"""GPU gate for chunk v0.2 (core/chunkv2.py, pattern N) — run on a
single card BEFORE any multi-GPU spend.

Checks (any failure exits non-zero):

  1. fast path resolves     flex_attention with return_lse exists.
  2. fast == reference      3-way LSE merge vs materialised concat
                            softmax, FORWARD AND BACKWARD (the
                            xformers-partial and flat-loss lessons:
                            never trust a forward-only check).
  3. compiled == eager      real-arm logits AND per-param grad cosines
                            (flat-loss lesson: compiled flex backward
                            was silently wrong on S layers once).
                            Zero-init projections (lm_head, c_proj,
                            cmlp_down) are randomized first, else their
                            upstream grads are legitimately 0 at step 1
                            and every cosine is vacuous. bf16 threshold
                            0.995: the read-side fetch top-k can
                            tie-flip compiled-vs-eager (benign -- the
                            mint loop itself runs eagerly in BOTH, so
                            the log is identical).
  4. memory + speed         real arm SSSNNNNNNSSS d1152 w128 btok128
                            K4 topk16 fetch4, T=4096: timed
                            fwd+bwd+step + peak VRAM, vs the window-only
                            control -- this measures the sequential-mint
                            tax we estimated at 1.15-2x.

Writes runs/chunkv2-validate/metrics.json and echoes VALIDATE_JSON
markers for `vast/launch.py logs`.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core.chunk as chunk_mod                          # noqa: E402
from core.chunkv2 import ChunkV2Attention               # noqa: E402
from core.model import GPT, GPTConfig                   # noqa: E402

RESULTS = {}
ARM = "SSSNNNNNNSSS"


def report(name, ok, fatal=True, **kw):
    RESULTS[name] = {"ok": bool(ok), **kw}
    print(f"VALIDATE_JSON {json.dumps({name: RESULTS[name]})}", flush=True)
    if not ok and fatal:
        finish(fail=name)


def finish(fail=None):
    os.makedirs("runs/chunkv2-validate", exist_ok=True)
    with open("runs/chunkv2-validate/metrics.json", "w") as f:
        json.dump(RESULTS, f, indent=2)
    with open("runs/LATEST", "w") as f:
        f.write("runs/chunkv2-validate")
    if fail:
        print(f"VALIDATE_FAILED at {fail}", flush=True)
        sys.exit(1)


def base_cfg(n_embd, n_head, **kw):
    base = dict(block_size=4096, vocab_size=50304, n_layer=12,
                n_head=n_head, n_embd=n_embd, pos="rope", norm="rms",
                mlp="relu2", bias=False, qk_norm=True, window=128,
                chunk_btok=128, chunk_k=4, chunk_topk=16, chunk_fetch_n=4,
                canon=True, canon_full=True, untied=True, zero_init=True,
                softcap=15.0, hg_frac=0.3, hg_bneck=8, hg_round=96)
    base.update(kw)
    return GPTConfig(**base)


def randomize_zero_inits(model):
    """Zero-init residual projections block their upstream grads for
    exactly one step -- randomize them so grad comparisons see signal."""
    with torch.no_grad():
        for n, p in model.named_parameters():
            if (n.endswith("c_proj.weight") or n.endswith("cmlp_down.weight")
                    or "lm_head" in n):
                p.normal_(0, 0.02)


def module_parity(n_embd, n_head, T, B, grads=False):
    torch.manual_seed(0)
    cfg = base_cfg(n_embd, n_head)
    m = ChunkV2Attention(cfg).cuda().to(torch.bfloat16)
    with torch.no_grad():
        m.cmlp_down.weight.normal_(0, 0.02)     # else cmlp_up grad is 0
    x = torch.randn(B, T, n_embd, device="cuda", dtype=torch.bfloat16)

    def run(fast):
        chunk_mod.USE_FAST = fast
        m.zero_grad(set_to_none=True)
        xi = x.clone().requires_grad_(grads)
        if grads:
            out = m(xi)
            out.float().square().mean().backward()
            return (out, xi.grad, m.wq.weight.grad, m.cmlp_up.weight.grad,
                    m.lam.grad)
        with torch.no_grad():
            return (m(xi), None, None, None, None)

    f = run(True)
    r = run(False)
    chunk_mod.USE_FAST = True
    diffs = {}
    for name, a, b in zip(("out", "x_grad", "wq_grad", "cmlp_grad",
                           "lam_grad"), f, r):
        if a is not None:
            d = (a.float() - b.float()).abs().max().item()
            s = b.float().abs().max().item()
            diffs[name] = (d, s)
    return diffs


def grad_cosines(model, compiled, idx):
    """Per-param grad cosine, compiled vs eager, on one bf16 step."""
    grads = {}
    for tag, m in (("eager", model), ("compiled", compiled)):
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = m(idx, targets=idx)
        loss.backward()
        grads[tag] = {n: p.grad.detach().float().clone()
                      for n, p in model.named_parameters()
                      if p.grad is not None}
    worst, worst_name = 1.0, None
    for n in grads["eager"]:
        a, b = grads["eager"][n], grads["compiled"][n]
        c = torch.nn.functional.cosine_similarity(
            a.flatten(), b.flatten(), dim=0).item()
        if c < worst:
            worst, worst_name = c, n
    return worst, worst_name


def timed_run(pattern, dbase, mb, iters):
    torch.manual_seed(0)
    cfg = base_cfg(dbase, 12, attn_pattern=pattern)
    model = GPT(cfg).cuda()
    opt = model.configure_optimizers(0.1, 6e-4, (0.9, 0.95), "cuda",
                                     opt="muon")
    model_c = torch.compile(model)
    idx = torch.randint(0, 50304, (mb, 4096), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    t0 = None
    for i in range(iters):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model_c(idx, targets=idx)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if i == 4:
            torch.cuda.synchronize()
            t0 = time.time()
    torch.cuda.synchronize()
    ms = (time.time() - t0) / (iters - 5) * 1000
    peak = torch.cuda.max_memory_allocated() / 2**30
    del model, model_c, opt
    torch.cuda.empty_cache()
    return ms, peak, mb * 4096 / (ms / 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--micro-bs", type=int, default=2)
    args, _ = ap.parse_known_args()

    assert torch.cuda.is_available()
    print(f"[validate_chunkv2] {torch.cuda.get_device_name(0)}", flush=True)
    for attr in ("recompile_limit", "cache_size_limit"):
        if hasattr(torch._dynamo.config, attr):
            setattr(torch._dynamo.config, attr, 64)

    from core.gated_swa import _HAVE_FLEX
    report("fast_path_resolves", _HAVE_FLEX)

    # 2. module parity, fast (flex + 3-way LSE) vs reference (concat)
    diffs = module_parity(384, 6, T=4096, B=1)
    d, s = diffs["out"]
    report("parity_T4096", d < 3e-2 * max(s, 1.0),
           max_abs_diff=d, ref_scale=s)
    diffs = module_parity(1152, 12, T=1024, B=2, grads=True)
    for name, (d, s) in diffs.items():
        report(f"parity_grads_{name}", d < 3e-2 * max(s, 1.0),
               max_abs_diff=d, ref_scale=s)

    # 3. compiled vs eager: logits + grad cosines on the real arm
    torch.manual_seed(0)
    cfg = base_cfg(1152, 12, attn_pattern=ARM)
    model = GPT(cfg).cuda()
    randomize_zero_inits(model)
    compiled = torch.compile(model)
    idx = torch.randint(0, 50304, (1, 4096), device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        eager_l = model(idx, targets=idx)[0].float()
        comp_l = compiled(idx, targets=idx)[0].float()
    cdiff = (eager_l - comp_l).abs().max().item()
    del eager_l, comp_l
    report("compiled_vs_eager_logits", cdiff < 0.1, max_abs_logit_diff=cdiff)
    worst, worst_name = grad_cosines(model, compiled, idx)
    report("compiled_vs_eager_grads", worst > 0.995,
           worst_cosine=round(worst, 5), worst_param=worst_name)
    del model, compiled
    torch.cuda.empty_cache()

    # 4. memory + speed: the arm vs the window-only control
    for pattern, name in ((ARM, "chunkv2"), ("S" * 12, "window_only")):
        oom = []
        for mb in (args.micro_bs, 1):
            try:
                ms, peak, tps = timed_run(pattern, 1152, mb, args.iters)
                report(f"bench_{name}", True, ms_per_iter=round(ms, 1),
                       peak_gb=round(peak, 2), tok_per_s=round(tps), mb=mb,
                       oom_at=oom)
                break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                oom.append(mb)
        else:
            report(f"bench_{name}", False, fatal=False, oom_at=oom)

    if all(r["ok"] for r in RESULTS.values()):
        finish()
        print("VALIDATE_PASSED", flush=True)
    else:
        finish(fail="bench_oom")


if __name__ == "__main__":
    main()
