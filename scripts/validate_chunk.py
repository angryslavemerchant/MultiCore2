"""GPU gate for chunk-latent attention (core/chunk.py) — run on the bench
box BEFORE any multi-GPU spend (bench protocol: single rented card, never
improvised on training boxes).

Checks, in order (any failure exits non-zero, so run_training.sh leaves
the box alive for inspection):

  1. fast path resolves     xformers memory_efficient_attention_partial
                            exists and passes core.chunk's numerics probe.
                            Real runs REQUIRE it: the reference path
                            materialises (B,H,T,T) logits and OOMs at 4k.
  2. fast == reference      module-level joint softmax vs LSE merge, bf16,
                            T=4096 (narrow module) and T=1024 (full width).
  3. compiled == eager      full d1056 K-model logits, T=4096 — the
                            flex-compile-mask lesson says NEVER trust
                            torch.compile with fresh mask machinery
                            unverified.
  4. memory + speed         real chunk arm (d1056 hourglass, KKKK...,
                            full canon, T=4096, mb4): timed fwd+bwd+step,
                            peak VRAM. Same for the blocksum control.

Writes runs/chunk-validate/metrics.json (+ runs/LATEST) and echoes
VALIDATE_JSON markers for `vast/launch.py logs`.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core.chunk as chunk_mod                          # noqa: E402
from core.chunk import ChunkAttention                   # noqa: E402
from core.model import GPT, GPTConfig                   # noqa: E402

RESULTS = {}


def report(name, ok, fatal=True, **kw):
    RESULTS[name] = {"ok": bool(ok), **kw}
    print(f"VALIDATE_JSON {json.dumps({name: RESULTS[name]})}", flush=True)
    if not ok and fatal:
        finish(fail=name)


def finish(fail=None):
    os.makedirs("runs/chunk-validate", exist_ok=True)
    with open("runs/chunk-validate/metrics.json", "w") as f:
        json.dump(RESULTS, f, indent=2)
    with open("runs/LATEST", "w") as f:
        f.write("runs/chunk-validate")
    if fail:
        print(f"VALIDATE_FAILED at {fail}", flush=True)
        sys.exit(1)


def attn_cfg(n_embd, n_head, window=256, **kw):
    base = dict(block_size=4096, vocab_size=50304, n_layer=12,
                n_head=n_head, n_embd=n_embd, pos="rope", norm="rms",
                mlp="relu2", bias=False, qk_norm=True, window=window,
                chunk_btok=256, chunk_k=16, canon=True, canon_full=True,
                untied=True, zero_init=True, softcap=15.0,
                hg_frac=0.3, hg_bneck=8, hg_round=96)
    base.update(kw)
    return GPTConfig(**base)


def real_cfg(pattern, dbase):
    return attn_cfg(dbase, 12, attn_pattern=pattern)


def module_parity(n_embd, n_head, T, B, grads=False):
    """Forward (and optionally BACKWARD) parity, fast vs reference.
    The xformers-partial dead end was caught only at backward time —
    never again trust a forward-only check for a training path."""
    torch.manual_seed(0)
    cfg = attn_cfg(n_embd, n_head)
    m = ChunkAttention(cfg).cuda().to(torch.bfloat16)
    x = torch.randn(B, T, n_embd, device="cuda", dtype=torch.bfloat16)

    def run(fast):
        chunk_mod.USE_FAST = fast
        m.zero_grad(set_to_none=True)
        xi = x.clone().requires_grad_(grads)
        if grads:
            out = m(xi)
            out.float().square().mean().backward()
            return out, xi.grad, m.wq.weight.grad, m.cmlp_up.weight.grad
        with torch.no_grad():
            return (m(xi), None, None, None)

    f = run(True)
    r = run(False)
    chunk_mod.USE_FAST = True
    diffs = {}
    for name, a, b in zip(("out", "x_grad", "wq_grad", "cmlp_grad"), f, r):
        if a is not None:
            d = (a.float() - b.float()).abs().max().item()
            s = b.float().abs().max().item()
            diffs[name] = (d, s)
    return diffs


def timed_run(pattern, dbase, mb, iters, compile_model=True):
    torch.manual_seed(0)
    cfg = real_cfg(pattern, dbase)
    model = GPT(cfg).cuda()
    opt = model.configure_optimizers(0.1, 6e-4, (0.9, 0.95), "cuda",
                                     opt="muon")
    if compile_model:
        model = torch.compile(model)
    idx = torch.randint(0, 50304, (mb, 4096), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    t0 = None
    for i in range(iters):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(idx, targets=idx)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if i == 4:                       # skip compile/warmup iters
            torch.cuda.synchronize()
            t0 = time.time()
    torch.cuda.synchronize()
    ms = (time.time() - t0) / (iters - 5) * 1000
    peak = torch.cuda.max_memory_allocated() / 2**30
    del model, opt
    torch.cuda.empty_cache()
    return ms, peak, mb * 4096 / (ms / 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--dbase", type=int, default=1056)
    ap.add_argument("--micro-bs", type=int, default=4)
    ap.parse_known_args()
    args, _ = ap.parse_known_args()

    assert torch.cuda.is_available()
    print(f"[validate_chunk] {torch.cuda.get_device_name(0)}", flush=True)
    # mirror the trainer: 12 hourglass widths > default recompile limit,
    # and eager-fallback flex is ~6x slow
    for attr in ("recompile_limit", "cache_size_limit"):
        if hasattr(torch._dynamo.config, attr):
            setattr(torch._dynamo.config, attr, 64)

    # 1. fast path must be available (flex with return_lse)
    from core.gated_swa import _HAVE_FLEX
    report("fast_path_resolves", _HAVE_FLEX)

    # 2. module parity, fast (flex + LSE merge) vs reference (concat
    # softmax) — forward at scale, forward+BACKWARD at full width
    diffs = module_parity(384, 6, T=4096, B=1)
    d, s = diffs["out"]
    report("parity_T4096", d < 3e-2 * max(s, 1.0),
           max_abs_diff=d, ref_scale=s)
    diffs = module_parity(1152, 12, T=1024, B=2, grads=True)
    for name, (d, s) in diffs.items():
        report(f"parity_grads_{name}", d < 3e-2 * max(s, 1.0),
               max_abs_diff=d, ref_scale=s)

    # 3. compiled vs eager logits on the real model (flex-mask lesson)
    torch.manual_seed(0)
    cfg = real_cfg("K" * 12, args.dbase)
    model = GPT(cfg).cuda()
    idx = torch.randint(0, 50304, (1, 4096), device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        eager = model(idx, targets=idx)[0].float()
        comp = torch.compile(model)(idx, targets=idx)[0].float()
    cdiff = (eager - comp).abs().max().item()
    del model, eager, comp
    torch.cuda.empty_cache()
    report("compiled_vs_eager", cdiff < 0.1, max_abs_logit_diff=cdiff)

    # 4. memory + speed, both arms at the matched configs
    # champion-era 5090 practice is mb2 single-card (mb4 was the 8x
    # setting) — try mb4 for headroom info, fall back to mb2
    for pattern, dbase, name in (("K" * 12, args.dbase, "chunk"),
                                 ("B" * 12, args.dbase, "blocksum"),
                                 ("S" * 12, 1152, "window_only")):
        oom = []
        for mb in (args.micro_bs, 2):
            try:
                ms, peak, tps = timed_run(pattern, dbase, mb, args.iters)
                report(f"bench_{name}", True, ms_per_iter=round(ms, 1),
                       peak_gb=round(peak, 2), tok_per_s=round(tps),
                       mb=mb, oom_at=oom, dbase=dbase)
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
