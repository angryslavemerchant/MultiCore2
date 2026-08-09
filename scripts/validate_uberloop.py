"""On-GPU gate for the uberloop arm (cfg.loops) before any run.

Checks, on the real run config (SSSFSSSFSSSF w128 hg0.3 d960 tiered
loops 1,1,1,2,2,4,4,4,4,2,1,1):
  1. compiled-vs-eager FORWARD logits AND per-param BACKWARD grad
     cosines (the flat-loss lesson: fwd-only checks pass while a
     compiled backward silently corrupts grads);
  2. loop gains/scales receive gradients (symmetry-breakers alive);
  3. Muon classification: gains/scales (1D ParameterList entries) must
     land in AdamW, shared block matrices in Muon;
  4. timed compiled steps vs the unlooped same-d control (loop overhead
     is ~w^2-cheap on paper; verify the compiled unroll agrees).

Usage (on the box):
  python scripts/validate_uberloop.py            # bf16, real cfg
  python scripts/validate_uberloop.py --fp32     # rounding-free leg
  python scripts/validate_uberloop.py --seq 1024 # smoke
Emits VALIDATE_JSON lines; every "ok" must be true.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model import GPT, GPTConfig  # noqa: E402

LOOPS = "1,1,1,2,2,4,4,4,4,2,1,1"


def emit(name, **kw):
    print("VALIDATE_JSON", json.dumps({name: kw}), flush=True)
    return kw.get("ok", True)


def make_cfg(seq, d, loops):
    return GPTConfig(
        block_size=seq, vocab_size=50304, n_layer=12, n_head=12,
        n_embd=d, pos="rope", norm="rms", mlp="relu2", bias=False,
        qk_norm=True, untied=True, zero_init=True, softcap=15.0,
        canon=True, canon_full=True, window=128,
        attn_pattern="SSSFSSSFSSSF", loops=loops,
        hg_frac=0.3, hg_bneck=8, hg_round=96)


def grads(model, x, y, compiled, fp32=False):
    from contextlib import nullcontext
    m = torch.compile(model) if compiled else model
    model.zero_grad(set_to_none=True)
    amp = (nullcontext() if fp32 else
           torch.autocast(device_type="cuda", dtype=torch.bfloat16))
    with amp:
        logits, loss = m(x, targets=y)
    loss.backward()
    return (logits.detach().float(),
            {n: p.grad.detach().float().clone()
             for n, p in model.named_parameters() if p.grad is not None})


def timed(cfg, iters=12, mb=2):
    torch.manual_seed(0)
    model = GPT(cfg).cuda()
    opt = model.configure_optimizers(0.1, 6e-4, (0.9, 0.95), "cuda",
                                     opt="muon")
    mc = torch.compile(model)
    idx = torch.randint(0, 50304, (mb, cfg.block_size), device="cuda")
    t0 = None
    for i in range(iters):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = mc(idx, targets=idx)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if i == 3:
            torch.cuda.synchronize()
            t0 = time.time()
    torch.cuda.synchronize()
    ms = (time.time() - t0) / (iters - 4) * 1000
    del model, mc, opt
    torch.cuda.empty_cache()
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--dbase", type=int, default=960)
    ap.add_argument("--fp32", action="store_true",
                    help="no-autocast leg: separates rounding noise from "
                         "real compiled-backward bugs")
    ap.add_argument("--cos-thresh", type=float, default=None)
    ap.add_argument("--skip-timing", action="store_true")
    args = ap.parse_args()
    thresh = args.cos_thresh or (0.9999 if args.fp32 else 0.995)
    if args.fp32:
        # fp32 flex tiles blow the 5090 smem limit; xformers is bf16-only
        from core import gated_swa
        gated_swa.USE_FLEX = False
    torch.manual_seed(1337)
    for attr in ("recompile_limit", "cache_size_limit"):
        if hasattr(torch._dynamo.config, attr):
            setattr(torch._dynamo.config, attr, 64)

    cfg = make_cfg(args.seq, args.dbase, LOOPS)
    model = GPT(cfg).cuda()
    with torch.no_grad():
        # zero-init head multiplies every trunk grad by 0 at step 1
        model.lm_head.weight.normal_(0, 0.02)
        # zero-init c_proj blocks each block's residual delta, which
        # would zero the loop scales' grads too
        for n, p in model.named_parameters():
            if n.endswith("c_proj.weight"):
                p.normal_(0, 0.02)

    x = torch.randint(0, 50304, (1, args.seq), device="cuda")
    y = torch.randint(0, 50304, (1, args.seq), device="cuda")

    lg_e, g_e = grads(model, x, y, compiled=False, fp32=args.fp32)
    lg_c, g_c = grads(model, x, y, compiled=True, fp32=args.fp32)

    ok_all = emit("fwd_compiled_vs_eager",
                  ok=bool((lg_e - lg_c).abs().max()
                          < (1e-3 if args.fp32 else 5e-2)),
                  max_abs_logit_diff=float((lg_e - lg_c).abs().max()))

    worst = []
    for n in g_e:
        a, b = g_e[n].flatten(), g_c[n].flatten()
        if a.norm() == 0 and b.norm() == 0:
            continue
        cos = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
        worst.append((cos, n))
    worst.sort()
    ok = all(c > thresh for c, _ in worst)
    ok_all &= emit("bwd_grad_cosine", ok=bool(ok),
                   n_params=len(worst), thresh=thresh,
                   worst=[(round(c, 6), n) for c, n in worst[:8]])

    loop_named = [n for n, _ in model.named_parameters()
                  if ".gains." in n or ".scales." in n]
    alive = [n for n in loop_named if n in g_e and g_e[n].norm() > 0]
    ok_all &= emit("loop_gains_grads_alive",
                   ok=len(alive) == len(loop_named) and len(loop_named) > 0,
                   alive=len(alive), total=len(loop_named))

    combo = model.configure_optimizers(0.1, 6e-4, (0.9, 0.95), "cuda",
                                       opt="muon")
    muon_ids = {id(p) for g in combo.opts[0].param_groups
                for p in g["params"]}
    leaked = [n for n, p in model.named_parameters()
              if (".gains." in n or ".scales." in n) and id(p) in muon_ids]
    shared_in = any(id(p) in muon_ids for n, p in model.named_parameters()
                    if "block.attn.c_attn.weight" in n)
    ok_all &= emit("muon_classification", ok=not leaked and shared_in,
                   gains_leaked_to_muon=leaked, shared_2d_in_muon=shared_in)
    del combo

    if not args.skip_timing:
        ms_loop = timed(cfg)
        ms_ctrl = timed(make_cfg(args.seq, args.dbase, ""))
        emit("timing", ok=True, looped_ms=round(ms_loop, 1),
             unlooped_same_d_ms=round(ms_ctrl, 1),
             ratio=round(ms_loop / ms_ctrl, 3))

    print("VALIDATE_RESULT", "PASS" if ok_all else "FAIL", flush=True)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
