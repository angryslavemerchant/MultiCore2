"""On-GPU gate for the hierarchical predictive-plan model before any run.

Checks on the real run config (T=4096, 32x128 blocks, 16k-slot PKM):
  1. compiled-vs-eager FORWARD logits AND per-param BACKWARD grad
     cosines (flat-loss lesson);
  2. spec section-10 causality perturbation: change all tokens from
     block b on -> logits before block b bitwise-unchanged (eager,
     fp32); repeated at a superblock boundary;
  3. gradient aliveness for the new machinery (values, keys, gates,
     summaries, aux heads, BOS params);
  4. Muon classification (stack weights in, tables/heads out);
  5. timed compiled optimizer steps (throughput planning).

Usage (on the box):
  python scripts/validate_hier.py            # bf16, real cfg
  python scripts/validate_hier.py --fp32
Emits VALIDATE_JSON lines; every "ok" must be true.
"""
import argparse
import json
import sys
import os
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.hier import HierGPT, HierConfig  # noqa: E402


def emit(name, **kw):
    print("VALIDATE_JSON", json.dumps({name: kw}), flush=True)
    return kw.get("ok", True)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-mode", default="block",
                    choices=("block", "swa"))
    ap.add_argument("--levels", type=int, default=2, choices=(2, 3))
    ap.add_argument("--btok", type=int, default=128)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--cos-thresh", type=float, default=None)
    ap.add_argument("--skip-timing", action="store_true")
    args = ap.parse_args()
    thresh = args.cos_thresh or (0.9999 if args.fp32 else 0.995)
    torch.manual_seed(1337)
    for attr in ("recompile_limit", "cache_size_limit"):
        if hasattr(torch._dynamo.config, attr):
            setattr(torch._dynamo.config, attr, 64)

    cfg = HierConfig(token_mode=args.token_mode, levels=args.levels,
                     btok=args.btok)
    model = HierGPT(cfg).cuda()
    with torch.no_grad():
        model.lm_head.weight.normal_(0, 0.02)
        for n, p in model.named_parameters():
            if n.endswith("c_proj.weight"):
                p.normal_(0, 0.02)
        model.mem_gate.fill_(0.5)      # exercise the memory path

    # ---- causality perturbation (spec section 10), eager fp32 -----
    model.eval()
    idx = torch.randint(0, cfg.vocab_size, (1, cfg.block_size),
                        device="cuda")
    with torch.no_grad():
        base, _ = model(idx, idx)
        for tag, start in (("block", 16 * cfg.btok),
                           ("superblock", 2 * cfg.btok
                            * cfg.blocks_per_super)):
            pert = idx.clone()
            pert[0, start:] = (pert[0, start:] + 1) % cfg.vocab_size
            after, _ = model(pert, pert)
            same = bool(torch.equal(base[0, :start], after[0, :start]))
            if not emit(f"causality_{tag}", ok=same, boundary=start):
                print("VALIDATE_RESULT FAIL", flush=True)
                sys.exit(1)
    model.train()

    x = torch.randint(0, cfg.vocab_size, (1, cfg.block_size),
                      device="cuda")
    y = torch.randint(0, cfg.vocab_size, (1, cfg.block_size),
                      device="cuda")
    lg_e, g_e = grads(model, x, y, compiled=False, fp32=args.fp32)
    lg_c, g_c = grads(model, x, y, compiled=True, fp32=args.fp32)

    ok_all = emit("fwd_compiled_vs_eager",
                  ok=bool((lg_e - lg_c).abs().max()
                          < (1e-3 if args.fp32 else 5e-2)),
                  max_abs_logit_diff=float((lg_e - lg_c).abs().max()))

    worst = []
    for n in g_e:
        a, b = g_e[n].flatten(), g_c.get(n, g_e[n] * 0).flatten()
        if a.norm() == 0 and b.norm() == 0:
            continue
        cos = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
        worst.append((cos, n))
    worst.sort()
    ok = all(c > thresh for c, _ in worst)
    ok_all &= emit("bwd_grad_cosine", ok=bool(ok),
                   n_params=len(worst), thresh=thresh,
                   worst=[(round(c, 6), n) for c, n in worst[:8]])

    need_alive = ("memory.values.weight", "memory.keys_a",
                  "memory.keys_b", "memory.wq.weight", "mem_gate",
                  "w_summary.weight", "w_super.weight", "w_c.weight",
                  "w_aux_blk.weight", "w_aux_sup.weight",
                  "block_bos", "super_bos", "cond_gate")
    dead = [n for n in need_alive
            if n not in g_e or g_e[n].norm() == 0]
    ok_all &= emit("new_machinery_grads_alive", ok=not dead, dead=dead)

    combo = model.configure_optimizers(0.1, 6e-4, (0.9, 0.95), "cuda",
                                       opt="muon")
    muon_ids = {id(p) for g in combo.opts[0].param_groups
                for p in g["params"]}
    named = dict(model.named_parameters())
    leaked = [n for n in ("memory.values.weight", "memory.keys_a",
                          "lm_head.weight", "transformer.wte.weight")
              if id(named[n]) in muon_ids]
    stack_in = any(
        id(p) in muon_ids for n, p in named.items()
        if n.startswith("transformer.analysis.") and p.dim() == 2)
    ok_all &= emit("muon_classification", ok=not leaked and stack_in,
                   leaked=leaked, stack_2d_in_muon=stack_in)
    del combo

    if not args.skip_timing:
        opt = model.configure_optimizers(0.1, 6e-4, (0.9, 0.95),
                                         "cuda", opt="muon")
        mc = torch.compile(model)
        xb = torch.randint(0, cfg.vocab_size, (2, cfg.block_size),
                           device="cuda")
        t0 = None
        for i in range(12):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = mc(xb, targets=xb)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            if i == 3:
                torch.cuda.synchronize()
                t0 = time.time()
        torch.cuda.synchronize()
        ms = (time.time() - t0) / 8 * 1000
        emit("timing", ok=True, ms_per_iter_mb2=round(ms, 1),
             tok_per_s_single_gpu=round(2 * cfg.block_size / ms * 1000))

    print("VALIDATE_RESULT", "PASS" if ok_all else "FAIL", flush=True)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
