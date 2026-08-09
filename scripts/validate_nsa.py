"""On-GPU gate for the NSA-with-registers arm (core/nsa.py).

Real run config: T=4096, hourglass d1080 f0.2 b8 r24, window 128,
nsa 32-blocks / top-12 / 1024 registers, speedrun stack. Checks:
  1. causality perturbation (eager): change tokens from position s on
     -> logits before s bitwise-unchanged (mid-block + block edge);
  2. compiled-vs-eager forward logits and per-param backward grad
     cosines (flat-loss lesson);
  3. registers + gates receive gradient; registers stay out of Muon;
  4. timed compiled optimizer steps at --micro-bs (throughput + OOM
     probe: run once with 2, once with 4).

Usage: python scripts/validate_nsa.py [--micro-bs 2] [--fp32]
Emits VALIDATE_JSON lines; every "ok" must be true.
"""
import argparse
import json
import sys
import os
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model import GPT, GPTConfig  # noqa: E402


def emit(name, **kw):
    print("VALIDATE_JSON", json.dumps({name: kw}), flush=True)
    return kw.get("ok", True)


def real_cfg(args):
    return GPTConfig(block_size=4096, vocab_size=50304, n_layer=12,
                     n_head=12, n_embd=args.dbase, bias=False,
                     attn="nsa", mlp="relu2", norm="rms", qk_norm=True,
                     pos="rope", untied=True, zero_init=True,
                     softcap=15, canon=True, window=args.window,
                     hg_frac=0.2, hg_bneck=8, hg_round=24,
                     nsa_block=args.nsa_block, nsa_topk=args.nsa_topk,
                     nsa_nreg=args.nsa_nreg)


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
    ap.add_argument("--dbase", type=int, default=1080)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--nsa-block", type=int, default=32)
    ap.add_argument("--nsa-topk", type=int, default=12)
    ap.add_argument("--nsa-nreg", type=int, default=1024)
    ap.add_argument("--micro-bs", type=int, default=2)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--cos-thresh", type=float, default=None)
    ap.add_argument("--skip-timing", action="store_true")
    args = ap.parse_args()
    thresh = args.cos_thresh or (0.9999 if args.fp32 else 0.995)
    torch.manual_seed(1337)
    for attr in ("recompile_limit", "cache_size_limit"):
        if hasattr(torch._dynamo.config, attr):
            setattr(torch._dynamo.config, attr, 64)

    cfg = real_cfg(args)
    model = GPT(cfg).cuda()
    with torch.no_grad():
        # zero-init head/proj hide signal; randomize for the checks
        model.lm_head.weight.normal_(0, 0.02)
        for n, p in model.named_parameters():
            if n.endswith("c_proj.weight"):
                p.normal_(0, 0.02)
    print(f"params {model.num_params()/1e6:.2f}M "
          f"flops/tok {model.flops_per_token(4096)/1e9:.3f}G", flush=True)

    # ---- causality perturbation, eager ----------------------------
    model.eval()
    idx = torch.randint(0, cfg.vocab_size, (1, cfg.block_size),
                        device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        base, _ = model(idx, idx)
        for s in (2048, 2069, 3072):    # block edge, mid-block, edge
            pert = idx.clone()
            pert[0, s:] = (pert[0, s:] + 1) % cfg.vocab_size
            after, _ = model(pert, pert)
            same = bool(torch.equal(base[0, :s], after[0, :s]))
            if not emit(f"causality_{s}", ok=same, boundary=s):
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

    dead = [n for n, p in model.named_parameters()
            if (n.endswith(".registers") or "w_gate" in n)
            and (n not in g_e or g_e[n].norm() == 0)]
    ok_all &= emit("registers_gates_grads_alive", ok=not dead, dead=dead)

    combo = model.configure_optimizers(0.1, 6e-4, (0.9, 0.95), "cuda",
                                       opt="muon")
    muon_ids = {id(p) for g in combo.opts[0].param_groups
                for p in g["params"]}
    leaked = [n for n, p in model.named_parameters()
              if n.endswith(".registers") and id(p) in muon_ids]
    stack_in = any(
        id(p) in muon_ids for n, p in model.named_parameters()
        if "attn.c_attn.weight" in n)
    ok_all &= emit("muon_classification", ok=not leaked and stack_in,
                   leaked=leaked, stack_2d_in_muon=stack_in)
    del combo

    if not args.skip_timing:
        opt = model.configure_optimizers(0.1, 6e-4, (0.9, 0.95),
                                         "cuda", opt="muon")
        mc = torch.compile(model)
        xb = torch.randint(0, cfg.vocab_size,
                           (args.micro_bs, cfg.block_size), device="cuda")
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
        emit("timing", ok=True, micro_bs=args.micro_bs,
             ms_per_iter=round(ms, 1),
             tok_per_s_single_gpu=round(
                 args.micro_bs * cfg.block_size / ms * 1000),
             peak_mem_gb=round(
                 torch.cuda.max_memory_allocated() / 2**30, 2))

    print("VALIDATE_RESULT", "PASS" if ok_all else "FAIL", flush=True)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
