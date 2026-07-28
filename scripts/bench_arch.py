"""Throughput shoot-out: dense GPT-2 vs SWA control vs admission-gated.

Times full training iterations (fwd + bwd + AdamW step) on random tokens —
no dataset needed, so it runs anywhere with a GPU:

    python scripts/bench_arch.py                     # all three archs
    python scripts/bench_arch.py --micro-bs 16 --compile
    python scripts/bench_arch.py --no-flex           # dense-mask SDPA only

Use this to (a) quantify the gated layer's router/mask overhead, (b) check
flex_attention actually wins over masked SDPA on the target GPU, (c) pick
--micro-bs for the rented card before spending money on it.
"""
import argparse
import os
import platform
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import gated_swa                                    # noqa: E402
from core.model import GPT, GPTConfig                         # noqa: E402

ARCHS = {
    "dense": dict(attn_pattern=""),
    "swa":   dict(attn_pattern="FSSSSFFSSSSF", window=512),
    "gated": dict(attn_pattern="FGGGGFFGGGGF", window=512, n_gates=8),
}


def bench(name, args, device):
    cfg = GPTConfig(**ARCHS[name])
    model = GPT(cfg).to(device)
    if args.compile:
        model = torch.compile(model)
    raw = model._orig_mod if args.compile else model
    opt = raw.configure_optimizers(0.1, 6e-4, (0.9, 0.95), device.type)
    B, T = args.micro_bs, args.seq_len
    idx = torch.randint(0, cfg.vocab_size, (B, T + 1), device=device)
    x, y = idx[:, :-1], idx[:, 1:]

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        opt.step()

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / args.iters * 1e3
    toks = B * T / (ms / 1e3)
    fpt = raw.flops_per_token(T)
    print(f"{name:>6}: {ms:8.1f} ms/iter  {toks / 1e3:7.1f}k tok/s  "
          f"{fpt / 1e6:.0f} MFLOPs/tok  "
          f"model-TFLOPs {toks * fpt / 1e12:.1f}", flush=True)
    del model, opt
    torch.cuda.empty_cache()
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="*", default=list(ARCHS))
    ap.add_argument("--micro-bs", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--no-flex", action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "benchmark needs a GPU"
    device = torch.device("cuda")
    if args.no_flex:
        gated_swa.USE_FLEX = False
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"torch {torch.__version__}  {platform.system()}  "
          f"B={args.micro_bs} T={args.seq_len} "
          f"compile={args.compile} flex={not args.no_flex}", flush=True)
    base = None
    for name in args.archs:
        ms = bench(name, args, device)
        base = base or ms
        if name != "dense" and base:
            print(f"        ({ms / base * 100:.1f}% of dense wall-clock)",
                  flush=True)


if __name__ == "__main__":
    main()
