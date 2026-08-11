"""Wallclock + memory profile: four-stroke launch config vs MiMo champion.

Run on a rented box (bench protocol: single GPU, never improvised on a
training box). Times compiled fwd+bwd+opt steps at T=4096 for both arms at
the same batch size, reports tok/s, peak memory, and achieved-vs-analytic
FLOP efficiency, and (--check) verifies compiled==eager numerics for the
M block -- the flex-compile-mask lesson says never trust torch.compile
around attention masking without checking.

  python scripts/bench_fourstroke_speed.py                 # both arms
  python scripts/bench_fourstroke_speed.py --arch fourstroke --batch 4
  python scripts/bench_fourstroke_speed.py --check         # numerics only
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model import GPT, GPTConfig  # noqa: E402

T = 4096
VOCAB = 50304
SPEEDRUN = dict(block_size=T, vocab_size=VOCAB, pos="rope", norm="rms",
                qk_norm=True, untied=True, zero_init=True, bias=False,
                mlp="relu2", window=128)


def mimo_cfg():
    return GPTConfig(n_layer=12, n_head=12, n_embd=768,
                     attn_pattern="SSSFSSSFSSSF", **SPEEDRUN)


def fourstroke_cfg():
    return GPTConfig(n_layer=12, n_head=8, n_embd=512,
                     attn_pattern="MMMFMMMFMMMF",
                     fs_n_machines=16, fs_d_machine=128, fs_n_head_m=2,
                     fs_backend="swa", fs_window=128, **SPEEDRUN)


def bench(name, cfg, batch, steps, compile_model=True):
    torch.manual_seed(0)
    model = GPT(cfg).cuda().bfloat16()
    n_params = model.num_params()
    fpt = model.flops_per_token(T)
    opt = model.configure_optimizers(0.1, 3e-4, (0.9, 0.95), "cuda",
                                     opt="muon")
    if compile_model:
        model = torch.compile(model)
    idx = torch.randint(0, VOCAB, (batch, T), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    for i in range(3):                                    # warmup + compile
        _, loss = model(idx, idx)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(steps):
        _, loss = model(idx, idx)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / steps
    toks = batch * T / dt
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"{name:>10}: {n_params/1e6:7.2f}M params  {fpt/1e6:7.1f}M f/tok  "
          f"{dt*1e3:7.1f} ms/step  {toks/1e3:7.1f}k tok/s  "
          f"{toks*fpt/1e12:6.2f} TFLOPs achieved  peak {peak:5.2f} GiB",
          flush=True)
    del model, opt
    torch.cuda.empty_cache()
    return toks


def check_compiled_numerics():
    """Compiled vs eager logits for the M block at fp32 (the flex mask
    gotcha: compiled masks can silently corrupt; verify before trusting
    any compiled timing or run)."""
    torch.manual_seed(1)
    cfg = GPTConfig(block_size=512, vocab_size=1024, n_layer=4, n_head=8,
                    n_embd=512, attn_pattern="MMMF", pos="rope",
                    norm="rms", qk_norm=True, untied=True, zero_init=True,
                    bias=False, mlp="relu2", window=128,
                    fs_n_machines=16, fs_d_machine=128, fs_n_head_m=2,
                    fs_backend="swa", fs_window=128)
    model = GPT(cfg).cuda().float()
    idx = torch.randint(0, 1024, (2, 512), device="cuda")
    with torch.no_grad():
        eager, _ = model(idx, idx)
        compiled, _ = torch.compile(model)(idx, idx)
    err = (eager - compiled).abs().max().item()
    print(f"compiled-vs-eager max |dlogit| = {err:.2e}", flush=True)
    assert err < 1e-3, "compiled M-block diverges from eager -- DO NOT TRAIN"
    print("NUMERICS_OK", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="both",
                    choices=("mimo", "fourstroke", "both"))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="compiled-vs-eager numerics only")
    args = ap.parse_args()
    print(torch.cuda.get_device_name(0), flush=True)
    if args.check:
        check_compiled_numerics()
        return
    if args.arch in ("fourstroke", "both"):
        check_compiled_numerics()
    if args.arch in ("mimo", "both"):
        bench("mimo", mimo_cfg(), args.batch, args.steps,
              not args.no_compile)
    if args.arch in ("fourstroke", "both"):
        bench("fourstroke", fourstroke_cfg(), args.batch, args.steps,
              not args.no_compile)


if __name__ == "__main__":
    main()
