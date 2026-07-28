"""Trust-but-verify for suspicious benchmark wins: measures what the gated
mask actually looks like at init and whether compile changes the math.

    python scripts/validate_gated.py [--micro-bs 8]

Checks:
  1. Mask density: mean visible keys/query for SWA vs gated death times on
     REAL model hidden states (not synthetic) — if gated is much sparser
     than SWA, the router has collapsed and any speed win is an artifact.
  2. Gate occupancy at init: fraction of tokens per gate at layer 1.
  3. Compiled vs eager forward on identical input: max |diff|.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.gated_swa import (GatedSWAttention, gated_death_times,
                            swa_death_times, interval_mask)   # noqa: E402
from core.model import GPT, GPTConfig                         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--micro-bs", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--pattern", default="FGGGGFFGGGGF")
    ap.add_argument("--no-flex", action="store_true")
    ap.add_argument("--compile-only", action="store_true",
                    help="skip density stats; just the compile check")
    args = ap.parse_args()
    if args.no_flex:
        from core import gated_swa
        gated_swa.USE_FLEX = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1234)

    B, T = args.micro_bs, args.seq_len
    cfg = GPTConfig(attn_pattern=args.pattern, window=512, n_gates=8)
    model = GPT(cfg).to(device).eval()
    idx = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    if args.compile_only:
        with torch.no_grad():
            eager_logits, _ = model(idx, None)
        cmodel = torch.compile(model)
        with torch.no_grad():
            comp_logits, _ = cmodel(idx, None)
        diff = float((comp_logits - eager_logits).abs().max())
        print(f"pattern={args.pattern} flex={not args.no_flex} "
              f"compiled vs eager max|diff|: {diff:.4e} "
              f"({'OK' if diff < 5e-2 else 'MISMATCH'})")
        return

    # -- 1+2: real router decisions at init, via the stats hook ------------
    gated = [(i, b.attn) for i, b in enumerate(model.transformer.h)
             if isinstance(b.attn, GatedSWAttention)]
    for _, m in gated:
        m.collect_stats = True
    with torch.no_grad():
        eager_logits, _ = model(idx, None)
    d_swa = swa_death_times(T, cfg.window, device).expand(B, T)
    swa_keys = interval_mask(d_swa, T)[:, 0].float().sum(-1).mean()
    print(f"swa   mean visible keys/query: {swa_keys:7.1f}")
    for i, m in gated:
        m.collect_stats = False
        # reconstruct this layer's death times from its router on the SAME
        # input path: rerun ln_1 input is awkward; occupancy tells the story
        frac = m.stats["gate_frac"]
        print(f"layer {i:2d}: gate occupancy max {float(frac.max()):.3f} "
              f"min {float(frac.min()):.3f} "
              f"entropy {m.stats['router_entropy']:.3f} "
              f"frac_evicted {m.stats['frac_evicted']:.3f} "
              f"mean_lifetime {m.stats['mean_lifetime']:.0f}")

    # density measured directly from each gated layer's router decisions on
    # the real hidden state: capture ln_1 outputs (the router's input) with
    # forward hooks, then recompute death times and count visible keys
    acts = {}
    handles = []
    for i, blk in enumerate(model.transformer.h):
        if isinstance(blk.attn, GatedSWAttention):
            handles.append(blk.ln_1.register_forward_hook(
                lambda m, inp, out, i=i: acts.__setitem__(i, out.detach())))
    with torch.no_grad():
        model(idx, None)
    for h in handles:
        h.remove()
    for i, blk in enumerate(model.transformer.h):
        if i in acts:
            att = blk.attn
            gate = att.router(acts[i]).argmax(-1)
            d = gated_death_times(gate, att.capacity)
            keys = interval_mask(d, T)[:, 0].float().sum(-1).mean()
            print(f"layer {i:2d}: gated mean visible keys/query "
                  f"{keys:7.1f}  (swa reference {float(swa_keys):.1f})")

    # -- 3: compile changes nothing numerically ----------------------------
    if device == "cuda":
        cmodel = torch.compile(model)
        with torch.no_grad():
            comp_logits, _ = cmodel(idx, None)
        diff = float((comp_logits - eager_logits).abs().max())
        print(f"compiled vs eager max|diff|: {diff:.4e} "
              f"({'OK' if diff < 5e-2 else 'MISMATCH'})")


if __name__ == "__main__":
    main()
