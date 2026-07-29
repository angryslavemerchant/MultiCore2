"""Parameter-match a slice-carry hourglass against the dense baseline,
before training anything.

The hourglass (core/model.py SliceBlock): the residual stream stays at
d_base for every layer; layer l reads and writes only the first w_l dims
and the trailing dims carry forward untouched. Widths fall by a constant
per-layer ratio from d_base to frac*d_base at layer --bneck, hold there
for --mid extra flat layers, then rise back to d_base. Geometry comes
from GPTConfig.layer_widths() — the same code the model builds from, so
this script cannot drift from the implementation.

Three knobs: --frac (how narrow), --bneck (where), --mid (extra layers
at the waist; they cost ~12*(frac*d_base)^2 params each, so depth at
the waist is nearly free in parameters — its real cost is serial
wall-clock). For each combo the script solves for the d_base that
matches the dense 124M baseline's parameter count.

    python scripts/hourglass_match.py --frac 0.2 --bneck 8
    python scripts/hourglass_match.py --frac 0.2 --bneck 8 --mid 0,4,8
    python scripts/hourglass_match.py --frac 0.1,0.2,0.35 --bneck 6,8,10

Trainer flags for a chosen row: --hg-frac --hg-bneck --hg-mid --hg-dbase.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model import GPTConfig                                 # noqa: E402

VOCAB = GPTConfig.vocab_size
N_HEAD = 12


def total_params(ws, d_base, pos="rope", block_size=4096):
    """Same accounting as core/model.py: tied wte/lm_head, biases
    everywhere, two LayerNorms per block, rope = no wpe."""
    n = VOCAB * d_base + 2 * d_base            # wte (tied) + ln_f
    if pos == "learned":
        n += block_size * d_base
    return n + sum(12 * w * w + 13 * w for w in ws)


def flops_per_token(ws, d_base, T):
    """6*N + 12*w_l*T per layer (dense attention at the layer's width)."""
    return 6 * total_params(ws, d_base) + sum(12 * w * T for w in ws)


def solve(frac, bneck, mid, rnd, target, pos, block_size):
    best = None
    for d_base in range(rnd * 4, 3072 + rnd, rnd):
        cfg = GPTConfig(n_embd=d_base, n_head=N_HEAD, hg_frac=frac,
                        hg_bneck=bneck, hg_mid=mid)
        ws = cfg.layer_widths(rnd)
        n = total_params(ws, d_base, pos, block_size)
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (d_base, n, ws)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", default="0.2", help="comma-separated fractions")
    ap.add_argument("--bneck", default="8", help="comma-separated layer idx")
    ap.add_argument("--mid", default="0",
                    help="comma-separated extra flat layers at the waist")
    ap.add_argument("--round", type=int, default=24, dest="rnd",
                    help="round widths to a multiple of this (24 = 12 heads "
                         "with even head_dim)")
    ap.add_argument("--pos", default="rope", choices=("rope", "learned"))
    ap.add_argument("--block-size", type=int, default=4096)
    args = ap.parse_args()

    dense = [768] * 12
    target = total_params(dense, 768, args.pos, args.block_size)
    dense_fpt = flops_per_token(dense, 768, args.block_size)
    print(f"dense baseline target: {target:,} params, "
          f"{dense_fpt / 1e9:.2f} GFLOPs/token at T={args.block_size}\n")

    for frac in (float(x) for x in args.frac.split(",")):
        for bneck in (int(x) for x in args.bneck.split(",")):
            for mid in (int(x) for x in args.mid.split(",")):
                d_base, n, ws = solve(frac, bneck, mid, args.rnd, target,
                                      args.pos, args.block_size)
                fpt = flops_per_token(ws, d_base, args.block_size)
                print(f"frac={frac} bneck=L{bneck} mid=+{mid}: "
                      f"d_base={d_base}  params={n:,} "
                      f"({(n - target) / target:+.3%})  "
                      f"flops/tok={fpt / dense_fpt:.3f}x dense  "
                      f"depth={len(ws)}")
                print(f"  widths    {ws}")
                print(f"  head_dim  {[w // N_HEAD for w in ws]}")
                print(f"  waist {min(ws)}/{d_base} = "
                      f"{min(ws) / d_base:.3f}\n")


if __name__ == "__main__":
    main()
