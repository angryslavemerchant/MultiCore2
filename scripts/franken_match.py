"""FLOPs-match the Frankenstein against the dense 124M baseline.

The Frankenstein changes both sides of the compute ledger: windowed G
layers spend less on attention than dense, diff attention spends 1.5x,
untied embeddings add parameters that do no compute, canon/router/norm
params are new. The honest matching currency (the project contract) is
analytic fwd+bwd FLOPs per token at T — so this script solves for the
d_base whose franken FLOPs/token equals the dense baseline's, using the
model's own GPTConfig.layer_widths() geometry and mirroring
GPT.flops_per_token()'s accounting, then verifies the winner by
instantiating the real GPT and comparing.

    python scripts/franken_match.py --frac 0.3 --bneck 8 --round 96 \
        --pattern FGGGGFFGGGGF --window 256 --n-gates 4 --recent-band 128
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model import GPT, GPTConfig                            # noqa: E402

VOCAB = GPTConfig.vocab_size
N_HEAD = 12


def franken_cfg(d_base, args):
    return GPTConfig(
        block_size=args.T, n_embd=d_base, n_head=N_HEAD,
        attn_pattern=args.pattern, window=args.window,
        n_gates=args.n_gates, recent_band=args.recent_band, pos="rope",
        hg_frac=args.frac, hg_bneck=args.bneck, hg_round=args.rnd,
        norm="rms", qk_norm=True, diff_attn=not getattr(args, "no_diff", False),
        canon=True,
        softcap=15.0, untied=True, zero_init=True, bias=False,
        mlp="relu2")


def params_and_flops(d_base, args):
    """Mirror of GPT param/FLOP accounting, no instantiation."""
    cfg = franken_cfg(d_base, args)
    ws = cfg.layer_widths()
    T = args.T

    def avg_keys(w):
        return T if T <= w else (w * (w + 1) / 2 + (T - w) * w) / T

    n = 2 * VOCAB * d_base + d_base          # wte + lm_head + ln_f (rms)
    n_compute = VOCAB * d_base + d_base      # wte is lookup-only (untied)
    score = 0.0
    for ch, w in zip(cfg.attn_pattern, ws):
        layer = 12 * w * w + 2 * w + 8 * w + 4 * (w // N_HEAD)
        if ch == "G":
            layer += cfg.n_gates * w
        n += layer
        n_compute += layer
        score += 12 * w * (T if ch == "F" else avg_keys(cfg.window))
    diff = 1.0 if getattr(args, "no_diff", False) else 1.5
    return n, 6 * n_compute + diff * score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--bneck", type=int, default=8)
    ap.add_argument("--round", type=int, default=96, dest="rnd")
    ap.add_argument("--pattern", default="FGGGGFFGGGGF")
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--n-gates", type=int, default=4)
    ap.add_argument("--recent-band", type=int, default=128)
    ap.add_argument("--T", type=int, default=4096)
    ap.add_argument("--no-diff", action="store_true",
                    help="score term 1.0x instead of diff attention's 1.5x")
    args = ap.parse_args()

    dense = [768] * 12
    dense_n = VOCAB * 768 + 2 * 768 + sum(12 * w * w + 13 * w for w in dense)
    dense_fpt = 6 * dense_n + sum(12 * w * args.T for w in dense)
    print(f"dense baseline: {dense_n:,} params, "
          f"{dense_fpt / 1e9:.3f} GFLOPs/token at T={args.T}\n")

    best = None
    for d_base in range(args.rnd * 4, 3072 + args.rnd, args.rnd):
        n, fpt = params_and_flops(d_base, args)
        if best is None or abs(fpt - dense_fpt) < abs(best[2] - dense_fpt):
            best = (d_base, n, fpt)
    d_base, n, fpt = best
    ws = franken_cfg(d_base, args).layer_widths()
    print(f"franken match: d_base={d_base}  params={n:,}  "
          f"flops/tok={fpt / 1e9:.3f}G ({fpt / dense_fpt:.3f}x dense)")
    print(f"  widths    {ws}")
    print(f"  head_dim  {[w // N_HEAD for w in ws]}")
    print(f"  waist     {min(ws)}/{d_base} = {min(ws) / d_base:.3f}")

    print("\nverifying against a real GPT instance...")
    model = GPT(franken_cfg(d_base, args))
    real_n = sum(p.numel() for p in model.parameters())
    real_fpt = model.flops_per_token(args.T)
    print(f"  real params {real_n:,} (analytic {n:,}, "
          f"drift {real_n - n:+,})")
    print(f"  real flops/tok {real_fpt / 1e9:.3f}G "
          f"({real_fpt / dense_fpt:.3f}x dense; analytic "
          f"{fpt / 1e9:.3f}G)")
    print(f"\ntrainer flags: --hg-frac {args.frac} --hg-bneck {args.bneck} "
          f"--hg-round {args.rnd} --hg-dbase {d_base}")


if __name__ == "__main__":
    main()
