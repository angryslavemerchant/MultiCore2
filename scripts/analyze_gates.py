"""Post-run gate autopsy: did fast and slow gates actually separate?

    python scripts/analyze_gates.py --run-name 124m-FGGGGFFGGGGF-w256-g8

Loads the trained checkpoint, routes real eval-split batches through it, and
reports PER GATE what the wandb aggregates hide:
  frac        admission share of the stream
  life mean/p50/p90   lifetime of its evicted tokens (positions survived)
  cens%       tokens still resident at sequence end (right-censored, i.e.
              lived longer than the window could measure)
  gapCV       burstiness of admissions: std/mean of inter-admission gaps.
              ~1 = Poisson-like steady traffic; >>1 = quiet stretches with
              bursts (long-lived residents WITHOUT unequal average traffic)

A "slow gate" shows up as low frac (or high gapCV) + long lifetimes + high
censoring; a "fast gate" as high frac + short lifetimes. If every row looks
alike, the load-balance loss has homogenized the gates.
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.data import open_data, DEFAULT_SHARDS                # noqa: E402
from core.gated_swa import GatedSWAttention, gated_death_times  # noqa: E402
from core.model import GPT, GPTConfig                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--shards", type=int, default=DEFAULT_SHARDS)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--micro-bs", type=int, default=8)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    path = os.path.join("runs", args.run_name, args.ckpt)
    ckpt = torch.load(path, map_location=device)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    T = cfg.block_size
    print(f"{path}: step {ckpt['step']}, val {ckpt.get('best_val'):.4f}, "
          f"pattern {cfg.attn_pattern} W={cfg.window} G={cfg.n_gates}")

    gated = [(i, blk) for i, blk in enumerate(model.transformer.h)
             if isinstance(blk.attn, GatedSWAttention)]
    acts = {}
    for i, blk in gated:
        blk.ln_1.register_forward_hook(
            lambda m, inp, out, i=i: acts.__setitem__(i, out.detach()))

    data = open_data(args.shards, args.data_dir)
    ev = data.batches(args.micro_bs, T, device, seed=0, split="eval")

    # per (layer, gate): lifetimes of evicted, censored count, admissions,
    # inter-admission gap CVs per row
    lives = defaultdict(list)
    cens = defaultdict(int)
    admits = defaultdict(int)
    gap_cvs = defaultdict(list)
    total_tokens = 0
    with torch.no_grad():
        for _ in range(args.batches):
            w = next(ev)
            model(w[:, :T])
            total_tokens += w.shape[0] * T
            for i, blk in gated:
                att = blk.attn
                gate = att.router(acts[i]).argmax(-1)          # (B,T)
                death = gated_death_times(gate, att.capacity)
                pos = torch.arange(T, device=device).expand_as(gate)
                life = (death - pos).cpu().numpy()
                g_np = gate.cpu().numpy()
                d_np = death.cpu().numpy()
                for g in range(att.n_gates):
                    sel = g_np == g
                    admits[(i, g)] += int(sel.sum())
                    died = sel & (d_np < T)
                    lives[(i, g)].extend(life[died].tolist())
                    cens[(i, g)] += int((sel & (d_np >= T)).sum())
                    for b in range(g_np.shape[0]):
                        p = np.flatnonzero(g_np[b] == g)
                        if len(p) > 3:
                            gaps = np.diff(p)
                            gap_cvs[(i, g)].append(
                                float(gaps.std() / max(gaps.mean(), 1e-9)))

    per_layer_tokens = total_tokens
    for i, blk in gated:
        G = blk.attn.n_gates
        print(f"\nlayer {i}  (capacity {blk.attn.capacity}/gate)")
        print("  gate   frac   evicted  life_mean  p50   p90   cens%  gapCV")
        means = []
        for g in range(G):
            n = admits[(i, g)]
            fr = n / per_layer_tokens
            lv = np.array(lives[(i, g)]) if lives[(i, g)] else np.array([0.0])
            cz = cens[(i, g)] / max(n, 1)
            cv = float(np.mean(gap_cvs[(i, g)])) if gap_cvs[(i, g)] else 0.0
            means.append(lv.mean())
            print(f"  {g:>4}  {fr:5.3f}  {len(lives[(i,g)]):>7}  "
                  f"{lv.mean():8.1f}  {np.median(lv):5.0f} "
                  f"{np.percentile(lv, 90):5.0f}  {cz*100:5.1f}  {cv:5.2f}")
        means = np.array(means)
        print(f"  spread: slowest/fastest gate mean lifetime = "
              f"{means.max() / max(means.min(), 1e-9):.2f}x")


if __name__ == "__main__":
    main()
