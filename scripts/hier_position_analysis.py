"""Run-one deliverable for the hier arm: decompose val loss by
WITHIN-BLOCK position, with the conditioning and memory switches
toggled. This is the pre-declared judge (aggregate val was never it).

    python scripts/hier_position_analysis.py --run-name <name>

Buckets: within-block positions [0-15, 16-31, 32-63, 64-127]. A token
at in-block position p has p raw-context tokens (block mode) plus
whatever the plans carry. Four eval variants:
    full        : plans + memory as trained
    no_cond     : conditioning gates off (plans severed)
    no_mem      : memory gate off
    no_both     : both off
plan value at position p  = loss(no_cond) - loss(full)   [+ = plans help]
memory value at position p = loss(no_mem) - loss(full)

Writes runs/<name>/position_analysis.json and prints the table.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.hier import HierGPT, HierConfig      # noqa: E402
from core.data import open_data                # noqa: E402

BUCKETS = ((0, 16), (16, 32), (32, 64), (64, 128))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--eval-iters", type=int, default=50)
    ap.add_argument("--micro-bs", type=int, default=4)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(os.path.join("runs", args.run_name, args.ckpt),
                      map_location=device)
    ccfg = {k: v for k, v in ckpt["config"].items() if k != "arch"}
    cfg = HierConfig(**ccfg)
    model = HierGPT(cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    T = cfg.block_size
    data = open_data(data_dir=args.data_dir)

    variants = {"full": (False, False), "no_cond": (True, False),
                "no_mem": (False, True), "no_both": (True, True)}
    sums = {v: np.zeros(len(BUCKETS)) for v in variants}
    cnts = np.zeros(len(BUCKETS))

    ev = data.batches(args.micro_bs, T, device, seed=0, split="eval")
    with torch.no_grad():
        for it in range(args.eval_iters):
            w = next(ev)
            x, y = w[:, :-1], w[:, 1:]
            first = True
            for vname, (dc, dm) in variants.items():
                model.disable_cond, model.disable_memory = dc, dm
                with torch.autocast(device, dtype=torch.bfloat16,
                                    enabled=device == "cuda"):
                    logits, _ = model(x, y)
                lt = F.cross_entropy(
                    logits.float().view(-1, logits.size(-1)),
                    y.reshape(-1), reduction="none").view(x.shape[0], T)
                # position t predicts token t+1 -> its in-block phase is
                # (t+1) % btok: the position of the PREDICTED token
                phase = (torch.arange(T) + 1) % cfg.btok
                for bi, (lo, hi) in enumerate(BUCKETS):
                    m = (phase >= lo) & (phase < hi)
                    sums[vname][bi] += float(lt[:, m].sum())
                    if first:
                        cnts[bi] += int(m.sum()) * x.shape[0]
                first = False
    model.disable_cond = model.disable_memory = False

    out = {}
    print(f"{'bucket':>10} | " + " | ".join(f"{v:>8}" for v in variants)
          + " |  plan_val |  mem_val")
    for bi, (lo, hi) in enumerate(BUCKETS):
        row = {v: sums[v][bi] / cnts[bi] for v in variants}
        plan_v = row["no_cond"] - row["full"]
        mem_v = row["no_mem"] - row["full"]
        out[f"pos_{lo}-{hi}"] = {**row, "plan_value": plan_v,
                                 "mem_value": mem_v}
        print(f"{lo:>4}-{hi:<5} | "
              + " | ".join(f"{row[v]:8.4f}" for v in variants)
              + f" | {plan_v:+9.4f} | {mem_v:+8.4f}")
    agg = {v: float(sums[v].sum() / cnts.sum()) for v in variants}
    out["aggregate"] = agg
    print("aggregate:", {k: round(v, 4) for k, v in agg.items()})

    path = os.path.join("runs", args.run_name, "position_analysis.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
