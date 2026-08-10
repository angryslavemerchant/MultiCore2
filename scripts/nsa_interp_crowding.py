"""NSA register interp tier 1e: does the null option get CROWDED OUT as
context grows?

Registers act as a learned null option in a forced-choice top-k: a slot
spent on a register block is a real block NOT fetched. But the bank has a
FIXED number of blocks with FIXED keys, competing against a pool of token
blocks that GROWS with position. By order statistics the 12th-highest
token logit rises as that pool grows, so registers should win fewer slots
the deeper into the context a query sits.

If so, the mechanism's abstention capacity decays with sequence length --
a direct problem for scaling T beyond 4096, and an argument for either
scaling n_reg with T or decoupling the null option from the competition
(reserved slots / an explicit logit-space threshold).

Position within the 4096 window is the proxy for effective context length:
a query at t has t/nsa_block token blocks to compete with.

    python scripts/nsa_interp_crowding.py --run-name 124m-nsa-...-t4096
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model import model_from_ckpt_config                   # noqa: E402
from core.nsa import NSARegisterAttention                       # noqa: E402
from scripts.needle_probe import ensure_corpus                  # noqa: E402

BUCKETS = [(256, 512), (512, 1024), (1024, 2048), (2048, 3072),
           (3072, 4096)]


@torch.no_grad()
def crowding(mod, x):
    B, T, C = x.shape
    nb, H = mod.nb, mod.n_head
    hd = C // H
    NB, NR = T // nb, mod.n_reg // nb
    q, k, _ = mod._qkv_nope(x)
    k_reg, _ = mod._reg_kv(B)
    k_blk = k.view(B, H, NB, nb, hd).mean(3)
    k_rblk = k_reg.view(B, H, NR, nb, hd).mean(3)
    K_cmp = torch.cat((k_rblk, k_blk), dim=2)
    scores = q @ K_cmp.transpose(-1, -2) / math.sqrt(hd)
    blk_id = torch.arange(T, device=x.device) // nb
    tok_ok = torch.arange(NB, device=x.device) < blk_id.unsqueeze(-1)
    allowed = torch.cat((tok_ok.new_ones(T, NR), tok_ok), dim=1)
    p = torch.softmax(scores.masked_fill(~allowed, float("-inf")),
                      dim=-1).float()
    mass = p[0, :, :, :NR].sum(-1).mean(0)                # (T,) over heads
    sel = p.sum(1)[0]
    idx = sel.topk(mod.topk, dim=-1).indices
    vis = torch.zeros_like(sel, dtype=torch.bool).scatter_(
        -1, idx, True) & allowed
    nreg = vis[:, :NR].sum(-1).float()                    # (T,)
    out = {}
    for lo, hi in BUCKETS:
        sl = slice(lo, min(hi, T))
        out[f"{lo}-{hi}"] = (float(nreg[sl].mean()),
                             float(mass[sl].mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--docs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(os.path.join("runs", args.run_name, args.ckpt),
                      map_location="cpu")
    cfg, model = model_from_ckpt_config(ckpt["config"])
    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(device).eval()
    mods = [m for m in model.modules()
            if isinstance(m, NSARegisterAttention)]
    corpus = ensure_corpus()
    rng = np.random.default_rng(args.seed)
    T = cfg.block_size

    acc = [[] for _ in mods]
    for d in range(args.docs):
        s = rng.integers(0, len(corpus) - T)
        x = torch.from_numpy(
            np.asarray(corpus[s:s + T]).astype(np.int64))[None].to(device)
        caught, hooks = {}, []
        for i, m in enumerate(mods):
            hooks.append(m.register_forward_pre_hook(
                lambda mod, inp, i=i: caught.__setitem__(i, inp[0].detach())))
        with torch.no_grad():
            if device == "cuda":
                dt = (torch.bfloat16 if torch.cuda.is_bf16_supported()
                      else torch.float16)
                with torch.autocast("cuda", dtype=dt):
                    model(x, x)
            else:
                model(x, x)
        for h in hooks:
            h.remove()
        for i, m in enumerate(mods):
            acc[i].append(crowding(m, caught[i].float()))
            del caught[i]
        print(f"[crowd] doc {d + 1}/{args.docs}", flush=True)

    keys = [f"{lo}-{hi}" for lo, hi in BUCKETS]
    print("\n  register blocks winning slc slots, by query position "
          f"(of topk)   [cmp mass in brackets]")
    print(f"{'':5} " + "  ".join(f"{k:>16}" for k in keys))
    out = []
    for i, m in enumerate(mods):
        row = {k: (float(np.mean([a[k][0] for a in acc[i]])),
                   float(np.mean([a[k][1] for a in acc[i]]))) for k in keys}
        out.append({"layer": i, "topk": m.topk,
                    "buckets": {k: {"n_reg_slots": v[0], "cmp_mass": v[1]}
                                for k, v in row.items()}})
        print(f"[L{i:>2}] " + "  ".join(
            f"{row[k][0]:>6.2f} [{row[k][1]:.2f}]".rjust(16) for k in keys),
            flush=True)

    path = os.path.join("runs", args.run_name, "nsa_interp_crowding.json")
    with open(path, "w") as f:
        json.dump({"run": args.run_name, "docs": args.docs,
                   "layers": out}, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
