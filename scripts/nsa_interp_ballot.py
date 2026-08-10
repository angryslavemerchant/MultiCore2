"""NSA register interp tier 1d: does per-head vote weighting actually
change WHICH REAL TOKEN BLOCKS get fetched?

The objection to the control-plane reading is sound on its face: mass that
routes to a register block routes to near-zero values, so it delivers
nothing. The claim under test is not about delivery, it is about the
cross-head ballot:

    sel_j = sum_h (1 - mass_h) * ptilde^h_j

where ptilde^h is head h's token-only distribution. Registers make heads
vote with UNEQUAL weight on a decision they share. Vanilla attention has
no analogue -- heads are concatenated, not summed, so a per-head gain
stays inside that head's own output.

This isolates that effect from the slot-displacement effect. For every
query we take n = (number of token blocks the real ballot actually
selected), then re-run the selection over token blocks ONLY with every
head weighted equally, taking the same n. Any difference in the resulting
block set is attributable purely to head reweighting.

    overlap = 1.0  -> reweighting is inert, the objection is right
    overlap < 1.0  -> registers steer which real tokens get fetched

    python scripts/nsa_interp_ballot.py --run-name 124m-nsa-...-t4096
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


@torch.no_grad()
def ballot(mod, x, win):
    B, T, C = x.shape
    nb, H = mod.nb, mod.n_head
    hd = C // H
    NB, NR = T // nb, mod.n_reg // nb
    q, k, _ = mod._qkv_nope(x)
    k_reg, _ = mod._reg_kv(B)
    k_blk = k.view(B, H, NB, nb, hd).mean(3)
    k_rblk = k_reg.view(B, H, NR, nb, hd).mean(3)
    K_cmp = torch.cat((k_rblk, k_blk), dim=2)
    q = q[:, :, T - win:, :]
    scores = q @ K_cmp.transpose(-1, -2) / math.sqrt(hd)
    blk_id = torch.arange(T - win, T, device=x.device) // nb
    tok_ok = torch.arange(NB, device=x.device) < blk_id.unsqueeze(-1)
    allowed = torch.cat((tok_ok.new_ones(win, NR), tok_ok), dim=1)
    p = torch.softmax(scores.masked_fill(~allowed, float("-inf")),
                      dim=-1).float()                     # (1,H,win,NR+NB)

    # ---- the real ballot -------------------------------------------
    sel = p.sum(1)[0]                                     # (win, NR+NB)
    idx = sel.topk(mod.topk, dim=-1).indices
    vis = torch.zeros_like(sel, dtype=torch.bool).scatter_(-1, idx, True)
    vis = vis & allowed
    tok_true = vis[:, NR:]                                # (win, NB)
    n_tok = tok_true.sum(-1)                              # per query

    # ---- the equal-weight ballot: renormalise each head to the token
    # half so every head votes with weight exactly 1 ------------------
    p_tok = p[0, :, :, NR:]                               # (H,win,NB)
    p_tilde = p_tok / p_tok.sum(-1, keepdim=True).clamp(min=1e-9)
    sel_flat = p_tilde.sum(0)                             # (win, NB)
    sel_flat = sel_flat.masked_fill(~tok_ok, -1.0)

    # take the SAME number of token blocks per query as the real ballot
    order = sel_flat.argsort(dim=-1, descending=True)
    ar = torch.arange(order.shape[-1], device=x.device).unsqueeze(0)
    keep = ar < n_tok.unsqueeze(-1)
    tok_flat = torch.zeros_like(tok_true).scatter_(
        -1, order, keep)

    ok = n_tok > 0
    inter = (tok_true & tok_flat).sum(-1).float()[ok]
    n = n_tok.float()[ok]
    return {
        "overlap": float((inter / n).mean()),
        "frac_queries_changed": float(((inter < n).float()).mean()),
        "mean_n_tok": float(n.mean()),
        "mean_blocks_swapped": float((n - inter).mean()),
        "frac_queries_no_tokens": float((~ok).float().mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--docs", type=int, default=8)
    ap.add_argument("--window", type=int, default=512)
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
    T, win = cfg.block_size, args.window

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
            acc[i].append(ballot(m, caught[i].float(), win))
            del caught[i]
        print(f"[ballot] doc {d + 1}/{args.docs}", flush=True)

    out = []
    print("\n     equal-weight ballot vs real ballot (same #token blocks)")
    for i, m in enumerate(mods):
        st = {k: float(np.mean([a[k] for a in acc[i]]))
              for k in acc[i][0]}
        st["layer"] = i
        out.append(st)
        print(f"[L{i:>2}] overlap {st['overlap']:.3f}   queries changed "
              f"{st['frac_queries_changed']:.1%}   blocks swapped "
              f"{st['mean_blocks_swapped']:.2f} of {st['mean_n_tok']:.1f}"
              f"   (no-token queries {st['frac_queries_no_tokens']:.0%})",
              flush=True)

    path = os.path.join("runs", args.run_name, "nsa_interp_ballot.json")
    with open(path, "w") as f:
        json.dump({"run": args.run_name, "docs": args.docs,
                   "window": win, "layers": out}, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
