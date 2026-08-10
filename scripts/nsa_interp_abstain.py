"""NSA register interp tier 1c: is drained mass a FIXED per-head gain
knob, or a query-dependent abstention signal?

Tier 1 measured the SHAPE of the register distribution (near-constant
across queries). It never measured the TOTAL drained mass per head, which
is the quantity the control-plane reading actually rests on:

    sel = p.sum(over heads) -> topk

so head h's token-block votes sum to 1 - mass_h, i.e. mass IS that head's
abstention weight in the shared selection ballot. If mass_h barely varies
with the query, registers are a fixed gain knob and the abstention story
is wrong. If it varies a lot, the story holds.

CONFOUND: mass falls with position by construction -- a query at t=200
has 6 token blocks to compete with the registers, one at t=4000 has 125.
So raw variance is mostly positional and proves nothing. This decomposes:

    var_pos      : variance of the per-position mean across DIFFERENT
                   documents -- the positional trend (uninteresting)
    var_content  : variance across documents AT A FIXED POSITION -- same
                   position, different text, so this is purely
                   content-driven (the number that matters)

Also measures channel 2 (adaptive k): how many of the topk slc slots go
to register blocks per query, with the same decomposition.

    python scripts/nsa_interp_abstain.py --run-name 124m-nsa-...-t4096
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
def mass_and_k(mod, x, win):
    """(mass per head, n register blocks selected) over the last `win`
    query positions. mass: (H, win); nregsel: (win,)"""
    B, T, C = x.shape
    nb, H = mod.nb, mod.n_head
    hd = C // H
    NB, NR = T // nb, mod.n_reg // nb
    q, k, v = mod._qkv_nope(x)
    k_reg, _ = mod._reg_kv(B)
    k_blk = k.view(B, H, NB, nb, hd).mean(3)
    k_rblk = k_reg.view(B, H, NR, nb, hd).mean(3)
    K_cmp = torch.cat((k_rblk, k_blk), dim=2)
    q = q[:, :, T - win:, :]                       # only the window
    scores = q @ K_cmp.transpose(-1, -2) / math.sqrt(hd)
    blk_id = torch.arange(T - win, T, device=x.device) // nb
    tok_ok = torch.arange(NB, device=x.device) < blk_id.unsqueeze(-1)
    allowed = torch.cat((tok_ok.new_ones(win, NR), tok_ok), dim=1)
    p = torch.softmax(scores.masked_fill(~allowed, float("-inf")),
                      dim=-1).float()              # (1,H,win,NR+NB)
    mass = p[0, :, :, :NR].sum(-1)                 # (H,win)
    sel = p.sum(1)                                 # (1,win,NR+NB)
    idx = sel.topk(mod.topk, dim=-1).indices
    vis = torch.zeros_like(sel, dtype=torch.bool).scatter_(
        -1, idx, True) & allowed
    nregsel = vis[0, :, :NR].sum(-1).float()       # (win,)
    return mass.cpu(), nregsel.cpu()


def decompose(M):
    """M: (D, ...) stacked over documents, sharing positions.
    Returns (mu, sd_content, sd_pos) where sd_content is the spread at a
    FIXED position across documents."""
    mu = float(M.mean())
    sd_content = float(M.var(0, unbiased=True).mean().sqrt())
    sd_pos = float(M.mean(0).std(unbiased=True))
    return mu, sd_content, sd_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--docs", type=int, default=8)
    ap.add_argument("--window", type=int, default=512,
                    help="last N query positions (limits positional drift)")
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

    # documents are independent samples AT THE SAME positions
    masses = [[] for _ in mods]
    kregs = [[] for _ in mods]
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
            mm, kk = mass_and_k(m, caught[i].float(), win)
            masses[i].append(mm)
            kregs[i].append(kk)
            del caught[i]
        print(f"[abstain] doc {d + 1}/{args.docs}", flush=True)

    out = []
    print(f"\n{'':4} {'mass':>26}  {'CV':>6}   "
          f"{'#reg blocks of topk':>22}")
    for i, m in enumerate(mods):
        M = torch.stack(masses[i])                 # (D,H,win)
        K = torch.stack(kregs[i])                  # (D,win)
        mu, sdc, sdp = decompose(M)
        # per-head: which head abstains most variably?
        per_head = [decompose(M[:, h]) for h in range(M.shape[1])]
        cvs = [c / max(u, 1e-9) for u, c, _ in per_head]
        head_mus = [u for u, _, _ in per_head]
        kmu, ksdc, ksdp = decompose(K)
        st = {"layer": i, "mass_mean": mu, "mass_sd_content": sdc,
              "mass_sd_pos": sdp, "mass_cv_content": sdc / max(mu, 1e-9),
              "head_cv_max": max(cvs), "head_cv_min": min(cvs),
              "head_mean_spread": float(np.std(head_mus)),
              "head_mean_min": min(head_mus), "head_mean_max": max(head_mus),
              "kreg_mean": kmu, "kreg_sd_content": ksdc,
              "kreg_sd_pos": ksdp, "topk": m.topk,
              "kreg_frac_zero": float((K == 0).float().mean()),
              "kreg_frac_all": float((K == m.topk).float().mean())}
        out.append(st)
        print(f"[L{i:>2}] {mu:.3f} +-{sdc:.3f}(content) "
              f"+-{sdp:.3f}(pos)  CV {sdc / max(mu, 1e-9):.3f}  "
              f"head CV {min(cvs):.2f}-{max(cvs):.2f}  "
              f"head mu {min(head_mus):.2f}-{max(head_mus):.2f}   "
              f"kreg {kmu:.1f}/{m.topk} +-{ksdc:.2f}(content) "
              f"[0:{st['kreg_frac_zero']:.0%} all:{st['kreg_frac_all']:.0%}]",
              flush=True)

    path = os.path.join("runs", args.run_name, "nsa_interp_abstain.json")
    with open(path, "w") as f:
        json.dump({"run": args.run_name, "docs": args.docs,
                   "window": win, "layers": out}, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
