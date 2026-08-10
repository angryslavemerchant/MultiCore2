"""NSA register interpretability, tier 1: what are the registers DOING?

The ON/OFF pseudo-ablation proved registers are load-bearing but confounds
three mechanisms (content removed / mass redistributed / scores shifted).
This script splits the first fork WITHOUT any behavioural run, from the
weights plus a single forward pass on real text:

  1. QUERY-CONDITIONALITY. Normalise the cmp attention distribution over
     register blocks only, per (batch, head, query). If every query uses
     the same distribution, the registers are a learned CONSTANT added to
     every token -- a bias/sink. If queries differ, they are a
     content-addressed dictionary and there is something to interpret.
     Reported as mean cosine to the per-head mean distribution (1.0 =
     pure sink) and as the fraction of distribution variance NOT
     explained by that mean (0.0 = pure sink).

  2. VALUE NORMS. The textbook sink signature (streaming-LLM): sinks hold
     near-zero VALUE vectors, so mass parked on them is a no-op that only
     drains the softmax. Compare ||v_reg|| to ||v_tok|| at the block-
     summary granularity cmp actually reads.

  3. EFFECTIVE RANK. SVD participation ratio of the (n_reg, C) register
     matrix. If 1024 registers span ~3 directions they are a handful of
     sinks wearing a big coat, and n_reg is a free speed lever.

  4. BLOCK USAGE. Effective number of register blocks used (exp of the
     entropy of the mean distribution) and the top block's share -- does
     the bank participate, or do 2 of 32 blocks do all the work?

    python scripts/nsa_interp.py --run-name 124m-nsa-...-t4096

Writes runs/<run-name>/nsa_interp.json.
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


def eff_rank(M):
    """Participation ratio of squared singular values: (sum s^2)^2 /
    sum s^4. Equals r for r equal directions, 1 for a rank-1 matrix."""
    s = torch.linalg.svdvals(M.float())
    s2 = s ** 2
    return float(s2.sum() ** 2 / (s2 ** 2).sum())


@torch.no_grad()
def layer_stats(mod, x):
    """Recompute the cmp branch externally (same maths as the module's
    forward) and measure the registers' role in it."""
    B, T, C = x.shape
    nb, H = mod.nb, mod.n_head
    hd = C // H
    NB, NR = T // nb, mod.n_reg // nb
    q, k, v = mod._qkv_nope(x)
    k_reg, v_reg = mod._reg_kv(B)

    k_blk = k.view(B, H, NB, nb, hd).mean(3)
    v_blk = v.view(B, H, NB, nb, hd).mean(3)
    k_rblk = k_reg.view(B, H, NR, nb, hd).mean(3)
    v_rblk = v_reg.view(B, H, NR, nb, hd).mean(3)
    K_cmp = torch.cat((k_rblk, k_blk), dim=2)

    scores = q @ K_cmp.transpose(-1, -2) / math.sqrt(hd)
    blk_id = torch.arange(T, device=x.device) // nb
    tok_ok = torch.arange(NB, device=x.device) < blk_id.unsqueeze(-1)
    allowed = torch.cat((tok_ok.new_ones(T, NR), tok_ok), dim=1)
    p = torch.softmax(scores.masked_fill(~allowed, float("-inf")),
                      dim=-1).float()                    # (B,H,T,NR+NB)

    p_reg = p[..., :NR]
    mass = p_reg.sum(-1)                                 # (B,H,T)

    # --- query-conditionality: shape of the register distribution ------
    # drop the first block's queries (no token keys -> degenerate row)
    ph = p_reg[:, :, nb:, :]
    ph = ph / ph.sum(-1, keepdim=True).clamp(min=1e-9)   # (B,H,T',NR)
    mean_h = ph.mean(2, keepdim=True)                    # per-head mean
    cos = torch.nn.functional.cosine_similarity(ph, mean_h, dim=-1)
    # fraction of total variation about the mean (0 = every query same)
    resid = ((ph - mean_h) ** 2).sum(-1).mean()
    total = (ph ** 2).sum(-1).mean()

    mh = mean_h.squeeze(2)                               # (B,H,NR)
    ent = -(mh * mh.clamp(min=1e-9).log()).sum(-1)

    return {
        "reg_cmp_mass": float(mass.mean()),
        "reg_cmp_mass_max_head": float(mass.mean((0, 2)).max()),
        # 1.0 => every query attends to registers identically (a sink)
        "query_cos_to_mean": float(cos.mean()),
        "query_var_frac": float(resid / total.clamp(min=1e-9)),
        # sink signature: near-zero values
        "v_reg_norm": float(v_rblk.norm(dim=-1).mean()),
        "v_tok_norm": float(v_blk.norm(dim=-1).mean()),
        "k_reg_norm": float(k_rblk.norm(dim=-1).mean()),
        "k_tok_norm": float(k_blk.norm(dim=-1).mean()),
        # how much of the bank participates
        "eff_reg_blocks": float(ent.exp().mean()),
        "n_reg_blocks": NR,
        "top_block_share": float(mh.max(-1).values.mean()),
        "eff_rank_registers": eff_rank(mod.registers),
        "n_reg": mod.n_reg,
        "d": C,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(os.path.join("runs", args.run_name, args.ckpt),
                      map_location="cpu")
    cfg, model = model_from_ckpt_config(ckpt["config"])
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    assert not unexpected, unexpected
    assert all(m.endswith(".gain") for m in missing), missing
    model = model.to(device).eval()

    mods = [m for m in model.modules()
            if isinstance(m, NSARegisterAttention)]
    assert mods, "no NSA layers in this checkpoint"
    print(f"[interp] {len(mods)} NSA layers, T={cfg.block_size}", flush=True)

    corpus = ensure_corpus()
    rng = np.random.default_rng(args.seed)
    T = cfg.block_size
    starts = rng.integers(0, len(corpus) - T, size=args.batch)
    x = torch.from_numpy(np.stack(
        [np.asarray(corpus[s:s + T]) for s in starts]).astype(np.int64)
    ).to(device)

    # capture each NSA layer's INPUT (post-norm hidden state)
    caught, hooks = {}, []
    for i, m in enumerate(mods):
        hooks.append(m.register_forward_pre_hook(
            lambda mod, inp, i=i: caught.__setitem__(i, inp[0].detach())))
    dt = (torch.bfloat16 if device == "cuda"
          and torch.cuda.is_bf16_supported() else torch.float16)
    with torch.no_grad():
        if device == "cuda":
            with torch.autocast("cuda", dtype=dt):
                model(x, x)
        else:
            model(x, x)
    for h in hooks:
        h.remove()

    out = []
    for i, m in enumerate(mods):
        st = layer_stats(m, caught[i].float())
        st["layer"] = i
        out.append(st)
        print(f"[L{i:>2}] mass {st['reg_cmp_mass']:.3f}  "
              f"cos_to_mean {st['query_cos_to_mean']:.4f}  "
              f"var_frac {st['query_var_frac']:.4f}  "
              f"v_reg/v_tok {st['v_reg_norm'] / st['v_tok_norm']:.3f}  "
              f"eff_blocks {st['eff_reg_blocks']:.1f}/{st['n_reg_blocks']}  "
              f"eff_rank {st['eff_rank_registers']:.1f}/{st['n_reg']}",
              flush=True)
        del caught[i]

    path = os.path.join("runs", args.run_name, "nsa_interp.json")
    with open(path, "w") as f:
        json.dump({"run": args.run_name, "layers": out}, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
