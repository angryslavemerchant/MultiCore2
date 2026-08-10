"""NSA register interp tier 1b: does ANY individual register escape the
sink role, at ANY layer?

Tier 1 reported per-layer aggregates. A mean of 0.02 is compatible with
"all 1024 registers are dead" AND with "1000 are dead and 24 carry real
content" -- very different architectural readings. This looks at the
per-register distributions:

  * value-norm TAIL: max / p99 / median of ||v_reg|| against ||v_tok||.
    A live memory slot would show up as a register whose value norm is
    in the token range.
  * slc attention at INDIVIDUAL register granularity (cmp only ever sees
    32-register block means; slc attends to registers one by one, so it
    is the only place per-register identity can express itself).
  * per-register usage concentration: effective number of registers used
    (exp entropy) and top-1 share, per layer.
  * cosine structure WITHIN the bank: are registers a few clusters or a
    continuum?

    python scripts/nsa_interp_reg.py --run-name 124m-nsa-...-t4096
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
def reg_detail(mod, x):
    B, T, C = x.shape
    nb, H = mod.nb, mod.n_head
    hd = C // H
    NB, NR = T // nb, mod.n_reg // nb
    R = mod.n_reg
    q, k, v = mod._qkv_nope(x)
    k_reg, v_reg = mod._reg_kv(B)

    # ---- per-register value norms vs per-token value norms ------------
    vr = v_reg[0].transpose(0, 1).reshape(R, -1).norm(dim=-1)     # (R,)
    vt = v[0].transpose(0, 1).reshape(T, -1).norm(dim=-1)         # (T,)
    vt_med = float(vt.median())

    # ---- cmp block scores (same maths as forward) ---------------------
    k_blk = k.view(B, H, NB, nb, hd).mean(3)
    k_rblk = k_reg.view(B, H, NR, nb, hd).mean(3)
    K_cmp = torch.cat((k_rblk, k_blk), dim=2)
    scores = q @ K_cmp.transpose(-1, -2) / math.sqrt(hd)
    blk_id = torch.arange(T, device=x.device) // nb
    tok_ok = torch.arange(NB, device=x.device) < blk_id.unsqueeze(-1)
    allowed = torch.cat((tok_ok.new_ones(T, NR), tok_ok), dim=1)
    p = torch.softmax(scores.masked_fill(~allowed, float("-inf")),
                      dim=-1).float()
    sel = p.detach().sum(1)                                       # (B,T,*)
    idx = sel.topk(mod.topk, dim=-1).indices
    vis = torch.zeros_like(sel, dtype=torch.bool).scatter_(
        -1, idx, True) & allowed                                  # (B,T,*)

    # ---- slc attention at INDIVIDUAL register granularity -------------
    # (only over the register half; token half not needed here)
    a = (q[0] @ k_reg[0].transpose(-1, -2)) / math.sqrt(hd)        # (H,T,R)
    m_reg = vis[0, :, :NR].repeat_interleave(nb, dim=-1)           # (T,R)
    # mask to visible registers only, then softmax over the register set
    a = a.masked_fill(~m_reg.unsqueeze(0), float("-inf")).float()
    live = m_reg.any(-1)                                           # (T,)
    a = torch.softmax(a, dim=-1)
    a = torch.nan_to_num(a, nan=0.0)[:, live, :]                   # (H,T',R)
    use = a.mean((0, 1))                                           # (R,)
    use = use / use.sum().clamp(min=1e-9)
    ent = float(-(use * use.clamp(min=1e-9).log()).sum())

    # ---- structure within the bank ------------------------------------
    Rm = torch.nn.functional.normalize(mod.registers.float(), dim=-1)
    cs = (Rm @ Rm.t())
    off = cs[~torch.eye(R, dtype=torch.bool, device=cs.device)]

    return {
        "v_reg_med": float(vr.median()), "v_reg_max": float(vr.max()),
        "v_reg_p99": float(vr.kthvalue(int(0.99 * R)).values),
        "v_tok_med": vt_med, "v_tok_p01": float(vt.kthvalue(
            max(1, int(0.01 * T))).values),
        # how many registers reach even 1/4 of the median token value norm
        "n_reg_over_quarter_tok": int((vr > 0.25 * vt_med).sum()),
        "n_reg_over_tok_p01": int((vr > vt.kthvalue(
            max(1, int(0.01 * T))).values).sum()),
        "slc_eff_registers": float(math.exp(ent)),
        "slc_top1_share": float(use.max()),
        "slc_top16_share": float(use.topk(16).values.sum()),
        "bank_cos_mean": float(off.mean()), "bank_cos_absmean":
            float(off.abs().mean()), "bank_cos_max": float(off.max()),
        "n_reg": R,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--ckpt", default="best.pt")
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

    out = []
    for i, m in enumerate(mods):
        st = reg_detail(m, caught[i].float())
        st["layer"] = i
        out.append(st)
        print(f"[L{i:>2}] v_reg med/p99/max "
              f"{st['v_reg_med']:.2f}/{st['v_reg_p99']:.2f}/"
              f"{st['v_reg_max']:.2f}  vs v_tok med {st['v_tok_med']:.1f} "
              f"(p01 {st['v_tok_p01']:.1f})  |  #reg>0.25*tok "
              f"{st['n_reg_over_quarter_tok']:>4}  #reg>tok_p01 "
              f"{st['n_reg_over_tok_p01']:>4}  |  slc eff_reg "
              f"{st['slc_eff_registers']:>6.1f}/{st['n_reg']} top1 "
              f"{st['slc_top1_share']:.3f} top16 {st['slc_top16_share']:.3f}"
              f"  |  bank |cos| {st['bank_cos_absmean']:.3f} max "
              f"{st['bank_cos_max']:.3f}", flush=True)
        del caught[i]

    path = os.path.join("runs", args.run_name, "nsa_interp_reg.json")
    with open(path, "w") as f:
        json.dump({"run": args.run_name, "layers": out}, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
