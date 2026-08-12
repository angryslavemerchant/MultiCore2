"""Four-stroke gate + conference autopsy: is the population alive, and is
it already a top-k router?

    python scripts/diag_fourstroke_gates.py --run-name d512-4stroke-...

Routes real FineWeb sequences through the trained checkpoint with every
MachineStrokes block instrumented, and reports PER BLOCK / PER MACHINE what
the loss curve hides:

  gates      sigmoid write-back openness g (B,T,K): per-machine mean/std,
             frac saturated open (>0.9) / shut (<0.1). Nine zero-init
             write-backs in series could reproduce the hier "memory dead"
             signature — this is the tripwire, run post-hoc.
  router     top-k concentration of the gate vector per token: mass of the
             top-1/2/4 gates over the sum of all K. If top-4 holds ~all the
             mass, the trained gates ARE a router and v2 top-k pruning is a
             prune, not a retrain.
  conference K x K attention: entropy, top-1/2/4 mass and self-mass per
             query machine. Diffuse rows = the conference genuinely mixes;
             peaky rows = machines mostly talk to themselves (conference
             prunable too).
  writeback  ||sum_k g_k W_O(c_k)|| vs ||x||: how much the population
             actually contributes to the residual, per block.
  channel    ||c_k|| per machine: dead machines show as norm outliers.
  params     addr_mix and gate bias per machine (static).

Non-fourstroke checkpoints write {"skipped": ...} and exit 0 so
bench_all_ckpts stays idempotent. Results: runs/<run>/fourstroke_gates.json.
"""
import argparse
import json
import math
import os
import sys
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.fourstroke import MachineStrokes                     # noqa: E402
from needle_probe import ensure_corpus                         # noqa: E402


def _acc(rec, key, val):
    """Running sum of a (K,)-or-scalar stat as float64 numpy."""
    v = val.detach().double().cpu().numpy() if torch.is_tensor(val) else val
    rec.setdefault(key, 0.0)
    rec[key] = rec[key] + v


def instrumented_forward(self, x, c_prev, rec):
    """MachineStrokes.forward with stats taps. Same math as the module
    (conference y computed from the explicit softmax so the recorded
    probabilities are exactly the ones used)."""
    B, T, C = x.shape
    K, d = self.K, self.d
    s = self.backend(x, c_prev)                       # stroke 1
    sn = self.ln_iface(s)
    q, ks, v = self.w_qkv(sn).chunk(3, dim=-1)        # stroke 2
    k = (self.anchor[None, :, None, :]
         + self.addr_mix[None, :, None, None] * ks)

    def heads(t):
        return (t.permute(0, 2, 1, 3)
                .reshape(B * T, K, self.H, self.hd).transpose(1, 2))

    qh, kh, vh = heads(q), heads(k), heads(v)
    attn = torch.softmax(qh @ kh.transpose(-2, -1) / math.sqrt(self.hd),
                         dim=-1)                      # (B*T, H, K, K)
    y = attn @ vh
    y = (y.transpose(1, 2).reshape(B, T, K, d).permute(0, 2, 1, 3))
    c = s + self.conf_out(y)                          # stroke 3
    g = torch.sigmoid(                                # stroke 4
        torch.einsum("btc,kc->btk", x, self.gate_x)
        + torch.einsum("bktd,kd->btk", c, self.gate_c)
        + self.gate_b)
    oc = self.w_o(c)                                  # (B,K,T,C)
    out = torch.einsum("btk,bktc->btc", g, oc)

    with torch.no_grad():
        gf = g.float()
        _acc(rec, "n_tok", B * T)
        _acc(rec, "g_sum", gf.sum(dim=(0, 1)))            # (K,)
        _acc(rec, "g_sq", (gf ** 2).sum(dim=(0, 1)))
        _acc(rec, "g_open", (gf > 0.9).double().sum(dim=(0, 1)))
        _acc(rec, "g_shut", (gf < 0.1).double().sum(dim=(0, 1)))
        # router view: per-token sorted gate mass
        gs = gf.reshape(-1, K).sort(dim=-1, descending=True).values
        tot = gs.sum(-1).clamp_min(1e-8)
        for kk in (1, 2, 4):
            _acc(rec, f"route_top{kk}",
                 (gs[:, :kk].sum(-1) / tot).sum())
        # conference stats, averaged over heads: (B*T, K, K)
        p = attn.float().mean(dim=1)
        _acc(rec, "conf_ent",
             (-(p.clamp_min(1e-9).log() * p).sum(-1)).sum(dim=0))  # (K,)
        ps = p.sort(dim=-1, descending=True).values
        for kk in (1, 2, 4):
            _acc(rec, f"conf_top{kk}", ps[:, :, :kk].sum(-1).sum(dim=0))
        _acc(rec, "conf_self",
             p.diagonal(dim1=-2, dim2=-1).sum(dim=0))              # (K,)
        # write-back contribution vs the residual it lands on
        _acc(rec, "out_norm", out.float().norm(dim=-1).sum())
        _acc(rec, "x_norm", x.float().norm(dim=-1).sum())
        _acc(rec, "wo_norm",                                       # (K,)
             (gf.permute(0, 2, 1)[..., None] * oc.float())
             .norm(dim=-1).sum(dim=(0, 2)))
        _acc(rec, "c_norm", c.float().norm(dim=-1).sum(dim=(0, 2)))
    return out, c


def finalize(rec, strokes):
    n = float(rec["n_tok"])
    K = strokes.K
    mean = rec["g_sum"] / n
    var = np.maximum(rec["g_sq"] / n - mean ** 2, 0.0)
    out = {
        "gate_mean": mean.tolist(),
        "gate_std": np.sqrt(var).tolist(),
        "gate_frac_open": (rec["g_open"] / n).tolist(),
        "gate_frac_shut": (rec["g_shut"] / n).tolist(),
        "gate_mean_overall": float(mean.mean()),
        "route_top_mass": {str(kk): float(rec[f"route_top{kk}"] / n)
                           for kk in (1, 2, 4)},
        "conf_entropy": (rec["conf_ent"] / n).tolist(),
        "conf_entropy_uniform": math.log(K),
        "conf_top_mass": {str(kk): (rec[f"conf_top{kk}"] / n).tolist()
                          for kk in (1, 2, 4)},
        "conf_self_mass": (rec["conf_self"] / n).tolist(),
        "writeback_over_x": float(rec["out_norm"] / rec["x_norm"]),
        "writeback_norm_per_machine": (rec["wo_norm"] / n).tolist(),
        "channel_norm_per_machine": (rec["c_norm"] / n).tolist(),
        "addr_mix": strokes.addr_mix.detach().float().cpu().tolist(),
        "gate_bias": strokes.gate_b.detach().float().cpu().tolist(),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--micro-bs", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    args, _ = ap.parse_known_args()

    out_path = os.path.join("runs", args.run_name, "fourstroke_gates.json")
    ckpt_path = os.path.join("runs", args.run_name, args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    from core.model import model_from_ckpt_config
    cfg, model = model_from_ckpt_config(ckpt["config"])
    sd = ckpt.get("model", ckpt.get("model_state", ckpt))
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd)

    blocks = [m for m in model.modules() if isinstance(m, MachineStrokes)]
    if not blocks:
        json.dump({"skipped": "no MachineStrokes blocks in this arch"},
                  open(out_path, "w"), indent=2)
        print("[diag_fs] not a fourstroke checkpoint, skipped", flush=True)
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    recs = [{} for _ in blocks]
    for i, b in enumerate(blocks):
        b.forward = MethodType(
            (lambda self, x, c, _r=recs[i]:
             instrumented_forward(self, x, c, _r)), b)

    corpus = ensure_corpus()
    T = min(args.seq_len, cfg.block_size)
    rng = np.random.default_rng(args.seed)
    amp = (torch.autocast("cuda", dtype=torch.bfloat16)
           if device == "cuda" else torch.autocast("cpu", enabled=False))
    with torch.no_grad():
        for bi in range(args.batches):
            starts = rng.integers(0, len(corpus) - T - 1, args.micro_bs)
            x = np.stack([np.asarray(corpus[s:s + T], dtype=np.int64)
                          for s in starts])
            with amp:
                model(torch.from_numpy(x).to(device))
            print(f"[diag_fs] batch {bi + 1}/{args.batches}", flush=True)

    report = {
        "run": args.run_name,
        "n_tokens": int(recs[0]["n_tok"]),
        "n_machines": blocks[0].K,
        "blocks": [finalize(r, b) for r, b in zip(recs, blocks)],
    }
    json.dump(report, open(out_path, "w"), indent=2)
    for li, blk in enumerate(report["blocks"]):
        print(f"[diag_fs] block {li}: gate_mean {blk['gate_mean_overall']:.3f}"
              f"  wb/x {blk['writeback_over_x']:.4f}"
              f"  route_top4 {blk['route_top_mass']['4']:.3f}"
              f"  conf_ent {np.mean(blk['conf_entropy']):.2f}"
              f"/{blk['conf_entropy_uniform']:.2f}", flush=True)
    print(f"DIAG_FS_DONE {out_path}", flush=True)


if __name__ == "__main__":
    main()
