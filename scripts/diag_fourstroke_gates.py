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

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.fourstroke import MachineStrokes                     # noqa: E402
from core.deltamachines import DeltaMachines                   # noqa: E402
from needle_probe import ensure_corpus                         # noqa: E402


def _acc(rec, key, val):
    """Running sum of a (K,)-or-scalar stat as float64 numpy."""
    v = val.detach().double().cpu().numpy() if torch.is_tensor(val) else val
    rec.setdefault(key, 0.0)
    rec[key] = rec[key] + v


def consume(hook, rec, K):
    """Fold one forward's hook record (core diag hook, REAL forward —
    see core/fourstroke.py MachineStrokes.diag) into running stats.
    The hook is equivalence-tested against the unhooked forward; this
    script never reimplements the math (2026-08-13 lesson)."""
    with torch.no_grad():
        g = hook.pop("g")                                 # (B,T,K) final
        B, T, _ = g.shape
        gf = g.float()
        _acc(rec, "n_tok", B * T)
        _acc(rec, "g_sum", gf.sum(dim=(0, 1)))            # (K,)
        _acc(rec, "g_sq", (gf ** 2).sum(dim=(0, 1)))
        _acc(rec, "g_open", (gf > 0.9).double().sum(dim=(0, 1)))
        _acc(rec, "g_shut", (gf < 0.1).double().sum(dim=(0, 1)))
        # router view: per-token sorted gate mass (post any topk mask)
        gs = gf.reshape(-1, K).sort(dim=-1, descending=True).values
        tot = gs.sum(-1).clamp_min(1e-8)
        for kk in (1, 2, 4):
            _acc(rec, f"route_top{kk}", (gs[:, :kk].sum(-1) / tot).sum())
        # conference probs: hook sums over rounds; head-average. Absent
        # when the conference is ablated (v3 probe rungs 1-2).
        attn = hook.pop("attn", None)
        if attn is not None:
            n_r = max(1, hook.pop("attn_rounds", 1))
            p = (attn / n_r).float().mean(dim=1)          # (B*T, K, K)
            _acc(rec, "conf_ent",
                 (-(p.clamp_min(1e-9).log() * p).sum(-1)).sum(dim=0))
            ps = p.sort(dim=-1, descending=True).values
            for kk in (1, 2, 4):
                _acc(rec, f"conf_top{kk}", ps[:, :, :kk].sum(-1).sum(dim=0))
            _acc(rec, "conf_self",
                 p.diagonal(dim1=-2, dim2=-1).sum(dim=0))          # (K,)
        # write-back contribution vs the residual it lands on
        _acc(rec, "out_norm", hook.pop("wb_out_norm"))
        _acc(rec, "x_norm", hook.pop("wb_x_norm"))
        _acc(rec, "wo_norm", hook.pop("wo_norm_k"))                # (K,)
        _acc(rec, "c_norm", hook.pop("c_norm_k"))
        # v2 router: per-round traffic + mid-block wake/sleep events
        routes = hook.pop("routes", [])
        if routes:
            tr = np.stack([(r_ > 0).double().mean(dim=(0, 1))
                           .cpu().numpy() for r_ in routes])       # (R,K)
            _acc(rec, "route_traffic", tr * (B * T))
            if len(routes) > 1:
                w = ((routes[1] > 0) & (routes[0] == 0)).double()
                s_ = ((routes[1] == 0) & (routes[0] > 0)).double()
                _acc(rec, "route_wake",
                     w.mean(dim=(0, 1)).cpu().numpy() * (B * T))
                _acc(rec, "route_sleep",
                     s_.mean(dim=(0, 1)).cpu().numpy() * (B * T))


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
        "writeback_over_x": float(rec["out_norm"] / rec["x_norm"]),
        "writeback_norm_per_machine": (rec["wo_norm"] / n).tolist(),
        "channel_norm_per_machine": (rec["c_norm"] / n).tolist(),
    }
    if "conf_ent" in rec:          # absent when conference is ablated
        out["conf_entropy"] = (rec["conf_ent"] / n).tolist()
        out["conf_entropy_uniform"] = math.log(K)
        out["conf_top_mass"] = {str(kk): (rec[f"conf_top{kk}"] / n).tolist()
                                for kk in (1, 2, 4)}
        out["conf_self_mass"] = (rec["conf_self"] / n).tolist()
    for name, attr in (("addr_mix", "addr_mix"), ("gate_bias", "gate_b")):
        p = getattr(strokes, attr, None)   # absent on ablated probe rungs
        if p is not None:
            out[name] = p.detach().float().cpu().tolist()
    if "route_traffic" in rec:
        out["route_traffic_per_round"] = (rec["route_traffic"] / n).tolist()
        if "route_wake" in rec:
            out["route_wake_frac"] = (rec["route_wake"] / n).tolist()
            out["route_sleep_frac"] = (rec["route_sleep"] / n).tolist()
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

    # v2 (M, MachineStrokes) and v3 (D, DeltaMachines) share the diag
    # hook contract, so one autopsy covers both populations
    blocks = [m for m in model.modules()
              if isinstance(m, (MachineStrokes, DeltaMachines))]
    if not blocks:
        json.dump({"skipped": "no machine-population blocks in this arch"},
                  open(out_path, "w"), indent=2)
        print("[diag_fs] not a fourstroke checkpoint, skipped", flush=True)
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    hooks = [{} for _ in blocks]
    recs = [{} for _ in blocks]
    for b, h in zip(blocks, hooks):
        b.diag = {"rec": h}

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
            for h, r, b in zip(hooks, recs, blocks):
                consume(h, r, b.K)
            print(f"[diag_fs] batch {bi + 1}/{args.batches}", flush=True)

    report = {
        "run": args.run_name,
        "n_tokens": int(recs[0]["n_tok"]),
        "n_machines": blocks[0].K,
        "blocks": [finalize(r, b) for r, b in zip(recs, blocks)],
    }
    json.dump(report, open(out_path, "w"), indent=2)
    for li, blk in enumerate(report["blocks"]):
        conf = ("  conf_ent "
                f"{np.mean(blk['conf_entropy']):.2f}"
                f"/{blk['conf_entropy_uniform']:.2f}"
                if "conf_entropy" in blk else "  conf ablated")
        print(f"[diag_fs] block {li}: gate_mean {blk['gate_mean_overall']:.3f}"
              f"  wb/x {blk['writeback_over_x']:.4f}"
              f"  route_top4 {blk['route_top_mass']['4']:.3f}"
              + conf, flush=True)
    print(f"DIAG_FS_DONE {out_path}", flush=True)


if __name__ == "__main__":
    main()
