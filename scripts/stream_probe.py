"""Stream-probe eval: grade a checkpoint on generated lifetimes.

For each lifetime the model does one forward pass; every grade site's
payload NLL is read off the same logits. Update sites get one extra
truncated pass ([prefix + stale-variant recur doc]) to grade the stale
value under identical conditions.

Aggregation (per condition group, bucketed by ACTUAL distance):
  real vs ghost delta  - the headline retrieval signal per cue
  update margin        - stale NLL minus live NLL (positive = the model
                         tracks the latest value)
  collide penalty      - collide-site NLL minus clean fuzzy at the same
                         distance bucket

Writes runs/<name>/stream_probe.json.
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model import GPT, GPTConfig                      # noqa: E402
from stream_probe.generate import (Lifetime, ProbeConfig,  # noqa: E402
                                   Tok, load_bank)

BUCKETS = (128, 256, 512, 1024, 2048, 3584)


def bucket_of(d):
    return min(BUCKETS, key=lambda b: abs(b - d))


@torch.no_grad()
def site_nlls(model, lifetimes, device, batch_size=4):
    """Yield (site, nll, stale_nll|None) across all lifetimes."""
    seqs = []   # (ids, [(uid, site, span, variant), ...])
    out = {}    # uid -> [site, live_nll, stale_nll]
    uid = 0
    for lt in lifetimes:
        tagged = []
        for s in lt.sites:
            out[uid] = [s, None, None]
            tagged.append((uid, s, s["span"], "live"))
            if s["update"]:
                alt = lt.ids[:s["doc_off"]] + s["alt_ids"]
                span = (s["doc_off"] + s["alt_span"][0],
                        s["doc_off"] + s["alt_span"][1])
                seqs.append((alt, [(uid, s, span, "stale")]))
            uid += 1
        seqs.append((lt.ids, tagged))
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i:i + batch_size]
        L = max(len(ids) for ids, _ in chunk)
        x = torch.zeros(len(chunk), L, dtype=torch.long, device=device)
        for j, (ids, _) in enumerate(chunk):
            x[j, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        logits, _ = model(x, x)
        logp = F.log_softmax(logits.float(), dim=-1)
        for j, (ids, tagged) in enumerate(chunk):
            for u, site, (s, e), variant in tagged:
                tgt = torch.tensor(ids[s:e], device=device)
                lp = logp[j, s - 1:e - 1, :].gather(
                    -1, tgt[:, None]).squeeze(-1)
                out[u][1 if variant == "live" else 2] = -lp.mean().item()
    return list(out.values())


def aggregate(rows):
    """rows: (site, live_nll, stale_nll|None)."""
    def stats(vals):
        n = len(vals)
        if not n:
            return None
        m = sum(vals) / n
        se = ((sum((v - m) ** 2 for v in vals) / (n * max(1, n - 1))) ** 0.5
              if n > 1 else 0.0)
        return {"mean": round(m, 4), "se": round(se, 4), "n": n}

    agg = {}
    ghosts = [(s, v) for s, v, _ in rows if s["kind"] == "ghost"]
    ghost_stats = stats([v for _, v in ghosts])
    for cue in sorted({s["cue"] for s, _, _ in rows if s["kind"] == "real"}):
        by_bucket = {}
        for s, v, _ in rows:
            if (s["kind"] == "real" and s["cue"] == cue
                    and not s["update"] and not s["collide"]):
                by_bucket.setdefault(bucket_of(s["actual_d"]), []).append(v)
        agg[cue] = {str(b): stats(vs) for b, vs in sorted(by_bucket.items())}
    upd = {}
    for s, v, sv in rows:
        if s["update"] and sv is not None:
            upd.setdefault(bucket_of(s["actual_d"]), []).append(sv - v)
    agg["update_margin"] = {str(b): stats(vs) for b, vs in sorted(upd.items())}
    col = {}
    for s, v, _ in rows:
        if s["collide"]:
            col.setdefault(bucket_of(s["actual_d"]), []).append(v)
    agg["collide"] = {str(b): stats(vs) for b, vs in sorted(col.items())}
    agg["ghost"] = ghost_stats
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--lifetimes", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--bank-part", default="hold")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(os.path.join("runs", args.run_name, args.ckpt),
                      map_location=device)
    from core.model import model_from_ckpt_config
    cfg, model = model_from_ckpt_config(ckpt["config"])
    model = model.to(device).eval()
    # v0-era chunk checkpoints predate the v0.1 writer gain (init 1.0 ==
    # the exact soft behavior they trained with) — tolerate ONLY that
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    assert not unexpected, unexpected
    assert all(m.endswith(".gain") for m in missing), missing

    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    bank = load_bank()
    def fits(dd, scale=1):
        return tuple(d for d in dd if scale * d + 512 <= cfg.block_size)
    pcfg = ProbeConfig(T=cfg.block_size, bank_part=args.bank_part,
                       distances=fits(ProbeConfig.distances),
                       update_distances=fits(ProbeConfig.update_distances, 2),
                       collide_distances=fits(ProbeConfig.collide_distances),
                       h2_distances=fits(ProbeConfig.h2_distances, 2),
                       ghost_distances=fits(ProbeConfig.ghost_distances))
    print(f"[stream_probe] {args.run_name}: pattern "
          f"{getattr(cfg, 'attn_pattern', '') or type(cfg).__name__}, "
          f"T={cfg.block_size}, "
          f"{args.lifetimes} lifetimes, distances {pcfg.distances}",
          flush=True)

    tok = Tok(hf)
    rows = []
    import random
    for i in range(args.lifetimes):
        lt = Lifetime(random.Random(args.seed + i), bank, tok, pcfg)
        rows.extend(site_nlls(model, [lt], device, args.batch_size))
        if (i + 1) % 8 == 0:
            print(f"  {i + 1}/{args.lifetimes} lifetimes", flush=True)

    agg = aggregate(rows)
    out = {"lifetimes": args.lifetimes, "seed": args.seed,
           "bank_part": args.bank_part, "T": cfg.block_size,
           "n_sites": len(rows), "results": agg}
    path = os.path.join("runs", args.run_name, "stream_probe.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(agg, indent=1)[:2000])
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
