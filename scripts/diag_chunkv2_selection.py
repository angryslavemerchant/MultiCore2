"""Post-mortem for chunkv2's needle null: which half of the fetch broke?

    python scripts/diag_chunkv2_selection.py --run-name 124m-chunkv2-...

For needle trials (same construction as needle_probe.py), instrument
every N layer's fetch and measure, per distance:

  mint_rate : fraction of (trial, head) where ANY chunk's raw pointer
              set cites a needle-payload position — can the log even
              carry the needle?
  sel_rate  : fraction of (trial, head, query-in-payload-window) whose
              top-fetch_n selected chunks include a payload-citing
              chunk — does addressing route to it?
  chance    : expected sel_rate if selection were uniform over visible
              chunks (fetch_n * containing / visible).

mint_rate ~0            -> writers never cite needles (mint failure).
mint high, sel ~chance  -> addressing failure (summaries not
                           content-addressable; the NSA-selection gap).
sel >> chance, acc 0    -> position-free fetch can't express adjacency
                           (the rope'd-fetch variant is the retest).
"""
import argparse
import json
import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model import GPT, GPTConfig                          # noqa: E402
from core.chunkv2 import ChunkV2Attention                      # noqa: E402
from scripts.needle_probe import ensure_corpus, build_trials   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--distances", default="256,512,1024")
    ap.add_argument("--needle-prefix", type=int, default=8)
    ap.add_argument("--needle-payload", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(os.path.join("runs", args.run_name, args.ckpt),
                      map_location=device)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    T = cfg.block_size
    k, m = args.needle_prefix, args.needle_payload

    # instrument every chunkv2 layer's fetch: record the selected chunk
    # indices (fi) alongside the mint pointers already kept in last_mint
    records = {}
    orig_fetch = ChunkV2Attention._fetch
    n_layers = []
    for li, blk in enumerate(model.transformer.h):
        attn = getattr(blk, "attn", None)
        if not isinstance(attn, ChunkV2Attention):
            continue
        n_layers.append(li)

        def patched(self, q, k_n, v, s_c, ptr, ok, _li=li):
            s_f, vf = orig_fetch(self, q, k_n, v, s_c, ptr, ok)
            n = min(self.fetch_n, s_c.shape[-1])
            fv, fi = s_c.topk(n, dim=-1)
            records[_li] = {"fi": fi.detach(),
                            "ok": ok.detach(),
                            "ptr": self.last_mint[0],
                            "vis": torch.isfinite(fv).detach()}
            return s_f, vf

        attn._fetch = types.MethodType(patched, attn)
    print(f"[diag] {len(n_layers)} chunkv2 layers: {n_layers}", flush=True)

    corpus = ensure_corpus()
    out = {}
    for dist in [int(d) for d in args.distances.split(",")]:
        rng = np.random.default_rng(args.seed + dist)
        w, _ = build_trials(corpus, rng, T, dist, args.trials, k, m)
        qs = T - m - k
        ns = qs - dist
        vp_lo, vp_hi = ns + k, ns + k + m      # payload's 1st occurrence
        stats = {li: [0, 0, 0, 0.0] for li in n_layers}  # mint,selhit,tot,chance
        for s in range(0, args.trials, args.batch_size):
            xb = torch.from_numpy(w[s:s + args.batch_size]).to(device)
            with torch.no_grad(), torch.autocast(
                    device, dtype=torch.bfloat16, enabled=device == "cuda"):
                model(xb, xb)
            for li in n_layers:
                r = records[li]
                ptr, ok, fi, vis = r["ptr"], r["ok"], r["fi"], r["vis"]
                B, H, S, kk = ptr.shape
                # chunk cites a payload position with a RAW pointer
                cites = (((ptr >= vp_lo) & (ptr < vp_hi) & ok)
                         .any(-1))                       # (B,H,S)
                # queries that predict the payload: last m+k positions
                qsl = slice(qs, T)
                fi_q = fi[:, :, qsl, :]                  # (B,H,k+m,n)
                hit = cites.unsqueeze(2).expand(
                    -1, -1, fi_q.shape[2], -1).gather(3, fi_q)
                selhit = (hit & vis[:, :, qsl, :]).any(-1)   # (B,H,k+m)
                nvis = vis[:, :, qsl, :].any(-1)
                stats[li][0] += int(cites.any(-1).sum())
                stats[li][1] += int(selhit.sum())
                stats[li][2] += int(nvis.sum())
                # chance: fetch_n * ncontaining / nvisible chunks
                nvis_chunks = torch.isfinite(
                    torch.zeros(1)).new_ones(1)  # placeholder, see below
                ncont = cites.sum(-1).float().mean()
                stats[li][3] += float(
                    (fi.shape[-1] * ncont / max(S, 1)) * nvis.sum())
        for li in n_layers:
            mint, selhit, tot, chance = stats[li]
            denom = (args.trials // args.batch_size
                     * args.batch_size * ptr.shape[1])
            print(f"[diag] d={dist:>5} layer {li:>2}  "
                  f"mint_rate {mint / denom:5.1%}  "
                  f"sel_rate {selhit / max(tot, 1):6.2%}  "
                  f"chance~{chance / max(tot, 1):6.2%}", flush=True)
            out[f"d{dist}_L{li}"] = {
                "mint_rate": mint / denom,
                "sel_rate": selhit / max(tot, 1),
                "chance": chance / max(tot, 1)}

    path = os.path.join("runs", args.run_name, "diag_selection.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
