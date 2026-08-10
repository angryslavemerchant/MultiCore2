"""NSA register interp tier 1f: could the registers serve as a CODEBOOK
that chunks are categorised against?

The registers' directions are the one thing training kept alive (values
are ~2% of token scale; keys are rms-normalised so only direction
survives). A codebook proposal uses that: measure BOTH queries and chunks
against the register bank, and let category agreement drive retrieval.
The register never delivers a value, so the broadcast-noise force that
killed the value path does not apply.

Precondition: chunk representations must already spread meaningfully over
the register directions. If every chunk collapses onto one or two
registers the bank is unusable as a codebook without anti-collapse
machinery. Measured in residual space at each layer -- chunk = mean of the
layer input over its nsa_block tokens, codebook = the register vectors --
so chunks and queries live in the same space the registers were trained in.

Reports, per layer: effective number of categories used (exp entropy of
the assignment histogram), top-1 category share, and how peaked each
chunk's assignment is (margin between best and second-best register).

    python scripts/nsa_interp_codebook.py --run-name 124m-nsa-...-t4096
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
from core.model import model_from_ckpt_config                   # noqa: E402
from core.nsa import NSARegisterAttention                       # noqa: E402
from scripts.needle_probe import ensure_corpus                  # noqa: E402


@torch.no_grad()
def codebook(mod, x):
    """Assign every token chunk to its nearest register direction."""
    B, T, C = x.shape
    nb = mod.nb
    NB = T // nb
    chunk = x.view(B, NB, nb, C).mean(2)[0]            # (NB, C)
    cb = mod.registers                                  # (R, C)
    cs = F.normalize(chunk.float(), dim=-1) @ F.normalize(
        cb.float(), dim=-1).t()                         # (NB, R)
    top2 = cs.topk(2, dim=-1)
    assign = top2.indices[:, 0]
    margin = (top2.values[:, 0] - top2.values[:, 1])
    hist = torch.bincount(assign, minlength=cb.shape[0]).float()
    hist = hist / hist.sum()
    nz = hist[hist > 0]
    ent = float(-(nz * nz.log()).sum())
    return {
        "eff_categories": math.exp(ent),
        "n_categories_used": int((hist > 0).sum()),
        "top1_share": float(hist.max()),
        "assign_margin": float(margin.mean()),
        "best_cos": float(top2.values[:, 0].mean()),
        "n_chunks": NB, "n_reg": cb.shape[0],
        "assign": assign.cpu(),
    }


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
            acc[i].append(codebook(m, caught[i].float()))
            del caught[i]
        print(f"[codebook] doc {d + 1}/{args.docs}", flush=True)

    out = []
    print("\n   chunk -> nearest-register assignment (cosine, residual "
          "space)")
    for i, m in enumerate(mods):
        # do different DOCUMENTS reuse the same categories? (a codebook
        # must generalise across text, not partition one document)
        sets = [set(a["assign"].tolist()) for a in acc[i]]
        shared = len(set.intersection(*sets)) if len(sets) > 1 else 0
        union = len(set.union(*sets))
        st = {k: float(np.mean([a[k] for a in acc[i]]))
              for k in acc[i][0] if k != "assign"}
        st.update({"layer": i, "cats_shared_across_docs": shared,
                   "cats_union_across_docs": union})
        out.append(st)
        print(f"[L{i:>2}] eff_cats {st['eff_categories']:>6.1f}  used "
              f"{st['n_categories_used']:>5.0f}/{st['n_reg']:.0f}  top1 "
              f"{st['top1_share']:.2f}  margin {st['assign_margin']:.4f}  "
              f"best_cos {st['best_cos']:+.3f}  |  across {args.docs} docs: "
              f"{shared} shared / {union} union", flush=True)

    path = os.path.join("runs", args.run_name, "nsa_interp_codebook.json")
    with open(path, "w") as f:
        json.dump({"run": args.run_name, "docs": args.docs,
                   "layers": out}, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
