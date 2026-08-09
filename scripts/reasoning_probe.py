"""Serial-depth (reasoning) probe for base checkpoints: chain-following
INSIDE one attention window, so retrieval reach is a non-factor and
serial computation is the only axis being varied.

    python scripts/reasoning_probe.py --run-names mimo-name,uberloop-name

Two tasks, both logprob-contrast graded (the needle trick: a distractor
chain with a different answer sits in the same context, so format
familiarity and "copy the only candidate" shortcuts cancel — the model
must actually follow the queried chain):

  chain   : interleaved variable chains "k = 7 . q = k . z = q . z = ?"
            vs a parallel chain ending in a different digit. Sweep hops.
  nlchain : transitive containment "the key is in the box . the box is
            in the shed . the key is in the ?" vs a parallel object
            chain. Sweep hops.

Score per (task, hops): acc = P(correct final token) > P(distractor's
final token), margin = lp(correct) - lp(distractor). A model that binds
but cannot compose stays at 50% beyond 1 hop; effective serial depth
shows up as the hop count where acc decays toward chance.

Runs on any GPU (or CPU) at T<=~120 tokens; forces the dense SDPA path
(no flex/flash) so pre-Ampere cards work.
"""
import argparse
import json
import os
import string
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("FLASH_SWA", "0")
from core import gated_swa                                     # noqa: E402
gated_swa.USE_FLEX = False
from core.model import GPT, GPTConfig                          # noqa: E402

HOPS = (0, 1, 2, 3, 4, 6, 8)   # 0 = direct re-query (pure induction
                               # control: needle-provable, must be high
                               # or the probe format itself is broken)
CONTAINERS = ["box", "bag", "jar", "cup", "pot", "bin", "crate", "chest",
              "drawer", "basket", "bucket", "sack", "tank", "case",
              "shed", "barn", "vault", "tray", "tube", "cart"]
OBJECTS = ["key", "coin", "ring", "stone", "shell", "bead",
           "marble", "button", "feather"]


def enc(tok, s):
    return tok(s, add_special_tokens=False)["input_ids"]


def build_chain_trial(tok, rng, hops):
    """Two interleaved variable chains; query one. Returns (ids,
    correct_first_token, distractor_first_token)."""
    # rare consonant bigrams, not single letters: "a"/"i" are common
    # words with crushing priors, and induction needs a distinctive cue
    # (the needle's cue is 8 random tokens; give it at least 2 rare ones)
    cons = list("bcdfghjklmnpqrstvwxz")
    seen = set()
    while len(seen) < 2 * (hops + 1):
        seen.add("".join(rng.choice(cons, 2)))
    names = list(seen)
    va = names[:hops + 1]
    vb = names[hops + 1:2 * (hops + 1)]
    da, db = rng.choice(10, size=2, replace=False)
    # counterbalance order within every pair AND which chain is queried,
    # else pure recency preference masquerades as (anti-)reasoning
    pair = [f"{va[0]} = {da} .", f"{vb[0]} = {db} ."]
    rng.shuffle(pair)
    lines = pair
    for i in range(1, hops + 1):
        pair = [f"{va[i]} = {va[i - 1]} .", f"{vb[i]} = {vb[i - 1]} ."]
        rng.shuffle(pair)
        lines += pair
    if rng.random() < 0.5:
        va, vb, da, db = vb, va, db, da
    prompt = " ".join(lines) + f" {va[hops]} ="
    # causal contrast: identical prompt with the QUERIED chain's root
    # digit swapped to dc. If the model resolves the chain, lp(da)
    # drops in the swapped version -> delta > 0. No in-context
    # competitor prior to fight (the needle probe's with/without trick).
    dc = int(rng.choice([x for x in range(10) if x not in (da, db)]))
    swapped = prompt.replace(f"= {da} .", f"= {dc} .", 1)
    tgt = enc(tok, f" {da}")[0]
    dis = enc(tok, f" {db}")[0]
    return (enc(tok, prompt), tgt, dis), enc(tok, swapped)


def build_nlchain_trial(tok, rng, hops):
    """Two containment chains; query one object's outermost container."""
    conts = rng.permutation(CONTAINERS)
    ca = conts[:hops + 1]
    cb = conts[hops + 1:2 * (hops + 1)]
    oa, ob = rng.permutation(OBJECTS)[:2]
    pair = [f"the {oa} is in the {ca[0]} .",
            f"the {ob} is in the {cb[0]} ."]
    rng.shuffle(pair)
    lines = pair
    for i in range(1, hops + 1):
        pair = [f"the {ca[i - 1]} is in the {ca[i]} .",
                f"the {cb[i - 1]} is in the {cb[i]} ."]
        rng.shuffle(pair)
        lines += pair
    if rng.random() < 0.5:
        ca, cb, oa, ob = cb, ca, ob, oa
    prompt = " ".join(lines) + f" so the {oa} is in the"
    # causal contrast: swap the queried object's ROOT container for an
    # unused one -- breaks the chain's first link, so lp(target) drops
    # iff the model was actually following the chain
    spare = [c for c in CONTAINERS if c not in list(ca) + list(cb)]
    swapped = prompt.replace(f"the {oa} is in the {ca[0]} .",
                             f"the {oa} is in the {rng.choice(spare)} .", 1)
    tgt = enc(tok, f" {ca[hops]}")[0]
    dis = enc(tok, f" {cb[hops]}")[0]
    return (enc(tok, prompt), tgt, dis), enc(tok, swapped)


@torch.no_grad()
def run_task(model, tok, build, trials, device, seed):
    def lp_last(rows, bs=96):
        # microbatched: the full (N, L, 50304) logits tensor OOMs at
        # fp32 x 512 trials; only the last position is ever used
        L = max(len(r) for r in rows)
        outs = []
        for s in range(0, len(rows), bs):
            chunk = rows[s:s + bs]
            x = torch.zeros(len(chunk), L, dtype=torch.long, device=device)
            for i, r in enumerate(chunk):   # LEFT-pad so query is last
                x[i, L - len(r):] = torch.tensor(r, device=device)
            logits, _ = model(x, x)
            outs.append(torch.log_softmax(logits[:, -1, :].float(), dim=-1))
            del logits
        return torch.cat(outs), L

    out = {}
    for hops in HOPS:
        rng = np.random.default_rng(seed + hops)
        rows, srows, tgts, diss = [], [], [], []
        for _ in range(trials):
            (ids, tgt, dis), sw = build(tok, rng, hops)
            rows.append(ids)
            srows.append(sw)
            tgts.append(tgt)
            diss.append(dis)
        lp, L = lp_last(rows)
        lps, _ = lp_last(srows)
        t = torch.tensor(tgts, device=device)
        d = torch.tensor(diss, device=device)
        lt = lp.gather(1, t[:, None])[:, 0]
        ld = lp.gather(1, d[:, None])[:, 0]
        lts = lps.gather(1, t[:, None])[:, 0]
        margin = (lt - ld).cpu().numpy()
        delta = (lt - lts).cpu().numpy()    # causal root-tracking
        acc = float((lt > ld).float().mean())
        out[hops] = {"acc": acc, "margin_mean": float(margin.mean()),
                     "margin_se": float(margin.std(ddof=1)
                                        / np.sqrt(trials)),
                     "delta_mean": float(delta.mean()),
                     "delta_se": float(delta.std(ddof=1)
                                       / np.sqrt(trials)),
                     "ctx_tokens": L}
        print(f"    hops={hops:>2}  acc {acc:5.1%}  margin "
              f"{margin.mean():+6.2f}  DELTA {delta.mean():+6.2f} "
              f"(se {delta.std(ddof=1) / np.sqrt(trials):.2f})  ctx~{L}",
              flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-names", required=True,
                    help="comma-separated run folder names under runs/")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--trials", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    # containment vocab must be single-token with leading space
    for w in CONTAINERS + OBJECTS:
        assert len(enc(tok, f" {w}")) == 1, w

    results = {}
    for name in args.run_names.split(","):
        name = name.strip()
        ckpt = torch.load(os.path.join("runs", name, args.ckpt),
                          map_location=device)
        cfg = GPTConfig(**ckpt["config"])
        model = GPT(cfg).to(device).eval()
        model.load_state_dict(ckpt["model"])
        print(f"[reason] {name} (pattern {cfg.attn_pattern}, "
              f"loops {cfg.loops or 'none'})", flush=True)
        res = {}
        for task, build in (("chain", build_chain_trial),
                            ("nlchain", build_nlchain_trial)):
            print(f"  -- {task}", flush=True)
            res[task] = run_task(model, tok, build, args.trials, device,
                                 args.seed)
        results[name] = res
        path = os.path.join("runs", name, "reasoning_probe.json")
        with open(path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"wrote {path}", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    if len(results) > 1:
        print("\n=== acc by hops ===")
        for task in ("chain", "nlchain"):
            print(f"-- {task}")
            print(" | ".join(["%-28s" % "model"]
                             + ["%5d" % h for h in HOPS]))
            for name, res in results.items():
                print(" | ".join(["%-28s" % name[:28]]
                                 + ["%5.1f" % (100 * res[task][h]["acc"])
                                    for h in HOPS]))


if __name__ == "__main__":
    main()
