"""FALLING-LINE world (RUNG2_SPEC.md) — no questions, no flags.

Lifetimes = intro narratives + RECURRENCE docs (continuation
predictable only via memory; value span = grade tokens) + entity
distractors + generic fillers. Patches stream_text_v2 like
stream_prose. Doc tuple reuse:
    (ids, kind, fid, a1, a2, span, apos)
    kind: "recur" / "recur_ctrl" (no-intro control) / others as before
    fid:  -1 always (NO flag filing anywhere in this world)
    a1:   fact index (diagnostics/bucketing), -1 for ctrl
    a2:   recurrence GAP in docs (retention curve), -1 otherwise
    span: grade-token span (the attribute value)
Gap curriculum: module global GAP_CAP (docs) — train loop anneals
it; None = uncapped (eval).
"""

import json
import math

import stream_text_v2 as W
import stream_prose

GAP_CAP = None  # None = full range; train sets small -> anneals up

REL_VALUE = {
    "founded": ("realize_founded", "C", lambda lt, k: lt.founder[k]),
    "industry": ("realize_industry", "C", lambda lt, k: lt.industry[k]),
    "based_in": ("realize_based_in", "C", lambda lt, k: lt.city[k]),
    "makes": ("realize_makes", "C", lambda lt, k: lt.makes[k]),
    "works_as": ("realize_works_as", "P", lambda lt, k: lt.job[k]),
    "lives_in": ("realize_lives_in", "P", lambda lt, k: lt.home[k]),
    "works_at": ("realize_works_at", "P", lambda lt, k: lt.employer[k]),
}
# partner handled separately (frozenset key)

_RB = None  # realize bank {cat: {train: [...], hold: [...]}}


def _render(lt, rng, cat, slot, ent, value, bank_part):
    tpl = rng.choice(_RB[cat][bank_part if bank_part in ("train",
                                                         "hold")
                             else "train"])
    kw = {slot: ent}
    # value slot name varies; fill every known placeholder
    for vs in ("P", "C", "I", "X", "PROD", "J", "C2", "P2"):
        if "{%s}" % vs in tpl and vs not in kw:
            kw[vs] = value
    prefix_tpl = tpl[:tpl.rfind("{")]
    prefix = prefix_tpl.format(**{k: v for k, v in kw.items()
                                  if "{%s}" % k in prefix_tpl})
    ids = W.enc_c(prefix.rstrip())
    a = W.enc_c(" " + value)
    span = (len(ids), len(ids) + len(a))
    return ids + a, span


def setup(l_doc=56, rec_per_fact=(1, 3), n_ctrl=3,
          distractor_frac=0.6, path="prose_bank.json"):
    """Call AFTER W.load_bank(); installs prose statements +
    realize templates + the FallLifetime."""
    stream_prose.setup(l_doc=l_doc, distractor_frac=distractor_frac,
                       path=path)
    global _RB
    with open(path, encoding="utf-8") as f:
        pb = json.load(f)
    _RB = {k: v for k, v in pb.items() if k.startswith("realize_")}
    assert len(_RB) == 8, f"realize bank incomplete: {list(_RB)}"
    base_lt = W.Lifetime  # = ProseLifetime (statements + distractors)

    class FallLifetime(base_lt):
        def __init__(self, rng, bank_part="train", **kw):
            kw["abstain_frac"] = 0.0
            kw["n_stream_q"] = 0
            super().__init__(rng, bank_part=bank_part, **kw)
            # strip ALL question docs (super appends a final quiz)
            self.docs = [d for d in self.docs
                         if not (d[1] or "").startswith("q_")]
            docs = self.docs
            last_pos = {}
            for pos, d in enumerate(docs):
                if d[1] == "fact":
                    last_pos[d[2]] = pos
            rec = []
            # estimated FINAL stream length (after insertions) so
            # gaps can reach whole-lifetime range regardless of
            # where the intro landed
            n_rec_est = int(self.n_facts
                            * (rec_per_fact[0] + rec_per_fact[1]) / 2)
            d_est = len(docs) + n_rec_est

            # enumerate facts through the graph (mirrors add() order)
            def add_rec(fkey, cat, slot, ent, value):
                fid = self.fid[fkey]
                n = rng.randint(*rec_per_fact)
                for _ in range(n):
                    ids, span = _render(self, rng, cat, slot, ent,
                                        value, bank_part)
                    base = last_pos.get(fid, 0)
                    room = max(6, d_est - base)
                    cap = min(room, GAP_CAP) if GAP_CAP else room
                    gap = int(math.exp(rng.uniform(
                        math.log(5), math.log(max(cap, 6)))))
                    apos = self._subj_pos(ids, ent)
                    rec.append((min(base + gap, len(docs)),
                                (ids, "recur", -1, fid, gap, span,
                                 apos)))
            for c in self.cos:
                add_rec(("founded", c), "realize_founded", "C", c,
                        self.founder[c])
                add_rec(("industry", c), "realize_industry", "C", c,
                        self.industry[c])
                add_rec(("based_in", c), "realize_based_in", "C", c,
                        self.city[c])
                add_rec(("makes", c), "realize_makes", "C", c,
                        self.makes[c])
            for p in self.persons:
                add_rec(("works_as", p), "realize_works_as", "P", p,
                        self.job[p])
                add_rec(("lives_in", p), "realize_lives_in", "P", p,
                        self.home[p])
                add_rec(("works_at", p), "realize_works_at", "P", p,
                        self.employer[p])
            done = set()
            for a in self.persons:
                if a in done or a not in self.partner:
                    continue
                b = self.partner[a]
                done |= {a, b}
                add_rec(("partner", frozenset((a, b))),
                        "realize_partner", "P", a, b)
            # no-intro CONTROLS: ghost entities, recurrence docs only
            for _ in range(n_ctrl):
                ghost = W.nonce_person(rng)
                val = rng.choice(W.PROFESSIONS)
                ids, span = _render(self, rng, "realize_works_as",
                                    "P", ghost, val, bank_part)
                pos = rng.randrange(len(docs) // 2, len(docs))
                rec.append((pos, (ids, "recur_ctrl", -1, -1, -1,
                                  span, max(0, len(ids) - 1))))
            for pos, doc in sorted(rec, key=lambda x: -x[0]):
                docs.insert(min(pos, len(docs)), doc)

    W.Lifetime = FallLifetime
    n_t = sum(len(v["train"]) for v in _RB.values())
    n_h = sum(len(v["hold"]) for v in _RB.values())
    print(f"[fall] world installed: {n_t} train / {n_h} hold "
          f"realize templates, rec/fact={rec_per_fact}, "
          f"ctrl={n_ctrl}", flush=True)
