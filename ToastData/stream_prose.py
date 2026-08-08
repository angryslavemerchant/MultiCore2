"""PROSE WORLD (2026-07-25): the honesty upgrade for the graft line.

Replaces the template bank's tidy single-fact statements with 2-3
sentence NARRATIVES (facts embedded mid-prose, pronoun coreference,
narrative flavor) and swaps a fraction of generic fillers for
ENTITY-MENTIONING DISTRACTORS — prose about a known person/company
that contains NO fact. The distractors are the write-all gate's hard
negatives: novelty alone cannot separate "fact about Vorlath" from
"chatter about Vorlath"; whether the learned key space can is the
experiment.

Mechanics: monkey-patches stream_text_v2 in place —
  - statement categories (train+hold) replaced with prose_bank.json
  - L_DOC raised (narratives are ~35-50 BPE tokens)
  - Lifetime wrapped so ~60% of generic fillers become entity
    distractors rendered with the lifetime's own entities
Question templates, stream assembly, min-gap, grading: unchanged.
Call setup() AFTER W.load_bank() and BEFORE W.build_idf().
"""

import json

import stream_text_v2 as W

REL_CATS = ["founded", "industry", "based_in", "makes",
            "works_as", "lives_in", "works_at", "partner"]


def setup(l_doc=56, distractor_frac=0.6, path="prose_bank.json"):
    with open(path, encoding="utf-8") as f:
        pb = json.load(f)
    for cat in REL_CATS:
        W._BANK["train"][cat] = pb[cat]["train"]
        W._BANK["hold"][cat] = pb[cat]["hold"]
    W.L_DOC = l_doc
    dp = pb["distractors_person"]
    dc = pb["distractors_company"]
    base_lt = W.Lifetime

    class ProseLifetime(base_lt):
        def __init__(self, rng, **kw):
            super().__init__(rng, **kw)
            for ix, d in enumerate(self.docs):
                if d[1] != "filler" or rng.random() > distractor_frac:
                    continue
                if rng.random() < 0.67:
                    t = rng.choice(dp).format(P=rng.choice(self.persons))
                else:
                    t = rng.choice(dc).format(C=rng.choice(self.cos))
                self.docs[ix] = (W.enc_c(t), "filler", -1, -1, -1,
                                 None, 0)

    W.Lifetime = ProseLifetime
    print(f"[prose] world patched: L_DOC={l_doc}, "
          f"{sum(len(pb[c]['train']) for c in REL_CATS)} train / "
          f"{sum(len(pb[c]['hold']) for c in REL_CATS)} hold "
          f"narratives, {len(dp)}+{len(dc)} distractors "
          f"(frac {distractor_frac})", flush=True)
