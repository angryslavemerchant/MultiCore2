# The Stream-World Generated Dataset ("fuzzy retrieval / induction")

A portable guide to the procedurally generated dataset family built in
NeocoreEpisodic for training and grading memory-augmented language
models. Written for an agent adopting it in another project. Everything
described here is self-contained in five files (section 2) plus
`tiktoken` and (for the batching helpers only) `torch`.

## 1. What it is, in one page

Each sample is a **lifetime**: a randomly generated **fact graph** over
freshly minted nonce entities, rendered into a long stream of short
documents (~28–56 BPE tokens each) through **paraphrase template
banks**. A lifetime is hundreds of documents long; the facts a model is
later graded on appear only 1–3 times, early, phrased differently each
time.

The dataset is engineered so that exactly one channel can produce the
answer — an explicit memory between the weights and the context:

- **Unsmearable across training.** Entities are multi-token nonce names
  (e.g. "Vorlath Kremar", "Tavrix Holdings") re-randomized per
  lifetime. Weights cannot memorize the facts; only the *skill* of
  filing and retrieving is trainable.
- **Unanswerable from context.** The reference consumers run attention
  DOCUMENT-LOCALLY (each doc is its own attention span), and question /
  recurrence docs are placed with a minimum gap (default ≥25 docs)
  after the fact they depend on. Even a full-context consumer faces
  gaps up to whole-lifetime range (log-uniform sampled, median ~26,
  max ~215 of ~275 docs in the falling-line config).
- **Fuzzy, not literal.** Statement templates, question templates, and
  recurrence ("realize") templates are three DISJOINT phrasings of the
  same fact, with additional held-out template splits for eval. String
  matching / induction-head copying cannot work; retrieval must be
  semantic.
- **Induction-shaped grading (falling-line config).** Recurrence docs
  are prose whose continuation is unpredictable from local context but
  predictable given the earlier fact: "The flight home felt endless.
  {P} never sleeps well until the plane touches down in {home_city}."
  The attribute-realizing tokens are **grade tokens**; next-token NLL
  on them, bucketed by position or by gap, IS the memory measurement.
  No questions, no labels — the LM loss is the only teacher and grade.

Three stacked configurations (each monkey-patches the previous):

| config | file | docs | supervision signal |
|---|---|---|---|
| **bank world** | `stream_text_v2.py` | tidy single-fact statements + fillers + interleaved questions + final quiz | exact-match QA (h1 / h2 / abstain) |
| **prose world** | `stream_prose.py` | 2–3-sentence narratives (pronoun coref) + entity-mention distractors | same QA, harder percepts |
| **falling line** | `stream_fall.py` | narratives + recurrence docs + ghost controls, NO questions | grade-token NLL only |

## 2. Files to copy

| file | role |
|---|---|
| `stream_text_v2.py` | core generator: fact graph, `Lifetime`, questions, vocab, IDF, `build_batch` |
| `stream_prose.py` | prose patch (narrative statements + distractors) |
| `stream_fall.py` | falling-line patch (recurrence docs, grade spans, gap curriculum) |
| `templates_bank.json` | 8 relations × ~44 statement paraphrases, 13 question types × ~16, 124 fillers |
| `prose_bank.json` | narrative statements, distractors (person/company), and `realize_*` recurrence templates per relation, each with train/hold splits |
| `vocab_text_v2.json` | *optional cache* of the reduced vocab; safe to delete (rebuilt on demand) |

**Dependency wart:** `stream_text_v2.py` does
`from toy_stream_icl import Block, Book, ReadHead` — those are only
used by its bundled model/training code, not by the dataset. For a
dataset-only extraction either copy `toy_stream_icl.py` too or delete
that import along with the `TextStreamLM` / `run_batch` / `train` /
`eval_arm` / `main` sections. The dataset proper needs only:
constants, `load_bank`, `build_vocab`, `enc_c`, `build_idf`,
`nonce_*`, `Lifetime`, `build_batch`.

## 3. The fact graph

Per lifetime (defaults `N_C=5` companies, `N_P=10` persons → **55
facts**):

- Companies: `founded` (by person i), `industry`, `based_in` (city),
  `makes` (nonce product).
- Persons: `works_as` (profession), `lives_in` (city), `works_at`
  (company; persons 0–4 work at "their" company), `partner`
  (5 disjoint spouse pairs).
- Attribute pools are small closed sets (8 industries, 8 professions,
  12 cities) — answers are guessable at a closed-set floor, which is
  why controls matter (section 7).
- **Two-hop chains** exist by construction: founder→industry,
  founder→city, spouse→job, spouse→city, employer→industry. Two-hop
  questions name only the first entity; the bridge entity must come
  from memory.
- **Abstention**: ~12% of questions ask about ghost entities that
  don't exist in the lifetime; correct answer is `" unknown"`.

## 4. The document tuple

`Lifetime(rng).docs` is a list of 7-tuples:

```
(ids, kind, fid, a1, a2, span, apos)
```

| field | meaning |
|---|---|
| `ids` | token ids (see section 6 for the two encoding modes) |
| `kind` | `"fact"`, `"filler"`, `"q_h1"`, `"q_h2"`, `"q_abstain"`, `"recur"`, `"recur_ctrl"` |
| `fid` | fact index for fact docs (a write cue for cheat-flagged filing); `-1` otherwise. **Always `-1` in the falling-line world** — that world forbids flag filing by design |
| `a1`, `a2` | questions: the fact indices needed (hop-1: both equal; hop-2: fact1, fact2). Recurrence docs: `a1` = fact index (diagnostics only), `a2` = the gap in docs since that fact's last statement (retention-curve bucketing). `-1` for `recur_ctrl`/others |
| `span` | `(start, end)` token span of the answer / grade tokens inside `ids`; `None` for fact/filler docs |
| `apos` | last token index of the subject mention (a hook for auxiliary read supervision at the subject position) |

Grading convention used everywhere: **teacher-forced exact match over
the span** (every argmax prediction inside the span must equal the
target — multi-token answers get no partial credit), or **mean NLL on
grade tokens** in the falling-line config.

`build_batch(B, device, rng, ...)` turns `B` lifetimes into padded
tensors `(toks[B,D,L_DOC], fids, aux1, aux2, ans_mask, q_pos, s_pos,
kinds, n_facts)` — convenient but optional; the tuples are easy to
consume directly. Note docs are truncated to `L_DOC`; keep `L_DOC`
large enough for your longest template + answer (28 for the bank
world, 56 for prose/fall) or spans get clipped.

## 5. Setup order (this is the part that bites)

The patches replace `stream_text_v2`'s globals in place, so **order is
mandatory**: `load_bank()` first, patches second, vocab/IDF last.

Bank world:

```python
import stream_text_v2 as W
W.load_bank()                      # templates_bank.json
nv = W.build_vocab()               # reduced vocab (or GPT-2 mode, §6)
W.UNKNOWN_IDS = W.enc_c(" unknown")
W.build_idf()                      # only if you use IDF pooling
lt = W.Lifetime(random.Random(0))  # ~275 docs, 55 facts
```

Prose world — insert before vocab/IDF:

```python
W.load_bank()
import stream_prose; stream_prose.setup(l_doc=56, distractor_frac=0.6)
# ... vocab / UNKNOWN_IDS / idf as above; W.Lifetime is now ProseLifetime
```

Falling line (calls `stream_prose.setup()` itself):

```python
W.load_bank()
import stream_fall; stream_fall.setup(l_doc=56, rec_per_fact=(1, 3), n_ctrl=3)
W.build_idf()
stream_fall.GAP_CAP = 20   # optional gap curriculum; None = uncapped (eval)
lt = W.Lifetime(random.Random(0))            # no question docs at all
hold = W.Lifetime(random.Random(1), bank_part="hold")  # held-out templates
```

`bank_part="hold"` selects the held-out template split (last 10
statements per relation, last 4 questions per type, held realize
templates) — the paraphrase-generalization eval. Entities are fresh
every lifetime regardless; "hold" holds out *phrasings*, not facts.

## 6. Two encoding modes

Token ids come out of `enc_c(text)`, which maps GPT-2 BPE ids through
`W._REMAP`.

**Reduced vocab (~2.2k, for from-scratch small models):** the default.
`build_vocab()` enumerates every token the banks + name generators can
produce, caches to `vocab_text_v2.json`, and remaps to a dense range
with `PAD=0`, `UNK=1`. **Gotcha:** the cache is built from whichever
bank is loaded at call time — if you use the prose/fall banks with
reduced vocab, delete the cache and call `build_vocab()` *after*
`setup()`, or prose-only tokens silently become `UNK`.

**Raw GPT-2 BPE (for pretrained models):** bypass the remap with an
identity map before any `enc_c` call (this is exactly what the graft
scripts do):

```python
class _IdMap(dict):
    def get(self, k, default=None): return k
W.load_bank()
W._REMAP = _IdMap()
W._NVOCAB = 50257
W.PAD = 50256          # eot as pad; PAD=0 would collide with token 0 ("!")
W.UNKNOWN_IDS = W.enc_c(" unknown")
stream_fall.setup()    # patches AFTER the remap override is fine
W.build_idf()
```

`build_idf()` samples 40 lifetimes and returns per-token
inverse-doc-frequency weights (`W._IDF`) — used for pooling document
embeddings into memory payloads (flat means drown the answer tokens in
function words; IDF hands the mass to names and values). Only needed
if your consumer pools; harmless to skip otherwise.

## 7. Knobs

| knob | where | default | effect |
|---|---|---|---|
| `N_C`, `N_P` | module globals / `main` args | 5 / 10 | world size (55 facts) |
| `stmts` | `Lifetime` kw | 2 | paraphrased statements per fact |
| `filler_frac` | `Lifetime` kw | 0.3 | generic-filler fraction |
| `abstain_frac` | `Lifetime` kw | 0.12 | ghost-entity question fraction |
| `n_stream_q` / `STREAM_Q` | `Lifetime` kw | 16 | mid-stream questions (0 = quiz-only) |
| `n_quiz` | `Lifetime` kw | 26 | final quiz block size |
| `MIN_GAP` | module global | 25 | min docs between a fact's last statement and a question on it |
| `L_DOC` | module global (set by `setup`) | 28 / 56 | tokens per doc (truncation!) |
| `bank_cap` | `load_bank(cap=)` | 0 | use only first N templates/category (diversity ablation) |
| `distractor_frac` | `stream_prose.setup` | 0.6 | fillers converted to entity-mention distractors |
| `rec_per_fact` | `stream_fall.setup` | (1, 3) | recurrence docs per fact (spec allows up to 5) |
| `n_ctrl` | `stream_fall.setup` | 3 | ghost recurrence controls per lifetime |
| `GAP_CAP` | `stream_fall` global | None | cap on recurrence gaps (anneal small→None as curriculum) |

## 8. Measurement contract and hard-won pitfalls

These are pre-registered instruments and empirical laws from the runs
that built this dataset (RUNG2_SPEC.md has the full falling-line
contract). Adopt them; each one exists because a naive version
produced a wrong or unignitable experiment.

**Instruments**
- *Headline (falling line)*: mean grade-token NLL bucketed by doc
  position (8 buckets), memory arm vs dense vs junk-memory. Success =
  the memory arm's line falls within a lifetime while dense stays flat.
- *Retention curve*: grade NLL vs recurrence gap (`a2`) — does memory
  hold value at ranges context can't reach.
- *Paraphrase honesty*: everything re-evaluated with
  `bank_part="hold"`.
- *Position curve (QA worlds)*: per-question accuracy by stream
  quartile.

**Controls (all cheap, all necessary)**
- `recur_ctrl` ghosts: same realize templates, entity never
  introduced. Real recurrences must beat ghosts, else you're measuring
  template priors, not memory.
- First-occurrence grade NLL must be equal across arms (leak check).
- Junk-memory arm (trained model, randomized store at eval) must
  crater; if it ties, the "memory" was never load-bearing.
- Closed-set answer floors are real: a memoryless model scores ~2–5%
  h1 / ~8–15% h2 by guessing over the small attribute pools, and
  uniform-over-answer-set gives flat NLL ≈ log(set size) on grade
  tokens. Never read small nonzero scores as retrieval.

**Leak validation (run once per template-bank edit)**
Realize templates must satisfy: ≥4 attribute values locally plausible;
no phrasing shared with introduction templates. Verify with a
pretrained LM: grade-span NLL should stay high (the original gate:
mean 11.3, min 4.7 nats under GPT-2) — check FULL sequences; per-token
checks false-alarm on nonce subword collisions.

**Training-dynamics laws (if you meta-train a memory circuit on this)**
- Mid-stream questions asked while the store is still filling
  suppress read-circuit ignition ~5x if present from step one, and are
  harmless curriculum afterwards. Warm up: no stream questions for the
  first ~35% of training, no abstention for the first ~50–60%
  (`" unknown"` is a degenerate always-valid basin pre-ignition).
- Pure LM-loss, no-scaffolding configs may not self-assemble at all
  (the cold falling-line run pinned its store at 3/96 slots);
  warm-starting the write/read heads from a QA-world checkpoint is the
  working ignition insurance.

## 9. Minimal standalone loop

```python
import random
import stream_text_v2 as W
import stream_fall

W.load_bank("templates_bank.json")
# (choose an encoding mode here, per section 6)
stream_fall.setup(path="prose_bank.json")

rng = random.Random(0)
for lifetime_ix in range(100):
    lt = W.Lifetime(rng)                 # fresh entities every time
    for ids, kind, fid, a1, a2, span, apos in lt.docs:
        feed(ids)                        # your model, doc-local context
        if kind == "recur":
            grade_nll(ids, span, gap=a2) # the falling line
        elif kind == "recur_ctrl":
            grade_nll(ids, span, ctrl=True)
```

That's the whole interface: a generator of unbounded, leak-gated,
paraphrase-split lifetimes where the only way to get the grade tokens
right is to have remembered.
