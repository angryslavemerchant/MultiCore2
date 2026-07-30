# Grafting a writable codebook into a pretrained LLM

Source repo: `NeocoreEpisodic`. Relevant files: `toy_stream_icl.py` (the
`Book` and `ReadHead` classes), `graft_gpt2.py` (the graft into GPT-2 +
`ReadHead` insertion), `graft_writer.py` (learned writer heads,
`WriterView`), `graft_cwriter.py` (continuous select-writer, newest/
least mature variant). All code below is quoted or paraphrased directly
from those files, not reconstructed from memory.

This describes a synthetic-benchmark experiment, not a natural-text
result — see the "scope" note at the end.

---

## 1. What problem this is solving

Standard transformer LMs have two knowledge stores: **weights** (learned
over training, then frozen — new facts can only enter via fine-tuning,
which risks overwriting old knowledge) and **context** (holds anything
instantly, retains nothing once it slides out, costs quadratically).
There is nothing in between — no persistent, cheap, *editable-after-
training* store. The codebook graft is an attempt to build that middle
layer directly into an existing pretrained model.

The task used to grade it: stream documents introduce entities and facts
about them (`person -> employer`, `person -> spouse -> profession`, plus
distractor filler docs). Later, with the source text long gone from any
practical context window, the model is asked:
- **hop-1**: single fact lookup ("what is X's employer?")
- **hop-2**: two facts chained ("what is X's partner's profession?")

The GPT-2 backbone's pretrained weights are never touched by the write
rule; the only channel that can possibly answer the question is
whatever got written to the store when the fact streamed by, and
whatever gets read back out at question time.

---

## 2. Architecture — where the book sits in the model

`GraftLM` (`graft_gpt2.py:83`) wraps a real, pretrained HF `GPT2LMHeadModel`
and inserts two `ReadHead` modules at two depths inside the 12-block
stack (default `read_depths=(5, 11)` — one mid-stack, one near the top):

```python
class GraftLM(nn.Module):
    def __init__(self, use_book=True, metabook=False, K=96,
                 name="gpt2", read_depths=(5, 11)):
        super().__init__()
        gpt = GPT2LMHeadModel.from_pretrained(name)
        self.tr = gpt.transformer
        self.lm_head = gpt.lm_head
        self.ri = read_depths
        self.read1 = ReadHead(self.d) if use_book else None
        self.read2 = ReadHead(self.d) if use_book else None

    def forward_docs(self, toks, book=None):
        x = self.tr.wte(toks) + self.tr.wpe(pos)
        for li, blk in enumerate(self.tr.h):
            x = blk(x)
            if self.use_book and li in self.ri:
                xb = x.reshape(B, N * L, self.d)
                if li == self.ri[0]:
                    xb, w1 = self.read1(xb, book)
                else:
                    xb, w2 = self.read2(xb, book)
                x = xb.reshape(...)
        logits = self.lm_head(self.tr.ln_f(x))
        return logits, w1, w2
```

Everything before/after/between those two insertion points is unmodified
GPT-2 forward computation. This is why two hops falls naturally out of
having two read points: block 5's read pulls in the *bridging* fact,
block 11's read (now seeing the bridging fact in its residual stream)
pulls in the fact that fact points to.

### The read head: a near-zero-init residual injection

```python
class ReadHead(nn.Module):
    """Query proj into key space + ZERO-INIT output proj (book starts
    as a no-op; aux ignites it)."""
    def __init__(self, d):
        self.wq = nn.Linear(d, d)
        self.wo = nn.Linear(d, d)
        nn.init.zeros_(self.wo.weight)   # <- starts as identity/no-op
        nn.init.zeros_(self.wo.bias)

    def forward(self, x, book):
        mix, w = book.read(self.wq(x))
        return x + self.wo(mix), w        # residual add
```

`wo` is zero-initialized so at step 0 the graft is a mathematical no-op
on top of vanilla GPT-2 — the model can only get *worse* by using the
book once gradients start flowing into `wo`, so there's no discontinuity
between "book off" and "book on" during training. This is the same
zero-init-residual trick used for adapters/LoRA-style grafts generally.

### The book itself (`toy_stream_icl.py:142`, class `Book`)

Batched per-lifetime key/payload store:

```python
self.keys   = (B, K, d)   # unit-normalized keys, random init
self.pays   = (B, K, d)   # payloads, zero init
self.counts = (B, K)      # write count per slot; 0 = unborn

def write(self, kv, pv):          # @torch.no_grad() — NOT backprop
    sim = cos(kv, self.keys)             # vs used slots only
    best, bestv = argmax(sim)
    birth = (bestv < theta) & any_free_slot
    i = free_slot if birth else best
    n = counts[i] + 1
    lr = 1 / min(n, cap)                  # count-capped running mean
    keys[i] += lr * (kv - keys[i])
    pays[i] += lr * (pv - pays[i])
    # + a lazy bar-merge step (see section 3)

def read(self, q):                # differentiable in q only
    sim = cos(q, self.keys.detach())      # masked to used slots
    w = softmax(sim * code_temp)          # code_temp ~ 12.0
    return w @ self.pays.detach(), w      # SOFT blend, not top-1 snap
```

Two load-bearing properties:
- **`write` is a hardcoded rule, not a gradient step.** It runs under
  `@torch.no_grad()`. This is the whole point — the store can be edited
  at *inference/deployment* time, when there is no optimizer running.
- **`read` is differentiable in the query only** (`keys`/`pays` are
  `.detach()`ed). Gradient flows from the LM/answer loss back through
  `wq` and `wo` in `ReadHead`, teaching the model *how to query and use*
  the store, without ever touching the store's contents via backprop.

---

## 3. The write rule in full: birth, update, merge

This is the "v6b" rule, arrived at after 10 documented failed variants
(full log in `toy_codebook_icl.py`'s docstring — summarized in section 5).

**Write/update** — online k-means:
- Compare the incoming key against all *used* slots by cosine similarity.
- If the best match is below a threshold `theta` (and a free slot
  exists) -> **birth** a new slot from this glance.
- Otherwise -> **update** the best-matching slot: `keys`/`pays` move
  toward the new value by `1/min(n, cap)`, i.e. an exact running mean,
  capped so old slots don't freeze completely rigid.

**Merge** — lazy, bar-triggered, happens right after each write:
```python
after updating slot i, compare i's NEW key against other used slots
if max_sim(i, j) > theta_merge  (theta_merge > theta -> hysteresis):
    keep = whichever of i,j has the larger count
    merged key/pay = count-weighted mean of i and j   # exact algebra
    drop the loser (its count -> 0, slot freed for future births)
```

`theta_merge > theta` is deliberate hysteresis: a code that was *just*
born (similarity to everything < theta) can never instantly merge back
into what it split from — merging only fires between two *mature*,
independently-converged duplicates.

**Capacity dynamic**: birth is liberal while slots remain, and forced
joins only happen once `K` is exhausted. This acts as an implicit
explore -> commit schedule (early: spawn freely to avoid collisions;
late: consolidate because there's no other option) — and multiple
attempts to replace it with more "principled" structural rules (see
section 5) made results *worse*, because they broke this schedule.

**Read**: never snaps to one slot. It's a full softmax blend over every
used slot, weighted by cosine similarity. This "soft content" design
was the single highest-leverage decision in the whole line (see v5/v6b
in section 5) — it made the store robust to imperfect/duplicate slot
bookkeeping, because a blend over near-duplicate codes converges to
roughly the same thing a structurally-merged code would represent
anyway.

---

## 4. Making the write *content* learned (the "writer")

The version above writes hand-picked content (e.g. a doc-mean of token
embeddings). `graft_writer.py` replaces that with **learned** key/payload
content while keeping the write/merge *rule* (theta, birth, merge — all
of section 3) completely unchanged:

```python
class WriterView:
    """Per-chunk differentiable view of the book: transforms stored
    percepts through the writer heads WITH graph."""
    def __init__(self, book, key_head, pay_head):
        p = book.pays.detach().clone()      # raw stored percepts
        self.keys = normalize(key_head(p))  # re-rendered EVERY forward pass
        self.pays = pay_head(p)

    def read(self, q):
        # same softmax-blend read as Book.read, but against
        # freshly-computed keys/pays instead of the raw store
        ...
```

The store still holds raw, non-differentiable percepts (written by the
exact same hardcoded rule). But every forward pass, two small MLPs
(`key_head`, `pay_head`, ~4.7M params total) **re-render those stored
percepts into keys/payloads fresh, inside the current chunk's autograd
graph.** Gradient from the downstream LM/answer loss flows back through
this re-rendering, chunk-locally — so the model learns *what a good key/
payload representation looks like*, without ever making the store's
raw contents or its edit rule differentiable.

This single change (content becomes learned, rule stays hardcoded) was
the biggest single jump in the whole campaign: two-hop accuracy went
from "modest" (33%) to "essentially solved" (97%) — see section 6.

**Autonomous filing** (no dataset-provided "this is a fact" flag): the
same novelty-gate rule from section 3, but run directly on the *learned*
key space instead of on hand-picked doc embeddings —
`Book.write(..., theta=0.9)` where `kv`/`pv` come from `key_head`/
`pay_head` applied to candidate percepts, and the model itself decides
per-document whether to write, with zero supervision on that decision.

---

## 5. The write-rule failure log (why it looks the way it does)

Ten iterations, run on a controlled synthetic toy world before ever
touching a real LM (`toy_codebook_icl.py`). This is the part worth
reading closely if you're implementing your own write rule — most of
the "obvious" improvements were tried and made things worse.

| step | change | outcome |
|---|---|---|
| v0 | snap to nearest of K random-init codes, no birth | learns, but same class scatters across many codes ("fragmentation") — consolidation captures ~0 of available headroom |
| v1 | add novelty-gated birth | first real test-time-learning curve, but `theta` miscalibrated to the actual embedding geometry -> under-catches same class -> duplicate births -> book exhausts fast |
| v1b | lower `theta` to fix under-catching | **now under-births instead**: two different classes get fused into one slot — a *collision*, and it's **permanent** (no split operation exists) |
| v2 | add a similarity-threshold merge as repair | works, but fires too late — duplicate slots only become similar enough to merge after 3-4 writes each |
| v3 | "witness merge": if a write's top-2 candidate slots are *both* above the birth bar, merge them immediately (no separate threshold needed) | merges fire early, but **performance drops** — freed slots get immediately re-populated by fresh single-glance noise; this broke the birth-until-full anneal that v1 was accidentally relying on |
| v4 | tombstone freed slots (never reusable) so capacity is strictly monotone | structure behaves exactly as designed, but **accuracy declines starting exactly at the training-horizon episode** — the model was never meta-trained on the consolidated book states its own rule eventually produces |
| v4b | fix training horizon to match eval horizon | decline is fixed, but accuracy still sits well below v1's, despite the *best*-looking bookkeeping of the whole campaign — merges jump the underlying vector mid-episode, staling every card written before the merge |
| v5 | **soft mixture content**: reads/writes-to-content blend over *all* used slots by similarity-softmax, instead of snapping to one slot; keep hard-assign for the update rule itself | **decisive win** — slope ~2x v1's, gap to an oracle nearly halved, cleanest training of the campaign. The lesson: the noise was never in the slot bookkeeping, it was always at the *content interface* |
| v6 | soft content + witness-merge + tombstone (retry structure now that content is soft) | worse than v5 — under soft content, a population of duplicate slots is a *richer* estimator than its merged collapse; merging throws away signal that's now an asset |
| v6b | soft content + **lazy bar-merge only**, no tombstone | **best of the whole campaign** — merge only fires between mature, already-converged duplicates, which is exactly when merging is (near) lossless |
| v7/v7b | DP-means "rent" formulation — one constant (`lam`) replaces both thresholds via an energy objective (birth iff distortion saved > rent; merge iff exact collapse cost < rent) | theoretically elegant (derives the v6b behavior from first principles) but **fails twice in practice**: fixed rent bankrupts the book while its geometry is still immature (nothing ever births); adaptive/quantile rent gets poisoned by its own early novelty (self-referential statistic) |

**Shipped rule** (used in every downstream real-LM experiment):
hard-assign updates (running-mean writes) + birth-until-capacity-full +
soft-mixture content at read time + lazy bar-merge, no tombstones. One
constant for birth (`theta`), one for merge (`theta_merge`).

**The two standing laws this produced:**
1. *Asymmetry*: over-birthing (duplicate slots) is recoverable — a soft
   read blends them back together, or a lazy merge cleans them up later.
   Under-birthing (collisions) is permanent — there's no split
   operation. **When in doubt, birth.**
2. *Interface beats structure*: fixing the content representation (v5)
   beat every attempt to fix the slot bookkeeping (v2/v3/v4/v6)
   combined. Structural tidying only pays off when it's lazy/confident
   (v6b); aggressive structural cleanup is worse than doing nothing.

---

## 6. Results

### 6a. Recipe writer (hand-picked content, learned rule unchanged)

GPT-2-small (124M), fully fine-tuned, bank-template synthetic stream
world, `K=96` slots. `metabook` = same book *interface*, but slots are
`nn.Parameter`s trained by backprop and frozen at eval (the "Meta Memory
Layers"-style regime) instead of hardcoded-rule-written.

| arm | hop-1 exact % | hop-2 exact % | LM loss |
|---|---|---|---|
| **live** (writable book) | 63.0 | 33.0 | 0.58 |
| live @ 60% filler docs | 63.0 | 30.5 | 0.43 |
| live, held-out paraphrase templates | 23.8 | 25.3 | 2.36 |
| frozen (book replaced with junk at eval) | 1.4 | 6.9 | 3.29 |
| dense (plain GPT-2 fine-tune, no book) | 3.4 | 11.7 | 1.21 |
| metabook (backprop-learned, frozen slots) | 3.4 | 11.7 | 1.21 |
| live-theta (autonomous filing, no dataset flags) | 0.0 | 0.0 | 7.94 |

Key readings:
- **The channel is load-bearing, provably**: swapping in a junk book at
  eval time craters both accuracy *and* LM loss. This isn't a "maybe
  it's ignoring the memory" ambiguity — the eval literally breaks
  without it.
- **`metabook` == `dense` to 4 decimal places.** A backprop-learned
  store, frozen at eval, stores nothing useful under per-lifetime
  randomization — both arms collapse to the exact same
  "abstain/guess" behavior. This is presented as the differentiated
  claim versus published memory-layer architectures (e.g. Meta's
  Memory Layers at Scale): the *write rule* is where the value is,
  not just having a memory-shaped layer.
- Two warts, both later fixed by the learned writer (section 6b):
  holdout-paraphrase accuracy nearly halves (63->24 hop-1), and
  autonomous filing (`live-theta`) is a total failure (0/0) — raw
  GPT-2 doc-mean embeddings don't organize into a space where naive
  novelty-gated filing works at all.

### 6b. Learned writer (learned content, same write/merge rule)

Same setup, `key_head`/`pay_head` MLPs added (`graft_writer.py`), one
GPU, ~3.4 h, single run.

| arm | hop-1 | hop-2 | abstain acc | slots used |
|---|---|---|---|---|
| **live**, dataset-flagged writes | 92.5 | 96.7 | 97.1 | 55.0 |
| held-out paraphrase | 80.7 | 86.9 | 98.3 | 55.0 |
| @ 60% filler docs | 93.2 | 95.8 | 98.5 | 55.0 |
| autonomous filing, `theta=0.9`, no flags | 91.5 | 93.4 | 98.3 | 95.9 |
| autonomous filing, `theta=0.75` | 75.8 | 70.2 | 99.2 | 79.4 |
| autonomous filing, `theta=0.6` | 52.1 | 35.9 | 98.3 | 41.0 |
| frozen (junk book) | 1.5 | 6.3 | — | — |

Key readings:
- **Two-hop went from 33% to 97%** purely by changing *what* gets
  stored (learned re-rendering) — the write/merge/read rule from
  section 3 was left untouched. The bottleneck all along was the
  content representation, not the store's logic.
- **Autonomy passed**: with zero flags telling the model which
  documents are fact-shaped, novelty-gated filing in the *learned*
  key space lands within ~3 points of the cheat-flagged version
  (91.5/93.4 vs 92.5/96.7). Compare to section 6a's `live-theta`
  0.0/0.0 on the raw (non-learned) embeddings — the learned space is
  what made autonomous filing possible at all.
- **The theta asymmetry reappears at LLM scale, a 4th independent
  time**: `theta=0.6` under-births (41 slots for 55 real facts,
  collisions, 52/36) while `theta=0.9` over-births (96 slots, dilution,
  91.5/93.4 — much better despite "wasting" capacity). Confirms: bias
  toward over-birthing.
- **Paraphrase generalization fixed**: 63->24 (hand-picked embeddings)
  became 92.5->80.7 (learned hidden-state-derived keys) — learned keys
  generalize across phrasing where bag-of-embedding keys didn't.

### 6c. LoRA instead of full fine-tuning

Same task, GPT-2 100% frozen, only a rank-16 LoRA adapter (2.36M
params) + the writer heads (4.7M) are trainable (~5% of the model):

| arm | hop-1 | hop-2 |
|---|---|---|
| LoRA r16, live | 89.5 | 96.4 |
| LoRA r16, autonomous (theta=0.9) | 85.7 | 85.7 |
| full fine-tune, live (for reference) | 92.5 | 96.7 |

Two-hop is **statistically equal to full fine-tuning** (96.4 vs 96.7);
hop-1 drops modestly (-3 points). Reading: the read/write circuit
does not require deep weight changes to the backbone — a LoRA pass is
enough. The graft is portable to "any pretrained LLM + a LoRA adapter,"
not something that needs full fine-tuning access.

### 6d. Prose world (more natural narrative structure, still synthetic)

Same idea, but documents are short narratives with pronoun coreference
and distractor entities instead of bare template sentences (full
fine-tune):

| arm | hop-1 | hop-2 |
|---|---|---|
| live | 86.8 | 26.1 |
| held-out narrative structures | 29.3 | 19.6 |
| @ 60% filler | 85.3 | 29.7 |
| autonomous (theta=0.9) | 59.1 | 20.7 |
| write-all (no gate at all, write every doc) | 58.9 | 21.2 |
| frozen (junk book) | 0.1 | 3.8 |

Key reading: **single-fact recall survives messier text; two-hop
composition does not** (96.7 in the template world -> 26.1 here). The
diagnosis: hop-1 queries come straight from clean question text, but
hop-2's second query has to be *built from the retrieved payload* —
pooling ~56 tokens of real narrative smears the identity of the
bridging entity, and the two-hop chain compounds that noise. The
write-all control matching the gated control shows the *gate* itself
isn't the weak point (entity-chatter distractors don't poison what
gets written); the percept/pooling step is the diagnosed bottleneck.

---

## 7. Scope — what this is *not* evidence of yet

- **No natural-corpus run exists.** Every number above comes from
  synthetic streams: either hand-authored sentence templates filling a
  fact graph, or short authored narratives with controlled fact
  placement. GPT-2's own pretraining data is natural text, but the
  *stream it's tested on* never is. As of the last recorded session,
  "real-corpus streams (not template banks)" is explicitly listed as
  still-missing work before an external-facing claim.
- No long-context or LoRA-per-stream *competitor* baseline has been run
  (i.e. "would just giving the model a bigger context window or a
  per-conversation LoRA do just as well or better" is unanswered).
- No multi-lifetime persistence test (does the book keep working if it
  has to hold facts from many unrelated sessions at once, well past
  its capacity `K`).
- `theta`/`theta_merge` are fixed constants, calibrated per-experiment
  by sweep, in whatever representation space that experiment happens to
  use. There's no principled, self-calibrating version that has worked
  yet (see the v7/v7b failures in section 5) — this is flagged as an
  open problem, not a hidden detail.
