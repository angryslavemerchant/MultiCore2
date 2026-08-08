# Chunk-Latent Attention — v0 Specification

*Working name. A GPT-2-scale testbed for runtime dynamic chunking: content-generated,
free-membership chunk latents minted over the KV cache on a fixed cadence, accumulating in a
per-layer chunk log that serves as the model's only long-range pathway.*

---

## 0. Thesis

Fixed tokenization and fixed attention blocks allocate resolution uniformly; information is not
uniform. This architecture lets the model mint **chunks** — latent KV pairs summarizing
arbitrary (non-contiguous) subsets of the cache — on a fixed schedule, as a **pure function of
the causal prefix**. Chunks accumulate in a growing log (NSA-style: every mint persists and
stays attendable); attention reads over {local raw tokens} ∪ {chunk log}. Long-range
information travels *only* through chunks, forcing them to become load-bearing abstractions.

Core claim to falsify: **learned free-membership chunks beat fixed contiguous block summaries
as the long-range pathway, at matched parameters and compute.**

Design philosophy: v0 contains only what the conceptual idea requires. Every defensive
mechanism (dedup, auxiliary losses, anchors, memoization) is demoted to §10 with the diagnostic
that would justify adding it back — so each is identified empirically as a fix or discarded as
superstition.

---

## 1. Base model

- nanoGPT / GPT-2 small: `d = 768`, `n_layer = 12`, `n_head = 12`, standard BPE tokenizer
  (tokenizer-level chunking is out of scope; this tests chunking of *context*, not of bytes).
- **Raw self-attention is restricted to a local causal window `w`** (default `w = 256`) in all
  layers. This is the forcing function: distant context is reachable only through chunks.
  Without it, chunks are optional garnish and the writer starves for gradient.

## 2. Chunk writer

**Schedule.** Fixed cadence: at every boundary `b` (every `B_tok` tokens, default
`B_tok = 256`), mint `K` chunks (default `K = 16`). Rigidity lives in the schedule; semantics
stay free.

**Per-layer, NSA-style.** Every transformer layer has its own writer and its own chunk log,
built from that layer's *own* attention keys and values (`K^ℓ`, `V^ℓ` — tensors the layer
already computes; no extra projections, no cross-layer plumbing). Layer ℓ's chunks live in
layer ℓ's representation space and are read only by layer ℓ.

**Generated queries.** Chunk queries are generated from content, MTP-head-style: a per-layer
head reads the trunk's hidden states over the just-completed block and emits K query vectors:

```
Q_b^ℓ = Head^ℓ( pool(h^ℓ over block b) )               # (K × d)
```

Generated queries are a pure function of the causal prefix — training parallelism and
causality unaffected.

**Head capacity is a first-class sweep, not a contingency.** The MTP/EAGLE "one block
suffices" datapoint does not bound this head: those heads do trunk-flavored work (predict a
token), so the trunk's representation is already nearly the right shape. Query generation —
deciding what deserves to become an object — is a different job, and the object-forming
literature (DETR's 6-block decoder, Perceiver's deep latent formation, slot attention's
iterative refinement) consistently spends real capacity on it. Ladder to sweep:
1. Per-layer linear (or 2-layer MLP) head on the mean-pooled block state — bets that layer
   ℓ's trunk beneath is already the deep generator.
2. One shared dedicated transformer block reading the block's hidden states (attends within
   the block instead of mean-pooling), per-layer output projections.
3. Deeper query stacks (2+ blocks). Affordable at this scale; watch the parameter-count
   confound in baseline comparisons.
Diagnostic: generic/positional membership maps at a given rung → climb before concluding free
membership itself failed.

**Membership.** For boundary `b` at layer ℓ:

```
A_b^ℓ = softmax( Q_b^ℓ · K^ℓ[:, :t_b]^T / sqrt(d) )    # (K × t_b), t_b = b · B_tok
R_b^ℓ = A_b^ℓ · V^ℓ[:t_b]                               # (K × d) pooled reads
```

`A_b` **is** the chunking: membership is attention affinity — soft, non-contiguous, global
over the whole prefix. A chunk minted at boundary 20 may bind a token from just now together
with tokens from thousands of positions back ("edge" chunks relating new material to old).

**Write.** An MLP maps each pooled read to a chunk KV pair:

```
(k_i, v_i) = MLP(R_b^ℓ[i] ⊕ Q_b^ℓ[i] ⊕ e_ℓ)            # k_i, v_i ∈ R^d
```

One MLP shared across all layers, conditioned by a learned layer embedding `e_ℓ` (per-layer
MLPs would be ~2–4M × 12 layers on a 124M model — ablation, not default). Optionally add a
small boundary-index embedding so chunks carry coarse temporal position.

**Slot anti-collapse fallback.** If all K slots collapse to one reading, switch `A_b`'s
normalization to slot-attention style (softmax over slots per token) so tokens are fought over
and owned.

## 3. Chunk log

Chunks **accumulate**: each layer's log after boundary `b` holds `b·K` entries — every mint
persists, immutable, attendable for the rest of the sequence. This is the NSA property
(compressed block summaries persist for the whole context) with membership freed from
positional slicing. No deduplication in v0: near-duplicate chunks coexist; softmax splits
attention mass between similar keys, but their values are similar too, so the mixture ≈ either
one (small sharpness cost, measured not pre-engineered away — see §10.1 for the trigger).

Population math: linear growth, `K/B_tok` chunks per token — at K=16, B_tok=256 that is one
chunk per 16 tokens; a 16k context accrues 1,024 chunks per layer. Trivial to attend over at
v0 scales; capacity control (§10.1) is a long-context problem.

**Immutability corollary.** A chunk is a pure function of prefix ≤ its mint boundary and is
never recomputed. "Revision" in v0 is implicit: later boundaries mint fresh chunks over the
updated prefix, and stale + fresh coexist. No staleness machinery, no memoization, no dirty
flags exist in v0 (§10.4).

## 4. Loss

```
L = L_LM        (next-token cross-entropy — nothing else)
```

The local window makes chunks the only long-range path, so every prediction depending on
distant context sends gradient through the writer. Auxiliary losses are §10 items with
triggers.

## 5. Reading chunks (main attention)

At **every layer**, each token's attention population is:

```
{ raw tokens in its local window w } ∪ { that layer's chunk log as of its block's last boundary }
```

Layer ℓ's chunks are written in layer ℓ's KV space, so they concatenate directly onto the
layer's key/value tensors — chunks are literally extra tokens, split across heads like any
other token, scored by ordinary attention. No adapters, no selection machinery.

**Causality.** A chunk minted at boundary `b` binds only tokens `≤ t_b`. A query at position
`t` in block `b+1` reads chunks minted at boundaries `≤ b`. Time is fine-grained for raw
attention, quantized to block ticks for chunk visibility (Transformer-XL masking geometry).

## 6. Training-time execution (fully parallel)

No recurrence anywhere — chunk tables are pure functions of prefixes, never of previous
tables. At each layer ℓ, given its input hidden states (all positions available
simultaneously):

1. Compute the layer's `K^ℓ, V^ℓ` as normal.
2. Pool hidden states per block; `Head^ℓ` emits all `B × K` queries in one batch.
3. One batched masked cross-attention: `(B × K)` queries vs `T` keys, each boundary's mask
   zeroing keys past `t_b`. Cost ~ `(K/B_tok)·T²` per layer — same order as the layer's own
   attention.
4. Shared MLP → the layer's `B·K` chunk KVs.
5. The layer's attention runs over the mixed population with block-structured visibility
   masks; output feeds the next layer, which repeats 1–5 on its own representations.
6. L_LM; backprop. Writer gradient flows through every later query that attended a chunk.

Teacher-forcing-style redundancy (computing chunk tables at every boundary when a given query
uses only some) is the standard price of parallelism.

## 7. Inference-time execution (serial decode)

Trivial in v0, because mints are immutable:

- Decode token-by-token; attend over local window + the layer's chunk log.
- Every `B_tok` tokens: pool the completed block, generate queries, one cross-attention over
  the prefix, MLP, **append** K chunks per layer. Nothing is ever recomputed, revised, or
  evicted.

## 8. Experiments

**Baselines (matched params — writer + heads add a few M; give baselines an equivalent bump):**
1. Local window only (`w = 256`).
2. Local window + **fixed contiguous block summaries** (same cadence, same K, same MLP, but
   membership = positional slicing) — the critical control: it is what the architecture
   degenerates into if generated free membership learns nothing.
3. (Optional) full-attention GPT-2 as a ceiling reference.

**Tasks.**
- OpenWebText perplexity (sanity: chunks shouldn't hurt short-range modeling).
- PG19 / long-document perplexity, stratified by token position — the thesis test is whether
  the gap over baseline 2 *grows* with distance.
- Synthetic needle-in-a-haystack retrieval at 4k–16k tokens.

**Core diagnostics.**
- **Membership contiguity index**: fraction of each chunk's attention mass outside its largest
  contiguous run. ≈ 0 means the writer reinvented fixed blocks — thesis fails informatively.
- **Instance separation**: do distinct same-type entities get distinct chunks, or smear?
- **Chunk usage**: attention mass to chunks vs raw, per layer / per distance bucket.
- **Cross-layer chunking divergence**: near-identical membership maps across layers argue for
  collapsing to one writer; divergence justifies the per-layer cost.
- **§10 trigger meters** (cheap to log from day one): revision rate (fraction of mints with
  high key-similarity to an existing chunk), stale-read errors (attention mass on superseded
  facts in probes designed to have updated answers), log-dilution (entropy of attention over
  the chunk log as it grows).

## 9. Hyperparameters (initial)

| knob | default | sweep |
|---|---|---|
| `B_tok` (write cadence) | 256 | 64 / 256 / 512 |
| `K` (slots per boundary) | 16 | 8 / 16 / 32 |
| `w` (local window) | 256 | 128 / 256 |
| Head capacity | rung 1 (linear/MLP) | rungs 1–3 (first-class sweep) |
| writer MLP sharing | shared + `e_ℓ` | per-layer MLPs |

## 10. Demoted machinery — v2 extensions, each with its trigger

Everything here was designed in full during development and is documented so it can be
reinstated deliberately, not rediscovered.

**10.1 Keep-latest dedup + annealed thresholds + key-repulsion loss.**
Design: chunk ineligible if a later mint's key exceeds `δ_hard` similarity (succession =
in-place revision; discard, don't merge); repulsion penalizes only the coexistence band
`δ_soft < sim < δ_hard` (squared hinge, over surviving pairs — succession stays free); both
thresholds anneal down with boundary index, a packing constraint forcing coarser, sublinear
populations at long range; unit-normalize keys if reinstated.
Triggers: stale-read errors (model retrieves superseded values — the stale+fresh value
mixture is the one real cost of no-dedup); log-dilution degrading retrieval as context grows;
population cost mattering at 100k+ contexts.

**10.2 Reconstruction auxiliary loss.** Chunk value predicts its members' pooled
representation (dense local gradient for the writer).
Trigger: writer visibly undertrained — chunks used but uninformative, LM-gradient starvation.

**10.3 Static anchor queries** (`Q = anchor + Head(·)`).
Trigger: revision-rate meter collapsing — concepts not re-visited across boundaries, evolving
content never re-minted.

**10.4 Memoization / dirty flags.** Only meaningful if chunks become recomputable again
(i.e., if 10.1's revision semantics are reinstated in recompute form).

**10.5 Iterative query refinement** (2–3 slot-attention rounds `q ← q + f(A·V)`).
Trigger: instance-separation diagnostic shows type-smearing that head-capacity rungs don't fix.

**10.6 Hierarchy.** Writers reading chunks as input → chunks-of-chunks, recursive coarsening,
raw-token eviction under stable chunks (foveated memory pyramid). v0 is decided raw-tokens-only
(chunk information still reaches writers indirectly, via trunk representations computed through
chunk-reading layers below).
Trigger: log growth, or chunks plateauing at first-order abstractions once contexts are long
enough to matter.
Accepted route if triggered: same-layer chunk consumption via **chunkwise training** — writers
at boundary b read chunks from boundaries < b; train with a block-granular sequential scan
(~T/B_tok small steps, the standard price of state), possibly with a smaller `B_tok` (e.g. 128)
so abstractions compound faster. Zero-serialism alternative: layer ℓ's writer reads layer
ℓ−1's chunk log (hierarchy climbs layers instead of time; needs one cross-layer projection).

**10.7 Persistent chunks (cross-session retention) — the endgame.**
The chunk log already accumulates over a lifetime; persistence extends it past the sequence
boundary: at end-of-session, retain the top-k chunks per layer by cumulative attention usage
(the readers' own revealed preference for which abstractions earned permanence) and seed the
next session's log with them. Design: Transformer-XL-style segment recurrence — session =
segment, carried state = retained chunks, **stop-gradient at the boundary** (full cross-session
BPTT is intractable; the writer gets only within-session gradient, so the load-bearing new
claim is that "useful at distance 3k within a session" transfers to "useful next session").
Known costs, in order of severity: (a) representation drift during training — persistent
chunks live in a checkpoint's KV space and decay into noise as weights move; keep the
retention horizon short relative to weight movement, and note the problem vanishes at
inference, where persistence matters most; (b) training data must contain cross-session
structure — on i.i.d. docs, inherited chunks are noise and the model correctly learns to
ignore them; needs per-lane contiguous document streams or multi-episode synthetic data
(extend a stream-probe lifetime across several sessions with the log culled to top-k at each
boundary, grading retrieval of pre-cull facts — the eval nearly for free); (c) top-k-by-usage
is non-differentiable and selects for what *this* session needed — fine for v1; a learned
retention score is the follow-on. Free ablation: retained-top-k vs retained-random-k vs
no-carry isolates whether usage-based selection specifically does the work.
Trigger: v0 holds (free membership beats contiguous summaries) — a strict prerequisite; if
chunks don't carry information well within a session, retaining them across sessions inherits
nothing. Costs §6's full parallelism only *across* sessions; within a session everything still
batches.

**10.8 Other.** Data-dependent cadence or K (reintroduces discrete decisions — the H-Net tar
pit; only if fixed cadence visibly wastes capacity). Learned thresholds via straight-through.
Byte-level base model. Query-conditioned *reading* (retrieval-time re-chunking).

## 11. Known risks, ranked

1. **Writer learns positional pooling** (chunks ≈ contiguous span means). Detectable day one
   via contiguity index; kills the interesting claim cheaply.
2. **Weak query generation at rung 1** mistaken for failure of the idea — climb the ladder
   before concluding.
3. **Stale-value mixtures** in long generations (the accepted no-dedup cost) — watch the
   stale-read meter; 10.1 is the fix.
4. **Slot collapse** (all K queries converge) — slot-normalized competition is the fallback.
5. **Within-block staleness**: a concept revised mid-block is invisible to chunks until the
   next tick. Bounded by `B_tok`; the local window covers the tail meanwhile.

---

## 12. v0.1 amendment (2026-08-08) — post-first-run redesign

v0 result: chunk arm (12x K, btok=256, K=16, soft membership, rung-1
head) val **3.1200** @ 2.5B tokens — 0.10 nats behind the pyramid
champion (3.0199) at matched tokens/FLOPs; controls not run before the
redesign. Changes, user-directed:

- **Middle (waist) layers only** carry chunk machinery (e.g.
  `SSSKKKKKKSSS`): the hourglass makes them the cheap layers; outer
  layers stay plain w=256 SWA.
- **Write more often, mint less**: btok=64, K=2.
- **Attention-pooled queries** (rung-2): the K probes attend over the
  just-completed block (PMA-style, residual on probe) instead of
  mean-pool+Linear.
- **Hard top-k membership** (topk=16): softmax renormalized over each
  query's top-k prefix keys; gradient reaches selected members' scores,
  the selection gets none (the NSA trade; their score-sharing trick is
  the known fix if it starves).
- Framing sharpened: blocksum == NSA's compression branch, so the
  comparison is *NSA-style positional compression vs learned top-k
  membership*, same cadence, same middle layers, one joint softmax
  both arms.
- Rejected en route: late-layer clone as query source (single writer
  fed by a late layer) — token-causal but not compute-causal; early
  layers can't read late-sourced chunks in one parallel pass. Revisit
  only as late-read-only (chunks visible to layers >= writer source).

---

## 13. v0.2 (2026-08-08) — recursion + dedup + raw fetch (pattern N)

Built after the full v0/v0.1 capability bench (2026-08-08): chunk log =
huge distance-flat gist memory, but ZERO exact fetch beyond the window,
weak update-tracking (append-only keeps stale values live), and the
verdict that capability is mechanism-intrinsic. v0.2 changes the
mechanism on all three axes; `core/chunkv2.py`, pattern letter **N**
(K stays v0.1, checkpoints keep loading).

- **Recursive minting, fully sequential** (user: no wavefront/lag
  affordances — stay true to the idea). Boundary b's writer candidates
  = raw prefix UNION all previously minted chunks. Chunk→chunk
  references fold in the LATENT only, never transitively expanded.
  Mint loop runs eagerly under `torch.compiler.disable`; the
  sequential tax (est. 1.15–2x) is measured by the gate before any run.
- **Soft dedup** (read-side): per-head learned λ, init 0 = exact
  append-only v0.1; each chunk's read logit penalized by
  λ·relu(max cos sim to any NEWER chunk). Newest-wins ≈ the cheap COW
  merge, which won update-margin.
- **Raw member fetch** (NSA fine branch; the bench showed pointers
  select right while summaries smear): each query attends the raw
  pointer sets of its top `chunk_fetch_n` chunks as a THIRD branch of
  the joint softmax (3-way LSE merge), position-free. Chunk-pointers
  are never fetched — summary-of-summary content stays gist.
- **Run shape** (settled): `SSSNNNNNNSSS` d1152 hourglass = cheapest-6
  layers exactly (widths 672/576/480/384/384/480; 6th place is a
  4-vs-10 tie at 672, kept at 4 for v0.1 comparability). Uniform
  **w=128** everywhere (MiMo-proven; pays for the fetch keys and
  pushes more gradient through the chunk path). btok=128, K=4,
  topk=16, fetch_n=4. 150.70M params, 0.9242 GFLOPs/tok — cheaper
  than the mimo control (142.99M, 0.9762) and ~level with v0.1's arm.
- Gate `scripts/validate_chunkv2.py`: fast-vs-reference parity FWD+BWD,
  compiled-vs-eager logits AND per-param grad cosines (flat-loss
  lesson), zero-init projections randomized first, sequential-mint
  wall-clock vs window-only control.

### 13.1 Gate saga addendum (2026-08-08, 1x5090)

Two kernel-shape lessons, both caught by the gate before any run:
1. **Gather-on-expanded-view backward materialises the source**: the
   fetch's per-query gather from a (B,H,T,T,hd) expand OOM'd at 21 GiB
   — gather along the sequence dim instead (backward = scatter-add).
2. **Per-query sparse fetch backward = atomic-scatter hotspot**: one
   fused scatter_add kernel was 78% of the training step (868ms,
   value-dependent — appears only once training sharpens selection and
   every query fetches the same popular chunks; 9.5x wall-clock, and
   invisible at init, so bench WITH optimizer steps). Fix: dense-with-
   mask fetch — members gathered once per chunk (S·kk rows), all
   scored per query, non-selected masked to -inf. Same math, GEMM
   gradients. FLOPs now honest at 0.9906 G/tok (+1.5% vs mimo ctrl).
Final gate numbers: parity fwd+bwd exact; compiled-vs-eager logits
0.0625, grads bf16 trunk 0.99853 / selection 0.97928 (tie-flips), fp32
no-flex leg 0.999944; bench 335.9ms vs 126.7 window-only = **2.65x**
(24.4k tok/s/GPU mb2, peak 20.1GB) — the honest fully-sequential tax.
