# Hierarchical GPT-2 with Predictive Plans and Block-Level Sparse Memory

## Status and purpose

This document specifies one testable, GPT-2-scale language model inspired by the architectural clues discussed around Incantor's Cantor model. It is **not** a reconstruction of Cantor. The undisclosed parts are treated as hypotheses.

The experiment tests one focused proposition:

> Stable hierarchical representations make a large sparse parametric memory easier to route and more useful than querying that memory only from token states.

The model must remain causally valid and substantially parallelizable during teacher-forced training.

## 1. Fixed configuration

| Item | Value |
|---|---:|
| Training sequence length | 4,096 tokens |
| Vocabulary | 50,257 BPE tokens |
| Core block size | 64 tokens |
| Backward overlap | 64 tokens |
| Token window | 128 tokens |
| Block stride | 64 tokens |
| Blocks per sequence | 64 |
| Blocks per superblock | 8 |
| Superblocks per sequence | 8 |
| Token width | 768 |
| Token SwiGLU hidden width | 2,048 |
| Block width | 512 |
| Block SwiGLU hidden width | 1,376 |
| Superblock width | 384 |
| Superblock SwiGLU hidden width | 1,024 |
| Token analysis layers | 4 |
| Token prediction layers | 4 |
| Block predictor layers | 4 |
| Superblock predictor layers | 2 |
| Sparse memory slots | 65,536 |
| Sparse memory value width | 128 |
| Memory query heads | 4 |
| Retrieved slots per head | 4 |
| Approximate total parameters | 121–124M |

Use pre-normalization, RoPE, RMSNorm, SwiGLU, tied token input/output embeddings, and no absolute token-position embeddings. The listed SwiGLU widths keep its three projections near the parameter cost of a conventional two-projection MLP with expansion factor four. Exact initialization and optimizer settings should initially follow a stable GPT-2-scale baseline.

## 2. High-level dataflow

```text
4,096 tokens
    |
    | 64 overlapping local windows, processed as a batch
    v
token analysis network (4 layers, width 768)
    |
    +--> 64 observed block summaries S[0:64]
              |
              +--> 8 observed superblock summaries U[0:8]
              |         |
              |         v
              |    shifted causal superblock predictor
              |         |
              |         +--> superblock plans G[0:8]
              |
              v
        shifted causal block predictor, conditioned by G
              |
              +--> product-key sparse memory
              |
              +--> block plans P[0:64]
                         |
                         v
token prediction network (4 layers, width 768), conditioned by P and G
    |
    v
next-token logits for all 4,096 target positions
```

The stages are sequential with respect to one another, but every token window, block position, and superblock position is processed in parallel within its stage during training.

## 3. Token windows and loss alignment

Let block `b` begin at global token index `t = 64b`.

Construct a 128-token window:

```text
x[t-64 : t+64]
```

For the first block, left-pad with a learned BOS/history token. Windows never include tokens beyond the end of their current 64-token core.

Standard causal logits at window positions `63..126` predict the 64 target tokens:

```text
x[t : t+64]
```

The representation at window position 127 corresponds to the final observed token of the block and is used when constructing its completed summary. Its next-token logit is excluded from this block's loss because that target belongs to the next block.

Consequences:

- Each target token contributes to the primary loss exactly once.
- The preceding 64 tokens are context only.
- The overlapping implementation approximately doubles token-stage training compute relative to nonoverlapping chunks. A sliding-window kernel may optimize this later without changing semantics.

## 4. Token analysis network

Each of the 64 windows is processed independently by a shared four-layer causal Transformer:

```text
H_b = TokenAnalysis(window_b)
H_b shape: [128, 768]
```

This network has no access to the completed summary of block `b`, its block plan, or any future plan. Its job is to produce locally contextual token representations from which an observed summary can be minted.

Construct the observed block summary:

```text
core_mean_b = mean(H_b[64:128], axis=token)
core_last_b = H_b[127]
S_b = RMSNorm(W_summary [core_mean_b ; core_last_b])
S_b shape: [512]
```

Using both mean and final state is deliberately simple. Learned attention pooling is deferred until an ablation justifies it.

## 5. Superblock pathway

Group every eight observed block summaries:

```text
group g = S[8g : 8g+8]
```

Create an observed superblock summary only after all eight constituent blocks have been observed:

```text
U_g = RMSNorm(W_super [mean(group_g) ; group_g[-1]])
U_g shape: [384]
```

Run a two-layer causal Transformer over a right-shifted sequence:

```text
super_input = [SUPER_BOS, U_0, U_1, ..., U_6]
G_0, G_1, ..., G_7 = SuperPredictor(super_input)
```

Thus `G_g` predicts/conditions superblock `g` using only `U_<g`. It must never consume `U_g` or later summaries.

`G_g` is not trained to reproduce `U_g` with mean-squared error. It is a predictive plan whose meaning is determined by whether it improves future-token prediction.

## 6. Block predictor

Run a four-layer causal block Transformer over a right-shifted block-summary sequence:

```text
block_input_b = W_block(S_{b-1}) + W_condition(G_floor(b/8))
block_input_0 = BLOCK_BOS + W_condition(G_0)
P_b = BlockPredictor(block_input_0 ... block_input_b)[b]
P_b shape: [512]
```

The entire shifted sequence is evaluated in parallel under a causal mask during training.

The block predictor may use summaries from earlier blocks in the same superblock. The superblock plan for the current group contains information only from previous completed superblocks.

### Block-level sparse memory placement

Add exactly one sparse-memory residual sublayer after the attention sublayer of block-predictor layer 3. Do not remove that layer's dense SwiGLU MLP in the initial model.

```text
h = h + BlockAttention(RMSNorm(h))
h = h + DenseSwiGLU(RMSNorm(h))
h = h + gate(h) * ProductKeyMemory(RMSNorm(h))
```

Initialize the memory residual gate near zero so early training behaves like the dense hierarchical model. A scalar gate is acceptable initially; a per-channel gate is optional.

## 7. Product-key sparse memory

### Address space

Use a shared table of 65,536 values:

```text
N = 256
addresses = N x N = 65,536
ValueTable shape: [65,536, 128]
```

Each memory head projects a 512-dimensional block state into a 128-dimensional query and splits it:

```text
q_h = Wq_h h                  # [128]
qA_h, qB_h = split(q_h)       # [64], [64]
KeyA_h shape: [256, 64]
KeyB_h shape: [256, 64]
```

For each of four heads:

1. Score all 256 entries in each sub-key codebook.
2. Retain the best 32 from each half.
3. Form the 1,024 Cartesian candidate pairs.
4. Score pair `(i,j)` by `scoreA_i + scoreB_j`.
5. Select the best four addresses.
6. Softmax their scores and return the weighted sum of their 128-dimensional values.

Concatenate the four head outputs to obtain 512 dimensions:

```text
memory_output = concat(head_0, head_1, head_2, head_3)
```

The heads share the value table but use separate query projections and sub-key codebooks. Deduplicate repeated addresses for utilization statistics, not for the mathematical output.

### Gradient behavior

Gradients flow through:

- the selected values;
- the selected product-key scores;
- the query projections;
- the residual gate.

Hard top-k routing means unselected values receive no gradient on that example. Track starvation explicitly.

### Required routing telemetry

Log at least:

- fraction of memory slots used per 1M training tokens;
- routing entropy per head;
- mean and maximum traffic per slot;
- fraction of duplicate selections across heads;
- gradient norm for keys, selected values, and the memory gate;
- memory contribution norm relative to the dense block MLP;
- utilization separately by block position and superblock position.

## 8. Top-down token conditioning

For every token in block `b`, construct a conditioning vector from its causal plans:

```text
C_b = Wc [P_b ; G_floor(b/8)]
C_b shape: [768]
```

Feed the analyzed token states through a separate four-layer causal token-prediction Transformer. Apply the same `C_b` to all positions in the window, but calculate loss only on the core targets.

Use gated residual conditioning in every prediction layer:

```text
h = h + TokenAttention(RMSNorm(h))
h = h + DenseSwiGLU(RMSNorm(h))
h = h + sigmoid(a_l) * W_l C_b
```

Initialize `a_l` so the conditioning path begins small but nonzero. FiLM/AdaLN conditioning may be tested later; simple gated addition is the reference implementation.

The final logits use the tied token embedding matrix.

## 9. Objectives

### Primary next-token loss

Compute cross-entropy over all 4,096 tokens, with each token assigned to exactly one block core:

```text
L_ntp = mean CE(token_logits_t, x_t)
```

### Block-plan auxiliary loss

Make each `P_b` predict eight sampled token positions from block `b`, at fixed offsets:

```text
offsets = [7, 15, 23, 31, 39, 47, 55, 63]
```

Add a learned offset embedding to `P_b`, project to token width, and use the tied vocabulary matrix. This objective does not claim that the plan contains an exact reconstruction of the future block; it only pressures the plan to carry information useful for its token distribution.

### Superblock-plan auxiliary loss

Make `G_g` predict one sampled token from each of the eight blocks in superblock `g`. Use a fixed offset within each block initially, rotating that offset between batches if convenient.

### Routing regularization

Use a small load-balancing penalty based on batch-level traffic across the two sub-key codebooks. Do not attempt a full 65,536-way uniformity loss.

Initial total objective:

```text
L = L_ntp
  + 0.10 * L_block_plan
  + 0.05 * L_super_plan
  + 0.01 * L_route_balance
```

These coefficients are starting points, not architectural claims. Log every component separately and reduce an auxiliary weight if it improves its proxy task while degrading validation NTP.

## 10. Causality requirements

These are correctness constraints, not optional tests:

1. Token targets in block `b` may use raw tokens before the target position.
2. They may use `S_<b`, never completed `S_b`.
3. They may use `U_<g`, never completed `U_g`, where `g=floor(b/8)`.
4. `P_b` is produced from shifted block summaries.
5. `G_g` is produced from shifted superblock summaries.
6. Auxiliary prediction heads receive only the corresponding causal plan, positional query, and globally fixed parameters.

Required perturbation test:

- Change every token at or after the beginning of block `b`.
- Logits predicting the first token of block `b` must remain numerically unchanged, apart from nondeterministic kernel tolerance.
- Repeat at a superblock boundary for the effect of `G_g`.

## 11. Parameter estimate

Approximate parameter allocation:

| Component | Parameters |
|---|---:|
| Tied token embedding/output | 38.6M |
| Eight token Transformer layers, width 768 | 56.6M |
| Four block layers, width 512 | 12.6M |
| Two superblock layers, width 384 | 3.5M |
| Sparse value table | 8.4M |
| Product keys and memory projections | ~0.5M |
| Summary, conditioning, and auxiliary projections | ~1–3M |
| **Total** | **~121–124M** |

Count parameters from the actual implementation and publish the exact total. If SwiGLU expansion choices push the total materially above 125M, reduce its hidden width before reducing the sparse table.

## 12. Training parallelism and inference state

### Teacher-forced training

For each 4,096-token example:

1. Stack all 64 token windows along the batch dimension.
2. Run token analysis for every window in parallel.
3. Mint all 64 observed block summaries.
4. Mint all eight observed superblock summaries.
5. Run the shifted superblock predictor over eight positions.
6. Run the shifted block predictor over 64 positions, including sparse lookup.
7. Stack conditioned token windows and run token prediction in parallel.
8. Compute primary, plan, and routing losses.

There is no sequential loop over the 4,096 training tokens.

### Autoregressive inference

Maintain:

- the most recent 64 raw tokens;
- completed block summaries;
- the block-predictor KV cache;
- completed superblock summaries;
- the superblock-predictor KV cache;
- the current block and superblock plans.

At a block boundary, finalize `S_b` and update the block predictor. Every eight blocks, finalize `U_g` and update the superblock predictor before producing the next superblock plan.

The reference implementation may recompute the current 128-token window while generating. KV caching inside the local token networks is an optimization milestone, not a prerequisite for validating the architecture.

## 13. Mandatory ablations

Train with identical data order and token budget wherever possible:

| ID | Model |
|---|---|
| A | Flat GPT-2-scale baseline |
| B | Hierarchy with no sparse memory and no plan auxiliary losses |
| C | Hierarchy plus block/superblock plan losses, no sparse memory |
| D | Full specified model: predictive hierarchy plus block memory |
| E | Same as D, but query the memory from token states instead of block states, compute-matched |
| F | Same as D, but random/frozen product keys |

The central comparison is `D` versus `C` and `E`. A gain over `C` shows memory utility; a gain over `E` supports the claim that hierarchical block states are better memory routers.

Report:

- validation loss and perplexity versus training FLOPs;
- validation loss versus tokens seen;
- wall-clock throughput;
- peak accelerator memory;
- long-context retrieval probes;
- factual completion probes;
- memory-routing statistics;
- performance when the memory residual gate is disabled at inference.

## 14. Implementation order

1. Implement window construction and verify every target appears exactly once.
2. Implement token analysis, summary minting, shifted block prediction, and token conditioning without the superblock pathway.
3. Pass causality perturbation tests and overfit a tiny dataset.
4. Add the superblock summary and shifted superblock plan.
5. Add plan auxiliary heads and verify that all targets are future-only.
6. Add the sparse memory with its residual gate initialized near zero.
7. Add routing telemetry and load balancing.
8. Run ablations before attempting runtime episodic memory.

## 15. Explicit non-goals for this model

Do not include in the first serious implementation:

- writable episodic memory at inference;
- fine-detail residual retrieval;
- dynamic memory compression or reorganization;
- replacement of every dense MLP;
- exact latent-summary regression;
- semantic actor/emotion labels;
- shared weights across hierarchy levels;
- a claim that this reproduces Cantor.

Those are follow-on experiments. This model isolates the most defensible novel interaction: predictive temporal hierarchy used as the router for a very large, sparsely activated parametric memory.
