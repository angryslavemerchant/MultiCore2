# Top-k side-stack: nonlinear attention aggregation (v0)

*2026-08-08. Next experiment after chunk-latent v0.1. Pattern letter `T`,
module `core/sidestack.py`.*

## 1. Hypothesis

A transformer's only cross-token mixing operator — the attention output
`softmax(qk)·V` — is **linear** in the values. A weighted average can blend
retrieved tokens but cannot compute a function *of their combination*
("A and B together imply X") within one layer; that takes depth. The
side-stack replaces one layer's MLP with a small transformer that operates
on the **top-k retrieved tokens kept as distinct items**, making the
combination nonlinear inside a single layer.

Project priors say val loss is insensitive to attention *layout*
(staircase == MiMo exact tie); this changes the aggregation *operator*,
a different axis. Judged like every arm: val loss must not regress, and
capability probes (needle/retrieval) decide value.

## 2. Design (user-settled, 2026-08-08)

Trunk: **MiMo SSSF×3** (the closing-ablation arm: val 3.0193 tie with
champion, needle 57%@512 — the proven layout with the best retrieval
profile), hourglass hg0.3/bneck8/round96 d1152, uniform w=128 S layers,
full canon, speedrun stack, T=4096. That arm is the exact control: same
trunk, same data; only layer 8's MLP slot differs.

The branch replaces the MLP across **the whole middle SSSF block**
(layers 5-8, hourglass widths 576/480/384/384 — the cheap part of the
network): pattern `SSSFRRRTSSSF`. Letter `T` = full causal attention +
branch (layer 8); letter `R` = sliding-window attention + branch (layers
5-7; identical branch, candidates limited to the layer's own window —
the fast kernels don't expose scores, so selection re-derives them in a
cheap no-grad banded sweep). At each of the four layers:

1. **Attention sublayer: completely standard full causal attention**
   (augment, not replace — the smooth full-width weighted average and its
   copy path are untouched).
2. **The MLP sublayer is replaced by the side branch**, which reuses that
   attention's internals:
   - **Selection (per head):** each head's top-`side_topk` (=16) keys by
     attention score. Head-shared selection was considered and dropped —
     per-head keeps each head's own view, and the items are that head's
     **v slices** (head_dim wide, already small: no down-projection).
     Selection indices come from a no-grad chunked qk pass (top-k over
     the causal prefix); indices are non-differentiable (the NSA trade,
     same as chunk_topk).
   - **Weights:** softmax over the k *selected* logits, recomputed with
     grad on the gathered keys — renormalized-over-survivors, exactly
     the chunk_topk precedent. Each slice is scaled by its weight, so
     q and the selected k get a learning signal through the branch.
   - **Tagging:** each slice gets a log2-bucketed relative-distance
     embedding and a head-identity embedding (v slices are anonymous
     without them; RoPE lives in q/k only, values carry no position).
   - **Slice MLP:** every slice through the SAME shared MLP (relu²,
     4x at head_dim, residual, zero-init).
   - **Set attention:** all H·k slices lined up as ONE joint sequence
     (heads mix; head-id embedding disambiguates) plus a CLS seed
     projected from the query token's own post-attention state — the
     query joins the set, so its per-token transform (the deleted MLP's
     job) happens inside the branch. One bidirectional attention block
     over the 1+H·k items at head_dim width.
   - **Read-out:** the CLS item, through a final MLP with the standard
     4x rule expressed in TOKEN dim (head_dim -> 4·d -> d), zero-init
     output, added onto the residual in the MLP's old slot.

Why width = head_dim, not token dim: 192 full-width items per position
puts ~1.0G MACs/position through the set attention (~2x the whole rest
of the model). At head_dim the same structure is ~1% of model FLOPs.
The one amendment to the original sketch: the up-projection to token dim
happens ONCE, after the read-out, not per-slice before the set attention.

## 3. Cost

Measured on the real config (T=4096): control `SSSFSSSFSSSF` 142.99M
params / 0.9762 GFLOPs/tok; arm `SSSFRRRTSSSF` **139.97M params
(3M LIGHTER — deleted MLPs outweigh the branches) / 1.1099 GFLOPs/tok
= +13.7%**. Params-cheaper does NOT mean compute-cheaper: the branch
reuses its shared parameters ~192x per position, so the slice MLP + set
projections alone are ~2x the deleted MLP's FLOPs at any width, plus the
193-item set-attention scores. This is why "put it everywhere" (all 12
MLPs) was rejected: ~+70% FLOPs — the compute-matched arm would train on
far fewer tokens. Wired into `flops_per_token` via `side_extra_flops`;
runs stay matched by the usual stop-at-equal-FLOPs rule.

## 4. What would falsify it

Four sites (the middle block), k=16, augment-mode. If neither loss nor
any capability probe moves vs the MiMo control arm, the hypothesis is in
serious trouble. Pre-declared knobs before killing the idea entirely (so
they don't become post-hoc excuses): all-F sites or everywhere-with-k=4
(~+15-20%), wider set attention (2x/3x head_dim), branch-reads-V-vs-h
ablation.

## 5. Open / deferred

- Entmax-1.5 on the selection weights (sharper membership) — after v0.
- Set-attention depth 2 — after v0.
- Global-softmax-mass gate (how much total attention the selected k
  actually held — renormalization discards it) as a scalar feature.
- EVERY run needs explicit per-run user confirmation (standing rule).
