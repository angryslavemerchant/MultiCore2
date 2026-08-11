# Four-Stroke / Machine Population — v1 Spec (post-workshop)

Supersedes v0. Core changes: soft everything (no routers, no aux losses),
one transport primitive, sigmoid write gate, state-derived addresses with
learned anchors, private conclusion channel, swappable state backend.

---

## 0. The one primitive

All information transport is a single operation:
**publishers expose (K, V) derived from their state; readers issue Q.**

- Intake      = machine reads the token stream   (machine Q, tokens KV)
- Conference  = machine reads other machines      (machine Q, machines KV)
- Consultation= token reads the machines          (token Q, machines KV)

No routers, no straight-through estimators, no load-balancing losses.
Specialization is emergent via address differentiation (see §3).

## 1. Objects

- N stream tokens, d_model. K machines (v0 run: K = 16).
- Machine k at layer ℓ, position t has state `s[k,ℓ,t]` (d_machine, e.g. 256).
- Learned init vector `s0_k` = the machine's innate character (state at t=0,
  and the pretrained "boot reputation").
- Learned anchor embedding `e_k` = the fixed component of its address.

## 2. Machine = backend + interface (hard module boundary)

### State backend (swappable plugin; config flag)
Produces `s[k,ℓ,t]` causally. Contract: function only of layer ℓ−1 values at
positions ≤ t (the delay-one-layer rule) + s0_k. Fixed-size output. Variants:
  - `attn`  : full attention over the machine's private channel (v1 default)
  - `swa`   : sliding-window attention over the private channel
  - `linear`: linear-attention / SSM-style running state
  - (later) `ttt`: gradient-updated mini-net — persistence experiments
The backend's internal projections are PRIVATE — never shared with §2b.

Private channel content (the machine's memory): its own past
post-conference outputs from layer ℓ−1 (conclusions, unmixed, full fidelity)
PLUS read access to the layer ℓ−1 residual stream (intake, via the primitive).

### Circuit interface (fixed; the architecture)
Dedicated projections from state — the machine's face:
  - `q[k] = W_Q^k · s[k]`            what it wants to know
  - `k[k] = e_k + W_K^k · s[k]`      its address: anchor + lived reputation
  - `v[k] = W_V^k · s[k]`            what it offers
Interface is identical regardless of backend. Persistence later = backend
patch (don't reset state between contexts); circuit unchanged.

## 3. Layer dataflow (per position t, all positions parallel)

1. **State update (intake included).** Backend computes `s[k,ℓ,t]` from
   private channel + current layer ℓ−1 residual window. Then private MLP_k.
2. **Publish.** Interface produces (q, k, v) for each machine.
3. **Conference.** K×K attention among machines' published tuples
   (batched over positions; cost N·K²·d_machine — negligible at K=16).
   Output per machine: conclusion `c[k,ℓ,t]`. This is what enters the
   machine's own private channel for layer ℓ+1.
4. **Write-back (sigmoid gate, NOT softmax).**
   `y[t] = Σ_k σ(w_g · [x_t ; c[k]]) · W_O^k · c[k]`
   Independent gates per machine; all gates can close (silence is legal;
   avoids forced-speech / attention-sink pathology). y added to residual.

## 4. Model & block layout

- ~15–30M params, d_model 512, 8 blocks, context 1–2k.
- Machine population is SHARED across all machine-blocks (same K machines,
  same identities through depth — required for coherent addresses; their
  states are per-layer but the anchors/inits/interfaces are one set).
  Ablation flag: per-layer independent populations.
- Interleave: standard block / machine block alternating. Machine block
  keeps its own FFN on the token path (strokes replace attention only).

## 5. Stability notes

- Drifting addresses: the anchor `e_k` in the published key is the
  stabilizer. Watch conference attention entropy early in training; if
  machines can't find each other, upweight anchor vs. state term
  (learnable mixing scalar, init biased toward anchor).
- Machine state norm: RMSNorm on s before interface projections.
- Init diversity: initialize `s0_k` and `e_k` with enough spread that
  machines start distinguishable (orthogonal init for anchors).

## 6. Baselines & evals (unchanged from v0, restated)

1. Matched-params, matched-FLOPs standard transformer (count K×K conference
   and per-machine projections honestly).
2. Dummy-register control: standard transformer + K value-less learned
   registers (structure-free token-budget match; already built).
3. Ablations: no-conference (skip step 3), shared-machine-params (Perceiver-
   izing — kills the society, keeps the topology), softmax-out (vs gate),
   backend swap (attn/swa/linear).
- General eval: LM loss, TinyStories or FineWeb-edu subset. Expect tie.
- Targeted eval (the real test): per-category accumulation streams —
  latent-category tokens, queries needing per-category aggregates
  (count / mean / most-recent) at ranges where full attention gets noisy;
  MQAR-style variants. Vary #categories vs K (below / at / above).
- Diagnostics: gate openness rates, conference attention maps (do machines
  develop stable non-uniform relationships?), address drift norms
  (‖W_K s‖ vs ‖e‖ over training), per-machine read patterns on the stream.

## 7. Deferred (explicitly out of v1)

- Cross-context persistence (backend patch + 2-segment truncated BPTT +
  gated/decayed state update). Blocked on v1 showing within-context signal.
- Hard routing / top-k consultation (cost optimization, only if soft wins).
- TTT backend.
