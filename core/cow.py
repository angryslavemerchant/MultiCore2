"""Copy-on-write codebook attention (the COW archive).

THE FORMULATION. Same silhouette as the hybrid arm (recent raw band +
archive beyond the window), but the archive holds copy-on-write CHAINS
instead of FIFO survivors:

  admission   a learned router assigns each token to one of G topic gates
              (stage 1 -- fixed keys, no state feedback, fully parallel).
  chains      inside a gate, the token compares against the gate's live
              chain-head keys (cosine, pre-rope content keys). Best sim
              >= theta -> MERGE: the chain's previous version dies, a new
              version is born at this token's position carrying the
              chain's running-mean value; the token's key becomes the
              chain head. Below theta -> BIRTH a new chain. A gate holds
              at most K chains; on overflow the stalest (oldest head) is
              evicted.
  read        every version is one KV entry visible to queries q in
              [t + band, death(t)) -- the raw token itself serves queries
              inside the band, and the chain version takes over exactly
              when the raw token leaves it. death(t) = position of the
              next token merging into the same chain, else eviction
              position, else T. One contiguous interval per entry, so the
              whole thing rides the existing interval-mask machinery.

Positions are never faked: a version born at t is RoPE-rotated at t, so a
chain's positional distance measures STALENESS of the object (last touch),
not its birth time.

Causality: a version born at t is a function of tokens <= t only, and is
visible only to queries >= t + band. The scan decides membership from
tokens <= t. Teacher-forcing-parallel by construction.

Parallelism: chain-membership decisions are genuinely sequential (whether
token t merges into chain c depends on c's current head). The scan
quarantines that into a chunked no-grad bookkeeping pass: heads are FROZEN
for `chunk` tokens, all comparisons inside a chunk run as one batched
matmul (against frozen heads AND earlier same-gate tokens in the chunk,
resolved by pointer-doubling), then one reconciliation step advances the
state. T=4096 / chunk=128 -> 32 sequential steps. The heavy math (value
accumulation, attention) is fully parallel and differentiable.

Write-rule choices follow the NeocoreEpisodic failure log (see
codebook_llm_integration.md): birth-liberal theta, NO chain-to-chain
merging ever, uncapped running mean for values, soft attention read over
all live versions.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

from core.gated_swa import SlidingWindowAttention, gated_death_times
import core.gated_swa as _gsw

try:
    from torch.nn.attention.flex_attention import (
        flex_attention, create_block_mask)
    _HAVE_FLEX = True
except ImportError:
    _HAVE_FLEX = False


# --------------------------------------------------------------- chain scan
@torch.no_grad()
def _scan_chunk(kc, gc, pos_c, c1_t, head_key, head_pos, birth_id,
                gate_arange, G, K, T, theta, n_jumps):
    """One chunk of the scan: compare, resolve roots, reconcile state.

    Branchless and static-shaped BY DESIGN: this function is
    torch.compile'd (fullgraph) on CUDA so the 32 sequential chunk steps
    cost a few fused kernels each instead of ~35 eager dispatches -- the
    eager version measured 3.3 s/iter on 8x5090 (5x the hybrid arm), all
    launch latency. c1_t (chunk end) must stay a TENSOR: as a python int
    it would burn one recompile per chunk.

    Returns (cid, n_birth_row, evict_upd, n_evict_row, head_key,
    head_pos, birth_id).
    """
    B, m, C = kc.shape
    GK = G * K

    # --- similarities to frozen heads (own gate, alive only) -------------
    sim_h = torch.einsum("bmc,bgkc->bmgk", kc, head_key)   # (B,m,G,K)
    own = F.one_hot(gc, G).bool().unsqueeze(-1)            # (B,m,G,1)
    alive = (head_pos >= 0).unsqueeze(1)                   # (B,1,G,K)
    sim_h = torch.where(own & alive, sim_h, torch.full_like(sim_h, -2.0))
    sim_h = sim_h.reshape(B, m, GK)

    # --- similarities to earlier same-gate tokens in this chunk ----------
    # Eligible in-chunk candidates are tokens that did NOT merge into a
    # frozen head: members of FRESH chains only. A token similar to an
    # existing chain's within-chunk member (but not its frozen head)
    # births instead of merging -- the error lands on the birth side, the
    # recoverable side of the Neocore asymmetry. Without this the
    # in-chunk superset ("match ANY earlier member") halves the birth
    # rate at chunk=128 (measured on random keys).
    head_merged = sim_h.max(dim=-1).values >= theta         # (B,m)
    sim_t = torch.einsum("bmc,bnc->bmn", kc, kc)           # (B,m,m)
    tri = torch.ones(m, m, dtype=torch.bool, device=kc.device).tril(-1)
    same_gate = gc.unsqueeze(-1) == gc.unsqueeze(1)
    elig = tri & same_gate & ~head_merged.unsqueeze(1)
    sim_t = torch.where(elig, sim_t, torch.full_like(sim_t, -2.0))

    # --- best candidate over both populations ----------------------------
    sim_all = torch.cat([sim_h, sim_t], dim=-1)            # (B,m,GK+m)
    best_sim, best_idx = sim_all.max(dim=-1)
    merge = best_sim >= theta                               # (B, m)
    to_head = merge & (best_idx < GK)
    to_tok = merge & (best_idx >= GK)

    # --- resolve chain ids: pointer-doubling to each token's root --------
    self_idx = torch.arange(m, device=kc.device).expand(B, m)
    parent = torch.where(to_tok, best_idx - GK, self_idx)
    for _ in range(n_jumps):
        parent = parent.gather(1, parent)                   # doubling
    root = parent                                           # (B,m)
    root_to_head = to_head.gather(1, root)
    root_head_idx = best_idx.gather(1, root).clamp(max=GK - 1)
    head_chain = birth_id.reshape(B, GK).gather(1, root_head_idx)
    root_pos = pos_c.gather(1, root)
    cid = torch.where(root_to_head, head_chain, root_pos)
    is_birth = (~to_head) & (~to_tok)                       # own root

    # --- reconcile state at the boundary ---------------------------------
    # 1) existing chains that got merges: newest member this chunk.
    flat_bid = birth_id.reshape(B, GK)                      # (B,GK)
    match = (cid.unsqueeze(-1) == flat_bid.unsqueeze(1)) \
        & (flat_bid.unsqueeze(1) >= 0)                      # (B,m,GK)
    newest = torch.where(match, pos_c.unsqueeze(-1),
                         torch.full_like(match, -1, dtype=torch.long))
    upd_pos, upd_tok = newest.max(dim=1)                    # (B,GK)
    has_upd = upd_pos >= 0
    upd_key = kc.gather(1, upd_tok.clamp(min=0)
                        .unsqueeze(-1).expand(-1, -1, C))   # (B,GK,C)
    head_key = torch.where(has_upd.reshape(B, G, K, 1),
                           upd_key.reshape(B, G, K, C), head_key)
    head_pos = torch.where(has_upd.reshape(B, G, K),
                           upd_pos.reshape(B, G, K), head_pos)

    # 2) fresh chains born this chunk: newest member of each.
    same_new = cid.unsqueeze(1) == pos_c.unsqueeze(-1)      # (B,m,m)
    memb_pos = torch.where(same_new, pos_c.unsqueeze(1),
                           torch.full_like(same_new, -1, dtype=torch.long))
    new_last, new_last_tok = memb_pos.max(dim=2)            # (B,m)
    new_key = kc.gather(1, new_last_tok.clamp(min=0)
                        .unsqueeze(-1).expand(-1, -1, C))

    # 3) per gate: pool existing + births, keep the newest K.
    neg1 = torch.full_like(pos_c, -1)
    pool_pos = torch.cat([head_pos.reshape(B, GK),
                          torch.where(is_birth, new_last, neg1)], 1)
    pool_gate = torch.cat([gate_arange.expand(B, -1), gc], 1)
    pool_bid = torch.cat([flat_bid,
                          torch.where(is_birth, pos_c, neg1)], 1)
    pool_key = torch.cat([head_key.reshape(B, GK, C), new_key], 1)

    dead = pool_pos < 0
    span = 2 * T + 2
    rank_key = pool_gate * span + (T - pool_pos)            # small=newer
    rank_key = torch.where(dead, torch.full_like(rank_key, G * span),
                           rank_key)
    order = rank_key.argsort(dim=1)                          # (B,GK+m)
    srt_gate = pool_gate.gather(1, order)
    src_pos = pool_pos.gather(1, order)
    src_bid = pool_bid.gather(1, order)
    src_key = pool_key.gather(1, order.unsqueeze(-1).expand(-1, -1, C))
    # index within each gate's sorted run: 0,1,2.. resetting per gate.
    run = torch.zeros_like(srt_gate)
    run[:, 1:] = (srt_gate[:, 1:] == srt_gate[:, :-1]).long()
    csum = run.cumsum(1)
    start_val = torch.where(run == 0, csum, torch.zeros_like(csum))
    idx_in_gate = csum - start_val.cummax(1).values
    keep_sorted = (idx_in_gate < K) & (src_pos >= 0)

    # evictions, branchless: alive-but-not-kept entries write their chunk
    # end into a T+1-wide amin buffer (dummy column T absorbs the rest).
    evicted = (src_pos >= 0) & ~keep_sorted
    ev_idx = torch.where(evicted, src_bid, torch.full_like(src_bid, T))
    ev_val = torch.where(evicted, c1_t.expand_as(src_bid),
                         torch.full_like(src_bid, T))
    evict_upd = torch.full((B, T + 1), T, dtype=torch.long,
                           device=kc.device)
    evict_upd.scatter_reduce_(1, ev_idx, ev_val, reduce="amin")

    # 4) rebuild packed state: kept entries scatter to their slot, all
    # others collide harmlessly on dummy slot GK.
    dst = torch.where(keep_sorted, srt_gate * K + idx_in_gate.clamp(max=K - 1),
                      torch.full_like(idx_in_gate, GK))
    new_hp = torch.full((B, GK + 1), -1, dtype=torch.long, device=kc.device)
    new_bi = torch.full((B, GK + 1), -1, dtype=torch.long, device=kc.device)
    new_hk = torch.zeros(B, GK + 1, C, device=kc.device)
    new_hp.scatter_(1, dst, torch.where(keep_sorted, src_pos, neg1[:, :1]
                                        .expand_as(src_pos)))
    new_bi.scatter_(1, dst, torch.where(keep_sorted, src_bid, neg1[:, :1]
                                        .expand_as(src_bid)))
    new_hk.scatter_(1, dst.unsqueeze(-1).expand(-1, -1, C),
                    torch.where(keep_sorted.unsqueeze(-1), src_key,
                                torch.zeros_like(src_key)))

    return (cid, is_birth.float().sum(1), evict_upd[:, :T],
            evicted.float().sum(1), new_hk[:, :GK].reshape(B, G, K, C),
            new_hp[:, :GK].reshape(B, G, K),
            new_bi[:, :GK].reshape(B, G, K))


COMPILE_SCAN = True          # module-level override for tests / debugging
_scan_chunk_compiled = None


def _get_scan_step(is_cuda):
    global _scan_chunk_compiled
    if not (COMPILE_SCAN and is_cuda):
        return _scan_chunk
    if _scan_chunk_compiled is None:
        _scan_chunk_compiled = torch.compile(_scan_chunk, fullgraph=True,
                                             dynamic=False)
    return _scan_chunk_compiled


@torch.no_grad()
def cow_chain_scan(keys, gate, n_gates, n_chains, theta, chunk=128):
    """Chunked copy-on-write chain-membership scan.

    keys  (B, T, C) float: PRE-rope content keys (L2-normalized inside).
    gate  (B, T) long: stage-1 gate id per token, in [0, n_gates).
    n_chains: K, max live chains per gate.
    theta: vigilance. best cosine >= theta -> merge, else birth.
    chunk: scan granularity; heads are frozen within a chunk.

    Returns (chain_id, evict_pos, stats):
      chain_id  (B, T) long -- chain each token joined; ids are the BIRTH
                POSITION of the chain, globally unique per row, never
                reused.
      evict_pos (B, T) long -- indexed by birth position: position at
                which that chain was evicted (T = never).
      stats     dict of floats.

    Approximations vs exact per-token semantics (both deliberate):
      * heads move only at chunk boundaries -- but a token also compares
        against ALL earlier same-gate tokens inside its own chunk (which
        is exactly where the heads would have moved to), so staleness
        only bites on cross-chunk boundaries;
      * eviction applies at chunk boundaries -- a gate may transiently
        hold more than K chains inside a chunk.

    The driver loop is dynamo-disabled with recursive=False: the outer
    model compile skips this frame (python loop, data-dependent state),
    while the inner per-chunk step still hits its own torch.compile.
    recursive=True would silently downgrade the step to eager.
    """
    B, T, C = keys.shape
    device = keys.device
    G, K = n_gates, n_chains
    kn = F.normalize(keys.float(), dim=-1)

    head_key = torch.zeros(B, G, K, C, device=device)
    head_pos = torch.full((B, G, K), -1, dtype=torch.long, device=device)
    birth_id = torch.full((B, G, K), -1, dtype=torch.long, device=device)

    chain_id = torch.empty(B, T, dtype=torch.long, device=device)
    evict_pos = torch.full((B, T), T, dtype=torch.long, device=device)
    n_births = torch.zeros(B, device=device)
    n_evicts = torch.zeros(B, device=device)
    gate_arange = torch.arange(G, device=device).repeat_interleave(K)
    pos_all = torch.arange(T, device=device).expand(B, T)
    n_jumps = max(1, (chunk - 1).bit_length())
    step = _get_scan_step(keys.is_cuda)

    for c0 in range(0, T, chunk):
        c1 = min(c0 + chunk, T)
        c1_t = torch.full((1,), c1, dtype=torch.long, device=device)
        (cid, nb, evict_upd, ne, head_key, head_pos, birth_id) = step(
            kn[:, c0:c1], gate[:, c0:c1], pos_all[:, c0:c1], c1_t,
            head_key, head_pos, birth_id, gate_arange, G, K, T, theta,
            n_jumps)
        chain_id[:, c0:c1] = cid
        evict_pos = torch.minimum(evict_pos, evict_upd)
        n_births += nb
        n_evicts += ne

    live = (head_pos >= 0).float().sum((1, 2))
    stats = {
        "birth_rate": float((n_births / T).mean()),
        "merge_rate": float(1.0 - (n_births / T).mean()),
        "n_chains": float(n_births.mean()),
        "n_evicted": float(n_evicts.mean()),
        "live_chains_end": float(live.mean()),
        "mean_chain_len": float(T / max(n_births.mean().item(), 1.0)),
    }
    return chain_id, evict_pos, stats


cow_chain_scan = torch.compiler.disable(cow_chain_scan, recursive=False)


def cow_death_times(chain_id, evict_pos):
    """Version death: next same-chain token, else chain eviction, else T.

    gated_death_times with capacity=1 IS 'position of the next same-chain
    token' -- the chain id plays the role of the gate id.
    """
    death = gated_death_times(chain_id, 1)
    ev = evict_pos.gather(1, chain_id)          # eviction of MY chain
    return torch.minimum(death, ev)


# ------------------------------------------------- value accumulation (grad)
def chain_running_mean(v, chain_id):
    """Exact running mean of v along each chain, differentiable.

    v (B, H, T, hd), chain_id (B, T) -> (B, H, T, hd) where output[t] is
    the mean of v over this chain's members at positions <= t.
    """
    B, H, T, hd = v.shape
    idx = torch.arange(T, device=v.device)
    order = (chain_id * T + idx).argsort(dim=-1)                 # (B,T)
    inv = torch.empty_like(order)
    inv.scatter_(1, order, idx.expand(B, T))
    cid_s = chain_id.gather(1, order)
    start = torch.ones(B, T, dtype=torch.bool, device=v.device)
    start[:, 1:] = cid_s[:, 1:] != cid_s[:, :-1]
    seg0 = torch.where(start, idx.expand(B, T),
                       torch.zeros_like(order)).cummax(1).values
    rank = (idx.expand(B, T) - seg0 + 1).to(v.dtype)             # 1,2,3..

    ov = order.view(B, 1, T, 1).expand(-1, H, -1, hd)
    v_s = v.gather(2, ov)
    cs = v_s.cumsum(dim=2)
    s0 = seg0.view(B, 1, T, 1).expand(-1, H, -1, hd)
    seg_sum = cs - cs.gather(2, s0) + v_s.gather(2, s0)          # incl. start
    mean_s = seg_sum / rank.view(B, 1, T, 1)
    return mean_s.gather(2, inv.view(B, 1, T, 1).expand(-1, H, -1, hd))


# ------------------------------------------------------------------ masking
def cow_interval_mask(death, band, T):
    """(B,1,T,2T) bool. Keys 0..T-1: raw band, k <= q < k+band.
    Keys T..2T-1: archive version, k+band <= q < death[b,k]."""
    q = torch.arange(T, device=death.device).view(1, T, 1)
    k = torch.arange(T, device=death.device).view(1, 1, T)
    raw = (k <= q) & (q < k + band)
    arch = (k + band <= q) & (q < death.view(-1, 1, T))
    return torch.cat([raw.expand(death.shape[0], -1, -1), arch],
                     dim=-1).unsqueeze(1)


def _build_cow_block_mask(death, band, B, T, device):
    """BlockMask over 2T keys. Built EAGERLY (compiler.disable) -- death is
    data-dependent; see core.gated_swa._build_block_mask's warning.

    _compile=True is load-bearing for MEMORY, not speed: the eager
    create_block_mask materializes the dense (B,1,T,2T) index grids in
    int64 -- ~4 GiB at B=16/T=4096 -- which OOMed rank 4 on the first COW
    launch (2026-07-30, 27.9/31.4 GiB in use). The compiled builder
    evaluates mask_mod blockwise. This is an explicit torch.compile of the
    BUILDER only, still outside the model's graph -- a different path from
    the data-dependent-mask-inside-compiled-graph corruption the eager
    rule guards against; compiled-vs-eager was re-verified on the box
    after this change."""
    def mask_mod(b, h, qi, ki):
        raw = (ki <= qi) & (qi < ki + band)
        kt = (ki - T).clamp(min=0)
        arch = (kt + band <= qi) & (qi < death[b, kt])
        return torch.where(ki < T, raw, arch)

    return create_block_mask(mask_mod, B, 1, T, 2 * T, device=str(device),
                             _compile=True)


if _HAVE_FLEX:
    # recursive=False: the outer model compile must skip this frame (the
    # data-dependent-mask gotcha), but create_block_mask's own
    # _compile=True path must still engage -- recursive=True would
    # silently run the memory-hungry eager builder.
    _build_cow_block_mask = torch.compiler.disable(_build_cow_block_mask,
                                                   recursive=False)


# ---------------------------------------------------------------- attention
class COWAttention(SlidingWindowAttention):
    """Recent raw band + copy-on-write chain archive. See module docstring.

    Budget contract: recent_band raw keys + n_gates*cow_chains live
    versions == cfg.window visible keys per query, the same currency as
    the SWA/hybrid arms.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        assert cfg.recent_band > 0, "COW needs a raw recent band"
        assert cfg.n_gates * cfg.cow_chains == cfg.window - cfg.recent_band, (
            f"budget: {cfg.n_gates}*{cfg.cow_chains} != "
            f"{cfg.window}-{cfg.recent_band}")
        self.recent_band = cfg.recent_band
        self.n_gates = cfg.n_gates
        self.n_chains = cfg.cow_chains
        self.theta = cfg.cow_theta
        self.chunk = cfg.cow_chunk
        self.router = nn.Linear(cfg.n_embd, cfg.n_gates, bias=False)
        self.collect_stats = False
        self.stats = {}
        self.lb_loss = None

    def forward(self, x):
        B, T, C = x.shape
        logits = self.router(x)
        probs = F.softmax(logits.float(), dim=-1)
        gate = probs.argmax(dim=-1)
        p_sel = probs.gather(-1, gate.unsqueeze(-1))
        frac = F.one_hot(gate, self.n_gates).float().mean(dim=(0, 1))
        self.lb_loss = self.n_gates * (frac.detach()
                                       * probs.mean(dim=(0, 1))).sum()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        k_cont = k.detach()                       # pre-rope content keys
        shp = (B, T, self.n_head, C // self.n_head)
        q = q.view(shp).transpose(1, 2)
        k = k.view(shp).transpose(1, 2)
        v = v.view(shp).transpose(1, 2)
        if self.use_rope:
            from core.rope import apply_rope
            q, k = apply_rope(q, k)

        # forward-identity router-gradient multiplier (see GatedSWAttention)
        scale = (p_sel / p_sel.detach()).to(v.dtype)
        v = v * scale.unsqueeze(1)

        chain_id, evict_pos, sstats = cow_chain_scan(
            k_cont, gate, self.n_gates, self.n_chains, self.theta,
            self.chunk)
        death = cow_death_times(chain_id, evict_pos)
        # cumsum autocasts to fp32; the extra precision is welcome for long
        # chains, but the attention kernel needs one dtype.
        v_chain = chain_running_mean(v, chain_id).to(v.dtype)

        k2 = torch.cat([k, k], dim=2)             # archive key = newest raw
        v2 = torch.cat([v, v_chain], dim=2)
        y = self._attend_cow(q, k2, v2, death)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        if self.collect_stats:
            with torch.no_grad():
                died = death < T
                life = (death - torch.arange(T, device=x.device))[died]
                self.stats = dict(
                    sstats,
                    gate_frac=frac.detach().cpu(),
                    mean_version_lifetime=float(life.float().mean())
                    if died.any() else float("nan"),
                )
        return self.resid_dropout(self.c_proj(y))

    def _attend_cow(self, q, k2, v2, death):
        B, H, T, _ = q.shape
        if _gsw.USE_FLEX and _HAVE_FLEX and q.is_cuda:
            bm = _build_cow_block_mask(death, self.recent_band, B, T,
                                       q.device)
            return flex_attention(q, k2, v2, block_mask=bm)
        mask = cow_interval_mask(death, self.recent_band, T)
        return F.scaled_dot_product_attention(
            q, k2, v2, attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0)
