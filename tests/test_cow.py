"""Correctness tests for the copy-on-write codebook archive (core/cow.py).

Run:  python -m pytest tests/test_cow.py -q     (CPU is enough; the
flex-vs-dense equivalence test self-skips without CUDA)
"""
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cow import (COWAttention, chain_running_mean, cow_chain_scan,
                      cow_death_times, cow_interval_mask)

torch.manual_seed(0)


def ref_scan(keys, gate, G, K, theta):
    """Exact per-token simulation. Eviction convention matches the chunked
    scan at chunk=1: the evicted chain's last version dies at t+1 (the
    boundary after the admitting token)."""
    B, T, C = keys.shape
    kn = F.normalize(keys.float(), dim=-1)
    chain_id = torch.empty(B, T, dtype=torch.long)
    evict_pos = torch.full((B, T), T, dtype=torch.long)
    for b in range(B):
        chains = {g: [] for g in range(G)}   # list of [head_vec, pos, bid]
        for t in range(T):
            g = int(gate[b, t])
            k = kn[b, t]
            best, best_c = -2.0, None
            for c in chains[g]:
                s = float(k @ c[0])
                if s > best:
                    best, best_c = s, c
            if best_c is not None and best >= theta:
                best_c[0], best_c[1] = k, t
                chain_id[b, t] = best_c[2]
            else:
                chains[g].append([k, t, t])
                chain_id[b, t] = t
                if len(chains[g]) > K:
                    si = min(range(len(chains[g])),
                             key=lambda i: chains[g][i][1])
                    stale = chains[g].pop(si)
                    evict_pos[b, stale[2]] = t + 1
    return chain_id, evict_pos


@pytest.mark.parametrize("T,G,K,theta,seed", [
    (64, 2, 4, 0.3, 0), (128, 4, 4, 0.5, 1), (96, 1, 2, 0.0, 2),
    (200, 8, 8, 0.7, 3)])
def test_chunk1_matches_exact_simulation(T, G, K, theta, seed):
    """chunk=1 removes every approximation: heads move each token,
    evictions land at each boundary. Must equal the reference exactly."""
    rng = torch.Generator().manual_seed(seed)
    # low-dim keys -> plenty of similarity structure at these thetas
    keys = torch.randn(2, T, 8, generator=rng)
    gate = torch.randint(0, G, (2, T), generator=rng)
    cid, ev, _ = cow_chain_scan(keys, gate, G, K, theta, chunk=1)
    rcid, rev = ref_scan(keys, gate, G, K, theta)
    assert torch.equal(cid, rcid)
    assert torch.equal(ev, rev)


def test_chunked_scan_is_close_to_exact():
    """chunk>1 is an approximation. Chain IDS diverge fast (a single early
    difference renames every later member), so judge what matters:
    co-membership structure (do two tokens share a chain in both scans?)
    and birth-rate, which must stay on the birth-liberal side of exact/2
    (under-birthing is the unrecoverable failure mode)."""
    rng = torch.Generator().manual_seed(4)
    keys = torch.randn(2, 512, 8, generator=rng)
    gate = torch.randint(0, 4, (2, 512), generator=rng)
    cid, _, st = cow_chain_scan(keys, gate, 4, 8, 0.5, chunk=32)
    rcid, _ = ref_scan(keys, gate, 4, 8, 0.5)
    idx = torch.randint(0, 512, (2, 4000, 2), generator=rng)
    a = cid.gather(1, idx[..., 0]) == cid.gather(1, idx[..., 1])
    r = rcid.gather(1, idx[..., 0]) == rcid.gather(1, idx[..., 1])
    co = (a == r).float().mean()
    assert co > 0.95, f"co-membership agreement only {co:.1%}"
    exact_birth = (rcid == torch.arange(512).expand(2, 512)).float().mean()
    assert st["birth_rate"] > 0.5 * float(exact_birth), (
        st["birth_rate"], float(exact_birth))


def test_death_times_semantics():
    """A version dies at the next same-chain token, its chain's eviction,
    or T -- and never at or before its own position."""
    rng = torch.Generator().manual_seed(5)
    keys = torch.randn(2, 256, 8, generator=rng)
    gate = torch.randint(0, 4, (2, 256), generator=rng)
    cid, ev, _ = cow_chain_scan(keys, gate, 4, 4, 0.5, chunk=32)
    death = cow_death_times(cid, ev)
    T = 256
    pos = torch.arange(T)
    assert (death > pos.unsqueeze(0)).all()
    for b in range(2):
        for t in range(T):
            nxt = [u for u in range(t + 1, T) if cid[b, u] == cid[b, t]]
            expect = min(nxt[0] if nxt else T, int(ev[b, cid[b, t]]))
            assert int(death[b, t]) == expect, (b, t)


def test_chain_running_mean_matches_loop():
    rng = torch.Generator().manual_seed(6)
    v = torch.randn(2, 3, 64, 5, generator=rng, requires_grad=True)
    cid = torch.randint(0, 7, (2, 64), generator=rng)
    out = chain_running_mean(v, cid)
    for b in range(2):
        for t in range(64):
            members = [u for u in range(t + 1) if cid[b, u] == cid[b, t]]
            expect = v[b, :, members].mean(dim=1)
            assert torch.allclose(out[b, :, t], expect, atol=1e-5), (b, t)
    out.sum().backward()          # differentiability
    assert v.grad is not None and torch.isfinite(v.grad).all()


def _cfg(**kw):
    from core.model import GPTConfig
    base = dict(block_size=256, vocab_size=128, n_layer=2, n_head=4,
                n_embd=32, window=48, n_gates=4, recent_band=16,
                cow_chains=8, cow_theta=0.6, cow_chunk=32, pos="rope",
                dropout=0.0)
    base.update(kw)
    return GPTConfig(**base)


def test_cow_attention_causal():
    """Changing the suffix must not change the prefix's outputs -- the
    strongest single check that scan + mask + accumulation are causal."""
    torch.manual_seed(7)
    attn = COWAttention(_cfg())
    attn.eval()
    x = torch.randn(2, 256, 32)
    y1 = attn(x)
    x2 = x.clone()
    x2[:, 200:] = torch.randn(2, 56, 32)
    y2 = attn(x2)
    assert torch.allclose(y1[:, :200], y2[:, :200], atol=1e-5)


def test_cow_attention_grad_flows():
    """Loss must reach the router (via p/p.detach()) and c_attn."""
    torch.manual_seed(8)
    attn = COWAttention(_cfg())
    x = torch.randn(2, 256, 32, requires_grad=True)
    attn(x).sum().backward()
    assert attn.router.weight.grad is not None
    assert attn.router.weight.grad.abs().sum() > 0
    assert attn.c_attn.weight.grad.abs().sum() > 0
    assert torch.isfinite(x.grad).all()


def test_archive_reaches_beyond_band():
    """A query attends to content older than the band: with theta > 1 every
    token births its own chain (no merges), so versions live until evicted
    and the mask must expose positions older than recent_band."""
    T, band = 256, 16
    death = torch.full((1, T), T, dtype=torch.long)
    mask = cow_interval_mask(death, band, T)
    q = 200
    # raw copy of key 100 is dead (200 - 100 >= band)...
    assert not mask[0, 0, q, 100]
    # ...but its archive version is visible
    assert mask[0, 0, q, T + 100]


def test_band_handoff_no_gap_no_overlap():
    """Raw copy is band-visible unconditionally (like hybrid's recent
    band); archive copy is visible on [t+band, death). The two never
    overlap, and a never-dying chain's coverage is gapless [t, T)."""
    T, band = 128, 8
    rng = torch.Generator().manual_seed(9)
    death = torch.randint(1, T + 1, (1, T), generator=rng)
    death = torch.maximum(death, torch.arange(T).unsqueeze(0) + 1)
    mask = cow_interval_mask(death, band, T)
    q = torch.arange(T).view(T, 1)
    t = torch.arange(T).view(1, T)
    assert not (mask[0, 0, :, :T] & mask[0, 0, :, T:]).any()  # no overlap
    assert torch.equal(mask[0, 0, :, :T],
                       (t <= q) & (q < t + band))             # raw = band
    assert torch.equal(mask[0, 0, :, T:],
                       (t + band <= q)
                       & (q < death.view(1, T).expand(T, T)))  # archive
    immortal = cow_interval_mask(torch.full((1, T), T, dtype=torch.long),
                                 band, T)
    covered = immortal[0, 0, :, :T] | immortal[0, 0, :, T:]
    assert torch.equal(covered, t <= q)                        # gapless


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_flex_matches_dense_sdpa():
    """flex BlockMask path vs dense-mask SDPA path on the same inputs."""
    import core.gated_swa as gsw
    torch.manual_seed(10)
    attn = COWAttention(_cfg()).cuda().eval()
    x = torch.randn(2, 256, 32, device="cuda")
    gsw.USE_FLEX = True
    y_flex = attn(x)
    gsw.USE_FLEX = False
    y_dense = attn(x)
    gsw.USE_FLEX = True
    diff = (y_flex - y_dense).abs().max()
    assert diff < 1e-3, f"flex vs dense max diff {diff}"
