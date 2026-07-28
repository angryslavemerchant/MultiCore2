"""Correctness tests for admission-gated windowed attention.

Run:  python -m pytest tests/test_gated.py -q     (CPU is enough; the
flex-vs-dense equivalence test self-skips without CUDA)
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import gated_swa
from core.gated_swa import (GatedSWAttention, SlidingWindowAttention,
                            gated_death_times, interval_mask, swa_death_times)
from core.model import GPT, GPTConfig

torch.manual_seed(0)


def brute_force_death(gates, capacity):
    """Simulate the FIFO gates literally, one token at a time."""
    T = len(gates)
    death = [T] * T
    fifo = {}
    for t, g in enumerate(gates):
        q = fifo.setdefault(g, [])
        q.append(t)
        if len(q) > capacity:
            death[q.pop(0)] = t     # evicted the moment t is admitted
    return death


@pytest.mark.parametrize("T,G,cap,seed", [(64, 4, 4, 0), (128, 8, 8, 1),
                                          (257, 8, 16, 2), (16, 2, 32, 3)])
def test_death_times_match_simulation(T, G, cap, seed):
    rng = torch.Generator().manual_seed(seed)
    gates = torch.randint(0, G, (3, T), generator=rng)
    death = gated_death_times(gates, cap)
    for b in range(3):
        expect = brute_force_death(gates[b].tolist(), cap)
        assert death[b].tolist() == expect


def test_one_gate_equals_sliding_window():
    """G=1 with capacity W admits everyone to one FIFO of size W — exactly
    the sliding window. The masks must be identical."""
    T, W = 96, 24
    gates = torch.zeros(2, T, dtype=torch.long)
    d_gated = gated_death_times(gates, W)
    d_swa = swa_death_times(T, W, gates.device).expand(2, T)
    assert torch.equal(interval_mask(d_gated, T), interval_mask(d_swa, T))


def test_self_and_recent_always_visible():
    """A token always sees itself, and the most recent `capacity` tokens are
    visible regardless of routing (nothing can evict them yet)."""
    T, G, cap = 200, 8, 8
    gates = torch.randint(0, G, (4, T))
    mask = interval_mask(gated_death_times(gates, cap), T)[:, 0]
    q = torch.arange(T).view(T, 1)
    k = torch.arange(T).view(1, T)
    recent = (k <= q) & (q - k < cap)
    assert bool(mask[:, torch.arange(T), torch.arange(T)].all()), "self"
    assert bool(mask[:, recent].all()), "recent-capacity window"


def test_resident_budget_never_exceeds_window():
    """At any query position, at most W keys are visible (the union of G
    gates of capacity W/G)."""
    T, G, cap = 512, 8, 16          # W = 128
    gates = torch.randint(0, G, (4, T))
    mask = interval_mask(gated_death_times(gates, cap), T)[:, 0]
    assert int(mask.sum(-1).max()) <= G * cap


def test_no_future_leak():
    """Perturbing tokens after position t must not change outputs at <= t,
    even though routing (and thus death times) of the future changes."""
    cfg = GPTConfig(block_size=64, vocab_size=97, n_layer=2, n_head=2,
                    n_embd=32, attn_pattern="FG", window=8, n_gates=2)
    model = GPT(cfg).eval()
    t = 40
    idx = torch.randint(0, 97, (2, 64))
    idx2 = idx.clone()
    idx2[:, t + 1:] = torch.randint(0, 97, (2, 64 - t - 1))
    with torch.no_grad():
        a, _ = model(idx, idx)
        b, _ = model(idx2, idx2)
    assert torch.allclose(a[:, :t + 1], b[:, :t + 1], atol=1e-5)


def test_router_receives_gradient():
    """The forward-identity value scaling must carry gradient into the
    router even though the gate choice is a hard argmax."""
    cfg = GPTConfig(block_size=32, vocab_size=53, n_layer=1, n_head=2,
                    n_embd=32, attn_pattern="G", window=8, n_gates=4)
    model = GPT(cfg)
    idx = torch.randint(0, 53, (2, 32))
    _, loss = model(idx, idx)
    loss.backward()
    router = model.transformer.h[0].attn.router.weight
    assert router.grad is not None and float(router.grad.abs().sum()) > 0


def test_forward_identity_scaling():
    """Router value-scaling is exactly 1.0 in the forward pass: a gated
    layer and an argmax-identical mask must produce identical outputs."""
    cfg = GPTConfig(block_size=48, vocab_size=53, n_layer=1, n_head=2,
                    n_embd=32, attn_pattern="G", window=8, n_gates=4)
    model = GPT(cfg).eval()
    attn = model.transformer.h[0].attn
    x = torch.randn(2, 48, 32)
    with torch.no_grad():
        y = attn(x)
        # recompute with the scaling forcibly removed
        logits = attn.router(x)
        gate = logits.argmax(dim=-1)
        q, k, v = attn._qkv(x)
        death = gated_death_times(gate, attn.capacity)
        ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=interval_mask(death, 48))
        ref = ref.transpose(1, 2).contiguous().view(2, 48, 32)
        ref = attn.c_proj(ref)
    assert torch.allclose(y, ref, atol=1e-6)


def test_pattern_flops_ordering():
    """Windowed patterns must report fewer FLOPs/token than dense, and the
    gated pattern must cost the same as its SWA control."""
    dense = GPT(GPTConfig(n_layer=4, n_head=2, n_embd=64, vocab_size=256))
    swa = GPT(GPTConfig(n_layer=4, n_head=2, n_embd=64, vocab_size=256,
                        attn_pattern="FSSF", window=256))
    gated = GPT(GPTConfig(n_layer=4, n_head=2, n_embd=64, vocab_size=256,
                          attn_pattern="FGGF", window=256, n_gates=8))
    f_dense = dense.flops_per_token(1024)
    f_swa = swa.flops_per_token(1024)
    # router params differ; compare score terms via num_params-adjusted
    assert f_swa < f_dense
    assert abs((gated.flops_per_token(1024) - 6 * gated.num_params())
               - (f_swa - 6 * swa.num_params())) < 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_flex_matches_dense_path():
    """The flex_attention fast path must agree with the dense-mask SDPA
    path (same weights, same input)."""
    cfg = GPTConfig(block_size=256, vocab_size=97, n_layer=2, n_head=4,
                    n_embd=64, attn_pattern="GS", window=64, n_gates=4)
    model = GPT(cfg).cuda().eval()
    idx = torch.randint(0, 97, (2, 256), device="cuda")
    try:
        gated_swa.USE_FLEX = True
        with torch.no_grad():
            a, _ = model(idx, idx)
        gated_swa.USE_FLEX = False
        with torch.no_grad():
            b, _ = model(idx, idx)
    finally:
        gated_swa.USE_FLEX = True
    assert torch.allclose(a, b, atol=2e-3, rtol=2e-3)
