"""Chunk v0.2 (core/chunkv2.py, pattern N): causality through the
recursive log, chunk->chunk references actually happen (and are never
fetched), dedup wiring, fetch on/off, grads, accounting. CPU tests --
the flex 3-way LSE-merge fast path is probed on-GPU (validate_chunkv2)
before runs, same split as v0/v0.1.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model import GPT, GPTConfig  # noqa: E402
from core.chunkv2 import ChunkV2Attention  # noqa: E402
from core.gated_swa import SlidingWindowAttention  # noqa: E402


def tiny(**kw):
    base = dict(block_size=256, vocab_size=128, n_layer=1, n_head=2,
                n_embd=32, pos="rope", norm="rms", mlp="relu2", bias=False,
                qk_norm=True, attn_pattern="N", window=8,
                chunk_btok=64, chunk_k=2, chunk_topk=8, chunk_fetch_n=2)
    base.update(kw)
    return GPTConfig(**base)


def logits_pair(cfg, idx, idx2):
    torch.manual_seed(1)
    m = GPT(cfg).eval()
    with torch.no_grad():
        return m(idx, targets=idx)[0], m(idx2, targets=idx2)[0]


# ------------------------------------------------------------- causality

def test_autoregressive_causality():
    """Perturbing token p never moves logits at positions < p: writer
    loop, recursion, dedup, and fetch are all pure prefix functions."""
    torch.manual_seed(0)
    idx = torch.randint(0, 128, (1, 200))
    for p in (5, 63, 64, 130, 199):
        idx2 = idx.clone()
        idx2[0, p] = (idx2[0, p] + 1) % 128
        a, b = logits_pair(tiny(), idx, idx2)
        assert torch.equal(a[0, :p], b[0, :p]), f"leak at p={p}"
        assert not torch.equal(a[0, p:], b[0, p:])


def test_block_tick_visibility():
    """Perturbing token 0 is invisible at t in [8, 64) (window passed,
    no chunk tick yet -- and no fetch either, fetch is gated on chunk
    visibility) but visible from t=64 on."""
    torch.manual_seed(0)
    idx = torch.randint(0, 128, (1, 200))
    idx2 = idx.clone()
    idx2[0, 0] = (idx2[0, 0] + 1) % 128
    a, b = logits_pair(tiny(), idx, idx2)
    assert torch.equal(a[0, 8:64], b[0, 8:64])
    assert not torch.equal(a[0, 64:128], b[0, 64:128])


def test_swa_degeneration_below_one_block():
    torch.manual_seed(0)
    cfg = tiny()
    attn = ChunkV2Attention(cfg).eval()
    x = torch.randn(2, 48, cfg.n_embd)
    with torch.no_grad():
        assert torch.equal(attn(x), SlidingWindowAttention.forward(attn, x))


# ------------------------------------------------------------- recursion

def test_chunks_reference_chunks():
    """With many boundaries and a small window the writer's top-k picks
    include chunk candidates -- recursion is live, and every such
    reference is flagged not-ok (never fetched)."""
    torch.manual_seed(0)
    cfg = tiny(chunk_btok=16, chunk_k=4, chunk_topk=8)
    m = GPT(cfg).eval()
    idx = torch.randint(0, 128, (1, 250))
    with torch.no_grad():
        m(idx, targets=idx)
    ptr, ok = m.transformer.h[0].attn.last_mint
    assert (~ok).any(), "no chunk->chunk reference was ever selected"
    # raw pointers stay in raw range; the causal bound: boundary b's
    # pointers (raw ones) are < (b+1)*btok
    S = ptr.shape[2]
    b_of = torch.arange(S) // cfg.chunk_k
    bound = ((b_of + 1) * cfg.chunk_btok).view(1, 1, S, 1)
    assert (ptr[ok] < bound.expand_as(ptr)[ok]).all()


def test_grads_reach_writer_dedup_and_mlp():
    torch.manual_seed(0)
    m = GPT(tiny())
    idx = torch.randint(0, 128, (1, 200))
    _, loss = m(idx, targets=idx)
    loss.backward()
    attn = m.transformer.h[0].attn
    for name in ("slot_emb", "wq.weight", "cmlp_up.weight", "gain", "lam"):
        p = attn.get_parameter(name) if "." in name else getattr(attn, name)
        assert p.grad is not None and p.grad.abs().max() > 0, name


# ---------------------------------------------------------- dedup / fetch

def test_dedup_lambda_moves_logits():
    """lam=0 is the exact no-dedup model (penalty identically 0);
    cranking lam changes the read path."""
    torch.manual_seed(1)
    cfg = tiny()
    m = GPT(cfg).eval()
    idx = torch.randint(0, 128, (1, 200))
    with torch.no_grad():
        base = m(idx, targets=idx)[0]
        m.transformer.h[0].attn.lam.fill_(5.0)
        hot = m(idx, targets=idx)[0]
    assert not torch.equal(base[0, 64:], hot[0, 64:])
    assert torch.equal(base[0, :64], hot[0, :64])   # no chunks visible yet


def test_fetch_disabled_runs():
    torch.manual_seed(0)
    idx = torch.randint(0, 128, (1, 200))
    a, _ = logits_pair(tiny(chunk_fetch_n=0), idx, idx)
    b, _ = logits_pair(tiny(chunk_fetch_n=2), idx, idx)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert not torch.equal(a[0, 64:], b[0, 64:])    # fetch branch is live


def test_topk_required():
    with pytest.raises(AssertionError):
        ChunkV2Attention(tiny(chunk_topk=0))


# ------------------------------------------------------------- accounting

def test_flops_ordering_and_fetch_accounting():
    T = 256
    fN = GPT(tiny()).flops_per_token(T)
    fK = GPT(tiny(attn_pattern="K", chunk_topk=8)).flops_per_token(T)
    fS = GPT(tiny(attn_pattern="S")).flops_per_token(T)
    assert fN > fK > fS          # recursion + fetch + dedup cost extra
    # dense-with-mask fetch: compute depends on the log size, not
    # fetch_n -- n only shapes the mask (on/off is the real knob)
    f0 = GPT(tiny(chunk_fetch_n=0)).flops_per_token(T)
    f8 = GPT(tiny(chunk_fetch_n=8)).flops_per_token(T)
    assert f8 == fN > f0


def test_hourglass_slice_widths():
    """N layers built inside SliceBlocks at narrowed widths; backward
    runs clean."""
    cfg = tiny(attn_pattern="NNNN", n_layer=4, n_embd=64,
               hg_frac=0.5, hg_bneck=2, hg_round=8)
    m = GPT(cfg)
    idx = torch.randint(0, 128, (1, 130))
    _, loss = m(idx, targets=idx)
    loss.backward()
