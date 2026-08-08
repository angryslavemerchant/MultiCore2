"""Chunk-latent attention (core/chunk.py): causality, block-tick chunk
visibility, long-range liveness through the log, SWA degeneration, the
blocksum control, params/flops accounting. CPU tests -- the xformers
LSE-merge fast path is probed on-GPU (validate_gated-style) before runs.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model import GPT, GPTConfig  # noqa: E402
from core.chunk import ChunkAttention, BlockSumAttention  # noqa: E402
from core.gated_swa import SlidingWindowAttention  # noqa: E402


def tiny(**kw):
    base = dict(block_size=256, vocab_size=128, n_layer=1, n_head=2,
                n_embd=32, pos="rope", norm="rms", mlp="relu2", bias=False,
                qk_norm=True, attn_pattern="K", window=8,
                chunk_btok=64, chunk_k=4)
    base.update(kw)
    return GPTConfig(**base)


def logits_pair(cfg, idx, idx2):
    torch.manual_seed(1)
    m = GPT(cfg).eval()
    with torch.no_grad():
        return m(idx, targets=idx)[0], m(idx2, targets=idx2)[0]


# ------------------------------------------------------------- causality

def test_autoregressive_causality_topk():
    """v0.1 hard selection is still a pure function of the prefix."""
    torch.manual_seed(0)
    idx = torch.randint(0, 128, (1, 200))
    cfg = tiny(chunk_topk=16, chunk_k=2)
    for p in (5, 64, 130):
        idx2 = idx.clone()
        idx2[0, p] = (idx2[0, p] + 1) % 128
        a, b = logits_pair(cfg, idx, idx2)
        assert torch.equal(a[0, :p], b[0, :p]), f"leak at p={p}"


def test_topk_grads_reach_writer():
    torch.manual_seed(0)
    m = GPT(tiny(chunk_topk=8, chunk_k=2))
    idx = torch.randint(0, 128, (1, 200))
    _, loss = m(idx, targets=idx)
    loss.backward()
    attn = m.transformer.h[0].attn
    for p in (attn.slot_emb, attn.wq.weight):
        assert p.grad is not None and p.grad.abs().max() > 0


def test_topk_exceeding_btok_asserts():
    with pytest.raises(AssertionError):
        ChunkAttention(tiny(chunk_topk=128, chunk_btok=64))


@pytest.mark.parametrize("pattern", ["K", "B"])
def test_autoregressive_causality(pattern):
    """Perturbing token p never moves logits at positions < p (writer,
    log, and read path are all pure functions of the causal prefix)."""
    torch.manual_seed(0)
    idx = torch.randint(0, 128, (1, 200))
    for p in (5, 63, 64, 130, 199):
        idx2 = idx.clone()
        idx2[0, p] = (idx2[0, p] + 1) % 128
        a, b = logits_pair(tiny(attn_pattern=pattern), idx, idx2)
        assert torch.equal(a[0, :p], b[0, :p]), f"leak at p={p}"
        assert not torch.equal(a[0, p:], b[0, p:])


@pytest.mark.parametrize("pattern", ["K", "B"])
def test_block_tick_visibility_and_liveness(pattern):
    """One w=8 layer, btok=64: perturbing token 0 is invisible at t=63
    (outside the window; block-0 chunks not yet readable within block 0)
    but visible at t >= 64 -- long-range information flows ONLY through
    the chunk log, and only from the next block on."""
    torch.manual_seed(0)
    idx = torch.randint(0, 128, (1, 200))
    idx2 = idx.clone()
    idx2[0, 0] = (idx2[0, 0] + 1) % 128
    a, b = logits_pair(tiny(attn_pattern=pattern), idx, idx2)
    assert torch.equal(a[0, 8:64], b[0, 8:64])      # window passed, no tick
    assert not torch.equal(a[0, 64:128], b[0, 64:128])   # via block-0 chunks


def test_swa_degeneration_below_one_block():
    """T <= btok mints nothing: ChunkAttention == its own SWA parent."""
    torch.manual_seed(0)
    cfg = tiny()
    attn = ChunkAttention(cfg).eval()
    x = torch.randn(2, 48, cfg.n_embd)
    with torch.no_grad():
        assert torch.equal(attn(x), SlidingWindowAttention.forward(attn, x))


# ----------------------------------------------------------- writer/grads

def test_writer_gets_gradient():
    """A long-range-dependent loss reaches the writer head params."""
    torch.manual_seed(0)
    cfg = tiny()
    m = GPT(cfg)
    idx = torch.randint(0, 128, (1, 200))
    _, loss = m(idx, targets=idx)
    loss.backward()
    attn = m.transformer.h[0].attn
    for p in (attn.slot_emb, attn.wq.weight, attn.cmlp_up.weight):
        assert p.grad is not None and p.grad.abs().max() > 0


def test_blocksum_has_no_writer_head():
    cfg = tiny(attn_pattern="B")
    attn = BlockSumAttention(cfg)
    assert not hasattr(attn, "slot_emb") and not hasattr(attn, "wq")


def test_blocksum_divisibility_assert():
    with pytest.raises(AssertionError):
        BlockSumAttention(tiny(chunk_btok=64, chunk_k=5))


# ------------------------------------------------------------- accounting

def test_flops_ordering():
    """chunk > blocksum (writer cross-attn) > plain SWA (log reads)."""
    T = 256
    f = {p: GPT(tiny(attn_pattern=p)).flops_per_token(T)
         for p in ("K", "B", "S")}
    assert f["K"] > f["B"] > f["S"]


def test_flops_no_chunks_equals_swa():
    """Below one block the log is empty and accounting must agree."""
    assert (GPT(tiny(attn_pattern="K")).flops_per_token(64)
            == GPT(tiny(attn_pattern="S")).flops_per_token(64)
            + 6 * (GPT(tiny(attn_pattern="K")).num_params()
                   - GPT(tiny(attn_pattern="S")).num_params()))


def test_hourglass_slice_widths():
    """Chunk layers built inside SliceBlocks at narrowed widths."""
    cfg = tiny(attn_pattern="KKKK", n_layer=4, n_embd=64,
               hg_frac=0.5, hg_bneck=2, hg_round=8)
    m = GPT(cfg)
    idx = torch.randint(0, 128, (1, 130))
    _, loss = m(idx, targets=idx)
    loss.backward()   # runs, and every width divides heads cleanly
