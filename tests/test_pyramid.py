"""Pyramid window schedule (cfg.windows) + full canon (B on qkv, D on MLP
hidden). CPU tests — the flex path is exercised on-GPU by validate_gated."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model import GPT, GPTConfig  # noqa: E402


def tiny(**kw):
    base = dict(block_size=64, vocab_size=128, n_layer=2, n_head=2,
                n_embd=32, pos="rope", norm="rms", mlp="relu2", bias=False)
    base.update(kw)
    return GPTConfig(**base)


# ---------------------------------------------------------------- windows

def test_layer_windows_uniform_default():
    cfg = tiny(attn_pattern="SF", window=8)
    assert cfg.layer_windows() == [8, None]


def test_layer_windows_schedule():
    cfg = tiny(attn_pattern="SFS", n_layer=3, windows="4,F,16")
    assert cfg.layer_windows() == [4, None, 16]


def test_layer_windows_wrong_count():
    cfg = tiny(attn_pattern="SF", windows="4,8,16")
    with pytest.raises(AssertionError):
        cfg.layer_windows()


def test_layer_windows_bad_entry():
    cfg = tiny(attn_pattern="SS", windows="4,0")
    with pytest.raises(AssertionError):
        cfg.layer_windows()


def test_window_schedule_receptive_field():
    """Two stacked w=4 SWA layers: token 0 reaches at most position 6
    (3 + 3); logits beyond are bit-identical under a token-0 swap. A
    64-window second layer must break that bound."""
    torch.manual_seed(0)
    idx = torch.randint(0, 128, (1, 64))
    idx2 = idx.clone()
    idx2[0, 0] = (idx2[0, 0] + 1) % 128

    def logits(cfg):
        torch.manual_seed(1)
        m = GPT(cfg).eval()
        with torch.no_grad():
            # targets => full (B,T,V) logits (targets=None keeps last only)
            return m(idx, targets=idx)[0], m(idx2, targets=idx2)[0]

    a, b = logits(tiny(attn_pattern="SS", windows="4,4"))
    assert not torch.allclose(a[:, 0], b[:, 0])          # sees the swap
    assert torch.equal(a[:, 8:], b[:, 8:])               # out of reach
    a, b = logits(tiny(attn_pattern="SS", windows="4,64"))
    assert not torch.allclose(a[:, 8:], b[:, 8:])        # wide layer 2


def test_per_layer_window_flops():
    small = GPT(tiny(attn_pattern="SS", windows="4,8"))
    big = GPT(tiny(attn_pattern="SS", windows="8,8"))
    assert small.flops_per_token(64) < big.flops_per_token(64)


# ---------------------------------------------------------------- canon B/D

def cfg_full_canon(**kw):
    return tiny(attn_pattern="SF", canon=True, canon_full=True, **kw)


def test_canon_full_identity_at_init():
    """Zero-init B/D convs: loading a no-canon-full state dict into a
    canon_full model reproduces its outputs exactly."""
    torch.manual_seed(0)
    base = GPT(tiny(attn_pattern="SF", canon=True)).eval()
    full = GPT(cfg_full_canon()).eval()
    missing, unexpected = full.load_state_dict(base.state_dict(),
                                              strict=False)
    assert not unexpected
    assert all("canon_b" in m or "canon_d" in m for m in missing)
    idx = torch.randint(0, 128, (1, 32))
    with torch.no_grad():
        assert torch.equal(base(idx, targets=idx)[0], full(idx, targets=idx)[0])


def test_canon_full_causal():
    """Perturbing token j never changes logits before j."""
    torch.manual_seed(0)
    m = GPT(cfg_full_canon()).eval()
    # nudge the convs off identity so B/D actually mix
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "canon_b" in n or "canon_d" in n:
                p.normal_(0, 0.02)
    idx = torch.randint(0, 128, (1, 32))
    idx2 = idx.clone()
    j = 16
    idx2[0, j] = (idx2[0, j] + 1) % 128
    with torch.no_grad():
        a, b = m(idx, targets=idx)[0], m(idx2, targets=idx2)[0]
    assert torch.equal(a[:, :j], b[:, :j])
    assert not torch.allclose(a[:, j], b[:, j])


def test_canon_full_grads_flow():
    torch.manual_seed(0)
    m = GPT(cfg_full_canon())
    idx = torch.randint(0, 128, (2, 32))
    _, loss = m(idx, targets=idx)
    loss.backward()
    seen = 0
    for n, p in m.named_parameters():
        if "canon_b" in n or "canon_d" in n:
            assert p.grad is not None and torch.isfinite(p.grad).all(), n
            seen += 1
    assert seen == 4           # (canon_b + canon_d) x 2 layers


def test_canon_full_param_count():
    """B adds k*3d per attention, D adds k*4d per MLP (k=4)."""
    base = GPT(tiny(attn_pattern="SF", canon=True))
    full = GPT(cfg_full_canon())
    d = 32
    per_layer = 4 * 3 * d + 4 * 4 * d
    assert (sum(p.numel() for p in full.parameters())
            - sum(p.numel() for p in base.parameters())) == 2 * per_layer
