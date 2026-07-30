"""flash-attn SWA path: semantics (window off-by-one) + agreement with the
flex/SDPA path. GPU + flash-attn only — skipped everywhere else; run on a
box as part of the gate:  python -m pytest tests/test_flash.py -q
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import gated_swa  # noqa: E402
from core.model import GPTConfig  # noqa: E402

needs_flash = pytest.mark.skipif(
    not (torch.cuda.is_available() and gated_swa._resolve_flash()),
    reason="needs CUDA + a working banded flash kernel")


@needs_flash
def test_flash_window_semantics_vs_reference():
    """flash window_size=(W-1, 0) must equal death(k)=k+W: q sees keys
    k with q-W < k <= q. Explicit fp32 softmax reference."""
    torch.manual_seed(0)
    B, H, T, hd, W = 2, 3, 300, 32, 32          # T deliberately not %128
    q = torch.randn(B, H, T, hd, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    flash = gated_swa._resolve_flash()
    out = flash(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                causal=True, window_size=(W - 1, 0)).transpose(1, 2)

    qi = torch.arange(T, device="cuda").view(T, 1)
    ki = torch.arange(T, device="cuda").view(1, T)
    mask = (ki <= qi) & (qi - ki <= W - 1)
    scores = (q.float() @ k.float().transpose(-1, -2)) / hd ** 0.5
    scores = scores.masked_fill(~mask, float("-inf"))
    ref = (scores.softmax(-1) @ v.float())

    diff = (out.float() - ref).abs().max().item()
    assert diff < 3e-2, f"max|flash - ref| = {diff}"
    # boundary: q=W-1 sees exactly W keys (0..W-1); q=W must NOT see k=0
    row = scores[0, 0, W].isfinite()
    assert not mask[W, 0] and row.sum().item() == W


@needs_flash
@pytest.mark.parametrize("window", [32, 128, 512])
def test_flash_module_agrees_with_flex(window):
    """SlidingWindowAttention end-to-end: flash on vs off, same weights,
    bf16, must agree to bf16 tolerance."""
    torch.manual_seed(0)
    cfg = GPTConfig(block_size=1024, vocab_size=128, n_layer=1, n_head=4,
                    n_embd=128, window=window, pos="rope", qk_norm=True,
                    norm="rms", bias=False)
    m = gated_swa.SlidingWindowAttention(cfg).cuda().to(torch.bfloat16).eval()
    x = torch.randn(2, 640, 128, device="cuda", dtype=torch.bfloat16)
    try:
        gated_swa.USE_FLASH = True
        with torch.no_grad():
            y_flash = m(x)
        gated_swa.USE_FLASH = False
        with torch.no_grad():
            y_flex = m(x)
    finally:
        gated_swa.USE_FLASH = True
    diff = (y_flash.float() - y_flex.float()).abs().max().item()
    assert diff < 3e-2, f"window {window}: max|flash - flex| = {diff}"


@needs_flash
def test_flash_backward_finite():
    torch.manual_seed(0)
    cfg = GPTConfig(block_size=1024, vocab_size=128, n_layer=1, n_head=4,
                    n_embd=128, window=32, pos="rope", norm="rms", bias=False)
    m = gated_swa.SlidingWindowAttention(cfg).cuda().to(torch.bfloat16)
    x = torch.randn(2, 512, 128, device="cuda", dtype=torch.bfloat16,
                    requires_grad=True)
    m(x).square().mean().backward()
    assert torch.isfinite(x.grad).all()
    for p in m.parameters():
        assert p.grad is None or torch.isfinite(p.grad).all()
