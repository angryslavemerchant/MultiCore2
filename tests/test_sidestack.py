"""Top-k side-stack (core/sidestack.py, pattern T): selection correctness
vs a naive materialised top-k, causality through the branch, no-op at init
under zero_init, gradient flow into q/k/v and every branch param, hourglass
(SliceBlock) wiring, flops accounting. CPU tests; compiled-vs-eager
FORWARD AND BACKWARD parity is probed on-GPU before any run (the flat-loss
lesson: never trust a forward-only check).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model import GPT, GPTConfig  # noqa: E402
from core.sidestack import (_select_topk, SideStack,  # noqa: E402
                            TopKSideBlock, side_extra_flops)


def tiny(**kw):
    base = dict(block_size=256, vocab_size=128, n_layer=2, n_head=2,
                n_embd=32, pos="rope", norm="rms", mlp="relu2", bias=False,
                qk_norm=True, attn_pattern="ST", window=8, side_topk=4)
    base.update(kw)
    return GPTConfig(**base)


def logits_pair(cfg, idx, idx2):
    torch.manual_seed(1)
    m = GPT(cfg).eval()
    with torch.no_grad():
        return m(idx, targets=idx)[0], m(idx2, targets=idx2)[0]


# ------------------------------------------------------------- selection

def test_select_matches_naive():
    torch.manual_seed(0)
    B, H, T, hd, k = 2, 3, 50, 8, 5
    q, kx = torch.randn(B, H, T, hd), torch.randn(B, H, T, hd)
    idx = _select_topk(q, kx, k, chunk=16)     # force multi-chunk merging
    sc = (q @ kx.transpose(-2, -1)).float() * hd ** -0.5
    t = torch.arange(T)
    sc = sc.masked_fill(t[None, None, None, :] > t[None, None, :, None],
                        float("-inf"))
    naive_val, _ = sc.topk(k, dim=-1)
    picked = sc.gather(-1, idx)
    # compare VALUES not indices (ties may order differently)
    assert torch.allclose(picked.sort(-1).values, naive_val.sort(-1).values)


def test_select_first_tokens_no_future():
    torch.manual_seed(0)
    q, kx = torch.randn(1, 2, 30, 8), torch.randn(1, 2, 30, 8)
    idx = _select_topk(q, kx, 8, chunk=16)
    # every SELECTED-and-used index must be causal; ghost slots (queries
    # with < k visible keys) are allowed junk — the forward masks them,
    # so here we only check that at least the top slots are causal where
    # enough candidates exist
    t = torch.arange(30).view(1, 1, 30, 1)
    n_valid = (idx <= t).sum(-1)
    assert (n_valid >= torch.minimum(torch.tensor(8),
                                     torch.arange(1, 31)).view(1, 1, 30)
            ).all()


# ------------------------------------------------------------- causality

def test_autoregressive_causality():
    torch.manual_seed(0)
    idx = torch.randint(0, 128, (1, 200))
    for pattern in ("TS", "RS"):
        cfg = tiny(attn_pattern=pattern)
        for p in (5, 64, 130):
            idx2 = idx.clone()
            idx2[0, p] = (idx2[0, p] + 1) % 128
            a, b = logits_pair(cfg, idx, idx2)
            assert torch.equal(a[0, :p], b[0, :p]), \
                f"leak at p={p} ({pattern})"


def test_r_branch_respects_window():
    """A single R layer (no canon) is a pure function of the last W
    tokens — through the attention AND the branch: perturbing a token
    outside the window must not change later logits."""
    torch.manual_seed(0)
    cfg = tiny(n_layer=1, attn_pattern="R", window=16, side_topk=4)
    idx = torch.randint(0, 128, (1, 120))
    p = 40
    idx2 = idx.clone()
    idx2[0, p] = (idx2[0, p] + 1) % 128
    a, b = logits_pair(cfg, idx, idx2)
    assert torch.equal(a[0, p + 16:], b[0, p + 16:]), \
        "R branch read beyond its window"
    assert not torch.equal(a[0, p:p + 16], b[0, p:p + 16])


def test_windowed_select_matches_naive():
    torch.manual_seed(0)
    B, H, T, hd, k, W = 2, 3, 90, 8, 5, 12
    q, kx = torch.randn(B, H, T, hd), torch.randn(B, H, T, hd)
    idx = _select_topk(q, kx, k, window=W, chunk=32)
    sc = (q @ kx.transpose(-2, -1)).float() * hd ** -0.5
    t = torch.arange(T)
    bad = (t[None, :] > t[:, None]) | (t[None, :] <= t[:, None] - W)
    sc = sc.masked_fill(bad[None, None], float("-inf"))
    naive_val, _ = sc.topk(k, dim=-1)
    picked = sc.gather(-1, idx)
    assert torch.allclose(picked.sort(-1).values, naive_val.sort(-1).values)


# ------------------------------------------------------------- init/grads

def test_branch_noop_at_init_zero_init():
    torch.manual_seed(0)
    cfg = tiny(zero_init=True, untied=True)
    m = GPT(cfg).eval()
    blk = m.transformer.h[1]
    assert isinstance(blk, TopKSideBlock)
    x = torch.randn(2, 40, cfg.n_embd)
    q = torch.randn(2, cfg.n_head, 40, cfg.n_embd // cfg.n_head)
    k, v = torch.randn_like(q), torch.randn_like(q)
    out = blk.side(x, q, k, v)
    assert out.abs().max() == 0, "zero-init branch must be a strict no-op"


def test_grads_reach_branch_and_qkv():
    torch.manual_seed(0)
    m = GPT(tiny())
    idx = torch.randint(0, 128, (1, 200))
    _, loss = m(idx, targets=idx)
    loss.backward()
    blk = m.transformer.h[1]
    side = blk.side
    for name, p in side.named_parameters():
        assert p.grad is not None and p.grad.abs().max() > 0, name
    # selection weights give the layer's q/k a second gradient path; at
    # minimum c_attn must see gradient (shared with the main attention)
    assert blk.attn.c_attn.weight.grad.abs().max() > 0


def test_branch_output_depends_on_v():
    """The branch must actually read the selected values."""
    torch.manual_seed(0)
    cfg = tiny()
    m = GPT(cfg).eval()
    side = m.transformer.h[1].side
    x = torch.randn(1, 30, cfg.n_embd)
    hd = cfg.n_embd // cfg.n_head
    q = torch.randn(1, cfg.n_head, 30, hd)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    with torch.no_grad():
        a = side(x, q, k, v)
        b = side(x, q, k, v + 1.0)
    assert not torch.allclose(a, b)


# ------------------------------------------------------------- wiring

def test_hourglass_sliceblock_wiring():
    torch.manual_seed(0)
    cfg = tiny(n_layer=4, n_head=2, n_embd=64, attn_pattern="SSTS",
               hg_frac=0.5, hg_bneck=3, hg_round=16)
    m = GPT(cfg)
    idx = torch.randint(0, 128, (1, 100))
    _, loss = m(idx, targets=idx)
    loss.backward()
    ws = cfg.layer_widths()
    assert ws[2] < cfg.n_embd          # the T layer sits at a narrow width
    inner = m.transformer.h[2].block
    assert isinstance(inner, TopKSideBlock)
    assert inner.side.c_proj.weight.grad.abs().max() > 0


def test_flops_accounting():
    cfg_t = tiny()
    cfg_f = tiny(attn_pattern="SF")
    extra = side_extra_flops(cfg_t, cfg_t.n_embd, 128)
    assert extra > 0
    ft = GPT(cfg_t).flops_per_token(128)
    ff = GPT(cfg_f).flops_per_token(128)
    # T layer = F layer + branch params + extra term, so strictly costlier
    assert ft > ff


def test_embeddings_not_2d():
    """head/dist embeddings must dodge the '2D under transformer.h. ->
    Muon' rule: orthogonalising a lookup table is nonsense."""
    side = SideStack(tiny())
    assert side.head_emb.dim() != 2 and side.dist_emb.dim() != 2
