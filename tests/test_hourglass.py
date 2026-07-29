"""Slice-carry hourglass: geometry, param accounting, carry semantics."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model import GPT, GPTConfig, SliceBlock, config_dict


def tiny_cfg(**kw):
    base = dict(block_size=64, vocab_size=256, n_layer=6, n_head=4,
                n_embd=96, pos="rope", hg_frac=0.5, hg_bneck=4)
    base.update(kw)
    return GPTConfig(**base)


def test_widths_geometry():
    cfg = GPTConfig(n_embd=1128, n_head=12, hg_frac=0.2, hg_bneck=8)
    ws = cfg.layer_widths()
    assert ws[0] == ws[-1] == 1128 and min(ws) == ws[7]
    assert all(w % 24 == 0 for w in ws)
    # monotone down to the waist, up after it
    assert all(a >= b for a, b in zip(ws[:8], ws[1:8]))
    assert all(a <= b for a, b in zip(ws[7:], ws[8:]))
    # mid layers extend the waist flat
    wm = GPTConfig(n_embd=1128, n_head=12, hg_frac=0.2, hg_bneck=8,
                   hg_mid=4).layer_widths()
    assert len(wm) == 16 and wm[7:12] == [ws[7]] * 5
    # hourglass off -> uniform
    assert GPTConfig().layer_widths() == [768] * 12


def test_param_count_matches_analytic():
    cfg = GPTConfig(block_size=128, n_embd=1128, n_head=12, pos="rope",
                    hg_frac=0.2, hg_bneck=8)
    m = GPT(cfg)
    ws = cfg.layer_widths()
    expected = (cfg.vocab_size * 1128 + 2 * 1128
                + sum(12 * w * w + 13 * w for w in ws))
    assert m.num_params() == expected
    # and the matcher's chosen config lands within 0.5% of dense
    dense = GPT(GPTConfig(block_size=128, pos="rope")).num_params()
    assert abs(m.num_params() - dense) / dense < 0.005


def test_slice_carry_leaves_tail_untouched():
    cfg = tiny_cfg()
    blk = SliceBlock(cfg, "causal", 48)
    x = torch.randn(2, 16, 96)
    y = blk(x)
    assert y.shape == x.shape
    assert torch.equal(y[..., 48:], x[..., 48:])
    assert not torch.allclose(y[..., :48], x[..., :48])


def test_forward_backward_all_layers_grad():
    cfg = tiny_cfg(hg_mid=2)
    m = GPT(cfg)
    assert len(m.transformer.h) == 8
    x = torch.randint(0, 256, (2, 32))
    logits, loss = m(x, x)
    assert logits.shape == (2, 32, 256) and torch.isfinite(loss)
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_flops_accounting():
    hg = GPT(tiny_cfg())
    uniform = GPT(tiny_cfg(hg_frac=0.0))
    assert hg.flops_per_token(64) < uniform.flops_per_token(64)


def test_old_configs_still_load():
    """Checkpoint configs from before the hourglass fields must rebuild."""
    old = {k: v for k, v in config_dict(GPTConfig()).items()
           if not k.startswith("hg_")}
    cfg = GPTConfig(**old)
    assert cfg.layer_widths() == [768] * 12 and cfg.n_layer_total() == 12
