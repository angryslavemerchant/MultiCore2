"""Uberloop (cfg.loops / LoopedBlock): weight-tied per-layer repetition
with per-iteration gains/scales. CPU tests — compiled/DDP behavior is
exercised on-GPU by validate_uberloop."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model import GPT, GPTConfig, LoopedBlock  # noqa: E402


def tiny(**kw):
    base = dict(block_size=64, vocab_size=128, n_layer=4, n_head=2,
                n_embd=32, pos="rope", norm="rms", mlp="relu2", bias=False,
                attn_pattern="SFSF", window=8)
    base.update(kw)
    return GPTConfig(**base)


def hg(**kw):
    # 4-layer hourglass: widths [32, 16, 16, 32] at hg_round=16
    base = dict(hg_frac=0.5, hg_bneck=2, hg_round=16)
    base.update(kw)
    return tiny(**base)


# ------------------------------------------------------------ cfg parsing

def test_layer_loops_default_all_one():
    assert tiny().layer_loops() == [1, 1, 1, 1]


def test_layer_loops_parse():
    assert tiny(loops="1,2,4,1").layer_loops() == [1, 2, 4, 1]


def test_layer_loops_wrong_count():
    with pytest.raises(AssertionError):
        tiny(loops="1,2").layer_loops()


def test_layer_loops_zero_rejected():
    with pytest.raises(AssertionError):
        tiny(loops="1,0,1,1").layer_loops()


# ------------------------------------------------------------- structure

def test_looped_layers_wrapped_and_shared():
    m = GPT(tiny(loops="1,2,4,1"))
    h = m.transformer.h
    assert not isinstance(h[0], LoopedBlock)
    assert isinstance(h[1], LoopedBlock) and h[1].loops == 2
    assert isinstance(h[2], LoopedBlock) and h[2].loops == 4
    # weights are shared: a LoopedBlock holds ONE inner block plus
    # loops*(gain+scale) vectors, nothing else
    inner = sum(p.numel() for p in h[2].block.parameters())
    total = sum(p.numel() for p in h[2].parameters())
    assert total == inner + 4 * 2 * 32


def test_loop_executes_n_times():
    m = GPT(tiny(loops="1,1,3,1"))
    calls = []
    m.transformer.h[2].block.register_forward_hook(
        lambda *a: calls.append(1))
    idx = torch.randint(0, 128, (1, 16))
    m(idx, targets=idx)
    assert len(calls) == 3


def test_loops_at_init_match_unrolled_shared_stack():
    # g=s=1 at init, so a 2x looped layer must equal applying its block
    # twice (up to fp reassociation: the wrapper computes
    # x + (block(x) - x), not block(x) directly)
    torch.manual_seed(0)
    m = GPT(tiny(loops="1,2,1,1"))
    m.eval()
    x = torch.randn(1, 16, 32)
    blk = m.transformer.h[1]
    with torch.no_grad():
        y = blk(x)
        y2 = blk.block(blk.block(x))
    assert torch.allclose(y, y2, atol=1e-6)


def test_loops_one_is_plain_block():
    # loops="1,...": no wrapper, model is bitwise the unlooped model
    torch.manual_seed(0)
    a = GPT(tiny(loops="1,1,1,1"))
    torch.manual_seed(0)
    b = GPT(tiny())
    idx = torch.randint(0, 128, (1, 16))
    with torch.no_grad():
        la, _ = a(idx, targets=idx)
        lb, _ = b(idx, targets=idx)
    assert torch.equal(la, lb)


# ---------------------------------------------------------- hourglass carry

def test_slice_carry_dims_untouched_by_loops():
    # gains on carry dims must be inert: the carried tail of the residual
    # stream passes a looped SliceBlock unchanged
    m = GPT(hg(loops="1,4,1,1"))
    blk = m.transformer.h[1]          # width 16 < n_embd 32, 4x looped
    assert isinstance(blk, LoopedBlock)
    with torch.no_grad():
        for g in blk.gains:
            g.mul_(3.0)               # would corrupt carry if applied
    x = torch.randn(1, 16, 32)
    with torch.no_grad():
        y = blk(x)
    assert torch.equal(y[..., 16:], x[..., 16:])


# ----------------------------------------------------------------- grads

def test_grads_reach_gains_scales_and_shared_weights():
    m = GPT(tiny(loops="1,2,4,1", untied=True, zero_init=True))
    # zero-init blocks the residual path at init: randomize the residual
    # projections and head so gradients actually flow
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("c_proj.weight") or "lm_head" in n:
                p.normal_(0, 0.02)
    idx = torch.randint(0, 128, (2, 16))
    _, loss = m(idx, targets=idx)
    loss.backward()
    blk = m.transformer.h[2]
    assert all(g.grad is not None and g.grad.abs().sum() > 0
               for g in blk.gains)
    assert all(s.grad is not None and s.grad.abs().sum() > 0
               for s in blk.scales)
    w = blk.block.attn.c_attn.weight
    assert w.grad is not None and w.grad.abs().sum() > 0


def test_gains_stay_out_of_muon():
    m = GPT(tiny(loops="1,2,1,1"))
    combo = m.configure_optimizers(0.1, 6e-4, (0.9, 0.95), "cpu",
                                   opt="muon")
    muon_ids = {id(p) for g in combo.opts[0].param_groups
                for p in g["params"]}
    blk = m.transformer.h[1]
    for p in list(blk.gains) + list(blk.scales):
        assert id(p) not in muon_ids
    assert id(blk.block.attn.c_attn.weight) in muon_ids


# ----------------------------------------------------------------- flops

def test_flops_charge_loops():
    f1 = GPT(tiny()).flops_per_token(64)
    f2 = GPT(tiny(loops="1,1,2,1")).flops_per_token(64)
    f4 = GPT(tiny(loops="1,1,4,1")).flops_per_token(64)
    assert f1 < f2 < f4
    # extra iterations charge param + score FLOPs of that layer only:
    # the 2->4 increment is twice the 1->2 increment (same layer, +1 vs
    # +3 iterations), modulo the tiny gain params counted once
    assert abs((f4 - f2) - 2 * (f2 - f1)) / f1 < 0.01


def test_causality_preserved_under_loops():
    torch.manual_seed(0)
    m = GPT(tiny(loops="2,2,2,2"))
    m.eval()
    idx = torch.randint(0, 128, (1, 32))
    with torch.no_grad():
        base, _ = m(idx, targets=idx)
        idx2 = idx.clone()
        idx2[0, 20:] = (idx2[0, 20:] + 1) % 128
        pert, _ = m(idx2, targets=idx2)
    assert torch.allclose(base[0, :19], pert[0, :19], atol=1e-5)


def test_state_dict_roundtrip():
    m = GPT(tiny(loops="1,2,4,1"))
    sd = m.state_dict()
    m2 = GPT(tiny(loops="1,2,4,1"))
    m2.load_state_dict(sd)
    idx = torch.randint(0, 128, (1, 16))
    with torch.no_grad():
        a, _ = m(idx, targets=idx)
        b, _ = m2(idx, targets=idx)
    assert torch.equal(a, b)
