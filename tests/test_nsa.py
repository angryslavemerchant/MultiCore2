"""NSA-with-registers (core/nsa.py) unit tests: causality, register
gradient flow, muon classification, hourglass composition, flops."""
import sys
import os

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model import GPT, GPTConfig  # noqa: E402


def tiny(**kw):
    base = dict(block_size=64, vocab_size=256, n_layer=2, n_head=4,
                n_embd=64, bias=False, attn="nsa", mlp="relu2",
                norm="rms", qk_norm=True, pos="rope", untied=True,
                zero_init=False, softcap=15,
                window=16, nsa_block=8, nsa_topk=2, nsa_nreg=32)
    base.update(kw)
    return GPTConfig(**base)


def build(**kw):
    torch.manual_seed(1337)
    m = GPT(tiny(**kw))
    with torch.no_grad():
        # zero-init head would freeze logits; give it signal for tests
        m.lm_head.weight.normal_(0, 0.02)
    return m


def test_forward_shapes_and_loss():
    m = build()
    x = torch.randint(0, 256, (2, 64))
    logits, loss = m(x, targets=x)
    assert logits.shape == (2, 64, 256)
    assert torch.isfinite(loss)


def test_causality_bitwise():
    """Changing tokens from position s on must not change logits < s.
    Exercises all three branches (s inside a block, at a block edge)."""
    m = build().eval()
    torch.manual_seed(0)
    idx = torch.randint(0, 256, (1, 64))
    with torch.no_grad():
        base, _ = m(idx, idx)
        for s in (12, 24, 33, 48):     # mid-block and block boundaries
            pert = idx.clone()
            pert[0, s:] = (pert[0, s:] + 1) % 256
            after, _ = m(pert, pert)
            assert torch.equal(base[0, :s], after[0, :s]), s


def test_registers_and_gates_get_grad():
    m = build()
    x = torch.randint(0, 256, (2, 64))
    _, loss = m(x, targets=x)
    loss.backward()
    for i, blk in enumerate(m.transformer.h):
        assert blk.attn.registers.grad is not None, i
        assert blk.attn.registers.grad.norm() > 0, i
        assert blk.attn.w_gate.weight.grad.norm() > 0, i


def test_registers_not_in_muon():
    m = build()
    combo = m.configure_optimizers(0.1, 6e-4, (0.9, 0.95), "cpu",
                                   opt="muon")
    muon_ids = {id(p) for g in combo.opts[0].param_groups
                for p in g["params"]}
    for n, p in m.named_parameters():
        if n.endswith(".registers"):
            assert id(p) not in muon_ids, n
        if n.endswith("attn.c_attn.weight"):
            assert id(p) in muon_ids, n


def test_hourglass_composition():
    """NSA at sliced widths under hg_frac (the run-three trunk)."""
    m = build(hg_frac=0.5, hg_bneck=2, n_layer=4, hg_round=8)
    x = torch.randint(0, 256, (1, 64))
    logits, loss = m(x, targets=x)
    assert torch.isfinite(loss)
    m.eval()
    with torch.no_grad():
        base, _ = m(x, x)
        pert = x.clone()
        pert[0, 40:] = (pert[0, 40:] + 1) % 256
        after, _ = m(pert, pert)
        assert torch.equal(base[0, :40], after[0, :40])


def test_flops_accounting():
    m = build()
    f = m.flops_per_token(64)
    # must exceed pure-param cost (attention keys are charged)
    assert f > 6 * m.num_params()


def test_registers_influence_output():
    """Registers must actually reach the logits (KV path live)."""
    m = build().eval()
    x = torch.randint(0, 256, (1, 64))
    with torch.no_grad():
        base, _ = m(x, x)
        for blk in m.transformer.h:
            blk.attn.registers.add_(1.0)
        after, _ = m(x, x)
    assert not torch.allclose(base, after)


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")


def test_disable_registers_ablation():
    """Pseudo-ablation: flag changes outputs, keeps causality, no NaN
    (incl. the first-block empty-row guard), stats populated."""
    m = build().eval()
    x = torch.randint(0, 256, (1, 64))
    with torch.no_grad():
        base, _ = m(x, x)
        for blk in m.transformer.h:
            blk.attn.disable_registers = True
        off, _ = m(x, x)
    assert torch.isfinite(off).all()
    assert not torch.allclose(base, off)
    st = m.transformer.h[0].attn.reg_stats
    assert st["reg_sel_frac"] == 0.0          # nothing selectable
    # causality still holds with the guard in place
    with torch.no_grad():
        pert = x.clone()
        pert[0, 40:] = (pert[0, 40:] + 1) % 256
        after, _ = m(pert, pert)
        assert torch.equal(off[0, :40], after[0, :40])


def test_reg_stats_live_when_enabled():
    m = build().eval()
    x = torch.randint(0, 256, (2, 64))
    with torch.no_grad():
        m(x, x)
    st = m.transformer.h[0].attn.reg_stats
    assert 0.0 < st["reg_cmp_mass"] <= 1.0
    assert 0.0 <= st["reg_sel_frac"] <= 1.0
