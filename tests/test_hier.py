"""Hierarchical GPT (core/hier.py): causality perturbation tests, PKM
correctness, target bookkeeping, gates, and Muon classification. CPU
tests -- compiled behavior is exercised on-GPU by validate_hier.py."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.hier import HierGPT, HierConfig  # noqa: E402
from core.model import GPT, GPTConfig      # noqa: E402


def tiny(**kw):
    # 512-token sequences: 8 blocks of 64, 2 superblocks of 4
    base = dict(block_size=512, vocab_size=256, btok=64,
                blocks_per_super=4, d_tok=64, d_blk=48, d_sup=32,
                n_head_tok=4, n_head_blk=4, n_head_sup=4,
                n_analysis=2, n_predict=2, n_blockpred=2, n_superpred=1,
                mem_slots=64, mem_val=32, mem_heads=2, mem_topk=2,
                mem_cand=4)
    base.update(kw)
    return HierConfig(**base)


def randomize_zero_inits(m):
    """zero-init lm_head / c_proj block gradients and make logits
    insensitive at init; randomize for behavioral tests."""
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lm_head" in n or n.endswith("c_proj.weight"):
                p.normal_(0, 0.02)
        m.mem_gate.fill_(0.5)
    return m


def logits_of(m, idx):
    with torch.no_grad():
        lg, _ = m(idx, idx.roll(-1, dims=1))
    return lg


# ------------------------------------------------------------- causality

def test_block_perturbation_leaves_earlier_logits_unchanged():
    torch.manual_seed(0)
    cfg = tiny()
    m = randomize_zero_inits(HierGPT(cfg)).eval()
    idx = torch.randint(0, 256, (1, 512))
    base = logits_of(m, idx)
    pert = idx.clone()
    pert[0, 256:] = (pert[0, 256:] + 1) % 256     # blocks 4..7 changed
    after = logits_of(m, pert)
    # logits for positions < 256 - 1 must be numerically identical:
    # they see raw tokens of their own window and plans built from
    # S_<b / U_<g only. (position 255 predicts token 256 -- unchanged
    # input, so it is also identical.)
    assert torch.equal(base[0, :256], after[0, :256])


def test_superblock_perturbation():
    torch.manual_seed(0)
    cfg = tiny()
    m = randomize_zero_inits(HierGPT(cfg)).eval()
    idx = torch.randint(0, 256, (1, 512))
    base = logits_of(m, idx)
    pert = idx.clone()
    pert[0, 256:320] = (pert[0, 256:320] + 1) % 256   # first block of SB1
    after = logits_of(m, pert)
    assert torch.equal(base[0, :256], after[0, :256])


def test_within_window_causality():
    torch.manual_seed(0)
    cfg = tiny()
    m = randomize_zero_inits(HierGPT(cfg)).eval()
    idx = torch.randint(0, 256, (1, 512))
    base = logits_of(m, idx)
    pert = idx.clone()
    pert[0, 32] = (pert[0, 32] + 1) % 256         # mid-window token
    after = logits_of(m, pert)
    assert torch.equal(base[0, :32], after[0, :32])
    assert not torch.equal(base[0, 32:64], after[0, 32:64])


def test_plans_only_from_past_blocks():
    # changing block b must not change P_<=b (P_b uses S_{<b})
    torch.manual_seed(0)
    cfg = tiny()
    m = randomize_zero_inits(HierGPT(cfg)).eval()
    idx = torch.randint(0, 256, (1, 512))
    with torch.no_grad():
        H = m._analyze(idx)
        S, U = m._summaries(H, 1)
        P, G, _ = m._plans(S, U)
        pert = idx.clone()
        pert[0, 192:256] = (pert[0, 192:256] + 1) % 256   # block 3
        H2 = m._analyze(pert)
        S2, U2 = m._summaries(H2, 1)
        P2, G2, _ = m._plans(S2, U2)
    assert torch.equal(P[:, :4], P2[:, :4])       # P_0..P_3 unchanged
    assert not torch.equal(P[:, 4], P2[:, 4])     # P_4 sees S_3


# ----------------------------------------------------------- loss/targets

def test_loss_matches_flat_shifted_ce_mapping():
    # every target counted exactly once with the standard shift: the
    # hier loss on (x, y=shift(x)) must equal a manual per-position CE
    torch.manual_seed(0)
    cfg = tiny()
    m = randomize_zero_inits(HierGPT(cfg)).eval()
    idx = torch.randint(0, 256, (2, 512))
    tgt = torch.randint(0, 256, (2, 512))
    lg, loss = m(idx, tgt)
    manual = torch.nn.functional.cross_entropy(
        lg.reshape(-1, 256), tgt.reshape(-1))
    assert torch.allclose(loss, manual, atol=1e-6)


def test_aux_losses_only_in_training_mode():
    torch.manual_seed(0)
    cfg = tiny()
    m = randomize_zero_inits(HierGPT(cfg))
    idx = torch.randint(0, 256, (1, 512))
    tgt = torch.randint(0, 256, (1, 512))
    m.eval()
    _, val_loss = m(idx, tgt)
    m.train()
    _, train_loss = m(idx, tgt)
    assert train_loss.item() != pytest.approx(val_loss.item())
    lg, _ = m.eval()(idx, tgt) if False else (None, None)  # noqa


# ------------------------------------------------------------------ PKM

def test_pkm_matches_bruteforce():
    torch.manual_seed(0)
    cfg = tiny()
    mem = __import__("core.hier", fromlist=["ProductKeyMemory"]) \
        .ProductKeyMemory(cfg)
    h = torch.randn(2, cfg.n_blocks(), cfg.d_blk)
    out = mem(h)
    # brute force: full n*n score table per head
    n = cfg.sub_keys()
    half = cfg.mem_val // 2
    q = mem.wq(h).view(2, cfg.n_blocks(), cfg.mem_heads, cfg.mem_val)
    qa, qb = q[..., :half], q[..., half:]
    sa = torch.einsum("bnhd,hkd->bnhk", qa, mem.keys_a)
    sb = torch.einsum("bnhd,hkd->bnhk", qb, mem.keys_b)
    full = sa.unsqueeze(-1) + sb.unsqueeze(-2)        # (B,NB,H,n,n)
    top, addr = full.view(2, cfg.n_blocks(), cfg.mem_heads,
                          n * n).topk(cfg.mem_topk, dim=-1)
    w = torch.softmax(top, dim=-1)
    vals = mem.values(addr)
    ref_heads = (w.unsqueeze(-1) * vals).sum(-2)
    ref = mem.out(ref_heads.reshape(2, cfg.n_blocks(), -1))
    # candidate pruning (mem_cand per half) can differ from exact topk
    # only when a top pair's halves are both outside their candidate
    # sets; with cand=4 of n=8 this is rare -- allow tiny mismatch count
    close = torch.isclose(out, ref, atol=1e-5).float().mean()
    assert close > 0.95


def test_memory_gate_zero_at_init_and_receives_grad():
    torch.manual_seed(0)
    cfg = tiny()
    m = HierGPT(cfg)
    randomize_zero_inits(m)
    with torch.no_grad():
        m.mem_gate.zero_()
    idx = torch.randint(0, 256, (1, 512))
    m.train()
    base = m.disable_memory
    # with gate exactly 0, output must equal disable_memory=True output
    m.eval()
    lg_on = logits_of(m, idx)
    m.disable_memory = True
    lg_off = logits_of(m, idx)
    m.disable_memory = base
    assert torch.allclose(lg_on, lg_off, atol=1e-6)
    # and the gate still gets gradient (leaf on the residual product)
    m.train()
    _, loss = m(idx, idx.roll(-1, dims=1))
    loss.backward()
    assert m.mem_gate.grad is not None and m.mem_gate.grad.abs() > 0


def test_disable_cond_switch():
    torch.manual_seed(0)
    cfg = tiny()
    m = randomize_zero_inits(HierGPT(cfg)).eval()
    idx = torch.randint(0, 256, (1, 512))
    lg_on = logits_of(m, idx)
    m.disable_cond = True
    lg_off = logits_of(m, idx)
    assert not torch.allclose(lg_on, lg_off)


# ------------------------------------------------------- optimizer/flops

def test_muon_classification():
    m = HierGPT(tiny())
    combo = m.configure_optimizers(0.1, 6e-4, (0.9, 0.95), "cpu",
                                   opt="muon")
    muon_ids = {id(p) for g in combo.opts[0].param_groups
                for p in g["params"]}
    named = dict(m.named_parameters())
    assert id(named["memory.values.weight"]) not in muon_ids
    assert id(named["memory.keys_a"]) not in muon_ids
    assert id(named["lm_head.weight"]) not in muon_ids
    assert id(named["transformer.wte.weight"]) not in muon_ids
    a_blk = [p for n, p in named.items()
             if n.startswith("transformer.analysis.0")
             and n.endswith("c_attn.weight")]
    assert all(id(p) in muon_ids for p in a_blk)


def test_grads_reach_memory_and_plans():
    torch.manual_seed(0)
    m = randomize_zero_inits(HierGPT(tiny())).train()
    idx = torch.randint(0, 256, (2, 512))
    _, loss = m(idx, idx.roll(-1, dims=1))
    loss.backward()
    for name in ("memory.values.weight", "memory.keys_a",
                 "w_summary.weight", "w_aux_blk.weight",
                 "w_aux_sup.weight", "block_bos", "super_bos"):
        p = dict(m.named_parameters())[name]
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_flops_and_params_sane():
    m = HierGPT(HierConfig())
    n = m.num_params()
    assert 90e6 < n < 135e6, n
    f = m.flops_per_token()
    assert 0.3e9 < f < 1.5e9, f


# ------------------------------------------------------------- swa mode

def test_swa_mode_cross_boundary_visibility():
    # a token just after a block boundary must SEE the previous block's
    # raw tokens (the whole point of swa mode vs block mode)
    torch.manual_seed(0)
    cfg = tiny(token_mode="swa")
    m = randomize_zero_inits(HierGPT(cfg)).eval()
    idx = torch.randint(0, 256, (1, 512))
    base = logits_of(m, idx)
    pert = idx.clone()
    pert[0, 60] = (pert[0, 60] + 1) % 256      # near end of block 0
    after = logits_of(m, pert)
    # position 65 (block 1) is within w=64 of position 60: must change
    assert not torch.equal(base[0, 65], after[0, 65])
    # and causality: positions before 60 unchanged
    assert torch.equal(base[0, :60], after[0, :60])


def test_swa_mode_plans_still_blockwise_causal():
    torch.manual_seed(0)
    cfg = tiny(token_mode="swa")
    m = randomize_zero_inits(HierGPT(cfg)).eval()
    idx = torch.randint(0, 256, (1, 512))
    with torch.no_grad():
        H = m._analyze(idx)
        S, U = m._summaries(H, 1)
        P, _, _ = m._plans(S, U)
        pert = idx.clone()
        pert[0, 192:256] = (pert[0, 192:256] + 1) % 256   # block 3
        H2 = m._analyze(pert)
        S2, U2 = m._summaries(H2, 1)
        P2, _, _ = m._plans(S2, U2)
    # swa reach backward: only blocks whose WINDOWS see block 3 change;
    # plans for blocks 0..3 use S_{<b} of untouched-or-earlier blocks.
    # P_0..P_3 depend on S_0..S_2; S_2's tokens can't see block 3
    # (future) -> unchanged
    assert torch.equal(P[:, :4], P2[:, :4])
    assert not torch.equal(P[:, 4], P2[:, 4])


def test_swa_mode_loss_and_shapes():
    torch.manual_seed(0)
    cfg = tiny(token_mode="swa")
    m = randomize_zero_inits(HierGPT(cfg))
    idx = torch.randint(0, 256, (2, 512))
    m.train()
    lg, loss = m(idx, idx.roll(-1, dims=1))
    assert lg.shape == (2, 512, 256)
    assert loss.isfinite()
    loss.backward()
    p = dict(m.named_parameters())["memory.values.weight"]
    assert p.grad is not None and p.grad.abs().sum() > 0


# ------------------------------------------------------------- v3 (3 levels)

def tiny3(**kw):
    # 512-token sequences: 16 blocks of 32, 4 supers of 4, 2 hypers of 2
    base = dict(levels=3, btok=32, blocks_per_super=4,
                supers_per_hyper=2, d_hyp=24, n_head_hyp=2,
                token_mode="swa")
    base.update(kw)
    return tiny(**base)


def test_v3_forward_and_latent_loss():
    torch.manual_seed(0)
    m = randomize_zero_inits(HierGPT(tiny3()))
    idx = torch.randint(0, 256, (2, 512))
    tgt = idx.roll(-1, dims=1)
    m.eval()
    _, val_loss = m(idx, tgt)
    m.train()
    _, train_loss = m(idx, tgt)
    assert train_loss.item() > val_loss.item()   # aux + latent added
    assert hasattr(m, "last_latent_loss") and m.last_latent_loss > 0
    stats = m.memory_stats()
    for k in ("collapse/S_paircos", "collapse/U_paircos",
              "collapse/V_paircos"):
        assert k in stats


def test_v3_causality_hyper_boundary():
    torch.manual_seed(0)
    m = randomize_zero_inits(HierGPT(tiny3())).eval()
    idx = torch.randint(0, 256, (1, 512))
    base = logits_of(m, idx)
    pert = idx.clone()
    pert[0, 256:] = (pert[0, 256:] + 1) % 256    # entire hyper 1
    after = logits_of(m, pert)
    assert torch.equal(base[0, :224], after[0, :224])


def test_v3_dynamic_gates_and_pool_grads():
    torch.manual_seed(0)
    m = randomize_zero_inits(HierGPT(tiny3())).train()
    # gate bias must be -2 (small-but-nonzero start), not init-zeroed
    for g in m.cond_dyn:
        assert torch.allclose(g.bias, torch.full_like(g.bias, -2.0))
    idx = torch.randint(0, 256, (2, 512))
    _, loss = m(idx, idx.roll(-1, dims=1))
    loss.backward()
    named = dict(m.named_parameters())
    for n in ("pool_blk.q", "pool_sup.w.weight", "pool_hyp.q",
              "cond_dyn.0.weight", "hyper_bos", "w_cond_hy.weight"):
        assert named[n].grad is not None and \
            named[n].grad.abs().sum() > 0, n


def test_v3_muon_classification():
    m = HierGPT(tiny3())
    combo = m.configure_optimizers(0.1, 6e-4, (0.9, 0.95), "cpu",
                                   opt="muon")
    muon_ids = {id(p) for g in combo.opts[0].param_groups
                for p in g["params"]}
    named = dict(m.named_parameters())
    assert any(id(p) in muon_ids for n, p in named.items()
               if n.startswith("transformer.hyperpred.") and p.dim() == 2)
    assert id(named["pool_blk.w.weight"]) not in muon_ids
    assert id(named["pool_blk.q"]) not in muon_ids


def test_v2_checkpoint_compat_unchanged():
    # levels=2 model must be state-dict-identical to before the v3 work
    torch.manual_seed(0)
    m = HierGPT(tiny())
    keys = set(m.state_dict().keys())
    assert not any("pool_" in k or "hyper" in k or "cond_dyn" in k
                   for k in keys)
    assert "w_summary.weight" in keys and "cond_gate" in keys
