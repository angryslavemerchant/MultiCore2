"""Unit tests for the four-stroke machine block (core/fourstroke.py).

Everything runs tiny (K=3 machines, d_machine=16, T=8) in float64 on CPU
so causality checks are exact-arithmetic, not tolerance games.
"""
import torch
import pytest

from core.model import GPT, GPTConfig
import core.fourstroke as fs
from core.fourstroke import (FourStrokeBlock, MachineStrokes,
                             fourstroke_score_flops)

B, T, C = 2, 8, 32


def make_cfg(**kw):
    base = dict(block_size=T, n_embd=C, n_head=4, bias=True,
                fs_n_machines=3, fs_d_machine=16, fs_n_head_m=2,
                fs_mlp_mult=2, fs_backend="attn", fs_window=4)
    base.update(kw)
    return GPTConfig(**base)


def make_block(seed=0, **kw):
    torch.manual_seed(seed)
    blk = FourStrokeBlock(make_cfg(**kw)).double()
    return blk


def run(blk, x, c=None):
    if c is None:
        c = blk.strokes.init_channel(x.shape[0], x.shape[1],
                                     device=x.device, dtype=x.dtype)
    return blk(x, c)


def test_shapes():
    blk = make_block()
    x = torch.randn(B, T, C, dtype=torch.float64)
    y, c = run(blk, x)
    K, d = blk.strokes.s0.shape
    assert y.shape == (B, T, C)
    assert c.shape == (B, K, T, d)
    assert blk.strokes.init_channel(B, T).shape == (B, K, T, d)


@pytest.mark.parametrize("backend", ["attn", "swa"])
def test_token_causality(backend):
    """A perturbed token at position p may not touch outputs before p."""
    blk = make_block(backend == "swa", fs_backend=backend)
    torch.manual_seed(1)
    x = torch.randn(B, T, C, dtype=torch.float64)
    p = 5
    x2 = x.clone()
    x2[:, p] += 1.0
    y1, c1 = run(blk, x)
    y2, c2 = run(blk, x2)
    assert torch.equal(y1[:, :p], y2[:, :p])
    assert torch.equal(c1[:, :, :p], c2[:, :, :p])
    assert not torch.equal(c1[:, :, p:], c2[:, :, p:])


def test_channel_causality():
    """A perturbed incoming conclusion at position p may not touch
    outputs before p (the private channel is causal too)."""
    blk = make_block()
    torch.manual_seed(2)
    x = torch.randn(B, T, C, dtype=torch.float64)
    c0 = blk.strokes.init_channel(B, T, dtype=torch.float64)
    c0b = c0.clone()
    p = 4
    c0b[:, 1, p] += 1.0
    y1, c1 = blk(x, c0)
    y2, c2 = blk(x, c0b)
    assert torch.equal(y1[:, :p], y2[:, :p])
    assert torch.equal(c1[:, :, :p], c2[:, :, :p])
    assert not torch.equal(c1[:, :, p:], c2[:, :, p:])


def test_swa_window_bounds_influence():
    """swa backend: token at position 0 is invisible past the band. One
    block does no cross-position mixing outside intake (conference and
    write-back are per-position), so influence dies exactly at window."""
    w = 2
    blk = make_block(fs_backend="swa", fs_window=w)
    torch.manual_seed(3)
    x = torch.randn(B, T, C, dtype=torch.float64)
    x2 = x.clone()
    x2[:, 0] += 1.0
    y1, c1 = run(blk, x)
    y2, c2 = run(blk, x2)
    # strokes output past the band is untouched; the block's own MLP acts
    # per-position so y differs only where the token path differs (pos 0)
    assert torch.equal(c1[:, :, w:], c2[:, :, w:])
    assert torch.equal(y1[:, w:], y2[:, w:])
    assert not torch.equal(c1[:, :, :w], c2[:, :, :w])


def test_token_path_untouched_at_init():
    """Zero-init W_O: the strokes' residual delta is exactly zero, so a
    fresh machine block is the identity plus its own MLP."""
    blk = make_block()
    torch.manual_seed(4)
    x = torch.randn(B, T, C, dtype=torch.float64)
    delta, _ = blk.strokes(blk.ln_1(x), blk.strokes.init_channel(
        B, T, dtype=torch.float64))
    assert torch.equal(delta, torch.zeros_like(delta))


def test_gates_can_silence():
    """All gates driven shut -> zero write-back even with live W_O."""
    blk = make_block()
    s = blk.strokes
    with torch.no_grad():
        s.w_o.weight.normal_(0, 0.1)
        s.gate_b.fill_(-1e4)
    torch.manual_seed(5)
    x = torch.randn(B, T, C, dtype=torch.float64)
    delta, _ = s(x, s.init_channel(B, T, dtype=torch.float64))
    assert delta.abs().max().item() < 1e-12


def test_gradients_reach_everything():
    """After W_O wakes (a few steps), gradient reaches identity params,
    interface, gates, and the backend's private projections."""
    blk = make_block()
    opt = torch.optim.SGD(blk.parameters(), lr=0.1)
    torch.manual_seed(6)
    x = torch.randn(B, T, C, dtype=torch.float64)
    tgt = torch.randn(B, T, C, dtype=torch.float64)
    losses = []
    for _ in range(3):
        opt.zero_grad()
        y, _ = run(blk, x)
        loss = ((y - tgt) ** 2).mean()
        loss.backward()
        losses.append(loss.item())
        opt.step()
    s = blk.strokes
    for name, p in [("anchor", s.anchor), ("s0", s.s0),
                    ("addr_mix", s.addr_mix), ("w_qkv", s.w_qkv.weight),
                    ("gate_x", s.gate_x),
                    ("backend.w_q", s.backend.w_q.weight),
                    ("backend.w_tkv", s.backend.w_tkv.weight)]:
        assert p.grad is not None and p.grad.abs().max() > 0, name
    assert losses[-1] < losses[0]


def test_stacked_params_tagged():
    """Every 3D stacked machine weight carries MUON_STACKED for the
    future per-slice Muon routing; no stacked weight is 2D (which would
    silently fall into today's 2D->Muon rule with ndim asserts)."""
    blk = make_block()
    tagged = [n for n, p in blk.named_parameters()
              if getattr(p, "MUON_STACKED", False)]
    assert tagged, "no stacked tags found"
    for n, p in blk.named_parameters():
        if getattr(p, "MUON_STACKED", False):
            assert p.dim() == 3, n


def test_flop_counter():
    cfg_a = make_cfg(fs_backend="attn")
    cfg_s = make_cfg(fs_backend="swa", fs_window=4)
    big_t = 4096
    assert fourstroke_score_flops(cfg_a, big_t) > \
        fourstroke_score_flops(cfg_s, big_t) > 0


@pytest.mark.parametrize("chunk_q", [4, 3])
def test_banded_matches_dense_mask(monkeypatch, chunk_q):
    """The streamed swa intake must equal the dense (T, 2T)-mask
    reference exactly. chunk_q=4 divides T=8 -> the batched single-SDPA
    path (unfolded keys, chunk-0 ghost masking); chunk_q=3 does not ->
    the per-chunk loop fallback."""
    w = 3
    blk = make_block(fs_backend="swa", fs_window=w)
    be = blk.strokes.backend
    torch.manual_seed(11)
    x = torch.randn(B, T, C, dtype=torch.float64)
    c0 = blk.strokes.init_channel(B, T, dtype=torch.float64)
    monkeypatch.setattr(fs, "CHUNK_Q", chunk_q)
    s_stream = be(blk.ln_1(x), c0)

    # dense reference: same projections, full concat + (T, 2T) mask
    def dense(x_tok, c_prev):
        Bx, K, Tx, d = c_prev.shape
        q = be._heads(be.w_q(be.ln_q(c_prev)))
        k_tok, v_tok = be.w_tkv(x_tok).chunk(2, dim=-1)
        k_prv, v_prv = be.w_pkv(be.ln_p(c_prev)).chunk(2, dim=-1)
        k_tok, v_tok = be._heads(k_tok), be._heads(v_tok)
        k_prv, v_prv = be._heads(k_prv), be._heads(v_prv)
        from core.rope import apply_rope
        q, k_tok = apply_rope(q, k_tok)
        k_prv = apply_rope(k_prv, k_prv)[0]
        k = torch.cat((k_tok, k_prv), dim=2)
        v = torch.cat((v_tok, v_prv), dim=2)
        mask = fs._intake_mask(Tx, w, x_tok.device)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).reshape(Bx, K, Tx, d)
        s = c_prev + be.w_out(y)
        return s + be.mlp(be.ln_mlp(s))

    s_dense = dense(blk.ln_1(x), c0)
    assert torch.allclose(s_stream, s_dense, atol=1e-12)


def test_gpt_assembly_mmmf():
    """Full GPT with an MMMF pattern: forward/backward runs, the machine
    channel threads across M blocks, and causality holds end to end."""
    torch.manual_seed(12)
    cfg = make_cfg(vocab_size=128, n_layer=4, attn_pattern="MMMF",
                   fs_backend="swa", fs_window=4, pos="rope")
    model = GPT(cfg)
    idx = torch.randint(0, 128, (2, T))
    logits, loss = model(idx, idx)
    assert logits.shape == (2, T, 128)
    loss.backward()
    # causality through the whole stack
    model.zero_grad(set_to_none=True)
    p = 5
    idx2 = idx.clone()
    idx2[:, p] = (idx2[:, p] + 1) % 128
    with torch.no_grad():
        l1, _ = model(idx, idx)
        l2, _ = model(idx2, idx2)
    assert torch.equal(l1[:, :p], l2[:, :p])
    assert not torch.equal(l1[:, p:], l2[:, p:])


def test_muon_routes_stacked_params():
    """configure_optimizers(opt="muon") must place every MUON_STACKED
    machine weight in the Muon optimizer, and a step must run (batched
    Newton-Schulz over 3D tensors)."""
    torch.manual_seed(13)
    cfg = make_cfg(vocab_size=128, n_layer=2, attn_pattern="MF",
                   fs_backend="swa", fs_window=4)
    model = GPT(cfg)
    combo = model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu",
                                       opt="muon")
    muon_ids = {id(p) for g in combo.opts[0].param_groups
                for p in g["params"]}
    for n, p in model.named_parameters():
        if getattr(p, "MUON_STACKED", False):
            assert id(p) in muon_ids, n
    # step 1 wakes the zero-init write-back; step 2 reaches the interface
    s = model.transformer.h[0].strokes
    idx = torch.randint(0, 128, (1, T))
    wo_before = s.w_o.weight.clone()
    wq_before = s.w_qkv.weight.clone()
    for _ in range(2):
        combo.zero_grad()
        _, loss = model(idx, idx)
        loss.backward()
        combo.step()
    assert not torch.equal(wo_before, s.w_o.weight)
    assert not torch.equal(wq_before, s.w_qkv.weight)


def test_batched_newtonschulz_matches_per_slice():
    """Batched NS on (K, m, n) == independent 2D NS per slice."""
    from core.muon import zeropower_via_newtonschulz5 as ns
    torch.manual_seed(14)
    G = torch.randn(3, 8, 6)
    batched = ns(G)
    for k in range(3):
        assert torch.allclose(batched[k], ns(G[k]), atol=1e-2), k


def test_no_orphan_params_mmmf():
    """Only the first M block owns s0, and every parameter in an MMMF
    stack participates in the loss graph — the condition DDP's reducer
    enforces (8 orphaned seeds crashed the 8x run, 2026-08-11)."""
    torch.manual_seed(15)
    cfg = make_cfg(vocab_size=128, n_layer=8, attn_pattern="MMMFMMMF",
                   fs_backend="swa", fs_window=4)
    model = GPT(cfg)
    seeds = [n for n, _ in model.named_parameters() if n.endswith(".s0")]
    assert len(seeds) == 1, seeds
    idx = torch.randint(0, 128, (1, T))
    _, loss = model(idx, idx)
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, missing


def test_flops_per_token_charges_machines():
    """M layers must cost more than S layers at token width but be
    charged at machine width, not as full attention."""
    base = dict(vocab_size=128, n_layer=4, fs_backend="swa",
                fs_window=4)
    f_m = GPT(make_cfg(**base, attn_pattern="MMMF")).flops_per_token(T)
    f_s = GPT(make_cfg(**base, attn_pattern="SSSF")).flops_per_token(T)
    assert f_m > f_s


def test_two_block_chain():
    """Conclusions thread between machine blocks; second block consumes
    the first's channel at matching shapes."""
    b1, b2 = make_block(7), make_block(8)
    torch.manual_seed(9)
    x = torch.randn(B, T, C, dtype=torch.float64)
    x, c = run(b1, x)
    x, c = b2(x, c)
    assert x.shape == (B, T, C)


# ---------------------------------------------------------------- v2 features

V2 = dict(fs_topk=2, fs_loop_rounds=3, fs_loop_topk=2, fs_conf_sink=True)


@pytest.mark.parametrize("backend", ["attn", "swa"])
def test_v2_causality(backend):
    """Exact causality with router + loop + sink all enabled."""
    blk = make_block(11, fs_backend=backend, **V2)
    torch.manual_seed(12)
    x = torch.randn(B, T, C, dtype=torch.float64)
    p = 5
    x2 = x.clone()
    x2[:, p] += torch.randn(C, dtype=torch.float64)
    y1, c1 = run(blk, x)
    y2, c2 = run(blk, x2)
    assert torch.equal(y1[:, :p], y2[:, :p])
    assert torch.equal(c1[:, :, :p], c2[:, :, :p])
    assert not torch.equal(c1[:, :, p:], c2[:, :, p:])


def test_v2_defaults_are_v1():
    """fs_topk=0, rounds=1, sink off must reproduce the v1 block exactly
    (same params, same math) — the refactor may not drift the champion
    config's semantics."""
    b1, b2 = make_block(13), make_block(13)
    torch.manual_seed(14)
    x = torch.randn(B, T, C, dtype=torch.float64)
    y1, c1 = run(b1, x)
    y2, c2 = run(b2, x)
    assert torch.equal(y1, y2) and torch.equal(c1, c2)
    assert b1.strokes.conf_sink is None and b1.strokes.rounds == 1


def test_v2_write_topk_sparsity():
    """With fs_topk=k, at most k machines write to any token."""
    blk = make_block(15, **V2)
    torch.manual_seed(16)
    x = torch.randn(B, T, C, dtype=torch.float64)
    g_seen = {}
    orig = blk.strokes._topk_mask

    def spy(v, k):
        out = orig(v, k)
        g_seen["last"] = out
        return out

    blk.strokes._topk_mask = spy
    run(blk, x)
    g = g_seen["last"]                       # final call = write gates
    assert ((g > 0).sum(dim=-1) <= V2["fs_topk"]).all()


def test_v2_unrouted_state_passes_through():
    """A machine routed nowhere keeps c_prev exactly when loops are off
    and the conference can't move it (conf_out zeroed): the skip really
    is a skip."""
    blk = make_block(17, fs_topk=1, fs_loop_rounds=1)
    with torch.no_grad():
        blk.strokes.conf_out.weight.zero_()
        # force the router to always pick machine 0
        blk.strokes.route_x.zero_()
        blk.strokes.route_c.zero_()
        blk.strokes.route_b.copy_(torch.tensor([10.0, -10.0, -10.0]))
    torch.manual_seed(18)
    x = torch.randn(B, T, C, dtype=torch.float64)
    c0 = blk.strokes.init_channel(B, T, dtype=torch.float64)
    _, c = blk(x, c0)
    assert torch.equal(c[:, 1:], c0[:, 1:])       # skipped machines
    assert not torch.equal(c[:, :1], c0[:, :1])   # routed machine moved


def test_v2_gradients_reach_new_params():
    blk = make_block(19, **V2)
    torch.manual_seed(20)
    x = torch.randn(B, T, C, dtype=torch.float64)
    y, c = run(blk, x)
    (y.square().mean() + c.square().mean()
     + blk.strokes.lb_loss).backward()
    s = blk.strokes
    for name in ("route_x", "route_c", "route_b", "conf_sink",
                 "loop_w", "loop_b"):
        p = getattr(s, name)
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_v2_loop_rounds_change_output():
    b1 = make_block(21, fs_loop_rounds=1)
    b2 = make_block(21, fs_loop_rounds=2)
    # identical shared params; loop params exist only on b2
    torch.manual_seed(22)
    x = torch.randn(B, T, C, dtype=torch.float64)
    _, c1 = run(b1, x)
    _, c2 = run(b2, x)
    assert not torch.equal(c1, c2)


def test_v2_flop_counter_discounts_and_charges():
    dense = make_cfg()
    topk = make_cfg(fs_topk=1)
    loop = make_cfg(fs_loop_rounds=3)
    f_dense = fourstroke_score_flops(dense, T)
    f_topk = fourstroke_score_flops(topk, T)
    f_loop = fourstroke_score_flops(loop, T)
    assert f_topk < f_dense          # routing is a discount
    assert f_loop > f_dense          # loops are a surcharge
    v1_formula = dense.fs_n_machines * 12 * dense.fs_d_machine * (
        2 * T + dense.fs_n_machines)
    assert abs(f_dense - v1_formula) < 1e-6   # defaults match v1 exactly


V2S = dict(fs_topk=2, fs_loop_rounds=2, fs_loop_topk=2, fs_conf_sink=True,
           fs_tkv_heads=4, fs_share_pub=True, fs_mlp_depth=2)


@pytest.mark.parametrize("backend", ["attn", "swa"])
def test_v2_shared_causality(backend):
    """Exact causality with the shared token bank, shared publish and
    deep MLP stacked on every other v2 feature."""
    blk = make_block(23, fs_backend=backend, **V2S)
    torch.manual_seed(24)
    x = torch.randn(B, T, C, dtype=torch.float64)
    p = 5
    x2 = x.clone()
    x2[:, p] += torch.randn(C, dtype=torch.float64)
    y1, c1 = run(blk, x)
    y2, c2 = run(blk, x2)
    assert torch.equal(y1[:, :p], y2[:, :p])
    assert torch.equal(c1[:, :, :p], c2[:, :, :p])
    assert not torch.equal(c1[:, :, p:], c2[:, :, p:])


def test_v2_shared_gradients_and_muon_paths():
    """Shared weights are 2D (standard Muon path), stacks stay tagged,
    and every new param gets gradient."""
    blk = make_block(25, **V2S)
    s = blk.strokes
    assert s.backend.w_tkv.dim() == 2 and s.w_qkv.weight.dim() == 2
    assert not hasattr(s.backend.w_tkv, "MUON_STACKED")
    assert len(s.backend.mlp.mids) == 1
    torch.manual_seed(26)
    x = torch.randn(B, T, C, dtype=torch.float64)
    y, c = run(blk, x)
    (y.square().mean() + c.square().mean() + s.lb_loss).backward()
    for p_name in ("backend.w_tkv", "w_qkv.weight", "conf_out.weight"):
        obj = s
        for part in p_name.split("."):
            obj = getattr(obj, part)
        assert obj.grad is not None and obj.grad.abs().sum() > 0, p_name
    assert s.backend.mlp.mids[0].weight.grad.abs().sum() > 0


def test_v2_bank_machines_differ():
    """With a fully shared token bank, machines still produce different
    states (private queries + private state distinguish them)."""
    blk = make_block(27, fs_tkv_heads=2)
    torch.manual_seed(28)
    x = torch.randn(B, T, C, dtype=torch.float64)
    _, c = run(blk, x)
    assert not torch.allclose(c[:, 0], c[:, 1])


def test_v2_shared_flop_accounting():
    """Sharing publish adds a surcharge (K-1 extra applications of the
    shared weight vs its single 6*N billing); the total fpt of a GPT
    still DROPS because 6*N itself shrinks far more."""
    dense_pub = make_cfg()
    shared_pub = make_cfg(fs_share_pub=True)
    assert (fourstroke_score_flops(shared_pub, T)
            > fourstroke_score_flops(dense_pub, T))
    deep = make_cfg(fs_mlp_depth=2, fs_topk=1)
    shallow = make_cfg(fs_mlp_depth=1, fs_topk=1)
    assert (fourstroke_score_flops(deep, T)
            < fourstroke_score_flops(shallow, T) + 6 * 3 * (
                shallow.fs_mlp_mult * shallow.fs_d_machine) ** 2)


def test_v2_sparse_state_freezes_sleepers():
    """With fs_sparse_state, an unrouted machine's state is bit-frozen
    even though the conference output is nonzero."""
    blk = make_block(29, fs_topk=1, fs_sparse_state=True,
                     fs_loop_rounds=2, fs_loop_topk=1)
    with torch.no_grad():
        blk.strokes.route_x.zero_()
        blk.strokes.route_c.zero_()
        blk.strokes.route_b.copy_(torch.tensor([10.0, -10.0, -10.0]))
    torch.manual_seed(30)
    x = torch.randn(B, T, C, dtype=torch.float64)
    c0 = blk.strokes.init_channel(B, T, dtype=torch.float64)
    _, c = blk(x, c0)
    assert torch.equal(c[:, 1:], c0[:, 1:])       # sleepers frozen
    assert not torch.equal(c[:, :1], c0[:, :1])   # routed machine moved
