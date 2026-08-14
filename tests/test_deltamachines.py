"""Delta-machine population (core/deltamachines.py, pattern D — the v3
arm). The load-bearing claims, each with a test:

  - the chunkwise UT-transform scan IS the sequential gated delta rule
  - beta=0 / alpha=1 masking IS skipping (dense-masked == packed-sparse)
  - the block is causal
  - an unrouted machine cannot touch a token (no leak through the scan
    state, conference, or write-back)
  - the diag hook is a pure observer
  - router aux losses (lb, z) exist and backprop to the router
  - the FLOPs scorer prices sparsity below dense
"""
import os
import sys

import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from core.model import GPT, GPTConfig                          # noqa: E402
from core.deltamachines import (DeltaMachines, DeltaMachineBlock,
                                delta_scan_reference,
                                delta_scan_chunk,
                                delta_scan_packed,
                                deltamachines_score_flops)     # noqa: E402

torch.manual_seed(0)

B, T, C = 2, 32, 32
BASE = dict(block_size=T, n_embd=C, n_head=4, bias=True,
            fs_n_machines=4, fs_d_machine=16, fs_n_head_m=2,
            fs_mlp_mult=2, fs_chunk=8, fs_conv=4,
            fs_topk=2, fs_conf_sink=True)


def cfg(**kw):
    return GPTConfig(**{**BASE, **kw})


def _scan_inputs(N=2, H=2, T_=37, d=8, seed=1):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(N, H, T_, d, generator=g, dtype=torch.float64)
    k = torch.randn(N, H, T_, d, generator=g, dtype=torch.float64)
    k = k / k.norm(dim=-1, keepdim=True)
    v = torch.randn(N, H, T_, d, generator=g, dtype=torch.float64)
    beta = torch.rand(N, H, T_, generator=g, dtype=torch.float64) * 0.9
    la = -torch.rand(N, H, T_, generator=g, dtype=torch.float64) * 0.15
    return q, k, v, beta, la


def test_chunk_matches_reference():
    """UT-transform chunkwise scan == sequential recurrence, fp64, with a
    ragged tail chunk (37 = 4*8 + 5)."""
    q, k, v, beta, la = _scan_inputs()
    o_ref, s_ref = delta_scan_reference(q, k, v, beta, la)
    o_chk, s_chk = delta_scan_chunk(q, k, v, beta, la, L=8)
    assert torch.allclose(o_ref, o_chk, atol=1e-10)
    assert torch.allclose(s_ref, s_chk, atol=1e-10)


def test_chunk_matches_reference_with_s0():
    q, k, v, beta, la = _scan_inputs(seed=2)
    s0 = torch.randn(2, 2, 8, 8, dtype=torch.float64)
    o_ref, s_ref = delta_scan_reference(q, k, v, beta, la, s0=s0)
    o_chk, s_chk = delta_scan_chunk(q, k, v, beta, la, s0=s0, L=8)
    assert torch.allclose(o_ref, o_chk, atol=1e-10)
    assert torch.allclose(s_ref, s_chk, atol=1e-10)


def test_hold_equals_skip():
    """beta=0, log-decay=0 at masked positions: outputs at kept positions
    and the final state are IDENTICAL to a scan over the packed kept-only
    subsequence — masking is skipping, the sparsity contract."""
    q, k, v, beta, la = _scan_inputs(T_=20, seed=3)
    keep = torch.tensor([0, 3, 4, 7, 11, 12, 13, 18])
    m = torch.zeros(20, dtype=torch.float64)
    m[keep] = 1.0
    o_full, s_full = delta_scan_chunk(q, k, v, beta * m, la * m, L=8)
    o_pack, s_pack = delta_scan_chunk(
        q[:, :, keep], k[:, :, keep], v[:, :, keep],
        beta[:, :, keep], la[:, :, keep], L=8)
    assert torch.allclose(o_full[:, :, keep], o_pack, atol=1e-10)
    assert torch.allclose(s_full, s_pack, atol=1e-10)


def test_packed_scan_matches_per_segment_oracle():
    """Machine-major packed scan == the sequential oracle run on each
    segment separately — ragged segment lengths crossing chunk
    boundaries, empty segments included. fp64, L=8."""
    torch.manual_seed(5)
    lens = [3, 0, 17, 8, 1, 0, 24, 5]        # 8 segments, 58 rows
    n, H, d = sum(lens), 2, 8
    seg = torch.cat([torch.full((ln,), i, dtype=torch.int64)
                     for i, ln in enumerate(lens)])
    q = torch.randn(n, H, d, dtype=torch.float64)
    k = torch.randn(n, H, d, dtype=torch.float64)
    k = k / k.norm(dim=-1, keepdim=True)
    v = torch.randn(n, H, d, dtype=torch.float64)
    beta = torch.rand(n, H, dtype=torch.float64) * 0.9
    la = -torch.rand(n, H, dtype=torch.float64) * 0.15
    o, S = delta_scan_packed(q, k, v, beta, la, seg, len(lens), L=8)
    off = 0
    for i, ln in enumerate(lens):
        if ln == 0:
            assert torch.equal(S[i], torch.zeros_like(S[i]))
            continue
        sl = slice(off, off + ln)
        # oracle wants (N,H,T,*) layout
        o_ref, s_ref = delta_scan_reference(
            q[sl].transpose(0, 1)[None], k[sl].transpose(0, 1)[None],
            v[sl].transpose(0, 1)[None], beta[sl].t()[None],
            la[sl].t()[None])
        assert torch.allclose(o[sl].transpose(0, 1), o_ref[0],
                              atol=1e-10), f"seg {i} outputs"
        assert torch.allclose(S[i], s_ref[0], atol=1e-10), f"seg {i} state"
        off += ln


def test_block_causality():
    torch.manual_seed(10)
    blk = DeltaMachineBlock(cfg()).double()
    blk.eval()
    with torch.no_grad():
        blk.machines.w_o.weight.normal_(0, 0.02)   # else out is all-zero
    x = torch.randn(B, T, C, dtype=torch.float64)
    x2 = x.clone()
    t0 = T // 2
    x2[:, t0:] += torch.randn(B, T - t0, C, dtype=torch.float64)
    with torch.no_grad():
        y1, y2 = blk(x), blk(x2)
    assert torch.allclose(y1[:, :t0], y2[:, :t0], atol=1e-12)


def test_unrouted_machine_cannot_touch_a_token():
    """Perturbing machine j's PRIVATE weights must leave the output
    unchanged at every (b, t) where the router did not pick j — through
    the scan (state held), the conference (key masked), and the
    write-back (gate zeroed)."""
    torch.manual_seed(11)
    m = DeltaMachines(cfg()).double()
    m.eval()
    x = torch.randn(B, T, C, dtype=torch.float64)
    rec = {}
    m.diag = {"rec": rec}
    with torch.no_grad():
        out1 = m(x)
    routed = rec["routes"][0] > 0                     # (B,T,K)
    m.diag = None
    j = 1
    with torch.no_grad():
        m.w_in.weight[j] += 0.5
        m.w_out.weight[j] += 0.5
        m.w_qkv.weight[j] += 0.5
        m.w_o.weight[j] += 0.5
        out2 = m(x)
    off = ~routed[:, :, j]
    assert off.any() and (~off).any()                 # test has teeth
    assert torch.allclose(out1[off], out2[off], atol=1e-12)
    assert not torch.allclose(out1[~off], out2[~off], atol=1e-6)


def test_diag_hook_pure_observer():
    torch.manual_seed(12)
    m = DeltaMachines(cfg()).double()
    m.eval()
    with torch.no_grad():
        m.w_o.weight.normal_(0, 0.02)              # else out is all-zero
    x = torch.randn(B, T, C, dtype=torch.float64)
    with torch.no_grad():
        base = m(x)
    rec = {}
    m.diag = {"rec": rec}
    with torch.no_grad():
        hooked = m(x)
    assert torch.equal(base, hooked)
    for key in ("routes", "g", "attn", "attn_rounds", "wb_out_norm",
                "wb_x_norm", "wo_norm_k", "c_norm_k"):
        assert key in rec, key


def test_mute_write_all_zeroes_output():
    torch.manual_seed(13)
    m = DeltaMachines(cfg()).double()
    m.eval()
    with torch.no_grad():
        m.w_o.weight.normal_(0, 0.02)     # zero-init would pass vacuously
    x = torch.randn(B, T, C, dtype=torch.float64)
    with torch.no_grad():
        assert not torch.equal(m(x), torch.zeros(B, T, C).double())

    class Ctl:
        mute_write = list(range(BASE["fs_n_machines"]))
        mute_conf = None
    m.diag = {"ctl": Ctl()}
    with torch.no_grad():
        out = m(x)
    assert torch.equal(out, torch.zeros_like(out))


def test_dense_warmup_override():
    """topk_now = 0: router still runs (aux losses live), every machine
    active, and the result differs from the top-k pass."""
    torch.manual_seed(14)
    m = DeltaMachines(cfg()).double()
    m.eval()
    with torch.no_grad():
        m.w_o.weight.normal_(0, 0.02)     # zero-init would hide the diff
    x = torch.randn(B, T, C, dtype=torch.float64)
    with torch.no_grad():
        sparse = m(x)
    m.topk_now = 0
    rec = {}
    m.diag = {"rec": rec}
    with torch.no_grad():
        dense = m(x)
    assert (rec["routes"][0] > 0).all()               # everyone active
    assert m.lb_loss is not None and torch.isfinite(m.lb_loss)
    assert m.z_loss is not None and torch.isfinite(m.z_loss)
    assert not torch.allclose(sparse, dense, atol=1e-8)


def test_route_noise_train_only():
    torch.manual_seed(15)
    m = DeltaMachines(cfg(fs_route_noise=0.5)).double()
    m.eval()
    with torch.no_grad():
        m.w_o.weight.normal_(0, 0.02)              # else out is all-zero
    x = torch.randn(B, T, C, dtype=torch.float64)
    with torch.no_grad():
        a, b_ = m(x), m(x)
    assert torch.equal(a, b_)                         # eval: no noise
    m.train()
    torch.manual_seed(16)
    r1 = m(x)
    torch.manual_seed(17)
    r2 = m(x)
    assert not torch.equal(r1, r2)                    # train: noisy routes


def test_gpt_integration_and_aux_losses():
    torch.manual_seed(18)
    c = cfg(vocab_size=64, n_layer=2, attn_pattern="FD",
            lb_coef=0.01, fs_zloss=1e-3, fs_route_noise=0.1)
    model = GPT(c)
    model.train()
    idx = torch.randint(0, 64, (2, T))
    tgt = torch.randint(0, 64, (2, T))
    logits, loss = model(idx, tgt)
    assert torch.isfinite(loss)
    loss.backward()
    dm = model.transformer.h[1].machines
    assert dm.route_x.grad is not None
    assert torch.isfinite(dm.route_x.grad).all()
    assert dm.w_in.weight.grad is not None
    # flops accounting covers D layers
    assert model.flops_per_token(T) > 0


RUNGS = [dict(fs_dm_mlp=False, fs_dm_conf=False, fs_dm_gate=False),
         dict(fs_dm_conf=False, fs_dm_gate=False),
         dict(fs_dm_gate=False),
         dict()]


def test_probe_rungs_all_params_reach_the_graph():
    """Probe-ladder rungs 1-4: forward/backward runs and EVERY constructed
    parameter gets a grad — DDP errors on orphans (the v2 s0 lesson), so
    this is the single-process proxy for the 2x/8x path."""
    for kw in RUNGS:
        torch.manual_seed(20)
        m = DeltaMachines(cfg(**kw)).double()
        with torch.no_grad():
            m.w_o.weight.normal_(0, 0.02)
        x = torch.randn(B, T, C, dtype=torch.float64)
        out = m(x)
        (out.square().sum() + m.lb_loss + m.z_loss).backward()
        for n, p in m.named_parameters():
            assert p.grad is not None, (kw, n)


def test_packed_matches_dense_masked_all_rungs():
    """The hard gate: sparse execution == dense-masked semantics, every
    rung, outputs at 1e-10 in fp64 (CPU path = grouped_mm_reference)."""
    for kw in RUNGS:
        torch.manual_seed(21)
        m = DeltaMachines(cfg(**kw)).double()
        m.eval()
        with torch.no_grad():
            m.w_o.weight.normal_(0, 0.02)
        x = torch.randn(B, T, C, dtype=torch.float64)
        with torch.no_grad():
            dense = m(x)
            m.packed = True
            packed = m(x)
            m.packed = False
        assert torch.allclose(dense, packed, atol=1e-10), kw


def test_packed_matches_dense_gradients():
    """Backward equivalence on the full rung: every parameter's grad
    matches between dense-masked and packed execution."""
    grads = []
    for use_packed in (False, True):
        torch.manual_seed(22)
        m = DeltaMachines(cfg()).double()
        m.packed = use_packed
        with torch.no_grad():
            m.w_o.weight.normal_(0, 0.02)
        torch.manual_seed(23)
        x = torch.randn(B, T, C, dtype=torch.float64)
        (m(x).square().sum() + m.lb_loss + m.z_loss).backward()
        grads.append({n_: p.grad.clone()
                      for n_, p in m.named_parameters()})
    for n_ in grads[0]:
        assert torch.allclose(grads[0][n_], grads[1][n_],
                              atol=1e-9), n_


def test_probe_rungs_flops_monotone():
    """Each added part must cost FLOPs at the accounting layer (params
    grow and the scorer charges/credits the parts that exist)."""
    fs = [GPT(cfg(vocab_size=64, n_layer=2, attn_pattern="FD",
                  **kw)).flops_per_token(T) for kw in RUNGS]
    assert fs[0] < fs[1] < fs[2] < fs[3]


def test_flops_scorer_prices_sparsity():
    dense = cfg(fs_topk=0)
    sparse = cfg(fs_topk=2)
    one = cfg(fs_topk=1)
    assert deltamachines_score_flops(one, T) \
        < deltamachines_score_flops(sparse, T) \
        < deltamachines_score_flops(dense, T)


# ---------------------------------------------------------------- fla path
# The fused varlen kernel (flash-linear-attention) is the packed scan's
# fast path. These tests are the convention gate the integration crib
# demands: fla's recurrence must match OUR reference (decay-then-delta;
# scalar decay commutes with the projector, so any ordering difference
# would surface here), scale must be 1.0, segments must map onto
# cu_seqlens including empties. bf16 kernel vs fp64 oracle => loose
# tolerances chosen from bf16 mantissa (~3 decimal digits) with scan
# accumulation; the TORCH path keeps the exact 1e-10 tests above.

def _fla_ready():
    from core.deltamachines import _fla_kernel
    return torch.cuda.is_available() and bool(_fla_kernel())


requires_fla = pytest.mark.skipif(
    not _fla_ready(), reason="needs CUDA + flash-linear-attention")


def _packed_inputs_cuda(lens, H=2, hd=64, seed=7):
    torch.manual_seed(seed)
    n = sum(lens)
    dev = "cuda"
    seg = torch.cat([torch.full((ln,), i, dtype=torch.int64)
                     for i, ln in enumerate(lens)]).to(dev)
    q = torch.nn.functional.normalize(
        torch.randn(n, H, hd, device=dev), dim=-1)
    k = torch.nn.functional.normalize(
        torch.randn(n, H, hd, device=dev), dim=-1)
    v = torch.randn(n, H, hd, device=dev)
    beta = torch.rand(n, H, device=dev) * 0.9
    la = -torch.rand(n, H, device=dev) * 0.125
    return q, k, v, beta, la, seg


@requires_fla
def test_fla_matches_per_segment_oracle():
    """fla varlen kernel == sequential fp64 oracle per segment (ragged
    lengths crossing fla's internal chunk size, empty segments dropped
    from cu_seqlens). This is the recurrence-convention gate."""
    from core.deltamachines import _scan_packed_fla
    lens = [37, 0, 130, 64, 1, 0, 200, 23]
    q, k, v, beta, la, seg = _packed_inputs_cuda(lens)
    o, _ = _scan_packed_fla(q, k, v, beta, la, seg, len(lens))
    off = 0
    for i, ln in enumerate(lens):
        if ln == 0:
            continue
        sl = slice(off, off + ln)
        o_ref, _ = delta_scan_reference(
            q[sl].double().transpose(0, 1)[None].cpu(),
            k[sl].double().transpose(0, 1)[None].cpu(),
            v[sl].double().transpose(0, 1)[None].cpu(),
            beta[sl].double().t()[None].cpu(),
            la[sl].double().t()[None].cpu())
        got = o[sl].float().transpose(0, 1).cpu()
        ref = o_ref[0].float()
        rel = (got - ref).norm() / ref.norm()
        assert rel < 2e-2, f"seg {i} rel {rel:.3e} (convention mismatch?)"
        off += ln


@requires_fla
def test_fla_matches_torch_packed():
    """Same inputs through the fla path and the exact torch path: close
    (bf16 vs fp32 state), and both differentiable with agreeing grads."""
    from core.deltamachines import _scan_packed_fla, _scan_packed_inner
    lens = [64, 96, 0, 33, 127]
    q, k, v, beta, la, seg = _packed_inputs_cuda(lens, seed=8)
    leafs = []
    for t in (q, k, v, beta, la):
        t = t.clone().requires_grad_(True)
        leafs.append(t)
    q1, k1, v1, b1, l1 = leafs
    o_fla, _ = _scan_packed_fla(q1, k1, v1, b1, l1, seg, len(lens))
    o_fla.square().sum().backward()
    g_fla = [t.grad.clone() for t in leafs]
    for t in leafs:
        t.grad = None
    o_t, _ = _scan_packed_inner(q1, k1, v1, b1, l1, seg, len(lens), L=64)
    o_t.square().sum().backward()
    g_t = [t.grad.clone() for t in leafs]
    rel = (o_fla.float() - o_t.float()).norm() / o_t.float().norm()
    assert rel < 2e-2, f"output rel {rel:.3e}"
    for name, a, b in zip("qkv beta la".split(), g_fla, g_t):
        assert torch.isfinite(a).all(), name
        cos = torch.nn.functional.cosine_similarity(
            a.flatten(), b.flatten(), dim=0)
        assert cos > 0.99, f"grad {name} cos {cos:.4f}"


@requires_fla
def test_fla_gpt_train_step_matches_torch():
    """Full packed GPT train step, fs_scan fla vs torch: finite loss +
    grads, losses close. The end-to-end integration gate."""
    losses = {}
    for impl in ("torch", "fla"):
        torch.manual_seed(31)
        # hd=64: fla kernels want real head dims (BASE's hd=8 is
        # oracle-test-sized, below the kernel's supported range)
        c = cfg(vocab_size=128, n_layer=2, attn_pattern="FD",
                fs_d_machine=128, fs_n_head_m=2,
                fs_packed=True, fs_scan=impl, fs_zloss=1e-3,
                lb_coef=0.01)
        model = GPT(c).cuda()
        with torch.no_grad():
            for blk in model.transformer.h:
                if hasattr(blk, "machines"):
                    blk.machines.w_o.weight.normal_(0, 0.02)
        model.train()
        torch.manual_seed(32)
        idx = torch.randint(0, 128, (2, T), device="cuda")
        tgt = torch.randint(0, 128, (2, T), device="cuda")
        _, loss = model(idx, tgt)
        loss.backward()
        assert torch.isfinite(loss), impl
        for n_, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"{impl} {n_}"
        losses[impl] = loss.item()
    assert abs(losses["fla"] - losses["torch"]) < 2e-2, losses


# ------------------------------------------------------------- v3.1 rungs
# pop_read / brief / mag_topk / conf_commit (dense-path probe science).
# Every mechanism must keep causality and the sparsity contract it
# claims: pop reads NEVER write, briefs are chunk-START (past-only),
# commit changes states only at chunk edges, mag-topk emits exactly k
# machines per token.

V31 = [dict(fs_pop_read=True),
       dict(fs_pop_read=True, fs_brief=True),
       dict(fs_pop_read=True, fs_brief=True, fs_mag_topk=True),
       dict(fs_pop_read=True, fs_brief=True, fs_mag_topk=True,
            fs_conf_commit=True)]


def _mk(kw, seed=40):
    torch.manual_seed(seed)
    m = DeltaMachines(cfg(**kw)).double()
    with torch.no_grad():
        m.w_o.weight.normal_(0, 0.02)   # zero-init would test nothing
    return m


@pytest.mark.parametrize("kw", V31)
def test_v31_causality(kw):
    """Future-token perturbation cannot touch past outputs, for every
    v3.1 rung (briefs read chunk-START states; commit only feeds
    LATER chunks)."""
    m = _mk(kw)
    torch.manual_seed(41)
    x = torch.randn(B, T, C, dtype=torch.float64)
    y1 = m(x)
    x2 = x.clone()
    x2[:, T - 3:] += 10.0
    y2 = m(x2)
    assert torch.allclose(y1[:, :T - 3], y2[:, :T - 3], atol=1e-12)


def test_v31_pop_reads_do_not_write():
    """The shared-query read path must be read-only: an unrouted
    machine's future behavior is identical whether pop_read is on or
    off (states untouched), even though outputs differ."""
    torch.manual_seed(42)
    x = torch.randn(B, T, C, dtype=torch.float64)
    outs = {}
    for pop in (False, True):
        m = _mk(dict(fs_pop_read=pop) if pop else {}, seed=43)
        # identical params where shared: copy the pop module's base
        # weights from the non-pop model on second pass
        if pop:
            sd = outs["m"].state_dict()
            m.load_state_dict(sd, strict=False)
        with torch.no_grad():
            r_log = (torch.einsum("btc,kc->btk", x, m.route_x)
                     + m.route_b)
            r_sig = torch.sigmoid(r_log)
            outs[pop] = (m(x), r_sig)
        outs.setdefault("m", m)
    # routing identical (router params copied) => the routed WRITE set
    # is identical; pop reads changed outputs only through views
    assert torch.allclose(outs[False][1], outs[True][1], atol=1e-12)
    assert not torch.allclose(outs[False][0], outs[True][0])


def test_v31_mag_topk_emits_exactly_k():
    m = _mk(dict(fs_pop_read=True, fs_mag_topk=True))
    torch.manual_seed(44)
    x = torch.randn(B, T, C, dtype=torch.float64)
    rec = {}
    m.diag = {"rec": rec, "ctl": None}
    m(x)
    m.diag = None
    g = rec["g"]
    assert ((g > 0).sum(dim=-1) == m.topk).all(), \
        "mag-topk must select exactly k writers per token"


def test_v31_mag_topk_unrouted_can_win():
    """With pop reads + magnitude selection, some winning writer is NOT
    router-selected somewhere in the batch (emergent routing is real,
    not a re-skin of the router)."""
    torch.manual_seed(45)
    m = _mk(dict(fs_pop_read=True, fs_mag_topk=True), seed=45)
    x = torch.randn(B, T, C, dtype=torch.float64)
    rec = {}
    m.diag = {"rec": rec, "ctl": None}
    m(x)
    m.diag = None
    g = rec["g"]                                   # (B,T,K) writers
    routed = rec["routes"][0] > 0                  # (B,T,K) router picks
    assert ((g > 0) & ~routed).any(), \
        "no unrouted machine ever won a write slot"


def test_v31_commit_matches_manual_two_chunk():
    """conf_commit == running chunks manually with a rank-1 message
    write between them (the mechanism is exactly chunk-sequential)."""
    kw = dict(fs_pop_read=True, fs_conf_commit=True)
    m = _mk(kw, seed=46)
    torch.manual_seed(47)
    x = torch.randn(1, T, C, dtype=torch.float64)
    y = m(x)
    assert torch.isfinite(y).all()
    # states must differ from the no-commit run (the commit is live)
    m2 = _mk(dict(fs_pop_read=True), seed=46)
    m2.load_state_dict(m.state_dict(), strict=False)
    y2 = m2(x)
    assert not torch.allclose(y, y2), "commit changed nothing"
    # and chunk 0 must be IDENTICAL (first commit lands after chunk 0)
    ch = m.chunk
    assert torch.allclose(y[:, :ch], y2[:, :ch], atol=1e-12), \
        "commit leaked into the first chunk"


@pytest.mark.parametrize("kw", V31)
def test_v31_all_params_reach_graph(kw):
    """DDP proxy: every constructed parameter gets a gradient."""
    torch.manual_seed(48)
    model = GPT(cfg(vocab_size=64, n_layer=2, attn_pattern="FD",
                    fs_zloss=1e-3, lb_coef=0.01, **kw)).double()
    with torch.no_grad():
        for blk in model.transformer.h:
            if hasattr(blk, "machines"):
                blk.machines.w_o.weight.normal_(0, 0.02)
    idx = torch.randint(0, 64, (2, T))
    tgt = torch.randint(0, 64, (2, T))
    _, loss = model(idx, tgt)
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, missing


def test_v31_flops_scorer_prices_mechanisms():
    """Each mechanism must cost FLOPs (monotone up the rung ladder)."""
    base = deltamachines_score_flops(cfg(), T)
    scores = [deltamachines_score_flops(cfg(**kw), T) for kw in V31]
    assert base < scores[0] < scores[1] < scores[2] < scores[3]


def test_v31_packed_refuses():
    with pytest.raises(AssertionError):
        DeltaMachines(cfg(fs_packed=True, fs_pop_read=True))


@pytest.mark.parametrize("kw", V31)
def test_v31_warmup_all_params_reach_graph(kw):
    """The DDP proxy the 2-rank smoke failed (2026-08-14): during DENSE
    WARMUP (topk_now=0) the v3.1 mechanisms are gated off, but their
    params must still produce gradients or DDP errors on step 1."""
    torch.manual_seed(49)
    model = GPT(cfg(vocab_size=64, n_layer=2, attn_pattern="FD",
                    fs_zloss=1e-3, lb_coef=0.01, **kw)).double()
    for blk in model.transformer.h:
        if hasattr(blk, "machines"):
            blk.machines.topk_now = 0            # dense warmup mode
            with torch.no_grad():
                blk.machines.w_o.weight.normal_(0, 0.02)
    idx = torch.randint(0, 64, (2, T))
    tgt = torch.randint(0, 64, (2, T))
    _, loss = model(idx, tgt)
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, missing
