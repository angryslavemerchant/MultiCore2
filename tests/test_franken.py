"""Correctness tests for the Frankenstein stack: diff attention, canon,
RMSNorm/ReLU2/softcap/untied/zero-init, Muon.

Run:  python -m pytest tests/test_franken.py -q    (CPU is enough)
"""
import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.gated_swa import GatedSWAttention, SlidingWindowAttention
from core.model import GPT, GPTConfig, Canon, config_dict
from core.muon import Muon, zeropower_via_newtonschulz5

torch.manual_seed(0)


def franken_smoke_cfg(**over):
    """Small all-flags-on config: 4 layers FGGF, hourglass, everything."""
    base = dict(block_size=64, vocab_size=256, n_layer=4, n_head=4,
                n_embd=64, bias=False, attn_pattern="FGGF", window=16,
                n_gates=2, recent_band=8, pos="rope", lb_coef=0.01,
                hg_frac=0.5, hg_bneck=3, hg_round=16, mlp="relu2",
                norm="rms", qk_norm=True, diff_attn=True, canon=True,
                softcap=5.0, untied=True, zero_init=True)
    base.update(over)
    return GPTConfig(**base)


# --------------------------------------------------------------- canon
def test_canon_identity_at_init():
    c = Canon(32)
    x = torch.randn(2, 20, 32)
    assert torch.equal(c(x), torch.zeros_like(x))


def test_canon_causal():
    c = Canon(16)
    torch.nn.init.normal_(c.conv.weight)
    x = torch.randn(1, 30, 16)
    y = c(x)
    x2 = x.clone()
    x2[0, 20] += 1.0
    y2 = c(x2)
    assert torch.allclose(y[0, :20], y2[0, :20])          # past unchanged
    assert not torch.allclose(y[0, 20:24], y2[0, 20:24])  # kernel-4 reach
    assert torch.allclose(y[0, 24:], y2[0, 24:])          # beyond reach


# ------------------------------------------------- differential attention
def ref_diff_swa(mod, x, window):
    """Explicit-softmax reference for diff SWA: same module params, dense
    masked math, paper equations."""
    B, T, C = x.shape
    q, k, v = mod._qkv(x)                       # (B,H,T,hd) post-norm+rope
    H, hd = q.shape[1], q.shape[-1]
    idx = torch.arange(T)
    vis = (idx.view(1, T) <= idx.view(T, 1)) & \
          (idx.view(T, 1) < idx.view(1, T) + window)

    def att(qh, kh, vp):
        s = (qh @ kh.transpose(-1, -2)) / math.sqrt(hd)
        s = s.masked_fill(~vis, float("-inf"))
        return s.softmax(-1) @ vp

    vp = (v.view(B, H // 2, 2, T, hd).permute(0, 1, 3, 2, 4)
          .reshape(B, H // 2, T, 2 * hd))
    lam = (torch.exp((mod.lambda_q1 * mod.lambda_k1).sum())
           - torch.exp((mod.lambda_q2 * mod.lambda_k2).sum())
           + mod.lambda_init)
    o = att(q[:, 0::2], k[:, 0::2], vp) - lam * att(q[:, 1::2], k[:, 1::2], vp)
    o = F.rms_norm(o, (o.shape[-1],)) * (1.0 - mod.lambda_init)
    y = o.transpose(1, 2).reshape(B, T, C)
    return mod.c_proj(y)


@pytest.mark.parametrize("window,depth", [(8, 0), (24, 3)])
def test_diff_swa_matches_reference(window, depth):
    cfg = GPTConfig(n_embd=32, n_head=4, window=window, bias=False,
                    pos="rope", qk_norm=True, diff_attn=True)
    mod = SlidingWindowAttention(cfg).eval()
    mod.set_depth(depth)
    x = torch.randn(2, 40, 32)
    with torch.no_grad():
        got = mod(x)
        want = ref_diff_swa(mod, x, window)
    assert torch.allclose(got, want, atol=1e-5), \
        (got - want).abs().max().item()


def test_diff_gated_one_gate_equals_swa():
    """G=1 gated (capacity == window) is exactly a sliding window; with
    identical weights the diff outputs must match too."""
    cfg_s = GPTConfig(n_embd=32, n_head=4, window=16, bias=False,
                      pos="rope", qk_norm=True, diff_attn=True)
    cfg_g = GPTConfig(n_embd=32, n_head=4, window=16, n_gates=1,
                      recent_band=0, bias=False, pos="rope", qk_norm=True,
                      diff_attn=True)
    swa = SlidingWindowAttention(cfg_s).eval()
    gated = GatedSWAttention(cfg_g).eval()
    gated.load_state_dict(swa.state_dict(), strict=False)  # router extra
    swa.set_depth(2), gated.set_depth(2)
    x = torch.randn(2, 48, 32)
    with torch.no_grad():
        assert torch.allclose(swa(x), gated(x), atol=1e-5)


def test_lambda_init_depth_schedule():
    cfg = franken_smoke_cfg()
    model = GPT(cfg)
    lams = [blk.attn.lambda_init for blk in model.transformer.h]
    want = [0.8 - 0.6 * math.exp(-0.3 * i) for i in range(4)]
    assert lams == pytest.approx(want)


# ----------------------------------------------- softcap / untied / inits
def test_softcap_bounds_logits():
    model = GPT(franken_smoke_cfg(untied=False)).eval()  # nonzero head
    idx = torch.randint(0, 256, (2, 32))
    with torch.no_grad():
        logits, _ = model(idx)
    assert logits.abs().max() < 5.0


def test_untied_and_zero_init():
    model = GPT(franken_smoke_cfg())
    assert model.lm_head.weight is not model.transformer.wte.weight
    assert torch.equal(model.lm_head.weight,
                       torch.zeros_like(model.lm_head.weight))
    assert model.transformer.wte.weight.abs().max() > 0
    for name, p in model.named_parameters():
        if name.endswith("c_proj.weight"):
            assert p.abs().max() == 0, name
    # untied wte is a pure lookup: excluded from the 6N FLOPs base
    diff = (model.num_params(non_embedding=False)
            - model.num_params(non_embedding=True))
    assert diff == model.transformer.wte.weight.numel()


def test_config_roundtrip_has_franken_fields():
    d = config_dict(franken_smoke_cfg())
    for f in ("norm", "qk_norm", "diff_attn", "canon", "softcap",
              "untied", "zero_init", "hg_round"):
        assert f in d


# ------------------------------------------------------------------ muon
def test_newtonschulz_orthogonalizes():
    """The tuned quintic lands singular values near 1 (deliberately loose:
    the speedrun coefficients trade exactness for steps)."""
    g = torch.randn(48, 32)
    u = zeropower_via_newtonschulz5(g).float()
    svs = torch.linalg.svdvals(u)
    assert svs.min() > 0.3 and svs.max() < 1.5, svs


def test_muon_reduces_toy_loss():
    torch.manual_seed(1)
    W = torch.nn.Parameter(torch.randn(16, 8) * 0.1)
    x = torch.randn(64, 8)
    y = x @ torch.randn(8, 16)                    # realizable target
    opt = Muon([W], lr=0.05)
    first = None
    for _ in range(30):
        loss = ((x @ W.T - y) ** 2).mean()
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 * first


def test_combo_optimizer_full_model_step():
    model = GPT(franken_smoke_cfg())
    opt = model.configure_optimizers(0.1, 6e-4, (0.9, 0.95), "cpu",
                                     opt="muon", muon_lr=0.02)
    scales = {g.get("lr_scale") for g in opt.param_groups}
    assert scales == {0.02 / 6e-4, 1.0}
    # zero-init head blocks trunk gradients on step 1 (speedrun behavior:
    # the head learns first, everything flows from step 2) — so take two
    idx = torch.randint(0, 256, (2, 33))
    before = model.transformer.h[0].attn.c_attn.weight.clone()
    wte_before = model.transformer.wte.weight.clone()
    for _ in range(2):
        _, loss = model(idx[:, :-1], idx[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert not torch.equal(before,
                           model.transformer.h[0].attn.c_attn.weight)
    assert not torch.equal(wte_before, model.transformer.wte.weight)
    sd = opt.state_dict()
    opt.load_state_dict(sd)                       # resume path round-trips


# ------------------------------------------------------------- end to end
def test_franken_forward_backward_grads_flow():
    # zero_init/untied off: this asserts CONNECTIVITY, and the zero-init
    # head/projections legitimately block trunk gradients at step 1
    model = GPT(franken_smoke_cfg(zero_init=False, untied=False))
    idx = torch.randint(0, 256, (2, 49))
    _, loss = model(idx[:, :-1], idx[:, 1:])
    assert torch.isfinite(loss)
    loss.backward()
    need_grad = ["transformer.wte.weight"]   # tied: lm_head IS wte here
    for name, p in model.named_parameters():
        if any(k in name for k in ("router", "lambda_q1", "canon_a")):
            need_grad.append(name)
    got = {n for n, p in model.named_parameters()
           if p.grad is not None and p.grad.abs().max() > 0}
    missing = [n for n in need_grad if n not in got]
    assert not missing, missing
