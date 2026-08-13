"""The diagnostic scripts' stat pipelines, fed by the REAL forward's
diag hook (never a reimplemented forward — the 2026-08-13 lesson).
Covers the key contract: every rec key the scripts consume is a key the
hook actually writes, on a v2 config."""
import os
import sys

import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))
from core.model import GPTConfig                                # noqa: E402
from core.fourstroke import FourStrokeBlock                     # noqa: E402
from diag_fourstroke_gates import consume, finalize             # noqa: E402

B, T, C = 2, 8, 32
V2 = dict(block_size=T, n_embd=C, n_head=4, bias=True,
          fs_n_machines=3, fs_d_machine=16, fs_n_head_m=2,
          fs_mlp_mult=2, fs_backend="attn", fs_window=4,
          fs_topk=2, fs_loop_rounds=2, fs_loop_topk=2,
          fs_conf_sink=True, fs_sparse_state=True)


def test_gates_pipeline_on_real_hook():
    torch.manual_seed(81)
    blk = FourStrokeBlock(GPTConfig(**V2)).double()
    hook, rec = {}, {}
    blk.strokes.diag = {"rec": hook}
    torch.manual_seed(82)
    x = torch.randn(B, T, C, dtype=torch.float64)
    c0 = blk.strokes.init_channel(B, T, dtype=torch.float64)
    with torch.no_grad():
        for _ in range(2):                       # two batches accumulate
            blk(x, c0)
            consume(hook, rec, blk.strokes.K)
    out = finalize(rec, blk.strokes)
    K = blk.strokes.K
    assert len(out["gate_mean"]) == K
    assert all(0.0 <= v <= 1.0 for v in out["gate_mean"])
    assert 0.0 < out["route_top_mass"]["2"] <= 1.0 + 1e-9
    # top-2-of-3 masked gates: top-2 mass must be exactly 1
    assert abs(out["route_top_mass"]["2"] - 1.0) < 1e-9
    assert len(out["conf_entropy"]) == K
    assert all(e <= out["conf_entropy_uniform"] + 1e-6
               for e in out["conf_entropy"])
    assert len(out["route_traffic_per_round"]) == V2["fs_loop_rounds"]
    assert all(abs(sum(r) - V2["fs_topk"]) < 1e-6
               for r in out["route_traffic_per_round"])  # top-k per token
    assert len(out["route_wake_frac"]) == K
    assert "writeback_over_x" in out and "channel_norm_per_machine" in out


def test_hook_keys_match_consume_contract():
    """Every key consume() pops must be written by one hooked forward."""
    torch.manual_seed(83)
    blk = FourStrokeBlock(GPTConfig(**V2)).double()
    hook = {}
    blk.strokes.diag = {"rec": hook}
    x = torch.randn(B, T, C, dtype=torch.float64)
    c0 = blk.strokes.init_channel(B, T, dtype=torch.float64)
    with torch.no_grad():
        blk(x, c0)
    for key in ("g", "attn", "attn_rounds", "wb_out_norm", "wb_x_norm",
                "wo_norm_k", "c_norm_k", "routes"):
        assert key in hook, key
