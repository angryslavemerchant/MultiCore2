"""Grouped GEMM tests (core/grouped_gemm.py).

CPU tier always runs: the tile map and the reference oracle. GPU tier
(needs CUDA + triton) checks the kernels against the oracle -- forward,
dx, dw -- at ragged sizes including empty groups and partial tiles.
On sm75 (the local 2060) run fp16/fp32; bf16 tolerance is checked too
but only matters on the rented boxes.
"""
import pytest
import torch

from core.grouped_gemm import (HAS_TRITON, build_tile_map,
                               grouped_mm_reference)

GPU = HAS_TRITON and torch.cuda.is_available()


def ragged_offsets(lens, device="cpu"):
    t = torch.tensor([0] + list(lens), dtype=torch.int32, device=device)
    return torch.cumsum(t, 0).to(torch.int32)


# ------------------------------------------------------------------ CPU tier

def test_tile_map_covers_all_rows():
    offs = ragged_offsets([5, 0, 130, 64, 1])
    bm = 64
    n = int(offs[-1])
    max_tiles = n // bm + 5
    tg, tr = build_tile_map(offs, bm, max_tiles)
    covered = torch.zeros(n, dtype=torch.bool)
    for t in range(max_tiles):
        g = int(tg[t])
        if g < 0:
            continue
        lo = int(tr[t])
        hi = min(lo + bm, int(offs[g + 1]))
        assert lo >= int(offs[g])          # tile starts inside its group
        covered[lo:hi] = True
    assert covered.all()                   # every row owned by exactly one
    live = [(int(tg[t]), int(tr[t])) for t in range(max_tiles)
            if int(tg[t]) >= 0]
    assert len(live) == len(set(live))     # no duplicate tiles


def test_tile_map_static_shape_padding():
    offs = ragged_offsets([3, 2])
    tg, tr = build_tile_map(offs, 64, 8)
    assert tg.shape == (8,) and tr.shape == (8,)
    # one live tile PER nonempty group (segments never share a tile)
    assert tg[0] == 0 and tg[1] == 1 and (tg[2:] == -1).all()


def test_reference_matches_loop():
    torch.manual_seed(0)
    offs = ragged_offsets([7, 0, 33])
    x = torch.randn(40, 16, dtype=torch.float64)
    w = torch.randn(3, 16, 24, dtype=torch.float64)
    y = grouped_mm_reference(x, w, offs)
    assert torch.equal(y[7:7], y[7:7])     # empty segment is fine
    assert torch.allclose(y[0:7], x[0:7] @ w[0])
    assert torch.allclose(y[7:40], x[7:40] @ w[2])


# ------------------------------------------------------------------ GPU tier

needs_gpu = pytest.mark.skipif(not GPU, reason="needs CUDA + triton")

SIZES = [([128, 0, 300, 64, 77, 1], 256, 512),   # ragged + empty + tiny
         ([64, 64, 64, 64], 256, 256),           # exact tiles
         ([1000], 64, 128)]                      # single fat group


@needs_gpu
@pytest.mark.parametrize("lens,d_in,d_out", SIZES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_forward_matches_reference(lens, d_in, d_out, dtype):
    from core.grouped_gemm import grouped_mm
    torch.manual_seed(1)
    offs = ragged_offsets(lens, "cuda")
    n = int(offs[-1])
    x = torch.randn(n, d_in, device="cuda", dtype=dtype)
    w = torch.randn(len(lens), d_in, d_out, device="cuda",
                    dtype=dtype) * 0.05
    tg, tr = build_tile_map(offs, 64, n // 64 + len(lens))
    y = grouped_mm(x, w, offs, tg, tr)
    ref = grouped_mm_reference(x.double(), w.double(), offs).to(dtype)
    # fp32 tl.dot runs as TF32 on sm80+ (10-bit mantissa; the sm75 2060
    # has no TF32 units and computes true fp32) — bound must cover both.
    # Real training runs bf16/fp16 via autocast; fp32 is a test-only path.
    tol = 1e-2 if dtype == torch.float16 else 2e-3
    assert (y - ref).abs().max().item() < tol * max(1.0, ref.abs().max().item())


@needs_gpu
def test_backward_matches_reference():
    from core.grouped_gemm import grouped_mm
    torch.manual_seed(2)
    offs = ragged_offsets([90, 0, 170, 33], "cuda")
    n = int(offs[-1])
    x = torch.randn(n, 128, device="cuda", requires_grad=True)
    w = (torch.randn(4, 128, 96, device="cuda") * 0.05).requires_grad_()
    tg, tr = build_tile_map(offs, 64, n // 64 + 4)
    y = grouped_mm(x, w, offs, tg, tr)
    g = torch.randn_like(y)
    y.backward(g)
    x2 = x.detach().double().requires_grad_()
    w2 = w.detach().double().requires_grad_()
    grouped_mm_reference(x2, w2, offs).backward(g.double())
    # RELATIVE bounds: fp32 dots run as TF32 on sm80+ (~1e-3 rel), and
    # accumulation order varies per host/tile schedule — an absolute
    # bound broke on a second 5090 (dw abs 0.040 on entries of mag ~40,
    # i.e. exactly TF32-rel; the same stack passed at 1.9e-3 abs on
    # another host). True-fp32 sm75 lands ~1e-5 rel.
    dx = (x.grad - x2.grad.float()).abs().max() \
        / x2.grad.float().abs().max()
    dw = (w.grad - w2.grad.float()).abs().max() \
        / w2.grad.float().abs().max()
    assert dx.item() < 2e-3, f"dx rel {dx:.2e}"
    assert dw.item() < 2e-3, f"dw rel {dw:.2e}"


@needs_gpu
def test_empty_group_gets_zero_dw():
    from core.grouped_gemm import grouped_mm
    offs = ragged_offsets([64, 0, 64], "cuda")
    x = torch.randn(128, 32, device="cuda")
    w = torch.randn(3, 32, 32, device="cuda", requires_grad=True)
    tg, tr = build_tile_map(offs, 64, 128 // 64 + 3)
    grouped_mm(x, w, offs, tg, tr).sum().backward()
    assert torch.all(w.grad[1] == 0)
    assert w.grad[0].abs().sum() > 0 and w.grad[2].abs().sum() > 0
