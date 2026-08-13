"""Grouped GEMM for per-machine (per-expert) linear layers (Triton).

The dispatch problem: K machines each own a (d_in, d_out) weight; every
token is routed to top-k machines. Dense-masked runs all K at 4x waste;
FCFS capacity buffers (fs_capacity) cut the FLOPs but leave 16 small
GEMMs that run at poor efficiency and drop overflow tokens. This module
is the third rung: ALL routed (token, machine) rows live in ONE flat
buffer sorted by machine (dropless, ragged segments), and one kernel
launch covers every segment -- machine 3's last tile and machine 4's
first tile run concurrently on different SMs. The only waste is each
segment's final partial tile (the irreducible granule tax).

Layout contract:
    x        (N, d_in)   rows sorted by group; N = B*T*topk is STATIC
                         for top-k routing (only segment lengths vary)
    w        (K, d_in, d_out)
    offsets  (K+1,) int32  row range of group g = [offsets[g], offsets[g+1])
    y        (N, d_out)

Tile map (host side, plain torch ops, static shape when max_tiles given):
    tile_group (MAX_TILES,) int32   group id of launch tile, -1 = padding
    tile_row   (MAX_TILES,) int32   absolute starting row of the tile
A group of L rows owns ceil(L / BLOCK_M) tiles; MAX_TILES = N//BLOCK_M + K
upper-bounds the total (each group adds at most one partial tile), so the
launch grid is shape-static and torch.compile-friendly: routing changes
tensor CONTENTS, never tensor shapes.

Backward is two more grouped GEMMs over the same segments:
    dx = dy @ w[g]^T          (same kernel, transposed weights)
    dw[g] = x_g^T @ dy_g      (second kernel, grid over groups x weight
                               tiles, ragged reduction over the segment)

Accumulation is always fp32 regardless of input dtype.

Triton is imported lazily: importing this module never requires a GPU,
and grouped_mm_reference() is pure PyTorch for CPU tests / fallback.
"""
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:                                       # CPU-only box
    HAS_TRITON = False


# --------------------------------------------------------------- host side

def build_tile_map(offsets, block_m, max_tiles):
    """Map linear launch-tile ids to (group, starting row). Pure torch,
    output shapes depend only on max_tiles (static under compile)."""
    dev = offsets.device
    lens = (offsets[1:] - offsets[:-1]).to(torch.int64)
    ntiles = (lens + block_m - 1) // block_m              # tiles per group
    starts = torch.cumsum(ntiles, 0) - ntiles             # first tile id
    tile_group = torch.full((max_tiles,), -1,
                            dtype=torch.int32, device=dev)
    tile_row = torch.zeros(max_tiles, dtype=torch.int32, device=dev)
    K = lens.numel()
    # scatter each group's tile range; K is tiny (16) so a host loop over
    # groups would sync -- build with a flat arange instead.
    tid = torch.arange(max_tiles, device=dev)
    # group of tile t = searchsorted(cumsum(ntiles), t, right) for t < total
    total = ntiles.sum()
    cum = torch.cumsum(ntiles, 0)
    g_of = torch.searchsorted(cum, tid, right=True).to(torch.int32)
    live = tid < total
    g_live = torch.where(live, g_of, torch.zeros_like(g_of))
    within = tid - starts.gather(0, g_live.to(torch.int64))
    row = (offsets[:-1].gather(0, g_live.to(torch.int64)).to(torch.int64)
           + within * block_m)
    tile_group = torch.where(live, g_live,
                             torch.full_like(g_live, -1))
    tile_row = torch.where(live, row, torch.zeros_like(row)).to(torch.int32)
    return tile_group, tile_row


def grouped_mm_reference(x, w, offsets):
    """Segment-by-segment torch.mm; the correctness oracle."""
    y = x.new_zeros(x.shape[0], w.shape[2])
    for g in range(w.shape[0]):
        a, b = int(offsets[g]), int(offsets[g + 1])
        if b > a:
            y[a:b] = x[a:b] @ w[g]
    return y


# ----------------------------------------------------------------- kernels

if HAS_TRITON:

    # Autotune over tile shapes: our segments are ~1-5k rows x 256-1024
    # features, small enough that the best tile is not the textbook one.
    # BLOCK_M is FIXED at 64 -- the host-side tile map is built with
    # block_m=64 and the kernel's row tiling must agree with it.
    _MM_CONFIGS = [
        triton.Config({"BLOCK_N": bn, "BLOCK_K": bk},
                      num_warps=w, num_stages=s)
        for bn in (64, 128) for bk in (32, 64)
        for w, s in ((4, 2), (4, 3), (8, 3))
    ]

    @triton.autotune(configs=_MM_CONFIGS, key=["d_in", "d_out"])
    @triton.jit
    def _grouped_mm_kernel(
        x_ptr, w_ptr, y_ptr, tile_group_ptr, tile_row_ptr, offs_ptr,
        d_in, d_out,
        sxm, sxk, swg, swk, swn, sym, syn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        g = tl.load(tile_group_ptr + pid_m)
        if g < 0:                                         # padding tile
            return
        row0 = tl.load(tile_row_ptr + pid_m)
        row_end = tl.load(offs_ptr + g + 1)
        rows = row0 + tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rmask = rows < row_end
        cmask = cols < d_out
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, d_in, BLOCK_K):
            ks = k0 + tl.arange(0, BLOCK_K)
            kmask = ks < d_in
            xt = tl.load(x_ptr + rows[:, None] * sxm + ks[None, :] * sxk,
                         mask=rmask[:, None] & kmask[None, :], other=0.0)
            wt = tl.load(w_ptr + g * swg + ks[:, None] * swk
                         + cols[None, :] * swn,
                         mask=kmask[:, None] & cmask[None, :], other=0.0)
            acc = tl.dot(xt, wt, acc)
        tl.store(y_ptr + rows[:, None] * sym + cols[None, :] * syn,
                 acc.to(y_ptr.dtype.element_ty),
                 mask=rmask[:, None] & cmask[None, :])

    _DW_CONFIGS = [
        triton.Config({"BLOCK_I": bi, "BLOCK_O": bo, "BLOCK_R": br},
                      num_warps=w, num_stages=s)
        for bi, bo in ((64, 64), (64, 128), (128, 64))
        for br in (32, 64)
        for w, s in ((4, 2), (8, 3))
    ]

    @triton.autotune(configs=_DW_CONFIGS, key=["d_in", "d_out"])
    @triton.jit
    def _grouped_dw_kernel(
        x_ptr, dy_ptr, dw_ptr, offs_ptr,
        d_in, d_out,
        sxm, sxk, sym, syn, swg, swk, swn,
        BLOCK_I: tl.constexpr, BLOCK_O: tl.constexpr, BLOCK_R: tl.constexpr,
    ):
        g = tl.program_id(0)
        pid_i = tl.program_id(1)
        pid_o = tl.program_id(2)
        row0 = tl.load(offs_ptr + g)
        row_end = tl.load(offs_ptr + g + 1)
        ins = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
        outs = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
        imask = ins < d_in
        omask = outs < d_out
        acc = tl.zeros((BLOCK_I, BLOCK_O), dtype=tl.float32)
        for r0 in range(row0, row_end, BLOCK_R):
            rs = r0 + tl.arange(0, BLOCK_R)
            rmask = rs < row_end
            xt = tl.load(x_ptr + rs[:, None] * sxm + ins[None, :] * sxk,
                         mask=rmask[:, None] & imask[None, :], other=0.0)
            dyt = tl.load(dy_ptr + rs[:, None] * sym + outs[None, :] * syn,
                          mask=rmask[:, None] & omask[None, :], other=0.0)
            acc = tl.dot(tl.trans(xt), dyt, acc)
        tl.store(dw_ptr + g * swg + ins[:, None] * swk + outs[None, :] * swn,
                 acc.to(dw_ptr.dtype.element_ty),
                 mask=imask[:, None] & omask[None, :])

    def _launch_mm(x, w, offsets, tile_group, tile_row, block_m=64):
        N, d_in = x.shape
        d_out = w.shape[2]
        y = x.new_empty(N, d_out)

        def grid(meta):
            return (tile_group.numel(),
                    triton.cdiv(d_out, meta["BLOCK_N"]))

        _grouped_mm_kernel[grid](
            x, w, y, tile_group, tile_row, offsets,
            d_in, d_out,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1), w.stride(2),
            y.stride(0), y.stride(1),
            BLOCK_M=block_m)
        return y

    def _launch_dw(x, dy, offsets, K):
        d_in, d_out = x.shape[1], dy.shape[1]
        dw = x.new_zeros(K, d_in, d_out)

        def grid(meta):
            return (K, triton.cdiv(d_in, meta["BLOCK_I"]),
                    triton.cdiv(d_out, meta["BLOCK_O"]))

        _grouped_dw_kernel[grid](
            x, dy, dw, offsets,
            d_in, d_out,
            x.stride(0), x.stride(1),
            dy.stride(0), dy.stride(1),
            dw.stride(0), dw.stride(1), dw.stride(2))
        return dw


if HAS_TRITON:
    # Registered as torch custom ops so torch.compile treats the launches
    # as OPAQUE graph nodes: dynamo must neither trace into the triton
    # source (it can't -- verified NaN corruption on 2026-08-13 when it
    # tried) nor graph-break around every launch (48 breaks per step).
    # register_fake gives inductor the shape/dtype contract it needs to
    # keep compiling the surrounding graph.

    @torch.library.custom_op("multicore::grouped_mm", mutates_args=())
    def _op_mm(x: torch.Tensor, w: torch.Tensor, offsets: torch.Tensor,
               tile_group: torch.Tensor,
               tile_row: torch.Tensor) -> torch.Tensor:
        return _launch_mm(x, w, offsets, tile_group, tile_row)

    @_op_mm.register_fake
    def _op_mm_fake(x, w, offsets, tile_group, tile_row):
        return x.new_empty(x.shape[0], w.shape[2])

    @torch.library.custom_op("multicore::grouped_dw", mutates_args=())
    def _op_dw(x: torch.Tensor, dy: torch.Tensor, offsets: torch.Tensor,
               n_groups: int) -> torch.Tensor:
        return _launch_dw(x, dy, offsets, n_groups)

    @_op_dw.register_fake
    def _op_dw_fake(x, dy, offsets, n_groups):
        return x.new_empty(n_groups, x.shape[1], dy.shape[1])

    def _setup_ctx(ctx, inputs, output):
        x, w, offsets, tile_group, tile_row = inputs
        ctx.save_for_backward(x, w, offsets, tile_group, tile_row)

    def _backward(ctx, dy):
        x, w, offsets, tile_group, tile_row = ctx.saved_tensors
        dy = dy.contiguous()
        # dx = dy @ w^T -- same segments, same tile map; the transpose
        # copy is K*d_in*d_out*2B (~8 MB at our shapes), negligible.
        dx = _op_mm(dy, w.transpose(1, 2).contiguous(), offsets,
                    tile_group, tile_row)
        dw = _op_dw(x, dy, offsets, w.shape[0])
        return dx, dw, None, None, None

    _op_mm.register_autograd(_backward, setup_context=_setup_ctx)


def grouped_mm(x, w, offsets, tile_group, tile_row):
    """y[a:b] = x[a:b] @ w[g] per segment; block_m of the tile map and
    the kernel launch must agree (build_tile_map(block_m=64)).

    Autocast: custom ops bypass amp's casting, so under autocast the
    trainer hands us bf16 activations and fp32 master weights and
    tl.dot refuses mixed dtypes (8x crash 2026-08-13). Cast both to
    the autocast dtype here, like a real matmul would be — the .to()
    is autograd-tracked, so fp32 params still receive fp32 grads."""
    if not HAS_TRITON:
        raise RuntimeError("triton not available; use grouped_mm_reference")
    if torch.is_autocast_enabled("cuda"):
        dt = torch.get_autocast_dtype("cuda")
        x, w = x.to(dt), w.to(dt)
    elif x.dtype != w.dtype:
        w = w.to(x.dtype)
    return _op_mm(x, w, offsets, tile_group, tile_row)
