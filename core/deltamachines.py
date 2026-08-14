r"""Delta-machine population block (v3 of the machine-population arm,
pattern letter D).

The v2 post-mortem (memory: multicore2-v2-review-brief) split the arm's
cost into three taxes -- thin per-machine GEMMs, the persistent (B,K,T,d_m)
channel, and per-machine softmax attention over the stream. v3 removes the
channel entirely (the user's original framing: gated writes to the
residual, nothing threaded across depth) and replaces each machine's
intake attention with a GATED DELTA RULE memory: a fixed-size (hd x hd)
matrix state per machine-head, updated recurrently as tokens stream past.
The scan state IS the machine's private memory across time -- O(1) size,
no KV cache, and per-token sparsity becomes exact by construction:

    unrouted (token, machine):  write strength beta = 0, decay alpha = 1
    -> the state HOLDS bit-exactly; masking IS skipping. Dense-masked
    execution therefore matches packed-sparse semantics with zero drops,
    so the science (does the router collapse?) runs before any kernel
    work. The grouped-GEMM fast path from v2 (core/grouped_gemm.py) can
    later pack the MLP/projection GEMMs; the scan packs by gathering each
    machine's routed tokens into an order-preserving subsequence.

Block anatomy (drops into a block's attention slot, plain (B,T,C) in/out):

  1. route      predict-before-work: logits from x alone, top-k machines
                per token. Anti-collapse package (the point of v3):
                training-time logit noise, router z-loss, Switch-style
                load-balance loss, and a trainer-driven dense->sparse
                warmup via .topk_now (0 = dense). No frozen sleepers.
  2. intake     per-machine delta memory over the token stream: private
                q/k/v/beta/alpha projections, silu + per-head L2 norm on
                q,k (contraction: ||k||=1 keeps I - beta k k^T stable),
                chunkwise scan, then w_out(norm(o)) + private MLP.
  3. conference K x K attention among the ACTIVE machines' views of this
                token (inactive machines are masked out of the keys --
                with no channel there is no persistent state to read, so
                v2's "services stay readable" semantics has no referent).
                Machine identity via anchor keys; optional per-head sink.
  4. write-back per-machine sigmoid gate x router value, zero-init W_O,
                gated sum into the residual. That's the whole output --
                nothing is carried to the next block.

Chunkwise math (gated delta rule; oracle-tested in
tests/test_deltamachines.py::test_chunk_matches_reference):
  sequential   S_t = a_t (I - b_t k_t k_t^T) S_{t-1} + b_t k_t v_t^T,
               o_t = S_t^T q_t
  substitution S_t = A_t \hat S_t with A_t = prod_{i<=t} a_i (per chunk)
               gives an UNGATED delta recurrence on \hat S with
               u_t = v_t / A_t, so the DeltaNet UT transform applies:
                 Tmat = (I + tril(diag(b) K K^T, -1))^{-1} diag(b)
                 W    = Tmat (U - K S_0)          "pseudo write values"
                 out  = (A*Q) S_0 + tril((A*Q) K^T) W
                 S_L  = A_L (S_0 + K^T W)
  1/A_t is the numerical hazard: per-step log-decay is bounded to
  -LA_TOTAL/chunk (via -LA_TOTAL/L * sigmoid), so 1/A <= e^LA_TOTAL
  within a chunk -- fp32-safe; the scan always runs in >= fp32 (states
  stay fp32 under bf16 autocast).
"""
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from core.model import make_norm, MLPS
from core.fourstroke import MachineLinear, TokenLinear, MachineMLP

# max total within-chunk forgetting: per-step log-decay floor is
# -LA_TOTAL / chunk, so 1/A_t <= e^LA_TOTAL ~ 3e3 stays comfortably fp32.
LA_TOTAL = 8.0

# flash-linear-attention's fused varlen kernel is the fast path for the
# packed scan (the pure-torch inner is launch-bound: ~5k tok/s on a 5090
# vs 0.81 ms/layer fwd+bwd for fla at the same shape). Lazy import:
# False = tried and absent, None = not tried yet.
_FLA_GDR = None


def _fla_kernel():
    global _FLA_GDR
    if _FLA_GDR is None:
        try:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule
            _FLA_GDR = chunk_gated_delta_rule
        except Exception:
            _FLA_GDR = False
    return _FLA_GDR


def delta_scan_reference(q, k, v, beta, la, s0=None):
    """Sequential oracle. q,k: (N,H,T,dk); v: (N,H,T,dv);
    beta, la (log decay, <= 0): (N,H,T). Returns o (N,H,T,dv), S (N,H,dk,dv).
    Test-only: T small python steps, any dtype (fp64 for oracles)."""
    N, H, T, dk = k.shape
    dv = v.shape[-1]
    S = (torch.zeros(N, H, dk, dv, dtype=q.dtype, device=q.device)
         if s0 is None else s0.clone())
    outs = []
    for t in range(T):
        kt, vt, qt = k[:, :, t], v[:, :, t], q[:, :, t]
        S = torch.exp(la[:, :, t])[..., None, None] * S
        err = vt - (S * kt[..., None]).sum(-2)            # v - S^T k
        S = S + (beta[:, :, t][..., None] * kt)[..., None] * err[..., None, :]
        outs.append((S * qt[..., None]).sum(-2))          # S^T q
    return torch.stack(outs, dim=2), S


def delta_scan_chunk(q, k, v, beta, la, s0=None, L=64):
    """Chunkwise-parallel gated delta scan (UT transform, module
    docstring). Same contract as the reference; computes in
    promote(input, fp32) and casts the output back. Autocast is
    DISABLED inside: amp would silently downcast the state matmuls to
    fp16/bf16, breaking the fp32-state guarantee."""
    with torch.autocast(q.device.type, enabled=False):
        return _scan_chunk_inner(q, k, v, beta, la, s0, L)


def _scan_chunk_inner(q, k, v, beta, la, s0, L):
    N, H, T, dk = k.shape
    dv = v.shape[-1]
    dt = torch.promote_types(q.dtype, torch.float32)
    q_, k_, v_ = q.to(dt), k.to(dt), v.to(dt)
    beta_, la_ = beta.to(dt), la.to(dt)
    S = (torch.zeros(N, H, dk, dv, dtype=dt, device=q.device)
         if s0 is None else s0.to(dt))
    eye = torch.eye(L, dtype=dt, device=q.device)
    outs = []
    for s_ in range(0, T, L):
        e = min(s_ + L, T)
        Lc = e - s_
        Kc, Vc, Qc = k_[:, :, s_:e], v_[:, :, s_:e], q_[:, :, s_:e]
        bc = beta_[:, :, s_:e]
        cl = la_[:, :, s_:e].cumsum(-1)                   # log A_t
        U = Vc * torch.exp(-cl)[..., None]                # v_t / A_t
        M = torch.tril(bc[..., None] * (Kc @ Kc.transpose(-2, -1)), -1)
        rhs = bc[..., None] * (U - Kc @ S)
        ey = eye if Lc == L else eye[:Lc, :Lc]
        W = torch.linalg.solve_triangular(ey + M, rhs, upper=False,
                                          unitriangular=True)
        Qt = Qc * torch.exp(cl)[..., None]                # A_t q_t
        outs.append(Qt @ S + torch.tril(Qt @ Kc.transpose(-2, -1)) @ W)
        S = (torch.exp(cl[..., -1])[..., None, None]
             * (S + Kc.transpose(-2, -1) @ W))
    return torch.cat(outs, dim=2).to(q.dtype), S


@torch._dynamo.disable
def delta_scan_packed(q, k, v, beta, la, seg, n_seg, L=64, impl="auto"):
    """Packed gated-delta scan over MACHINE-MAJOR flat rows (the sparse
    fast path: only routed (token, machine) pairs exist as rows).

    q, k: (n, H, dk); v: (n, H, dv); beta, la (log decay <= 0): (n, H);
    seg: (n,) int64, NON-DECREASING segment ids in [0, n_seg); rows
    within a segment are time-ordered. Returns o (n, H, dv) and final
    states S (n_seg, H, dk, dv).

    Same UT-transform chunkwise math as delta_scan_chunk, with two
    segment adaptations inside each L-row chunk: (1) the intra-chunk
    interaction matrices are masked to same-segment pairs (the
    triangular system block-diagonalizes, so one solve still works);
    (2) cumulative decays reset at segment starts, and each segment's
    state picks up its own end-of-chunk decay. A segment absent from a
    chunk keeps its state bit-exactly (matching dense-masked hold).
    Autocast disabled inside (fp32-state guarantee, as delta_scan_chunk).

    The inner scan is activation-checkpointed under grad: its per-row
    state gathers save O(n * H * hd^2) per chunk-tensor for backward
    (~GBs per layer at real shapes — OOMed a 32 GB 5090), while the scan
    itself is FLOPs-light, so recompute is the right trade.

    Kept OUT of torch.compile (@dynamo.disable): the row-chunk loop is
    n/L iterations (~512 at real shapes) and dynamo unrolls it into an
    hour-scale compile (observed hang on a 5090); the surrounding
    grouped GEMMs and conference still compile around the break.

    impl: "torch" = the pure-torch inner (exact, CPU-capable, oracle);
    "fla" = flash-linear-attention's fused varlen kernel (CUDA-only,
    bf16 I/O with fp32 state — fast path; final state not returned);
    "auto" = fla when importable and on CUDA, else torch."""
    use_fla = (impl == "fla" or (impl == "auto" and q.is_cuda
                                 and _fla_kernel()))
    if use_fla:
        return _scan_packed_fla(q, k, v, beta, la, seg, n_seg)
    with torch.autocast(q.device.type, enabled=False):
        if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad
                                        or v.requires_grad):
            return torch.utils.checkpoint.checkpoint(
                lambda a, b, c, d, e: _scan_packed_inner(
                    a, b, c, d, e, seg, n_seg, L),
                q, k, v, beta, la, use_reentrant=False)
        return _scan_packed_inner(q, k, v, beta, la, seg, n_seg, L)


def _scan_packed_fla(q, k, v, beta, la, seg, n_seg):
    """fla fast path. Same row contract as _scan_packed_inner; segments
    become varlen sequences via cu_seqlens (empty segments simply have
    no rows — a zero-length "sequence" never materializes, matching the
    hold-is-identity semantics). Recurrence convention verified against
    delta_scan_reference in tests (decay-then-delta; scalar decay
    commutes with the delta projector, so order is not a distinction).
    q,k,v,beta go bf16 (kernel accumulates state in fp32), g = log decay
    stays fp32, scale=1.0 (our q is L2-normalized, o = S^T q unscaled).
    Final state is not returned (None): the packed forward discards it,
    and skipping output_final_state avoids the (n_seg,H,dk,dv) write."""
    fn = _fla_kernel()
    assert fn, "flash-linear-attention not installed (impl='fla')"
    n = q.shape[0]
    bounds = torch.nonzero(seg[1:] != seg[:-1], as_tuple=True)[0] + 1
    cu = torch.cat([bounds.new_zeros(1), bounds,
                    bounds.new_full((1,), n)]).to(torch.int32)
    with torch.autocast(q.device.type, enabled=False):
        o, _ = fn(q.unsqueeze(0).to(torch.bfloat16),
                  k.unsqueeze(0).to(torch.bfloat16),
                  v.unsqueeze(0).to(torch.bfloat16),
                  g=la.unsqueeze(0).to(torch.float32),
                  beta=beta.unsqueeze(0).to(torch.bfloat16),
                  scale=1.0, output_final_state=False, cu_seqlens=cu)
    return o.squeeze(0).to(q.dtype), None


def _scan_packed_inner(q, k, v, beta, la, seg, n_seg, L):
    n, H, dk = k.shape
    dv = v.shape[-1]
    dt = torch.promote_types(q.dtype, torch.float32)
    q_, k_, v_ = q.to(dt), k.to(dt), v.to(dt)
    beta_, la_ = beta.to(dt), la.to(dt)
    S = torch.zeros(n_seg, H, dk, dv, dtype=dt, device=q.device)
    outs = []
    for s0 in range(0, n, L):
        e = min(s0 + L, n)
        Lc = e - s0
        sg = seg[s0:e]
        Kc = k_[s0:e].transpose(0, 1)                    # (H, Lc, dk)
        Vc = v_[s0:e].transpose(0, 1)
        Qc = q_[s0:e].transpose(0, 1)
        bc = beta_[s0:e].transpose(0, 1)                 # (H, Lc)
        same = sg[:, None] == sg[None, :]                # (Lc, Lc)
        # segment-reset cumulative log-decay: cl[i] = sum of la over this
        # segment's rows in this chunk up to and including i
        cg = la_[s0:e].transpose(0, 1).cumsum(-1)        # (H, Lc)
        prev = F.pad(cg[:, :-1], (1, 0))                 # cumsum before row
        ar = torch.arange(Lc, device=q.device)
        firsts = torch.cat([torch.ones(1, dtype=torch.bool,
                                       device=q.device),
                            sg[1:] != sg[:-1]])
        fidx = torch.cummax(torch.where(firsts, ar,
                                        torch.zeros_like(ar)), 0).values
        cl = cg - prev.gather(-1, fidx.expand(H, Lc))
        A = torch.exp(cl)[..., None]                     # (H, Lc, 1)
        U = Vc * torch.exp(-cl)[..., None]
        S_g = S[sg]                                      # (Lc, H, dk, dv)
        KS0 = torch.einsum("hld,lhdv->hlv", Kc, S_g)
        M = torch.tril(bc[..., None] * (Kc @ Kc.transpose(-2, -1)),
                       -1) * same
        rhs = bc[..., None] * (U - KS0)
        eye = torch.eye(Lc, dtype=dt, device=q.device)
        W = torch.linalg.solve_triangular(eye + M, rhs, upper=False,
                                          unitriangular=True)
        Qt = Qc * A
        O = (torch.einsum("hld,lhdv->hlv", Qt, S_g)
             + (torch.tril(Qt @ Kc.transpose(-2, -1)) * same) @ W)
        outs.append(O.transpose(0, 1))                   # (Lc, H, dv)
        # state update: S[s] = A_end[s] * (S[s] + sum_{r in s} k_r w_r^T)
        lasts = torch.cat([sg[1:] != sg[:-1],
                           torch.ones(1, dtype=torch.bool,
                                      device=q.device)])
        a_end_seg = torch.ones(n_seg, H, dtype=dt, device=q.device)
        a_end_seg[sg[lasts]] = torch.exp(cl)[:, lasts].transpose(0, 1)
        a_row = a_end_seg[sg]                            # (Lc, H)
        contrib = torch.einsum(
            "lhk,lhv->lhkv",
            Kc.transpose(0, 1) * a_row[..., None], W.transpose(0, 1))
        S = S * a_end_seg[..., None, None]
        S = S.index_add(0, sg, contrib)
    return torch.cat(outs, dim=0).to(q.dtype), S


class DeltaMachines(nn.Module):
    """The population; drops into a block's attention slot.
    forward(x) -> residual delta (B,T,C). No channel."""

    def __init__(self, cfg):
        super().__init__()
        K, C, d = cfg.fs_n_machines, cfg.n_embd, cfg.fs_d_machine
        H = cfg.fs_n_head_m
        assert d % H == 0
        self.K, self.d, self.H, self.hd = K, d, H, d // H
        self.chunk = cfg.fs_chunk
        # causal depthwise conv on the stream feeding the machine
        # projections (GDN keeps a short conv before q/k/v; one shared
        # zero-init conv = exact no-op at init, cheap local mixing after)
        self.conv = (nn.Conv1d(C, C, cfg.fs_conv, groups=C, bias=False,
                               padding=0) if cfg.fs_conv else None)
        if self.conv is not None:
            nn.init.zeros_(self.conv.weight)
        # intake projections, fused per machine: q, k, v (d each), then
        # per-head write strength beta and decay logits (H each)
        self.w_in = TokenLinear(K, C, 3 * d + 2 * H)
        self.ln_o = make_norm(cfg, d)
        self.w_out = MachineLinear(K, d, d)
        # probe-ladder ablations (fs_dm_*): rung 1 = plain-MoM substrate
        # (memories only), +mlp, +conf, +gate = full v3. Disabled parts are
        # NOT constructed — DDP errors on params that never touch the loss
        # (the v2 orphaned-s0 lesson, 8x 2026-08-11).
        self.use_mlp = getattr(cfg, "fs_dm_mlp", True)
        self.use_conf = getattr(cfg, "fs_dm_conf", True)
        self.use_gate = getattr(cfg, "fs_dm_gate", True)
        if self.use_mlp:
            self.mlp = MachineMLP(cfg)
            self.ln_mlp = make_norm(cfg, d)
        # conference (identical bones to v2: anchor keys, addr_mix, sink)
        if self.use_conf:
            self.anchor = nn.Parameter(torch.empty(K, d))
            nn.init.orthogonal_(self.anchor)
            self.addr_mix = nn.Parameter(torch.full((K,), cfg.fs_addr_mix))
            self.ln_iface = make_norm(cfg, d)
            self.w_qkv = MachineLinear(K, d, 3 * d)
            self.conf_out = MachineLinear(K, d, d)
            self.conf_sink = (nn.Parameter(torch.zeros(H))
                              if cfg.fs_conf_sink else None)
        # write-back gate on [x_t ; c_k], zero-init W_O. Ungated rungs use
        # the router value alone (MoM-style sum of routed reads).
        if self.use_gate:
            self.gate_x = nn.Parameter(torch.randn(K, C) * 0.02)
            self.gate_c = nn.Parameter(torch.randn(K, d) * 0.02)
            self.gate_b = nn.Parameter(torch.zeros(K))
        self.w_o = MachineLinear(K, d, C, zero=True)
        # router: predict-before-work from x alone. topk_now is the LIVE
        # k -- the trainer's dense->sparse warmup sets it to 0 (dense,
        # router still runs, aux losses still on) then back to fs_topk.
        self.topk = cfg.fs_topk
        self.topk_now = cfg.fs_topk
        self.noise = cfg.fs_route_noise
        if self.topk:
            self.route_x = nn.Parameter(torch.randn(K, C) * 0.02)
            self.route_b = nn.Parameter(torch.zeros(K))
        # packed-sparse execution (fs_packed): only routed rows exist —
        # grouped GEMMs over machine-major flat rows + the segment-packed
        # scan. Semantically identical to dense-masked (equivalence-
        # tested); requires a router; dense warmup and diag hooks fall
        # back to the dense path.
        self.packed = getattr(cfg, "fs_packed", False)
        self.scan_impl = getattr(cfg, "fs_scan", "auto")
        if self.packed:
            assert self.topk, "fs_packed needs fs_topk routing"
        self.lb_loss = None
        self.z_loss = None
        # diagnostics hook, same contract as MachineStrokes.diag (hooks in
        # the REAL forward, never a reimplemented one -- 2026-08-13
        # lesson). rec keys: routes, g, attn, attn_rounds, wb_out_norm,
        # wb_x_norm, wo_norm_k, c_norm_k, lens_wo. ctl: mute_write,
        # mute_conf.
        self.diag = None

    @staticmethod
    def _topk_mask(v, k):
        kth = v.topk(k, dim=-1).values[..., -1:]
        return v * (v >= kth).to(v.dtype)

    @staticmethod
    def _gmm(x, w, offs, tg, tr):
        """Per-machine linear on machine-major flat rows: Triton grouped
        GEMM on GPU (core/grouped_gemm.py), reference einsum on CPU."""
        if x.is_cuda:
            from core.grouped_gemm import grouped_mm
            return grouped_mm(x, w, offs, tg, tr)
        from core.grouped_gemm import grouped_mm_reference
        return grouped_mm_reference(x, w, offs)

    def _heads(self, x):                      # (B,K,T,d) -> (B*K,H,T,hd)
        B, K, T, d = x.shape
        return x.reshape(B * K, T, self.H, self.hd).transpose(1, 2)

    def _confer(self, s, mask, diag=None):
        """K x K conference at each token over the ACTIVE machines' views
        (mask (B,T,K) bool or None = all active). Same bones as v2."""
        B, K, T, d = s.shape
        sn = self.ln_iface(s)
        q, ks, v = self.w_qkv(sn).chunk(3, dim=-1)
        k = (self.anchor[None, :, None, :]
             + self.addr_mix[None, :, None, None] * ks)

        def heads(t):     # (B,K,T,d) -> (B*T, H, K, hd)
            return (t.permute(0, 2, 1, 3)
                    .reshape(B * T, K, self.H, self.hd).transpose(1, 2))

        ctl = diag.get("ctl") if diag else None
        rec = diag.get("rec") if diag else None
        mute = getattr(ctl, "mute_conf", None) if ctl is not None else None
        if (mask is None and self.conf_sink is None and mute is None
                and rec is None):
            y = F.scaled_dot_product_attention(heads(q), heads(k), heads(v))
        else:
            sc = (heads(q) @ heads(k).transpose(-2, -1)
                  / math.sqrt(self.hd))                   # (B*T, H, K, K)
            if mask is not None:
                sc = sc.masked_fill(
                    ~mask.reshape(B * T, 1, 1, K), float("-inf"))
            if mute is not None:
                sc = sc.clone()
                sc[..., mute] = float("-inf")
            if self.conf_sink is not None:
                sink = (self.conf_sink[None, :, None, None]
                        .expand(sc.shape[0], -1, K, 1))
                a = torch.softmax(torch.cat((sc, sink), dim=-1), dim=-1)
                probs = a[..., :K]
            else:
                probs = torch.softmax(sc, dim=-1)
            y = probs @ heads(v)
            if rec is not None:
                with torch.no_grad():
                    rec["attn"] = (rec.get("attn", 0)
                                   + probs.detach().float())
                    rec["attn_rounds"] = rec.get("attn_rounds", 0) + 1
        y = y.transpose(1, 2).reshape(B, T, K, d).permute(0, 2, 1, 3)
        return s + self.conf_out(y)

    def forward(self, x):
        B, T, C = x.shape
        K, d, H = self.K, self.d, self.H
        diag = self.diag
        ctl = diag.get("ctl") if diag else None
        rec = diag.get("rec") if diag else None
        xi = x
        if self.conv is not None:
            w = self.conv.kernel_size[0]
            xi = x + self.conv(
                F.pad(x.transpose(1, 2), (w - 1, 0))).transpose(1, 2)
        # --- route FIRST (predict-before-work; anti-collapse package) ---
        r_gate = mask = None
        k_now = self.topk_now if self.topk else 0
        if self.topk:
            r_log = (torch.einsum("btc,kc->btk", x, self.route_x)
                     + self.route_b)
            if self.training and self.noise:
                r_log = r_log + torch.randn_like(r_log) * self.noise
            r_sig = torch.sigmoid(r_log)
            r_gate = self._topk_mask(r_sig, k_now) if k_now else r_sig
            # dense warmup (k_now = 0): every machine active, sigmoid > 0
            # everywhere -> no conference masking needed (static branch,
            # compile-safe; the flip recompiles once)
            mask = (r_gate > 0) if k_now else None        # (B,T,K) bool
            if rec is not None:
                rec["routes"] = [r_gate.detach().float()]
            with torch.no_grad():
                f = (r_gate > 0).float().mean(dim=(0, 1))
            self.lb_loss = K * (f * r_sig.mean(dim=(0, 1))).sum()
            self.z_loss = torch.logsumexp(r_log, dim=-1).square().mean()
            if self.packed and k_now and diag is None:
                # sparse fast path: only routed rows are ever computed.
                # Identical semantics (equivalence-tested); dense warmup
                # (k_now = 0) and diag runs stay on the dense path.
                return self._forward_packed(x, xi, r_sig, k_now)
        pr = self.w_in(xi)                                # (B,K,T,3d+2H)
        q = self._heads(pr[..., :d])
        k = self._heads(pr[..., d:2 * d])
        v = self._heads(pr[..., 2 * d:3 * d])
        q = F.normalize(F.silu(q), dim=-1)
        k = F.normalize(F.silu(k), dim=-1)
        beta = torch.sigmoid(
            pr[..., 3 * d:3 * d + H]).permute(0, 1, 3, 2).reshape(
            B * K, H, T)
        la = (-(LA_TOTAL / self.chunk) * torch.sigmoid(
            pr[..., 3 * d + H:])).permute(0, 1, 3, 2).reshape(B * K, H, T)
        if r_gate is not None:
            rk = r_gate.permute(0, 2, 1).reshape(B * K, 1, T)
            # unrouted: beta -> 0 AND log-decay -> 0 (alpha = 1): the
            # state holds bit-exactly, masking == skipping. Routed writes
            # scale by the router sigmoid (its gradient path).
            beta = beta * rk.to(beta.dtype)
            la = la * (rk > 0).to(la.dtype)
        o, _ = delta_scan_chunk(q, k, v, beta, la, L=self.chunk)
        o = (o.view(B, K, self.H, T, self.hd).permute(0, 1, 3, 2, 4)
             .reshape(B, K, T, d))
        s = self.w_out(self.ln_o(o))
        if self.use_mlp:
            s = s + self.mlp(self.ln_mlp(s))
        c = self._confer(s, mask, diag) if self.use_conf else s
        if self.use_gate:
            g = torch.sigmoid(
                torch.einsum("btc,kc->btk", x, self.gate_x)
                + torch.einsum("bktd,kd->btk", c, self.gate_c)
                + self.gate_b)
            if r_gate is not None:
                g = g * r_gate
        else:
            g = (r_gate if r_gate is not None
                 else x.new_ones(B, T, K))    # MoM-style routed sum
        if (ctl is not None
                and getattr(ctl, "mute_write", None) is not None):
            g = g.clone()
            g[..., ctl.mute_write] = 0.0
        out = torch.einsum("btk,bktc->btc", g, self.w_o(c))
        if rec is not None:
            with torch.no_grad():
                rec["g"] = g.detach().float()
                oc = self.w_o(c).float()
                rec["wb_out_norm"] = out.detach().float().norm(dim=-1).sum()
                rec["wb_x_norm"] = x.detach().float().norm(dim=-1).sum()
                rec["wo_norm_k"] = (g.float().permute(0, 2, 1)[..., None]
                                    * oc).norm(dim=-1).sum(dim=(0, 2))
                rec["c_norm_k"] = c.detach().float().norm(dim=-1).sum(
                    dim=(0, 2))
                if rec.get("lens_pos") is not None:
                    rec["lens_wo"] = self.w_o(
                        c[:, :, rec["lens_pos"], :]).detach()
        return out


    def _forward_packed(self, x, xi, r_sig, kk):
        """Sparse execution: the n = B*T*kk routed (token, machine) rows,
        MACHINE-MAJOR (stable sort by machine keeps (b, t) order inside
        each machine — the layout both the grouped GEMMs and the packed
        scan want). All shapes static: routing changes tensor contents,
        never shapes."""
        B, T, C = x.shape
        K, d, H, hd = self.K, self.d, self.H, self.hd
        dev = x.device
        vals, midx = r_sig.topk(kk, dim=-1)              # (B,T,kk)
        n = B * T * kk
        i = torch.arange(n, device=dev)
        b, t = i // (T * kk), (i // kk) % T
        m = midx.reshape(-1).to(torch.int64)
        order = torch.argsort(m, stable=True)
        m_s, b_s, t_s = m[order], b[order], t[order]
        bt = b_s * T + t_s                               # row -> token
        rv = vals.reshape(-1)[order]                     # router sigmoid
        seg = m_s * B + b_s                              # non-decreasing
        counts = torch.bincount(m_s, minlength=K)
        offs = torch.zeros(K + 1, dtype=torch.int32, device=dev)
        offs[1:] = counts.cumsum(0).to(torch.int32)
        if dev.type == "cuda":
            from core.grouped_gemm import build_tile_map
            tg, tr = build_tile_map(offs, 64, n // 64 + K)
        else:
            tg = tr = None
        gmm = self._gmm
        xf = xi.reshape(B * T, C).index_select(0, bt)    # (n, C)
        pr = gmm(xf, self.w_in.weight, offs, tg, tr)     # (n, 3d+2H)
        q = F.normalize(F.silu(pr[:, :d].view(n, H, hd)), dim=-1)
        k = F.normalize(F.silu(pr[:, d:2 * d].view(n, H, hd)), dim=-1)
        v = pr[:, 2 * d:3 * d].view(n, H, hd)
        beta = torch.sigmoid(pr[:, 3 * d:3 * d + H]) * rv[:, None]
        la = -(LA_TOTAL / self.chunk) * torch.sigmoid(pr[:, 3 * d + H:])
        # row-chunk L is a compute blocking factor, NOT the decay-clamp
        # window (that is fs_chunk, already baked into la above): larger
        # L = fewer python-level chunk iterations (the packed scan is
        # launch-bound in eager). Overflow bound: worst in-chunk 1/A is
        # e^(LA_TOTAL/fs_chunk * L) = e^32 at 256/64 — fp32-safe.
        o, _ = delta_scan_packed(k=k, q=q, v=v, beta=beta, la=la,
                                 seg=seg, n_seg=K * B,
                                 L=min(4 * self.chunk, 256),
                                 impl=self.scan_impl)
        s = gmm(self.ln_o(o.reshape(n, d)), self.w_out.weight,
                offs, tg, tr)
        if self.use_mlp:
            h = F.relu(gmm(self.ln_mlp(s), self.mlp.fc.weight,
                           offs, tg, tr)).square()
            for mid in self.mlp.mids:
                h = F.relu(gmm(h, mid.weight, offs, tg, tr)).square()
            s = s + gmm(h, self.mlp.proj.weight, offs, tg, tr)
        if self.use_conf:
            sn = self.ln_iface(s)
            qc, kc, vc = gmm(sn, self.w_qkv.weight,
                             offs, tg, tr).split(d, dim=-1)
            kc = self.anchor[m_s] + self.addr_mix[m_s, None] * kc
            inv = torch.empty_like(order)
            inv[order] = i                               # -> token-major

            def tok(z):      # (n, d) -> (B*T, H, kk, hd)
                return (z.index_select(0, inv)
                        .view(B * T, kk, H, hd).transpose(1, 2))

            sc = tok(qc) @ tok(kc).transpose(-2, -1) / math.sqrt(hd)
            if self.conf_sink is not None:
                sink = (self.conf_sink[None, :, None, None]
                        .expand(B * T, -1, kk, 1))
                a = torch.softmax(torch.cat((sc, sink), -1),
                                  -1)[..., :kk]
            else:
                a = torch.softmax(sc, -1)
            y = (a @ tok(vc)).transpose(1, 2).reshape(n, d)
            y = y.index_select(0, order)                 # machine-major
            c = s + gmm(y, self.conf_out.weight, offs, tg, tr)
        else:
            c = s
        if self.use_gate:
            xr = x.reshape(B * T, C).index_select(0, bt)  # gate reads RAW x
            g = torch.sigmoid((xr * self.gate_x[m_s]).sum(-1)
                              + (c * self.gate_c[m_s]).sum(-1)
                              + self.gate_b[m_s]) * rv
        else:
            g = rv
        wo = gmm(c, self.w_o.weight, offs, tg, tr)       # (n, C)
        out = x.new_zeros(B * T, C).index_add(0, bt, wo * g[:, None])
        return out.view(B, T, C)


class DeltaMachineBlock(nn.Module):
    """Pre-LN block, attention slot = the population, own FFN on the
    token path. Plain forward(x) -> x: no channel threading."""

    def __init__(self, cfg, attn_key=None):
        super().__init__()
        self.ln_1 = make_norm(cfg, cfg.n_embd)
        self.machines = DeltaMachines(cfg)
        self.ln_2 = make_norm(cfg, cfg.n_embd)
        self.mlp = MLPS[cfg.mlp](cfg)

    @property
    def attn(self):
        # model-level loops (lb_loss, z_loss) probe blk.attn
        return self.machines

    def forward(self, x):
        x = x + self.machines(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


def deltamachines_score_flops(cfg, T):
    """Per-D-layer fwd+bwd FLOPs per token beyond (or below) what 6*N
    counts. 6*N bills every machine param once per token; corrections:

      + scan spend per ROUTED (token, machine): intra-chunk band charged
        as attention with a full chunk of keys (12*d*L convention --
        generous: covers the K K^T / q K^T tiles, the triangular solve,
        and cumulative-decay overhead) plus state read/write/erase
        (~3 rank-1 ops on H (hd x hd) states, fwd+bwd ~ 18*d*hd)
      + conference scores among the k_act active views: 12*d*k_act^2
      - unrouted pairs skip ALL private params (w_in, w_out, MLP,
        w_qkv/conf_out, w_o) that 6*N billed: routing is decided from x
        before any machine work, so the skips are real
      router/gate/anchor params are tiny and billed by 6*N as-is."""
    K, d, C, H = (cfg.fs_n_machines, cfg.fs_d_machine, cfg.n_embd,
                  cfg.fs_n_head_m)
    hd = d // H
    frac = (cfg.fs_topk / K) if cfg.fs_topk else 1.0
    k_act = cfg.fs_topk or K
    mlp_p = ((2 * cfg.fs_mlp_mult * d * d
              + (cfg.fs_mlp_depth - 1) * (cfg.fs_mlp_mult * d) ** 2)
             if getattr(cfg, "fs_dm_mlp", True) else 0)
    conf_p = (4 * d * d if getattr(cfg, "fs_dm_conf", True) else 0)
    p_priv = (C * (3 * d + 2 * H)      # w_in
              + d * d + mlp_p          # w_out, private MLP (if built)
              + conf_p                 # conference w_qkv+conf_out (if built)
              + d * C)                 # w_o
    scan = 12 * d * cfg.fs_chunk + 18 * d * hd
    score = K * frac * scan
    if getattr(cfg, "fs_dm_conf", True):
        score += 12 * d * k_act * k_act
    score -= 6 * (1 - frac) * K * p_priv
    return score
