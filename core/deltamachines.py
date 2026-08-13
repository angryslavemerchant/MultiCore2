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
    promote(input, fp32) and casts the output back."""
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
        # --- route (predict-before-work; anti-collapse package) ---
        r_gate = mask = None
        if self.topk:
            r_log = (torch.einsum("btc,kc->btk", x, self.route_x)
                     + self.route_b)
            if self.training and self.noise:
                r_log = r_log + torch.randn_like(r_log) * self.noise
            r_sig = torch.sigmoid(r_log)
            k_now = self.topk_now
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
