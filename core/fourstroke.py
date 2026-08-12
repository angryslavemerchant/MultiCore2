"""Four-stroke machine population block (four_stroke_spec_v1.md).

A machine block replaces attention with a four-beat cycle over K persistent
"machines" that live alongside the token stream:

  1. intake      machine reads the stream + its own private channel
                 (backend, PRIVATE projections), then private MLP_k
  2. publish     interface projections from state: q, k = e_k + mix*W_K s, v
  3. conference  K x K attention among the machines -> conclusion c[k]
  4. write-back  per-machine sigmoid gate, gated sum into the residual;
                 c[k] also becomes the machine's private-channel entry for
                 the NEXT machine block

The machine channel is threaded between machine blocks as the conclusions
tensor (B, K, T, d_machine); the first machine block seeds it from the
learned inits s0_k (init_channel()).

Implementation decisions the spec leaves open (v1 choices, all cheap to
revisit):
  - Intake query source: the machine's previous-layer conclusion c_prev
    (the only per-position layer l-1 machine value that exists; satisfies
    the delay-one-layer rule). At the first machine block this is s0_k, so
    intake at t=0 of training is driven purely by innate character.
  - RoPE on intake q and both key sources (token keys and private-channel
    keys are position-indexed). Conference is position-free: all K
    machines sit at the same position t.
  - Zero-init ONLY the write-back W_O (token residual untouched at init).
    The intake/conference/MLP output projections land on the MACHINE
    residual and are normal-init: zero-initting all of them stacks three
    zero residuals in series on one channel and gradients would take
    several steps to reach the backend.
  - Conference output is a residual on state (c = s + proj(conf)) so a
    machine's identity survives its own conclusion.
  - Per-machine gate vectors (spec writes one w_g; "independent gates per
    machine" reads more naturally as per-machine parameters).

Optimizer note: per-machine weights are stored stacked as (K, d_in, d_out)
3D tensors (einsum-batched -- K separate nn.Linears would serialize).
They carry .MUON_STACKED = True; configure_optimizers routes tagged
tensors to Muon, whose Newton-Schulz runs per K-slice (batched matmuls,
per-slice norms) -- true per-machine orthogonalisation.

Cost honesty: the attn backend is K full causal attentions at d_machine
over the stream -- at K=16, d_machine=256, d_model=512 the intake scores
alone are ~8x one standard attention layer's scores. fs_backend="swa"
bounds it; fourstroke_extra_flops() charges it either way.
"""
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from core.model import make_norm, MLPS
from core.rope import apply_rope

_MASK_CACHE = {}

# swa-backend query-chunk size for the streamed intake sweep. Each chunk
# of Q queries attends only its own <= 2*(Q + window - 1) candidate keys,
# so nothing (T, 2T)-shaped is ever materialised. Module-level so tests
# can force multi-chunk execution at tiny T.
CHUNK_Q = 256


def _intake_mask(T, window, device):
    """(T, 2T) bool, True = may attend. Keys are [tokens 0..T-1,
    private-channel conclusions 0..T-1]; both sources causal at <= t,
    banded to `window` when set (swa backend)."""
    key = (T, window, str(device))
    hit = _MASK_CACHE.get(key)
    if hit is None:
        t = torch.arange(T, device=device)
        vis = t[None, :] <= t[:, None]
        if window is not None:
            vis = vis & (t[None, :] > t[:, None] - window)
        hit = torch.cat((vis, vis), dim=1)
        _MASK_CACHE[key] = hit
    return hit


def _band_mask(nq, nk, off, window, device):
    """(nq, 2*nk) bool for one query chunk: query i (absolute t = s + i)
    may see per-source key j' (absolute lo + j') iff lo + j' in
    (t - window, t]; off = s - lo. Pattern depends only on shapes and
    off, so chunks after the first share one cached mask."""
    key = (nq, nk, off, window, str(device))
    hit = _MASK_CACHE.get(key)
    if hit is None:
        i = torch.arange(nq, device=device)[:, None]
        j = torch.arange(nk, device=device)[None, :]
        vis = (j <= off + i) & (j > off + i - window)
        hit = torch.cat((vis, vis), dim=1)
        _MASK_CACHE[key] = hit
    return hit


def _band_mask_all(nc, Q, w, device):
    """(nc, Q, 2*(Q+w-1)) bool for the single-kernel path: chunk c's
    query i sits at absolute t = c*Q + i; per-source key slot j' holds
    absolute position c*Q + j' - (w-1) (front-padded), valid iff that
    position is in (t-w, t] and >= 0 (chunk 0's pad slots are ghosts)."""
    key = ("all", nc, Q, w, str(device))
    hit = _MASK_CACHE.get(key)
    if hit is None:
        n = Q + w - 1
        c = torch.arange(nc, device=device)[:, None, None]
        i = torch.arange(Q, device=device)[None, :, None]
        j = torch.arange(n, device=device)[None, None, :]
        pos = c * Q + j - (w - 1)
        t = c * Q + i
        vis = (pos <= t) & (pos > t - w) & (pos >= 0)
        hit = torch.cat((vis, vis), dim=-1)
        _MASK_CACHE[key] = hit
    return hit


def _unfold_keys(x, Q, w):
    """(B_, H, T, hd) -> (B_, nc, H, Q+w-1, hd): front-pad w-1 ghost
    positions, then window per query chunk (dup factor ~(Q+w)/Q)."""
    B_, H, T, hd = x.shape
    xp = F.pad(x, (0, 0, w - 1, 0))
    u = xp.unfold(2, Q + w - 1, Q)            # (B_, H, nc, hd, n)
    return u.permute(0, 2, 1, 4, 3)


class MachineLinear(nn.Module):
    """Per-machine linear, weights stacked (K, d_in, d_out).
    Input (B, K, T, d_in) -> (B, K, T, d_out). No bias (machine identity
    lives in s0/e_k, not in projection offsets)."""

    def __init__(self, K, d_in, d_out, zero=False):
        super().__init__()
        w = (torch.zeros(K, d_in, d_out) if zero
             else torch.randn(K, d_in, d_out) * 0.02)
        self.weight = nn.Parameter(w)
        self.weight.MUON_STACKED = True

    def forward(self, x):
        return torch.einsum("bktd,kde->bkte", x, self.weight)


class TokenLinear(nn.Module):
    """Shared token input, per-machine output: (B, T, C) -> (B, K, T, d)."""

    def __init__(self, K, C, d):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(K, C, d) * 0.02)
        self.weight.MUON_STACKED = True

    def forward(self, x):
        # one (B*T, C) @ (C, K*d) GEMM instead of a K-way bmm
        B, T, C = x.shape
        K, _, d = self.weight.shape
        w = self.weight.permute(1, 0, 2).reshape(C, K * d)
        return (x.reshape(B * T, C) @ w).view(B, T, K, d).transpose(1, 2)


class SharedLinear(nn.Module):
    """One weight applied to every machine: (B, K, T, d_in) -> (..., d_out).
    Plain 2D parameter, so Muon picks it up on the standard path."""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_in, d_out) * 0.02)

    def forward(self, x):
        return torch.einsum("bktd,de->bkte", x, self.weight)


class MachineMLP(nn.Module):
    """Private per-machine MLP, relu^2, residual applied by the caller.
    fs_mlp_depth > 1 inserts square (mult*d x mult*d) hidden layers — the
    most arithmetic-dense GEMMs in the arm, and routed, so their params
    are cheap on the FLOPs ledger at top-k."""

    def __init__(self, cfg):
        super().__init__()
        K, d, m = cfg.fs_n_machines, cfg.fs_d_machine, cfg.fs_mlp_mult
        self.fc = MachineLinear(K, d, m * d)
        self.mids = nn.ModuleList(
            MachineLinear(K, m * d, m * d)
            for _ in range(getattr(cfg, "fs_mlp_depth", 1) - 1))
        self.proj = MachineLinear(K, m * d, d)

    def forward(self, x):
        h = F.relu(self.fc(x)).square()
        for mid in self.mids:
            h = F.relu(mid(h)).square()
        return self.proj(h)


class AttnBackend(nn.Module):
    """State backend "attn"/"swa": multi-head attention from the machine's
    previous conclusion over [token stream, own private channel], both at
    positions <= t (banded for swa). All projections PRIVATE per machine.
    Contract: forward(x_tok, c_prev) -> s, function only of layer l-1
    values at positions <= t."""

    def __init__(self, cfg, window=None):
        super().__init__()
        K, C, d = cfg.fs_n_machines, cfg.n_embd, cfg.fs_d_machine
        H = cfg.fs_n_head_m
        assert d % H == 0
        self.H, self.hd, self.window = H, d // H, window
        self.use_rope = cfg.fs_rope
        self.ln_q = make_norm(cfg, d)
        self.ln_p = make_norm(cfg, d)     # private-channel keys/values
        self.w_q = MachineLinear(K, d, d)
        # token intake k/v: private per machine (v1), or — fs_tkv_heads>0 —
        # ONE shared bank of that many heads, projected once per block;
        # machine m's head h reads bank head (m*H + h) % bank_h, so
        # machines are spread across the bank and differ by their private
        # queries (MQA/GQA applied across the population)
        self.bank_h = getattr(cfg, "fs_tkv_heads", 0)
        if self.bank_h:
            self.w_tkv = nn.Parameter(
                torch.randn(C, self.bank_h * 2 * self.hd) * 0.02)
            idx = torch.arange(K * H) % self.bank_h
            self.register_buffer("bank_idx", idx, persistent=False)
        else:
            self.w_tkv = TokenLinear(K, C, 2 * d)  # private k,v fused
        self.w_pkv = MachineLinear(K, d, 2 * d)  # private-channel k,v fused
        self.w_out = MachineLinear(K, d, d)
        self.mlp = MachineMLP(cfg)
        self.ln_mlp = make_norm(cfg, d)

    def _heads(self, x):                      # (B,K,T,d) -> (B*K,H,T,hd)
        B, K, T, d = x.shape
        return (x.reshape(B * K, T, self.H, self.hd).transpose(1, 2))

    def prep(self, x_tok):
        """Token-side k/v, computed ONCE per block: within a block the
        tokens never change, so loop rounds reuse this (the (B*K,H,T,hd)
        token tensors are the bandwidth-heavy part of intake)."""
        if self.bank_h:
            B, T, C = x_tok.shape
            kv = (x_tok.reshape(B * T, C) @ self.w_tkv).view(
                B, T, self.bank_h, 2, self.hd)
            k_b = kv[:, :, :, 0].transpose(1, 2)     # (B, bank_h, T, hd)
            v_b = kv[:, :, :, 1].transpose(1, 2)
            if self.use_rope:
                k_b = apply_rope(k_b, k_b)[0]
            K = self.bank_idx.numel() // self.H
            k_tok = (k_b.index_select(1, self.bank_idx)
                     .view(B, K, self.H, T, self.hd)
                     .reshape(B * K, self.H, T, self.hd))
            v_tok = (v_b.index_select(1, self.bank_idx)
                     .view(B, K, self.H, T, self.hd)
                     .reshape(B * K, self.H, T, self.hd))
            return k_tok, v_tok
        k_tok, v_tok = self.w_tkv(x_tok).chunk(2, dim=-1)
        k_tok, v_tok = self._heads(k_tok), self._heads(v_tok)
        if self.use_rope:
            k_tok = apply_rope(k_tok, k_tok)[0]
        return k_tok, v_tok

    def step(self, tok_kv, c_prev):
        """State-dependent intake + private MLP for one loop round."""
        B, K, T, d = c_prev.shape
        k_tok, v_tok = tok_kv
        q = self._heads(self.w_q(self.ln_q(c_prev)))
        k_prv, v_prv = self.w_pkv(self.ln_p(c_prev)).chunk(2, dim=-1)
        k_prv, v_prv = self._heads(k_prv), self._heads(v_prv)
        if self.use_rope:
            q = apply_rope(q, q)[0]
            k_prv = apply_rope(k_prv, k_prv)[0]
        if self.window is not None:
            y = self._banded(q, k_tok, v_tok, k_prv, v_prv)
        else:
            k = torch.cat((k_tok, k_prv), dim=2)  # (B*K,H,2T,hd)
            v = torch.cat((v_tok, v_prv), dim=2)
            mask = _intake_mask(T, None, k.device)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = (y.transpose(1, 2).reshape(B, K, T, d))
        s = c_prev + self.w_out(y)
        return s + self.mlp(self.ln_mlp(s))

    def forward(self, x_tok, c_prev):
        return self.step(self.prep(x_tok), c_prev)

    def _banded(self, q, k_tok, v_tok, k_prv, v_prv):
        """Streamed swa intake: query chunks against their own band of
        [token, private] keys. One softmax over both sources per query,
        never anything (T, 2T)-shaped. When T divides into CHUNK_Q the
        chunks are batched into ONE SDPA call (chunk dim folded into
        batch, keys unfolded with front padding) -- 16 small masked
        kernels per block collapse into one saturating one."""
        B_, H, T, hd = q.shape
        w = self.window
        if T % CHUNK_Q == 0 and T > CHUNK_Q:
            Q = CHUNK_Q
            nc, n = T // Q, Q + w - 1
            qq = (q.view(B_, H, nc, Q, hd).permute(0, 2, 1, 3, 4)
                  .reshape(B_ * nc, H, Q, hd))
            k = torch.cat((_unfold_keys(k_tok, Q, w),
                           _unfold_keys(k_prv, Q, w)),
                          dim=3).reshape(B_ * nc, H, 2 * n, hd)
            v = torch.cat((_unfold_keys(v_tok, Q, w),
                           _unfold_keys(v_prv, Q, w)),
                          dim=3).reshape(B_ * nc, H, 2 * n, hd)
            mask = (_band_mask_all(nc, Q, w, q.device)
                    .view(1, nc, 1, Q, 2 * n)
                    .expand(B_, nc, 1, Q, 2 * n)
                    .reshape(B_ * nc, 1, Q, 2 * n))
            y = F.scaled_dot_product_attention(qq, k, v, attn_mask=mask)
            return (y.view(B_, nc, H, Q, hd).permute(0, 2, 1, 3, 4)
                    .reshape(B_, H, T, hd))
        outs = []
        for s in range(0, T, CHUNK_Q):
            e = min(s + CHUNK_Q, T)
            lo = max(0, s - w + 1)
            k = torch.cat((k_tok[:, :, lo:e], k_prv[:, :, lo:e]), dim=2)
            v = torch.cat((v_tok[:, :, lo:e], v_prv[:, :, lo:e]), dim=2)
            mask = _band_mask(e - s, e - lo, s - lo, w, q.device)
            outs.append(F.scaled_dot_product_attention(
                q[:, :, s:e], k, v, attn_mask=mask))
        return torch.cat(outs, dim=2)


def make_backend(cfg):
    if cfg.fs_backend == "attn":
        return AttnBackend(cfg, window=None)
    if cfg.fs_backend == "swa":
        return AttnBackend(cfg, window=cfg.fs_window)
    raise NotImplementedError(f"fs_backend={cfg.fs_backend!r} "
                              "(linear/ttt are deferred past v1)")


class MachineStrokes(nn.Module):
    """The four strokes; drops into a block's attention slot.
    forward(x, c_prev) -> (residual delta (B,T,C), conclusions (B,K,T,d))."""

    def __init__(self, cfg, seed=True):
        super().__init__()
        K, C, d = cfg.fs_n_machines, cfg.n_embd, cfg.fs_d_machine
        H = cfg.fs_n_head_m
        self.H, self.hd = H, d // H
        self.K, self.d = K, d
        # identity: innate character + fixed address component, orthogonal
        # rows so machines start distinguishable (spec section 5). Only the
        # FIRST machine block owns s0 — later blocks' channels come from
        # their predecessor, so their seeds would never touch the loss and
        # DDP's reducer errors on the orphaned params (8x, 2026-08-11).
        if seed:
            self.s0 = nn.Parameter(torch.empty(K, d))
            nn.init.orthogonal_(self.s0)
        else:
            self.s0 = None
        self.anchor = nn.Parameter(torch.empty(K, d))
        nn.init.orthogonal_(self.anchor)
        # anchor-vs-state mixing scalar on the published key, per machine
        self.addr_mix = nn.Parameter(torch.full((K,), cfg.fs_addr_mix))
        self.backend = make_backend(cfg)
        self.ln_iface = make_norm(cfg, d)     # RMSNorm-on-s (spec section 5)
        # publish: private per machine (v1) or — fs_share_pub — one shared
        # interface projection; machine identity survives via the anchor
        # keys and the private state content
        share = getattr(cfg, "fs_share_pub", False)
        self.w_qkv = (SharedLinear(d, 3 * d) if share
                      else MachineLinear(K, d, 3 * d))
        self.conf_out = (SharedLinear(d, d) if share
                         else MachineLinear(K, d, d))
        # write-back: per-machine gate on [x_t ; c_k], zero-init W_O so the
        # token residual is untouched at init
        self.gate_x = nn.Parameter(torch.randn(K, C) * 0.02)
        self.gate_c = nn.Parameter(torch.randn(K, d) * 0.02)
        self.gate_b = nn.Parameter(torch.zeros(K))
        self.w_o = MachineLinear(K, d, C, zero=True)
        # --- v2 features, all inert at their config defaults ---
        # top-k activation (fs_topk > 0): per (token, machine) router from
        # (x, c_prev); unrouted pairs SKIP the state refresh (intake+MLP)
        # and pass c_prev through — still published/readable in conference
        # (interp 2026-08-12: services are read without writing). The same
        # k also caps the write-back gates. Routed pairs' refresh is scaled
        # by the router sigmoid (its gradient path).
        self.topk = cfg.fs_topk
        if self.topk:
            self.route_x = nn.Parameter(torch.randn(K, C) * 0.02)
            self.route_c = nn.Parameter(torch.randn(K, d) * 0.02)
            self.route_b = nn.Parameter(torch.zeros(K))
        self.lb_loss = None
        # conference sink (fs_conf_sink): ONE learned logit per head
        # appended to the conference softmax with a zero value — reading
        # "nothing" is legal (NSA register mech-interp recommendation).
        self.conf_sink = (nn.Parameter(torch.zeros(cfg.fs_n_head_m))
                          if cfg.fs_conf_sink else None)
        # loop (fs_loop_rounds R > 1): strokes 1-4 iterate on state with
        # tied weights and frozen token k/v; rounds >= 2 apply a learned
        # per-machine "still thinking" sigmoid on the state delta
        # (relative-magnitude halting, differentiable), optionally
        # top-fs_loop_topk machines per token per round.
        self.rounds = cfg.fs_loop_rounds
        self.loop_topk = cfg.fs_loop_topk
        if self.rounds > 1:
            self.loop_w = nn.Parameter(torch.zeros(K, d))
            self.loop_b = nn.Parameter(torch.full((K,), 2.0))

    def init_channel(self, B, T, device=None, dtype=None):
        """Seed conclusions for the first machine block: s0 at every t."""
        assert self.s0 is not None, "only the seed block can init the channel"
        s0 = self.s0.to(device=device, dtype=dtype)
        return s0[None, :, None, :].expand(B, -1, T, -1).contiguous()

    @staticmethod
    def _topk_mask(v, k):
        """Zero all but the top-k entries of v along its last dim."""
        kth = v.topk(k, dim=-1).values[..., -1:]
        return v * (v >= kth).to(v.dtype)

    def _confer(self, s):                                 # strokes 2 + 3
        B, K, T, d = s.shape
        sn = self.ln_iface(s)
        q, ks, v = self.w_qkv(sn).chunk(3, dim=-1)
        k = (self.anchor[None, :, None, :]
             + self.addr_mix[None, :, None, None] * ks)

        def heads(t):     # (B,K,T,d) -> (B*T, H, K, hd): attention over K
            return (t.permute(0, 2, 1, 3)
                    .reshape(B * T, K, self.H, self.hd).transpose(1, 2))

        if self.conf_sink is None:
            y = F.scaled_dot_product_attention(heads(q), heads(k), heads(v))
        else:
            sc = (heads(q) @ heads(k).transpose(-2, -1)
                  / math.sqrt(self.hd))                   # (B*T, H, K, K)
            sink = (self.conf_sink[None, :, None, None]
                    .expand(sc.shape[0], -1, K, 1))
            a = torch.softmax(torch.cat((sc, sink), dim=-1), dim=-1)
            y = a[..., :K] @ heads(v)                     # sink value = 0
        y = (y.transpose(1, 2).reshape(B, T, K, d).permute(0, 2, 1, 3))
        return s + self.conf_out(y)

    def forward(self, x, c_prev):
        B, T, C = x.shape
        K = self.K
        tok_kv = self.backend.prep(x)                     # once per block
        route = None
        if self.topk:
            r_log = (torch.einsum("btc,kc->btk", x, self.route_x)
                     + torch.einsum("bktd,kd->btk", c_prev, self.route_c)
                     + self.route_b)
            r_gate = self._topk_mask(torch.sigmoid(r_log), self.topk)
            route = r_gate.permute(0, 2, 1)[..., None]    # (B,K,T,1)
            with torch.no_grad():
                f = (r_gate > 0).float().mean(dim=(0, 1))  # traffic frac
            p = torch.sigmoid(r_log).mean(dim=(0, 1))      # router mass
            self.lb_loss = K * (f * p).sum()               # Switch-style
        c = c_prev
        for r in range(self.rounds):
            s = self.backend.step(tok_kv, c)              # stroke 1
            if route is not None:
                s = c + route * (s - c)     # unrouted: state passes through
            cand = self._confer(s)
            if r == 0:
                c = cand
            else:
                u = torch.sigmoid(                        # still thinking?
                    torch.einsum("bktd,kd->btk", cand, self.loop_w)
                    + self.loop_b)
                if self.loop_topk:
                    u = self._topk_mask(u, self.loop_topk)
                c = c + u.permute(0, 2, 1)[..., None] * (cand - c)
        g = torch.sigmoid(                                # stroke 4
            torch.einsum("btc,kc->btk", x, self.gate_x)
            + torch.einsum("bktd,kd->btk", c, self.gate_c)
            + self.gate_b)
        if self.topk:
            g = self._topk_mask(g, self.topk)
        out = torch.einsum("btk,bktc->btc", g, self.w_o(c))
        return out, c


class FourStrokeBlock(nn.Module):
    """Pre-LN block whose attention slot holds the strokes; keeps its own
    FFN on the token path (spec section 4). forward threads the machine
    channel: (x, c_prev) -> (x, c)."""

    def __init__(self, cfg, attn_key=None, seed=True):
        super().__init__()
        self.ln_1 = make_norm(cfg, cfg.n_embd)
        self.strokes = MachineStrokes(cfg, seed=seed)
        self.ln_2 = make_norm(cfg, cfg.n_embd)
        self.mlp = MLPS[cfg.mlp](cfg)

    @property
    def attn(self):
        # model-level loops (set_depth, lb_loss) probe blk.attn
        return self.strokes

    def forward(self, x, c_prev):
        y, c = self.strokes(self.ln_1(x), c_prev)
        x = x + y
        x = x + self.mlp(self.ln_2(x))
        return x, c


def fourstroke_score_flops(cfg, T):
    """Per-M-layer fwd+bwd FLOPs per token beyond (or below) what 6*N
    counts. 6*N bills every machine param once per token; corrections:

      + intake scores (12*d*keys convention, two key sources), spent only
        by the routed fraction frac = topk/K (top-k skips the refresh)
      + conference scores K x K at d_machine (dense — every machine stays
        readable; sink column is one extra key, ignored as ~1/K)
      - unrouted pairs skip intake private params (w_q, w_pkv, w_out, MLP)
        and the write-back w_o that 6*N already billed
      - w_tkv is NOT discounted: token k/v are computed once per block
        for all machines regardless of routing (and reused by loops)
      + rounds 2..R re-spend, per round: routed intake (params + scores),
        the dense publish/conference (w_qkv, conf_out, scores). Charged at
        the full routed fraction even when fs_loop_topk < topk
        (conservative: the halting gate's savings are not credited)."""
    K, d, w, C = (cfg.fs_n_machines, cfg.fs_d_machine, cfg.fs_window,
                  cfg.n_embd)
    if cfg.fs_backend == "attn":
        avg = T
    else:
        avg = T if T <= w else (w * (w + 1) / 2 + (T - w) * w) / T
    frac = (cfg.fs_topk / K) if cfg.fs_topk else 1.0
    mult, depth = cfg.fs_mlp_mult, cfg.fs_mlp_depth
    intake_scores = K * 12 * d * 2 * avg * frac
    conf_scores = K * 12 * d * K
    mlp_p = 2 * mult * d * d + (depth - 1) * (mult * d) ** 2
    p_intake = K * (d * d + 2 * d * d + d * d          # w_q, w_pkv, w_out
                    + mlp_p)                           # MLP (incl. mids)
    p_wo = K * d * C
    score = intake_scores + conf_scores
    score -= 6 * (1 - frac) * (p_intake + p_wo)
    # publish/conference param spend per round: private stacks are billed
    # once by 6*N (numel = K * 4d^2); a SHARED interface is billed once by
    # 6*N but applied K times per token, so the extra (K-1) applications
    # are a surcharge here — and every loop round re-spends the full K.
    p_pub_apply = K * (3 * d * d + d * d)              # round-1 spend
    if cfg.fs_share_pub:
        score += 6 * (K - 1) * (3 * d * d + d * d)     # round 1 surcharge
    # rounds >= 2: a machine re-publishes only if its state moved, and
    # fs_loop_topk caps movers per token — unchanged machines' published
    # k/v are bit-identical to last round's (cacheable), so re-publish is
    # charged at the loop fraction; conference scores stay dense (all K
    # remain readable every round).
    loop_frac = (cfg.fs_loop_topk / K) if cfg.fs_loop_topk else 1.0
    score += ((cfg.fs_loop_rounds - 1)
              * (6 * frac * p_intake + 6 * loop_frac * p_pub_apply
                 + intake_scores + conf_scores))
    return score
