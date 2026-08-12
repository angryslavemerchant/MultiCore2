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


class MachineMLP(nn.Module):
    """Private per-machine MLP, relu^2, residual is applied by the caller."""

    def __init__(self, cfg):
        super().__init__()
        K, d = cfg.fs_n_machines, cfg.fs_d_machine
        self.fc = MachineLinear(K, d, cfg.fs_mlp_mult * d)
        self.proj = MachineLinear(K, cfg.fs_mlp_mult * d, d)

    def forward(self, x):
        return self.proj(F.relu(self.fc(x)).square())


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
        self.w_tkv = TokenLinear(K, C, 2 * d)  # token intake k,v fused
        self.w_pkv = MachineLinear(K, d, 2 * d)  # private-channel k,v fused
        self.w_out = MachineLinear(K, d, d)
        self.mlp = MachineMLP(cfg)
        self.ln_mlp = make_norm(cfg, d)

    def _heads(self, x):                      # (B,K,T,d) -> (B*K,H,T,hd)
        B, K, T, d = x.shape
        return (x.reshape(B * K, T, self.H, self.hd).transpose(1, 2))

    def forward(self, x_tok, c_prev):
        B, K, T, d = c_prev.shape
        q = self._heads(self.w_q(self.ln_q(c_prev)))
        k_tok, v_tok = self.w_tkv(x_tok).chunk(2, dim=-1)
        k_prv, v_prv = self.w_pkv(self.ln_p(c_prev)).chunk(2, dim=-1)
        k_tok, v_tok = self._heads(k_tok), self._heads(v_tok)
        k_prv, v_prv = self._heads(k_prv), self._heads(v_prv)
        if self.use_rope:
            q, k_tok = apply_rope(q, k_tok)
            k_prv = apply_rope(k_prv, k_prv)[0]
        if self.window is not None:
            y = self._banded(q, k_tok, v_tok, k_prv, v_prv)
        else:
            k = torch.cat((k_tok, k_prv), dim=2)  # (B*K,H,2T,hd)
            v = torch.cat((v_tok, v_prv), dim=2)
            mask = _intake_mask(T, None, x_tok.device)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = (y.transpose(1, 2).reshape(B, K, T, d))
        s = c_prev + self.w_out(y)
        return s + self.mlp(self.ln_mlp(s))

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
        self.w_qkv = MachineLinear(K, d, 3 * d)  # interface q,k,v fused
        self.conf_out = MachineLinear(K, d, d)
        # write-back: per-machine gate on [x_t ; c_k], zero-init W_O so the
        # token residual is untouched at init
        self.gate_x = nn.Parameter(torch.randn(K, C) * 0.02)
        self.gate_c = nn.Parameter(torch.randn(K, d) * 0.02)
        self.gate_b = nn.Parameter(torch.zeros(K))
        self.w_o = MachineLinear(K, d, C, zero=True)

    def init_channel(self, B, T, device=None, dtype=None):
        """Seed conclusions for the first machine block: s0 at every t."""
        assert self.s0 is not None, "only the seed block can init the channel"
        s0 = self.s0.to(device=device, dtype=dtype)
        return s0[None, :, None, :].expand(B, -1, T, -1).contiguous()

    def forward(self, x, c_prev):
        B, T, C = x.shape
        K, d = self.K, self.d
        s = self.backend(x, c_prev)                       # stroke 1
        sn = self.ln_iface(s)
        q, ks, v = self.w_qkv(sn).chunk(3, dim=-1)        # stroke 2
        k = (self.anchor[None, :, None, :]
             + self.addr_mix[None, :, None, None] * ks)

        def heads(t):     # (B,K,T,d) -> (B*T, H, K, hd): attention over K
            return (t.permute(0, 2, 1, 3)
                    .reshape(B * T, K, self.H, self.hd).transpose(1, 2))

        y = F.scaled_dot_product_attention(heads(q), heads(k), heads(v))
        y = (y.transpose(1, 2).reshape(B, T, K, d).permute(0, 2, 1, 3))
        c = s + self.conf_out(y)                          # stroke 3
        g = torch.sigmoid(                                # stroke 4
            torch.einsum("btc,kc->btk", x, self.gate_x)
            + torch.einsum("bktd,kd->btk", c, self.gate_c)
            + self.gate_b)
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
    """Per-M-layer fwd+bwd score FLOPs per token beyond what 6*N counts
    (every machine param is used once per token, so projections and the
    token FFN are already covered): intake = K machines' scores over two
    sources of avg_t min(t, w) visible keys each (12*d*keys convention),
    conference = K x K at d_machine."""
    K, d, w = cfg.fs_n_machines, cfg.fs_d_machine, cfg.fs_window
    if cfg.fs_backend == "attn":
        avg = T
    else:
        avg = T if T <= w else (w * (w + 1) / 2 + (T - w) * w) / T
    return K * 12 * d * (2 * avg + K)
