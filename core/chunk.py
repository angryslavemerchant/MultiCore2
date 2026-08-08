"""Chunk-latent attention (spec: chunk-latent-attention-spec.md).

Per-layer runtime chunking: every `chunk_btok` tokens the layer mints
`chunk_k` latent KV pairs -- attention-weighted mixtures over ARBITRARY
(non-contiguous) subsets of the layer's own keys/values, selected by
content-generated queries. Chunks accumulate in an append-only log and
are the ONLY long-range pathway: raw self-attention is windowed
(cfg.window), so every prediction that depends on distant context sends
gradient through the writer.

Read path is ONE joint softmax over {local raw tokens} U {chunk log}:
chunks compete with raw tokens for attention mass directly (no NSA-style
branch gates). A token in block c sees chunks minted at boundaries < c
(visibility quantized to block ticks, Transformer-XL geometry).

Positions, NSA-analogue: minting is position-aware (writer queries are
RoPE-rotated to their boundary position and score against the layer's
rotated keys), the log is position-free (chunk KVs are built from
UNROTATED keys/values and read with unrotated queries).

Two execution paths, equal to float tolerance:
  reference -- materialised concat logits, single softmax. Runs
    everywhere; O(T^2) memory, so tests/smoke only at scale.
  fast      -- banded local branch via xformers
    memory_efficient_attention_partial (returns out + LSE) merged with
    an eager chunk branch by log-sum-exp; algebraically the same joint
    softmax. Resolved lazily and numerics-probed on first CUDA call,
    gated_swa-style. Kill switch: FLASH_CHUNK=0 / core.chunk.USE_FAST.

Control arm (`blocksum`, pattern letter B): identical machinery but
membership is positional slicing -- chunk j of block b is the mean of
the j-th contiguous sub-span's keys/values (no queries, no head). This
is NSA's compression branch; the spec's thesis is that free membership
beats it.

Writer head is capacity rung 1: K learned slot embeddings + one shared
Linear on the pooled block representation (a few M params across the
stack, per spec section 8). The chunk MLP is a per-layer 2C->C->2C
bottleneck with residual -- per-layer because hourglass widths differ,
so the spec's cross-layer sharing cannot apply.
"""
import os

import torch
import torch.nn as nn
from torch.nn import functional as F

from core.diffattn import rms
from core.gated_swa import SlidingWindowAttention

USE_FAST = os.environ.get("FLASH_CHUNK", "1") != "0"

_PARTIAL_FN = None      # resolved (fn, layout) or False = resolution failed
_MASK_CACHE = {}        # (kind, T, ...) -> bool mask, built once


def rope_at(x, pos):
    """Rotate (B, H, N, hd) queries to arbitrary positions pos (N,)."""
    from core.rope import _cos_sin
    hd = x.shape[-1]
    T = int(pos.max().item()) + 1
    cos, sin = _cos_sin(T, hd, x.device, x.dtype)     # (1,1,T,hd/2)
    cos, sin = cos[..., pos, :], sin[..., pos, :]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin,
                        x1 * sin + x2 * cos), dim=-1).flatten(-2)


def _band_mask(T, window, device):
    """(T, T) bool: k <= t and t - k <= window - 1 (death(k) = k + W)."""
    key = ("band", T, window, str(device))
    m = _MASK_CACHE.get(key)
    if m is None:
        t = torch.arange(T, device=device)
        m = (t[None, :] <= t[:, None]) & (t[:, None] - t[None, :] < window)
        _MASK_CACHE[key] = m
    return m


def _chunk_vis_mask(T, btok, nblk, k_slots, device):
    """(T, nblk*k_slots) bool: token t sees chunk of boundary b iff
    t // btok > b (chunks visible from the NEXT block on)."""
    key = ("vis", T, btok, nblk, k_slots, str(device))
    m = _MASK_CACHE.get(key)
    if m is None:
        blk_of_t = torch.arange(T, device=device) // btok        # (T,)
        b_of_c = (torch.arange(nblk * k_slots, device=device)
                  // k_slots)                                    # (S,)
        m = blk_of_t[:, None] > b_of_c[None, :]
        _MASK_CACHE[key] = m
    return m


def _writer_mask(T, btok, nblk, k_slots, device):
    """(nblk*k_slots, T) bool: boundary-b queries see tokens < (b+1)*btok."""
    key = ("writer", T, btok, nblk, k_slots, str(device))
    m = _MASK_CACHE.get(key)
    if m is None:
        b_of_q = (torch.arange(nblk * k_slots, device=device)
                  // k_slots)                                    # (S,)
        m = torch.arange(T, device=device)[None, :] < (b_of_q[:, None]
                                                       + 1) * btok
        _MASK_CACHE[key] = m
    return m


def _resolve_partial():
    """xformers partial attention (out + LSE) for the banded local branch,
    numerics-probed against an eager reference before being trusted."""
    global _PARTIAL_FN
    if _PARTIAL_FN is not None:
        return _PARTIAL_FN or None
    _PARTIAL_FN = False
    try:
        from xformers.ops import fmha

        def fn(q, k, v, window):
            # xformers wants (B, T, H, hd)
            bias = fmha.attn_bias.LocalAttentionFromBottomRightMask(
                window_left=window - 1, window_right=0)
            out, lse = fmha.memory_efficient_attention_partial(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                attn_bias=bias)
            return out.transpose(1, 2), lse       # (B,H,T,hd), (B,H,T)

        # probe: banded eager softmax must match (out, lse) to bf16 tol
        B, H, T, hd, W = 1, 2, 64, 32, 8
        q = torch.randn(B, H, T, hd, device="cuda", dtype=torch.bfloat16)
        k, v = torch.randn_like(q), torch.randn_like(q)
        out, lse = fn(q, k, v, W)
        s = (q.float() @ k.float().transpose(-2, -1)) * hd ** -0.5
        s = s.masked_fill(~_band_mask(T, W, q.device), float("-inf"))
        ref = torch.softmax(s, -1) @ v.float()
        assert (out.float() - ref).abs().max() < 3e-2
        assert (lse.float() - torch.logsumexp(s, -1)).abs().max() < 3e-2
        _PARTIAL_FN = fn
    except Exception:
        _PARTIAL_FN = False
    return _PARTIAL_FN or None


class ChunkAttention(SlidingWindowAttention):
    """Windowed attention + per-layer chunk log (free membership)."""

    FREE = True

    def __init__(self, cfg):
        assert not cfg.diff_attn, "chunk layers do not implement diff attn"
        super().__init__(cfg)
        C, K = cfg.n_embd, cfg.chunk_k
        self.btok = cfg.chunk_btok
        self.k_slots = K
        if self.FREE:
            # rung-1 writer head: slot embeddings + shared content proj
            self.slot_emb = nn.Parameter(torch.randn(K, C) * 0.02)
            self.wq = nn.Linear(C, C, bias=cfg.bias)
        # chunk MLP: 2C -> C -> 2C bottleneck, residual on the raw mixes
        self.cmlp_up = nn.Linear(2 * C, C, bias=cfg.bias)
        self.cmlp_down = nn.Linear(C, 2 * C, bias=cfg.bias)
        self.cmlp_down.RESIDUAL_SCALE_INIT = True   # zero-init => identity

    # ------------------------------------------------------------ writer
    def _membership(self, x, k_n, k_r, v, nblk):
        """Free membership: generated queries cross-attend the prefix.
        Returns raw chunk (key, value) mixes, (B, H, S, hd) each."""
        B, T, C = x.shape
        H, hd = self.n_head, C // self.n_head
        S = nblk * self.k_slots
        pooled = x[:, :nblk * self.btok].view(
            B, nblk, self.btok, C).mean(2)                    # (B,nblk,C)
        qw = self.slot_emb + self.wq(pooled)[:, :, None, :]   # (B,nblk,K,C)
        qw = qw.view(B, S, H, hd).transpose(1, 2)             # (B,H,S,hd)
        if self.qk_norm:
            qw = rms(qw)
        if self.use_rope:
            # position-aware minting: query b sits at its boundary
            pos = ((torch.arange(nblk, device=x.device) + 1) * self.btok
                   - 1).repeat_interleave(self.k_slots)
            qw = rope_at(qw, pos)
        scores = (qw @ k_r.transpose(-2, -1)) * hd ** -0.5    # (B,H,S,T)
        mask = _writer_mask(T, self.btok, nblk, self.k_slots, x.device)
        A = torch.softmax(
            scores.masked_fill(~mask, float("-inf")), dim=-1)
        return A @ k_n, A @ v

    def _mint(self, x, k_n, k_r, v, nblk):
        """Write the log: raw membership mixes -> chunk MLP -> chunk KVs
        in the layer's own (unrotated) KV space. (B, H, S, hd) each."""
        B, T, C = x.shape
        H, hd = self.n_head, C // self.n_head
        S = nblk * self.k_slots
        ck_raw, cv_raw = self._membership(x, k_n, k_r, v, nblk)
        h = torch.cat((ck_raw, cv_raw), dim=-1)               # (B,H,S,2hd)
        h = h.transpose(1, 2).reshape(B, S, 2 * C)
        h = h + self.cmlp_down(F.relu(self.cmlp_up(h)).square())
        h = h.view(B, S, H, 2, hd)          # undo the per-head interleave
        ck = h[..., 0, :].transpose(1, 2)                     # (B,H,S,hd)
        cv = h[..., 1, :].transpose(1, 2)
        if self.qk_norm:
            ck = rms(ck)
        return ck, cv

    # -------------------------------------------------------------- read
    def forward(self, x):
        B, T, C = x.shape
        H, hd = self.n_head, C // self.n_head
        dropout_p = self.dropout if self.training else 0.0

        qkv = self.c_attn(x)
        if self.canon_b is not None:
            qkv = qkv + self.canon_b(qkv)
        q, k, v = qkv.split(C, dim=2)
        shp = (B, T, H, hd)
        q = q.view(shp).transpose(1, 2)
        k = k.view(shp).transpose(1, 2)
        v = v.view(shp).transpose(1, 2)
        if self.qk_norm:
            q, k = rms(q), rms(k)
        from core.rope import apply_rope
        if self.use_rope:
            q_r, k_r = apply_rope(q, k)
        else:
            q_r, k_r = q, k

        nblk = T // self.btok
        # the final boundary's chunks are visible to no token in-sequence;
        # skip minting them (training) -- decode-time incremental minting
        # keeps them (v0 decodes by full re-forward, so this is exact).
        if nblk * self.btok == T:
            nblk -= 1
        if nblk <= 0:                       # no complete block: pure SWA
            return super().forward(x)

        ck, cv = self._mint(x, k, k_r, v, nblk)
        scale = hd ** -0.5
        vis = _chunk_vis_mask(T, self.btok, nblk, self.k_slots, x.device)

        if (USE_FAST and dropout_p == 0.0 and x.device.type == "cuda"
                and q.dtype in (torch.float16, torch.bfloat16)):
            fast = _resolve_partial()
            if fast is not None:
                out_l, lse_l = fast(q_r, k_r, v, self.window)
                s_c = (q @ ck.transpose(-2, -1)) * scale      # (B,H,T,S)
                s_c = s_c.masked_fill(~vis, float("-inf"))
                lse_c = torch.logsumexp(s_c.float(), dim=-1)  # (B,H,T)
                out_c = torch.softmax(s_c, dim=-1).nan_to_num() @ cv
                # joint softmax via LSE merge (rows with no visible
                # chunks: lse_c = -inf -> weight 0 -> pure local)
                m = torch.maximum(lse_l.float(), lse_c)
                wl = (lse_l.float() - m).exp()[..., None]
                wc = (lse_c - m).exp().nan_to_num()[..., None]
                y = (out_l.float() * wl + out_c.float() * wc) / (wl + wc)
                y = y.to(x.dtype).transpose(1, 2).reshape(B, T, C)
                return self.resid_dropout(self.c_proj(y))

        # reference: materialised concat logits, one softmax
        s_l = (q_r @ k_r.transpose(-2, -1)) * scale           # (B,H,T,T)
        s_l = s_l.masked_fill(~_band_mask(T, self.window, x.device),
                              float("-inf"))
        s_c = (q @ ck.transpose(-2, -1)) * scale              # (B,H,T,S)
        s_c = s_c.masked_fill(~vis, float("-inf"))
        A = torch.softmax(torch.cat((s_l, s_c), dim=-1), dim=-1)
        if dropout_p > 0:
            A = F.dropout(A, dropout_p)
        y = A[..., :T] @ v + A[..., T:] @ cv
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class BlockSumAttention(ChunkAttention):
    """Control arm: membership = positional slicing (NSA's compression
    branch). Chunk j of block b is the mean of the j-th contiguous
    sub-span's (unrotated) keys/values; same cadence, K, and MLP."""

    FREE = False

    def __init__(self, cfg):
        assert cfg.chunk_btok % cfg.chunk_k == 0, (
            f"blocksum needs chunk_btok % chunk_k == 0, got "
            f"{cfg.chunk_btok} % {cfg.chunk_k}")
        super().__init__(cfg)

    def _membership(self, x, k_n, k_r, v, nblk):
        B, T, C = x.shape
        H, hd = self.n_head, C // self.n_head
        span = self.btok // self.k_slots
        S = nblk * self.k_slots

        def pool(t):
            t = t[:, :, :nblk * self.btok]
            return t.view(B, H, S, span, hd).mean(3)

        return pool(k_n), pool(v)


def chunk_extra_keys(T, btok, k_slots, free):
    """Per-token extra attention keys of a chunk layer, for
    flops_per_token: log reads (K * mean visible boundaries) plus, for
    the free arm, the writer's cross-attention (K * (nblk+1)/2 keys per
    token, from sum_b K*t_b / T). blocksum's span means are ~free."""
    nblk = max(T // btok - (1 if T % btok == 0 else 0), 0)
    if nblk <= 0:
        return 0.0
    blk_sum = sum(min(t // btok, nblk) for t in range(T))
    read = k_slots * blk_sum / T
    write = (k_slots * sum((b + 1) * btok for b in range(nblk)) / T
             if free else 0.0)
    return read + write
