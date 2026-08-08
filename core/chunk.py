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
  fast      -- banded local branch via flex_attention(return_lse=True)
    under a cached static BlockMask (built eagerly, compiler.disable --
    the flex-compile-mask lesson), merged with an eager chunk branch by
    log-sum-exp; algebraically the same joint softmax, differentiable
    end to end. Flex's 128-tile granularity is harmless here: every
    chunk layer runs the uniform w=256 band, not the w=32 staircase
    that made flex unusable for the pyramid. Kill switch:
    FLASH_CHUNK=0 / core.chunk.USE_FAST.
    (Dead end, do not retry: xformers memory_efficient_attention_partial
    returns out+LSE but registers NO autograd formula -- inference-only,
    found on the bench box 2026-08-07.)

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
from core.gated_swa import SlidingWindowAttention, _HAVE_FLEX

USE_FAST = os.environ.get("FLASH_CHUNK", "1") != "0"

_MASK_CACHE = {}        # (kind, T, ...) -> bool mask, built once
_FLEX_MASK_CACHE = {}   # (T, window, device) -> static BlockMask


def rope_at(x, pos, T):
    """Rotate (B, H, N, hd) queries to positions pos (N,); T is a static
    python int bounding pos (passing it avoids a .item() graph break)."""
    from core.rope import _cos_sin
    hd = x.shape[-1]
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


def _swa_flex_mask(T, window, device):
    """Static banded BlockMask, built EAGERLY and cached. Building a
    BlockMask inside a compiled graph silently corrupts data-dependent
    masks (measured on the gated arms, 2026-07-30); these are static per
    (T, window), so build-once-outside is both safe and free."""
    key = (T, window, str(device))
    m = _FLEX_MASK_CACHE.get(key)
    if m is None:
        from torch.nn.attention.flex_attention import create_block_mask

        def mask_mod(b, h, qi, ki):
            return (ki <= qi) & (qi - ki < window)

        m = create_block_mask(mask_mod, None, None, T, T,
                              device=str(device))
        _FLEX_MASK_CACHE[key] = m
    return m


if _HAVE_FLEX:
    _swa_flex_mask = torch.compiler.disable(_swa_flex_mask)


class ChunkAttention(SlidingWindowAttention):
    """Windowed attention + per-layer chunk log (free membership)."""

    FREE = True

    def __init__(self, cfg):
        assert not cfg.diff_attn, "chunk layers do not implement diff attn"
        super().__init__(cfg)
        C, K = cfg.n_embd, cfg.chunk_k
        self.btok = cfg.chunk_btok
        self.k_slots = K
        self.topk = cfg.chunk_topk
        if self.topk:
            assert self.topk <= self.btok, (
                "chunk_topk must not exceed chunk_btok: the first "
                "boundary has only btok candidates")
        if self.FREE:
            # v0.1 writer head: K learned probe vectors ATTENTION-POOL
            # the just-completed block (PMA-style) to form the queries --
            # replaces v0's mean-pool + shared Linear (rung 1), same
            # param names so checkpoints/tests stay stable: slot_emb =
            # probes, wq = the pooling key/value projection.
            self.slot_emb = nn.Parameter(torch.randn(K, C) * 0.02)
            self.wq = nn.Linear(C, C, bias=cfg.bias)
        # chunk MLP: 2C -> C -> 2C bottleneck, residual on the raw mixes
        self.cmlp_up = nn.Linear(2 * C, C, bias=cfg.bias)
        self.cmlp_down = nn.Linear(C, 2 * C, bias=cfg.bias)
        self.cmlp_down.RESIDUAL_SCALE_INIT = True   # zero-init => identity

    # ------------------------------------------------------------ writer
    def _queries(self, x, nblk):
        """Attention-pooled queries: the K probes attend over the just-
        completed block's hidden states (PMA-style), residual on the
        probe. (B, H, S, hd)."""
        B, T, C = x.shape
        H, hd = self.n_head, C // self.n_head
        K, S = self.k_slots, nblk * self.k_slots
        blocks = self.wq(x[:, :nblk * self.btok]).view(
            B, nblk, self.btok, H, hd).permute(0, 1, 3, 2, 4)
        pr = self.slot_emb.view(K, H, hd).permute(1, 0, 2)    # (H,K,hd)
        sc = (pr.unsqueeze(0).unsqueeze(0) @ blocks.transpose(-2, -1)
              ) * hd ** -0.5                          # (B,nblk,H,K,btok)
        qw = pr + torch.softmax(sc, dim=-1) @ blocks  # (B,nblk,H,K,hd)
        return qw.permute(0, 2, 1, 3, 4).reshape(B, H, S, hd)

    def _membership(self, x, k_n, k_r, v, nblk):
        """Free membership: generated queries cross-attend the prefix --
        soft mixture (topk=0) or hard top-k selection with the softmax
        renormalized over the survivors (gradient reaches the selected
        members' scores; the selection itself gets none -- the NSA
        trade). Returns raw chunk (key, value) mixes, (B,H,S,hd)."""
        B, T, C = x.shape
        H, hd = self.n_head, C // self.n_head
        S = nblk * self.k_slots
        qw = self._queries(x, nblk)
        if self.qk_norm:
            qw = rms(qw)
        if self.use_rope:
            # position-aware minting: query b sits at its boundary
            pos = ((torch.arange(nblk, device=x.device) + 1) * self.btok
                   - 1).repeat_interleave(self.k_slots)
            qw = rope_at(qw, pos, nblk * self.btok)
        scores = (qw @ k_r.transpose(-2, -1)) * hd ** -0.5    # (B,H,S,T)
        mask = _writer_mask(T, self.btok, nblk, self.k_slots, x.device)
        scores = scores.masked_fill(~mask, float("-inf"))
        if not self.topk:
            A = torch.softmax(scores, dim=-1)
            return A @ k_n, A @ v
        val, idx = scores.topk(self.topk, dim=-1)             # (B,H,S,k)
        A = torch.softmax(val, dim=-1)

        def take(t):   # (B,H,T,hd) -> selected members (B,H,S,k,hd)
            src = t.unsqueeze(2).expand(B, H, S, T, hd)
            return src.gather(3, idx.unsqueeze(-1).expand(
                B, H, S, self.topk, hd))

        return ((A.unsqueeze(-1) * take(k_n)).sum(3),
                (A.unsqueeze(-1) * take(v)).sum(3))

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

        if (USE_FAST and _HAVE_FLEX and dropout_p == 0.0
                and x.device.type == "cuda"):
            from torch.nn.attention.flex_attention import flex_attention
            mask = _swa_flex_mask(T, self.window, x.device)
            out_l, lse_l = flex_attention(q_r, k_r, v, block_mask=mask,
                                          return_lse=True)
            s_c = (q @ ck.transpose(-2, -1)) * scale          # (B,H,T,S)
            s_c = s_c.masked_fill(~vis, float("-inf"))
            lse_c = torch.logsumexp(s_c.float(), dim=-1)      # (B,H,T)
            out_c = torch.softmax(s_c, dim=-1).nan_to_num() @ cv
            # joint softmax via LSE merge (rows with no visible chunks:
            # lse_c = -inf -> weight 0 -> pure local)
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
    token, from sum_b K*t_b / T -- top-k does not change this: every
    prefix key is still scored) and the K-probe block pooling (~K keys
    per token). blocksum's span means are ~free."""
    nblk = max(T // btok - (1 if T % btok == 0 else 0), 0)
    if nblk <= 0:
        return 0.0
    blk_sum = sum(min(t // btok, nblk) for t in range(T))
    read = k_slots * blk_sum / T
    write = ((k_slots * sum((b + 1) * btok for b in range(nblk)) / T
              + k_slots * nblk * btok / T)
             if free else 0.0)
    return read + write
