"""Chunk-latent attention v0.2 (pattern letter N): recursion + dedup +
raw member fetch. Extends core/chunk.py's v0.1 (pattern K), which stays
untouched so v0/v0.1 checkpoints keep loading.

Three mechanisms, each attacking a measured v0/v0.1 failure
(2026-08-08 bench: flat gist memory, zero exact fetch, stale values
stay live):

1. RECURSIVE MINTING (fully sequential -- the honest version, no
   wavefront/lag affordances). Boundary b's writer candidates are the
   raw prefix (tokens < (b+1)*btok, scored position-aware with the
   RoPE'd writer query) UNION all previously minted chunks (scored
   position-free with the unrotated query, mirroring the read path's
   duality). A chunk that selects an older chunk folds in its LATENT
   only -- references are never transitively expanded into raw tokens.
   The loop runs eagerly under torch.compiler.disable: per-boundary
   shapes grow, and 31 growing graphs would either recompile forever or
   fragment the outer graph. Cost measured on-box before any run.

2. SOFT DEDUP (read-side). Per-head learned lam (init 0 = exactly the
   append-only v0.1 behavior; the model must EARN dedup): each chunk's
   read logit is penalized by lam * relu(max cosine sim to any NEWER
   chunk), so a superseded chunk fades in proportion to how well a
   newer one covers it. Newest-wins -- the cheap approximation of COW's
   merge, which won update-margin. Fully differentiable, no threshold.

3. RAW MEMBER FETCH (NSA's fine branch, the v0.2 thesis: pointers
   select right, summaries smear). Each query takes its top
   chunk_fetch_n chunks by (penalized, visibility-masked) log score and
   attends the union of their RAW-token pointers as a third branch of
   the joint softmax -- position-free (unrotated q against unrotated
   k), consistent with the log and immune to RoPE extrapolation.
   Chunk->chunk pointers are NOT fetched (latent-only, see 1): content
   reachable only through a summary-of-summary stays gist.
   Two accepted biases, identical in fast and reference paths so parity
   holds: duplicate pointers (two selected chunks citing the same
   token) and pointers landing inside the local band double that key's
   softmax mass.

Read path = ONE joint softmax over {local band} U {chunk log} U
{fetched raws}: reference materialises the concat (tests/smoke only),
fast path LSE-merges flex-banded local + eager chunk + eager fetch
branches (the 3-way generalisation of v0.1's 2-way merge).
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

from core.diffattn import rms
from core.gated_swa import SlidingWindowAttention, _HAVE_FLEX
import core.chunk as chunk_mod
from core.chunk import (ChunkAttention, _band_mask, _chunk_vis_mask,
                        _swa_flex_mask, rope_at)


class ChunkV2Attention(ChunkAttention):
    """Windowed attention + recursive chunk log + dedup + raw fetch."""

    def __init__(self, cfg):
        assert cfg.chunk_topk > 0, (
            "chunkv2 needs hard top-k membership: the pointer sets ARE "
            "the fetch mechanism")
        super().__init__(cfg)
        self.fetch_n = cfg.chunk_fetch_n
        # dedup strength, per head; init 0 == append-only v0.1
        self.lam = nn.Parameter(torch.zeros(cfg.n_head))

    # ------------------------------------------------------------ writer
    def _mint_recursive(self, x, k_n, k_r, v, nblk):
        """Sequential mint. Returns (ck, cv, ptr, ok): the log
        (B,H,S,hd), each chunk's member pointers (B,H,S,kk) as indices
        into cat(raw, log) candidate space, and ok = pointer-is-raw
        (False marks a chunk->chunk reference, never fetched).

        STATIC SHAPES on purpose: every boundary scores the FULL
        (T + S)-wide candidate set with -inf masks over
        not-yet-visible raw tokens / not-yet-minted chunks, and writes
        into a preallocated log via (differentiable, out-of-place)
        index_copy. Identical math to prefix-sliced scoring -- masked
        softmax over -inf is the sliced softmax -- but every iteration
        of the loop has the same shapes, so torch.compile unrolls the
        31 iterations into fused kernels instead of the eager
        Python-dispatch loop that a compiler.disable'd dynamic-shape
        version was stuck with (58ms fwd/layer measured). The wasted
        masked FLOPs are writer-score-sized, i.e. noise."""
        B, T, C = x.shape
        H, hd = self.n_head, C // self.n_head
        K, kk = self.k_slots, self.topk
        S = nblk * K
        scale = hd ** -0.5
        qw = self._queries(x, nblk)                       # (B,H,S,hd)
        if self.qk_norm:
            qw = rms(qw)
        qw_n = qw
        if self.use_rope:
            pos = ((torch.arange(nblk, device=x.device) + 1) * self.btok
                   - 1).repeat_interleave(K)
            qw_r = rope_at(qw, pos, nblk * self.btok)
        else:
            qw_r = qw
        tcol = torch.arange(T, device=x.device)           # raw col ids
        scol = torch.arange(S, device=x.device)           # log slot ids
        ck_log = k_n.new_zeros(B, H, S, hd)
        cv_log = v.new_zeros(B, H, S, hd)
        ptrs, oks = [], []
        for b in range(nblk):
            p = (b + 1) * self.btok                # visible raw prefix
            qr = qw_r[:, :, b * K:(b + 1) * K]     # (B,H,K,hd)
            qn = qw_n[:, :, b * K:(b + 1) * K]
            s_raw = (qr @ k_r.transpose(-2, -1)) * scale   # (B,H,K,T)
            s_chk = (qn @ ck_log.transpose(-2, -1)) * scale  # (B,H,K,S)
            s = torch.cat((s_raw, s_chk), dim=-1)
            s = s * self.gain.view(1, -1, 1, 1)
            # mask AFTER the gain: gain * -inf backpropagates
            # 0 * -inf = NaN into the gain grad
            dead = torch.cat((tcol >= p, scol >= b * K))
            s = s.masked_fill(dead, float("-inf"))
            val, idx = s.topk(kk, dim=-1)          # (B,H,K,kk)
            A = torch.softmax(val, dim=-1)
            cand_k = torch.cat((k_n, ck_log), dim=2)   # (B,H,T+S,hd)
            cand_v = torch.cat((v, cv_log), dim=2)

            def take(t):   # (B,H,T+S,hd) -> selected (B,H,K,kk,hd)
                # dim-2 gather: backward is a scatter-add into t, NOT a
                # zeros_like of an expanded (B,H,K,T+S,hd) source
                idxe = idx.reshape(B, H, K * kk, 1).expand(
                    B, H, K * kk, hd)
                return t.gather(2, idxe).view(B, H, K, kk, hd)

            ck_raw = (A.unsqueeze(-1) * take(cand_k)).sum(3)
            cv_raw = (A.unsqueeze(-1) * take(cand_v)).sum(3)
            h = torch.cat((ck_raw, cv_raw), dim=-1)       # (B,H,K,2hd)
            h = h.transpose(1, 2).reshape(B, K, 2 * C)
            h = h + self.cmlp_down(F.relu(self.cmlp_up(h)).square())
            h = h.view(B, K, H, 2, hd)
            ckb = h[..., 0, :].transpose(1, 2)            # (B,H,K,hd)
            cvb = h[..., 1, :].transpose(1, 2)
            if self.qk_norm:
                ckb = rms(ckb)
            slots = scol[b * K:(b + 1) * K]
            ck_log = ck_log.index_copy(2, slots, ckb)
            cv_log = cv_log.index_copy(2, slots, cvb)
            ptrs.append(idx)
            oks.append(idx < p)                     # raw iff < T & < p
        ptr = torch.cat(ptrs, dim=2).clamp(max=T - 1)     # refs masked by ok
        ok = torch.cat(oks, dim=2)
        return ck_log, cv_log, ptr, ok

    # -------------------------------------------------------------- read
    def _dedup_penalty(self, ck):
        """(B,H,1,S) logit penalty: lam_h * relu(max cos sim to any
        newer chunk). Rows with no newer chunk (last boundary) get 0."""
        B, H, S, hd = ck.shape
        K = self.k_slots
        chd = F.normalize(ck.float(), dim=-1)
        sim = chd @ chd.transpose(-2, -1)                 # (B,H,S,S)
        b_of = torch.arange(S, device=ck.device) // K
        newer = b_of[None, :] > b_of[:, None]
        sim = sim.masked_fill(~newer, float("-inf"))
        pen = F.relu(sim.amax(-1))                        # (B,H,S)
        return (self.lam.view(1, -1, 1) * pen).unsqueeze(2).to(ck.dtype)

    def _fetch(self, q, k_n, v, s_c, ptr, ok):
        """Raw member fetch: top fetch_n chunks per query by log score,
        attend their raw pointers. Returns (s_f, vf) = fetch logits
        (B,H,T,S*kk), -inf outside the query's selection, and the
        per-CHUNK member values (B,H,S*kk,hd) shared by all queries.

        Dense-with-mask on purpose: the per-query sparse formulation
        (gather each query's 64 members, scatter-add grads back) put
        78% of the training step into one atomic scatter kernel --
        every query fetches the same few popular chunks, so their
        members' grad rows serialize (measured 868ms/iter, value-
        dependent: appears only once training sharpens selection).
        Scoring ALL chunks' members densely and masking non-selected
        ones to -inf is the same math with GEMM gradients; members are
        gathered once per chunk (S*kk rows, contention-free)."""
        B, H, T, S = s_c.shape
        hd = q.shape[-1]
        kk = self.topk
        n = min(self.fetch_n, S)
        fv, fi = s_c.topk(n, dim=-1)                      # (B,H,T,n)
        sel_ok = torch.isfinite(fv)                       # visible chunk

        pidx = ptr.reshape(B, H, S * kk, 1).expand(B, H, S * kk, hd)
        mem_k = k_n.gather(2, pidx)                       # (B,H,S*kk,hd)
        mem_v = v.gather(2, pidx)
        s_f = (q @ mem_k.transpose(-2, -1)) * hd ** -0.5  # (B,H,T,S*kk)
        # member visible iff its chunk is in this query's top-n AND the
        # pointer is raw (chunk->chunk references are never fetched)
        sel = torch.zeros(B, H, T, S, dtype=torch.bool, device=q.device)
        sel.scatter_(3, fi, sel_ok)
        okm = sel.unsqueeze(-1) & ok.reshape(B, H, 1, S, kk)
        s_f = s_f.view(B, H, T, S, kk).masked_fill(
            ~okm, float("-inf")).view(B, H, T, S * kk)
        return s_f, mem_v

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
        if nblk * self.btok == T:
            nblk -= 1
        if nblk <= 0:                       # no complete block: pure SWA
            return SlidingWindowAttention.forward(self, x)

        ck, cv, ptr, ok = self._mint_recursive(x, k, k_r, v, nblk)
        # diagnostics/tests: which members each chunk cited, raw or ref
        self.last_mint = (ptr.detach(), ok.detach())
        scale = hd ** -0.5
        vis = _chunk_vis_mask(T, self.btok, nblk, self.k_slots, x.device)
        s_c = (q @ ck.transpose(-2, -1)) * scale          # (B,H,T,S)
        s_c = s_c - self._dedup_penalty(ck)
        s_c = s_c.masked_fill(~vis, float("-inf"))
        if self.fetch_n:
            s_f, vf = self._fetch(q, k, v, s_c, ptr, ok)

        if (chunk_mod.USE_FAST and _HAVE_FLEX and dropout_p == 0.0
                and x.device.type == "cuda"):
            from torch.nn.attention.flex_attention import flex_attention
            mask = _swa_flex_mask(T, self.window, x.device)
            out_l, lse_l = flex_attention(q_r, k_r, v, block_mask=mask,
                                          return_lse=True)
            lse_c = torch.logsumexp(s_c.float(), dim=-1)  # (B,H,T)
            out_c = torch.softmax(s_c, dim=-1).nan_to_num() @ cv
            m = torch.maximum(lse_l.float(), lse_c)
            if self.fetch_n:
                lse_f = torch.logsumexp(s_f.float(), dim=-1)
                A_f = torch.softmax(s_f, dim=-1).nan_to_num()
                out_f = A_f @ vf
                m = torch.maximum(m, lse_f)
            wl = (lse_l.float() - m).exp()[..., None]
            wc = (lse_c - m).exp().nan_to_num()[..., None]
            y = out_l.float() * wl + out_c.float() * wc
            wsum = wl + wc
            if self.fetch_n:
                wf = (lse_f - m).exp().nan_to_num()[..., None]
                y = y + out_f.float() * wf
                wsum = wsum + wf
            y = (y / wsum).to(x.dtype).transpose(1, 2).reshape(B, T, C)
            return self.resid_dropout(self.c_proj(y))

        # reference: materialised concat logits, one softmax
        s_l = (q_r @ k_r.transpose(-2, -1)) * scale       # (B,H,T,T)
        s_l = s_l.masked_fill(~_band_mask(T, self.window, x.device),
                              float("-inf"))
        parts = [s_l, s_c] + ([s_f] if self.fetch_n else [])
        A = torch.softmax(torch.cat(parts, dim=-1), dim=-1)
        if dropout_p > 0:
            A = F.dropout(A, dropout_p)
        y = A[..., :T] @ v + A[..., T:T + s_c.shape[-1]] @ cv
        if self.fetch_n:
            y = y + A[..., T + s_c.shape[-1]:] @ vf
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.resid_dropout(self.c_proj(y))


def chunkv2_extra_keys(T, btok, k_slots, topk, fetch_n):
    """Per-token extra attention keys of a chunkv2 layer, for
    flops_per_token. Log reads as v0.1; writer additionally scores the
    growing chunk log (recursion); the dense-with-mask fetch scores ALL
    chunks' members (S*topk keys per token wherever a chunk is visible
    -- fetch_n only shapes the mask, not the compute); dedup's S^2
    cosine matrix amortises to ~S^2/(3T) key-equivalents."""
    nblk = max(T // btok - (1 if T % btok == 0 else 0), 0)
    if nblk <= 0:
        return 0.0
    S = nblk * k_slots
    blk_sum = sum(min(t // btok, nblk) for t in range(T))
    read = k_slots * blk_sum / T
    write = k_slots * sum((b + 1) * btok + b * k_slots
                          for b in range(nblk)) / T
    write += k_slots * nblk * btok / T                # PMA block pooling
    fetch = ((S * topk if fetch_n else 0)
             * sum(1 for t in range(T) if t // btok >= 1) / T)
    dedup = S * S / (3 * T)
    return read + write + fetch + dedup
