"""Top-k side-stack: nonlinear attention aggregation (pattern letter T).

The attention output softmax(qk)@V is linear in the values -- it can blend
retrieved tokens but not compute a function of their combination within a
layer. A T layer keeps its full causal attention untouched and replaces the
MLP sublayer with a small transformer over the retrieved set: per head, the
top-k keys by attention score contribute their v slices (head_dim wide, no
down-projection needed), each scaled by a softmax weight renormalized over
the survivors (the chunk_topk precedent -- selection indices themselves get
no gradient, the NSA trade), tagged with head-identity and log-distance
embeddings (v carries neither -- RoPE lives in q/k), pushed through one
shared slice MLP, then a single bidirectional attention block over each
head's (1+k)-item set -- the head's slices plus a per-head CLS seed from
the query token's own state (the query joins the set, so the deleted
MLP's per-token transform happens inside the branch). Sets are PER-HEAD
so SDPA's heads dimension does the batching (the joint (1+H*k)-item
variant cost ~10x on set-attention tile shapes alone, JP profile
2026-08-08); cross-head mixing happens in the read-out instead: the H
CLS items concat back to token dim and go through a 4x MLP (hidden =
4*d, the standard rule in TOKEN units) onto the residual in the MLP's
old slot -- the same place c_proj mixes heads in regular attention.

Width discipline: everything between the v slices and the final MLP runs at
head_dim. Per-slice return to token dim before the set attention would put
~1G MACs/position through it (~2x the rest of the model); at head_dim the
whole branch is ~1% of model FLOPs. See topk-sidestack-spec.md.
"""
import os

import torch
import torch.nn as nn
from torch.nn import functional as F

from core import gated_swa
from core.model import CausalSelfAttention, Canon, make_norm
from core.diffattn import rms

N_DIST_BUCKETS = 16     # log2 buckets cover deltas up to 2^15 (T=4096 uses 13)


class TopKSideAttention(CausalSelfAttention):
    """Standard full causal attention that also hands the branch its
    internals: post-norm/post-rope q,k (so selection sees exactly the
    scores the attention itself used) and the raw v."""

    def __init__(self, cfg):
        assert not cfg.diff_attn, "T layers do not implement diff attn"
        super().__init__(cfg)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        if self.canon_b is not None:
            qkv = qkv + self.canon_b(qkv)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.qk_norm:
            q, k = rms(q), rms(k)
        if self.use_rope:
            from core.rope import apply_rope
            q, k = apply_rope(q, k)
        dropout_p = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=dropout_p)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y)), q, k, v


class SWASideAttention(gated_swa.SlidingWindowAttention):
    """Sliding-window attention (pattern letter R) that also hands the
    branch its internals. The fast kernels don't expose scores, so the
    branch's selection re-derives them in a cheap no-grad banded sweep."""

    def __init__(self, cfg):
        assert not cfg.diff_attn, "R layers do not implement diff attn"
        super().__init__(cfg)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self._qkv(x)
        dropout_p = self.dropout if self.training else 0.0
        if (gated_swa.USE_FLASH and dropout_p == 0.0
                and q.device.type == "cuda"
                and q.dtype in (torch.float16, torch.bfloat16)):
            flash = gated_swa._resolve_flash()
            if flash is not None:
                y = flash(q.transpose(1, 2), k.transpose(1, 2),
                          v.transpose(1, 2), causal=True,
                          window_size=(self.window - 1, 0))
                return (self.resid_dropout(
                    self.c_proj(y.reshape(B, T, C))), q, k, v)
        y = self._run_attn(q, k, v, self._death(x), dropout_p,
                           cache_key=("swa", self.window))
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y)), q, k, v


@torch.no_grad()
def _select_topk(q, k, topk, window=None, chunk=None):
    """Per-head top-k visible-key indices by attention score, (B,H,T,k)
    int64. window=None: causal prefix, streamed over KEY chunks so nothing
    T x T is ever materialised. window=W: last-W band (death(k) = k + W,
    i.e. keys t-W+1..t), streamed over QUERY chunks against their band.
    Scores stay in the matmul dtype (bf16 under autocast) -- topk only
    needs ranks, and the fp32 upcast doubled the sweep's memory traffic
    for nothing but tie-breaks (JP profile 2026-08-08).
    SIDESTACK_SWEEP_FP32=1 restores fp32 scores: the validation gate sets
    it so compiled and eager pick IDENTICAL sets (bf16 reduction-order
    diffs flip near-tied picks between implementations -- benign for
    training, but it breaks the gate's compiled==eager premise)."""
    B, H, T, hd = q.shape
    scale = hd ** -0.5
    fp32 = os.environ.get("SIDESTACK_SWEEP_FP32", "0") == "1"

    def cast(sc):
        return sc.float() if fp32 else sc
    t_idx = torch.arange(T, device=q.device)
    if window is not None:
        # chunk 1024 computes a wastefully wide strip for small windows,
        # but fewer/fatter kernels still beat tight strips at chunk 256
        # (2x, bench 2026-08-08) -- launch overhead dominates band waste
        chunk = chunk or 1024
        out = []
        kk = min(topk, T)
        for s in range(0, T, chunk):
            e = min(s + chunk, T)
            lo = max(0, s - window + 1)
            sc = cast(q[:, :, s:e] @ k[:, :, lo:e].transpose(-2, -1)) * scale
            tq = torch.arange(s, e, device=q.device).view(1, 1, -1, 1)
            tk = torch.arange(lo, e, device=q.device).view(1, 1, 1, -1)
            sc = sc.masked_fill((tk > tq) | (tk <= tq - window),
                                float("-inf"))
            take = min(kk, e - lo)
            _, idx = sc.topk(take, dim=-1)
            idx = idx + lo
            if take < kk:
                # degenerate strip (only window+chunk < topk configs, never
                # the real one): pad with the query's own index — always
                # visible, duplicates are harmless
                pad = tq.expand(B, H, e - s, kk - take).clone()
                idx = torch.cat((idx, pad), dim=-1)
            out.append(idx)
        return torch.cat(out, dim=2)
    chunk = chunk or 1024
    best_val = best_idx = None
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        sc = cast(q @ k[:, :, s:e].transpose(-2, -1)) * scale
        j = torch.arange(s, e, device=q.device)
        sc = sc.masked_fill(j[None, None, None, :] > t_idx[None, None, :, None],
                            float("-inf"))
        kk = min(topk, e - s)
        val, idx = sc.topk(kk, dim=-1)
        idx = idx + s
        if best_val is not None:
            val = torch.cat((best_val, val), dim=-1)
            idx = torch.cat((best_idx, idx), dim=-1)
        if val.shape[-1] > topk:
            val, sel = val.topk(topk, dim=-1)
            idx = idx.gather(-1, sel)
        best_val, best_idx = val, idx
    return best_idx


class SliceMLP(nn.Module):
    """The shared per-slice MLP: relu^2, 4x at head_dim, residual. Named
    c_fc/c_proj so GPT's init loop treats c_proj as a residual projection
    (zero-init under cfg.zero_init => identity at init)."""

    def __init__(self, cfg, hd):
        super().__init__()
        self.c_fc = nn.Linear(hd, 4 * hd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * hd, hd, bias=cfg.bias)
        self.c_proj.RESIDUAL_SCALE_INIT = True

    def forward(self, x):
        return x + self.c_proj(F.relu(self.c_fc(x)).square())


class SideStack(nn.Module):
    """The branch that lives in a T layer's MLP slot."""

    def __init__(self, cfg):
        super().__init__()
        C, H = cfg.n_embd, cfg.n_head
        hd = C // H
        self.n_head, self.hd = H, hd
        self.topk = cfg.side_topk
        # embeddings are stored 3D (an extra leading/middle singleton) so
        # configure_optimizers' "2D under transformer.h. -> Muon" rule
        # routes them to AdamW: orthogonalising a lookup table is nonsense
        self.head_emb = nn.Parameter(torch.randn(H, 1, hd) * 0.02)
        self.dist_emb = nn.Parameter(
            torch.randn(N_DIST_BUCKETS, 1, hd) * 0.02)
        # per-head seeds: one hd-slice of the projected query state per
        # head (2026-08-08 speed redesign: sets are PER-HEAD, 1+k items,
        # so SDPA's heads dimension does the batching -- the joint
        # (1+H*k)-item set ran at ~10x the score cost for tile-shape
        # reasons; cross-head mixing now happens in the read-out concat +
        # final MLP, exactly like c_proj does for regular attention)
        self.cls_proj = nn.Linear(C, C, bias=cfg.bias)
        self.smlp = SliceMLP(cfg, hd)
        self.ln_set = make_norm(cfg, hd)
        self.wqkv = nn.Linear(hd, 3 * hd, bias=cfg.bias)
        self.wo = nn.Linear(hd, hd, bias=cfg.bias)
        self.ln_out = make_norm(cfg, hd)
        # final MLP: 4x rule in TOKEN units on the concatenated per-head
        # read-outs (C -> 4C -> C); c_proj naming => zero-init => the
        # branch is a strict no-op at init
        self.c_fc = nn.Linear(C, 4 * C, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * C, C, bias=cfg.bias)
        self.c_proj.RESIDUAL_SCALE_INIT = True

    def forward(self, x, q, k, v, window=None):
        """x: (B,T,C) post-ln_2 seed. q,k,v: (B,H,T,hd) from the layer's
        own attention (q,k post-rope). window: the host layer's window (R
        layers) or None (T layers). Returns (B,T,C) residual delta."""
        B, H, T, hd = k.shape
        kk = min(self.topk, T)
        idx = _select_topk(q, k, kk, window=window)           # (B,H,T,kk)

        def take(t):        # gather selected tokens: (B,H,T,kk,hd)
            flat = idx.reshape(B, H, T * kk, 1).expand(-1, -1, -1, hd)
            return t.gather(2, flat).view(B, H, T, kk, hd)

        k_sel, v_sel = take(k), take(v)
        # weights recomputed WITH grad on the survivors, causally masked
        # (ghost slots -- queries with fewer than kk visible keys -- are
        # -inf here regardless of what selection returned)
        logits = (q.unsqueeze(-2) @ k_sel.transpose(-2, -1)
                  ).squeeze(-2) * hd ** -0.5                  # (B,H,T,kk)
        t_idx = torch.arange(T, device=x.device)
        valid = idx <= t_idx.view(1, 1, T, 1)
        if window is not None:
            valid = valid & (idx > t_idx.view(1, 1, T, 1) - window)
        w = torch.softmax(
            logits.masked_fill(~valid, float("-inf")), dim=-1)
        delta = (t_idx.view(1, 1, T, 1) - idx).clamp(min=0)
        bucket = (delta + 1).float().log2().floor().long().clamp(
            0, N_DIST_BUCKETS - 1)
        emb = (self.head_emb.view(1, H, 1, 1, hd)
               + self.dist_emb[bucket].squeeze(-2))
        slices = (w.unsqueeze(-1) * v_sel
                  + valid.unsqueeze(-1).to(emb.dtype) * emb)  # (B,H,T,kk,hd)
        items = slices.permute(0, 2, 1, 3, 4)                 # (B,T,H,kk,hd)
        items = self.smlp(items)
        seed = self.cls_proj(x).view(B, T, H, 1, hd)
        seq = torch.cat((seed, items), dim=3)                 # (B,T,H,N,hd)
        N = seq.shape[3]
        qkv = self.wqkv(self.ln_set(seq)).view(B, T, H, N, 3, hd)
        sq, sk, sv = qkv.unbind(dim=4)                        # (B,T,H,N,hd)
        # explicit bmm+softmax: at 17x17 the score matrices are tiny and
        # every fused attention kernel loses to two batched GEMMs
        # (perhead_bmm 2-3ms vs sdpa 8-9ms/layer, bench 2026-08-08)
        sc = (sq @ sk.transpose(-2, -1)) * hd ** -0.5         # (B,T,H,N,N)
        y = torch.softmax(sc, dim=-1) @ sv                    # bidirectional
        seq = seq + self.wo(y)
        cls = self.ln_out(seq[:, :, :, 0])                    # (B,T,H,hd)
        return self.c_proj(
            F.relu(self.c_fc(cls.reshape(B, T, H * hd))).square())


class TopKSideBlock(nn.Module):
    """Pre-LN block whose MLP slot holds the side-stack. Mirrors
    core.model.Block exactly on the attention half. attn_key picks the
    host attention: "topkside" = full causal (T), "swaside" = sliding
    window (R) — the branch is identical, only the candidate set (and
    therefore the selection sweep and validity band) differs."""

    def __init__(self, cfg, attn_key="topkside"):
        super().__init__()
        self.use_canon = cfg.canon
        if cfg.canon:
            self.canon_a = Canon(cfg.n_embd)
            self.canon_c = Canon(cfg.n_embd)
        self.ln_1 = make_norm(cfg, cfg.n_embd)
        if attn_key == "swaside":
            self.attn = SWASideAttention(cfg)
            self.side_window = cfg.window
        else:
            self.attn = TopKSideAttention(cfg)
            self.side_window = None
        self.ln_2 = make_norm(cfg, cfg.n_embd)
        self.side = SideStack(cfg)

    def forward(self, x):
        if self.use_canon:
            x = x + self.canon_a(x)
        y, q, k, v = self.attn(self.ln_1(x))
        x = x + y
        if self.use_canon:
            x = x + self.canon_c(x)
        x = x + self.side(self.ln_2(x), q, k, v,
                          window=self.side_window)
        return x


def side_extra_flops(cfg, w, T, window=None):
    """Branch fwd+bwd FLOPs per token beyond what 6*N already counts,
    at layer width w. 6*N covers each branch param used ONCE per token;
    the slice MLP runs H*k times and the set-attention projections N
    times, so their extra uses are charged here, plus the set-attention
    scores (12*d*keys convention at d=hd over N items) and the no-grad
    selection sweep (one fwd qk matmul: 2 FLOPs * hd * H * avg keys per
    token = w*T causal, ~w*2W banded)."""
    H = cfg.n_head
    hd = w // H
    kk = min(cfg.side_topk, T)
    n = 1 + kk                       # per-head set length
    sel = w * (T if window is None else 2 * window)
    slice_mlp = 6 * 8 * hd * hd * (H * kk - 1)
    set_proj = 6 * 4 * hd * hd * (H * n - 1)
    set_scores = 12 * hd * H * n * n
    return sel + slice_mlp + set_proj + set_scores
