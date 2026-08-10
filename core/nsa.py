"""NSA with register banks (2026-08-09, the one-night arm).

Every layer is DeepSeek-NSA's trinity on top of the project trunk:

  1. swa: 128-token sliding window over the stream (RoPE'd, flash
     kernels — the pyramid/MiMo local branch, unchanged);
  2. cmp: dense attention over mean-pooled `nsa_block`-token block
     summaries of the whole past (position-free — content addressing);
  3. slc: fine attention over the top `nsa_topk` blocks scored by the
     cmp branch's own attention mass (per-token selection, realised as
     dense-with-mask SDPA — the chunkv2 fetch lesson: masks over static
     shapes compile; gathers don't).

combined per head by sigmoid gates computed from the token.

REGISTERS: `nsa_nreg` learned vectors sit "before" the stream, grouped
into nsa_block-sized blocks. They are pure parameters — never processed
by the stack, never attending to anything (they have no queries). Each
layer projects them with its own K/V weights. Their block summaries are
ALWAYS visible to the cmp branch (so every register receives score
gradient every step — the anti-PKM-death insurance), and their blocks
compete with token blocks for slc slots. RoPE is not applied to
registers or to the cmp/slc branches: retrieval is positionless.

Causality: a query in token block b sees token blocks strictly < b in
cmp/slc (its own block is covered by swa), so no per-token masking is
needed inside selected blocks. Registers are visible to everyone.

Selection scores are detached (selection is an argmax; gradients reach
the scorer through the cmp branch, which shares it).
"""
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from core.gated_swa import SlidingWindowAttention, USE_FLASH, \
    _resolve_flash
from core.diffattn import rms


class NSARegisterAttention(SlidingWindowAttention):
    def __init__(self, cfg):
        super().__init__(cfg)
        assert not cfg.diff_attn, "nsa: no diff-attn support"
        assert cfg.nsa_nreg % cfg.nsa_block == 0, \
            (cfg.nsa_nreg, cfg.nsa_block)
        self.nb = cfg.nsa_block
        self.topk = cfg.nsa_topk
        self.n_reg = cfg.nsa_nreg
        C = cfg.n_embd
        self.registers = nn.Parameter(
            torch.empty(cfg.nsa_nreg, C).normal_(0, 0.02))
        # per-head, per-token branch gates: [swa, cmp, slc]
        self.w_gate = nn.Linear(C, 3 * cfg.n_head, bias=cfg.bias)
        # eval-time pseudo-ablation (the hier disable_memory trick):
        # mask register blocks out of cmp scores AND the selection pool.
        # Set on the module post-load; never used in training.
        self.disable_registers = False
        # eval-time telemetry: {"reg_cmp_mass", "reg_sel_frac"} written
        # each forward when not training (cheap; both are scalars)
        self.reg_stats = {}

    def _reg_kv(self, B):
        """Project registers with this layer's K/V weights: (B,H,R,hd)."""
        C = self.registers.shape[1]
        W = self.c_attn.weight                       # (3C, C)
        k = self.registers @ W[C:2 * C].t()
        v = self.registers @ W[2 * C:].t()
        if self.c_attn.bias is not None:
            k = k + self.c_attn.bias[C:2 * C]
            v = v + self.c_attn.bias[2 * C:]
        shp = (self.n_reg, self.n_head, C // self.n_head)
        k = k.view(shp).transpose(0, 1)              # (H,R,hd)
        v = v.view(shp).transpose(0, 1)
        if self.qk_norm:
            k = rms(k)
        return (k.unsqueeze(0).expand(B, -1, -1, -1),
                v.unsqueeze(0).expand(B, -1, -1, -1))

    def _qkv_nope(self, x):
        """qkv WITHOUT rope (rope is applied per-branch)."""
        B, T, C = x.shape
        qkv = self.c_attn(x)
        if self.canon_b is not None:
            qkv = qkv + self.canon_b(qkv)
        q, k, v = qkv.split(C, dim=2)
        shp = (B, T, self.n_head, C // self.n_head)
        q = q.view(shp).transpose(1, 2)
        k = k.view(shp).transpose(1, 2)
        v = v.view(shp).transpose(1, 2)
        if self.qk_norm:
            q, k = rms(q), rms(k)
        return q, k, v

    def forward(self, x):
        B, T, C = x.shape
        nb, hd = self.nb, C // self.n_head
        assert T % nb == 0, (T, nb)
        NB, NR = T // nb, self.n_reg // nb
        q, k, v = self._qkv_nope(x)
        k_reg, v_reg = self._reg_kv(B)
        dropout_p = self.dropout if self.training else 0.0

        # ---- branch 1: swa (RoPE'd local window) ----------------------
        if self.use_rope:
            from core.rope import apply_rope
            q_r, k_r = apply_rope(q, k)
        else:
            q_r, k_r = q, k
        flash = _resolve_flash() if (
            USE_FLASH and dropout_p == 0.0 and q.device.type == "cuda"
            and q.dtype in (torch.float16, torch.bfloat16)) else None
        if flash is not None:
            y_swa = flash(q_r.transpose(1, 2), k_r.transpose(1, 2),
                          v.transpose(1, 2), causal=True,
                          window_size=(self.window - 1, 0)
                          ).transpose(1, 2)
        else:
            y_swa = self._run_attn(q_r, k_r, v, self._death(x), dropout_p,
                                   cache_key=("swa", self.window))

        # ---- branch 2: cmp (block summaries + register summaries) -----
        k_blk = k.view(B, self.n_head, NB, nb, hd).mean(3)
        v_blk = v.view(B, self.n_head, NB, nb, hd).mean(3)
        k_rblk = k_reg.view(B, self.n_head, NR, nb, hd).mean(3)
        v_rblk = v_reg.view(B, self.n_head, NR, nb, hd).mean(3)
        K_cmp = torch.cat((k_rblk, k_blk), dim=2)    # (B,H,NR+NB,hd)
        V_cmp = torch.cat((v_rblk, v_blk), dim=2)
        scores = q @ K_cmp.transpose(-1, -2) / math.sqrt(hd)
        blk_id = torch.arange(T, device=x.device) // nb      # (T,)
        tok_ok = (torch.arange(NB, device=x.device)
                  < blk_id.unsqueeze(-1))            # (T,NB) strictly past
        reg_ok = tok_ok.new_full((T, NR), not self.disable_registers)
        allowed = torch.cat((reg_ok, tok_ok), dim=1)
        scores = scores.masked_fill(~allowed, float("-inf"))
        p = F.softmax(scores, dim=-1)                # (B,H,T,NR+NB)
        if self.disable_registers:
            # first-block queries have NO valid cmp keys -> softmax NaN
            p = torch.nan_to_num(p, nan=0.0)
        y_cmp = p @ V_cmp

        # ---- branch 3: slc (top-k blocks, dense-with-mask fetch) ------
        sel = p.detach().sum(1)                      # (B,T,NR+NB)
        idx = sel.topk(self.topk, dim=-1).indices
        vis = torch.zeros_like(sel, dtype=torch.bool
                               ).scatter_(-1, idx, True) & allowed
        if (not self.training
                and not torch.compiler.is_compiling()):
            self.reg_stats = {
                "reg_cmp_mass": float(p[..., :NR].sum(-1).mean()),
                # fraction of SURVIVING selections that are registers
                # (raw topk picks zero-score fillers that vis discards)
                "reg_sel_frac": float(vis[..., :NR].sum()
                                      / vis.sum().clamp(min=1))}
        empty = None
        if self.disable_registers:
            # rows with nothing visible (first block when registers are
            # disabled): give them register block 0 to keep SDPA finite
            # (registers are input-independent; output zeroed below).
            # With registers enabled this cannot happen — no extra work
            # on the trained/compiled path.
            empty = ~vis.any(-1)                     # (B,T)
            vis[..., 0] = vis[..., 0] | empty
        vis = vis.repeat_interleave(nb, dim=-1)      # (B,T,R+T)
        K_all = torch.cat((k_reg, k), dim=2)
        V_all = torch.cat((v_reg, v), dim=2)
        y_slc = F.scaled_dot_product_attention(
            q, K_all, V_all, attn_mask=vis.unsqueeze(1),
            dropout_p=dropout_p)
        if empty is not None:
            y_slc = y_slc * (~empty).unsqueeze(1).unsqueeze(-1)

        # ---- per-head sigmoid gates -----------------------------------
        g = torch.sigmoid(self.w_gate(x))            # (B,T,3H)
        g = g.view(B, T, 3, self.n_head).permute(2, 0, 3, 1)  # (3,B,H,T)
        y = (g[0].unsqueeze(-1) * y_swa
             + g[1].unsqueeze(-1) * y_cmp
             + g[2].unsqueeze(-1) * y_slc)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


def nsa_extra_keys(T, nb, topk, n_reg, window):
    """Executed visible-key count per query for one NSA layer (honest
    dense-with-mask accounting: the slc branch PAYS T + n_reg scores
    even though only topk*nb survive the mask)."""
    cmp = (T / nb) / 2 + n_reg / nb
    slc = T + n_reg
    swa = window if T > window else T
    return swa + cmp + slc
