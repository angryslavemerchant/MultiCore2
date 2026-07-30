"""Differential attention (Ye et al. 2024), as a mixin over this repo's
attention classes.

Heads are split into pairs; the pair's output is the DIFFERENCE of two
softmax attention maps over a shared value vector:

    o_p = attn(q_2p, k_2p, v_p) - lambda * attn(q_2p+1, k_2p+1, v_p)

with v_p the concatenation of the pair's two value heads (2*head_dim wide),
a learned per-layer scalar lambda (reparameterised through four head_dim
vectors so its gradient is well-scaled), per-pair parameter-free RMSNorm on
the output, and a (1 - lambda_init) rescale. Common-mode attention mass --
the "attend to everything a little" haze -- subtracts out, which is the
paper's mechanism for sharper long-range retrieval.

The two passes share one mask, so callers hand this mixin an `attend`
closure (built once per forward by core.gated_swa._prepare_attend) rather
than a mask. Score FLOPs match standard attention (half-width heads, two
passes); value application doubles -- flops_per_token accounts the layer at
1.5x its score term.

lambda_init follows the paper's depth schedule 0.8 - 0.6*exp(-0.3*l),
applied by GPT.__init__ via set_depth(l) after the blocks are built.
"""
import math

import torch
import torch.nn as nn
from torch.nn import functional as F


def rms(x):
    """Parameter-free RMS norm over the last dim, computed in fp32."""
    return F.rms_norm(x.float(), (x.shape[-1],)).type_as(x)


class DiffMixin:
    def _init_diff(self, cfg):
        self.diff = cfg.diff_attn
        if not self.diff:
            return
        assert self.n_head % 2 == 0, "diff attention pairs heads"
        hd = cfg.n_embd // cfg.n_head
        for name in ("lambda_q1", "lambda_k1", "lambda_q2", "lambda_k2"):
            setattr(self, name, nn.Parameter(torch.randn(hd) * 0.1))
        self.lambda_init = 0.5      # overwritten by set_depth

    def set_depth(self, layer_idx):
        if getattr(self, "diff", False):
            self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * layer_idx)

    def _diff_attend(self, attend, q, k, v, dropout_p):
        """q, k, v: (B, H, T, hd) post-rope. Returns (B, H/2, T, 2*hd);
        .transpose(1, 2).reshape(B, T, C) recovers the layer layout."""
        B, H, T, hd = v.shape
        vp = (v.view(B, H // 2, 2, T, hd).permute(0, 1, 3, 2, 4)
              .reshape(B, H // 2, T, 2 * hd))
        lam = (torch.exp((self.lambda_q1 * self.lambda_k1).sum().float())
               - torch.exp((self.lambda_q2 * self.lambda_k2).sum().float())
               + self.lambda_init)
        o1 = attend(q[:, 0::2], k[:, 0::2], vp, dropout_p)
        o2 = attend(q[:, 1::2], k[:, 1::2], vp, dropout_p)
        o = o1 - lam.to(o1.dtype) * o2
        return rms(o) * (1.0 - self.lambda_init)
