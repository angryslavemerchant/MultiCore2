"""A faithful GPT-2, with seams for swapping in a new architecture.

Architecture is GPT-2 exactly: learned position embeddings, pre-LN blocks,
GELU 4x MLP, tied input/output embeddings, residual-projection init scaled
by 1/sqrt(2*n_layer). The only departures are implementation-level:
F.scaled_dot_product_attention (flash) instead of materialised score
matrices, and vocab padded to 50304 (the NeoX vocab this project's token
cache uses; a multiple of 64 keeps matmuls on fast paths).

THE SEAMS: a block is assembled from ATTENTIONS[cfg.attn] and
MLPS[cfg.mlp], and the block class itself from BLOCKS[cfg.block]. The new
architecture registers its pieces here and selects them by name from the
training script -- nothing else in the pipeline changes, so a baseline run
and a variant run differ by exactly one CLI flag.

Compute matching: `flops_per_token(T)` is the analytic fwd+bwd cost
(nanoGPT's convention: 6*N + 12*L*d*T). The training script multiplies by
tokens consumed and logs cumulative FLOPs; two runs are compute-matched by
stopping at the same total, not the same step count. A variant whose
per-token cost differs from the baseline's MUST override this estimate to
stay honestly matched.
"""
import math
from dataclasses import dataclass, asdict, replace

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True          # GPT-2 has biases everywhere
    block: str = "gpt2"        # BLOCKS registry key
    attn: str = "causal"       # ATTENTIONS registry key (all layers...)
    mlp: str = "gelu"          # MLPS registry key
    # ...unless attn_pattern is set: one char per layer, F=full causal,
    # S=sliding window, G=admission-gated window. E.g. the 2:1 sandwich
    # "FGGGGFFGGGGF". Empty = every layer uses cfg.attn.
    attn_pattern: str = ""
    window: int = 512          # total window budget for S and G layers
    n_gates: int = 8           # G layers: FIFO gates share (window - recent_band)
    lb_coef: float = 0.0       # weight of the router load-balance aux loss
    # G layers: reserve this many of `window` as a GUARANTEED recent band
    # (plain sliding visibility); the gates manage only the remainder.
    # Phase-1 lesson (2026-07-28): pure gates guarantee just window/n_gates
    # recent tokens and lost 0.04 nats to recency starvation at T=1024.
    recent_band: int = 0
    # "learned" = GPT-2 wpe (capped at block_size); "rope" = rotary, no
    # position parameters — the phase-2 long-context default.
    pos: str = "learned"
    # Hourglass (slice-carry bottleneck), hg_frac > 0 enables. The residual
    # stream stays n_embd wide everywhere; layer l reads/writes only the
    # FIRST layer_widths()[l] dims (trailing dims carry forward untouched —
    # no learned transition projections). Widths fall by a constant
    # per-layer ratio from n_embd (layer 1) to hg_frac*n_embd at layer
    # hg_bneck, hold there for hg_mid extra flat layers, then rise by a
    # constant ratio back to n_embd at layer n_layer. Total depth is
    # n_layer + hg_mid. d_base (n_embd) comes from
    # scripts/hourglass_match.py, which parameter-matches the dense arm.
    hg_frac: float = 0.0
    hg_bneck: int = 8
    hg_mid: int = 0

    def n_layer_total(self):
        return self.n_layer + (self.hg_mid if self.hg_frac else 0)

    def layer_widths(self, rnd=24):
        """Per-layer residual read/write width. Rounded to `rnd` (24 keeps
        12 heads with an even head_dim, which rope requires)."""
        if not self.hg_frac:
            return [self.n_embd] * self.n_layer
        ws = []
        for layer in range(1, self.n_layer + 1):
            if layer <= self.hg_bneck:
                t = (layer - 1) / (self.hg_bneck - 1)
            else:
                t = (self.n_layer - layer) / (self.n_layer - self.hg_bneck)
            w = self.n_embd * self.hg_frac ** t
            ws.append(max(rnd, rnd * round(w / rnd)))
        return (ws[:self.hg_bneck]
                + [ws[self.hg_bneck - 1]] * self.hg_mid
                + ws[self.hg_bneck:])

    def layer_attns(self):
        """Per-layer ATTENTIONS keys, resolving attn_pattern."""
        if not self.attn_pattern:
            return [self.attn] * self.n_layer_total()
        assert len(self.attn_pattern) == self.n_layer_total(), (
            f"attn_pattern {self.attn_pattern!r} has "
            f"{len(self.attn_pattern)} chars for "
            f"{self.n_layer_total()} layers")
        key = {"F": self.attn, "S": "swa", "G": "gated"}
        return [key[c] for c in self.attn_pattern]


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout
        self.use_rope = cfg.pos == "rope"
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.c_proj.RESIDUAL_SCALE_INIT = True
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.use_rope:
            from core.rope import apply_rope
            q, k = apply_rope(q, k)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class GeluMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.c_proj.RESIDUAL_SCALE_INIT = True
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x),
                                               approximate="tanh")))


class Block(nn.Module):
    """Pre-LN transformer block: x + attn(ln(x)), then x + mlp(ln(x))."""

    def __init__(self, cfg, attn_key=None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = ATTENTIONS[attn_key or cfg.attn](cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLPS[cfg.mlp](cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class SliceBlock(nn.Module):
    """Hourglass seam: a standard block built at `width`, reading and
    writing only the first `width` dims of the full-width residual
    stream. Trailing dims pass through untouched (slice-and-carry — the
    skip connection is the transition, no learned projections)."""

    def __init__(self, cfg, attn_key, width):
        super().__init__()
        assert width % cfg.n_head == 0, (width, cfg.n_head)
        self.width = width
        self.block = BLOCKS[cfg.block](replace(cfg, n_embd=width),
                                       attn_key=attn_key)

    @property
    def attn(self):
        return self.block.attn

    def forward(self, x):
        return torch.cat((self.block(x[..., :self.width]),
                          x[..., self.width:]), dim=-1)


from core.gated_swa import SlidingWindowAttention, GatedSWAttention  # noqa: E402

# The seams. The new architecture adds entries here; the config selects them.
ATTENTIONS = {"causal": CausalSelfAttention,
              "swa": SlidingWindowAttention,
              "gated": GatedSWAttention}
MLPS = {"gelu": GeluMLP}
BLOCKS = {"gpt2": Block}


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.pos in ("learned", "rope"), cfg.pos
        self.cfg = cfg
        modules = dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            drop=nn.Dropout(cfg.dropout),
            h=nn.ModuleList(
                BLOCKS[cfg.block](cfg, attn_key=key) if w == cfg.n_embd
                else SliceBlock(cfg, key, w)
                for key, w in zip(cfg.layer_attns(), cfg.layer_widths())),
            ln_f=nn.LayerNorm(cfg.n_embd, bias=cfg.bias),
        )
        if cfg.pos == "learned":
            modules["wpe"] = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.transformer = nn.ModuleDict(modules)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight   # tied

        self.apply(self._init_weights)
        # GPT-2 paper: scale residual-projection init by 1/sqrt(N residual
        # additions) so the stream's variance is depth-independent.
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0,
                                std=0.02 / math.sqrt(2 * cfg.n_layer_total()))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size
        x = self.transformer.wte(idx)
        if self.cfg.pos == "learned":
            pos = torch.arange(T, device=idx.device)
            x = x + self.transformer.wpe(pos)
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is None:
            return self.lm_head(x[:, [-1], :]), None
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               targets.reshape(-1))
        # Aux term only while TRAINING: val loss is the cross-arm judge and
        # must stay pure cross-entropy for every architecture.
        if self.cfg.lb_coef > 0 and self.training:
            lb = [b.attn.lb_loss for b in self.transformer.h
                  if getattr(b.attn, "lb_loss", None) is not None]
            if lb:
                loss = loss + self.cfg.lb_coef * torch.stack(lb).mean()
        return logits, loss

    # ------------------------------------------------------------- accounting
    def num_params(self, non_embedding=True):
        """Parameter count. `non_embedding` drops wpe (wte stays: it is tied
        to lm_head, which does real compute)."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and self.cfg.pos == "learned":
            n -= self.transformer.wpe.weight.numel()
        return n

    def flops_per_token(self, T=None):
        """Analytic fwd+bwd FLOPs per trained token (nanoGPT convention):
        6*N for the matmuls plus 12*d*keys for attention scores per layer.
        Full layers average (T+1)/2 causal keys... no — the 12*L*d*T
        convention already folds the causal 1/2 into its constant, so we
        keep T for full layers and swap in the windowed layers' actual
        average visible-key count, avg_t min(t, W): windowed and gated
        layers touch at most W keys per query (the gated union is exactly
        the same <=W budget). This number times tokens-consumed is the
        compute-matching currency between arms; router matmuls (d*G) are
        ~0.005% and ignored."""
        T = T or self.cfg.block_size
        W = self.cfg.window
        # mean over t in [1..T] of min(t, W), the visible keys of a
        # windowed layer at training shape T
        win_keys = T if T <= W else (W * (W + 1) / 2 + (T - W) * W) / T
        # hourglass layers score at their own (narrower) width
        score = sum(12 * w * (T if k == self.cfg.attn else win_keys)
                    for k, w in zip(self.cfg.layer_attns(),
                                    self.cfg.layer_widths()))
        return 6 * self.num_params() + score

    def configure_optimizers(self, weight_decay, lr, betas, device_type):
        """AdamW with weight decay on >=2D params only (no decay on biases,
        layernorms, embeddings-as-1D), fused on CUDA."""
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        fused = device_type == "cuda"
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.block_size:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        return idx


def config_dict(cfg: GPTConfig):
    return asdict(cfg)
