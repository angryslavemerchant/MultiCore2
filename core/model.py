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

from core.diffattn import DiffMixin, rms


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
    # Per-layer window schedule (the pyramid): comma-separated, one entry
    # per layer, ints for S/G layers; F layers' entries are ignored (write
    # "F" or "0"). Empty = every windowed layer uses cfg.window. E.g. the
    # pyramid-SWA hourglass "32,64,128,256,512,F,F,F,F,2048,1024,512".
    windows: str = ""
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
    # width rounding for layer_widths(); 24 keeps 12 heads at even head_dim
    # (the phase-3 arms), 96 keeps head_dim a multiple of 8 — required for
    # the franken run where every hourglass width goes through flex
    hg_round: int = 24
    # COW archive (attn="cow" / pattern letter C): recent_band raw keys +
    # n_gates*cow_chains live chain versions == window total budget.
    cow_chains: int = 32       # K live chains per gate
    cow_theta: float = 0.7     # vigilance: best cosine >= theta -> merge
    cow_chunk: int = 128       # chain-scan chunk (heads frozen within)
    # Chunk-latent attention (attn="chunk" / pattern K; control arm
    # "blocksum" / pattern B): every chunk_btok tokens each layer mints
    # chunk_k latent KVs over its own cache -- the only long-range path
    # (raw attention stays windowed at cfg.window). core/chunk.py.
    chunk_btok: int = 256      # write cadence (boundary every B_tok)
    chunk_k: int = 16          # chunks minted per boundary
    # 0 = soft membership over the whole prefix (v0). >0 = hard top-k:
    # each query keeps its chunk_topk best prefix tokens, softmax
    # renormalized over the survivors (v0.1, NSA-selection-style).
    chunk_topk: int = 0
    # Chunk v0.2 (attn="chunkv2" / pattern N, core/chunkv2.py): recursive
    # minting (writer candidates include older chunks, latent-only refs),
    # soft dedup (per-head lam, init 0), and raw member fetch -- each
    # query attends the raw-token pointers of its top chunk_fetch_n
    # chunks as a third softmax branch. 0 disables the fetch branch.
    chunk_fetch_n: int = 4
    # Top-k side-stack (pattern letter T, core/sidestack.py): full causal
    # attention whose MLP slot is replaced by a small transformer over the
    # per-head top-k retrieved v slices. This is the per-head k.
    side_topk: int = 16
    # NSA-with-registers (attn="nsa", core/nsa.py, 2026-08-09): every
    # layer = 128-swa + block-summary cmp branch + top-k selected fine
    # branch, with nsa_nreg learned register vectors (KV-only, grouped
    # into nsa_block-sized blocks) always visible to cmp and selectable
    # by slc. cfg.window is the swa branch's window.
    nsa_block: int = 32        # summary/selection granularity
    nsa_topk: int = 12         # blocks fetched by the slc branch
    nsa_nreg: int = 1024       # learned register vectors per layer
    # Uberloop (2026-08-09): per-layer weight-tied loop counts, comma-
    # separated, one entry per layer ("1,1,1,2,2,4,4,4,4,2,1,1"). A layer
    # with count L runs its block L times per forward (shared weights);
    # symmetry across iterations is broken by per-iteration channel gains
    # on the block input (equivalent to iteration-specific norm gains
    # under rms) and channel scales on the residual delta, both init 1 —
    # the consensus-cheap fix (MobileLLM needs none at 2x; Relaxed
    # Recursive Transformers' LoRA reserved for whole-model recursion).
    # Empty = every layer once. Loops buy re-query cadence where width is
    # cheap: iterations cost ~w^2 so 4x at the waist is ~1/3 of one
    # full-width layer.
    loops: str = ""
    # Frankenstein stack (2026-07-30). Every default is OFF so earlier
    # arms' checkpoint configs round-trip unchanged.
    norm: str = "ln"           # "ln" (GPT-2) | "rms" (weight only, no bias)
    qk_norm: bool = False      # parameter-free RMS on q,k per head, pre-rope
    diff_attn: bool = False    # differential attention (core/diffattn.py)
    canon: bool = False        # depthwise causal conv residuals (k=4)
    # Full A/B/C/D canon (Allen-Zhu 2025): on top of A (pre-attn) and C
    # (pre-MLP) residual convs, B puts the conv on the concatenated q,k,v
    # after c_attn and D on the MLP hidden after the activation. All
    # zero-init (exact identity at init). Independent of cfg.canon so the
    # trainer states both explicitly; B is skipped by attention classes
    # that don't implement it (cow).
    canon_full: bool = False
    softcap: float = 0.0       # logits = cap*tanh(logits/cap); 0 disables
    untied: bool = False       # separate wte / lm_head (lm_head zero-init)
    zero_init: bool = False    # zero-init residual c_proj (vs 1/sqrt(2L))

    def n_layer_total(self):
        return self.n_layer + (self.hg_mid if self.hg_frac else 0)

    def layer_widths(self, rnd=None):
        """Per-layer residual read/write width, rounded to cfg.hg_round."""
        rnd = rnd or self.hg_round
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

    def layer_loops(self):
        """Per-layer loop counts from cfg.loops; empty = all 1."""
        if not self.loops:
            return [1] * self.n_layer_total()
        parts = [int(p) for p in self.loops.split(",")]
        assert len(parts) == self.n_layer_total(), (
            f"loops {self.loops!r} has {len(parts)} entries for "
            f"{self.n_layer_total()} layers")
        assert all(p >= 1 for p in parts), self.loops
        return parts

    def layer_attns(self):
        """Per-layer ATTENTIONS keys, resolving attn_pattern."""
        if not self.attn_pattern:
            return [self.attn] * self.n_layer_total()
        assert len(self.attn_pattern) == self.n_layer_total(), (
            f"attn_pattern {self.attn_pattern!r} has "
            f"{len(self.attn_pattern)} chars for "
            f"{self.n_layer_total()} layers")
        key = {"F": self.attn, "S": "swa", "G": "gated", "C": "cow",
               "K": "chunk", "B": "blocksum", "N": "chunkv2",
               "T": "topkside", "R": "swaside"}
        return [key[c] for c in self.attn_pattern]

    def layer_windows(self):
        """Per-layer window: an int for windowed (S/G/C) layers, None for
        full layers. Resolves cfg.windows; empty = uniform cfg.window."""
        attns = self.layer_attns()
        # topkside layers are full-attention layers (the branch replaces
        # the MLP, not the window) — no window entry applies to them
        full = [k == self.attn or k == "topkside" for k in attns]
        if not self.windows:
            return [None if f else self.window
                    for f, k in zip(full, attns)]
        parts = [p.strip() for p in self.windows.split(",")]
        assert len(parts) == self.n_layer_total(), (
            f"windows {self.windows!r} has {len(parts)} entries for "
            f"{self.n_layer_total()} layers")
        out = []
        for p, f in zip(parts, full):
            if f:
                out.append(None)     # full layer: entry is decorative
                continue
            w = int(p)
            assert w > 0, f"windowed layer needs a positive window, got {p}"
            out.append(w)
        return out


class CausalSelfAttention(DiffMixin, nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout
        self.use_rope = cfg.pos == "rope"
        self.qk_norm = cfg.qk_norm
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.canon_b = Canon(3 * cfg.n_embd) if cfg.canon_full else None
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.c_proj.RESIDUAL_SCALE_INIT = True
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self._init_diff(cfg)

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
        if self.diff:
            # full-causal death (k + T >= T: nothing ever dies) reuses the
            # interval machinery so both diff passes share one cached mask
            from core.gated_swa import _prepare_attend, swa_death_times
            death = swa_death_times(T, T, x.device).expand(B, T)
            attend = _prepare_attend(death, B, T, x.device,
                                     cache_key=("causal",))
            y = self._diff_attend(attend, q, k, v, dropout_p)
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=dropout_p)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class RMSNorm(nn.Module):
    """Weight-only RMS norm, fp32 compute (the speedrun/Llama norm)."""

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x.float(), (x.shape[-1],),
                          self.weight.float()).type_as(x)


def make_norm(cfg, dim):
    return (RMSNorm(dim) if cfg.norm == "rms"
            else nn.LayerNorm(dim, bias=cfg.bias))


class Canon(nn.Module):
    """Canon layer (Allen-Zhu 2025): depthwise causal conv residual,
    kernel 4 (current + 3 past tokens), zero-init so it starts as the
    identity. Cheap local token mixing ahead of attention and the MLP."""

    def __init__(self, dim, kernel=4):
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv1d(dim, dim, kernel, groups=dim, bias=False)
        nn.init.zeros_(self.conv.weight)

    def forward(self, x):
        y = F.pad(x.transpose(1, 2), (self.kernel - 1, 0))
        return self.conv(y).transpose(1, 2)


class GeluMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.canon_d = Canon(4 * cfg.n_embd) if cfg.canon_full else None
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.c_proj.RESIDUAL_SCALE_INIT = True
        self.dropout = nn.Dropout(cfg.dropout)

    def _act(self, h):
        return F.gelu(h, approximate="tanh")

    def forward(self, x):
        h = self._act(self.c_fc(x))
        if self.canon_d is not None:
            h = h + self.canon_d(h)
        return self.dropout(self.c_proj(h))


class Relu2MLP(GeluMLP):
    """Same shapes, ReLU^2 activation (the speedrun MLP — measured equal
    or better than GELU at this scale, and param-identical, so the
    hourglass width matcher needs no special case)."""

    def _act(self, h):
        return F.relu(h).square()


class Block(nn.Module):
    """Pre-LN transformer block: x + attn(ln(x)), then x + mlp(ln(x))."""

    def __init__(self, cfg, attn_key=None):
        super().__init__()
        self.use_canon = cfg.canon
        if cfg.canon:
            self.canon_a = Canon(cfg.n_embd)
            self.canon_c = Canon(cfg.n_embd)
        self.ln_1 = make_norm(cfg, cfg.n_embd)
        self.attn = ATTENTIONS[attn_key or cfg.attn](cfg)
        self.ln_2 = make_norm(cfg, cfg.n_embd)
        self.mlp = MLPS[cfg.mlp](cfg)

    def forward(self, x):
        if self.use_canon:
            x = x + self.canon_a(x)
        x = x + self.attn(self.ln_1(x))
        if self.use_canon:
            x = x + self.canon_c(x)
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
        self.block = make_block(replace(cfg, n_embd=width), attn_key)

    @property
    def attn(self):
        return self.block.attn

    def forward(self, x):
        return torch.cat((self.block(x[..., :self.width]),
                          x[..., self.width:]), dim=-1)


class LoopedBlock(nn.Module):
    """Uberloop: run one weight-tied block `loops` times. Per-iteration
    channel gain g_i on the block input and channel scale s_i on the
    residual delta (both init 1) break iteration symmetry without
    adapters: under rms norms an input channel gain IS an
    iteration-specific norm gain (rms divides by the whole-vector norm,
    so relative channel weights pass through), and iteration i computes
    x <- x + s_i * (block(g_i * x) - g_i * x), which at init (g=s=1)
    is exactly block(x). For SliceBlocks the delta is zero on carry
    dims, so gains there are inert and the carry stream is never
    rescaled. Gains live in a ParameterList (1D each) so Muon's
    2D-under-h rule leaves them to AdamW."""

    def __init__(self, block, loops, dim):
        super().__init__()
        assert loops >= 2, loops
        self.block = block
        self.loops = loops
        self.gains = nn.ParameterList(
            nn.Parameter(torch.ones(dim)) for _ in range(loops))
        self.scales = nn.ParameterList(
            nn.Parameter(torch.ones(dim)) for _ in range(loops))

    @property
    def attn(self):
        return self.block.attn

    def forward(self, x):
        for g, s in zip(self.gains, self.scales):
            u = x * g
            x = x + s * (self.block(u) - u)
        return x


from core.gated_swa import SlidingWindowAttention, GatedSWAttention  # noqa: E402
from core.cow import COWAttention  # noqa: E402
from core.chunk import ChunkAttention, BlockSumAttention  # noqa: E402
from core.chunkv2 import ChunkV2Attention  # noqa: E402
from core.nsa import NSARegisterAttention  # noqa: E402

# The seams. The new architecture adds entries here; the config selects them.
ATTENTIONS = {"causal": CausalSelfAttention,
              "swa": SlidingWindowAttention,
              "gated": GatedSWAttention,
              "cow": COWAttention,
              "chunk": ChunkAttention,
              "blocksum": BlockSumAttention,
              "chunkv2": ChunkV2Attention,
              "nsa": NSARegisterAttention}
MLPS = {"gelu": GeluMLP, "relu2": Relu2MLP}
BLOCKS = {"gpt2": Block}


def make_block(cfg, attn_key):
    """Block construction seam: topkside (T, full) and swaside (R,
    windowed) layers need a block whose MLP slot holds the side branch
    (it consumes the attention's internals, so attention and branch
    can't be independent registry entries)."""
    if attn_key in ("topkside", "swaside"):
        from core.sidestack import TopKSideBlock
        return TopKSideBlock(cfg, attn_key=attn_key)
    return BLOCKS[cfg.block](cfg, attn_key=attn_key)


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.pos in ("learned", "rope"), cfg.pos
        self.cfg = cfg
        def build(key, w, win, lp):
            lcfg = (cfg if win is None or win == cfg.window
                    else replace(cfg, window=win))
            b = (make_block(lcfg, key) if w == cfg.n_embd
                 else SliceBlock(lcfg, key, w))
            return LoopedBlock(b, lp, cfg.n_embd) if lp > 1 else b

        modules = dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            drop=nn.Dropout(cfg.dropout),
            h=nn.ModuleList(
                build(key, w, win, lp)
                for key, w, win, lp in zip(cfg.layer_attns(),
                                           cfg.layer_widths(),
                                           cfg.layer_windows(),
                                           cfg.layer_loops())),
            ln_f=make_norm(cfg, cfg.n_embd),
        )
        if cfg.pos == "learned":
            modules["wpe"] = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.transformer = nn.ModuleDict(modules)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if not cfg.untied:
            self.transformer.wte.weight = self.lm_head.weight   # tied

        self.apply(self._init_weights)
        if cfg.untied:
            # speedrun: untied head starts at zero — logits are exactly 0
            # (uniform prediction) at init and the head learns from scratch
            nn.init.zeros_(self.lm_head.weight)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                if cfg.zero_init:
                    # speedrun: residual branches start as strict no-ops
                    nn.init.zeros_(p)
                else:
                    # GPT-2 paper: scale residual-projection init by
                    # 1/sqrt(N residual additions) so the stream's variance
                    # is depth-independent.
                    nn.init.normal_(
                        p, mean=0.0,
                        std=0.02 / math.sqrt(2 * cfg.n_layer_total()))
        # diff attention: per-layer lambda_init depth schedule
        for i, blk in enumerate(self.transformer.h):
            if hasattr(blk.attn, "set_depth"):
                blk.attn.set_depth(i)

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

        def cap(lg):
            return (self.cfg.softcap * torch.tanh(lg / self.cfg.softcap)
                    if self.cfg.softcap else lg)

        if targets is None:
            return cap(self.lm_head(x[:, [-1], :])), None
        logits = cap(self.lm_head(x))
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
        """Parameter count. `non_embedding` drops wpe, and — when untied —
        wte too: both are pure lookups. (Tied wte stays: it IS lm_head,
        which does real compute.) This is the N that 6*N FLOPs sees."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and self.cfg.pos == "learned":
            n -= self.transformer.wpe.weight.numel()
        if non_embedding and self.cfg.untied:
            n -= self.transformer.wte.weight.numel()
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

        def avg_keys(w):
            # mean over t in [1..T] of min(t, w): visible keys at width w
            return T if T <= w else (w * (w + 1) / 2 + (T - w) * w) / T

        # cow layers: raw band + at most n_gates*cow_chains live versions.
        # Upper bound (band + G*K == window by the budget assert), counted
        # as two populations because both saturate independently.
        cow_keys = (avg_keys(self.cfg.recent_band)
                    + avg_keys(self.cfg.n_gates * self.cfg.cow_chains))
        # chunk layers: local band + log reads + (free arm) writer
        # cross-attention, all counted as extra visible keys per token
        from core.chunk import chunk_extra_keys
        from core.chunkv2 import chunkv2_extra_keys
        chunk_keys = {
            key: chunk_extra_keys(T, self.cfg.chunk_btok,
                                  self.cfg.chunk_k, free)
            for key, free in (("chunk", True), ("blocksum", False))}
        chunk_keys["chunkv2"] = chunkv2_extra_keys(
            T, self.cfg.chunk_btok, self.cfg.chunk_k,
            self.cfg.chunk_topk, self.cfg.chunk_fetch_n)
        # nsa layers: swa window + cmp summaries + dense-with-mask slc
        # (charged at its executed T + n_reg, not the surviving top-k)
        from core.nsa import nsa_extra_keys
        nsa_keys = nsa_extra_keys(T, self.cfg.nsa_block,
                                  self.cfg.nsa_topk, self.cfg.nsa_nreg,
                                  self.cfg.window)
        # hourglass layers score at their own (narrower) width; windowed
        # layers at their own (per-layer, cfg.windows) window; looped
        # layers score once per iteration
        score = sum(lp * 12 * w * (nsa_keys if k == "nsa"
                                   else T if win is None
                                   else cow_keys if k == "cow"
                                   else avg_keys(win) + chunk_keys.get(k, 0.0))
                    for k, w, win, lp in zip(self.cfg.layer_attns(),
                                             self.cfg.layer_widths(),
                                             self.cfg.layer_windows(),
                                             self.cfg.layer_loops()))
        # uberloop: each extra iteration re-spends the layer's matmul
        # FLOPs on weight-tied params that 6*N bills only once
        score += sum(6 * (lp - 1)
                     * sum(p.numel() for p in blk.block.parameters())
                     for lp, blk in zip(self.cfg.layer_loops(),
                                        self.transformer.h) if lp > 1)
        if self.cfg.diff_attn:
            # two half-width score passes (parity) but both attention maps
            # hit the full value width: value application doubles -> 1.5x
            # on the attention term
            score *= 1.5
        # side-stack branch (T/R layers): 6*N covers each param used once
        # per token; the slice MLP / set-attention multiplicity, set
        # scores, and the no-grad selection sweep are charged on top
        # (core/sidestack.py). R layers' banded host attention is already
        # counted at avg_keys(win) above.
        attns = self.cfg.layer_attns()
        if "topkside" in attns or "swaside" in attns:
            from core.sidestack import side_extra_flops
            score += sum(side_extra_flops(self.cfg, w, T, window=win)
                         for k, w, win in zip(attns,
                                              self.cfg.layer_widths(),
                                              self.cfg.layer_windows())
                         if k in ("topkside", "swaside"))
        return 6 * self.num_params() + score

    def configure_optimizers(self, weight_decay, lr, betas, device_type,
                             opt="adamw", muon_lr=0.02):
        """opt="adamw": AdamW with weight decay on >=2D params only (no
        decay on biases, layernorms, embeddings-as-1D), fused on CUDA.

        opt="muon": Muon over the 2D hidden matrices inside the blocks,
        AdamW over everything else (embeddings, lm_head, norms, canon
        convs, diff lambdas). Each param group carries lr_scale so the
        trainer's one schedule drives both optimizers:
        group lr = schedule(step) * lr_scale, with Muon's base rate at
        muon_lr while the schedule is expressed in AdamW units."""
        params = [(n, p) for n, p in self.named_parameters()
                  if p.requires_grad]
        fused = device_type == "cuda"
        if opt == "muon":
            from core.muon import Muon, ComboOptimizer
            hidden = [p for n, p in params
                      if p.dim() == 2 and n.startswith("transformer.h.")
                      and not n.endswith(".registers")]  # embeddings-like
            hid_ids = {id(p) for p in hidden}
            rest = [p for _, p in params if id(p) not in hid_ids]
            decay = [p for p in rest if p.dim() >= 2]
            no_decay = [p for p in rest if p.dim() < 2]
            muon = Muon(hidden, lr=muon_lr, momentum=0.95)
            adamw = torch.optim.AdamW(
                [{"params": decay, "weight_decay": weight_decay},
                 {"params": no_decay, "weight_decay": 0.0}],
                lr=lr, betas=betas, fused=fused)
            for g in muon.param_groups:
                g["lr_scale"] = muon_lr / lr
            for g in adamw.param_groups:
                g["lr_scale"] = 1.0
            return ComboOptimizer([muon, adamw])
        decay = [p for _, p in params if p.dim() >= 2]
        no_decay = [p for _, p in params if p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
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


class PadForward(nn.Module):
    """Bench-time shim for arms with sequence-length constraints: right-
    pads idx/targets to `full` (fixed length) or the next multiple of
    `mult`, runs the inner model, and slices logits back to the caller's
    length. Causal-safe: trailing pads cannot influence earlier
    positions in any arm. The returned loss includes pad positions —
    bench scripts consume logits, not loss."""

    def __init__(self, model, mult=1, full=None):
        super().__init__()
        self.model, self.mult, self.full = model, mult, full

    def load_state_dict(self, sd, strict=True):
        return self.model.load_state_dict(sd, strict=strict)

    def forward(self, idx, targets=None):
        n = idx.shape[1]
        tgt = self.full or -(-n // self.mult) * self.mult
        if tgt == n:
            return self.model(idx, targets)
        pad = idx.new_zeros(idx.shape[0], tgt - n)
        padded = torch.cat((idx, pad), dim=1)
        if targets is None:
            # inner would return only the LAST (pad!) position; force
            # full logits with dummy targets, hand back the last REAL one
            logits, _ = self.model(padded, padded)
            return logits[:, n - 1:n], None
        logits, loss = self.model(padded,
                                  torch.cat((targets, pad), dim=1))
        return logits[:, :n], loss


def model_from_ckpt_config(config: dict):
    """(cfg, model) for a checkpoint's config dict, dispatching on
    family: hier checkpoints (HierConfig fields) vs everything else.
    Bench/probe scripts share this so new families patch one place.
    Length-constrained arms come back wrapped in PadForward (hier needs
    T == block_size exactly; nsa needs T % nsa_block == 0)."""
    config = dict(config)
    arch = config.pop("arch", None)       # hier_config_dict stamps this
    if arch == "hier" or "btok" in config:
        from core.hier import HierGPT, HierConfig
        cfg = HierConfig(**config)
        return cfg, PadForward(HierGPT(cfg), full=cfg.block_size)
    cfg = GPTConfig(**config)
    model = GPT(cfg)
    if cfg.attn == "nsa":
        model = PadForward(model, mult=cfg.nsa_block)
    return cfg, model
