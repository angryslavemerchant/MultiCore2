"""Hierarchical GPT-2 with predictive plans and block-level sparse memory.

Spec: hierarchical_gpt2_sparse_memory_spec.md, as amended for run one
(2026-08-09 workshop): NeoX 50304 vocab, project recipe (relu2 / rms /
qk-norm / rope / softcap / untied+zero-init / Muon), STRIDE 128 = no
window overlap (32 blocks/seq, 4 superblocks of 8), PKM at 16,384 slots
(128x128 product keys).

Dataflow per 4096-token sequence (all stages parallel in training):

  tokens -> [32 windows of 128] -> TokenAnalysis (4L, d768)
         -> observed block summaries S[32] (d512)
         -> observed superblock summaries U[4] (d384)
         -> SuperPredictor (2L, shifted causal) -> plans G[4]
         -> BlockPredictor (4L, shifted causal, PKM after layer 3) -> P[32]
         -> TokenPrediction (4L, d768, per-layer gated conditioning on
            C_b = Wc[P_b ; G_g]) -> logits

Causality: window b's logits depend on raw tokens of block b (causal
within the window), S_{<b} via P_b, U_{<g} via G_g -- never completed
S_b / U_g. Verified by tests/test_hier.py perturbation tests.

Losses: primary shifted CE over the tiled windows (every target exactly
once, identical mapping to a flat GPT loss) + block-plan aux (P_b
predicts 8 fixed offsets of block b) + super-plan aux (G_g predicts the
mid-block token of each of its 8 blocks) + product-key load balance.
Aux terms train-time only; val loss is pure CE (the cross-arm judge).

Design notes tied to project history:
- Plans have their own gradients (aux heads): the chunkv2 autopsy showed
  summaries with no direct training pressure never become addressable.
- Nothing sequential: shifted teacher forcing everywhere.
- Memory gate is a zero-init scalar: exact dense behavior at init; the
  gate is a leaf so it receives gradient immediately (lm_head-zero-init
  dynamics, not a dead path).
- Canon A/B/C/D on the token stacks only (temporal convs are
  well-defined within a window; block/super stacks are 32/4-position
  sequences where k=4 canon is not obviously meaningful).
- Muon takes 2D weights inside the four transformer stacks; the value
  table, key codebooks, embeddings and heads stay in AdamW.
"""
import math
from dataclasses import dataclass, asdict, replace

import torch
import torch.nn as nn
from torch.nn import functional as F

from core.model import GPTConfig, make_block, make_norm


@dataclass
class HierConfig:
    block_size: int = 4096
    vocab_size: int = 50304
    btok: int = 128            # window == stride (no overlap, run one)
    blocks_per_super: int = 8
    # levels=3 (run three) switches on the v3 feature set as a bundle:
    # a third hierarchy level (supers_per_hyper supers -> hyper units),
    # learned attention pooling for all summaries, per-token per-channel
    # dynamic conditioning gates, and per-level latent predictive losses
    # (1 - cos(pred, sg(target)), JEPA-style, targets stop-gradded;
    # token-aux heads stay as the collapse anchor). levels=2 preserves
    # runs one/two exactly (mean+last pooling, static scalar gates).
    levels: int = 2
    supers_per_hyper: int = 8
    d_hyp: int = 256
    n_head_hyp: int = 4
    n_hyperpred: int = 2
    latent_coef: float = 0.05
    d_tok: int = 768
    d_blk: int = 512
    d_sup: int = 384
    n_head_tok: int = 12
    n_head_blk: int = 8
    n_head_sup: int = 6
    n_analysis: int = 4
    n_predict: int = 4
    n_blockpred: int = 4
    n_superpred: int = 2
    # product-key sparse memory (16,384 = 128 x 128)
    mem_slots: int = 16384
    mem_val: int = 128
    mem_heads: int = 4
    mem_topk: int = 4
    mem_cand: int = 16         # kept per sub-key codebook before pairing
    # objective coefficients
    aux_block_coef: float = 0.10
    aux_super_coef: float = 0.05
    route_coef: float = 0.01
    # token-stack attention mode:
    #   "block" = independent 128-token windows (run one; block-diagonal,
    #             cold boundaries -- the no-overlap ablation)
    #   "swa"   = true sliding window w128 over the full sequence (every
    #             token sees its trailing 127; stacked layers compound
    #             reach; NSA-style: window + plan + sparse branches).
    #             Summaries still mint every btok tokens.
    token_mode: str = "block"
    # recipe
    softcap: float = 15.0
    dropout: float = 0.0

    def n_blocks(self):
        assert self.block_size % self.btok == 0
        return self.block_size // self.btok

    def n_supers(self):
        nb = self.n_blocks()
        assert nb % self.blocks_per_super == 0
        return nb // self.blocks_per_super

    def n_hypers(self):
        ns = self.n_supers()
        assert ns % self.supers_per_hyper == 0
        return ns // self.supers_per_hyper

    def sub_keys(self):
        n = int(math.isqrt(self.mem_slots))
        assert n * n == self.mem_slots, "mem_slots must be a square"
        return n


def _stack_cfg(cfg, d, n_layer, n_head, T, canon, window=0):
    """A GPTConfig for one internal transformer stack (reuses the
    project's Block / CausalSelfAttention / SlidingWindowAttention
    with rope + qk-norm). window > 0 selects the SWA attention."""
    return GPTConfig(block_size=T, vocab_size=1, n_layer=n_layer,
                     n_head=n_head, n_embd=d, dropout=cfg.dropout,
                     bias=False, pos="rope", norm="rms", mlp="relu2",
                     qk_norm=True, canon=canon, canon_full=canon,
                     untied=True, zero_init=True,
                     window=window or 512)


class AttnPool(nn.Module):
    """Learned attention pooling (v3 summaries): one learned query
    scores the window's states; output = W[pooled ; last] normalized.
    Strictly richer than the v2 mean+last statistic."""

    def __init__(self, cfg, d_in, d_out):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d_in) * 0.02)
        self.w = nn.Linear(2 * d_in, d_out, bias=False)
        self.ln = make_norm(_stack_cfg(cfg, d_out, 1, 1, 1, False), d_out)

    def forward(self, h):
        # h: (..., W, d_in), window on dim -2
        a = torch.softmax(
            (h @ self.q) / h.shape[-1] ** 0.5, dim=-1)
        pooled = (a.unsqueeze(-1) * h).sum(-2)
        return self.ln(self.w(torch.cat((pooled, h[..., -1, :]), -1)))


class ProductKeyMemory(nn.Module):
    """PKM (Lample 2019): factorized addressing over sub_keys^2 slots.
    Queries come from BLOCK states (the spec's central bet). Shared
    value table, per-head query projections and codebooks."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        n = cfg.sub_keys()
        half = cfg.mem_val // 2
        self.values = nn.Embedding(cfg.mem_slots, cfg.mem_val)
        nn.init.normal_(self.values.weight, std=0.02)
        self.wq = nn.Linear(cfg.d_blk, cfg.mem_heads * cfg.mem_val,
                            bias=False)
        self.keys_a = nn.Parameter(
            torch.randn(cfg.mem_heads, n, half) * 0.02)
        self.keys_b = nn.Parameter(
            torch.randn(cfg.mem_heads, n, half) * 0.02)
        self.out = nn.Linear(cfg.mem_heads * cfg.mem_val, cfg.d_blk,
                             bias=False)
        self.last_stats = None

    def forward(self, h):
        """h: (B, NB, d_blk) -> (B, NB, d_blk), plus load-balance loss
        and routing stats stashed on self."""
        cfg = self.cfg
        B, NB, _ = h.shape
        n = cfg.sub_keys()
        half = cfg.mem_val // 2
        H, K, C = cfg.mem_heads, cfg.mem_topk, cfg.mem_cand
        q = self.wq(h).view(B, NB, H, cfg.mem_val)
        qa, qb = q[..., :half], q[..., half:]           # (B,NB,H,half)
        sa = torch.einsum("bnhd,hkd->bnhk", qa, self.keys_a)
        sb = torch.einsum("bnhd,hkd->bnhk", qb, self.keys_b)

        va, ia = sa.topk(C, dim=-1)                     # (B,NB,H,C)
        vb, ib = sb.topk(C, dim=-1)
        pair = va.unsqueeze(-1) + vb.unsqueeze(-2)      # (B,NB,H,C,C)
        top, flat = pair.view(B, NB, H, C * C).topk(K, dim=-1)
        ra, rb = flat // C, flat % C                    # (B,NB,H,K)
        addr = (ia.gather(-1, ra) * n + ib.gather(-1, rb))
        w = F.softmax(top, dim=-1)                      # (B,NB,H,K)
        vals = self.values(addr)                        # (B,NB,H,K,val)
        head_out = (w.unsqueeze(-1) * vals).sum(-2)     # (B,NB,H,val)
        out = self.out(head_out.reshape(B, NB, H * cfg.mem_val))

        # switch-style load balance over each codebook's mean softmax
        lb = 0.0
        for s in (sa, sb):
            p = F.softmax(s.float(), dim=-1).mean(dim=(0, 1))  # (H,n)
            lb = lb + (n * (p * p).sum(-1)).mean()
        self.lb_loss = lb / 2

        with torch.no_grad():
            uniq = addr.view(B, NB, -1)
            self.last_stats = {
                "mem/slots_used_frac": float(
                    torch.unique(addr).numel() / cfg.mem_slots),
                "mem/dup_frac": float(1 - torch.unique(
                    uniq[0, 0]).numel() / uniq.shape[-1]) if B else 0.0,
                "mem/route_entropy": float(
                    -(w * (w + 1e-9).log()).sum(-1).mean()),
            }
        return out


class HierGPT(nn.Module):
    def __init__(self, cfg: HierConfig):
        super().__init__()
        self.cfg = cfg
        NB, NS = cfg.n_blocks(), cfg.n_supers()
        tok_T = cfg.btok

        def stack(d, layers, heads, T, canon, key="causal", window=0):
            scfg = _stack_cfg(cfg, d, layers, heads, T, canon, window)
            return nn.ModuleList(make_block(scfg, key)
                                 for _ in range(layers))

        swa = cfg.token_mode == "swa"
        tok_key = "swa" if swa else "causal"
        tok_len = cfg.block_size if swa else tok_T
        tok_win = cfg.btok if swa else 0
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.d_tok),
            analysis=nn.ModuleList(
                stack(cfg.d_tok, cfg.n_analysis, cfg.n_head_tok,
                      tok_len, True, tok_key, tok_win)),
            predict=nn.ModuleList(
                stack(cfg.d_tok, cfg.n_predict, cfg.n_head_tok,
                      tok_len, True, tok_key, tok_win)),
            blockpred=nn.ModuleList(
                stack(cfg.d_blk, cfg.n_blockpred, cfg.n_head_blk,
                      NB, False)),
            superpred=nn.ModuleList(
                stack(cfg.d_sup, cfg.n_superpred, cfg.n_head_sup,
                      NS, False)),
            ln_f=make_norm(_stack_cfg(cfg, cfg.d_tok, 1, 1, 1, False),
                           cfg.d_tok),
        ))
        if cfg.levels == 3:
            self.transformer["hyperpred"] = nn.ModuleList(
                stack(cfg.d_hyp, cfg.n_hyperpred, cfg.n_head_hyp,
                      cfg.n_hypers(), False))
            # v3: learned attention pooling at every level
            self.pool_blk = AttnPool(cfg, cfg.d_tok, cfg.d_blk)
            self.pool_sup = AttnPool(cfg, cfg.d_blk, cfg.d_sup)
            self.pool_hyp = AttnPool(cfg, cfg.d_sup, cfg.d_hyp)
            self.hyper_bos = nn.Parameter(torch.zeros(cfg.d_hyp))
            self.w_cond_hy = nn.Linear(cfg.d_hyp, cfg.d_sup, bias=False)
        else:
            # v2: [mean ; last] linear summaries
            self.w_summary = nn.Linear(2 * cfg.d_tok, cfg.d_blk,
                                       bias=False)
            self.ln_summary = make_norm(
                _stack_cfg(cfg, cfg.d_blk, 1, 1, 1, False), cfg.d_blk)
            self.w_super = nn.Linear(2 * cfg.d_blk, cfg.d_sup,
                                     bias=False)
            self.ln_super = make_norm(
                _stack_cfg(cfg, cfg.d_sup, 1, 1, 1, False), cfg.d_sup)
        # shifted-sequence inputs
        self.block_bos = nn.Parameter(torch.zeros(cfg.d_blk))
        self.super_bos = nn.Parameter(torch.zeros(cfg.d_sup))
        self.w_block_in = nn.Linear(cfg.d_blk, cfg.d_blk, bias=False)
        self.w_cond_g = nn.Linear(cfg.d_sup, cfg.d_blk, bias=False)
        # memory (after block-predictor layer 3)
        self.memory = ProductKeyMemory(cfg)
        self.mem_gate = nn.Parameter(torch.zeros(()))
        self.ln_mem = make_norm(
            _stack_cfg(cfg, cfg.d_blk, 1, 1, 1, False), cfg.d_blk)
        # top-down conditioning into the prediction stack
        c_in = cfg.d_blk + cfg.d_sup + (cfg.d_hyp if cfg.levels == 3
                                        else 0)
        self.w_c = nn.Linear(c_in, cfg.d_tok, bias=False)
        self.cond_proj = nn.ModuleList(
            nn.Linear(cfg.d_tok, cfg.d_tok, bias=False)
            for _ in range(cfg.n_predict))
        if cfg.levels == 3:
            # per-token per-channel dynamic gates; bias -2 so the
            # conditioning path starts small-but-nonzero like v2
            self.cond_dyn = nn.ModuleList(
                nn.Linear(cfg.d_tok, cfg.d_tok)
                for _ in range(cfg.n_predict))
            with torch.no_grad():
                for m in self.cond_dyn:
                    m.bias.fill_(-2.0)
        else:
            self.cond_gate = nn.Parameter(
                torch.full((cfg.n_predict,), -2.0))
        # aux heads (share the lm_head vocab matrix)
        self.aux_offsets = tuple(
            cfg.btok // 8 * (k + 1) - 1 for k in range(8))  # 15..127
        self.aux_off_emb = nn.Embedding(8, cfg.d_blk)
        self.w_aux_blk = nn.Linear(cfg.d_blk, cfg.d_tok, bias=False)
        self.aux_blk_emb = nn.Embedding(cfg.blocks_per_super, cfg.d_sup)
        self.w_aux_sup = nn.Linear(cfg.d_sup, cfg.d_tok, bias=False)
        self.super_aux_offset = cfg.btok // 2 - 1
        self.lm_head = nn.Linear(cfg.d_tok, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        nn.init.zeros_(self.lm_head.weight)      # untied speedrun head
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.zeros_(p)                # zero-init residuals
        if cfg.levels == 3:
            with torch.no_grad():                # _init_weights zeroed it
                for m in self.cond_dyn:
                    m.bias.fill_(-2.0)
        # eval-time switches (capability attribution, not training knobs)
        self.disable_memory = False
        self.disable_cond = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    # ----------------------------------------------------------- stages
    def _analyze(self, idx):
        cfg = self.cfg
        B, T = idx.shape
        NB = T // cfg.btok
        x = self.transformer.wte(idx)                   # (B,T,768)
        if cfg.token_mode == "swa":
            h = x                                       # full sequence
        else:
            h = x.view(B * NB, cfg.btok, cfg.d_tok)
        for blk in self.transformer.analysis:
            h = blk(h)
        return h            # (B,T,768) swa | (B*NB,128,768) block

    def _summaries(self, h, B):
        cfg = self.cfg
        hb = (h.view(B, -1, cfg.btok, cfg.d_tok)
              if cfg.token_mode == "swa"
              else h.view(B, h.shape[0] // B, cfg.btok, cfg.d_tok))
        if cfg.levels == 3:
            S = self.pool_blk(hb)                       # (B,NB,d_blk)
            g = S.view(B, cfg.n_supers(), cfg.blocks_per_super,
                       cfg.d_blk)
            U = self.pool_sup(g)                        # (B,NS,d_sup)
            v = U.view(B, cfg.n_hypers(), cfg.supers_per_hyper,
                       cfg.d_sup)
            V = self.pool_hyp(v)                        # (B,NH,d_hyp)
            with torch.no_grad():                       # collapse tripwire
                self.latent_stats = {}
                for tag, X in (("S", S), ("U", U), ("V", V)):
                    n = torch.nn.functional.normalize(
                        X[0].float(), dim=-1)
                    sim = n @ n.T
                    off = sim.numel() - sim.shape[0]
                    self.latent_stats[f"collapse/{tag}_paircos"] = float(
                        (sim.sum() - sim.trace()) / max(off, 1))
            return S, U, V
        if cfg.token_mode == "swa":
            core = torch.cat((hb.mean(dim=2), hb[:, :, -1]), dim=-1)
            S = self.ln_summary(self.w_summary(core))   # (B,NB,512)
        else:
            NB = hb.shape[1]
            core = torch.cat((hb.mean(dim=2), hb[:, :, -1]), dim=-1)
            S = self.ln_summary(self.w_summary(core))
        g = S.view(B, cfg.n_supers(), cfg.blocks_per_super, cfg.d_blk)
        U = self.ln_super(self.w_super(
            torch.cat((g.mean(dim=2), g[:, :, -1]), dim=-1)))
        return S, U                                     # (B,NB,512),(B,NS,384)

    def _plans(self, S, U, V=None):
        cfg = self.cfg
        B, NB, _ = S.shape
        NS = cfg.n_supers()
        Hy = None
        # hyper plans from shifted V (v3 only)
        if cfg.levels == 3:
            hy_in = torch.cat(
                (self.hyper_bos.expand(B, 1, -1), V[:, :-1]), dim=1)
            hh = hy_in
            for blk in self.transformer.hyperpred:
                hh = blk(hh)
            Hy = hh                                     # (B,NH,d_hyp)
        # super plans from shifted U (conditioned by hyper plans in v3)
        sup_in = torch.cat(
            (self.super_bos.expand(B, 1, -1), U[:, :-1]), dim=1)
        if Hy is not None:
            sup_in = sup_in + self.w_cond_hy(Hy).repeat_interleave(
                cfg.supers_per_hyper, dim=1)
        gh = sup_in
        for blk in self.transformer.superpred:
            gh = blk(gh)
        G = gh                                          # (B,NS,384)
        # block plans from shifted S, conditioned by this group's G
        blk_in = torch.cat(
            (self.block_bos.expand(B, 1, -1),
             self.w_block_in(S[:, :-1])), dim=1)
        cond = self.w_cond_g(G).unsqueeze(2).expand(
            -1, -1, cfg.blocks_per_super, -1).reshape(B, NB, cfg.d_blk)
        ph = blk_in + cond
        self.memory.lb_loss = S.new_zeros(())
        mem_layer = min(2, cfg.n_blockpred - 1)         # after layer 3
        for i, blk in enumerate(self.transformer.blockpred):
            ph = blk(ph)
            if i == mem_layer and not self.disable_memory:
                ph = ph + self.mem_gate * self.memory(self.ln_mem(ph))
        return ph, G, Hy                                # P:(B,NB,512)

    def _predict(self, idx, P, G, Hy=None):
        cfg = self.cfg
        B, T = idx.shape
        NB = T // cfg.btok
        g_per_block = G.unsqueeze(2).expand(
            -1, -1, cfg.blocks_per_super, -1).reshape(B, NB, cfg.d_sup)
        parts = [P, g_per_block]
        if cfg.levels == 3:
            per_hyper = cfg.blocks_per_super * cfg.supers_per_hyper
            parts.append(Hy.unsqueeze(2).expand(
                -1, -1, per_hyper, -1).reshape(B, NB, cfg.d_hyp))
        C = self.w_c(torch.cat(parts, dim=-1))          # (B,NB,768)

        def inject(h, i, c_tok):
            if self.disable_cond:
                return h
            if cfg.levels == 3:
                return h + torch.sigmoid(self.cond_dyn[i](h)) * c_tok
            return h + torch.sigmoid(self.cond_gate[i]) * c_tok

        x = self.transformer.wte(idx)
        if cfg.token_mode == "swa":
            h = x                                       # (B,T,768)
            for i, blk in enumerate(self.transformer.predict):
                h = blk(h)
                ci = self.cond_proj[i](C)               # per-block GEMM
                h = inject(h, i,
                           ci.repeat_interleave(cfg.btok, dim=1))
            h = self.transformer.ln_f(h)
            return h
        Cw = C.view(B * NB, 1, cfg.d_tok)
        h = x.view(B * NB, cfg.btok, cfg.d_tok)
        for i, blk in enumerate(self.transformer.predict):
            h = blk(h)
            h = inject(h, i, self.cond_proj[i](Cw))
        h = self.transformer.ln_f(h)
        return h.view(B, T, cfg.d_tok)

    # ---------------------------------------------------------- forward
    def _cap(self, lg):
        return (self.cfg.softcap * torch.tanh(lg / self.cfg.softcap)
                if self.cfg.softcap else lg)

    def forward(self, idx, targets=None):
        cfg = self.cfg
        B, T = idx.shape
        assert T == cfg.block_size, (
            f"hier run-one supports full-length sequences only, got {T}")
        H = self._analyze(idx)
        if cfg.levels == 3:
            S, U, V = self._summaries(H, B)
            P, G, Hy = self._plans(S, U, V)
        else:
            S, U = self._summaries(H, B)
            P, G, Hy = self._plans(S, U)
        h = self._predict(idx, P, G, Hy)
        if targets is None:
            return self._cap(self.lm_head(h[:, [-1], :])), None
        logits = self._cap(self.lm_head(h))
        # windows tile the sequence, so the flat shifted CE mapping is
        # exactly the standard GPT loss (targets == idx shifted by the
        # trainer's convention: targets[t] is the token AFTER idx[t])
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               targets.reshape(-1))
        if self.training:
            loss = loss + cfg.aux_block_coef * self._aux_block(P, idx) \
                        + cfg.aux_super_coef * self._aux_super(G, idx) \
                        + cfg.route_coef * self.memory.lb_loss
            if cfg.levels == 3:
                # JEPA-style per-level latent prediction: predictor
                # chases the (stop-gradded) observed summary. Token-aux
                # heads above are the collapse anchor; the tripwire in
                # _summaries watches pairwise cos per level.
                lat = (1 - F.cosine_similarity(
                           P, S.detach(), dim=-1).mean()) \
                    + (1 - F.cosine_similarity(
                           G, U.detach(), dim=-1).mean()) \
                    + (1 - F.cosine_similarity(
                           Hy, V.detach(), dim=-1).mean())
                self.last_latent_loss = float(lat.detach())
                loss = loss + cfg.latent_coef * lat
        return logits, loss

    def _aux_block(self, P, idx):
        """P_b predicts block b's tokens at 8 fixed offsets."""
        cfg = self.cfg
        B, NB, _ = P.shape
        offs = torch.tensor(self.aux_offsets, device=idx.device)
        # targets: (B, NB, 8) token ids at 128*b + off
        pos = (torch.arange(NB, device=idx.device)[:, None] * cfg.btok
               + offs[None, :])
        tgt = idx[:, pos.view(-1)].view(B, NB, 8)
        q = P.unsqueeze(2) + self.aux_off_emb.weight[None, None]
        lg = self._cap(self.lm_head(self.w_aux_blk(q)))
        return F.cross_entropy(lg.view(-1, lg.size(-1)), tgt.reshape(-1))

    def _aux_super(self, G, idx):
        """G_g predicts the mid-block token of each of its 8 blocks."""
        cfg = self.cfg
        B, NS, _ = G.shape
        k = cfg.blocks_per_super
        pos = (torch.arange(NS, device=idx.device)[:, None] * k
               + torch.arange(k, device=idx.device)[None, :]) \
            * cfg.btok + self.super_aux_offset
        tgt = idx[:, pos.view(-1)].view(B, NS, k)
        q = G.unsqueeze(2) + self.aux_blk_emb.weight[None, None]
        lg = self._cap(self.lm_head(self.w_aux_sup(q)))
        return F.cross_entropy(lg.view(-1, lg.size(-1)), tgt.reshape(-1))

    # ------------------------------------------------------- accounting
    def num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wte.weight.numel()    # untied lookup
        return n

    def flops_per_token(self, T=None):
        """6*N for matmul params touched once per token, with the block/
        super stacks amortized by their rates, plus attention scores."""
        cfg = self.cfg
        T = T or cfg.block_size

        def stack_params(mods):
            return sum(p.numel() for m in mods for p in m.parameters())

        tok = (stack_params(self.transformer.analysis)
               + stack_params(self.transformer.predict)
               + sum(p.numel() for p in self.cond_proj.parameters()))
        blk = (stack_params(self.transformer.blockpred)
               + sum(p.numel() for p in self.memory.parameters())
               + (sum(p.numel() for p in self.pool_blk.parameters())
                  if cfg.levels == 3
                  else self.w_summary.weight.numel())
               + self.w_block_in.weight.numel()
               + self.w_cond_g.weight.numel())
        sup = stack_params(self.transformer.superpred)
        hyp = 0.0
        if cfg.levels == 3:
            sup += sum(p.numel() for p in self.pool_sup.parameters())
            hyp = (stack_params(self.transformer.hyperpred)
                   + sum(p.numel() for p in self.pool_hyp.parameters())
                   + self.w_cond_hy.weight.numel())
        else:
            sup += self.w_super.weight.numel()
        head = self.lm_head.weight.numel() + self.w_c.weight.numel()
        # aux heads run once per block/super (train-time compute, but
        # the matching currency is the training graph)
        aux_blk = 8 * (self.w_aux_blk.weight.numel()
                       + self.lm_head.weight.numel())
        aux_sup = cfg.blocks_per_super * (
            self.w_aux_sup.weight.numel() + self.lm_head.weight.numel())
        f = 6 * (tok + head)
        f += 6 * (blk + aux_blk) / cfg.btok              # once per block
        f += 6 * (sup + aux_sup) / (cfg.btok * cfg.blocks_per_super)
        if cfg.levels == 3:
            f += 6 * hyp / (cfg.btok * cfg.blocks_per_super
                            * cfg.supers_per_hyper)
            f += (12 * cfg.d_hyp * cfg.n_hypers() * cfg.n_hyperpred
                  / (cfg.btok * cfg.blocks_per_super
                     * cfg.supers_per_hyper))
        # attention scores, project convention (12*d*keys, causal 1/2
        # folded into the constant exactly as core/model.py does)
        layers_tok = cfg.n_analysis + cfg.n_predict
        if cfg.token_mode == "swa":
            w = cfg.btok
            avg = (w * (w + 1) / 2 + (T - w) * w) / T
            f += 12 * cfg.d_tok * avg * layers_tok
        else:
            f += 12 * cfg.d_tok * cfg.btok * layers_tok
        f += 12 * cfg.d_blk * cfg.n_blocks() * cfg.n_blockpred / cfg.btok
        f += (12 * cfg.d_sup * cfg.n_supers() * cfg.n_superpred
              / (cfg.btok * cfg.blocks_per_super))
        return f

    def configure_optimizers(self, weight_decay, lr, betas, device_type,
                             opt="adamw", muon_lr=0.02):
        params = [(n, p) for n, p in self.named_parameters()
                  if p.requires_grad]
        fused = device_type == "cuda"
        if opt == "muon":
            from core.muon import Muon, ComboOptimizer
            stacks = ("transformer.analysis.", "transformer.predict.",
                      "transformer.blockpred.", "transformer.superpred.",
                      "transformer.hyperpred.")
            hidden = [p for n, p in params
                      if p.dim() == 2 and n.startswith(stacks)]
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
        return torch.optim.AdamW(
            [{"params": decay, "weight_decay": weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=lr, betas=betas, fused=fused)

    def memory_stats(self):
        stats = dict(self.memory.last_stats or {})
        stats.update(getattr(self, "latent_stats", {}))
        if hasattr(self, "last_latent_loss"):
            stats["latent/loss"] = self.last_latent_loss
        return stats


def hier_config_dict(cfg: HierConfig):
    d = asdict(cfg)
    d["arch"] = "hier"
    return d
