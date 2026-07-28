"""Interval-lifetime attention: sliding-window and admission-gated variants.

THE FORMULATION. Both variants are "each key k is visible to queries q with
k <= q < death(k)" — only death() differs:

  sliding window   death(k) = k + W. Every key lives exactly W steps.
  admission gates  the layer keeps G FIFO gates of capacity W/G. A learned
                   router assigns each token to one gate as it arrives; when
                   a gate's (capacity+1)-th token arrives, its oldest
                   resident is evicted. death(k) = position of the
                   capacity-th LATER token routed to k's gate (or infinity).
                   Total resident budget is G * W/G = W keys — same KV
                   memory and same per-query FLOPs bound as the window — but
                   lifetime is learned: tokens in a busy gate churn fast,
                   tokens in a quiet gate survive for thousands of steps.

Causality: whether "q < death(k)" holds is decidable from tokens <= q alone
(it asks "have `capacity` same-gate tokens arrived in (k, q]?"), so the mask
is causal even though the death array is computed from the full sequence in
one vectorized pass (teacher forcing). Eviction happens ON admission: the
arriving token that overflows a gate does NOT see the token it evicts, which
also makes gated attention with G=1 EXACTLY a W-window — the equivalence the
tests lean on. A key always sees itself (death(k) > k), and the most recent
W/G tokens are visible regardless of routing (a token cannot be evicted
before W/G same-gate successors arrive), so the worst-case router collapse
degrades to a W/G-window, never to nothing.

Router gradient: gate choice is a hard argmax (eviction must be discrete or
the ring-buffer/KV-cache story falls apart), and the chosen gate's softmax
probability p multiplies the token's VALUE vector as p / p.detach() —
exactly 1.0 in the forward pass, but backward pushes gradient from every
future query that attends to the token into the router. No Gumbel noise, no
value-magnitude distortion.

Kernels: a dense (B,1,T,T) bool mask through F.scaled_dot_product_attention
everywhere (correct, portable, zero FLOPs savings), or flex_attention with a
BlockMask (CUDA + torch>=2.5) which actually SKIPS dead blocks — that path
is what realizes the windowed layers' compute savings on rented GPUs.
Selection is automatic; core.gated_swa.USE_FLEX = False forces it off.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

USE_FLEX = True     # module-level override for tests / debugging

try:
    from torch.nn.attention.flex_attention import (
        flex_attention, create_block_mask)
    _HAVE_FLEX = True
    _flex_compiled = None
except ImportError:                                   # torch < 2.5
    _HAVE_FLEX = False


def swa_death_times(T, window, device):
    """(T,) long: death(k) = k + window."""
    return torch.arange(T, device=device) + window


def gated_death_times(gate_ids, capacity):
    """(B, T) long gate ids -> (B, T) long death positions (T = never dies
    inside this sequence).

    Vectorized: stable-sort positions by (gate, position); inside the sorted
    array a gate's members form a run in arrival order, so the death of the
    element at sorted slot i is the position at slot i + capacity iff that
    slot still belongs to the same gate.
    """
    B, T = gate_ids.shape
    device = gate_ids.device
    pos = torch.arange(T, device=device).expand(B, T)
    order = (gate_ids * T + pos).argsort(dim=-1)          # (B,T) positions
    sorted_gate = gate_ids.gather(-1, order)
    death_sorted = torch.full((B, T), T, dtype=torch.long, device=device)
    if capacity < T:
        cand = order[:, capacity:]                        # slot i+capacity
        same = sorted_gate[:, capacity:] == sorted_gate[:, :T - capacity]
        death_sorted[:, :T - capacity] = torch.where(
            same, cand, torch.full_like(cand, T))
    death = torch.empty_like(death_sorted)
    death.scatter_(-1, order, death_sorted)
    return death


def interval_mask(death, T):
    """death (B,T) -> (B,1,T,T) bool: True where k <= q < death[b,k]."""
    q = torch.arange(T, device=death.device).view(1, T, 1)
    k = torch.arange(T, device=death.device).view(1, 1, T)
    return ((k <= q) & (q < death.view(-1, 1, T))).unsqueeze(1)


def _attend(q, k, v, death, dropout_p):
    """Dispatch to flex_attention (skips dead blocks) or dense-mask SDPA."""
    B, H, T, _ = q.shape
    if USE_FLEX and _HAVE_FLEX and q.is_cuda:
        d = death                                        # captured (B,T)

        def mask_mod(b, h, qi, ki):
            return (ki <= qi) & (qi < d[b, ki])

        block_mask = create_block_mask(mask_mod, B, 1, T, T,
                                       device=str(q.device))
        return flex_attention(q, k, v, block_mask=block_mask)
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=interval_mask(death, T), dropout_p=dropout_p)


class SlidingWindowAttention(nn.Module):
    """Causal attention over the last `cfg.window` tokens (the control arm)."""

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.window = cfg.window
        self.dropout = cfg.dropout
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.c_proj.RESIDUAL_SCALE_INIT = True
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def _death(self, x):
        B, T, _ = x.shape
        return swa_death_times(T, self.window, x.device).expand(B, T)

    def _qkv(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        shp = (B, T, self.n_head, C // self.n_head)
        return (q.view(shp).transpose(1, 2), k.view(shp).transpose(1, 2),
                v.view(shp).transpose(1, 2))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self._qkv(x)
        y = _attend(q, k, v, self._death(x),
                    self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class GatedSWAttention(SlidingWindowAttention):
    """Admission-gated windowed attention: G learned FIFO gates of capacity
    window/G, union-visible to all queries. See module docstring."""

    def __init__(self, cfg):
        super().__init__(cfg)
        assert cfg.window % cfg.n_gates == 0, (
            f"window {cfg.window} not divisible by n_gates {cfg.n_gates}")
        self.n_gates = cfg.n_gates
        self.capacity = cfg.window // cfg.n_gates
        self.router = nn.Linear(cfg.n_embd, cfg.n_gates, bias=False)
        self.collect_stats = False   # set by eval code; read after forward
        self.stats = {}

    def forward(self, x):
        B, T, C = x.shape
        logits = self.router(x)                       # (B,T,G)
        probs = F.softmax(logits.float(), dim=-1)
        gate = probs.argmax(dim=-1)                   # (B,T) hard choice
        p_sel = probs.gather(-1, gate.unsqueeze(-1))  # (B,T,1)

        q, k, v = self._qkv(x)
        # forward-identity multiplier: ==1.0 in fwd, routes gradient from
        # every future query that reads this token back into the router.
        scale = (p_sel / p_sel.detach()).to(v.dtype)
        v = v * scale.unsqueeze(1)                    # broadcast over heads

        death = gated_death_times(gate, self.capacity)
        y = _attend(q, k, v, death,
                    self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        if self.collect_stats:
            with torch.no_grad():
                hist = torch.bincount(gate.reshape(-1),
                                      minlength=self.n_gates).float()
                hist = hist / hist.sum()
                ent = -(probs * probs.clamp_min(1e-9).log()).sum(-1).mean()
                died = death < T
                life = (death - torch.arange(T, device=x.device))[died]
                self.stats = {
                    "gate_frac": hist.cpu(),
                    "router_entropy": float(ent),
                    "frac_evicted": float(died.float().mean()),
                    "mean_lifetime": float(life.float().mean())
                    if died.any() else float("nan"),
                }
        return self.resid_dropout(self.c_proj(y))
