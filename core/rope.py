"""Rotary position embeddings (RoPE), interleaved-pair convention.

Applied to q and k after head split, before attention. Position enters as a
rotation of each (even, odd) feature pair by angle pos * base^(-2i/d), so
scores depend only on relative offset. No parameters — which is the point
for long-context runs: a learned 4096-slot wpe trains each slot once per
sequence, while rotations need no training at all.

Works unchanged with the admission gates: an evicted-or-retained key keeps
the rotation from its ORIGINAL position (exactly like every production KV
cache stores rotated keys), so retention policy and position encoding stay
orthogonal.
"""
import torch

_CACHE = {}


def _cos_sin(T, head_dim, device, dtype, base=10000.0):
    key = (T, head_dim, str(device), dtype)
    hit = _CACHE.get(key)
    if hit is None:
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device,
                                           dtype=torch.float32) / head_dim))
        ang = torch.outer(torch.arange(T, device=device,
                                       dtype=torch.float32), inv)
        hit = (ang.cos().to(dtype)[None, None],   # (1,1,T,hd/2)
               ang.sin().to(dtype)[None, None])
        _CACHE[key] = hit
    return hit


def apply_rope(q, k):
    """q, k: (B, H, T, head_dim) -> rotated copies, same shape/dtype."""
    T, hd = q.shape[-2], q.shape[-1]
    cos, sin = _cos_sin(T, hd, q.device, q.dtype)

    def rot(x):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack((x1 * cos - x2 * sin,
                            x1 * sin + x2 * cos), dim=-1).flatten(-2)

    return rot(q), rot(k)
