"""Decompose the v0 vs v0.1 chunk-arm training-speed gap on one GPU.

History (8x, different boxes — confounded): v0 (K12 d1056 btok256 K16
soft) ran 639k tok/s on the original box; v0.1 (SSSKKKKKKSSS d1152
btok64 K2 topk16) ran 403k tok/s on the mediocre-ring EPYC box
(~400ms/iter comm tax measured then). Suspect beyond the box: v0.1's
hard top-k membership gathers from an expanded (B,H,S,T,hd) view whose
BACKWARD materialises the full source and scatter-adds with atomics —
the exact bug family measured at 78%-of-step in chunkv2 (2026-08-08).
v0's soft path is pure GEMMs and never runs it.

Three timed configs (synthetic tokens, no cache needed):
    v0 exact / v0.1 exact / v0.1-but-soft (isolates the top-k gather).
Single-GPU wall clock; the leftover vs the historical 8x numbers is
the box's ring quality.
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model import GPT, GPTConfig                          # noqa: E402

for attr in ("recompile_limit", "cache_size_limit"):
    if hasattr(torch._dynamo.config, attr):
        setattr(torch._dynamo.config, attr, 64)


def cfg(**kw):
    base = dict(block_size=4096, vocab_size=50304, n_layer=12, n_head=12,
                pos="rope", norm="rms", mlp="relu2", bias=False,
                qk_norm=True, canon=True, canon_full=True, untied=True,
                zero_init=True, softcap=15.0, hg_frac=0.3, hg_bneck=8,
                hg_round=96)
    base.update(kw)
    return GPTConfig(**base)


def bench(tag, c, iters=18, mb=2):
    torch.manual_seed(0)
    model = GPT(c).cuda()
    opt = model.configure_optimizers(0.1, 6e-4, (0.9, 0.95), "cuda",
                                     opt="muon")
    mc = torch.compile(model)
    idx = torch.randint(0, 50304, (mb, 4096), device="cuda")
    t0 = None
    for i in range(iters):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = mc(idx, targets=idx)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if i == 4:
            torch.cuda.synchronize()
            t0 = time.time()
    torch.cuda.synchronize()
    ms = (time.time() - t0) / (iters - 5) * 1000
    print(f"BENCH {tag}: {ms:.0f}ms/iter (mb{mb})", flush=True)
    del model, mc, opt
    torch.cuda.empty_cache()


def main():
    assert torch.cuda.is_available()
    print(f"[speed] {torch.cuda.get_device_name(0)}", flush=True)
    bench("v0_exact_K12_d1056_w256_b256K16_soft",
          cfg(n_embd=1056, attn_pattern="K" * 12, window=256,
              chunk_btok=256, chunk_k=16, chunk_topk=0))
    bench("v01_exact_K6mid_d1152_w256_b64K2_topk16",
          cfg(n_embd=1152, attn_pattern="SSSKKKKKKSSS", window=256,
              chunk_btok=64, chunk_k=2, chunk_topk=16))
    bench("v01_soft_control_same_but_topk0",
          cfg(n_embd=1152, attn_pattern="SSSKKKKKKSSS", window=256,
              chunk_btok=64, chunk_k=2, chunk_topk=0))
    print("SPEED_BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
