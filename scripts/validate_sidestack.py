"""On-GPU gate for the top-k side-stack (pattern T) before any run.

The flat-loss lesson (2026-08-08): a forward-only compiled_vs_eager check
passed while compiled flex backward silently corrupted every S-layer
gradient. This gate compares compiled vs eager FORWARD AND BACKWARD on the
real run config: per-parameter grad cosine must be ~1 everywhere. The
zero-init lm_head is randomized first -- at step 1 it multiplies every
trunk gradient by exactly 0, which makes any grad comparison vacuous.

Usage (on the box):
  python scripts/validate_sidestack.py            # real cfg, T=4096
  python scripts/validate_sidestack.py --seq 1024 # smoke
Emits VALIDATE_JSON lines; every "ok" must be true.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model import GPT, GPTConfig  # noqa: E402


def emit(name, **kw):
    print("VALIDATE_JSON", json.dumps({name: kw}), flush=True)
    return kw.get("ok", True)


def grads(model, x, y, compiled):
    m = torch.compile(model) if compiled else model
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, loss = m(x, targets=y)
    loss.backward()
    return (logits.detach().float(),
            {n: p.grad.detach().float().clone()
             for n, p in model.named_parameters() if p.grad is not None})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=4096)
    args = ap.parse_args()
    torch.manual_seed(1337)
    for attr in ("recompile_limit", "cache_size_limit"):
        if hasattr(torch._dynamo.config, attr):
            setattr(torch._dynamo.config, attr, 64)

    cfg = GPTConfig(
        block_size=args.seq, vocab_size=50304, n_layer=12, n_head=12,
        n_embd=1152, pos="rope", norm="rms", mlp="relu2", bias=False,
        qk_norm=True, untied=True, zero_init=True, softcap=15.0,
        canon=True, canon_full=True, window=128,
        attn_pattern="SSSFRRRTSSSF", side_topk=16,
        hg_frac=0.3, hg_bneck=8, hg_round=96)
    model = GPT(cfg).cuda()
    with torch.no_grad():
        # the zero-init head blocks every trunk gradient at step 1
        model.lm_head.weight.normal_(0, 0.02)
        # likewise the branch's zero-init output projection blocks every
        # upstream branch gradient for exactly one step -- randomize so
        # the aliveness check sees real backward traffic
        for n, p in model.named_parameters():
            if ".side." in n and n.endswith("c_proj.weight"):
                p.normal_(0, 0.02)

    x = torch.randint(0, 50304, (1, args.seq), device="cuda")
    y = torch.randint(0, 50304, (1, args.seq), device="cuda")

    lg_e, g_e = grads(model, x, y, compiled=False)
    lg_c, g_c = grads(model, x, y, compiled=True)

    ok_all = emit("fwd_compiled_vs_eager",
                  ok=bool((lg_e - lg_c).abs().max() < 5e-2),
                  max_abs_logit_diff=float((lg_e - lg_c).abs().max()))

    worst = []
    for n in g_e:
        a, b = g_e[n].flatten(), g_c[n].flatten()
        if a.norm() == 0 and b.norm() == 0:
            continue
        cos = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
        worst.append((cos, n))
    worst.sort()
    ok = all(c > 0.999 for c, _ in worst)
    ok_all &= emit("bwd_grad_cosine", ok=bool(ok),
                   n_params=len(worst),
                   worst=[(round(c, 6), n) for c, n in worst[:8]])

    side_named = [n for n, _ in model.named_parameters() if ".side." in n]
    side_grads = [n for n in side_named if n in g_e and g_e[n].norm() > 0]
    ok_all &= emit("side_branch_grads_alive",
                   ok=len(side_grads) == len(side_named),
                   alive=len(side_grads), total=len(side_named))

    print("VALIDATE_RESULT", "PASS" if ok_all else "FAIL", flush=True)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
