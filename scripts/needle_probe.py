"""Needle-logprob probe: can the model recall a sequence it saw once,
thousands of tokens ago?

    python scripts/needle_probe.py --run-name 124m-FGGGGFFGGGGF-w256-g4-r128-rope-t4096

The instruction-following needle-in-a-haystack test is out of reach for a
124M base model, but its logprob cousin is not: induction-style copying is
one of the first capabilities small transformers develop. Plant a needle
[prefix P + payload Y] of random tokens inside real FineWeb text, end the
sequence with P again, and measure the model's logprob on Y. Subtract the
logprob of Y in the identical sequence WITHOUT the planted needle and the
prior cancels: delta > 0 means the model retrieved the earlier occurrence.
Sweeping the needle's distance from the query turns that into a curve of
recall vs distance -- the direct test of whether archive gates hold
information a sliding window has forgotten.

Haystack text comes from the token cache if present, else a small prefix of
the bank file is pulled (`rclone cat --count`), so this runs on a bench box
without the 7.5 GB cache download.

Results go to runs/<run-name>/needle_probe.json.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bank                                                    # noqa: E402
from core.data import token_cache_path, DEFAULT_DATA_DIR       # noqa: E402
from core.model import GPT, GPTConfig                          # noqa: E402

# 50277 real NeoX ids; 0 is eos. Needles sample [1, 50277) so every needle
# token is a real, non-separator id.
REAL_VOCAB = 50277
DISTANCES = (128, 256, 512, 1024, 2048, 3584)
PROBE_TOKENS = 20_000_000       # 40 MB prefix is plenty of haystack


def ensure_corpus():
    """A uint16 token memmap for haystack text: the full cache if this
    machine has it, else a pulled prefix of the bank file."""
    full = token_cache_path()
    if os.path.exists(full) and os.path.getsize(full) > 0:
        return np.memmap(full, dtype=np.uint16, mode="r")
    prefix = os.path.join(DEFAULT_DATA_DIR, "probe_prefix_u16.bin")
    want = PROBE_TOKENS * 2
    if not (os.path.exists(prefix) and os.path.getsize(prefix) == want):
        assert bank._rclone_ready(), "no cache and no rclone/credentials"
        conf_args, cleanup = bank._conf_args()
        try:
            os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)
            print(f"[probe] pulling {want / 1e6:.0f} MB haystack prefix",
                  flush=True)
            with open(prefix + ".tmp", "wb") as f:
                subprocess.run(
                    ["rclone", "cat", f"{bank.BANK_REMOTE}/{bank.BANK_FILE}",
                     "--count", str(want)] + conf_args,
                    stdout=f, check=True, timeout=1800)
            assert os.path.getsize(prefix + ".tmp") == want, "short read"
            os.replace(prefix + ".tmp", prefix)
        finally:
            cleanup()
    return np.memmap(prefix, dtype=np.uint16, mode="r")


def build_trials(corpus, rng, T, dist, n, k, m):
    """(with, without) int64 arrays of shape (n, T).

    Layout of each row (positions):      query prefix   payload
        [ ...haystack... P Y ...haystack... P            Y       ]
                         ^ needle starts at qs - dist    ^ [T-m, T)
    `without` is the identical row with untouched haystack where the needle
    was.  Logits at positions [T-m-1, T-1) predict the payload.
    """
    qs = T - m - k                        # query prefix start
    ns = qs - dist                        # needle start
    assert ns >= 0, f"distance {dist} does not fit in T={T}"
    assert dist >= k + m, "needle would overlap the query prefix"
    starts = rng.integers(0, len(corpus) - T, size=n)
    hay = np.stack([np.asarray(corpus[s:s + T]) for s in starts]).astype(
        np.int64)
    needle = rng.integers(1, REAL_VOCAB, size=(n, k + m))
    with_ = hay.copy()
    with_[:, ns:ns + k + m] = needle
    with_[:, qs:qs + k] = needle[:, :k]
    with_[:, T - m:] = needle[:, k:]
    without = hay.copy()
    without[:, qs:qs + k] = needle[:, :k]
    without[:, T - m:] = needle[:, k:]
    return with_, without


@torch.no_grad()
def payload_logprob(model, x, m, device, batch_size):
    """Mean per-token logprob of the last `m` tokens of each row, plus
    greedy top-1 accuracy over those tokens. x: (n, T) int64 array."""
    T = x.shape[1]
    lps, accs = [], []
    for s in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[s:s + batch_size]).to(device)
        amp = (torch.autocast(device, dtype=torch.bfloat16)
               if device == "cuda" else torch.autocast("cpu", enabled=False))
        with amp:
            logits, _ = model(xb, xb)
        # only the m positions that predict the payload; float AFTER the
        # slice so the (B, T, V) tensor never materialises in fp32
        rows = logits[:, T - m - 1:T - 1, :].float()
        lp = torch.log_softmax(rows, dim=-1)
        tgt = xb[:, T - m:]
        lps.append(lp.gather(-1, tgt[..., None])[..., 0].mean(-1).cpu())
        accs.append((lp.argmax(-1) == tgt).float().mean(-1).cpu())
    return torch.cat(lps).numpy(), torch.cat(accs).numpy()


def run_probe(model, corpus, T, distances, trials, k, m, batch_size, device,
              seed):
    out = {}
    for dist in distances:
        rng = np.random.default_rng(seed + dist)
        w, wo = build_trials(corpus, rng, T, dist, trials, k, m)
        lp_w, acc_w = payload_logprob(model, w, m, device, batch_size)
        lp_wo, acc_wo = payload_logprob(model, wo, m, device, batch_size)
        delta = lp_w - lp_wo
        out[dist] = {
            "delta_mean": float(delta.mean()),
            "delta_se": float(delta.std(ddof=1) / np.sqrt(trials)),
            "logprob_with": float(lp_w.mean()),
            "logprob_without": float(lp_wo.mean()),
            "acc_with": float(acc_w.mean()),
            "acc_without": float(acc_wo.mean()),
        }
        r = out[dist]
        print(f"[probe] d={dist:>5}  delta={r['delta_mean']:+7.3f} "
              f"(se {r['delta_se']:.3f})  acc {r['acc_without']:.1%}"
              f" -> {r['acc_with']:.1%}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--seq-len", type=int, default=4096,
                    help="capped at the model's block_size")
    ap.add_argument("--trials", type=int, default=64)
    ap.add_argument("--needle-prefix", type=int, default=8)
    ap.add_argument("--needle-payload", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(os.path.join("runs", args.run_name, args.ckpt),
                      map_location=device)
    from core.model import model_from_ckpt_config
    cfg, model = model_from_ckpt_config(ckpt["config"])
    model = model.to(device).eval()
    # v0-era chunk checkpoints predate the v0.1 writer gain (init 1.0 ==
    # the exact soft behavior they trained with) — tolerate ONLY that
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    assert not unexpected, unexpected
    assert all(m.endswith(".gain") for m in missing), missing
    T = min(args.seq_len, cfg.block_size)
    k, m = args.needle_prefix, args.needle_payload
    dists = [d for d in DISTANCES if d + k + m + m <= T]
    print(f"[probe] {args.run_name}: pattern "
          f"{getattr(cfg, 'attn_pattern', '') or type(cfg).__name__}, "
          f"T={T}, distances {dists}, {args.trials} trials each", flush=True)

    corpus = ensure_corpus()
    results = run_probe(model, corpus, T, dists, args.trials, k, m,
                        args.batch_size, device, args.seed)
    out = {"seq_len": T, "trials": args.trials,
           "needle": {"prefix": k, "payload": m}, "seed": args.seed,
           "distances": results}
    path = os.path.join("runs", args.run_name, "needle_probe.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
