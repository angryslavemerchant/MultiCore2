"""Deterministic, cached FineWeb-Edu tokens. Adapted from MultiCore's
scripts/m5_data.py, token mode only (the byte corpus was that project's
concern, not this one's).

The corpus is `HuggingFaceFW/fineweb-edu` `sample/100BT`: GPT-NeoX subword
ids + `<|endoftext|>` (id 0) as the document separator, concatenated into one
flat uint16 file. Two model configs sampling from the same file with the same
seed see the identical sequence of batches, which is the whole point -- the
baseline and the new architecture must be fed the same data or their losses
are not comparable.

uint16 caps the vocabulary at 65535; the NeoX vocab is 50277 (50304 padded)
and `build_token_cache` asserts every id fits rather than trusting that.

Pipeline:
  1. bank.try_pull the finished cache from the Drive bank (~90 s). The
     5-shard cache (3.76B tokens) already exists there, built 2026-07-28.
  2. Failing that, hf_hub_download the deterministic first N parquet shards
     and tokenise locally (minutes per shard; progress is recorded per shard
     in a `.progress` sidecar so a restart resumes rather than starting over).
  3. Sample batches as B random contiguous (T+1)-token windows from a seeded
     numpy Generator. The last EVAL_FRAC of the file is eval-only; training
     samples strictly from the head, so eval tokens are never trained on.

Standalone sanity check:  python core/data.py --shards 5
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

REPO_ID = "HuggingFaceFW/fineweb-edu"
SUBDIR = "sample/100BT"       # ~0.75B NeoX tokens per 2.15 GB parquet shard
TOKENIZER = "EleutherAI/gpt-neox-20b"
VOCAB_SIZE = 50304            # 50277 real ids, padded like open-sci-ref
EOS_ID = 0                    # <|endoftext|> in the NeoX vocab
DEFAULT_SHARDS = 5
DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
EVAL_FRAC = 0.05              # last 5% of the file is eval-only


# ---------------------------------------------------------------- hub fetch
def _retry(fn, what, attempts=5, base_delay=2.0):
    """Run fn(), retrying transient hub/IO failures with exponential backoff."""
    for a in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            if a == attempts:
                raise
            delay = base_delay * 2 ** (a - 1)
            print(f"[data] {what}: attempt {a}/{attempts} failed "
                  f"({type(e).__name__}: {e}); retrying in {delay:.0f}s",
                  flush=True)
            time.sleep(delay)


def shard_names(n_shards, repo_id=REPO_ID, subdir=SUBDIR):
    """The first `n_shards` parquet shard names, as listed by the hub."""
    names = None
    try:
        from huggingface_hub import list_repo_files
        files = _retry(lambda: list_repo_files(repo_id, repo_type="dataset"),
                       "list_repo_files", attempts=3)
        names = sorted(f for f in files
                       if f.startswith(subdir + "/") and f.endswith(".parquet"))
    except Exception as e:
        print(f"[data] list_repo_files failed ({type(e).__name__}: {e}); "
              f"falling back to the known {subdir} naming", flush=True)
    if not names:
        names = [f"{subdir}/{i:03d}_00000.parquet" for i in range(n_shards)]
    if len(names) < n_shards:
        raise RuntimeError(f"{repo_id}/{subdir} has only {len(names)} shards, "
                           f"asked for {n_shards}")
    return names[:n_shards]


def _fetch(name, repo_id):
    from huggingface_hub import hf_hub_download
    print(f"[data] fetching {name}", flush=True)
    return _retry(lambda: hf_hub_download(repo_id=repo_id, filename=name,
                                          repo_type="dataset"),
                  f"download {name}")


# --------------------------------------------------------------- token cache
def token_cache_path(n_shards=DEFAULT_SHARDS, data_dir=None,
                     tokenizer=TOKENIZER):
    tag = tokenizer.split("/")[-1].replace(".", "-")
    return os.path.join(data_dir or DEFAULT_DATA_DIR,
                        f"fineweb100_{tag}_{n_shards}shards_u16.bin")


def build_token_cache(n_shards=DEFAULT_SHARDS, data_dir=None, repo_id=REPO_ID,
                      subdir=SUBDIR, tokenizer=TOKENIZER, doc_batch=1024):
    """Return the path to the flat uint16 token file, building it if needed.

    Documents are concatenated as `ids + [eos]`, so a sampled window may span
    a document boundary and the model sees the separator as an ordinary
    token. That is what essentially every pretraining pipeline does.

    `tokenizer` is a hub name, or an already-loaded object exposing
    `__call__(list_of_str) -> {"input_ids": ...}` and `eos_token_id` -- which
    is how tests drive this without a network round trip or a 300 MB import.
    """
    name = tokenizer if isinstance(tokenizer, str) else getattr(
        tokenizer, "name_or_path", "custom")
    out = token_cache_path(n_shards, data_dir, name)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        n = os.path.getsize(out) // 2
        print(f"[data] cache hit: {out} ({n} tokens, {n / 1e9:.2f}B)",
              flush=True)
        return out
    # Drive before tokenising: the pull is ~90 s against ~58 min of CPU for
    # 5 shards. Failure here is not an error -- try_pull never raises, and a
    # False just means this box builds the cache the slow way (and can then
    # publish it with `python core/data.py --push`).
    try:
        from bank import try_pull
    except ImportError:
        try_pull = None
    if try_pull is not None and try_pull(out):
        n = os.path.getsize(out) // 2
        print(f"[data] bank hit: {out} ({n} tokens, {n / 1e9:.2f}B)",
              flush=True)
        return out
    import pyarrow.parquet as pq
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp, prog = out + ".partial", out + ".progress"
    names = shard_names(n_shards, repo_id, subdir)
    if isinstance(tokenizer, str):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer)
    else:
        tok = tokenizer
    eos = tok.eos_token_id
    assert eos is not None, f"{name} has no eos_token_id to separate docs"

    done, total, ndocs = 0, 0, 0
    if os.path.exists(prog) and os.path.exists(tmp):
        st = json.load(open(prog))
        if st.get("names") == names and st.get("tokenizer") == name:
            done, total, ndocs = st["shards"], st["tokens"], st["docs"]
            with open(tmp, "r+b") as f:          # drop a half-written shard
                f.truncate(total * 2)
            print(f"[data] resuming at shard {done}/{len(names)} "
                  f"({total / 1e9:.2f}B tokens already written)", flush=True)
        else:
            os.remove(prog)
    print(f"[data] building {out} from {len(names)} shard(s) of {subdir} "
          f"with {name}", flush=True)
    t0 = time.time()
    reported = total
    with open(tmp, "r+b" if done else "wb", buffering=1 << 22) as f:
        f.seek(total * 2)
        for si, shard in enumerate(names):
            if si < done:
                continue
            path = _fetch(shard, repo_id)
            pf = pq.ParquetFile(path)
            for rb in pf.iter_batches(batch_size=doc_batch, columns=["text"]):
                texts = [t for t in rb.column("text").to_pylist() if t]
                if not texts:
                    continue
                ids = tok(texts, add_special_tokens=False)["input_ids"]
                arr = np.concatenate(
                    [np.asarray(d + [eos], dtype=np.int64) for d in ids])
                hi = int(arr.max())
                assert hi < 65536, (
                    f"token id {hi} does not fit uint16; {name} is too "
                    f"large for this cache format")
                f.write(arr.astype(np.uint16).tobytes())
                total += int(arr.size)
                ndocs += len(texts)
                if total - reported > (1 << 27):    # every ~134M tokens
                    reported = total
                    print(f"[data]   {total / 1e9:.3f}B tokens, {ndocs} docs, "
                          f"{time.time() - t0:.0f}s", flush=True)
            f.flush()
            json.dump({"names": names, "tokenizer": name, "shards": si + 1,
                       "tokens": total, "docs": ndocs}, open(prog, "w"))
    if total == 0:
        os.remove(tmp)
        raise RuntimeError("no tokens written -- parquet had no 'text' column?")
    os.replace(tmp, out)
    if os.path.exists(prog):
        os.remove(prog)
    print(f"[data] wrote {out}: {total} tokens ({total / 1e9:.3f}B, "
          f"{total * 2 / 1e9:.2f} GB) from {ndocs} docs in "
          f"{time.time() - t0:.0f}s", flush=True)
    return out


# ---------------------------------------------------------------- sampler
class TokenData:
    """Seeded random contiguous windows over the flat uint16 token file.

    The last `eval_frac` of the file is reserved for eval; train windows lie
    entirely in the head, so the two splits share no tokens.
    """

    def __init__(self, path, eval_frac=EVAL_FRAC):
        self.path = path
        self.arr = np.memmap(path, dtype=np.uint16, mode="r")
        self.n = int(self.arr.shape[0])
        self.n_train = int(self.n * (1.0 - eval_frac))

    def bounds(self, split):
        if split == "train":
            return 0, self.n_train
        if split == "eval":
            return self.n_train, self.n
        raise ValueError(f"unknown split {split!r}")

    def batches(self, B, T, device, seed=0, split="train", skip=0, stride=1):
        """Yields (B, T+1) long tensors on `device`.

        `skip` advances the generator past the first `skip` batches WITHOUT
        reading them: a resumed run has to continue the stream, not replay
        it. Only the RNG draw is repeated, never the gather, so
        fast-forwarding 15k batches costs microseconds.

        `stride` skips (stride - 1) batches after each yield. With
        skip=rank, stride=world_size, the N ranks of a data-parallel run
        partition ONE stream between them -- rank r takes batches r, r+W,
        r+2W. Their union is exactly the single-GPU stream, so an 8-GPU run
        and a 1-GPU run at the same token count see the same data. Without
        this every rank draws the identical batch and 8 GPUs buy 8x the cost
        for 1x the data, which trains and converges and looks completely
        normal on a loss curve.
        """
        import torch
        lo, hi = self.bounds(split)
        last = hi - (T + 1)
        if last < lo:
            raise RuntimeError(
                f"{split} split of {self.path} holds {hi - lo} tokens, too "
                f"few for a {T + 1}-token window (build more shards)")
        rng = np.random.default_rng(seed)
        for _ in range(skip):
            rng.integers(lo, last, size=B, endpoint=True)
        buf = np.empty((B, T + 1), dtype=np.uint16)
        while True:
            starts = rng.integers(lo, last, size=B, endpoint=True)
            for i, s in enumerate(starts):
                buf[i] = self.arr[s:s + T + 1]
            # torch has no from_numpy path for uint16; the int32 widening is
            # nothing next to the step it feeds.
            yield torch.from_numpy(buf.astype(np.int32)).to(device).long()
            for _ in range(stride - 1):        # the other ranks' batches
                rng.integers(lo, last, size=B, endpoint=True)


def open_data(n_shards=DEFAULT_SHARDS, data_dir=None, rank=0, world=1,
              wait_s=7200):
    """The token cache, building it if needed.

    ONE builder under data parallel. Every rank runs this code, and eight
    processes building the same cache share one `.partial` and one
    `os.replace`, so the first to finish renames the file out from under the
    others and they die with FileNotFoundError. Rank 0 builds; the others
    poll for the finished file. Deliberately a filesystem poll rather than
    `dist.barrier()`: NCCL's watchdog aborts collectives that block for tens
    of minutes, and a cold build legitimately takes ~50.
    """
    if world > 1 and rank != 0:
        path = token_cache_path(n_shards, data_dir)
        t0 = time.time()
        while not (os.path.exists(path) and os.path.getsize(path) > 0):
            if time.time() - t0 > wait_s:
                raise RuntimeError(
                    f"rank {rank} waited {wait_s}s for rank 0 to build {path}")
            time.sleep(10)
        return TokenData(path)
    return TokenData(build_token_cache(n_shards, data_dir))


# ---------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(
        description="build and sanity-check the FineWeb-Edu token cache")
    ap.add_argument("--shards", type=int, default=DEFAULT_SHARDS)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--push", action="store_true",
                    help="publish the finished cache to the Drive bank")
    args = ap.parse_args()

    import torch
    d = open_data(args.shards, args.data_dir)
    print(f"[data] {d.path}: {d.n} tokens ({d.n / 1e9:.3f}B); "
          f"train [0, {d.n_train}), eval [{d.n_train}, {d.n})")
    if args.push:
        from bank import push
        push(d.path)

    def first(split):
        return next(TokenData(d.path).batches(
            args.batch, args.seq_len, "cpu", seed=args.seed, split=split))

    a, b = first("train"), first("train")
    print(f"[data] determinism (train, seed={args.seed}): "
          f"{'IDENTICAL' if torch.equal(a, b) else 'MISMATCH'}")
    e1, e2 = first("eval"), first("eval")
    print(f"[data] determinism (eval,  seed={args.seed}): "
          f"{'IDENTICAL' if torch.equal(e1, e2) else 'MISMATCH'}")
    print("[data] first 200 tokens of a training sample:")
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(TOKENIZER)
    print(repr(tk.decode(a[0, :200].tolist())))


if __name__ == "__main__":
    main()
