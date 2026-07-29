"""Benchmark every trained model in the Drive bank on the LM suite.

    python scripts/bench_all_ckpts.py [--batch-size 128] [--force]

For each run folder under gdrive:multicore2-runs/ that has a best.pt:
pull the checkpoint, run scripts/lm_eval_bench.py on it, push the
resulting lm_eval.json back next to the checkpoint. IDEMPOTENT: run
folders that already show an lm_eval.json on Drive are skipped (--force
overrides), so re-invoking after later training arms finish benches only
the new arrivals. Installs lm-eval on demand (not in requirements.txt —
training instances don't need it).

Designed to be a vast.ai --train-script: ignores unknown args (--wandb).
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bank

RUNS_FOLDER = "multicore2-runs"


def lsjson(path):
    proc = subprocess.run(["rclone", "lsjson", path], capture_output=True,
                          text=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"rclone lsjson {path}: {proc.stderr[-300:]}")
    return json.loads(proc.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--force", action="store_true")
    args, _ = ap.parse_known_args()

    try:
        import lm_eval  # noqa: F401
    except ImportError:
        print("[bench_all] installing lm-eval", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "lm-eval"], check=True, timeout=900)

    assert bank._rclone_ready(), "no rclone/credentials"
    remote = f"{bank.REMOTE}:{RUNS_FOLDER}"
    runs = [d["Name"] for d in lsjson(remote) if d.get("IsDir")]
    print(f"[bench_all] {len(runs)} run(s) on the bank: {runs}", flush=True)

    done, skipped, failed = [], [], []
    for name in runs:
        files = {f["Name"]: f["Size"] for f in lsjson(f"{remote}/{name}")}
        if "best.pt" not in files:
            print(f"[bench_all] {name}: no best.pt, skipping", flush=True)
            skipped.append(name)
            continue
        if "lm_eval.json" in files and not args.force:
            print(f"[bench_all] {name}: already benchmarked, skipping",
                  flush=True)
            skipped.append(name)
            continue
        local = os.path.join("runs", name, "best.pt")
        if not (os.path.exists(local)
                and os.path.getsize(local) == files["best.pt"]):
            print(f"[bench_all] pulling {name}/best.pt "
                  f"({files['best.pt'] / 1e9:.2f} GB)", flush=True)
            if not bank.try_pull(local, folder=f"{RUNS_FOLDER}/{name}"):
                print(f"[bench_all] {name}: pull failed", flush=True)
                failed.append(name)
                continue
        proc = subprocess.run(
            [sys.executable, "scripts/lm_eval_bench.py", "--run-name", name,
             "--batch-size", str(args.batch_size)], timeout=7200)
        if proc.returncode != 0:
            print(f"[bench_all] {name}: eval FAILED rc={proc.returncode}",
                  flush=True)
            failed.append(name)
            continue
        bank.push(os.path.join("runs", name, "lm_eval.json"),
                  folder=f"{RUNS_FOLDER}/{name}")
        done.append(name)
        print(f"BENCHED {name}", flush=True)

    print(f"BENCH_ALL_DONE done={done} skipped={skipped} failed={failed}",
          flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
