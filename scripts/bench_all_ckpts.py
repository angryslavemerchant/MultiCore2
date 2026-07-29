"""Benchmark every trained model in the Drive bank on the LM suite.

    python scripts/bench_all_ckpts.py [--batch-size 128] [--force]

For each run folder under gdrive:multicore2-runs/ that has a best.pt:
pull the checkpoint, run scripts/lm_eval_bench.py and
scripts/needle_probe.py on it, push the resulting lm_eval.json /
needle_probe.json back next to the checkpoint. IDEMPOTENT per artifact:
whichever result files a run folder already shows on Drive are skipped
(--force overrides), so re-invoking benches only what is missing.
Installs lm-eval on demand (not in requirements.txt — training instances
don't need it).

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

    # result file -> the command that produces it (run from repo root)
    jobs = {
        "lm_eval.json": [sys.executable, "scripts/lm_eval_bench.py",
                         "--batch-size", str(args.batch_size)],
        "needle_probe.json": [sys.executable, "scripts/needle_probe.py"],
        "stream_probe.json": [sys.executable, "scripts/stream_probe.py"],
    }
    done, skipped, failed = [], [], []
    for name in runs:
        files = {f["Name"]: f["Size"] for f in lsjson(f"{remote}/{name}")}
        if "best.pt" not in files:
            print(f"[bench_all] {name}: no best.pt, skipping", flush=True)
            skipped.append(name)
            continue
        todo = [a for a in jobs if a not in files or args.force]
        if not todo:
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
        ok = True
        for artifact in todo:
            proc = subprocess.run(jobs[artifact] + ["--run-name", name],
                                  timeout=7200)
            if proc.returncode != 0:
                print(f"[bench_all] {name}: {artifact} FAILED "
                      f"rc={proc.returncode}", flush=True)
                ok = False
                continue
            bank.push(os.path.join("runs", name, artifact),
                      folder=f"{RUNS_FOLDER}/{name}")
        (done if ok else failed).append(name)
        if ok:
            print(f"BENCHED {name}", flush=True)

    print(f"BENCH_ALL_DONE done={done} skipped={skipped} failed={failed}",
          flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
