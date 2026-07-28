"""vast/upload_results.py — push run outputs to the Google Drive bank.

wandb Artifacts abandoned 2026-07-28 (no storage on the account); metrics
still stream to wandb, but checkpoints live on Drive:

    gdrive:multicore2-runs/<run-name>/{best.pt, latest.pt, metrics.json, ...}

bank.push verifies the committed remote size and raises otherwise, and this
script exits non-zero if any push ultimately fails — run_training.sh gates
self-destroy on that, so a Drive outage can never destroy the only copy of
the weights. Fetch later with:

    rclone copy gdrive:multicore2-runs/<run-name>/ runs/<run-name>/
"""
import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bank import push

RUNS_FOLDER = "multicore2-runs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viz_dir",  type=str, default="figures")
    ap.add_argument("--ckpt_dir", type=str, default="runs")
    ap.add_argument("--extra",    type=str, nargs="*", default=[])
    args = ap.parse_args()

    run_name = os.path.basename(os.path.normpath(args.ckpt_dir))
    folder = f"{RUNS_FOLDER}/{run_name}"
    files = [os.path.join(args.ckpt_dir, n)
             for n in ("best.pt", "latest.pt", "metrics.json")]
    files += list(args.extra)
    files += sorted(glob.glob(os.path.join(args.viz_dir, "*.png")))

    pushed, failed = 0, 0
    for p in files:
        if not os.path.exists(p):
            continue
        for attempt in range(1, 4):
            try:
                push(p, folder=folder)
                pushed += 1
                break
            except Exception as e:
                print(f"push {p} failed ({e!r}) — attempt {attempt}/3",
                      flush=True)
                time.sleep(20 * attempt)
        else:
            failed += 1
    if failed or pushed == 0:
        sys.exit(f"upload incomplete: {failed} failed, {pushed} pushed")
    print(f"DRIVE_UPLOAD_OK {pushed} file(s) -> {folder}", flush=True)


if __name__ == "__main__":
    main()
