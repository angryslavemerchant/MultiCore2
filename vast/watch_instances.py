"""Emit an event line whenever a watched vast.ai instance goes down.

    python vast/watch_instances.py [ID ...] [--interval 120] [--once]

Polls `vastai show instances --raw` (VAST_API_KEY from env) and prints one
line per state transition, so it slots straight into a Monitor/watcher:

    INSTANCE_DOWN 47225408 (was: running)        <- destroyed / reaped
    INSTANCE_STATUS 47205967 running -> exited   <- stopped, crashed host
    INSTANCE_UP 47230000 running                 <- new id appeared (no-arg mode)
    VAST_API_UNREACHABLE (xN)                    <- CLI errors, with tripwire

With explicit IDs, exits once ALL watched ids are gone (natural end for a
single-notification watcher). With no IDs it watches everything on the
account forever. API-unreachable never counts as down — a box is only
DOWN when a successful listing omits it (the 2026-08-09 bench box
self-destroyed unseen; SSH-silence is ambiguous, the account listing is
authoritative).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time


def cli():
    exe_dir = os.path.dirname(sys.executable)
    for d in (None, exe_dir, os.path.join(exe_dir, "Scripts"), "/venv/main/bin"):
        for cand in ("vastai", "vastai.exe"):
            p = shutil.which(cand, path=d) if d else shutil.which(cand)
            if p:
                return [p]
    raise RuntimeError("vastai CLI not found near " + sys.executable)


def poll():
    proc = subprocess.run(cli() + ["show", "instances", "--raw"],
                          capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[-200:])
    return {int(d["id"]): d.get("actual_status") or "unknown"
            for d in json.loads(proc.stdout)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", type=int)
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    assert os.environ.get("VAST_API_KEY"), "VAST_API_KEY not in env"

    prev, fails = None, 0
    while True:
        try:
            cur = poll()
            fails = 0
        except Exception as e:  # noqa: BLE001 - transient API errors
            fails += 1
            print(f"VAST_API_UNREACHABLE (x{fails}): {e}", flush=True)
            if fails >= 5:
                print("VAST_API_TRIPWIRE: 5 consecutive failures", flush=True)
            time.sleep(args.interval)
            continue
        watched = args.ids or (sorted(prev) if prev is not None else [])
        if prev is not None:
            for i in watched:
                if i in prev and i not in cur:
                    print(f"INSTANCE_DOWN {i} (was: {prev[i]})", flush=True)
                elif i in prev and i in cur and prev[i] != cur[i]:
                    print(f"INSTANCE_STATUS {i} {prev[i]} -> {cur[i]}",
                          flush=True)
            if not args.ids:
                for i in sorted(set(cur) - set(prev)):
                    print(f"INSTANCE_UP {i} {cur[i]}", flush=True)
        else:
            for i in (args.ids or sorted(cur)):
                print(f"WATCHING {i} {cur.get(i, 'ABSENT')}", flush=True)
                if args.ids and i not in cur:
                    print(f"INSTANCE_DOWN {i} (was: unseen)", flush=True)
        prev = cur
        if args.ids and all(i not in cur for i in args.ids):
            print("ALL_WATCHED_INSTANCES_GONE", flush=True)
            return
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
