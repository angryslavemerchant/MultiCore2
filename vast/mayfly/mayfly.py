#!/usr/bin/env python3
"""mayfly — a dumb, loud, disposable remote-log watcher.

The WATCHTOWER mechanism layer (see ../WATCHTOWER.md) as one portable
file with zero dependencies. It polls a log on a remote box over ssh at
a fixed interval, applies regex rules, and — this is the whole design —
**exits the moment anything decision-worthy happens**. It never judges,
never fixes, never retries beyond its patience. In an agent harness,
run it as a *background task from the main session*: the process
exiting is the one wake-up signal a harness never loses. The agent that
wakes then does the judging (or forwards to a cheap judge model).

Exit codes ARE the protocol:
    0  DONE        done-re matched (the run finished)
    3  ERROR       error-re matched (traceback/OOM/NaN — context printed)
    4  STALL       progress-re output unchanged for --stall-min minutes
    5  UNREACHABLE ssh failed --max-ssh-fails consecutive times
    6  DEADLINE    --deadline-min elapsed without DONE (absolute cap)
    7  DEADMAN     (deadman mode) watched heartbeat file went stale
    2  BADARGS     misconfiguration (fail fast, fail loud)

The last ~40 lines of the log and the event line are printed to stdout
on exit, so the woken agent has context without another round-trip.

Typical launch (agent, background task):
    python vast/mayfly/mayfly.py \
        --ssh "ssh -p 54019 -o BatchMode=yes root@1.2.3.4" \
        --log /workspace/run.log \
        --done-re "LADDER_DONE|SPEEDBENCH_DONE" \
        --interval 60 --stall-min 20 --deadline-min 480 \
        --heartbeat /tmp/mayfly.hb

Shadow deadman (second background task; watches the FILE, not the agent):
    python vast/mayfly/mayfly.py --deadman /tmp/mayfly.hb \
        --interval 90 --stale-min 6 --deadline-min 500

Every poll appends one line to --heartbeat (if given): epoch, status,
last progress token. The deadman only ever reads that file's mtime —
no ssh, no process inspection, no agent-watching-agent.
"""
import argparse
import os
import re
import subprocess
import sys
import time

DEFAULT_ERROR_RE = (r"Traceback \(most recent call last\)"
                    r"|CUDA out of memory|CUDA error"
                    r"|loss(?:=|: ?)nan|\bNaN\b detected"
                    r"|torch\.OutOfMemoryError|RuntimeError")
DEFAULT_PROGRESS_RE = r"step\s*[:=]?\s*(\d+)|(\d+)/(\d+)|it/s|tok/s"


def say(msg):
    print(f"[mayfly {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def bail(code, event, tail=""):
    if tail:
        print("---- log tail ----", flush=True)
        print(tail[-4000:], flush=True)
        print("------------------", flush=True)
    say(f"EVENT:{event}")
    sys.exit(code)


def run_wait_up(a):
    """WAIT-UP MODE: the inverse watch — ssh failure is EXPECTED, first
    success is the event. For box provisioning, reboots, key
    propagation. Exits 0 (DONE) when --ready-cmd first succeeds over
    ssh, 6 (DEADLINE) at the bound. A bare `until ssh; do sleep; done`
    loop is this mode done wrong: no deadline, no heartbeat, no loud
    exit — an agent behind one is indistinguishable from a dead one
    (observed live 2026-08-14; a human had to poke)."""
    t0 = time.time()
    dl = a.deadline_min or 15          # provisioning must be bounded
    tries = 0
    while True:
        tries += 1
        try:
            r = subprocess.run(a.ssh.split() + [a.ready_cmd],
                               capture_output=True, text=True,
                               timeout=a.ssh_timeout,
                               encoding="utf-8", errors="replace")
            if r.returncode == 0:
                heartbeat(a, "TERMINAL DONE", f"up after {tries} tries")
                bail(0, f"DONE box is up ({tries} tries, "
                        f"{(time.time() - t0) / 60:.1f} min)")
        except (subprocess.TimeoutExpired, OSError):
            pass
        heartbeat(a, "WAITING", f"try {tries}")
        if (time.time() - t0) / 60 > dl:
            bail(6, f"DEADLINE box not up after {dl} min "
                    f"({tries} tries) — check provider status / "
                    "destroy and re-rent")
        time.sleep(a.interval)


def run_deadman(a):
    """Watch a heartbeat FILE. Exits 7 when it goes stale, 0 if the
    watched mayfly wrote a terminal status, 6 at deadline."""
    t0 = time.time()
    while True:
        if a.deadline_min and (time.time() - t0) / 60 > a.deadline_min:
            bail(6, "DEADLINE deadman outlived its watch window")
        try:
            age_min = (time.time() - os.path.getmtime(a.deadman)) / 60
            with open(a.deadman, "r", encoding="utf-8",
                      errors="replace") as f:
                last = f.readlines()[-1].strip() if f else ""
        except OSError:
            age_min, last = float("inf"), "(heartbeat file missing)"
        if "TERMINAL" in last:
            bail(0, f"DONE watched mayfly ended cleanly: {last}")
        if age_min > a.stale_min:
            bail(7, f"DEADMAN heartbeat stale {age_min:.1f} min "
                    f"(> {a.stale_min}); last: {last}")
        time.sleep(a.interval)


def poll_once(a):
    """One ssh round-trip: log tail (+ optional gpu line). Returns
    (ok, text)."""
    remote = (f"tail -c 100000 {a.log} 2>&1"
              + ("; nvidia-smi --query-gpu=utilization.gpu,memory.used"
                 " --format=csv,noheader 2>/dev/null" if a.gpu else ""))
    try:
        r = subprocess.run(a.ssh.split() + [remote], capture_output=True,
                           text=True, timeout=a.ssh_timeout,
                           encoding="utf-8", errors="replace")
        return r.returncode == 0, (r.stdout or "") + (r.stderr or "")
    except (subprocess.TimeoutExpired, OSError) as e:
        return False, str(e)


def heartbeat(a, status, progress):
    if not a.heartbeat:
        return
    with open(a.heartbeat, "a", encoding="utf-8") as f:
        f.write(f"{int(time.time())} {status} {progress}\n")


def run_watch(a):
    done_re = re.compile(a.done_re)
    err_re = re.compile(a.error_re)
    prog_re = re.compile(a.progress_re)
    t0 = time.time()
    last_prog, last_prog_t = None, time.time()
    ssh_fails = 0
    say(f"watching {a.log} every {a.interval}s "
        f"(stall {a.stall_min}m, deadline {a.deadline_min}m)")
    while True:
        ok, text = poll_once(a)
        if not ok:
            ssh_fails += 1
            heartbeat(a, f"SSH_FAIL_{ssh_fails}", last_prog or "-")
            say(f"ssh fail {ssh_fails}/{a.max_ssh_fails}: "
                f"{text.strip()[:200]}")
            if ssh_fails >= a.max_ssh_fails:
                heartbeat(a, "TERMINAL UNREACHABLE", last_prog or "-")
                bail(5, f"UNREACHABLE {ssh_fails} consecutive ssh "
                        "failures")
            time.sleep(a.interval)
            continue
        ssh_fails = 0
        if err_re.search(text):
            heartbeat(a, "TERMINAL ERROR", last_prog or "-")
            bail(3, f"ERROR pattern {err_re.pattern!r} matched", text)
        if done_re.search(text):
            heartbeat(a, "TERMINAL DONE", last_prog or "-")
            bail(0, "DONE", text)
        m = list(prog_re.finditer(text))
        prog = m[-1].group(0) if m else None
        if prog != last_prog:
            last_prog, last_prog_t = prog, time.time()
        stalled_min = (time.time() - last_prog_t) / 60
        if stalled_min > a.stall_min:
            heartbeat(a, "TERMINAL STALL", last_prog or "-")
            bail(4, f"STALL no progress change for {stalled_min:.1f} "
                    f"min (last: {last_prog!r})", text)
        if a.deadline_min and (time.time() - t0) / 60 > a.deadline_min:
            heartbeat(a, "TERMINAL DEADLINE", last_prog or "-")
            bail(6, f"DEADLINE {a.deadline_min} min elapsed", text)
        heartbeat(a, "OK", prog or "-")
        time.sleep(a.interval)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ssh", help='full ssh prefix, e.g. '
                   '"ssh -p 54019 -o BatchMode=yes root@1.2.3.4"')
    p.add_argument("--log", help="remote log path to tail")
    p.add_argument("--done-re", default=r"\bDONE\b")
    p.add_argument("--error-re", default=DEFAULT_ERROR_RE)
    p.add_argument("--progress-re", default=DEFAULT_PROGRESS_RE,
                   help="stall detection: last match must CHANGE")
    p.add_argument("--interval", type=float, default=60)
    p.add_argument("--stall-min", type=float, default=20)
    p.add_argument("--deadline-min", type=float, default=0,
                   help="absolute cap in minutes (0 = none)")
    p.add_argument("--max-ssh-fails", type=int, default=5)
    p.add_argument("--ssh-timeout", type=float, default=30)
    p.add_argument("--gpu", action="store_true",
                   help="also fetch a one-line nvidia-smi each poll")
    p.add_argument("--heartbeat", help="local file to append liveness to")
    p.add_argument("--deadman", metavar="HB_FILE",
                   help="DEADMAN MODE: watch this heartbeat file "
                        "instead of a remote log")
    p.add_argument("--stale-min", type=float, default=6,
                   help="deadman: minutes without heartbeat = dead")
    p.add_argument("--wait-up", action="store_true",
                   help="WAIT-UP MODE: exit 0 when ssh first succeeds "
                        "(box provisioning); deadline-bounded (15 min "
                        "default)")
    p.add_argument("--ready-cmd", default="true",
                   help="wait-up: remote command that must succeed to "
                        "count as up")
    a = p.parse_args()
    if a.deadman:
        run_deadman(a)
    if a.wait_up:
        if not a.ssh:
            say("wait-up needs --ssh")
            sys.exit(2)
        run_wait_up(a)
    if not a.ssh or not a.log:
        say("need --ssh and --log (or --deadman / --wait-up)")
        sys.exit(2)
    run_watch(a)


if __name__ == "__main__":
    main()
