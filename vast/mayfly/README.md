# mayfly — drop-in remote-run watcher for agentic projects

One file, stdlib-only, cross-platform (Windows/Linux, any Python 3.8+).
It polls a remote log over ssh at a fixed interval and **exits loudly**
the moment something decision-worthy happens. That's the entire trick:
in every agent harness, a *background process exiting* is the one wake
signal that can't be forgotten, rate-limited, or silently dropped —
unlike "an agent that promised to check back."

Design provenance: `../WATCHTOWER.md` (mechanism/judgment/authority
layers, the anti-stall invariant, the failure-mode table). mayfly is the
mechanism layer; judgment stays with whatever model wakes up.

## Quick start (agent harness)

Launch as a **main-session background task** (not from a subagent —
subagent→background-task wake chains have been observed to silently
drop; main-session task exits always re-invoke):

```
python vast/mayfly/mayfly.py \
  --ssh "ssh -p PORT -o BatchMode=yes -o StrictHostKeyChecking=no root@IP" \
  --log /workspace/run.log \
  --done-re "RUN_DONE|SPEEDBENCH_DONE" \
  --interval 60 --stall-min 20 --deadline-min 480 \
  --heartbeat /path/to/local/mayfly.hb
```

Optionally arm the **shadow deadman** as a second background task. It
watches the heartbeat *file* (mtime + last line), never the process —
no agent-watching-agent, no ssh of its own:

```
python vast/mayfly/mayfly.py --deadman /path/to/local/mayfly.hb \
  --interval 90 --stale-min 6 --deadline-min 500
```

## Exit codes are the protocol

| code | event | meaning | typical playbook |
|---|---|---|---|
| 0 | DONE | done-re matched | pull results, tear down box |
| 3 | ERROR | error-re matched (tail printed) | read tail; judge: fix vs abort |
| 4 | STALL | progress-re unchanged `--stall-min` min | check GPU util; hung → restart run |
| 5 | UNREACHABLE | N consecutive ssh failures | check instance state via provider CLI |
| 6 | DEADLINE | absolute time cap | escalate to human — never rearm blindly |
| 7 | DEADMAN | heartbeat stale | the watcher itself died; relaunch both |
| 2 | BADARGS | misconfigured | fix the invocation |

On every exit the log tail (~4 KB) plus a final `EVENT:<NAME>` line is
on stdout, so the woken agent has context with zero extra round-trips.

## Rules for the agent that launches it

1. **Write the playbook before sleeping.** For each exit code, one
   imperative action (table above is the template). Judgment calls go
   to the cheapest capable model; the expensive model wakes only for
   terminal events or repeated watcher death.
2. **Never end a turn "waiting"** without a live harness-tracked
   mayfly. An armed intention is not a mechanism (anti-stall invariant).
3. **Rearm after every wake** that isn't terminal: relaunch mayfly (and
   deadman) before doing anything else. A handled event with no new
   watcher is a silent hole.
4. **Tune `--progress-re` to the actual log.** Default matches
   `step N` / `N/M` / `it/s`; if the run logs something else, stall
   detection is only as good as this regex. Test it: the LAST match in
   the tail must change while the run is healthy.
5. `--error-re` should over-trigger rather than under-trigger; a false
   ERROR wake costs one cheap judgment, a missed one costs a night.
6. ssh must be non-interactive: key-based auth, `BatchMode=yes`. If the
   provider gives a fresh host key each rental, add
   `-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null`
   (already fine on vast).

## Reusing outside this repo

Copy the single file. There is no config file, no state beyond the
heartbeat, no imports beyond the stdlib — the point is that it works
identically from a Windows laptop, a Linux CI runner, or inside any
agent sandbox that has `ssh` on PATH.
