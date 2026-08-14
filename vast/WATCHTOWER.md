# WATCHTOWER — bulletproof cheap monitoring for live boxes & experiments

Spec v1, 2026-08-13. Distilled from this project's monitoring failures:
monitor agents stalling passively after decision events (3 incidents),
`vastai logs` silent on healthy instances, detached pollers that break
the notification chain, a false STALL from a quoting bug, and the tmux
server death + spot outbid that only tripwires caught.

## The core insight

"An agent that checks every 10 minutes" is the wrong primitive — models
are for JUDGMENT, not for polling. Polling is mechanical: a script does
it for zero tokens and cannot get bored, drift off task, or hallucinate.
The design separates three things that our failures kept conflating:

  MECHANISM  (scripts)  watch continuously, classify by regex, exit loudly
  JUDGMENT   (cheap model)  woken ONLY on events, decides benign vs real
  AUTHORITY  (main session)  woken ONLY on real, acts with evidence in hand

And the "watcher watching the watcher" is NOT another agent — agents
watching agents multiplies the stall problem (either can go passive; now
you have two things to babysit). The correct primitive is the **deadman
switch**: every layer emits heartbeats as a side effect of working, and
liveness is checked by things that CANNOT stall — process-exit
notifications and timestamp staleness. Nothing is ever "assumed running."

## The four layers

### L0 — on-box sentinels (free, survive local failures)
Ground truth lives ON the box, written by the workload itself:
- the run's own log (train.log / ladder.log) with step lines
- explicit terminal markers (`RUN_COMPLETE`, `LADDER_DONE`, `TRAIN_EXIT n`)
- a `setsid` watchdog that writes a synthetic exit marker if the process
  vanishes (tmux server death, OOM-kill) — v2 pattern, battle-tested
- optional reaper for verify-then-destroy endgames
L0 exists so that no matter what dies locally, the box's disk holds an
authoritative record the moment anyone can read it again.

### L1 — the dumb poller (free, local, self-terminating)
A shell loop on the operator's machine. Every N minutes (10 default; 1-2
for critical windows) it SSHes in and mirrors to a LOCAL file:
  log tail bytes + tmux/process liveness + GPU utilization
Then it classifies with REGEX ONLY into one status per poll:
  OK          expected progress markers advancing
  EVENT:<x>   terminal marker, process gone, error strings, N ssh
              failures, log frozen > threshold with idle GPUs
  CONFUSED    anything it cannot parse
Rules that make L1 bulletproof:
- **It exits on EVENT, CONFUSED, or its own deadline.** Exit is the
  signal. A harness background task's exit ALWAYS re-invokes its owner —
  that notification is the one wake-up that cannot be forgotten.
- It never repairs, never decides benign-vs-real (that's judgment).
- Error patterns must EXCLUDE known noise (vast ssh banner, torchrun
  teardown after DONE) — the false-STALL bug came from parsing stderr
  it didn't expect. When in doubt: CONFUSED, exit, let the judge look.
- The local mirror file IS the evidence bundle. Judges read it instead
  of re-polling; if the box dies, the mirror survives.

### L1b — the shadow deadman (free; watches L1 without being an agent)
A second trivial loop, different failure mode, one job: if the L1
mirror's mtime goes stale beyond 3 poll intervals OR L1's process is
gone without a status line, emit EVENT:poller-dead and exit. L1 and L1b
each independently wake the layer above; both dying silently in the
same window is the only uncovered case, and both are 15-line scripts
with no shared state. (Tonight's terminal_waiter was this, ad hoc.)

### L2 — the judge (cheap model, event-driven, stateless)
A FRESH cheap agent (Sonnet; Haiku when classification is trivial)
dispatched only when L1/L1b exits. It gets: this spec section, the
imperative playbook for the workload, and the mirror file path. It:
1. reads the mirror (and MAY do one confirming ssh read — logs on box
   are authoritative, `vastai logs` never is),
2. classifies: BENIGN (compile pause, eval pause, expected phase
   transition, known noise) → relaunch L1 with adjusted thresholds and
   report one line; REAL → escalate with a prepared evidence bundle
   (status word, last 100 relevant lines, utilization, its reasoning);
   AMBIGUOUS → escalate as REAL. Never treat ambiguity as benign.
3. never fixes, never destroys, never launches workloads.
Stateless-per-event beats one long-lived monitor agent: no context
drift, no passive stalls, each wake costs a few thousand tokens, and a
judge that crashes loses nothing (the next event spawns a fresh one).

### L3 — authority (main session / Fable)
Woken only by a judge's REAL escalation or by L2 dispatch failing.
Acts: restart runs, re-rent, run endgames, wake the user. Every action
ends by re-arming L1 — see the invariant below.

## The one invariant (the anti-stall law)

**No layer may go quiet without a successor armed.** Concretely: every
turn of every agent in this system ends in exactly one of
  (a) a terminal report delivered upward, or
  (b) a live, harness-tracked background process whose EXIT re-invokes it.
"Armed and waiting" with nothing tracked is the bug — it happened twice
tonight. Enforce it in every playbook's closing line: "Before ending
your turn, verify `TaskList`/background state shows your poller alive;
if not, relaunch it or exit with a report."

## Playbooks are imperative, or they are nothing

Condition lists ("watch for problems and report") produce passive
monitors — proven three times. Every workload ships an event→action
table: WHEN <observable> DO <exact command / exact report> THEN <next
state>. Include expected-benign events (recompiles at known steps, eval
pauses, phase prints) so the judge isn't guessing. Include the workload's
expected timeline; deadline expiry is an EVENT (escalate as suspicious),
never a silent stop.

## Cost profile

Quiet night: L0+L1+L1b only — **zero tokens**. Benign blip: one judge
wake, ~3-10k Sonnet tokens (~fractions of a cent). Real incident: one
judge wake + one authority wake with evidence pre-bundled (no expensive
re-investigation). Compare: a Fable agent self-polling every 10 min
burns full-model tokens ~144x/night to mostly observe "still fine."

## Known failure modes → which layer eats them

  box dies / network drops        L1 ssh-fail counter → EVENT
  run crashes, tmux survives      L0 marker or L1 log-freeze → EVENT
  tmux server dies                L0 setsid watchdog marker → L1 EVENT
  run hangs, GPUs idle            L1 freeze+idle rule → EVENT
  compile/eval pause (benign)     L1 EVENT → L2 says BENIGN, re-arms
  L1 script bug / bad parse       CONFUSED exit → judge looks
  L1 dies silently                L1b staleness deadman → EVENT
  judge stalls or crashes         its exit notifies L3; ambiguity→REAL
  spot outbid / price spike       L1 ssh-fail + vast status in playbook
  everything down at once         L1/L1b deadline expiry is itself loud

## Implementation status

- Patterns proven ad hoc this project: on-box reaper/watchdog (v2 8x),
  self-terminating pollers with tripwires, shadow waiter (tonight).
- TODO (WATCHTOWER v1 hardening): templated `watchtower_poll` script
  (parameterized: ssh target, log path, ok-regex, event-regex, noise
  allowlist, interval, deadline) + `watchtower_deadman` + a judge brief
  template that embeds this spec's L2 section. Keep them in vast/.
