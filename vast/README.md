# Vast.ai training automation — MultiCore2

Rent a GPU, health-check it, train, push results to wandb, destroy the
instance — driven from `vast/launch.py` on the local machine.

Ported 2026-07-28 from the SmallCore/MultiCore setup. `BLUEPRINT.md` is the
portable spec and `OFFER_JUDGEMENT.md` the machine-selection ledger; both
carry operational history worth reading before renting anything.

## What is different about this project

- **GPU-bound again.** MultiCore2 trains a 124M-parameter GPT-2 on cached
  FineWeb-Edu tokens. Unlike SmallCore (kernel-launch bound, CPU was the
  story) this is a straight FLOP workload: rent 4090/5090-class compute and
  gate on real bf16 throughput (`thresholds_gpt2.json`, the onstart
  default). The SmallCore rule "shop for the CPU" does NOT apply here.
- **The dataset is a 7.5 GB token cache on the Drive bank**
  (`gdrive:multicore-cache/fineweb100_gpt-neox-20b_5shards_u16.bin`, 3.76B
  GPT-NeoX tokens, built by MultiCore 2026-07-28 and shared between the two
  projects — it is immutable, do not overwrite it). Boot order:
  `core/data.py` first tries `bank.try_pull` (~90 s), falls back to
  tokenising from the HF hub (~58 min of CPU). The `bank` gate test
  ranged-reads it, so a host with throttled Drive peering self-destructs at
  boot instead of stalling training.
- **Compute matching is the experiment's contract.** Baseline and variant
  runs share the data seed (identical batch stream) and stop at equal
  cumulative FLOPs (`--target-flops`), not equal steps. See
  `scripts/train_gpt2.py`.
- **`destroy --all` is scoped to this repo.** The account runs instances for
  other projects concurrently; `--all` destroys only what
  `.vast/instances.json` records. `--all-remote` is the unscoped version and
  must be asked for by name.

## One-time setup

`vast/secrets.env` (gitignored — this repo is PUBLIC) with:

```
VAST_API_KEY=...
WANDB_API_KEY=...
HF_TOKEN=...
RCLONE_DRIVE_TOKEN=...
```

## Commands

```bash
python vast/launch.py search                       # candidate offers
python vast/launch.py scan --n 3                   # bench 3 machines, suggest thresholds
python vast/launch.py launch --smoke               # tiny pipeline test (keep-alive)
python vast/launch.py launch                       # default: train_gpt2.py 124M baseline
python vast/launch.py launch --train-args "--wandb --target-tokens 2.5e9"
python vast/launch.py status                       # live instances (all projects)
python vast/launch.py logs [--id ID]
python vast/launch.py pull [--id ID]               # copy runs/ back before destroying
python vast/launch.py destroy [--id ID | --all]
```

## Lifecycle

1. `launch.py` picks an offer (median price, never the cheapest — the bottom
   of the range over-samples lemons) and creates the instance.
2. Onstart clones this repo and runs `vast/onstart.sh`: installs
   `requirements.txt`, runs the health gate (a sick machine **destroys
   itself**), then starts `vast/run_training.sh` in tmux.
3. `run_training.sh` runs the training script, uploads checkpoints to the
   wandb run, and **destroys the instance** on success. On failure it stays
   alive for inspection. `--keep-alive` disables auto-destroy.

## Monitoring

- wandb project `multicore2` — losses live; checkpoints under Artifacts as
  `multicore2-<run_id>` (periodic insurance uploads + `final`).
- `python vast/launch.py logs` — markers: `ONSTART_BEGIN`,
  `BENCHMARK_JSON {...}`, `GATE_PASSED`/`GATE_FAILED`, `TRAIN_LAUNCHED`,
  `TRAIN_EXIT`, `RUN_COMPLETE`, `SELF_DESTROY`, plus `BANK_PULL_OK` from the
  cache download.
- **`vastai logs` can be silent on a perfectly healthy instance.** On image
  tag `pytorch_cuda-13.2.1-auto` (2026-07-25) provisioning output never
  reached `vastai logs` at all. The old rule "empty logs past ~8 minutes
  means a zombie, destroy it" would have killed it. **SSH in and read
  `/workspace/onstart.log` and `/workspace/train.log` before destroying
  anything.**

      python vast/launch.py ssh --id <ID>      # prints ssh://user@host:port
      ssh -p <port> root@<host> 'tail -30 /workspace/onstart.log'
      ssh -p <port> root@<host> 'tail -20 /workspace/train.log'

  Only treat an instance as dead if SSH also shows nothing running.
- **The hedged race can destroy healthy instances** when provisioning output
  doesn't reach `vastai logs` (see above): all racers "time out" and get
  destroyed while healthy. Until the poller reads `/workspace/onstart.log`
  over SSH, launch with `--hedge 1` and verify by SSH.

## Training scripts

Any script under `scripts/` works as `--train-script`. They log to wandb
with `--wandb` and write `runs/<name>/{best,latest}.pt` plus `metrics.json`,
with `runs/LATEST` naming the current run — which is what the upload step
reads. Multi-GPU: pass `--nproc N` to launch, which runs the script under
torchrun; the data sampler partitions ONE stream across ranks, so an N-GPU
run sees the same data as a 1-GPU run at equal token counts.
