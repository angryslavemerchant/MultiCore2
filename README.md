# MultiCore2

One-shot architecture comparison at GPT-2 scale: a faithful GPT-2 small
(124M) baseline, then a new architecture trained compute-matched against it
on the identical data stream.

## Layout

| Path | What |
|---|---|
| `core/model.py` | GPT-2 with registry seams (`ATTENTIONS`/`MLPS`/`BLOCKS`) — a variant registers its pieces and is selected by CLI flag |
| `core/data.py` | Deterministic FineWeb-Edu token cache (GPT-NeoX ids, flat uint16) + seeded window sampler; same seed => identical batches for every arch |
| `bank.py` | Google Drive "bank" via rclone: the 3.76B-token cache is pulled in ~90 s instead of ~58 min of tokenising |
| `scripts/train_gpt2.py` | Training: bf16, flash SDPA, AdamW, cosine, DDP-capable, resumable, wandb + checkpoint insurance |
| `vast/` | Rent-a-GPU automation (see `vast/README.md`) |

## The comparison contract

1. Both arms read the same cache file with the same `--seed`: identical
   token stream, batch for batch.
2. Both arms stop at equal cumulative fwd+bwd FLOPs — run the baseline,
   read `flops` from its `metrics.json`, pass it to the variant as
   `--target-flops`. A variant with a different per-token cost must
   override `GPT.flops_per_token`.
3. The judge is validation loss on the held-out tail (last 5% of the cache,
   never trained on) at matched compute.

## Quick start

```bash
# local pipeline test (tiny model, ~1 min)
python scripts/train_gpt2.py --smoke --data-dir data/smoke

# the 124M baseline on a rented GPU (see vast/README.md)
python vast/launch.py launch --smoke        # first: pipeline test in the cloud
python vast/launch.py launch                # then: the real run
```
