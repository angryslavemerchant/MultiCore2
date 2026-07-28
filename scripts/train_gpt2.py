"""Train the GPT-2 baseline (or a registered variant) on cached FineWeb-Edu.

The comparison contract: baseline and variant runs share the token stream
(same cache file, same seed => identical batch sequence) and are stopped at
the same cumulative FLOPs, not the same step count. `--arch` plus the
registry seams in core/model.py are the only place a variant differs.

    python scripts/train_gpt2.py --smoke              # tiny local pipe test
    python scripts/train_gpt2.py --wandb              # the 124M baseline
    torchrun --standalone --nproc_per_node=N scripts/train_gpt2.py --wandb

Stopping rule priority: --iters > --target-flops > --target-tokens
(default 2.5e9 tokens, Chinchilla-ish for 124M).

Checkpoints: runs/<name>/{latest,best}.pt + metrics.json, with runs/LATEST
naming the run dir -- the layout vast/run_training.sh and
vast/upload_results.py read. Resume with --resume: the data stream
fast-forwards past consumed batches, so a resumed run continues the stream
rather than replaying it.
"""
import argparse
import json
import math
import os
import platform
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.data import open_data, VOCAB_SIZE, DEFAULT_SHARDS      # noqa: E402
from core.model import GPT, GPTConfig, config_dict               # noqa: E402


# Named model scales. A variant usually keeps a scale and changes only the
# seam keys (--block/--attn/--mlp).
SCALES = {
    "124m":  dict(n_layer=12, n_head=12, n_embd=768),
    "small": dict(n_layer=8, n_head=8, n_embd=512),
    "smoke": dict(n_layer=4, n_head=4, n_embd=256, block_size=256),
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default=None,
                    help="default: <scale>-<block>-<attn>-<mlp>")
    ap.add_argument("--scale", default="124m", choices=sorted(SCALES))
    ap.add_argument("--block", default="gpt2")
    ap.add_argument("--attn", default="causal")
    ap.add_argument("--mlp", default="gelu")
    ap.add_argument("--attn-pattern", default="",
                    help="per-layer attention, one char per layer: F=full, "
                         "S=sliding window, G=admission-gated. The 2:1 "
                         "sandwich is FGGGGFFGGGGF (gated) / FSSSSFFSSSSF "
                         "(SWA control). Empty = all layers --attn")
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--n-gates", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.0)
    # stopping rule (first non-None wins, in this order)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--target-flops", type=float, default=None,
                    help="stop when cumulative fwd+bwd FLOPs reach this -- "
                         "the compute-matching knob for variant runs")
    ap.add_argument("--target-tokens", type=float, default=2.5e9)
    # optimisation
    ap.add_argument("--tokens-per-step", type=int, default=524288,
                    help="global tokens per optimizer step (0.5M, GPT-3 "
                         "small); grad accumulation is derived from this")
    ap.add_argument("--micro-bs", type=int, default=16,
                    help="per-GPU sequences per micro-batch")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--warmup-iters", type=int, default=700)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    ap.add_argument("--grad-clip", type=float, default=1.0)
    # data
    ap.add_argument("--shards", type=int, default=DEFAULT_SHARDS)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--seed", type=int, default=1337,
                    help="the DATA seed too: two runs comparing archs must "
                         "share it to see the same batches")
    # plumbing
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=25)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--artifact-every", type=int, default=1000,
                    help="steps between latest.pt wandb uploads (checkpoint "
                         "insurance against the instance dying); 0 disables")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny model, ~40 iters, no compile: end-to-end "
                         "pipeline test")
    args = ap.parse_args()

    if args.smoke:
        args.scale = "smoke"
        args.seq_len = 256
        args.iters = args.iters or 40
        args.tokens_per_step = 8 * 256 * 2
        args.micro_bs = 8
        args.warmup_iters = 10
        args.eval_every = 20
        args.eval_iters = 5
        args.artifact_every = 0
        args.no_compile = True
    if args.run_name is None:
        if args.attn_pattern:
            args.run_name = (f"{args.scale}-{args.attn_pattern}-w{args.window}"
                             + (f"-g{args.n_gates}"
                                if "G" in args.attn_pattern else ""))
        else:
            args.run_name = f"{args.scale}-{args.block}-{args.attn}-{args.mlp}"
    return args


def main():
    args = parse_args()

    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        torch.distributed.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        rank, world = 0, 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
    master = rank == 0
    device_type = "cuda" if "cuda" in device else "cpu"

    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    amp = (torch.autocast(device_type=device_type, dtype=torch.bfloat16)
           if device_type == "cuda" else nullcontext())

    # ------------------------------------------------------------ model
    scale = dict(SCALES[args.scale])
    block_size = scale.pop("block_size", 1024)
    assert args.seq_len <= block_size
    cfg = GPTConfig(block_size=block_size, vocab_size=VOCAB_SIZE,
                    dropout=args.dropout, block=args.block, attn=args.attn,
                    mlp=args.mlp, attn_pattern=args.attn_pattern,
                    window=args.window, n_gates=args.n_gates, **scale)
    model = GPT(cfg).to(device)
    n_params = model.num_params()
    fpt = model.flops_per_token(args.seq_len)

    # ------------------------------------------------------------ schedule
    B, T = args.micro_bs, args.seq_len
    grad_accum = max(1, args.tokens_per_step // (B * T * world))
    tokens_per_step = B * T * world * grad_accum
    if args.iters is not None:
        iters = args.iters
    elif args.target_flops is not None:
        iters = math.ceil(args.target_flops / (fpt * tokens_per_step))
    else:
        iters = math.ceil(args.target_tokens / tokens_per_step)

    def lr_at(step):
        if step < args.warmup_iters:
            return args.lr * (step + 1) / args.warmup_iters
        t = (step - args.warmup_iters) / max(1, iters - args.warmup_iters)
        coef = 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))
        floor = args.lr * args.min_lr_ratio
        return floor + coef * (args.lr - floor)

    if master:
        print(f"[train] {args.run_name}: {n_params / 1e6:.1f}M params, "
              f"{fpt / 1e9:.2f} GFLOPs/token, {iters} iters x "
              f"{tokens_per_step} tokens/step (accum {grad_accum} x {world} "
              f"rank(s)) = {iters * tokens_per_step / 1e9:.2f}B tokens, "
              f"{iters * tokens_per_step * fpt:.3e} total FLOPs", flush=True)

    # ------------------------------------------------------------ data
    data = open_data(args.shards, args.data_dir, rank, world)

    # ------------------------------------------------------------ resume
    run_dir = os.path.join("runs", args.run_name)
    ckpt_path = os.path.join(run_dir, "latest.pt")
    start_step, best_val = 0, float("inf")
    ckpt = None
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        assert ckpt["config"] == config_dict(cfg), (
            "checkpoint config != requested config; refusing to resume "
            f"{ckpt['config']} as {config_dict(cfg)}")
        model.load_state_dict(ckpt["model"])
        start_step = ckpt["step"]
        best_val = ckpt.get("best_val", float("inf"))
        if master:
            print(f"[train] resuming {args.run_name} at step {start_step}",
                  flush=True)

    compile_ok = (device_type == "cuda" and not args.no_compile
                  and platform.system() != "Windows")
    if compile_ok:
        model = torch.compile(model)
    raw_model = model._orig_mod if compile_ok else model

    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank])

    optimizer = raw_model.configure_optimizers(
        args.weight_decay, args.lr, tuple(args.betas), device_type)
    if ckpt is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    del ckpt

    # ------------------------------------------------------------ wandb
    use_wandb = args.wandb and master
    if use_wandb:
        import wandb
        wandb.init(project=os.environ.get("WANDB_PROJECT", "multicore2"),
                   name=args.run_name,
                   id=os.environ.get("WANDB_RUN_ID"),
                   resume="allow",
                   config={**vars(args), **config_dict(cfg),
                           "n_params": n_params, "flops_per_token": fpt,
                           "world_size": world, "iters": iters,
                           "tokens_per_step": tokens_per_step})

    # ------------------------------------------------------------ stream
    stream = data.batches(B, T, device, seed=args.seed, split="train",
                          skip=start_step * grad_accum * world + rank,
                          stride=world)

    def val_loss():
        """Mean eval loss, plus per-gated-layer router stats (collected on
        the last eval batch only — one batch of occupancy is representative
        and keeps the stats hook out of the hot path)."""
        from core.gated_swa import GatedSWAttention
        gated = [(i, blk.attn) for i, blk in enumerate(raw_model.transformer.h)
                 if isinstance(blk.attn, GatedSWAttention)]
        raw_model.eval()
        losses = []
        ev = data.batches(B, T, device, seed=0, split="eval")
        with torch.no_grad():
            for it in range(args.eval_iters):
                if it == args.eval_iters - 1:
                    for _, m in gated:
                        m.collect_stats = True
                w = next(ev)
                with amp:
                    _, loss = raw_model(w[:, :-1], w[:, 1:])
                losses.append(loss.item())
        stats = {}
        for i, m in gated:
            m.collect_stats = False
            s = m.stats
            stats[f"gates/L{i}/max_frac"] = float(s["gate_frac"].max())
            stats[f"gates/L{i}/entropy"] = s["router_entropy"]
            stats[f"gates/L{i}/mean_lifetime"] = s["mean_lifetime"]
            stats[f"gates/L{i}/frac_evicted"] = s["frac_evicted"]
        raw_model.train()
        return float(np.mean(losses)), stats

    def save(step, best):
        os.makedirs(run_dir, exist_ok=True)
        torch.save({"model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step, "best_val": best,
                    "config": config_dict(cfg), "args": vars(args),
                    "tokens": step * tokens_per_step,
                    "flops": step * tokens_per_step * fpt},
                   ckpt_path)

    def upload_ckpt(alias):
        import wandb
        # type must match vast/upload_results.py, which appends the final
        # version to the SAME artifact name — wandb forbids reusing a name
        # with a different type (found by the 2026-07-28 cloud smoke).
        art = wandb.Artifact(
            f"{wandb.run.project}-{wandb.run.id}", type="model")
        art.add_file(ckpt_path)
        wandb.log_artifact(art, aliases=[alias])

    if master:
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join("runs", "LATEST"), "w") as f:
            f.write(run_dir.replace(os.sep, "/"))

    # ------------------------------------------------------------ loop
    model.train()
    t_last = time.time()
    for step in range(start_step, iters):
        lr = lr_at(step)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for micro in range(grad_accum):
            w = next(stream)
            if ddp:
                model.require_backward_grad_sync = (micro == grad_accum - 1)
            with amp:
                _, loss = model(w[:, :-1], w[:, 1:])
                loss = loss / grad_accum
            loss_acc += loss.item()
            loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        done = step + 1
        if master and (done % args.log_every == 0 or done == iters):
            if device_type == "cuda":
                torch.cuda.synchronize()
            now = time.time()
            dt = (now - t_last) / args.log_every
            t_last = now
            toks = done * tokens_per_step
            print(f"[train] step {done}/{iters} loss {loss_acc:.4f} "
                  f"lr {lr:.2e} {dt * 1e3:.0f} ms/iter "
                  f"{tokens_per_step / dt / 1e3:.0f}k tok/s "
                  f"{toks / 1e9:.3f}B tokens", flush=True)
            if use_wandb:
                import wandb
                wandb.log({"train/loss": loss_acc, "lr": lr,
                           "tokens": toks, "flops": toks * fpt,
                           "perf/ms_per_iter": dt * 1e3,
                           "perf/tok_per_s": tokens_per_step / dt},
                          step=done)

        if master and (done % args.eval_every == 0 or done == iters):
            vl, gate_stats = val_loss()
            best_val = min(best_val, vl)
            print(f"[train] step {done} VAL {vl:.4f} (best {best_val:.4f})",
                  flush=True)
            save(done, best_val)
            if vl <= best_val:
                import shutil
                shutil.copyfile(ckpt_path, os.path.join(run_dir, "best.pt"))
            with open(os.path.join(run_dir, "metrics.json"), "w") as f:
                json.dump({"step": done, "val_loss": vl,
                           "best_val": best_val, "train_loss": loss_acc,
                           "tokens": done * tokens_per_step,
                           "flops": done * tokens_per_step * fpt,
                           "n_params": n_params,
                           "config": config_dict(cfg)}, f, indent=2)
            if use_wandb:
                import wandb
                wandb.log({"val/loss": vl, "val/best": best_val,
                           **gate_stats}, step=done)
            t_last = time.time()          # don't bill eval time to the iter

        if (use_wandb and args.artifact_every
                and done % args.artifact_every == 0 and done < iters):
            upload_ckpt("latest")
            t_last = time.time()

    if master:
        print(f"[train] done: {iters} steps, "
              f"{iters * tokens_per_step / 1e9:.3f}B tokens, "
              f"{iters * tokens_per_step * fpt:.3e} FLOPs, "
              f"best val {best_val:.4f}", flush=True)
        if use_wandb:
            upload_ckpt("final")
            import wandb
            wandb.finish()
    if ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
