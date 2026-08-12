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
    # four-stroke MMMF run: 12 layers at d512 (capacity deliberately moved
    # from the token stream into the machine population; K*d_machine =
    # 2048 of structured state on a 512-wide carrier)
    "d512":  dict(n_layer=12, n_head=8, n_embd=512),
    "small": dict(n_layer=8, n_head=8, n_embd=512),
    "smoke": dict(n_layer=4, n_head=4, n_embd=256, block_size=256),
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default=None,
                    help="default: <scale>-<block>-<attn>-<mlp>")
    ap.add_argument("--scale", default="124m", choices=sorted(SCALES))
    ap.add_argument("--arch", default="gpt2", choices=("gpt2", "hier"),
                    help="hier = hierarchical predictive-plan model with "
                         "block-level product-key memory (core/hier.py); "
                         "ignores the attention-pattern/hourglass flags")
    ap.add_argument("--mem-slots", type=int, default=16384,
                    help="hier: product-key memory slots (must be square)")
    ap.add_argument("--token-mode", default="block",
                    choices=("block", "swa"),
                    help="hier token stacks: independent 128-windows "
                         "(block) or true w128 sliding window over the "
                         "full sequence (swa, NSA-style)")
    ap.add_argument("--levels", type=int, default=2, choices=(2, 3),
                    help="hier: 3 switches on the v3 bundle (3rd level, "
                         "learned pooling, dynamic gates, latent losses)")
    ap.add_argument("--btok", type=int, default=128,
                    help="hier: block length in tokens (v3 run: 32)")
    ap.add_argument("--block", default="gpt2")
    ap.add_argument("--attn", default="causal")
    ap.add_argument("--mlp", default="gelu")
    ap.add_argument("--attn-pattern", default="",
                    help="per-layer attention, one char per layer: F=full, "
                         "S=sliding window, G=admission-gated, C=COW "
                         "archive, M=four-stroke machine block. The 2:1 "
                         "sandwich is FGGGGFFGGGGF (gated) / FSSSSFFSSSSF "
                         "(SWA control) / FCCCCFFCCCCF (COW); the machine "
                         "run is MMMFMMMFMMMF. Empty = all layers --attn")
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--windows", default="",
                    help="per-layer window schedule (pyramid), comma-"
                         "separated, one entry per layer; F layers' entries "
                         "ignored. E.g. 32,64,128,256,512,F,F,F,F,2048,"
                         "1024,512. Empty = uniform --window")
    ap.add_argument("--n-gates", type=int, default=8)
    ap.add_argument("--recent-band", type=int, default=0,
                    help="G layers: guaranteed-visible recent tokens out of "
                         "--window; gates share the rest. Phase-1's pure "
                         "gates lost 0.04 nats to recency starvation")
    ap.add_argument("--pos", default="learned", choices=("learned", "rope"),
                    help="rope has no position params and is the "
                         "long-context default")
    # hourglass (slice-carry bottleneck; see core/model.py SliceBlock and
    # scripts/hourglass_match.py for the d_base that parameter-matches dense)
    ap.add_argument("--hg-frac", type=float, default=0.0,
                    help="waist width as a fraction of d_base; 0 disables")
    ap.add_argument("--hg-bneck", type=int, default=8,
                    help="which layer is narrowest")
    ap.add_argument("--hg-mid", type=int, default=0,
                    help="extra flat layers at the waist (total depth = "
                         "n_layer + this)")
    ap.add_argument("--hg-dbase", type=int, default=0,
                    help="residual stream width, from hourglass_match.py; "
                         "overrides the scale's n_embd. Required with "
                         "--hg-frac")
    ap.add_argument("--hg-round", type=int, default=24,
                    help="width rounding; 96 keeps head_dim a multiple of "
                         "8 (franken/flex), 24 matches the phase-3 arms")
    ap.add_argument("--lb-coef", type=float, default=0.01,
                    help="router load-balance aux weight (G/C layers; "
                         "the router measurably collapses at init without "
                         "it). 0 disables")
    # COW archive (C layers): recent_band raw + n_gates*cow_chains versions
    ap.add_argument("--cow-chains", type=int, default=32,
                    help="C layers: K live chains per gate; budget assert "
                         "n_gates*K == window - recent_band")
    ap.add_argument("--cow-theta", type=float, default=0.7,
                    help="C layers: vigilance; best cosine >= theta merges "
                         "into the chain, below births a new one")
    ap.add_argument("--chunk-btok", type=int, default=256,
                    help="chunk layers (pattern K/B): write cadence")
    ap.add_argument("--chunk-k", type=int, default=16,
                    help="chunk layers: chunks minted per boundary")
    ap.add_argument("--chunk-topk", type=int, default=0,
                    help="chunk layers: hard top-k membership per query "
                         "(0 = soft mixture over the whole prefix)")
    ap.add_argument("--chunk-fetch-n", type=int, default=4,
                    help="N layers (chunk v0.2): raw-fetch the pointer "
                         "sets of each query's top-n chunks (0 disables)")
    ap.add_argument("--nsa-block", type=int, default=32,
                    help="NSA summary/selection block size (--attn nsa)")
    ap.add_argument("--nsa-topk", type=int, default=12,
                    help="NSA blocks fetched by the slc branch")
    ap.add_argument("--nsa-nreg", type=int, default=1024,
                    help="learned register vectors per NSA layer")
    ap.add_argument("--side-topk", type=int, default=16,
                    help="T layers (top-k side-stack): per-head k slices "
                         "fed to the branch in the MLP slot")
    ap.add_argument("--loops", default="",
                    help="uberloop: per-layer weight-tied loop counts, "
                         "comma-separated, one per layer (e.g. "
                         "1,1,1,2,2,4,4,4,4,2,1,1). Empty = no looping")
    ap.add_argument("--cow-chunk", type=int, default=128,
                    help="C layers: chain-scan chunk (heads frozen within)")
    # Four-stroke machine blocks (M layers, core/fourstroke.py)
    ap.add_argument("--fs-n-machines", type=int, default=16,
                    help="M layers: machines per block (K)")
    ap.add_argument("--fs-d-machine", type=int, default=256,
                    help="M layers: machine state width")
    ap.add_argument("--fs-n-head-m", type=int, default=4,
                    help="M layers: heads inside intake + conference")
    ap.add_argument("--fs-mlp-mult", type=int, default=4,
                    help="M layers: private per-machine MLP expansion")
    ap.add_argument("--fs-backend", default="attn",
                    choices=("attn", "swa"),
                    help="M layers: state backend over stream + private "
                         "channel (swa = banded intake, the launch config)")
    ap.add_argument("--fs-window", type=int, default=512,
                    help="M layers: swa backend's intake band")
    ap.add_argument("--fs-no-rope", action="store_true",
                    help="M layers: disable RoPE on intake q/k")
    ap.add_argument("--fs-topk", type=int, default=0,
                    help="four-stroke: machines refreshed/writing per token "
                         "(0 = dense population)")
    ap.add_argument("--fs-loop-rounds", type=int, default=1,
                    help="four-stroke: strokes 1-4 rounds per block (tied)")
    ap.add_argument("--fs-loop-topk", type=int, default=0,
                    help="four-stroke: machines updating per loop round")
    ap.add_argument("--fs-conf-sink", action="store_true",
                    help="four-stroke: learned null logit in conference")
    ap.add_argument("--fs-tkv-heads", type=int, default=0,
                    help="four-stroke: shared token-KV bank heads "
                         "(0 = private per machine)")
    ap.add_argument("--fs-share-pub", action="store_true",
                    help="four-stroke: share publish/conference projections")
    ap.add_argument("--fs-mlp-depth", type=int, default=1,
                    help="four-stroke: hidden layers in the private MLP")
    ap.add_argument("--fs-sparse-state", action="store_true",
                    help="four-stroke: only routed pairs absorb the round")
    ap.add_argument("--fs-addr-mix", type=float, default=1.0,
                    help="M layers: init of the anchor-vs-state key "
                         "mixing scalar")
    # Frankenstein stack (all default off; see core/model.py)
    ap.add_argument("--norm", default="ln", choices=("ln", "rms"))
    ap.add_argument("--qk-norm", action="store_true")
    ap.add_argument("--diff-attn", action="store_true")
    ap.add_argument("--canon", action="store_true")
    ap.add_argument("--canon-full", action="store_true",
                    help="full A/B/C/D canon: adds convs on q,k,v (B) and "
                         "the MLP hidden (D); use WITH --canon for A/C")
    ap.add_argument("--softcap", type=float, default=0.0)
    ap.add_argument("--untied", action="store_true")
    ap.add_argument("--zero-init", action="store_true",
                    help="zero-init residual projections (speedrun) instead "
                         "of GPT-2's 1/sqrt(2L) scaling")
    ap.add_argument("--no-bias", action="store_true",
                    help="drop every Linear/LayerNorm bias")
    ap.add_argument("--opt", default="adamw", choices=("adamw", "muon"))
    ap.add_argument("--muon-lr", type=float, default=0.02,
                    help="Muon base lr for hidden matrices; --lr still "
                         "drives the AdamW side and the schedule shape")
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
    if args.hg_frac:
        assert args.hg_dbase, "--hg-dbase required with --hg-frac"
    if args.run_name is None:
        if args.arch == "hier":
            nb = args.seq_len // args.btok
            args.run_name = (f"{args.scale}-hier-b{args.btok}x{nb}"
                             f"-L{args.levels}"
                             f"-mem{args.mem_slots // 1024}k"
                             + ("-swa" if args.token_mode == "swa"
                                else ""))
        elif args.attn == "nsa":
            args.run_name = (f"{args.scale}-nsa-b{args.nsa_block}"
                             f"k{args.nsa_topk}-reg{args.nsa_nreg}"
                             f"-w{args.window}"
                             + (f"-hg{args.hg_frac}d{args.hg_dbase}"
                                if args.hg_frac else ""))
        elif args.hg_frac:
            args.run_name = (f"{args.scale}-hg-f{args.hg_frac}"
                             f"-b{args.hg_bneck}"
                             + (f"-m{args.hg_mid}" if args.hg_mid else "")
                             + f"-d{args.hg_dbase}")
        elif args.attn_pattern:
            args.run_name = (f"{args.scale}-{args.attn_pattern}-w{args.window}"
                             + (f"-g{args.n_gates}"
                                if "G" in args.attn_pattern else "")
                             + (f"-g{args.n_gates}k{args.cow_chains}"
                                f"-th{args.cow_theta}"
                                if "C" in args.attn_pattern else "")
                             + (f"-r{args.recent_band}"
                                if args.recent_band else "")
                             + (f"-ck{args.chunk_k}b{args.chunk_btok}"
                                if ("K" in args.attn_pattern
                                    or "B" in args.attn_pattern) else "")
                             + (f"-nk{args.chunk_k}b{args.chunk_btok}"
                                f"t{args.chunk_topk}f{args.chunk_fetch_n}"
                                if "N" in args.attn_pattern else "")
                             + (f"-st{args.side_topk}"
                                if ("T" in args.attn_pattern
                                    or "R" in args.attn_pattern) else ""))
        else:
            args.run_name = f"{args.scale}-{args.block}-{args.attn}-{args.mlp}"
        if args.loops:
            args.run_name += "-L" + args.loops.replace(",", "")
        if args.pos == "rope":
            args.run_name += "-rope"
        if args.seq_len != 1024:
            args.run_name += f"-t{args.seq_len}"
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
    block_size = max(scale.pop("block_size", 1024), args.seq_len)
    if args.hg_frac:
        scale["n_embd"] = args.hg_dbase
    cfg = GPTConfig(block_size=block_size, vocab_size=VOCAB_SIZE,
                    dropout=args.dropout, block=args.block, attn=args.attn,
                    mlp=args.mlp, attn_pattern=args.attn_pattern,
                    window=args.window, windows=args.windows,
                    n_gates=args.n_gates,
                    recent_band=args.recent_band, pos=args.pos,
                    lb_coef=(args.lb_coef
                             if ("G" in args.attn_pattern
                                 or "C" in args.attn_pattern) else 0.0),
                    hg_frac=args.hg_frac, hg_bneck=args.hg_bneck,
                    hg_mid=args.hg_mid, hg_round=args.hg_round,
                    cow_chains=args.cow_chains,
                    cow_theta=args.cow_theta, cow_chunk=args.cow_chunk,
                    chunk_btok=args.chunk_btok, chunk_k=args.chunk_k,
                    chunk_topk=args.chunk_topk,
                    chunk_fetch_n=args.chunk_fetch_n,
                    side_topk=args.side_topk, loops=args.loops,
                    nsa_block=args.nsa_block, nsa_topk=args.nsa_topk,
                    nsa_nreg=args.nsa_nreg,
                    fs_n_machines=args.fs_n_machines,
                    fs_d_machine=args.fs_d_machine,
                    fs_n_head_m=args.fs_n_head_m,
                    fs_mlp_mult=args.fs_mlp_mult,
                    fs_backend=args.fs_backend,
                    fs_window=args.fs_window,
                    fs_rope=not args.fs_no_rope,
                    fs_addr_mix=args.fs_addr_mix,
                    fs_topk=args.fs_topk,
                    fs_loop_rounds=args.fs_loop_rounds,
                    fs_loop_topk=args.fs_loop_topk,
                    fs_conf_sink=args.fs_conf_sink,
                    fs_tkv_heads=args.fs_tkv_heads,
                    fs_share_pub=args.fs_share_pub,
                    fs_mlp_depth=args.fs_mlp_depth,
                    fs_sparse_state=args.fs_sparse_state,
                    norm=args.norm, qk_norm=args.qk_norm,
                    diff_attn=args.diff_attn, canon=args.canon,
                    canon_full=args.canon_full,
                    softcap=args.softcap, untied=args.untied,
                    zero_init=args.zero_init, bias=not args.no_bias,
                    **scale)
    if args.arch == "hier":
        from core.hier import HierGPT, HierConfig, hier_config_dict
        assert args.seq_len == 4096, "hier run-one is specced at T=4096"
        cfg = HierConfig(block_size=args.seq_len, vocab_size=VOCAB_SIZE,
                         mem_slots=args.mem_slots, softcap=args.softcap,
                         dropout=args.dropout,
                         token_mode=args.token_mode,
                         levels=args.levels, btok=args.btok)
        model = HierGPT(cfg).to(device)
        cfg_to_dict = hier_config_dict
    else:
        model = GPT(cfg).to(device)
        cfg_to_dict = config_dict
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
        assert ckpt["config"] == cfg_to_dict(cfg), (
            "checkpoint config != requested config; refusing to resume "
            f"{ckpt['config']} as {cfg_to_dict(cfg)}")
        model.load_state_dict(ckpt["model"])
        start_step = ckpt["step"]
        best_val = ckpt.get("best_val", float("inf"))
        if master:
            print(f"[train] resuming {args.run_name} at step {start_step}",
                  flush=True)

    compile_ok = (device_type == "cuda" and not args.no_compile
                  and platform.system() != "Windows")
    if args.diff_attn:
        # diff's 2*hd value heads need explicit flex tiles on consumer
        # GPUs (see core/gated_swa._flex_opts) — opt in here so plain
        # runs keep the autotuner (and DDPOptimizer) untouched
        os.environ.setdefault("FLEX_BLOCK_M", "64")
        os.environ.setdefault("FLEX_BLOCK_N", "32")
        os.environ.setdefault("FLEX_NUM_STAGES", "2")
    if compile_ok:
        from core import gated_swa as _gsw
        # SliceBlock.forward specializes per hourglass width: 12 distinct
        # widths blow the default limit (8) and later layers fall back to
        # EAGER — harmless when attention is a flash kernel either way,
        # fatal for chunk layers whose flex_attention then runs the slow
        # fallback implementation (6.3x, bench box 2026-08-07).
        for attr in ("recompile_limit", "cache_size_limit"):
            if hasattr(torch._dynamo.config, attr):
                setattr(torch._dynamo.config, attr, 64)
        # flash-attn covering every windowed layer removes flex from the
        # training graph entirely (S layers -> flash, F -> SDPA causal),
        # so DDPOptimizer's graph splitting is safe again and the
        # backward/allreduce overlap comes back.
        # K/B (chunk) layers keep flex_attention in the graph — flash
        # never covers them, and DDPOptimizer chokes on flex ("'int'
        # object has no attribute 'meta'", reconfirmed 8x 2026-08-08)
        flash_covers = (_gsw.USE_FLASH and not args.diff_attn
                        and not any(c in args.attn_pattern for c in "GCKBN")
                        and torch.cuda.is_available()
                        and _gsw._resolve_flash() is not None)
        if ddp and (args.arch == "hier" or args.attn == "nsa"
                    or "M" in args.attn_pattern
                    or (not flash_covers
                        and (args.diff_attn
                             or any(c in args.attn_pattern
                                    for c in "GSCKBN")))):
            # DDPOptimizer's graph-splitting chokes on flex_attention's
            # higher-order op inside the franken graph shape ("'int'
            # object has no attribute 'meta'", 2026-07-30: single-process
            # trains, DDP crashes — with or without diff attention or
            # flex kernel_options). The hier arch trips the same splitter
            # failure through its PKM/aux graph (2026-08-09: single-GPU
            # compiles, 8x DDP crashes with the identical message).
            # NSA NaNs under DDP graph splits (2026-08-09: single-GPU
            # compiled trains clean 10.8->0.46 in 30 steps, 8x DDP loss
            # is nan by step 10) — silent-corruption cousin of the same
            # splitter, so whole-graph compile for nsa too.
            # Four-stroke (M) threads the machine channel BETWEEN blocks
            # — exactly the cross-bucket dependency the splitter mangles;
            # whole-graph preemptively (nsa's NaN was silent).
            # Compile the whole graph instead; costs the
            # backward/allreduce overlap (~few % at 124M).
            torch._dynamo.config.optimize_ddp = False
        model = torch.compile(model)
    raw_model = model._orig_mod if compile_ok else model

    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank])

    optimizer = raw_model.configure_optimizers(
        args.weight_decay, args.lr, tuple(args.betas), device_type,
        opt=args.opt, muon_lr=args.muon_lr)
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
                   config={**vars(args), **cfg_to_dict(cfg),
                           "n_params": n_params, "flops_per_token": fpt,
                           "world_size": world, "iters": iters,
                           "tokens_per_step": tokens_per_step})

    # ------------------------------------------------------------ stream
    stream = data.batches(B, T, device, seed=args.seed, split="train",
                          skip=start_step * grad_accum * world + rank,
                          stride=world)

    # position-bucket edges for val loss: at long context the aggregate
    # dilutes the long-range signal — the discriminating readout is loss at
    # positions far beyond the window vs inside it
    BUCKETS = [e for e in (0, 256, 512, 1024, 2048, 4096) if e < T] + [T]

    def val_loss():
        """(mean eval loss, stats dict) — stats carry per-gated-layer router
        panels plus per-position-bucket loss."""
        from torch.nn import functional as F
        from core.gated_swa import GatedSWAttention
        layers = getattr(raw_model.transformer, "h", [])
        gated = [(i, blk.attn) for i, blk in enumerate(layers)
                 if isinstance(blk.attn, GatedSWAttention)]
        raw_model.eval()
        tok_sum = torch.zeros(len(BUCKETS) - 1, dtype=torch.float64)
        tok_cnt = torch.zeros(len(BUCKETS) - 1, dtype=torch.float64)
        ev = data.batches(B, T, device, seed=0, split="eval")
        with torch.no_grad():
            for it in range(args.eval_iters):
                if it == args.eval_iters - 1:
                    for _, m in gated:
                        m.collect_stats = True
                w = next(ev)
                x, y = w[:, :-1], w[:, 1:]
                with amp:
                    logits, _ = raw_model(x, y)
                lt = F.cross_entropy(
                    logits.float().view(-1, logits.size(-1)),
                    y.reshape(-1), reduction="none").view(x.shape[0], T)
                for bi in range(len(BUCKETS) - 1):
                    seg = lt[:, BUCKETS[bi]:BUCKETS[bi + 1]]
                    tok_sum[bi] += float(seg.sum())
                    tok_cnt[bi] += seg.numel()
        stats = {}
        for bi in range(len(BUCKETS) - 1):
            stats[f"val/pos_{BUCKETS[bi]}-{BUCKETS[bi + 1]}"] = \
                float(tok_sum[bi] / tok_cnt[bi])
        for i, m in gated:
            m.collect_stats = False
            s = m.stats
            stats[f"gates/L{i}/max_frac"] = float(s["gate_frac"].max())
            stats[f"gates/L{i}/entropy"] = s["router_entropy"]
            stats[f"gates/L{i}/mean_lifetime"] = s["mean_lifetime"]
            stats[f"gates/L{i}/frac_evicted"] = s["frac_evicted"]
        if hasattr(raw_model, "memory_stats"):
            stats.update(raw_model.memory_stats())
            stats["mem/gate"] = float(raw_model.mem_gate.detach())
        raw_model.train()
        return float(tok_sum.sum() / tok_cnt.sum()), stats

    def save(step, best):
        os.makedirs(run_dir, exist_ok=True)
        torch.save({"model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step, "best_val": best,
                    "config": cfg_to_dict(cfg), "args": vars(args),
                    "tokens": step * tokens_per_step,
                    "flops": step * tokens_per_step * fpt},
                   ckpt_path)

    def upload_ckpt(alias):
        """Checkpoint insurance to the DRIVE BANK (wandb Artifacts dropped
        2026-07-28: no storage on that account). Async so a 1.5 GB network
        push never stalls the train loop; if latest.pt is overwritten by the
        next save mid-push, bank.push's size verification fails loudly and
        the next insurance cycle retries. `alias` kept for call-site compat.
        """
        import threading
        from bank import push

        def _push():
            try:
                push(ckpt_path, folder=f"multicore2-runs/{args.run_name}")
            except Exception as e:
                print(f"[train] insurance push failed ({e!r}) — "
                      f"next cycle retries", flush=True)

        threading.Thread(target=_push, daemon=True).start()

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
            g["lr"] = lr * g.get("lr_scale", 1.0)

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
                           "config": cfg_to_dict(cfg)}, f, indent=2)
            if use_wandb:
                import wandb
                wandb.log({"val/loss": vl, "val/best": best_val,
                           **gate_stats}, step=done)
            t_last = time.time()          # don't bill eval time to the iter

        if (master and args.artifact_every
                and done % args.artifact_every == 0 and done < iters):
            upload_ckpt("latest")
            t_last = time.time()

    if master:
        print(f"[train] done: {iters} steps, "
              f"{iters * tokens_per_step / 1e9:.3f}B tokens, "
              f"{iters * tokens_per_step * fpt:.3e} FLOPs, "
              f"best val {best_val:.4f}", flush=True)
        # final checkpoint push is vast/upload_results.py's job (verified,
        # gates the instance's self-destroy) — no duplicate push here
        if use_wandb:
            import wandb
            wandb.finish()
    if ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
