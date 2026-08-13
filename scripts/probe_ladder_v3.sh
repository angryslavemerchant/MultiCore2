#!/bin/bash
# v3 delta-machines probe ladder: does a routed machine population with
# the anti-collapse package keep per-token routing alive, and what do the
# parts buy? Four rungs, SAME seed/data stream, ~300M tokens each:
#   r1 plain-MoM substrate (memories only)   r3 + conference
#   r2 + private MLPs                        r4 + learned write gate (full)
# Gates autopsy after each rung -> runs/<name>/fourstroke_gates.json;
# collapse readout = route_traffic_per_round (v2 collapsed to static
# committees: 4 machines ~1.0, rest ~0 -- memory multicore2-v3-deltamachines).
#
#   NPROC=2 bash scripts/probe_ladder_v3.sh          # user's 2x box
#   SCALE=small PATTERN=FDDFFDDF TOKENS=3e8 ...      # env overrides
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
NPROC=${NPROC:-1}
SCALE=${SCALE:-small}
PATTERN=${PATTERN:-FDDFFDDF}
TOKENS=${TOKENS:-3e8}
COMMON="--scale $SCALE --seq-len 1024 --attn-pattern $PATTERN
  --fs-n-machines 16 --fs-d-machine 256 --fs-n-head-m 4 --fs-topk 4
  --fs-dense-warmup 60 --fs-route-noise 0.3 --fs-zloss 1e-3
  --lb-coef 0.01 --opt muon --micro-bs ${MICRO_BS:-8}
  --target-tokens $TOKENS --eval-every 100 --eval-iters 10"
if [ "$NPROC" -gt 1 ]; then
  TRAIN="torchrun --standalone --nproc_per_node $NPROC scripts/train_gpt2.py"
else
  TRAIN="$PY scripts/train_gpt2.py"
fi

run_rung () {  # name, extra ablation flags...
  name=$1; shift
  echo "=== RUNG $name ==="
  $TRAIN $COMMON --run-name "$name" "$@"
  $PY scripts/diag_fourstroke_gates.py --run-name "$name" \
      --batches 4 --micro-bs 2 --seq-len 1024
}

run_rung dmprobe-r1-mom  --fs-dm-no-mlp --fs-dm-no-conf --fs-dm-no-gate
run_rung dmprobe-r2-mlp  --fs-dm-no-conf --fs-dm-no-gate
run_rung dmprobe-r3-conf --fs-dm-no-gate
run_rung dmprobe-r4-full

echo "=== SUMMARY (best val per rung) ==="
for r in dmprobe-r1-mom dmprobe-r2-mlp dmprobe-r3-conf dmprobe-r4-full; do
  $PY - "$r" <<'EOF'
import json, sys
m = json.load(open(f"runs/{sys.argv[1]}/metrics.json"))
print(f"{sys.argv[1]}: best val {m.get('best_val')}")
EOF
done
echo "LADDER_DONE"
