#!/bin/bash
# v3.1 probe ladder: population reads / briefings / magnitude writes /
# conference state-commit, stacked on the full v3 block. Four rungs,
# SAME seed/data/schedule as probe_ladder_v3.sh (ladder2 2026-08-14) so
# its numbers are direct baselines:
#   sctrl2 4.1492 | r4-full 4.0846 (token-matched, 300M tok, 2x5090)
# Rungs (each includes full v3: mlp+conf+gate):
#   rA + pop_read              rC + pop + brief + mag_topk
#   rB + pop_read + brief      rD + all four (+conf_commit)
# Gates autopsy after each rung (route_traffic vs WRITE traffic g:
# under mag_topk they diverge — that divergence IS the finding).
#
#   NPROC=2 bash scripts/probe_ladder_v31.sh
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

run_rung () {  # name, extra flags...
  name=$1; shift
  echo "=== RUNG $name ==="
  $TRAIN $COMMON --run-name "$name" "$@"
  $PY scripts/diag_fourstroke_gates.py --run-name "$name" \
      --batches 4 --micro-bs 2 --seq-len 1024
}

run_rung dmprobe31-rA-pop    --fs-pop-read
run_rung dmprobe31-rB-brief  --fs-pop-read --fs-brief
run_rung dmprobe31-rC-mag    --fs-pop-read --fs-brief --fs-mag-topk
run_rung dmprobe31-rD-commit --fs-pop-read --fs-brief --fs-mag-topk \
                             --fs-conf-commit

echo "=== SUMMARY (best val per rung; baselines sctrl2 4.1492 / r4 4.0846) ==="
for r in dmprobe31-rA-pop dmprobe31-rB-brief dmprobe31-rC-mag \
         dmprobe31-rD-commit; do
  $PY - "$r" <<'EOF'
import json, sys
m = json.load(open(f"runs/{sys.argv[1]}/metrics.json"))
print(f"{sys.argv[1]}: best val {m.get('best_val')}")
EOF
done
echo "LADDER31_DONE"
