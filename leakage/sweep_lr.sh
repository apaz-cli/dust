#!/usr/bin/env bash
# Short LR sweep for the small model, one run at a time on one GPU.
# Usage: LRS="0.125 0.25 0.5" STEPS=300 leakage/sweep_lr.sh [python]
set -euo pipefail
PY=${1:-python}
STEPS=${STEPS:-300}
mkdir -p runs/lr-sweep
for lr in ${LRS:-0.0625 0.125 0.25 0.5 1 2 4}; do
  out=runs/lr-sweep/lr$lr
  rm -rf "$out"
  "$PY" leakage/train_small.py --steps "$STEPS" --save-at -1 --lr "$lr" --output "$out" > "$out.log" 2>&1 \
    || echo "lr=$lr failed"
  echo "lr=$lr $(grep -E 'val_loss|diverged' "$out.log" | tail -1)"
done
