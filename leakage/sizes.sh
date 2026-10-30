#!/usr/bin/env bash
# Reproduce the leakage figures: three model sizes trained with DUST (population-256 recipe), kappa measured
# on the training batch every 25 steps. One GPU. Run from the repo root: leakage/sizes.sh [python]
# Each size's LR is its best 300-step LR (leakage/edge.py --axis lr) unless given; the medium model uses 0.25,
# tied with its best probe (0.354: val 3.457 vs 3.458).
set -euo pipefail
PY=${1:-python}
[ -f data/tinystories/train.pt ] || "$PY" leakage/prepare_tinystories.py
for spec in "2 128" "4 256 0.25" "8 512"; do
  set -- $spec
  out=runs/sizes/L$1-D$2
  [ -f "$out/metrics.jsonl" ] && [ "$(wc -l < "$out/metrics.jsonl")" -ge 3000 ] && continue
  lr=${3:-}
  if [ -z "$lr" ]; then
    "$PY" leakage/edge.py --axis lr --n-layer "$1" --n-embd "$2" --bisect 1 --probe-only --python "$PY"
    lr=$("$PY" -c "import json; print(json.load(open('runs/edge/L$1-D$2/edge.json'))['best_val_probe'])")
  fi
  rm -rf "$out"
  "$PY" leakage/train_small.py --n-layer "$1" --n-embd "$2" --lr "$lr" --steps 3000 --leak-every 25 \
    --save-at 0 3000 --output "$out" > "$out.log" 2>&1
  echo "L$1-D$2 lr=$lr: $(tail -1 "$out/metrics.jsonl")"
done
for r in L2-D128 L4-D256 L8-D512; do
  "$PY" leakage/plot.py --by-block --run runs/sizes/$r --output runs/sizes/kappa_$r.svg
done
"$PY" leakage/plot.py --loss --run runs/sizes/L2-D128 runs/sizes/L4-D256 runs/sizes/L8-D512 --output runs/sizes/loss.svg

# The 8-layer model hasn't started using context by step 3000, so it also gets a 9000-step run on its own,
# longer training split (the first 24000 rows match data/tinystories; same validation split).
[ -f data/tinystories-long/train.pt ] || "$PY" leakage/prepare_tinystories.py --train-rows 72000 --output data/tinystories-long
out=runs/long/L8-D512
if ! { [ -f "$out/metrics.jsonl" ] && [ "$(wc -l < "$out/metrics.jsonl")" -ge 9000 ]; }; then
  rm -rf "$out" && mkdir -p runs/long
  "$PY" leakage/train_small.py --n-layer 8 --n-embd 512 --lr 0.25 --steps 9000 --leak-every 50 --save-at 0 3000 9000 \
    --data-dir data/tinystories-long --output "$out" > "$out.log" 2>&1
fi
"$PY" leakage/plot.py --by-block --run "$out" --output runs/long/kappa_L8-D512.svg
