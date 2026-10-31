#!/usr/bin/env bash
# After queue_n3b: L12 own-token bias (K=16 fits in memory). GPU 0 (RTX 5090) only.
set -u
export CUDA_VISIBLE_DEVICES=0
until grep -q "queue b done" runs/n3.log; do sleep 30; done
echo "$(date +%T) bias L12-bp (K=16)"
.venv/bin/python yoco/measure_bias.py --sequences 2 --K 16 --run runs/n3/L12-bp-0.05 > runs/n3/bias-L12-bp.log 2>&1
.venv/bin/python yoco/measure_bias.py --sequences 2 --K 16 --run runs/n3/L12-bp-0.05 --gamma 0.98 > runs/n3/bias-L12-bp05-g98.log 2>&1
echo "$(date +%T) queue c done"
