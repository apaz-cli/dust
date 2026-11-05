#!/usr/bin/env bash
# Track B step 4: Dust on a dense MLP as wide as the MoE's total hidden (E1 k1 h3072 + shared 256 = 3328).
# Prediction from per-weight SNR ~ tokens / A: about the same loss as the MoE under Dust. 4070 only.
set -u
export CUDA_VISIBLE_DEVICES=1
PY=.venv/bin/python
D=runs/moe
run() { name=$1; shift; out=$D/$name; [ -f $out/step1500.pt ] && return; rm -rf "${out:?}"; mkdir -p $out
  echo "$(date +%T) start $name"; $PY yoco/train.py --output $out --steps 1500 --save-at 1500 "$@" > $out.log 2>&1
  echo "$(date +%T) done $name: $(tail -1 $out.log)"; }
W="--moe 1,1,3072,256"
run bp1500-wide   --arch yoco --method bp --lr 0.2 $W
BEST="--arch yoco --method dust --precision fast16 --chain 256 --chain-own --chain-hubs-only --local-scale 4 --emb-scale 4 --wsd 0.2"
run dust1500-wide $BEST $W
echo "$(date +%T) queue5 done"
