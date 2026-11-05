#!/usr/bin/env bash
# Night 5, track A: leak-free local k/v/gate/ve (--local-separate) in training. RTX 5090 (index 0) only.
# Baseline (night 2, same recipe without the flag): 2.733 (seed 42) / 2.749 (seed 7).
set -u
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
D=runs/n5
run() { name=$1; shift; out=$D/$name; [ -f $out/step3000.pt ] && return; rm -rf "${out:?}"; mkdir -p $out
  echo "$(date +%T) start $name"; $PY yoco/train.py --output $out "$@" > $out.log 2>&1
  echo "$(date +%T) done $name: $(tail -1 $out.log)"; }
B="--arch yoco --method dust --precision fast16 --chain 256 --chain-own --chain-hubs-only --emb-scale 4 --wsd 0.2"
run sep-l4-s42   $B --local-scale 4 --local-separate
run sep-l1-s42   $B --local-scale 1 --local-separate
run stock-l1-s42 $B --local-scale 1
run sep-l4-s7    $B --local-scale 4 --local-separate --seed 7
run sep-l1-s7    $B --local-scale 1 --local-separate --seed 7
echo "$(date +%T) queue done"
