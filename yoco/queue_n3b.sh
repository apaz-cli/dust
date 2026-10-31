#!/usr/bin/env bash
# After queue_n3: per-site gamma for the YOCO cache (L4 best ZO recipe). GPU 0 (RTX 5090) only.
set -u
export CUDA_VISIBLE_DEVICES=0
until grep -q "queue done" runs/n3.log; do sleep 30; done
PY=.venv/bin/python
D=runs/n3
run() { name=$1; shift; out=$D/$name; [ -f $out/step3000.pt ] && return; rm -rf "${out:?}"; mkdir -p $out
  echo "$(date +%T) start $name"; $PY yoco/train.py --output $out "$@" > $out.log 2>&1
  echo "$(date +%T) done $name: $(tail -1 $out.log)"; }
BEST="--arch yoco --method dust --precision fast16 --chain 256 --chain-own --chain-hubs-only --local-scale 4 --emb-scale 4 --wsd 0.2"
run L4-cacheg995    $BEST --cache-gamma 0.995
run L4-cacheg995-s7 $BEST --cache-gamma 0.995 --seed 7
echo "$(date +%T) queue b done"
