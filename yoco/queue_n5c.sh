#!/usr/bin/env bash
# Track A step 3: leak-free local k/v at L8-D512 (night 2 reference L8-best-zo 2.918, 3552 s). RTX 5090 only.
set -u
export CUDA_VISIBLE_DEVICES=0
until grep -q "queue b done" runs/n5.log; do sleep 30; done
PY=.venv/bin/python
D=runs/n5
run() { name=$1; shift; out=$D/$name; [ -f $out/step3000.pt ] && return; rm -rf "${out:?}"; mkdir -p $out
  echo "$(date +%T) start $name"; $PY yoco/train.py --output $out "$@" > $out.log 2>&1
  echo "$(date +%T) done $name: $(tail -1 $out.log)"; }
B="--arch yoco --method dust --n-layer 8 --n-embd 512 --precision fast16 --chain 256 --chain-own --chain-hubs-only --emb-scale 4 --wsd 0.2"
run L8-sep-l1 $B --local-scale 1 --local-separate
run L8-sep-l4 $B --local-scale 4 --local-separate
echo "$(date +%T) queue c done"
