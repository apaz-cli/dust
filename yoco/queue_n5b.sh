#!/usr/bin/env bash
# Track A step 2: can Dust learn to copy? TinyStories-repeat (story + the same story). RTX 5090 only.
set -u
export CUDA_VISIBLE_DEVICES=0
until grep -q "queue done" runs/n5.log; do sleep 30; done
PY=.venv/bin/python
D=runs/n5
run() { name=$1; shift; out=$D/$name; [ -f $out/step3000.pt ] && return; rm -rf "${out:?}"; mkdir -p $out
  echo "$(date +%T) start $name"; $PY yoco/train.py --output $out "$@" > $out.log 2>&1
  echo "$(date +%T) done $name: $(tail -1 $out.log)"; }
R="--arch yoco --data-dir data/tsrepeat"
run rep-ed-g98   $R --method ed --gamma 0.98 --lr 0.2
run rep-ed-g1    $R --method ed --gamma 1 --lr 0.2
run rep-dust     $R --method dust --precision fast16 --chain 256 --chain-own --chain-hubs-only --local-scale 4 --emb-scale 4 --wsd 0.2
run rep-dust-stock $R --method dust --precision fast16
$PY yoco/halves.py $D/rep-bp $D/rep-ed-g98 $D/rep-ed-g1 $D/rep-dust $D/rep-dust-stock
echo "$(date +%T) queue b done"
