#!/usr/bin/env bash
# Track A step 4: with leak-free local scoring, is a longer credit horizon (gamma 1) now cheap? RTX 5090 only.
# Copy task (gamma .98 costs .24 nat noise-free vs .065 at gamma 1) and TinyStories.
set -u
export CUDA_VISIBLE_DEVICES=0
until grep -q "queue c done" runs/n5.log; do sleep 30; done
PY=.venv/bin/python
D=runs/n5
run() { name=$1; shift; out=$D/$name; [ -f $out/step3000.pt ] && return; rm -rf "${out:?}"; mkdir -p $out
  echo "$(date +%T) start $name"; $PY yoco/train.py --output $out "$@" > $out.log 2>&1
  echo "$(date +%T) done $name: $(tail -1 $out.log)"; }
B="--arch yoco --method dust --precision fast16 --chain 256 --chain-own --chain-hubs-only --emb-scale 4 --wsd 0.2 --local-scale 1 --local-separate"
run rep-sep-g98  $B --data-dir data/tsrepeat
run rep-sep-g1   $B --data-dir data/tsrepeat --gamma 1
run sep-l1-g1    $B --gamma 1
$PY yoco/halves.py $D/rep-bp $D/rep-ed-g98 $D/rep-ed-g1 $D/rep-dust $D/rep-sep-g98 $D/rep-sep-g1
echo "$(date +%T) queue d done"
