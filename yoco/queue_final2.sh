#!/usr/bin/env bash
# Final runs, part 2: add night 1's live-unit c_fc jitter + head-error control variate (--live-fc --cv) to the
# fast recipe and the leak-free scoring. Never combined before: on night-1 code it took 2.765 -> 2.464 (2 seeds).
# Starts after queue_final. RTX 5090 only.
set -u
until grep -q "queue final done" runs/final.log; do sleep 60; done
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
D=runs/final
run() { name=$1; shift; out=$D/$name; [ -f $out/step3000.pt ] && return; rm -rf "${out:?}"; mkdir -p $out
  echo "$(date +%T) start $name"; $PY yoco/train.py --output $out "$@" > $out.log 2>&1
  echo "$(date +%T) done $name: $(tail -1 $out.log)"; }
BEST="--arch yoco --method dust --precision fast16 --chain 256 --chain-own --chain-hubs-only --emb-scale 4 --wsd 0.2"
F="$BEST --local-separate --cache-separate"
L8="--n-layer 8 --n-embd 512"
run ts-L4-lcv-s42   $F --local-scale 1 --live-fc --cv
run ts-L4-lcv-s7    $F --local-scale 1 --live-fc --cv --seed 7
run ts-L4-fastlcv   $BEST --local-scale 4 --live-fc --cv
run copy-L4-lcv     $F --local-scale 1 --live-fc --cv --data-dir data/tsrepeat --gamma 1 --cache-gamma 0.98
run ts-L8-lcv       $F --local-scale 4 --live-fc --cv $L8
$PY yoco/halves.py runs/n6/rep-sc-g1self $D/copy-L4-s7 $D/copy-L4-lcv
echo "$(date +%T) queue final2 done"
