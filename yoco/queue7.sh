#!/usr/bin/env bash
# GPU 0 only. Seed-7 confirmations of queue 6, and the queue-3 best recipe rerun on an idle GPU for wall time.
set -u
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
while pgrep -f "bash yoco/queue[23456].sh" > /dev/null || pgrep -f "yoco/train.py" > /dev/null; do sleep 20; done
run() { name=$1; shift; out=runs/q7/$name; [ -f $out/step3000.pt ] && return; rm -rf $out; mkdir -p $out
  echo "$(date +%T) start $name"; $PY yoco/train.py --output $out "$@" > $out.log 2>&1; echo "$(date +%T) done $name: $(tail -1 $out.log)"; }
BEST="--arch yoco --method dust --fp32 --local-scale 4 --hub-scale 4 --emb-scale 4"
HALF='{"writers": [20, 16, 16, 10], "hidden": [20, 16, 16, 10], "embedding": 20}'
run best-live-cv-s7      $BEST --live-fc --cv --seed 7
run best-live-cv-half-s7 $BEST --live-fc --cv --alloc "$HALF" --seed 7
run best-timing          $BEST
run stock-timing         --arch yoco --method dust
