#!/usr/bin/env bash
# GPU 0 only. How far live-fc + cv lets the early (full-forward) draws drop.
set -u
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
while pgrep -f "bash yoco/queue[234567].sh" > /dev/null || pgrep -f "yoco/train.py" > /dev/null; do sleep 20; done
run() { name=$1; shift; out=runs/q8/$name; [ -f $out/step3000.pt ] && return; rm -rf $out; mkdir -p $out
  echo "$(date +%T) start $name"; $PY yoco/train.py --output $out "$@" > $out.log 2>&1; echo "$(date +%T) done $name: $(tail -1 $out.log)"; }
BEST="--arch yoco --method dust --fp32 --local-scale 4 --hub-scale 4"
run best-live-cv-quarter $BEST --live-fc --cv --alloc '{"writers": [10, 8, 16, 10], "hidden": [10, 8, 16, 10], "embedding": 10}'
run best-live-cv-eighth  $BEST --live-fc --cv --alloc '{"writers": [6, 4, 16, 10], "hidden": [6, 4, 16, 10], "embedding": 6}'
run stock-live-cv-half   --arch yoco --method dust --live-fc --cv --alloc '{"writers": [20, 16, 16, 10], "hidden": [20, 16, 16, 10], "embedding": 6}'
