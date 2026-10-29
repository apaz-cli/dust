#!/usr/bin/env bash
# YOCO, derivative-free. GPU 0 (RTX 5090) only, sequential. Waits for any running train.py.
set -u
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
while pgrep -f "yoco/train.py" > /dev/null; do sleep 20; done
run() { name=$1; shift; out=runs/q2/$name; [ -f $out/step3000.pt ] && return; rm -rf $out; mkdir -p $out
  echo "$(date +%T) start $name"; $PY yoco/train.py --output $out "$@" > $out.log 2>&1; echo "$(date +%T) done $name: $(tail -1 $out.log)"; }
run yoco-local4          --arch yoco --method dust --local-scale 4
run yoco-fp32-local4     --arch yoco --method dust --fp32 --local-scale 4
run yoco-local16-hd4     --arch yoco --method dust --local-scale 16 --head-div 4
run yoco-fp32-local4-hub4 --arch yoco --method dust --fp32 --local-scale 4 --hub-scale 4
run gpt-fp32             --arch gpt  --method dust --fp32
run yoco-dust-s7         --arch yoco --method dust --seed 7
