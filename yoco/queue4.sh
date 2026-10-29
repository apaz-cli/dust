#!/usr/bin/env bash
# After queue3. GPU 0 only. How the remaining draw-noise gap shrinks with 4x the population.
set -u
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
while pgrep -f "bash yoco/queue[123].sh" > /dev/null; do sleep 20; done
run() { name=$1; shift; out=runs/q4/$name; [ -f $out/step3000.pt ] && return; rm -rf $out; mkdir -p $out
  echo "$(date +%T) start $name"; $PY yoco/train.py --output $out "$@" > $out.log 2>&1; echo "$(date +%T) done $name: $(tail -1 $out.log)"; }
run gpt-vjp-head-p1024  --arch gpt --method dust --local vjp --head exact --population 1024
run gpt-dust-p1024      --arch gpt --method dust --population 1024
