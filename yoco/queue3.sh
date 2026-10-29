#!/usr/bin/env bash
# After queue2. GPU 0 only. Reallocation: more attention-output draws now that the internals are exact.
set -u
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
while pgrep -f "bash yoco/queue[12].sh" > /dev/null; do sleep 20; done
run() { name=$1; shift; out=runs/q3/$name; [ -f $out/step3000.pt ] && return; rm -rf $out; mkdir -p $out
  echo "$(date +%T) start $name"; $PY yoco/train.py --output $out "$@" > $out.log 2>&1; echo "$(date +%T) done $name: $(tail -1 $out.log)"; }
run gpt-vjp-head-hub4   --arch gpt  --method dust --local vjp --head exact --hub-scale 4
run yoco-vjp-head-hub4  --arch yoco --method dust --local vjp --head exact --hub-scale 4
