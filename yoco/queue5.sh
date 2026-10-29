#!/usr/bin/env bash
# After queue3. GPU 0 only. Robustness of the best derivative-free recipe (fp32 + 4x local draws).
set -u
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
while pgrep -f "bash yoco/queue[234].sh" > /dev/null || pgrep -f "yoco/train.py" > /dev/null; do sleep 20; done
run() { name=$1; shift; out=runs/q5/$name; [ -f $out/step3000.pt ] && return; rm -rf $out; mkdir -p $out
  echo "$(date +%T) start $name"; $PY yoco/train.py --output $out "$@" > $out.log 2>&1; echo "$(date +%T) done $name: $(tail -1 $out.log)"; }
run yoco-fp32-local4-hub4-emb16 --arch yoco --method dust --fp32 --local-scale 4 --hub-scale 4 --emb-scale 16
run yoco-fp32-local4-hub4-flat --arch yoco --method dust --fp32 --local-scale 4 --hub-scale 4 --flat-alloc
run yoco-hub4               --arch yoco --method dust --hub-scale 4
