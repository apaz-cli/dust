#!/usr/bin/env bash
# After queue2. GPU 0 only. Precision and hub-draw arms (derivative-free).
set -u
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
while pgrep -f "bash yoco/queue2.sh" > /dev/null || pgrep -f "yoco/train.py" > /dev/null; do sleep 20; done
run() { name=$1; shift; out=runs/q3/$name; [ -f $out/step3000.pt ] && return; rm -rf $out; mkdir -p $out
  echo "$(date +%T) start $name"; $PY yoco/train.py --output $out "$@" > $out.log 2>&1; echo "$(date +%T) done $name: $(tail -1 $out.log)"; }
run yoco-attn-fp32-hub4 --arch yoco --method dust --fp32 --credit attn --hub-scale 4
run yoco-attn-hub4      --arch yoco --method dust --credit attn --hub-scale 4
