#!/usr/bin/env bash
# YOCO, derivative-free. GPU 0 (RTX 5090) only, sequential. Waits for any running train.py.
set -u
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
while pgrep -f "yoco/train.py" > /dev/null; do sleep 20; done
run() { name=$1; shift; out=runs/q2/$name; [ -f $out/step3000.pt ] && return; rm -rf $out; mkdir -p $out
  echo "$(date +%T) start $name"; $PY yoco/train.py --output $out "$@" > $out.log 2>&1; echo "$(date +%T) done $name: $(tail -1 $out.log)"; }
run yoco-attn           --arch yoco --method dust --credit attn
run yoco-fp32           --arch yoco --method dust --fp32
run yoco-attn-fp32      --arch yoco --method dust --fp32 --credit attn
run gpt-attn            --arch gpt  --method dust --credit attn
run yoco-attn-s7        --arch yoco --method dust --credit attn --seed 7
run yoco-dust-s7        --arch yoco --method dust --seed 7
run yoco-attn-local4    --arch yoco --method dust --credit attn --local-scale 4
