#!/usr/bin/env bash
# Seeds, handoff draw count, and the L8-D512 shape. GPU 0 only, sequential, after queue1.
set -u
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
while pgrep -f yoco/queue1.sh > /dev/null; do sleep 20; done
run() { name=$1; shift; out=runs/q2/$name; [ -f $out/step3000.pt ] && return; rm -rf $out; mkdir -p $out
  echo "$(date +%T) start $name"; $PY yoco/train.py --output $out "$@" > $out.log 2>&1; echo "$(date +%T) done $name: $(tail -1 $out.log)"; }
run yoco-handoff-h128   --arch yoco --method dust --local vjp --handoff --handoff-draws 128
run gpt-dust-s2         --arch gpt  --method dust --seed 7
run yoco-dust-s2        --arch yoco --method dust --seed 7
run yoco-handoff-s2     --arch yoco --method dust --local vjp --handoff --seed 7
run yoco-vjp-s2         --arch yoco --method dust --local vjp --seed 7
run L8-gpt-dust         --arch gpt  --method dust --n-layer 8 --n-embd 512
run L8-yoco-dust        --arch yoco --method dust --n-layer 8 --n-embd 512
run L8-yoco-handoff     --arch yoco --method dust --local vjp --handoff --n-layer 8 --n-embd 512
