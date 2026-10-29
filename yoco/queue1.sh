#!/usr/bin/env bash
# L4-D256 TinyStories, 3000 steps, population-256 recipe. GPU 0 (RTX 5090) only, sequential.
set -u
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
run() { name=$1; shift; out=runs/q1/$name; [ -f $out/step3000.pt ] && return; rm -rf $out; mkdir -p $out
  echo "$(date +%T) start $name"; $PY yoco/train.py --output $out "$@" > $out.log 2>&1; echo "$(date +%T) done $name: $(tail -1 $out.log)"; }
run gpt-bp        --arch gpt  --method bp --lr 0.25
run yoco-bp       --arch yoco --method bp --lr 0.25
run gpt-dust      --arch gpt  --method dust
run yoco-dust     --arch yoco --method dust
run yoco-vjp      --arch yoco --method dust --local vjp
run yoco-handoff  --arch yoco --method dust --local vjp --handoff
run gpt-vjp       --arch gpt  --method dust --local vjp
run gpt-exacthead --arch gpt  --method dust --head exact
