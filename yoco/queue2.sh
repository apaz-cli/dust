#!/usr/bin/env bash
# After queue1. GPU 0 (RTX 5090) only, sequential.
# The exact local attention VJP is the finding to confirm: seeds, then L8, then where the remaining noise lives.
set -u
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
while pgrep -f "bash yoco/queue1.sh" > /dev/null; do sleep 20; done
run() { name=$1; shift; out=runs/q2/$name; [ -f $out/step3000.pt ] && return; rm -rf $out; mkdir -p $out
  echo "$(date +%T) start $name"; $PY yoco/train.py --output $out "$@" > $out.log 2>&1; echo "$(date +%T) done $name: $(tail -1 $out.log)"; }
run yoco-vjp-s7         --arch yoco --method dust --local vjp --seed 7
run gpt-vjp-s7          --arch gpt  --method dust --local vjp --seed 7
run L8-gpt-dust         --arch gpt  --method dust --n-layer 8 --n-embd 512
run L8-gpt-vjp          --arch gpt  --method dust --local vjp --n-layer 8 --n-embd 512
run L8-yoco-vjp         --arch yoco --method dust --local vjp --n-layer 8 --n-embd 512
run gpt-oracle-attn     --arch gpt  --method dust --oracle attn
run gpt-oracle-mlp      --arch gpt  --method dust --oracle writers hidden
run gpt-oracle-wte      --arch gpt  --method dust --oracle wte
run gpt-dust-s7         --arch gpt  --method dust --seed 7
run yoco-dust-s7        --arch yoco --method dust --seed 7
run yoco-handoff-h128   --arch yoco --method dust --local vjp --handoff --handoff-draws 128
run gpt-oracle-all      --arch gpt  --method dust --oracle writers hidden attn wte head
