#!/usr/bin/env bash
# After queue1. GPU 0 (RTX 5090) only, sequential.
# 1) which group's draw noise costs the most: stock DUST with one group given its noise-free expected update
# 2) gamma on real DUST  3) seeds  4) handoff draw count  5) L8-D512
set -u
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
while pgrep -f "bash yoco/queue1.sh" > /dev/null; do sleep 20; done
run() { name=$1; shift; out=runs/q2/$name; [ -f $out/step3000.pt ] && return; rm -rf $out; mkdir -p $out
  echo "$(date +%T) start $name"; $PY yoco/train.py --output $out "$@" > $out.log 2>&1; echo "$(date +%T) done $name: $(tail -1 $out.log)"; }
run gpt-oracle-attn     --arch gpt  --method dust --oracle attn
run gpt-oracle-mlp      --arch gpt  --method dust --oracle writers hidden
run gpt-oracle-wte      --arch gpt  --method dust --oracle wte
run gpt-oracle-all      --arch gpt  --method dust --oracle writers hidden attn wte head
run gpt-dust-g999       --arch gpt  --method dust --gamma 0.999
run gpt-dust-s2         --arch gpt  --method dust --seed 7
run yoco-handoff-s2     --arch yoco --method dust --local vjp --handoff --seed 7
run yoco-dust-s2        --arch yoco --method dust --seed 7
run yoco-handoff-h128   --arch yoco --method dust --local vjp --handoff --handoff-draws 128
run L8-gpt-dust         --arch gpt  --method dust --n-layer 8 --n-embd 512
run L8-yoco-handoff     --arch yoco --method dust --local vjp --handoff --n-layer 8 --n-embd 512
run L8-yoco-dust        --arch yoco --method dust --n-layer 8 --n-embd 512
