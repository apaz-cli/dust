#!/usr/bin/env bash
# Derivative-free follow-up. GPU 0 (RTX 5090) only, sequential.
# The local VJP showed the k/v credit is lost to ZO noise; can more (cheap) ZO local draws recover it?
# Oracle arms are diagnostics only (autograd computes what the estimator targets).
set -u
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
run() { name=$1; shift; out=runs/q2/$name; [ -f $out/step3000.pt ] && return; rm -rf $out; mkdir -p $out
  echo "$(date +%T) start $name"; $PY yoco/train.py --output $out "$@" > $out.log 2>&1; echo "$(date +%T) done $name: $(tail -1 $out.log)"; }
run gpt-local4          --arch gpt  --method dust --local-scale 4
run gpt-local16         --arch gpt  --method dust --local-scale 16
run gpt-headdiv4        --arch gpt  --method dust --head-div 4
run gpt-local16-hd4     --arch gpt  --method dust --local-scale 16 --head-div 4
run yoco-local16-hd4    --arch yoco --method dust --local-scale 16 --head-div 4
run gpt-local64-hd4     --arch gpt  --method dust --local-scale 64 --head-div 4
run gpt-local16-hd4-g1  --arch gpt  --method dust --local-scale 16 --head-div 4 --gamma 1.0
run gpt-oracle-attn     --arch gpt  --method dust --oracle attn
run gpt-oracle-mlp      --arch gpt  --method dust --oracle writers hidden
run gpt-dust-s7         --arch gpt  --method dust --seed 7
run gpt-local16-hd4-s7  --arch gpt  --method dust --local-scale 16 --head-div 4 --seed 7
run L8-gpt-dust         --arch gpt  --method dust --n-layer 8 --n-embd 512
run L8-gpt-local16-hd4  --arch gpt  --method dust --local-scale 16 --head-div 4 --n-layer 8 --n-embd 512
