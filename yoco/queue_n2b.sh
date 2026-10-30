#!/usr/bin/env bash
# L8-D512 check of the best derivative-free recipe. GPU 0 (RTX 5090) only, sequential.
set -u
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
run() { name=$1; shift; out=runs/n2/$name; [ -f $out/step3000.pt ] && return; rm -rf $out; mkdir -p $out
  echo "$(date +%T) start $name"; $PY yoco/train.py --output $out "$@" > $out.log 2>&1
  echo "$(date +%T) done $name: $(tail -1 $out.log)"; }
run L8-best-zo     --arch yoco --method dust --n-layer 8 --n-embd 512 --precision fast16 --chain 256 --chain-own --chain-hubs-only --local-scale 4 --emb-scale 4 --wsd 0.2
run L8-stock-fast  --arch yoco --method dust --n-layer 8 --n-embd 512 --precision fast16
