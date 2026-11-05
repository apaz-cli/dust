#!/usr/bin/env bash
# Track B step 2: Dust on MoE vs Dust on dense, with backprop references, 1500 steps (4070 is slow). 4070 only.
# MoE config E8 k2 h384 + shared 256: active hidden 1024 = dense; Dust needs the shared expert's c_proj site.
set -u
export CUDA_VISIBLE_DEVICES=1
until grep -q "queue2 done" runs/moe.log; do sleep 20; done
PY=.venv/bin/python
D=runs/moe
run() { name=$1; shift; out=$D/$name; [ -f $out/step1500.pt ] && return; rm -rf "${out:?}"; mkdir -p $out
  echo "$(date +%T) start $name"; $PY yoco/train.py --output $out --steps 1500 --save-at 1500 "$@" > $out.log 2>&1
  echo "$(date +%T) done $name: $(tail -1 $out.log)"; }
M="--moe 8,2,384,256"
run bp1500-dense      --arch yoco --method bp --lr 0.2
run bp1500-e8k2s      --arch yoco --method bp --lr 0.2 $M
BEST="--arch yoco --method dust --precision fast16 --chain 256 --chain-own --chain-hubs-only --local-scale 4 --emb-scale 4 --wsd 0.2"
run dust1500-dense    $BEST
run dust1500-e8k2s    $BEST $M
echo "$(date +%T) queue3 done"
