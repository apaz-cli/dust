#!/usr/bin/env bash
# Track B step 3: Dust with AdamW on dense and MoE (SGD hides the MoE gain even under backprop). 4070 only.
set -u
export CUDA_VISIBLE_DEVICES=1
until grep -q "queue3 done" runs/moe.log; do sleep 20; done
PY=.venv/bin/python
D=runs/moe
run() { name=$1; shift; out=$D/$name; [ -f $out/step1500.pt ] && return; rm -rf "${out:?}"; mkdir -p $out
  echo "$(date +%T) start $name"; $PY yoco/train.py --output $out --steps 1500 --save-at 1500 "$@" > $out.log 2>&1
  echo "$(date +%T) done $name: $(tail -1 $out.log)"; }
M="--moe 8,2,384,256"
run bp1500-adam-dense   --arch yoco --method bp --opt adamw --adam-lr 3e-3
run bp1500-adam-e8k2s   --arch yoco --method bp --opt adamw --adam-lr 3e-3 $M
BEST="--arch yoco --method dust --precision fast16 --chain 256 --chain-own --chain-hubs-only --local-scale 4 --emb-scale 4 --wsd 0.2 --opt adamw"
run dust1500-adam1e-3-dense  $BEST --adam-lr 1e-3
run dust1500-adam1e-3-e8k2s  $BEST --adam-lr 1e-3 $M
echo "$(date +%T) queue4 done"
