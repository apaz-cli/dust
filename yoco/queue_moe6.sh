#!/usr/bin/env bash
# Track B step 5: does Dust-MoE catch up with dense as tokens per step grow (out of the noise-limited regime)?
# Same total tokens as the 1500 x 8 runs (dust1500-dense 2.987, dust1500-e8k2s 3.085). RTX 4070 only.
set -u
export CUDA_VISIBLE_DEVICES=1
PY=.venv/bin/python
D=runs/moe
run() { name=$1; steps=$2; shift 2; out=$D/$name; [ -f $out/step$steps.pt ] && return; rm -rf "${out:?}"; mkdir -p $out
  echo "$(date +%T) start $name"; $PY yoco/train.py --output $out --steps $steps --save-at $steps --eval-every 25 "$@" > $out.log 2>&1
  echo "$(date +%T) done $name: $(tail -1 $out.log)"; }
M="--moe 8,2,384,256"
BEST="--arch yoco --method dust --precision fast16 --chain 256 --chain-own --chain-hubs-only --local-scale 4 --emb-scale 4 --wsd 0.2"
run dust-b32-dense 375 $BEST --batch-seqs 32
run dust-b32-e8k2s 375 $BEST --batch-seqs 32 $M
run dust-b64-dense 188 $BEST --batch-seqs 64 --microbatch-size 32
run dust-b64-e8k2s 188 $BEST --batch-seqs 64 --microbatch-size 32 $M
echo "$(date +%T) queue6 done"
