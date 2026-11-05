#!/usr/bin/env bash
# Track B step 6: fixed 1500 steps, more tokens per step (32 and 64 sequences = 4x and 8x the tokens).
# Does Dust-MoE pull ahead of dense once the draw-noise limit is relaxed? References at 8 seqs: dense 2.987,
# MoE 3.085. RTX 4070 only (~2.7-7.7 h per run).
set -u
export CUDA_VISIBLE_DEVICES=1
PY=.venv/bin/python
D=runs/moe
run() { name=$1; shift; out=$D/$name; [ -f $out/step1500.pt ] && return; rm -rf "${out:?}"; mkdir -p $out
  echo "$(date +%T) start $name"; $PY yoco/train.py --output $out --steps 1500 --save-at 1500 "$@" > $out.log 2>&1
  echo "$(date +%T) done $name: $(tail -1 $out.log)"; }
M="--moe 8,2,384,256"
BEST="--arch yoco --method dust --precision fast16 --chain 256 --chain-own --chain-hubs-only --local-scale 4 --emb-scale 4 --wsd 0.2"
run dust1500-b32-dense $BEST --batch-seqs 32
run dust1500-b32-e8k2s $BEST --batch-seqs 32 $M
run dust1500-b64-dense $BEST --batch-seqs 64 --microbatch-size 32
run dust1500-b64-e8k2s $BEST --batch-seqs 64 --microbatch-size 32 $M
echo "$(date +%T) queue7 done"
