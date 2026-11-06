#!/usr/bin/env bash
# Track B step 6: fixed 1500 steps, more tokens per step (32 and 48 sequences = 4x and 6x the tokens).
# Data: tinystories-long (72k rows; same validation set, first 24k rows = data/tinystories). 48 x 1500 uses
# every row once; 64 x 1500 would need 96k. The 8-sequence references are rerun on the same data at the end.
# Does Dust-MoE pull ahead of dense once the draw-noise limit is relaxed? RTX 4070 only.
set -u
export CUDA_VISIBLE_DEVICES=1
PY=.venv/bin/python
D=runs/moe
run() { name=$1; shift; out=$D/$name; [ -f $out/step1500.pt ] && return; rm -rf "${out:?}"; mkdir -p $out
  echo "$(date +%T) start $name"; $PY yoco/train.py --output $out --steps 1500 --save-at 1500 "$@" > $out.log 2>&1
  echo "$(date +%T) done $name: $(tail -1 $out.log)"; }
M="--moe 8,2,384,256"
BEST="--arch yoco --method dust --precision fast16 --chain 256 --chain-own --chain-hubs-only --local-scale 4 --emb-scale 4 --wsd 0.2 --data-dir /home/apaz/git/dust/data/tinystories-long"
run dust1500-b32-dense $BEST --batch-seqs 32
run dust1500-b32-e8k2s $BEST --batch-seqs 32 $M
run dust1500-b48-dense $BEST --batch-seqs 48 --microbatch-size 24
run dust1500-b48-e8k2s $BEST --batch-seqs 48 --microbatch-size 24 $M
run dust1500-b8L-dense $BEST
run dust1500-b8L-e8k2s $BEST $M
echo "$(date +%T) queue7 done"
