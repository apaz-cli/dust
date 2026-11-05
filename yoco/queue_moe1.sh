#!/usr/bin/env bash
# Night 5, track B step 1: does MoE beat dense under backprop at L4-D256? RTX 4070 (index 1) only.
set -u
export CUDA_VISIBLE_DEVICES=1
PY=.venv/bin/python
D=runs/moe
run() { name=$1; shift; out=$D/$name; [ -f $out/step3000.pt ] && return; rm -rf "${out:?}"; mkdir -p $out
  echo "$(date +%T) start $name"; $PY yoco/train.py --output $out "$@" > $out.log 2>&1
  echo "$(date +%T) done $name: $(tail -1 $out.log)"; }
B="--arch yoco --method bp --lr 0.2"
for s in 42 7; do
  run dense-s$s          $B --seed $s
  run e8k2-s$s           $B --seed $s --moe 8,2,512,0
  run e16k2s-s$s         $B --seed $s --moe 16,2,256,512
  run e32k4s-s$s         $B --seed $s --moe 32,4,128,512
  run e64k4s-s$s         $B --seed $s --moe 64,4,128,512
  run e16k2s-lr4-s$s     $B --seed $s --moe 16,2,256,512 --moe-lr-mult 4
  run e64k4s-lr4-s$s     $B --seed $s --moe 64,4,128,512 --moe-lr-mult 4
done
echo "$(date +%T) queue done"
