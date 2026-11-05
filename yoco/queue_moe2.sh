#!/usr/bin/env bash
# Track B step 1b: MoE vs dense under backprop with AdamW (the usual MoE setting). RTX 4070 only.
# SGD (step 1a) had every MoE behind dense; this checks whether the optimizer is the reason.
set -u
export CUDA_VISIBLE_DEVICES=1
until grep -q "queue done" runs/moe.log; do sleep 20; done
PY=.venv/bin/python
D=runs/moe
run() { name=$1; shift; out=$D/$name; [ -f $out/step3000.pt ] && return; rm -rf "${out:?}"; mkdir -p $out
  echo "$(date +%T) start $name"; $PY yoco/train.py --output $out "$@" > $out.log 2>&1
  echo "$(date +%T) done $name: $(tail -1 $out.log)"; }
B="--arch yoco --method bp --opt adamw"
for lr in 1e-3 3e-3; do
  run adam$lr-dense   $B --adam-lr $lr
  run adam$lr-e8k2    $B --adam-lr $lr --moe 8,2,512,0
  run adam$lr-e16k2s  $B --adam-lr $lr --moe 16,2,256,512
  run adam$lr-e32k4s  $B --adam-lr $lr --moe 32,4,128,512
done
echo "$(date +%T) queue2 done"
