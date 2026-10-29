#!/usr/bin/env bash
# Seeds 7 and 11 for backprop and noise-free arms (seed 42 already run). GPU 0 only.
set -u
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
for shape in "4 256 runs/ed/" "8 512 runs/ed/L8-"; do
  set -- $shape; L=$1; D=$2; P=$3
  for seed in 7 11; do for lr in 0.1 0.2; do
    for spec in "gpt bp x" "yoco bp x" "gpt 0.98 x" "gpt 1 x" "yoco 0.98 x" "yoco 1 x" "yoco 1 --handoff"; do
      set -- $spec; h=""; [ "$3" = "--handoff" ] && h="--handoff"
      if [ "$2" = bp ]; then name=${P}bp-$1-lr$lr-s$seed; m="--method bp"
      else name=${P}ed-$1-g$2${h:+-handoff}-lr$lr-s$seed; m="--method ed --gamma $2 $h"; fi
      [ -f $name/step3000.pt ] && continue; rm -rf $name
      $PY yoco/train.py --arch $1 $m --lr $lr --seed $seed --n-layer $L --n-embd $D --output $name > $name.log 2>&1
      echo "$name $(tail -1 $name.log | grep -o '"val_loss": [0-9.]*')"
    done
  done; done
done
