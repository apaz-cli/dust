#!/usr/bin/env bash
# Noise-free expected-DUST training, L4-D256. GPU 0 only.
set -u
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
for lr in 0.1 0.2; do
  for spec in "gpt 0 x" "gpt 0.98 x" "gpt 0.995 x" "gpt 0.999 x" "gpt 1 x" "yoco 0.98 x" "yoco 1 x" "yoco 1 --handoff" "yoco 0 x"; do
    set -- $spec; h=""; [ "$3" = "--handoff" ] && h="--handoff"
    name=ed-$1-g$2${h:+-handoff}-lr$lr; out=runs/ed/$name
    [ -f $out/step3000.pt ] && continue; rm -rf $out
    $PY yoco/train.py --arch $1 --method ed --gamma $2 $h --lr $lr --output $out > $out.log 2>&1
    echo "$name $(tail -1 $out.log | grep -o '"val_loss": [0-9.]*')"
  done
done
