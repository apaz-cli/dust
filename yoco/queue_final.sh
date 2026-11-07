#!/usr/bin/env bash
# Final confirmation runs for the write-up. RTX 5090 (index 0) only; the 4070 stays off at night (200 W limit).
# A: final TinyStories recipe = best + leak-free self-decoder and cache scoring, gamma .98.
#    References: L4 best recipe 2.733 / 2.749; L4 leak-free self only, local x1 2.683 / 2.692; L8 best 2.918,
#    L8 leak-free self, local x4 2.842.
# B: final copy-task config = gamma 1 in the self-decoder, .98 on the cache. Reference L4 seed 42: 1.552
#    (second half 0.156). L8 at gamma 1 everywhere and lr .25 copied badly (1.46) and was unstable; lower lr here.
set -u
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
D=runs/final
run() { name=$1; shift; out=$D/$name; [ -f $out/step3000.pt ] && return; rm -rf "${out:?}"; mkdir -p $out
  echo "$(date +%T) start $name"; $PY yoco/train.py --output $out "$@" > $out.log 2>&1
  echo "$(date +%T) done $name: $(tail -1 $out.log)"; }
F="--arch yoco --method dust --precision fast16 --chain 256 --chain-own --chain-hubs-only --emb-scale 4 --wsd 0.2 --local-separate --cache-separate"
L8="--n-layer 8 --n-embd 512"
run ts-L4-s42   $F --local-scale 1
run ts-L4-s7    $F --local-scale 1 --seed 7
run copy-L4-s7  $F --local-scale 1 --data-dir data/tsrepeat --gamma 1 --cache-gamma 0.98 --seed 7
run ts-L8       $F --local-scale 4 $L8
run copy-L8-lr15 $F --local-scale 4 --data-dir data/tsrepeat --gamma 1 --cache-gamma 0.98 $L8 --lr 0.15
run copy-L8-lr10 $F --local-scale 4 --data-dir data/tsrepeat --gamma 1 --cache-gamma 0.98 $L8 --lr 0.1
$PY yoco/halves.py runs/n6/rep-sc-g1self $D/copy-L4-s7 runs/n6/rep-L8-bp runs/n6/rep-L8-ed-g1 runs/n6/rep-L8-sc-g1 \
    $D/copy-L8-lr15 $D/copy-L8-lr10
echo "$(date +%T) queue final done"
