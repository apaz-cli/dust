#!/usr/bin/env bash
# Night 3: scaling ladder L4-D256 / L8-D512 / L12-D768. GPU 0 (RTX 5090) only, sequential.
set -u
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
D=runs/n3
L12="--arch yoco --n-layer 12 --n-embd 768"
run() { name=$1; shift; out=$D/$name; [ -f $out/step3000.pt ] && return; rm -rf "${out:?}"; mkdir -p $out
  echo "$(date +%T) start $name"; $PY yoco/train.py --output $out "$@" > $out.log 2>&1
  echo "$(date +%T) done $name: $(tail -1 $out.log)"; }
bias() { tag=$1; shift; echo "$(date +%T) bias $tag"; $PY yoco/measure_bias.py --sequences 2 "$@" > $D/bias-$tag.log 2>&1
  echo "$(date +%T) bias $tag done"; }

# 1. exact bias and leakage (own-token and gamma-discounted) on existing checkpoints
bias L4-bp         --run runs/bpsweep/yoco-0.1
bias L4-bp-g98     --run runs/bpsweep/yoco-0.1 --gamma 0.98
bias L8-bp         --run runs/bpsweep/L8-yoco-0.1
bias L8-bp-g98     --run runs/bpsweep/L8-yoco-0.1 --gamma 0.98
bias L4-dust-g98   --run runs/n2/chain-hubs-wsd2 --gamma 0.98
bias L8-dust-g98   --run runs/n2/L8-best-zo --gamma 0.98

# 2. L12 backprop and noise-free Dust (bias ruler), two learning rates
run L12-bp-0.1      $L12 --method bp --lr 0.1
run L12-bp-0.05     $L12 --method bp --lr 0.05
run L12-ed-g1-0.1   $L12 --method ed --gamma 1 --lr 0.1
run L12-ed-g98-0.1  $L12 --method ed --gamma 0.98 --lr 0.1
run L12-ed-g1-0.05  $L12 --method ed --gamma 1 --lr 0.05
run L12-ed-g98-0.05 $L12 --method ed --gamma 0.98 --lr 0.05
bias L12-bp        --run $D/L12-bp-0.1
bias L12-bp-g98    --run $D/L12-bp-0.1 --gamma 0.98

# 3. L12 real Dust (lr 0.25 diverges by step 9 at L12; 0.15 is stable in a 40-step check)
run L12-best-zo    $L12 --method dust --lr 0.15 --precision fast16 --chain 256 --chain-own --chain-hubs-only --local-scale 4 --emb-scale 4 --wsd 0.2
run L12-stock-fast $L12 --method dust --lr 0.15 --precision fast16
bias L12-dust-g98  --run $D/L12-best-zo --gamma 0.98
echo "$(date +%T) queue done"
