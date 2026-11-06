#!/usr/bin/env bash
# Next step on context use (copy task, data/tsrepeat). Starts when the 4070 queue (queue_moe7) is done, then runs
# on the RTX 5090 (index 0) only. References (L4, night 5): bp 1.104 (second half 0.049); noise-free g1 1.169 (0.090);
# old best recipe 2.752 (2.645); leak-free self-decoder, local x1: g.98 2.649 (2.490), g1 2.035 (0.983).
set -u
until grep -q "queue7 done" runs/moe.log; do sleep 60; done
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
D=runs/n6
run() { name=$1; shift; out=$D/$name; [ -f $out/step3000.pt ] && return; rm -rf "${out:?}"; mkdir -p $out
  echo "$(date +%T) start $name"; $PY yoco/train.py --output $out "$@" > $out.log 2>&1
  echo "$(date +%T) done $name: $(tail -1 $out.log)"; }
# 1. cache estimator: noise cut of leak-free scoring (bp-trained copy model, exact own-token o@ targets)
echo "$(date +%T) bench cache"; $PY yoco/bench_separate.py runs/n5/rep-bp 1024 cache > $D/bench_cache.log 2>&1
echo "$(date +%T) bench cache done: $(grep -c cut $D/bench_cache.log) rows"
R="--arch yoco --data-dir data/tsrepeat --method dust --precision fast16 --chain 256 --chain-own --chain-hubs-only --emb-scale 4 --wsd 0.2 --local-separate"
# 2. leak-free cache too; which sites need the long horizon?
run rep-sc-g1        $R --local-scale 1 --cache-separate --gamma 1
run rep-sc-g98       $R --local-scale 1 --cache-separate
run rep-sc-g1self    $R --local-scale 1 --cache-separate --gamma 1 --cache-gamma 0.98
run rep-sc-g1cache   $R --local-scale 1 --cache-separate --cache-gamma 1
# 3. confirm: second seed of the gamma-1 config, then L8 with references
run rep-sc-g1-s7     $R --local-scale 1 --cache-separate --gamma 1 --seed 7
L8="--n-layer 8 --n-embd 512"
run rep-L8-bp        --arch yoco --data-dir data/tsrepeat --method bp --lr 0.1 $L8
run rep-L8-ed-g1     --arch yoco --data-dir data/tsrepeat --method ed --gamma 1 --lr 0.1 $L8
run rep-L8-sc-g1     $R --local-scale 4 --cache-separate --gamma 1 $L8
$PY yoco/halves.py runs/n5/rep-bp runs/n5/rep-sep-g1 $D/rep-sc-g1 $D/rep-sc-g98 $D/rep-sc-g1self $D/rep-sc-g1cache \
    $D/rep-sc-g1-s7 $D/rep-L8-bp $D/rep-L8-ed-g1 $D/rep-L8-sc-g1
echo "$(date +%T) queue n6 done"
