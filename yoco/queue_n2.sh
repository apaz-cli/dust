#!/usr/bin/env bash
# Night 2. GPU 0 (RTX 5090) only, sequential. YOCO L4-D256, 3000 steps.
set -u
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
while pgrep -f "yoco/train.py" > /dev/null; do sleep 20; done
run() { name=$1; shift; out=runs/n2/$name; [ -f $out/step3000.pt ] && return; rm -rf $out; mkdir -p $out
  echo "$(date +%T) start $name"; $PY yoco/train.py --output $out --save-at 500 1000 1500 2000 2500 "$@" > $out.log 2>&1
  echo "$(date +%T) done $name: $(tail -1 $out.log)"; }
BEST="--precision fast16 --local-scale 4 --hub-scale 4 --emb-scale 4"
run best-fast       --arch yoco --method dust $BEST
run best-fast-wsd2  --arch yoco --method dust $BEST --wsd 0.2
run best-pool       --arch yoco --method dust $BEST --pool
run chain-a         --arch yoco --method dust --precision fast16 --chain 256 --local-scale 4 --emb-scale 4
run chain-hubs      --arch yoco --method dust --precision fast16 --chain 256 --chain-own --chain-hubs-only --local-scale 4 --emb-scale 4
run chain-hubs-wsd2 --arch yoco --method dust --precision fast16 --chain 256 --chain-own --chain-hubs-only --local-scale 4 --emb-scale 4 --wsd 0.2
run stock-fast      --arch yoco --method dust --precision fast16
run best-pool-wsd2  --arch yoco --method dust $BEST --pool --wsd 0.2
run alloc-stat      --arch yoco --method dust --precision fast16 --local-scale 4 --alloc '{"writers": [56, 24, 22, 26], "hidden": [68, 28, 28, 22], "hubs": [18, 4, 4, 2], "embedding": 66, "head": 1344}'
run best-ramp-wsd2  --arch yoco --method dust $BEST --draw-ramp --wsd 0.2
run quant-master    --arch yoco --method dust $BEST --quant master
run quant-sr        --arch yoco --method dust $BEST --quant sr
run best-linv       --arch yoco --method dust $BEST --linear-v
run stock-wsd2      --arch yoco --method dust --wsd 0.2
run alloc-stat-hub4 --arch yoco --method dust --precision fast16 --local-scale 4 --alloc '{"writers": [46, 20, 18, 22], "hidden": [56, 22, 22, 18], "hubs": [24, 24, 16, 8], "embedding": 54, "head": 1088}'
run best-linv-kvc   --arch yoco --method dust $BEST --linear-v --kv-credit
