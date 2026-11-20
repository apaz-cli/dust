#!/bin/bash
# NVFP4 verification, 1500 steps, MoE 8 experts width 512 (the scale_moe8_sgd setup), on the local GPUs directly.
# RTX 4070: reference without NVFP4. RTX 5090: NVFP4 linears + experts (emulated FP4), then NVFP4 linears only.
cd /home/apaz/git/dust_experiments
O=/home/apaz/git/dust_experiments/runs/nvfp4_dev
C=/home/apaz/.cache/torchinductor_dust
B="--arch yoco --seed 42 --data-dir data/fineweb_edu --n-layer 4 --n-embd 512 --steps 1500 --batch-seqs 64 --save-every 0 --local-scale 4 --microbatch-size 16 --method dust --precision fast16 --chain 256 --chain-own --chain-hubs-only --emb-scale 4 --wsd 0.2 --local-separate --cache-separate --live-fc --cv --moe 8,2,768,512 --moe-fix --router-cap 2 --router-wd 10"
A='--alloc {"writers":[152,128,64,40]}'
run() { gpu=$1; name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu systemd-run --user --scope -q -p MemoryMax=20G -E TORCHINDUCTOR_CACHE_DIR=$C \
    .venv/bin/python yoco/train.py $B $A "$@" --output "$O/v_$name" > "$O/v_$name.log" 2>&1
  echo "$name done: $(tail -n1 $O/v_$name/metrics.jsonl | cut -c1-80)"; }
run 1 base &
( run 0 fp4_all --nvfp4 --nvfp4-experts; run 0 fp4_lin --nvfp4 ) &
wait
echo all done
