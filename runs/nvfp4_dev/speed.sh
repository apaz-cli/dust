#!/bin/bash
# Speed on the RTX 5090 with the GPU to itself: standard configuration vs NVFP4 (linear layers) vs NVFP4 + MXFP4
# experts. 40 steps each, time per step over steps 10-40. MoE (8 experts) and dense, widths 512 and 1024.
cd /home/apaz/git/dust_experiments
O=/home/apaz/git/dust_experiments/runs/nvfp4_dev/speed
C=/home/apaz/.cache/torchinductor_dust
mkdir -p "$O"
COMMON="--arch yoco --seed 42 --data-dir data/fineweb_edu --n-layer 4 --steps 40 --eval-every 1000 --save-every 0 --local-scale 4 --microbatch-size 16 --method dust --precision fast16 --chain 256 --chain-own --chain-hubs-only --emb-scale 4 --wsd 0.2 --local-separate --cache-separate --live-fc --cv"
FIX="--moe-fix --router-cap 2 --router-wd 10"
declare -A SETUP
SETUP[moe_w512]="--n-embd 512 --batch-seqs 64 --moe 8,2,768,512 $FIX"
SETUP[dense_w512]="--n-embd 512 --batch-seqs 64"
SETUP[moe_w1024]="--n-embd 1024 --batch-seqs 32 --moe 8,2,1536,1024 $FIX"
SETUP[dense_w1024]="--n-embd 1024 --batch-seqs 32"
for s in moe_w512 dense_w512 moe_w1024 dense_w1024; do
  for v in "std" "fp4 --nvfp4" "fp4mx --nvfp4 --nvfp4-experts"; do
    set -- $v; n=$1; shift
    case $s in dense*) [ "$n" = fp4mx ] && continue;; esac
    CUDA_VISIBLE_DEVICES=0 systemd-run --user --scope -q -p MemoryMax=24G -E TORCHINDUCTOR_CACHE_DIR=$C \
      .venv/bin/python yoco/train.py $COMMON ${SETUP[$s]} "$@" --output "$O/${s}_$n" > "$O/${s}_$n.log" 2>&1
    rm -f "$O/${s}_$n"/*.pt
    python3 - "$O/${s}_$n/metrics.jsonl" "$s" "$n" <<'PY'
import json, sys
try:
    rs = {r['step']: r for r in (json.loads(l) for l in open(sys.argv[1]))}
    print(f"{sys.argv[2]:12s} {sys.argv[3]:6s} {(rs[40]['seconds'] - rs[10]['seconds']) / 30:7.2f} s/step   loss@40 {rs[40]['train_loss']:.4f}", flush=True)
except Exception as e:
    print(sys.argv[2], sys.argv[3], 'FAILED', e, flush=True)
PY
  done
done
