#!/usr/bin/env bash
# Night 4, phase 1: where does draw noise cost loss? L4 best ZO recipe, one site group noise-free per run
# (oracle = expected-Dust update for that group; diagnostic only). GPU 0 (RTX 5090) only, sequential.
set -u
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
D=runs/n4
BEST="--arch yoco --method dust --precision fast16 --chain 256 --chain-own --chain-hubs-only --local-scale 4 --emb-scale 4 --wsd 0.2"
run() { name=$1; shift; out=$D/$name; [ -f $out/step3000.pt ] && return; rm -rf "${out:?}"; mkdir -p $out
  echo "$(date +%T) start $name"; $PY yoco/train.py --output $out "$@" > $out.log 2>&1
  echo "$(date +%T) done $name: $(tail -1 $out.log)"; }
run or-selfattn  $BEST --oracle-re 'h\.[01]\.attn\.(c_q|c_k|c_v|ve_gate)|value_embeds'
run or-cache     $BEST --oracle-re 'cache_[kv]'
run or-selfwrite $BEST --oracle-re 'h\.[01]\.(attn|mlp)\.c_proj'
run or-selfhid   $BEST --oracle-re 'h\.[01]\.mlp\.c_fc'
run or-wte       $BEST --oracle-re 'transformer\.wte'
run or-cross     $BEST --oracle-re 'h\.[23]\.'
run or-head      $BEST --oracle-re 'lm_head'
echo "$(date +%T) queue done"
