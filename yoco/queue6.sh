#!/usr/bin/env bash
# GPU 0 only. Noise PR #5 live-unit c_fc jitter and PR #6 head-error-keyed control variate, on the best
# derivative-free YOCO recipe (fp32 + 4x local + 4x hub + 4x embedding, 2.777), and at reduced early draws.
set -u
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
while pgrep -f "bash yoco/queue[2345].sh" > /dev/null || pgrep -f "yoco/train.py" > /dev/null; do sleep 20; done
run() { name=$1; shift; out=runs/q6/$name; [ -f $out/step3000.pt ] && return; rm -rf $out; mkdir -p $out
  echo "$(date +%T) start $name"; $PY yoco/train.py --output $out "$@" > $out.log 2>&1; echo "$(date +%T) done $name: $(tail -1 $out.log)"; }
BEST="--arch yoco --method dust --fp32 --local-scale 4 --hub-scale 4 --emb-scale 4"
# half the draws at blocks 0-1 and the embedding (the full-forward sites)
HALF='{"writers": [20, 16, 16, 10], "hidden": [20, 16, 16, 10], "embedding": 20}'
run best-live-cv      $BEST --live-fc --cv
run best-live-cv-half $BEST --live-fc --cv --alloc "$HALF"
run best-cv           $BEST --cv
run best-live         $BEST --live-fc
run stock-live-cv     --arch yoco --method dust --live-fc --cv
