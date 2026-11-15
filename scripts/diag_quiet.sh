#!/bin/bash
# Diagnostic (results/LOG.md "2e-6 quietness check"): the 40-file eval subset (eval_full_subset_v2:
# 37 eval v2 files + the 3 human files) through the base model and every kept checkpoint, all in
# one batch of 40 at seed 1 (same batch composition for every model), plus --batch 1 runs of base
# and train_1h_v2_lr2e-6 step 50. Outputs runs/diag40/<model>/, scores results/diagnostics/quiet40/.
# Usage: scripts/diag_quiet.sh GPU MODEL...   (MODEL = base | RUN__sNNN | <either>__b1 for batch 1)
cd /root/moshi-finetune
GPU=$1; shift
IX=data/indexes/eval_full_subset_v2.jsonl
for M in "$@"; do
  B=40; MM=$M
  case $M in *__b1) B=1; MM=${M%__b1};; esac
  LORA=()
  if [ "$MM" != base ]; then
    R=${MM%__s*}; S=${MM##*__s}
    LORA=(--lora runs/$R/checkpoints/checkpoint_000$S/consolidated)
  fi
  [ -f results/diagnostics/quiet40/$M.jsonl ] && continue
  .venv/bin/python scripts/run_inference.py $IX "${LORA[@]}" --seeds 1 --batch $B --gpu $GPU --out-dir runs/diag40/$M \
    > logs/diag40_$M.log 2>&1 || { echo "FAILED $M"; continue; }
  .venv/bin/python scripts/score.py --outputs runs/diag40/$M --events $IX --name $M --quiet \
    --out-dir results/diagnostics/quiet40 >> logs/diag40_$M.log 2>&1 && echo "$(date +%H:%M:%S) done $M"
done
