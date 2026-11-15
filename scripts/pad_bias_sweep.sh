#!/bin/bash
# Decoding-time silence penalty sweep (results/LOG.md): MODEL on the 40-file diagnostic set
# (eval_full_subset_v2, one batch of 40, seed 1) with run_inference.py --pad-bias B for each B.
# Bias 0 = the existing diagnostic run (results/diagnostics/quiet40/MODEL.jsonl, same settings).
# Outputs runs/diag40_padbias/MODEL__pbB/, scores results/diagnostics/pad_bias/MODEL__pbB.jsonl.
# Usage: scripts/pad_bias_sweep.sh GPU MODEL B...
cd /root/moshi-finetune
GPU=$1; M=$2; shift 2
IX=data/indexes/eval_full_subset_v2.jsonl
R=${M%__s*}; S=${M##*__s}
for B in "$@"; do
  N=${M}__pb$B
  [ -f results/diagnostics/pad_bias/$N.jsonl ] && continue
  .venv/bin/python scripts/run_inference.py $IX --lora runs/$R/checkpoints/checkpoint_000$S/consolidated --seeds 1 --batch 40 \
    --pad-bias $B --gpu $GPU --out-dir runs/diag40_padbias/$N > logs/padbias_$N.log 2>&1 \
  && .venv/bin/python scripts/score.py --outputs runs/diag40_padbias/$N --events $IX --name $N --quiet \
    --out-dir results/diagnostics/pad_bias >> logs/padbias_$N.log 2>&1 && echo "$(date +%H:%M:%S) done $N" || echo "FAILED $N"
done
