#!/bin/bash
# Evaluate one training run on one GPU: inference with the final checkpoint (400 eval v2 files,
# seed 1, batches of 32; human files seeds 1 2 3), eval_loss.py on all 403 eval_full_v2 files for
# every saved checkpoint, then score.py (results/metrics/RUN.jsonl).
# Usage: scripts/eval_run.sh RUN GPU
set -u
cd /root/moshi-finetune
RUN=$1; GPU=$2
CK=runs/$RUN/checkpoints; [ -d runs/$RUN/checkpoints_all ] && CK=runs/$RUN/checkpoints_all
FINAL=$(ls -d $CK/checkpoint_*/consolidated | sort | tail -n 1)
echo "=== $(date +%H:%M:%S) $RUN: inference with $FINAL on GPU $GPU"
SKIP_HUMAN=0 scripts/infer_model.sh $RUN $FINAL "$GPU"
for c in $(ls -d $CK/checkpoint_*/consolidated | sort); do
  echo "=== $(date +%H:%M:%S) $RUN: eval_loss $c"
  .venv/bin/python scripts/eval_loss.py --checkpoint $c --gpu $GPU > logs/eval_loss_${RUN}_$(basename $(dirname $c)).log 2>&1 \
    || echo "eval_loss FAILED for $c (see log)"
done
echo "=== $(date +%H:%M:%S) $RUN: scoring"
.venv/bin/python scripts/score.py --outputs runs/$RUN/infer/eval runs/$RUN/infer/human \
  --events data/indexes/eval_full_v2.jsonl data/indexes/human.jsonl --name $RUN --quiet > logs/score_$RUN.log 2>&1
echo "=== $(date +%H:%M:%S) $RUN: evaluation done"
