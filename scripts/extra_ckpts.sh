#!/bin/bash
# Behavioral tests for extra (non-candidate) checkpoints, exactly as scripts/eval_run.py tests candidates:
# 400 eval v2 files (seed 1, batch 32), human files and TTS twins (seeds 1 2 3), then score.py.
# They are reported as extra checkpoints; results/best/<run>.json is not changed.
# Usage: scripts/extra_ckpts.sh RUN GPU STEP [STEP ...]   Log: logs/extra_<run>.log
cd /root/moshi-finetune
RUN=$1; GPU=$2; shift 2; PY=.venv/bin/python; LOG=logs/extra_$RUN.log
for S in "$@"; do
  s=$(printf %03d $S); M=${RUN}__s$s; D=runs/$RUN/infer/s$s; L=runs/$RUN/checkpoints/checkpoint_000$s/consolidated
  echo "$(date +%T) $M: inference on GPU $GPU" >> $LOG
  $PY scripts/run_inference.py data/indexes/eval_v2.jsonl --lora $L --seeds 1 --batch 32 --gpu $GPU --out-dir $D/eval >> $LOG 2>&1 &&
  $PY scripts/run_inference.py data/indexes/human.jsonl --lora $L --seeds 1 2 3 --gpu $GPU --out-dir $D/human >> $LOG 2>&1 &&
  $PY scripts/run_inference.py data/indexes/human_tts.jsonl --lora $L --seeds 1 2 3 --gpu $GPU --out-dir $D/human_tts >> $LOG 2>&1 &&
  $PY scripts/score.py --outputs $D/eval $D/human --events data/indexes/eval_full_v2.jsonl data/indexes/human.jsonl --name $M --quiet >> $LOG 2>&1 &&
  $PY scripts/score.py --outputs $D/human_tts --events data/indexes/human_tts.jsonl --name $M --quiet --out-dir results/human_tts/metrics >> $LOG 2>&1 &&
  echo "$(date +%T) $M: DONE" >> $LOG || echo "$(date +%T) $M: FAILED" >> $LOG
done
echo "$(date +%T) ALL_DONE" >> $LOG
