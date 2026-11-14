#!/bin/bash
# Inference for one model on the 400 eval v2 files (seed 1, batches of 32 streams) and the human files (seeds 1 2 3).
# Usage: scripts/infer_model.sh NAME LORA_DIR|- "GPU GPU ..."   (one worker per listed GPU entry;
#        list a GPU several times to run several workers on it, e.g. "0 0 0 1 1 1")
# Outputs: runs/NAME/infer/{eval,human}/<id>_seed<N>.{wav,json}; logs in logs/infer_NAME_<k>.log.
# Resumable: existing outputs are skipped.
set -u
cd /root/moshi-finetune
NAME=$1; LORA=$2; GPUS=($3); N=${#GPUS[@]}
LORA_ARG=(); [ "$LORA" != "-" ] && LORA_ARG=(--lora "$LORA")
OUT=runs/$NAME/infer
for k in $(seq 0 $((N - 1))); do
  (
    .venv/bin/python scripts/run_inference.py data/indexes/eval_v2.jsonl "${LORA_ARG[@]}" --seeds 1 --batch 32 \
      --gpu ${GPUS[$k]} --shard $k/$N --out-dir $OUT/eval
    if [ $k -eq 0 ] && [ "${SKIP_HUMAN:-0}" != 1 ]; then
      .venv/bin/python scripts/run_inference.py data/indexes/human.jsonl "${LORA_ARG[@]}" --seeds 1 2 3 \
        --gpu ${GPUS[$k]} --out-dir $OUT/human
    fi
  ) > logs/infer_${NAME}_$k.log 2>&1 &
done
wait
echo "=== $(date +%H:%M:%S) inference done: $NAME ($(ls $OUT/eval/*.json | wc -l) eval, $(ls $OUT/human/*.json | wc -l) human)"
