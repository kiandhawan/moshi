#!/bin/bash
# Talk to a model in the browser: starts moshi.server on port 8998.
# Usage: scripts/talk.sh RUN|base|NAME [STEP] [GPU]   (GPU default 1)
#   RUN without STEP = the run's best checkpoint (results/best/RUN.json, scripts/eval_run.py).
#   Short names: onset3 = train_10h_v2_lr3e-5_onsetw3 step 150 (onset x3, best = final).
#   e.g. scripts/talk.sh base;  scripts/talk.sh onset3;  scripts/talk.sh train_1h_v2_lr3e-5 80 0
set -eu
cd /root/moshi-finetune
RUN=${1:?usage: scripts/talk.sh RUN|base|NAME [STEP] [GPU]}; STEP=${2:-}; GPU=${3:-1}
case $RUN in
  onset3) RUN=train_10h_v2_lr3e-5_onsetw3; STEP=${STEP:-150} ;;
esac
export HF_HUB_ENABLE_HF_TRANSFER=0 CUDA_VISIBLE_DEVICES=$GPU
ARGS=()
if [ "$RUN" = base ]; then
  echo "Loading the base model kyutai/moshiko-pytorch-bf16 (no LoRA) on GPU $GPU"
else
  if [ -z "$STEP" ]; then
    [ -f results/best/$RUN.json ] || { echo "No best checkpoint recorded for $RUN (results/best/$RUN.json); give a STEP"; exit 1; }
    STEP=$(.venv/bin/python -c "import json;print(json.load(open('results/best/$RUN.json'))['step'])")
    WHY="best checkpoint"
  else
    WHY="step $STEP (given)"
  fi
  CK=runs/$RUN/checkpoints/$(printf "checkpoint_%06d" $STEP)/consolidated
  [ -f $CK/lora.safetensors ] || CK=runs/$RUN/checkpoints_all/$(printf "checkpoint_%06d" $STEP)/consolidated
  [ -f $CK/lora.safetensors ] || { echo "No weights for $RUN step $STEP (pruned or not saved): $CK"; exit 1; }
  echo "Loading $RUN, $WHY: $CK (LoRA on kyutai/moshiko-pytorch-bf16) on GPU $GPU"
  ARGS=(--lora-weight $CK/lora.safetensors --config-path $CK/config.json)
fi
echo "Open port 8998 (e.g. the Runpod HTTP proxy for 8998) once it says it is ready."
exec .venv/bin/python -m moshi.server --host 0.0.0.0 --port 8998 --hf-repo kyutai/moshiko-pytorch-bf16 "${ARGS[@]}"
