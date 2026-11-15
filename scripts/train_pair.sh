#!/bin/bash
# Train a pair: RUN_1H on GPU 0 and RUN_10H on GPU 1, each in its own tmux window (session
# "setup") with its log in logs/<run>.log; returns when both have their final checkpoint.
# Usage: scripts/train_pair.sh RUN_1H RUN_10H   (configs/<run>.yaml)
set -u
cd /root/moshi-finetune
RUNS="$1 $2"
port=$((29500 + RANDOM % 400))
for spec in "0 $1 $port" "1 $2 $((port + 1))"; do
  set -- $spec
  [ -d runs/$2 ] && { echo "runs/$2 exists, not starting it"; exit 1; }
  tmux new-window -d -t setup -n $2 "cd /root/moshi-finetune && CUDA_VISIBLE_DEVICES=$1 .venv/bin/torchrun --nproc-per-node 1 --master_port $3 -m train configs/$2.yaml 2>&1 | tee logs/$2.log; exec bash"
done
echo "=== $(date +%H:%M:%S) started $RUNS (GPU 0, GPU 1)"
sleep 30
for R in $RUNS; do
  steps=$(grep -E "^max_steps" configs/$R.yaml | awk '{print $2}')
  final=runs/$R/checkpoints/$(printf "checkpoint_%06d" $steps)/consolidated/lora.safetensors
  until [ -f $final ] && ! pgrep -f "m train configs/$R.yaml" > /dev/null; do
    pgrep -f "m train configs/$R.yaml" > /dev/null || [ -f $final ] || { echo "$R stopped without $final"; exit 1; }
    sleep 30
  done
  echo "=== $(date +%H:%M:%S) $R trained"
done
