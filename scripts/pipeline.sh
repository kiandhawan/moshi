#!/bin/bash
# Pairs 1-3 of the grid: when both runs of a pair have their step-250 checkpoint, start the next
# pair's training (GPU 0 = 1 h, GPU 1 = 10 h; own tmux windows) and evaluate the finished pair on
# the same GPUs alongside it. Pair 4 (controls) is started by hand after the lr choice.
# Usage: scripts/pipeline.sh   (pair 1 already started)
set -u
cd /root/moshi-finetune
LRS=(1e-5 3e-5 2e-6)
start_pair() {  # lr
  local lr=$1 port=$((29500 + RANDOM % 400))
  for spec in "0 train_1h_v2_lr$lr t1h_$lr $port" "1 train_10h_v2_lr$lr t10h_$lr $((port + 1))"; do
    set -- $spec
    tmux new-window -d -t setup -n $3 "cd /root/moshi-finetune && CUDA_VISIBLE_DEVICES=$1 .venv/bin/torchrun --nproc-per-node 1 --master_port $4 -m train configs/$2.yaml 2>&1 | tee logs/$2.log; exec bash"
  done
  echo "=== $(date +%H:%M:%S) started pair lr $lr"
}
done_run() { [ -f runs/$1/checkpoints/checkpoint_000250/consolidated/lora.safetensors ] && ! pgrep -f "configs/$1.yaml" > /dev/null; }
for i in 0 1 2; do
  lr=${LRS[$i]}
  until done_run train_1h_v2_lr$lr && done_run train_10h_v2_lr$lr; do sleep 60; done
  echo "=== $(date +%H:%M:%S) pair lr $lr trained"
  [ $i -lt 2 ] && start_pair ${LRS[$((i + 1))]}
  scripts/eval_run.sh train_1h_v2_lr$lr 0 > logs/eval_run_train_1h_v2_lr$lr.log 2>&1 &
  scripts/eval_run.sh train_10h_v2_lr$lr 1 > logs/eval_run_train_10h_v2_lr$lr.log 2>&1 &
  wait
  echo "=== $(date +%H:%M:%S) pair lr $lr evaluated"
done
