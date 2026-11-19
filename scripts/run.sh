#!/bin/bash
# Train one config and evaluate it: torchrun train.py in a tmux window (log logs/<run>.log, copied to
# runs/<run>/train.log), stop it if it diverges (scripts/check_divergence.py every 30 s), then
# scripts/eval_run.py on the same GPU (loss eval of every checkpoint, top 3 by whole-wait silence + final,
# inference + scoring, best checkpoint, pruning, tables and plots). Status: logs/run_<run>.log.
# Usage: scripts/run.sh RUN GPU      (configs/RUN.yaml; run two at once on GPUs 0 and 1)
set -u
cd /root/moshi-finetune
R=$1; G=$2; LOG=logs/run_$R.log; PY=.venv/bin/python
say() { echo "$(date +%H:%M:%S) $*" | tee -a $LOG; }
tmux new-window -d -t setup -n $R "cd /root/moshi-finetune && CUDA_VISIBLE_DEVICES=$G .venv/bin/torchrun --nproc-per-node 1 --master_port $((29740 + G)) -m train configs/$R.yaml 2>&1 | tee logs/$R.log; exec bash"
say "started $R on GPU $G (tmux window $R)"
sleep 60
while pgrep -f "[m] train configs/$R.yaml" > /dev/null; do
  c=$($PY scripts/check_divergence.py $R); [ $? -eq 2 ] && { say "$c - STOPPING"; pkill -f "[m] train configs/$R.yaml"; exit 1; }
  sleep 30
done
say "$($PY scripts/check_divergence.py $R)"
cp logs/$R.log runs/$R/train.log
ls runs/$R/checkpoints/checkpoint_*/consolidated/lora.safetensors > /dev/null 2>&1 || { say "TRAINING FAILED"; exit 1; }
say "evaluating on GPU $G"
if $PY scripts/eval_run.py $R $G > logs/eval_run_$R.log 2>&1; then say "$(grep -E '^BEST' logs/eval_run_$R.log)"
else say "EVALUATION FAILED (logs/eval_run_$R.log)"; exit 1; fi
say "disk free: $(df -BG --output=avail / | tail -1 | tr -d ' ')"
