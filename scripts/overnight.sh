#!/bin/bash
# Unattended: onset_w3 (GPU 0) and onset_w6 (GPU 1) - train, then scripts/eval_run.py - in parallel.
# Waits for the PAD-bias sweep first (it needs the GPUs). A failure stops only that run's chain and is
# logged (logs/overnight.log, status in logs/overnight_<run>.status). Touches logs/overnight.done at the end.
cd /root/moshi-finetune
LOG=logs/overnight.log
say() { echo "$(date +%H:%M:%S) $*" | tee -a $LOG; }
while pgrep -f "[p]ad_bias_sweep.sh" > /dev/null; do sleep 30; done
say "PAD-bias sweep finished; starting training"
chain() {  # RUN GPU PORT
  local R=$1 G=$2 P=$3 final
  final=runs/$R/checkpoints/checkpoint_000150/consolidated/lora.safetensors
  [ -d runs/$R ] && { say "$R: runs/$R exists, not starting"; echo "FAILED: run dir exists" > logs/overnight_$R.status; return; }
  tmux new-window -d -t setup -n $R "cd /root/moshi-finetune && CUDA_VISIBLE_DEVICES=$G .venv/bin/torchrun --nproc-per-node 1 --master_port $P -m train configs/$R.yaml 2>&1 | tee logs/$R.log; exec bash"
  say "$R: training started on GPU $G"
  sleep 60
  while pgrep -f "m train configs/$R.yaml" > /dev/null; do sleep 30; done
  if [ ! -f $final ]; then
    say "$R: TRAINING FAILED (no step-150 checkpoint); last log lines:"; tail -n 5 logs/$R.log | tee -a $LOG
    echo "FAILED: training (see logs/$R.log)" > logs/overnight_$R.status; return
  fi
  say "$R: trained; evaluating on GPU $G"
  if .venv/bin/python scripts/eval_run.py $R $G > logs/eval_run_$R.log 2>&1; then
    say "$R: evaluation done: $(grep -E '^BEST' logs/eval_run_$R.log | cut -c1-160)"
    echo "DONE" > logs/overnight_$R.status
  else
    say "$R: EVALUATION FAILED; last log lines:"; tail -n 5 logs/eval_run_$R.log | tee -a $LOG
    echo "FAILED: evaluation (see logs/eval_run_$R.log)" > logs/overnight_$R.status
  fi
}
chain train_10h_v2_lr3e-5_onsetw3 0 29611 &
chain train_10h_v2_lr3e-5_onsetw6 1 29613 &
wait
say "all chains finished"
touch logs/overnight.done
