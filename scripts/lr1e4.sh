#!/bin/bash
# lr 1e-4 grid point (unattended). train_1h_v2_lr1e-4 (GPU 0) and train_10h_v2_lr1e-4 (GPU 1) were
# started by hand in tmux windows; this watches them (scripts/check_divergence.py every 30 s: a
# diverged run is stopped and logged), evaluates each on its GPU when trained (scripts/eval_run.py),
# and on the first GPU to free starts train_10h_v2_lr1e-4_onsetw3 (+ its evaluation) in the
# background. Returns when both 1e-4 runs are evaluated (control decision is made after that).
# Log: logs/lr1e4.log; status per run: logs/lr1e4_<run>.status.
cd /root/moshi-finetune
LOG=logs/lr1e4.log
say() { echo "$(date +%H:%M:%S) $*" | tee -a $LOG; }
watch_train() {  # RUN -> 0 trained, 1 failed/diverged
  local R=$1 final=runs/$1/checkpoints/checkpoint_000150/consolidated/lora.safetensors c
  while pgrep -f "[m] train configs/$R.yaml" > /dev/null; do
    c=$(.venv/bin/python scripts/check_divergence.py $R); [ $? -eq 2 ] && {
      say "$c - STOPPING $R"; pkill -f "[m] train configs/$R.yaml"; sleep 10
      echo "DIVERGED: $c" > logs/lr1e4_$R.status; return 1; }
    sleep 30
  done
  say "$R: $(.venv/bin/python scripts/check_divergence.py $R)"
  [ -f $final ] || { say "$R: TRAINING FAILED (no step-150 checkpoint)"; tail -n 5 logs/$R.log >> $LOG
                     echo "FAILED: training" > logs/lr1e4_$R.status; return 1; }
}
evaluate() {  # RUN GPU
  say "$1: trained; evaluating on GPU $2"
  if .venv/bin/python scripts/eval_run.py $1 $2 > logs/eval_run_$1.log 2>&1; then
    say "$1: evaluation done: $(grep -E '^BEST' logs/eval_run_$1.log | cut -c1-200)"; echo DONE > logs/lr1e4_$1.status
  else
    say "$1: EVALUATION FAILED"; tail -n 5 logs/eval_run_$1.log >> $LOG; echo "FAILED: evaluation" > logs/lr1e4_$1.status
  fi
  say "disk free: $(df -BG --output=avail / | tail -1 | tr -d ' ')"
}
chain() { watch_train $1 && evaluate $1 $2; }
onset() {  # GPU
  local R=train_10h_v2_lr1e-4_onsetw3
  [ -d runs/$R ] && { say "$R: run dir exists, not starting"; return; }
  tmux new-window -d -t setup -n $R "cd /root/moshi-finetune && CUDA_VISIBLE_DEVICES=$1 .venv/bin/torchrun --nproc-per-node 1 --master_port $((29700 + $1)) -m train configs/$R.yaml 2>&1 | tee logs/$R.log; exec bash"
  say "$R: training started on GPU $1"; sleep 60
  chain $R $1; say "$R: chain finished"; touch logs/lr1e4_onset.done
}
say "watching train_1h_v2_lr1e-4 (GPU 0) and train_10h_v2_lr1e-4 (GPU 1)"
chain train_1h_v2_lr1e-4 0 & P0=$!
chain train_10h_v2_lr1e-4 1 & P1=$!
wait -n $P0 $P1
if kill -0 $P0 2>/dev/null; then G=1; else G=0; fi
say "GPU $G is free first"
setsid nohup bash -c "$(declare -f say watch_train evaluate chain onset); LOG=$LOG; onset $G" >> logs/lr1e4_onset.out 2>&1 < /dev/null &
wait
say "both lr 1e-4 runs finished (onset x3 continues on GPU $G)"
