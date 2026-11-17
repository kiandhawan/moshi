#!/bin/bash
# Controls at lr 1e-4 (1e-4 beat 3e-5 at both data sizes, results/LOG.md): control_10h_v2_lr1e-4 on
# GPU 0 now; control_1h_v2_lr1e-4 on GPU 1 once the onset x3 1e-4 chain is done (logs/lr1e4_onset.done).
# Same watch / evaluation as scripts/lr1e4.sh (functions sourced from it). Log: logs/lr1e4.log.
cd /root/moshi-finetune
LOG=logs/lr1e4.log
eval "$(sed -n '/^say()/,/^chain()/p' scripts/lr1e4.sh)"
start() {  # RUN GPU
  [ -d runs/$1 ] && { say "$1: run dir exists, not starting"; return 1; }
  tmux new-window -d -t setup -n $1 "cd /root/moshi-finetune && CUDA_VISIBLE_DEVICES=$2 .venv/bin/torchrun --nproc-per-node 1 --master_port $((29720 + $2)) -m train configs/$1.yaml 2>&1 | tee logs/$1.log; exec bash"
  say "$1: training started on GPU $2"; sleep 60; chain $1 $2
}
start control_10h_v2_lr1e-4 0 &
( until [ -f logs/lr1e4_onset.done ]; do sleep 30; done; start control_1h_v2_lr1e-4 1 ) &
wait
say "controls at lr 1e-4 finished"
