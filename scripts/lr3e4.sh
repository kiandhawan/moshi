#!/bin/bash
# lr 3e-4 (10 h only): train_10h_v2_lr3e-4 (GPU 0) and train_10h_v2_lr3e-4_onsetw3 (GPU 1), started by
# hand in tmux windows. Watches both (scripts/check_divergence.py every 30 s; a diverged run is stopped
# and logged) and evaluates each on its GPU when trained (scripts/eval_run.py). Functions from
# scripts/lr1e4.sh. Log: logs/lr3e4.log; status per run: logs/lr1e4_<run>.status.
cd /root/moshi-finetune
LOG=logs/lr3e4.log
eval "$(sed -n '/^say()/,/^chain()/p' scripts/lr1e4.sh)"
say "watching train_10h_v2_lr3e-4 (GPU 0) and train_10h_v2_lr3e-4_onsetw3 (GPU 1)"
chain train_10h_v2_lr3e-4 0 &
chain train_10h_v2_lr3e-4_onsetw3 1 &
wait
say "both lr 3e-4 runs finished"
