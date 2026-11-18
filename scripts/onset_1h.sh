#!/bin/bash
# 1h x onset (2x2 with the 10h runs): train_1h_v2_lr1e-4_onsetw3 (GPU 0, started first) and
# train_1h_v2_lr3e-5_onsetw3 (GPU 1). Each trains in its own tmux window (log: logs/<run>.log, linked as
# runs/<run>/train.log once the run dir exists and copied there at the end), is watched for divergence
# (scripts/check_divergence.py) and then evaluated on its GPU by scripts/eval_run.py (loss eval, top 3 +
# final, inference + scoring, best checkpoint, pruning). Helpers from scripts/lr1e4.sh. Log: logs/onset_1h.log.
cd /root/moshi-finetune
LOG=logs/onset_1h.log
eval "$(sed -n '/^say()/,/^chain()/p' scripts/lr1e4.sh)"
start() {  # GPU RUN
  tmux new-window -d -t setup -n $2 "cd /root/moshi-finetune && CUDA_VISIBLE_DEVICES=$1 .venv/bin/torchrun --nproc-per-node 1 --master_port $((29740 + $1)) -m train configs/$2.yaml 2>&1 | tee logs/$2.log; exec bash"
  say "started $2 on GPU $1 (tmux window $2)"
  for i in $(seq 120); do [ -d runs/$2 ] && break; sleep 2; done
  ln -sfn ../../logs/$2.log runs/$2/train.log
}
run() {  # GPU RUN
  start $1 $2; sleep 20; chain $2 $1
  [ -L runs/$2/train.log ] && { rm runs/$2/train.log; cp logs/$2.log runs/$2/train.log; }
}
run 0 train_1h_v2_lr1e-4_onsetw3 &
sleep 60
run 1 train_1h_v2_lr3e-5_onsetw3 &
wait
say "both 1h onset runs finished"
