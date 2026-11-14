#!/usr/bin/env bash
# Render the v2 datasets (see results/LOG.md). Safe to rerun: clips are cached.
# Train and control: right-channel augmentation (render.py --augment), clean copies in
# data/v2/clean/ for verification. Eval v2: no augmentation (360 v1 files re-rendered from
# the cache + 40 other_name files). Usage (from the repo root, inside tmux): bash scripts/render_all_v2.sh
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ~/.config/fish_audio/env; set +a     # FISH_API_KEY, never printed
mkdir -p logs
run() {  # name, then render.py arguments
  name=$1; shift
  echo "=== $(date -u +%H:%M:%S) rendering $name" | tee -a logs/render_all_v2.log
  .venv/bin/python scripts/render.py --name "$name" --workers 12 "$@" 2>&1 \
    | grep --line-buffered -v '^\[0.0' | tee "logs/render_$name.log"
}
run train_all_v2   --scripts data/v2/scripts/train   --out data/v2/train   --augment --clean-out data/v2/clean/train
run control_all_v2 --scripts data/v2/scripts/control --out data/v2/control --augment --clean-out data/v2/clean/control
rm -f data/indexes/eval_full_v2.jsonl
run eval_v2        --scripts data/v2/scripts/eval    --out data/v2/eval    --full-dir data/v2/eval_full --full-index eval_full_v2
# the 3 human answer keys (unchanged) belong to eval_full v2 as well
grep '"../eval_full/human_' data/indexes/eval_full.jsonl >> data/indexes/eval_full_v2.jsonl
echo "=== $(date -u +%H:%M:%S) all done" | tee -a logs/render_all_v2.log
