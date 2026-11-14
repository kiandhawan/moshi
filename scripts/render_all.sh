#!/usr/bin/env bash
# Render the full datasets (see results/data_summary.md). Safe to rerun: clips are cached.
# Usage (from the repo root, inside tmux):  bash scripts/render_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ~/.config/fish_audio/env; set +a     # FISH_API_KEY, never printed
mkdir -p logs
run() {  # name scripts_dir out_dir
  echo "=== $(date -u +%H:%M:%S) rendering $1" | tee -a logs/render_all.log
  .venv/bin/python scripts/render.py --scripts "$2" --out "$3" --name "$1" --workers 12 2>&1 \
    | grep --line-buffered -v '^\[0.0' | tee "logs/render_$1.log"
}
run train_all   data/scripts/train   data/train
run eval        data/scripts/eval    data/eval
run control_all data/scripts/control data/control
echo "=== $(date -u +%H:%M:%S) all done" | tee -a logs/render_all.log
