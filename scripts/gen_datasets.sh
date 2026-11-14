#!/usr/bin/env bash
# Generate the content-unique script lists (see results/data_summary.md, results/LOG.md).
# Train list: 720 scripts (cut to ~10 h and ~1 h after rendering); eval: 400; control:
# ordinary-only, excluding every block of the train list so control stays content-unique.
set -euo pipefail
cd "$(dirname "$0")/.."
G=".venv/bin/python scripts/generate_scripts.py --ordered --unique"
rm -rf data/scripts/train data/scripts/eval data/scripts/control
$G --split train --n 720 --seed 10 --out data/scripts/train
$G --split eval --n 400 --seed 12 --out data/scripts/eval
$G --split train --only ordinary --n 500 --seed 11 --id-prefix control_11 \
   --exclude-blocks-from data/scripts/train --out data/scripts/control
