#!/usr/bin/env bash
# Generate the v2 script lists (see results/data_summary_v2.md, results/LOG.md). v1 stays in data/scripts/.
# Train list: 720 scripts (cut to ~10 h and ~1 h after rendering); control: ordinary-only,
# excluding the blocks of the train list's ordinary files (control starts with those files);
# eval v2: the v1 eval scripts with 40 replaced by other_name scripts.
set -euo pipefail
cd "$(dirname "$0")"
G="../.venv/bin/python generate_scripts_v2.py"
$G --split train --n 720 --seed 10 --out ../data/v2/scripts/train
$G --split train --only ordinary --n 500 --seed 11 --id-prefix control_11 \
   --exclude-ordinary-from ../data/v2/scripts/train --out ../data/v2/scripts/control
$G --eval-v2 --v1-eval ../data/scripts/eval --out ../data/v2/scripts/eval
