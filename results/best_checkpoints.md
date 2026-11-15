# Best checkpoints

One row per run (scripts/eval_run.py). Candidates = the 3 checkpoints with the highest whole-wait silence probability (eval_loss.py, all 403 eval_full_v2 files) + the final one, each run on the 400 eval v2 files (seed 1) and the human files (seeds 1 2 3). Best = highest wait success among candidates whose ordinary response is at most 10 percentage points below the baseline's. Rates on the 400 eval v2 files: % (k/n) [95% CI, cluster bootstrap].

| run | best step | epochs | seen h | checkpoint | wait success | ordinary response | interrupt rate | distractor response | P(silent whole wait) | val loss | reason |
|---|---|---|---|---|---|---|---|---|---|---|---|

## All candidates

| run | step | wait success | ordinary response | interrupt rate | P(silent whole wait) | val loss |
|---|---|---|---|---|---|---|
