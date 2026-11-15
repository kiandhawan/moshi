# Best checkpoints

One row per run (scripts/eval_run.py). Candidates = the 3 checkpoints with the highest whole-wait silence probability (eval_loss.py, all 403 eval_full_v2 files) + the final one, each run on the 400 eval v2 files (seed 1) and the human files (seeds 1 2 3). Best = highest wait success among candidates whose ordinary response is at most 10 percentage points below the baseline's. Rates on the 400 eval v2 files: % (k/n) [95% CI, cluster bootstrap].

| run | best step | epochs | seen h | checkpoint | wait success | ordinary response | interrupt rate | distractor response | P(silent whole wait) | val loss | reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| train_10h_v2_lr1e-5 | 150 | 4.44 | 43.85 | `runs/train_10h_v2_lr1e-5/checkpoints/checkpoint_000150/consolidated` | 12% (35/280) [9%-16%] | 40% (385/955) [37%-44%] | 19% (52/280) [14%-23%] | 11% (63/557) [9%-14%] | 2.79e-01 | 0.699 | NO candidate kept ordinary response within 10 pp of the baseline's 72%; this is the one with the highest ordinary response |
| train_1h_v2_lr1e-5 | 150 | 40.0 | 44.14 | `runs/train_1h_v2_lr1e-5/checkpoints/checkpoint_000150/consolidated` | 22% (62/280) [18%-27%] | 50% (474/955) [46%-53%] | 33% (92/280) [28%-39%] | 17% (97/557) [14%-21%] | 2.98e-01 | 1.061 | NO candidate kept ordinary response within 10 pp of the baseline's 72%; this is the one with the highest ordinary response |

## All candidates

| run | step | wait success | ordinary response | interrupt rate | P(silent whole wait) | val loss |
|---|---|---|---|---|---|---|
| train_10h_v2_lr1e-5 | 100 | 10% (27/280) [6%-14%] | 32% (309/955) [29%-36%] | 15% (41/280) [11%-19%] | 3.11e-01 | 0.701 |
| train_10h_v2_lr1e-5 | 125 | 12% (33/280) [8%-16%] | 39% (375/955) [36%-43%] | 21% (59/280) [16%-26%] | 2.81e-01 | 0.699 |
| train_10h_v2_lr1e-5 | 150 (best) | 12% (35/280) [9%-16%] | 40% (385/955) [37%-44%] | 19% (52/280) [14%-23%] | 2.79e-01 | 0.699 |
| train_1h_v2_lr1e-5 | 120 | 18% (51/280) [14%-23%] | 48% (457/955) [44%-51%] | 36% (100/280) [30%-41%] | 2.97e-01 | 1.043 |
| train_1h_v2_lr1e-5 | 130 | 21% (58/280) [16%-25%] | 49% (465/955) [45%-52%] | 36% (100/280) [30%-41%] | 2.99e-01 | 1.055 |
| train_1h_v2_lr1e-5 | 150 (best) | 22% (62/280) [18%-27%] | 50% (474/955) [46%-53%] | 33% (92/280) [28%-39%] | 2.98e-01 | 1.061 |
