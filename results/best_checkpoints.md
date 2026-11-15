# Best checkpoints

One row per run (scripts/eval_run.py). Candidates = the 3 checkpoints with the highest whole-wait silence probability (eval_loss.py, all 403 eval_full_v2 files) + the final one, each run on the 400 eval v2 files (seed 1) and the human files (seeds 1 2 3). Best = highest wait success among candidates whose ordinary response is at most 10 percentage points below the baseline's. Rates on the 400 eval v2 files: % (k/n) [95% CI, cluster bootstrap].

| run | best step | epochs | seen h | checkpoint | wait success | ordinary response | interrupt rate | distractor response | P(silent whole wait) | val loss | reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| train_10h_v2_lr1e-5 | 150 | 4.44 | 43.85 | `runs/train_10h_v2_lr1e-5/checkpoints/checkpoint_000150/consolidated` | 12% (35/280) [9%-16%] | 40% (385/955) [37%-44%] | 19% (52/280) [14%-23%] | 11% (63/557) [9%-14%] | 2.79e-01 | 0.699 | NO candidate kept ordinary response within 10 pp of the baseline's 72%; this is the one with the highest ordinary response |
| train_10h_v2_lr3e-5 | 125 | 3.7 | 36.54 | `runs/train_10h_v2_lr3e-5/checkpoints/checkpoint_000125/consolidated` | 49% (138/280) [43%-55%] | 73% (700/955) [70%-76%] | 13% (36/280) [9%-17%] | 15% (81/557) [12%-18%] | 5.29e-01 | 0.724 | highest wait success (49% (138/280) [43%-55%]) among the candidates whose ordinary response is within 10 pp of the baseline's 72% (>= 62%: s125, s150) |
| train_1h_v2_lr1e-5 | 150 | 40.0 | 44.14 | `runs/train_1h_v2_lr1e-5/checkpoints/checkpoint_000150/consolidated` | 22% (62/280) [18%-27%] | 50% (474/955) [46%-53%] | 33% (92/280) [28%-39%] | 17% (97/557) [14%-21%] | 2.98e-01 | 1.061 | NO candidate kept ordinary response within 10 pp of the baseline's 72%; this is the one with the highest ordinary response |
| train_1h_v2_lr3e-5 | 80 | 21.33 | 23.54 | `runs/train_1h_v2_lr3e-5/checkpoints/checkpoint_000080/consolidated` | 36% (100/280) [30%-41%] | 75% (719/955) [72%-78%] | 44% (122/280) [38%-49%] | 24% (133/557) [20%-28%] | 4.69e-01 | 1.374 | highest wait success (36% (100/280) [30%-41%]) among the candidates whose ordinary response is within 10 pp of the baseline's 72% (>= 62%: s80, s90, s130, s150) |

## All candidates

| run | step | wait success | ordinary response | interrupt rate | P(silent whole wait) | val loss |
|---|---|---|---|---|---|---|
| train_10h_v2_lr1e-5 | 100 | 10% (27/280) [6%-14%] | 32% (309/955) [29%-36%] | 15% (41/280) [11%-19%] | 3.11e-01 | 0.701 |
| train_10h_v2_lr1e-5 | 125 | 12% (33/280) [8%-16%] | 39% (375/955) [36%-43%] | 21% (59/280) [16%-26%] | 2.81e-01 | 0.699 |
| train_10h_v2_lr1e-5 | 150 (best) | 12% (35/280) [9%-16%] | 40% (385/955) [37%-44%] | 19% (52/280) [14%-23%] | 2.79e-01 | 0.699 |
| train_10h_v2_lr3e-5 | 75 | 32% (89/280) [26%-38%] | 57% (546/955) [54%-61%] | 14% (40/280) [10%-19%] | 4.91e-01 | 0.698 |
| train_10h_v2_lr3e-5 | 125 (best) | 49% (138/280) [43%-55%] | 73% (700/955) [70%-76%] | 13% (36/280) [9%-17%] | 5.29e-01 | 0.724 |
| train_10h_v2_lr3e-5 | 150 (final) | 48% (134/280) [42%-54%] | 69% (656/955) [66%-72%] | 11% (32/280) [8%-15%] | 5.66e-01 | 0.723 |
| train_1h_v2_lr1e-5 | 120 | 18% (51/280) [14%-23%] | 48% (457/955) [44%-51%] | 36% (100/280) [30%-41%] | 2.97e-01 | 1.043 |
| train_1h_v2_lr1e-5 | 130 | 21% (58/280) [16%-25%] | 49% (465/955) [45%-52%] | 36% (100/280) [30%-41%] | 2.99e-01 | 1.055 |
| train_1h_v2_lr1e-5 | 150 (best) | 22% (62/280) [18%-27%] | 50% (474/955) [46%-53%] | 33% (92/280) [28%-39%] | 2.98e-01 | 1.061 |
| train_1h_v2_lr3e-5 | 80 (best) | 36% (100/280) [30%-41%] | 75% (719/955) [72%-78%] | 44% (122/280) [38%-49%] | 4.69e-01 | 1.374 |
| train_1h_v2_lr3e-5 | 90 | 32% (90/280) [26%-38%] | 78% (744/955) [75%-81%] | 54% (152/280) [48%-60%] | 4.80e-01 | 1.430 |
| train_1h_v2_lr3e-5 | 130 | 24% (67/280) [19%-29%] | 88% (840/955) [86%-90%] | 69% (192/280) [63%-74%] | 4.24e-01 | 1.542 |
| train_1h_v2_lr3e-5 | 150 (final) | 25% (69/280) [20%-30%] | 92% (878/955) [90%-94%] | 71% (198/280) [65%-76%] | 3.33e-01 | 1.550 |
