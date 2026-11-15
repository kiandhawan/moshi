# Best checkpoints

One row per run (scripts/eval_run.py). Candidates = the 3 checkpoints with the highest whole-wait silence probability (eval_loss.py, all 403 eval_full_v2 files) + the final one, each run on the 400 eval v2 files (seed 1) and the human files (seeds 1 2 3). Best = highest wait success among candidates whose ordinary response is at most 10 percentage points below the baseline's. Rates on the 400 eval v2 files: % (k/n) [95% CI, cluster bootstrap].

| run | run (raw) | best step | epochs | seen h | checkpoint | wait success | ordinary response | interrupt rate | distractor response | P(silent whole wait) | val loss | reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1h · lr 2e-6 | `train_1h_v2_lr2e-6` | 140 | 37.33 | 41.2 | `runs/train_1h_v2_lr2e-6/checkpoints/checkpoint_000140/consolidated` | 6% (16/280) [3%-9%] | 34% (326/955) [30%-38%] | 39% (108/280) [33%-45%] | 15% (85/557) [12%-18%] | 1.28e-01 | 0.784 | NO candidate kept ordinary response within 10 pp of the baseline's 72%; this is the one with the highest ordinary response |
| 1h · lr 1e-5 | `train_1h_v2_lr1e-5` | 150 | 40.0 | 44.14 | `runs/train_1h_v2_lr1e-5/checkpoints/checkpoint_000150/consolidated` | 22% (62/280) [18%-27%] | 50% (474/955) [46%-53%] | 33% (92/280) [28%-39%] | 17% (97/557) [14%-21%] | 2.98e-01 | 1.061 | NO candidate kept ordinary response within 10 pp of the baseline's 72%; this is the one with the highest ordinary response |
| 1h · lr 3e-5 | `train_1h_v2_lr3e-5` | 80 | 21.33 | 23.54 | `runs/train_1h_v2_lr3e-5/checkpoints/checkpoint_000080/consolidated` | 36% (100/280) [30%-41%] | 75% (719/955) [72%-78%] | 44% (122/280) [38%-49%] | 24% (133/557) [20%-28%] | 4.69e-01 | 1.374 | highest wait success (36% (100/280) [30%-41%]) among the candidates whose ordinary response is within 10 pp of the baseline's 72% (>= 62%: s80, s90, s130, s150) |
| 10h · lr 2e-6 | `train_10h_v2_lr2e-6` | 100 | 2.96 | 29.24 | `runs/train_10h_v2_lr2e-6/checkpoints/checkpoint_000100/consolidated` | 7% (19/280) [4%-10%] | 35% (339/955) [31%-40%] | 42% (118/280) [36%-48%] | 16% (90/557) [13%-19%] | 1.28e-01 | 0.788 | NO candidate kept ordinary response within 10 pp of the baseline's 72%; this is the one with the highest ordinary response |
| 10h · lr 1e-5 | `train_10h_v2_lr1e-5` | 150 | 4.44 | 43.85 | `runs/train_10h_v2_lr1e-5/checkpoints/checkpoint_000150/consolidated` | 12% (35/280) [9%-16%] | 40% (385/955) [37%-44%] | 19% (52/280) [14%-23%] | 11% (63/557) [9%-14%] | 2.79e-01 | 0.699 | NO candidate kept ordinary response within 10 pp of the baseline's 72%; this is the one with the highest ordinary response |
| 10h · lr 3e-5 | `train_10h_v2_lr3e-5` | 125 | 3.7 | 36.54 | `runs/train_10h_v2_lr3e-5/checkpoints/checkpoint_000125/consolidated` | 49% (138/280) [43%-55%] | 73% (700/955) [70%-76%] | 13% (36/280) [9%-17%] | 15% (81/557) [12%-18%] | 5.29e-01 | 0.724 | highest wait success (49% (138/280) [43%-55%]) among the candidates whose ordinary response is within 10 pp of the baseline's 72% (>= 62%: s125, s150) |
| 10h · lr 3e-5 · onset ×3 | `train_10h_v2_lr3e-5_onsetw3` | 150 | 4.44 | 43.85 | `runs/train_10h_v2_lr3e-5_onsetw3/checkpoints/checkpoint_000150/consolidated` | 54% (152/280) [49%-60%] | 90% (858/955) [88%-92%] | 28% (78/280) [23%-33%] | 22% (125/557) [19%-27%] | 4.64e-01 | 0.745 | highest wait success (54% (152/280) [49%-60%]) among the candidates whose ordinary response is within 10 pp of the baseline's 72% (>= 62%: s100, s125, s150) |
| 10h · lr 3e-5 · onset ×6 | `train_10h_v2_lr3e-5_onsetw6` | 150 | 4.44 | 43.85 | `runs/train_10h_v2_lr3e-5_onsetw6/checkpoints/checkpoint_000150/consolidated` | 46% (130/280) [41%-52%] | 94% (901/955) [93%-96%] | 40% (113/280) [35%-46%] | 30% (167/557) [26%-34%] | 3.44e-01 | 0.760 | highest wait success (46% (130/280) [41%-52%]) among the candidates whose ordinary response is within 10 pp of the baseline's 72% (>= 62%: s75, s125, s150) |
| control 1h · lr 3e-5 | `control_1h_v2` | 80 | 16.0 | 18.1 | `runs/control_1h_v2/checkpoints/checkpoint_000080/consolidated` | 4% (12/280) [2%-7%] | 82% (784/955) [80%-84%] | 92% (257/280) [88%-95%] | 46% (254/557) [41%-50%] | 5.39e-02 | 1.269 | highest wait success (4% (12/280) [2%-7%]) among the candidates whose ordinary response is within 10 pp of the baseline's 72% (>= 62%: s70, s80, s90, s150) |
| control 10h · lr 3e-5 | `control_10h_v2` | 150 | 3.4 | 33.99 | `runs/control_10h_v2/checkpoints/checkpoint_000150/consolidated` | 5% (13/280) [2%-7%] | 67% (643/955) [64%-70%] | 92% (258/280) [89%-95%] | 49% (273/557) [45%-53%] | 6.65e-02 | 0.732 | highest wait success (5% (13/280) [2%-7%]) among the candidates whose ordinary response is within 10 pp of the baseline's 72% (>= 62%: s125, s150) |

## All candidates

| checkpoint | run (raw) | step | wait success | ordinary response | interrupt rate | P(silent whole wait) | val loss |
|---|---|---|---|---|---|---|---|
| 1h · lr 2e-6 · step 130 | `train_1h_v2_lr2e-6` | 130 | 5% (14/280) [3%-8%] | 33% (313/955) [29%-37%] | 38% (106/280) [32%-43%] | 1.27e-01 | 0.784 |
| 1h · lr 2e-6 · step 140 (best) | `train_1h_v2_lr2e-6` | 140 | 6% (16/280) [3%-9%] | 34% (326/955) [30%-38%] | 39% (108/280) [33%-45%] | 1.28e-01 | 0.784 |
| 1h · lr 2e-6 · step 150 (final) | `train_1h_v2_lr2e-6` | 150 | 7% (20/280) [4%-10%] | 34% (321/955) [30%-37%] | 39% (109/280) [33%-45%] | 1.28e-01 | 0.784 |
| 1h · lr 1e-5 · step 120 | `train_1h_v2_lr1e-5` | 120 | 18% (51/280) [14%-23%] | 48% (457/955) [44%-51%] | 36% (100/280) [30%-41%] | 2.97e-01 | 1.043 |
| 1h · lr 1e-5 · step 130 | `train_1h_v2_lr1e-5` | 130 | 21% (58/280) [16%-25%] | 49% (465/955) [45%-52%] | 36% (100/280) [30%-41%] | 2.99e-01 | 1.055 |
| 1h · lr 1e-5 · step 150 (best = final) | `train_1h_v2_lr1e-5` | 150 | 22% (62/280) [18%-27%] | 50% (474/955) [46%-53%] | 33% (92/280) [28%-39%] | 2.98e-01 | 1.061 |
| 1h · lr 3e-5 · step 80 (best) | `train_1h_v2_lr3e-5` | 80 | 36% (100/280) [30%-41%] | 75% (719/955) [72%-78%] | 44% (122/280) [38%-49%] | 4.69e-01 | 1.374 |
| 1h · lr 3e-5 · step 90 (extra: neighbour of step 100) | `train_1h_v2_lr3e-5` | 90 | 32% (90/280) [26%-38%] | 78% (744/955) [75%-81%] | 54% (152/280) [48%-60%] | 4.80e-01 | 1.430 |
| 1h · lr 3e-5 · step 130 | `train_1h_v2_lr3e-5` | 130 | 24% (67/280) [19%-29%] | 88% (840/955) [86%-90%] | 69% (192/280) [63%-74%] | 4.24e-01 | 1.542 |
| 1h · lr 3e-5 · step 150 (final) | `train_1h_v2_lr3e-5` | 150 | 25% (69/280) [20%-30%] | 92% (878/955) [90%-94%] | 71% (198/280) [65%-76%] | 3.33e-01 | 1.550 |
| 10h · lr 2e-6 · step 100 (best) | `train_10h_v2_lr2e-6` | 100 | 7% (19/280) [4%-10%] | 35% (339/955) [31%-40%] | 42% (118/280) [36%-48%] | 1.28e-01 | 0.788 |
| 10h · lr 2e-6 · step 125 | `train_10h_v2_lr2e-6` | 125 | 6% (17/280) [3%-9%] | 34% (327/955) [30%-38%] | 38% (106/280) [32%-43%] | 1.36e-01 | 0.782 |
| 10h · lr 2e-6 · step 150 (final) | `train_10h_v2_lr2e-6` | 150 | 6% (17/280) [3%-9%] | 34% (320/955) [30%-37%] | 34% (96/280) [29%-40%] | 1.37e-01 | 0.781 |
| 10h · lr 1e-5 · step 100 | `train_10h_v2_lr1e-5` | 100 | 10% (27/280) [6%-14%] | 32% (309/955) [29%-36%] | 15% (41/280) [11%-19%] | 3.11e-01 | 0.701 |
| 10h · lr 1e-5 · step 125 | `train_10h_v2_lr1e-5` | 125 | 12% (33/280) [8%-16%] | 39% (375/955) [36%-43%] | 21% (59/280) [16%-26%] | 2.81e-01 | 0.699 |
| 10h · lr 1e-5 · step 150 (best = final) | `train_10h_v2_lr1e-5` | 150 | 12% (35/280) [9%-16%] | 40% (385/955) [37%-44%] | 19% (52/280) [14%-23%] | 2.79e-01 | 0.699 |
| 10h · lr 3e-5 · step 75 | `train_10h_v2_lr3e-5` | 75 | 32% (89/280) [26%-38%] | 57% (546/955) [54%-61%] | 14% (40/280) [10%-19%] | 4.91e-01 | 0.698 |
| 10h · lr 3e-5 · step 125 (best) | `train_10h_v2_lr3e-5` | 125 | 49% (138/280) [43%-55%] | 73% (700/955) [70%-76%] | 13% (36/280) [9%-17%] | 5.29e-01 | 0.724 |
| 10h · lr 3e-5 · step 150 (final) | `train_10h_v2_lr3e-5` | 150 | 48% (134/280) [42%-54%] | 69% (656/955) [66%-72%] | 11% (32/280) [8%-15%] | 5.66e-01 | 0.723 |
| 10h · lr 3e-5 · onset ×3 · step 100 | `train_10h_v2_lr3e-5_onsetw3` | 100 | 46% (128/280) [40%-51%] | 94% (895/955) [92%-95%] | 36% (101/280) [31%-42%] | 3.52e-01 | 0.723 |
| 10h · lr 3e-5 · onset ×3 · step 125 | `train_10h_v2_lr3e-5_onsetw3` | 125 | 53% (148/280) [47%-59%] | 93% (884/955) [91%-94%] | 31% (88/280) [26%-37%] | 4.05e-01 | 0.747 |
| 10h · lr 3e-5 · onset ×3 · step 150 (best = final) | `train_10h_v2_lr3e-5_onsetw3` | 150 | 54% (152/280) [49%-60%] | 90% (858/955) [88%-92%] | 28% (78/280) [23%-33%] | 4.64e-01 | 0.745 |
| 10h · lr 3e-5 · onset ×6 · step 75 | `train_10h_v2_lr3e-5_onsetw6` | 75 | 31% (87/280) [26%-37%] | 94% (902/955) [93%-96%] | 60% (169/280) [54%-66%] | 2.48e-01 | 0.736 |
| 10h · lr 3e-5 · onset ×6 · step 125 | `train_10h_v2_lr3e-5_onsetw6` | 125 | 36% (102/280) [31%-42%] | 95% (904/955) [93%-96%] | 55% (154/280) [49%-61%] | 2.81e-01 | 0.762 |
| 10h · lr 3e-5 · onset ×6 · step 150 (best = final) | `train_10h_v2_lr3e-5_onsetw6` | 150 | 46% (130/280) [41%-52%] | 94% (901/955) [93%-96%] | 40% (113/280) [35%-46%] | 3.44e-01 | 0.760 |
| control 1h · lr 3e-5 · step 70 | `control_1h_v2` | 70 | 2% (7/280) [1%-5%] | 83% (788/955) [80%-85%] | 95% (266/280) [92%-98%] | 4.94e-02 | 1.163 |
| control 1h · lr 3e-5 · step 80 (best) | `control_1h_v2` | 80 | 4% (12/280) [2%-7%] | 82% (784/955) [80%-84%] | 92% (257/280) [88%-95%] | 5.39e-02 | 1.269 |
| control 1h · lr 3e-5 · step 90 | `control_1h_v2` | 90 | 2% (5/280) [0%-4%] | 87% (835/955) [85%-89%] | 97% (272/280) [95%-99%] | 4.95e-02 | 1.333 |
| control 1h · lr 3e-5 · step 150 (final) | `control_1h_v2` | 150 | 1% (4/280) [0%-3%] | 91% (866/955) [89%-92%] | 98% (274/280) [96%-99%] | 4.22e-02 | 1.533 |
| control 10h · lr 3e-5 · step 75 | `control_10h_v2` | 75 | 5% (13/280) [2%-7%] | 61% (587/955) [59%-64%] | 90% (252/280) [86%-94%] | 7.56e-02 | 0.714 |
| control 10h · lr 3e-5 · step 125 | `control_10h_v2` | 125 | 2% (6/280) [1%-4%] | 67% (641/955) [64%-70%] | 94% (263/280) [91%-96%] | 6.64e-02 | 0.733 |
| control 10h · lr 3e-5 · step 150 (best = final) | `control_10h_v2` | 150 | 5% (13/280) [2%-7%] | 67% (643/955) [64%-70%] | 92% (258/280) [89%-95%] | 6.65e-02 | 0.732 |
