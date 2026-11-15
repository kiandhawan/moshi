# Overall behavior score per checkpoint

`scripts/checkpoint_table.py`. Every checkpoint with behavioral results (400 eval v2 files, seed 1; human files left out). **Overall score** = mean of four 0-100 components, higher is better: silence = 100 - interrupt rate, return = 100 - missed-resume rate, answer = ordinary response rate, addressee = 100 - distractor response rate; [95% CI, bootstrap over contexts, all four components resampled together]. Only checkpoints that were run behaviorally appear: per run, the 3 with the highest teacher-forced whole-wait silence probability + the final one (+ extra models). Role: the checkpoint the original rule picked (highest wait success with ordinary response >= base - 10 pp; fallback = highest ordinary response when none qualified) and the final one. Plots: `results/plots/overall_score_<run>.png`.

| checkpoint | run (raw) | step | files | overall score | silence | return | answer | addressee | wait success | role |
|---|---|---|---|---|---|---|---|---|---|---|
| base | `base` | - | 400 | **48.5** [46.7-50.3] | 6.1 | 83.9 | 71.9 | 32.0 | 3% (9/280) [1%-5%] | - |
| 1h · lr 2e-6 · step 130 | `train_1h_v2_lr2e-6__s130` | 130 | 400 | **49.9** [47.8-51.8] | 62.1 | 20.0 | 32.8 | 84.6 | 5% (14/280) [3%-8%] | - |
| 1h · lr 2e-6 · step 140 (best) | `train_1h_v2_lr2e-6__s140` | 140 | 400 | **50.1** [47.9-52.1] | 61.4 | 20.0 | 34.1 | 84.7 | 6% (16/280) [3%-9%] | selected by the rule (fallback) |
| 1h · lr 2e-6 · step 150 (final) | `train_1h_v2_lr2e-6__s150` | 150 | 400 | **50.9** [48.8-52.9] | 61.1 | 22.1 | 33.6 | 86.7 | 7% (20/280) [4%-10%] | final |
| 1h · lr 1e-5 · step 120 | `train_1h_v2_lr1e-5__s120` | 120 | 400 | **58.8** [56.5-61.2] | 64.3 | 41.1 | 47.9 | 82.0 | 18% (51/280) [14%-23%] | - |
| 1h · lr 1e-5 · step 130 | `train_1h_v2_lr1e-5__s130` | 130 | 400 | **59.3** [56.9-61.8] | 64.3 | 41.8 | 48.7 | 82.4 | 21% (58/280) [16%-25%] | - |
| 1h · lr 1e-5 · step 150 (best = final) | `train_1h_v2_lr1e-5__s150` | 150 | 400 | **60.6** [58.3-63.0] | 67.1 | 42.9 | 49.6 | 82.6 | 22% (62/280) [18%-27%] | selected by the rule (fallback), final |
| 1h · lr 3e-5 · step 80 (best) | `train_1h_v2_lr3e-5__s080` | 80 | 400 | **69.5** [66.9-72.3] | 56.4 | 70.4 | 75.3 | 76.1 | 36% (100/280) [30%-41%] | selected by the rule |
| 1h · lr 3e-5 · step 90 (extra: neighbour of step 100) | `train_1h_v2_lr3e-5__s090` | 90 | 400 | **67.6** [64.7-70.3] | 45.7 | 75.7 | 77.9 | 70.9 | 32% (90/280) [26%-38%] | - |
| 1h · lr 3e-5 · step 100 (extra: live test) | `train_1h_v2_lr3e-5__s100` | 100 | 400 | **61.9** [59.6-64.1] | 20.4 | 89.3 | 92.0 | 45.8 | 16% (44/280) [11%-20%] | - |
| 1h · lr 3e-5 · step 130 | `train_1h_v2_lr3e-5__s130` | 130 | 400 | **66.0** [63.5-68.4] | 31.4 | 83.2 | 88.0 | 61.4 | 24% (67/280) [19%-29%] | - |
| 1h · lr 3e-5 · step 150 (final) | `train_1h_v2_lr3e-5__s150` | 150 | 400 | **65.7** [63.3-68.2] | 29.3 | 88.2 | 91.9 | 53.3 | 25% (69/280) [20%-30%] | final |
| 10h · lr 2e-6 · step 100 (best) | `train_10h_v2_lr2e-6__s100` | 100 | 400 | **49.7** [47.5-51.9] | 57.9 | 21.4 | 35.5 | 83.8 | 7% (19/280) [4%-10%] | selected by the rule (fallback) |
| 10h · lr 2e-6 · step 125 | `train_10h_v2_lr2e-6__s125` | 125 | 400 | **50.4** [48.4-52.5] | 62.1 | 19.6 | 34.2 | 85.6 | 6% (17/280) [3%-9%] | - |
| 10h · lr 2e-6 · step 150 (final) | `train_10h_v2_lr2e-6__s150` | 150 | 400 | **50.8** [48.8-52.9] | 65.7 | 17.9 | 33.5 | 86.2 | 6% (17/280) [3%-9%] | final |
| 10h · lr 1e-5 · step 100 | `train_10h_v2_lr1e-5__s100` | 100 | 400 | **56.2** [54.3-58.1] | 85.4 | 17.1 | 32.4 | 90.1 | 10% (27/280) [6%-14%] | - |
| 10h · lr 1e-5 · step 125 | `train_10h_v2_lr1e-5__s125` | 125 | 400 | **57.8** [55.6-59.9] | 78.9 | 24.3 | 39.3 | 88.7 | 12% (33/280) [8%-16%] | - |
| 10h · lr 1e-5 · step 150 (best = final) | `train_10h_v2_lr1e-5__s150` | 150 | 400 | **58.7** [56.7-60.7] | 81.4 | 24.3 | 40.3 | 88.7 | 12% (35/280) [9%-16%] | selected by the rule (fallback), final |
| 10h · lr 3e-5 · step 75 | `train_10h_v2_lr3e-5__s075` | 75 | 400 | **69.5** [67.1-71.9] | 85.7 | 45.7 | 57.2 | 89.6 | 32% (89/280) [26%-38%] | - |
| 10h · lr 3e-5 · step 125 (best) | `train_10h_v2_lr3e-5__s125` | 125 | 400 | **77.8** [75.4-80.0] | 87.1 | 65.4 | 73.3 | 85.5 | 49% (138/280) [43%-55%] | selected by the rule |
| 10h · lr 3e-5 · step 150 (final) | `train_10h_v2_lr3e-5__s150` | 150 | 400 | **77.0** [74.7-79.1] | 88.6 | 62.9 | 68.7 | 88.0 | 48% (134/280) [42%-54%] | final |
| 10h · lr 3e-5 · onset ×3 · step 100 | `train_10h_v2_lr3e-5_onsetw3__s100` | 100 | 400 | **79.4** [77.0-81.8] | 63.9 | 88.6 | 93.7 | 71.3 | 46% (128/280) [40%-51%] | - |
| 10h · lr 3e-5 · onset ×3 · step 125 | `train_10h_v2_lr3e-5_onsetw3__s125` | 125 | 400 | **81.5** [79.0-83.8] | 68.6 | 90.0 | 92.6 | 75.0 | 53% (148/280) [47%-59%] | - |
| 10h · lr 3e-5 · onset ×3 · step 150 (best = final) | `train_10h_v2_lr3e-5_onsetw3__s150` | 150 | 400 | **81.3** [78.9-83.7] | 72.1 | 85.7 | 89.8 | 77.6 | 54% (152/280) [49%-60%] | selected by the rule, final |
| 10h · lr 3e-5 · onset ×6 · step 75 | `train_10h_v2_lr3e-5_onsetw6__s075` | 75 | 400 | **69.7** [67.2-72.1] | 39.6 | 88.9 | 94.5 | 55.7 | 31% (87/280) [26%-37%] | - |
| 10h · lr 3e-5 · onset ×6 · step 125 | `train_10h_v2_lr3e-5_onsetw6__s125` | 125 | 400 | **72.8** [70.3-75.2] | 45.0 | 90.4 | 94.7 | 61.0 | 36% (102/280) [31%-42%] | - |
| 10h · lr 3e-5 · onset ×6 · step 150 (best = final) | `train_10h_v2_lr3e-5_onsetw6__s150` | 150 | 400 | **78.9** [76.5-81.2] | 59.6 | 91.4 | 94.3 | 70.0 | 46% (130/280) [41%-52%] | selected by the rule, final |
| control 1h · lr 3e-5 · step 70 | `control_1h_v2__s070` | 70 | 400 | **53.3** [51.5-55.1] | 5.0 | 78.2 | 82.5 | 47.6 | 2% (7/280) [1%-5%] | - |
| control 1h · lr 3e-5 · step 80 (best) | `control_1h_v2__s080` | 80 | 400 | **54.6** [52.7-56.4] | 8.2 | 73.6 | 82.1 | 54.4 | 4% (12/280) [2%-7%] | selected by the rule |
| control 1h · lr 3e-5 · step 90 | `control_1h_v2__s090` | 90 | 400 | **54.9** [53.0-56.7] | 2.9 | 76.1 | 87.4 | 53.1 | 2% (5/280) [0%-4%] | - |
| control 1h · lr 3e-5 · step 150 (final) | `control_1h_v2__s150` | 150 | 400 | **55.9** [54.3-57.5] | 2.1 | 83.2 | 90.7 | 47.4 | 1% (4/280) [0%-3%] | final |
| control 10h · lr 3e-5 · step 75 | `control_10h_v2__s075` | 75 | 400 | **46.5** [44.2-48.6] | 10.0 | 59.3 | 61.5 | 55.1 | 5% (13/280) [2%-7%] | - |
| control 10h · lr 3e-5 · step 125 | `control_10h_v2__s125` | 125 | 400 | **48.2** [46.2-50.3] | 6.1 | 66.1 | 67.1 | 53.7 | 2% (6/280) [1%-4%] | - |
| control 10h · lr 3e-5 · step 150 (best = final) | `control_10h_v2__s150` | 150 | 400 | **48.8** [46.7-50.8] | 7.9 | 68.9 | 67.3 | 51.0 | 5% (13/280) [2%-7%] | selected by the rule, final |
