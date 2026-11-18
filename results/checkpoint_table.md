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
| 1h · lr 3e-5 · onset ×3 · step 110 | `train_1h_v2_lr3e-5_onsetw3__s110` | 110 | 400 | **67.1** [64.8-69.5] | 32.5 | 87.5 | 91.8 | 56.7 | 24% (66/280) [19%-29%] | - |
| 1h · lr 3e-5 · onset ×3 · step 140 (best) | `train_1h_v2_lr3e-5_onsetw3__s140` | 140 | 400 | **66.4** [63.8-68.8] | 32.1 | 90.4 | 92.5 | 50.6 | 25% (70/280) [20%-30%] | selected by the rule |
| 1h · lr 3e-5 · onset ×3 · step 150 (final) | `train_1h_v2_lr3e-5_onsetw3__s150` | 150 | 400 | **65.8** [63.3-68.2] | 32.9 | 87.5 | 92.8 | 50.1 | 22% (62/280) [18%-27%] | final |
| 1h · lr 1e-4 · step 50 (best) | `train_1h_v2_lr1e-4__s050` | 50 | 400 | **73.3** [70.7-75.8] | 68.2 | 65.4 | 75.2 | 84.4 | 41% (116/280) [36%-47%] | selected by the rule |
| 1h · lr 1e-4 · step 140 | `train_1h_v2_lr1e-4__s140` | 140 | 400 | **64.9** [62.4-67.5] | 34.6 | 81.4 | 91.8 | 51.9 | 23% (64/280) [18%-28%] | - |
| 1h · lr 1e-4 · step 150 (final) | `train_1h_v2_lr1e-4__s150` | 150 | 400 | **65.8** [63.3-68.2] | 33.6 | 83.6 | 91.2 | 54.9 | 21% (60/280) [16%-26%] | final |
| 1h · lr 1e-4 · onset ×3 · step 120 | `train_1h_v2_lr1e-4_onsetw3__s120` | 120 | 400 | **66.4** [63.9-68.9] | 33.9 | 86.1 | 93.6 | 52.1 | 24% (66/280) [19%-29%] | - |
| 1h · lr 1e-4 · onset ×3 · step 130 | `train_1h_v2_lr1e-4_onsetw3__s130` | 130 | 400 | **67.7** [65.3-70.3] | 33.6 | 89.6 | 95.7 | 51.9 | 25% (71/280) [20%-31%] | - |
| 1h · lr 1e-4 · onset ×3 · step 150 (best = final) | `train_1h_v2_lr1e-4_onsetw3__s150` | 150 | 400 | **68.4** [66.1-70.9] | 33.6 | 90.0 | 94.9 | 55.3 | 27% (75/280) [21%-32%] | selected by the rule, final |
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
| 10h · lr 1e-4 · step 100 | `train_10h_v2_lr1e-4__s100` | 100 | 400 | **84.4** [82.3-86.3] | 89.6 | 75.7 | 84.4 | 87.8 | 61% (170/280) [55%-66%] | - |
| 10h · lr 1e-4 · step 125 (best) | `train_10h_v2_lr1e-4__s125` | 125 | 400 | **85.1** [83.0-87.1] | 89.3 | 81.1 | 81.8 | 88.3 | 62% (175/280) [57%-68%] | selected by the rule |
| 10h · lr 1e-4 · step 150 (final) | `train_10h_v2_lr1e-4__s150` | 150 | 400 | **83.6** [81.7-85.5] | 93.2 | 74.3 | 75.8 | 91.2 | 60% (167/280) [54%-65%] | final |
| 10h · lr 1e-4 · onset ×3 · step 75 | `train_10h_v2_lr1e-4_onsetw3__s075` | 75 | 400 | **85.7** [83.6-87.6] | 82.9 | 85.7 | 90.9 | 83.5 | 61% (171/280) [55%-67%] | - |
| 10h · lr 1e-4 · onset ×3 · step 100 | `train_10h_v2_lr1e-4_onsetw3__s100` | 100 | 400 | **86.9** [84.8-88.8] | 82.1 | 87.9 | 94.1 | 83.3 | 64% (178/280) [58%-69%] | - |
| 10h · lr 1e-4 · onset ×3 · step 150 (best = final) | `train_10h_v2_lr1e-4_onsetw3__s150` | 150 | 400 | **86.5** [84.3-88.5] | 80.7 | 88.9 | 90.9 | 85.5 | 65% (181/280) [59%-70%] | selected by the rule, final |
| 10h · lr 3e-4 · step 100 | `train_10h_v2_lr3e-4__s100` | 100 | 400 | **83.7** [81.7-85.5] | 94.3 | 67.5 | 79.6 | 93.4 | 53% (149/280) [48%-59%] | - |
| 10h · lr 3e-4 · step 125 (best) | `train_10h_v2_lr3e-4__s125` | 125 | 400 | **84.9** [83.0-86.7] | 91.4 | 77.5 | 78.3 | 92.5 | 62% (173/280) [56%-68%] | selected by the rule |
| 10h · lr 3e-4 · step 150 (final) | `train_10h_v2_lr3e-4__s150` | 150 | 400 | **85.8** [83.9-87.6] | 92.1 | 76.4 | 82.3 | 92.3 | 60% (168/280) [54%-66%] | final |
| 10h · lr 3e-4 · onset ×3 · step 75 | `train_10h_v2_lr3e-4_onsetw3__s075` | 75 | 400 | **85.5** [83.3-87.5] | 81.1 | 84.3 | 91.2 | 85.5 | 61% (171/280) [55%-66%] | - |
| 10h · lr 3e-4 · onset ×3 · step 100 (best) | `train_10h_v2_lr3e-4_onsetw3__s100` | 100 | 400 | **87.7** [85.6-89.6] | 84.3 | 88.9 | 90.3 | 87.4 | 66% (184/280) [60%-71%] | selected by the rule |
| 10h · lr 3e-4 · onset ×3 · step 150 (final) | `train_10h_v2_lr3e-4_onsetw3__s150` | 150 | 400 | **87.7** [85.9-89.5] | 86.8 | 85.7 | 89.4 | 88.9 | 64% (179/280) [58%-70%] | final |
| control 1h · lr 3e-5 · step 70 | `control_1h_v2__s070` | 70 | 400 | **53.3** [51.5-55.1] | 5.0 | 78.2 | 82.5 | 47.6 | 2% (7/280) [1%-5%] | - |
| control 1h · lr 3e-5 · step 80 (best) | `control_1h_v2__s080` | 80 | 400 | **54.6** [52.7-56.4] | 8.2 | 73.6 | 82.1 | 54.4 | 4% (12/280) [2%-7%] | selected by the rule |
| control 1h · lr 3e-5 · step 90 | `control_1h_v2__s090` | 90 | 400 | **54.9** [53.0-56.7] | 2.9 | 76.1 | 87.4 | 53.1 | 2% (5/280) [0%-4%] | - |
| control 1h · lr 3e-5 · step 150 (final) | `control_1h_v2__s150` | 150 | 400 | **55.9** [54.3-57.5] | 2.1 | 83.2 | 90.7 | 47.4 | 1% (4/280) [0%-3%] | final |
| control 1h · lr 1e-4 · step 40 (best) | `control_1h_v2_lr1e-4__s040` | 40 | 400 | **51.0** [48.7-53.1] | 16.8 | 59.3 | 64.3 | 63.6 | 5% (15/280) [3%-8%] | selected by the rule |
| control 1h · lr 1e-4 · step 50 | `control_1h_v2_lr1e-4__s050` | 50 | 400 | **55.5** [53.7-57.2] | 7.5 | 79.6 | 86.5 | 48.3 | 4% (12/280) [2%-7%] | - |
| control 1h · lr 1e-4 · step 100 | `control_1h_v2_lr1e-4__s100` | 100 | 400 | **58.4** [56.7-60.0] | 1.8 | 87.1 | 93.9 | 50.8 | 1% (4/280) [0%-3%] | - |
| control 1h · lr 1e-4 · step 150 (final) | `control_1h_v2_lr1e-4__s150` | 150 | 400 | **56.8** [55.3-58.2] | 0.0 | 91.4 | 95.9 | 39.7 | 0% (0/280) [0%-1%]* | final |
| control 10h · lr 3e-5 · step 75 | `control_10h_v2__s075` | 75 | 400 | **46.5** [44.2-48.6] | 10.0 | 59.3 | 61.5 | 55.1 | 5% (13/280) [2%-7%] | - |
| control 10h · lr 3e-5 · step 125 | `control_10h_v2__s125` | 125 | 400 | **48.2** [46.2-50.3] | 6.1 | 66.1 | 67.1 | 53.7 | 2% (6/280) [1%-4%] | - |
| control 10h · lr 3e-5 · step 150 (best = final) | `control_10h_v2__s150` | 150 | 400 | **48.8** [46.7-50.8] | 7.9 | 68.9 | 67.3 | 51.0 | 5% (13/280) [2%-7%] | selected by the rule, final |
| control 10h · lr 1e-4 · step 50 | `control_10h_v2_lr1e-4__s050` | 50 | 400 | **51.5** [49.4-53.3] | 3.6 | 75.7 | 81.7 | 44.9 | 1% (4/280) [0%-3%] | - |
| control 10h · lr 1e-4 · step 125 | `control_10h_v2_lr1e-4__s125` | 125 | 400 | **53.8** [51.5-55.8] | 9.6 | 71.1 | 76.2 | 58.2 | 5% (14/280) [3%-8%] | - |
| control 10h · lr 1e-4 · step 150 (best = final) | `control_10h_v2_lr1e-4__s150` | 150 | 400 | **54.4** [52.2-56.3] | 9.3 | 73.9 | 79.3 | 54.9 | 7% (20/280) [4%-10%] | selected by the rule, final |
