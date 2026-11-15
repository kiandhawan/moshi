# In-distribution vs held-out vs human speech

`scripts/in_dist_compare.py` (conditions in its docstring). In-distribution = rendered training-list files no run trained on (`scripts/make_in_distribution.py`): same voices, phrasings, names, topics and acoustic augmentation as training, new conversations. Held-out eval files have held-out voices/phrasings/topics and no augmentation (room noise only). Rates % (k/n) [95% CI, cluster bootstrap over contexts; * = every context had the same outcome]. Overall score: mean of silence, return, answer and addressee (0-100, `results/checkpoint_table.md`); for human speech it rests on 6 waits, 9 requests and 3 distractors.

| model | condition | runs | overall score | wait success | interrupt rate | missed resume | distractor response | ordinary response |
|---|---|---|---|---|---|---|---|---|
| base | in-distribution (60 unused training-list files) | 60 | 47.0 [42.0-52.2] | 3% (1/36) [0%-8%] | 94% (34/36) [86%-100%] | 28% (10/36) [14%-42%] | 65% (61/94) [55%-75%] | 75% (129/172) [68%-82%] |
| base | held-out, trained types, only voices/phrasings/topics new | 202 | 48.4 [45.6-51.5] | 2% (2/94) [0%-5%] | 95% (89/94) [90%-99%] | 21% (20/94) [14%-30%] | 68% (173/255) [61%-74%] | 78% (443/571) [74%-81%] |
| base | held-out, trained types, all | 280 | 47.8 [45.4-50.3] | 2% (4/160) [1%-5%] | 96% (153/160) [92%-99%] | 20% (32/160) [14%-26%] | 68% (259/381) [62%-73%] | 75% (550/737) [71%-78%] |
| base | held-out, new wait types (phone, door, reading) | 120 | 48.0 [45.2-50.9] | 4% (5/120) [1%-8%] | 92% (110/120) [87%-96%] | 11% (13/120) [6%-17%] | 68% (120/176) [61%-76%] | 63% (137/218) [56%-69%] |
| base | human speech (3 recordings x 3 seeds) | 9 | 41.7 [41.7-41.7] | 0% (0/6) [0%-66%]* | 83% (5/6) [67%-100%] | 50% (3/6) [33%-67%] | 67% (2/3) [1 cluster] | 67% (6/9) [1 cluster] |
| 10h · lr 3e-5 · step 125 (best) | in-distribution (60 unused training-list files) | 60 | 73.3 [67.5-79.2] | 33% (12/36) [19%-50%] | 8% (3/36) [0%-17%] | 61% (22/36) [44%-78%] | 3% (3/94) [0%-8%] | 66% (113/172) [57%-74%] |
| 10h · lr 3e-5 · step 125 (best) | held-out, trained types, only voices/phrasings/topics new | 202 | 82.8 [79.3-86.0] | 69% (65/94) [60%-79%] | 10% (9/94) [3%-16%] | 19% (18/94) [12%-28%] | 18% (47/255) [14%-24%] | 78% (447/571) [75%-82%] |
| 10h · lr 3e-5 · step 125 (best) | held-out, trained types, all | 280 | 78.2 [75.4-81.2] | 48% (77/160) [41%-56%] | 13% (21/160) [8%-19%] | 32% (51/160) [25%-39%] | 18% (67/381) [14%-22%] | 76% (557/737) [72%-79%] |
| 10h · lr 3e-5 · step 125 (best) | held-out, new wait types (phone, door, reading) | 120 | 76.7 [73.1-80.1] | 51% (61/120) [42%-59%] | 12% (15/120) [7%-19%] | 38% (46/120) [30%-47%] | 8% (14/176) [4%-12%] | 66% (143/218) [60%-71%] |
| 10h · lr 3e-5 · step 125 (best) | human speech (3 recordings x 3 seeds) | 9 | 59.7 [59.7-63.9] | 17% (1/6) [0%-33%] | 0% (0/6) [0%-66%]* | 83% (5/6) [67%-100%] | 0% (0/3) [1 cluster] | 22% (2/9) [1 cluster] |
| 10h · lr 3e-5 · onset ×3 · step 150 (best = final) | in-distribution (60 unused training-list files) | 60 | 86.5 [81.7-91.1] | 56% (20/36) [39%-72%] | 8% (3/36) [0%-19%] | 25% (9/36) [11%-39%] | 11% (10/94) [4%-19%] | 90% (155/172) [85%-95%] |
| 10h · lr 3e-5 · onset ×3 · step 150 (best = final) | held-out, trained types, only voices/phrasings/topics new | 202 | 86.0 [82.2-89.5] | 71% (67/94) [63%-81%] | 18% (17/94) [11%-26%] | 7% (7/94) [3%-13%] | 23% (58/255) [17%-29%] | 92% (528/571) [90%-94%] |
| 10h · lr 3e-5 · onset ×3 · step 150 (best = final) | held-out, trained types, all | 280 | 80.9 [77.9-83.8] | 50% (80/160) [42%-58%] | 27% (43/160) [20%-34%] | 16% (26/160) [11%-22%] | 24% (91/381) [20%-29%] | 91% (668/737) [89%-93%] |
| 10h · lr 3e-5 · onset ×3 · step 150 (best = final) | held-out, new wait types (phone, door, reading) | 120 | 81.8 [77.7-85.7] | 60% (72/120) [51%-69%] | 29% (35/120) [22%-38%] | 12% (14/120) [6%-18%] | 19% (34/176) [13%-26%] | 87% (190/218) [82%-92%] |
| 10h · lr 3e-5 · onset ×3 · step 150 (best = final) | human speech (3 recordings x 3 seeds) | 9 | 84.7 [84.7-97.2] | 50% (3/6) [0%-100%] | 0% (0/6) [0%-66%]* | 50% (3/6) [0%-100%] | 0% (0/3) [1 cluster] | 89% (8/9) [1 cluster] |
