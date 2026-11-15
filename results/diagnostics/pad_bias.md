# Decoding-time silence penalty (PAD logit bias)

`scripts/pad_bias_sweep.sh` + `scripts/pad_bias_report.py`. The 40-file diagnostic set (eval_full_subset_v2: 34 waits, 73 ordinary requests, 3 human files included), one batch of 40 at seed 1. At every decoding step the text padding token's logit is lowered by the bias before sampling (`run_inference.py --pad-bias`): a global decoding setting applied identically everywhere, not a mute rule - nothing looks at the conversation. Bias 0 = the existing diagnostic run. no-reply = ordinary requests with no Moshi speech start within 5 s.

| model | PAD bias | ordinary no-reply | missed resume | interrupt rate | wait success | Moshi speech s/min |
|---|---|---|---|---|---|---|
| base | 0 | 23% (17/73) [14%-34%] | 15% (5/34) [3%-26%] | 94% (32/34) [85%-100%] | 6% (2/34) [0%-15%] | 13.5 |
| 1h · lr 3e-5 · step 80 (best) | 0 | 32% (23/73) [24%-39%] | 38% (13/34) [24%-56%] | 53% (18/34) [38%-71%] | 29% (10/34) [15%-44%] | 10.9 |
| 1h · lr 3e-5 · step 80 (best) | 0.5 | 22% (16/73) [14%-31%] | 35% (12/34) [18%-53%] | 56% (19/34) [38%-74%] | 26% (9/34) [12%-41%] | 13.4 |
| 1h · lr 3e-5 · step 80 (best) | 1 | 15% (11/73) [8%-22%] | 24% (8/34) [9%-38%] | 68% (23/34) [50%-82%] | 18% (6/34) [6%-32%] | 14.2 |
| 1h · lr 3e-5 · step 80 (best) | 1.5 | 14% (10/73) [7%-20%] | 18% (6/34) [6%-32%] | 74% (25/34) [59%-88%] | 18% (6/34) [6%-32%] | 15.7 |
| 1h · lr 3e-5 · step 80 (best) | 2 | 8% (6/73) [3%-14%] | 18% (6/34) [6%-29%] | 76% (26/34) [62%-91%] | 18% (6/34) [6%-32%] | 17.8 |
| 10h · lr 3e-5 · step 125 (best) | 0 | 30% (22/73) [21%-41%] | 44% (15/34) [26%-62%] | 12% (4/34) [3%-24%] | 44% (15/34) [26%-62%] | 8.4 |
| 10h · lr 3e-5 · step 125 (best) | 0.5 | 25% (18/73) [17%-32%] | 26% (9/34) [12%-41%] | 18% (6/34) [6%-29%] | 44% (15/34) [26%-62%] | 10.3 |
| 10h · lr 3e-5 · step 125 (best) | 1 | 10% (7/73) [4%-16%] | 18% (6/34) [6%-29%] | 47% (16/34) [29%-62%] | 41% (14/34) [26%-59%] | 13.1 |
| 10h · lr 3e-5 · step 125 (best) | 1.5 | 8% (6/73) [3%-14%] | 9% (3/34) [0%-21%] | 62% (21/34) [44%-76%] | 29% (10/34) [15%-44%] | 14.0 |
| 10h · lr 3e-5 · step 125 (best) | 2 | 8% (6/73) [3%-14%] | 12% (4/34) [3%-24%] | 74% (25/34) [59%-88%] | 15% (5/34) [6%-26%] | 17.1 |
