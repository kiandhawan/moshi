# Extra measurements for the report

`scripts/report_extra.py` (docstring for definitions). Rates % (k/n) [95% CI, cluster bootstrap over contexts; * = every context had the same outcome, Wilson interval over contexts].

## 1. Reply to the greeting answer

Training data: files with a greeting answer, files where Moshi replies to it, median gap from the greeting answer's end to the user's next line.

| set | files with a greeting answer | Moshi replies to it | median gap to the user's next line |
|---|---|---|---|
| train_1h_v2 | 60 | 0 | 0.41 s |
| train_10h_v2 | 540 | 0 | 0.42 s |
| control_1h_v2 | 80 | 0 | 0.44 s |
| control_10h_v2 | 706 | 0 | 0.40 s |

Human file regularconvo (seeds 1 2 3; anecdotal, one recording): request 1 is the answer to Moshi's greeting ("Hi, I'm okay. How are you?"), requests 2-3 are ordinary questions.

| model | audio | answers the greeting answer | answers requests 2-3 |
|---|---|---|---|
| base | real | 3/3 | 3/6 |
| base | TTS twin | 3/3 | 6/6 |
| 1h · lr 3e-5 · step 80 (best) | real | 3/3 | 5/6 |
| 1h · lr 3e-5 · step 80 (best) | TTS twin | 3/3 | 6/6 |
| 10h · lr 3e-5 · step 125 (best) | real | 0/3 | 2/6 |
| 10h · lr 3e-5 · step 125 (best) | TTS twin | 3/3 | 4/6 |
| 10h · lr 3e-5 · onset ×3 · step 150 (best = final) | real | 2/3 | 6/6 |
| 10h · lr 3e-5 · onset ×3 · step 150 (best = final) | TTS twin | 3/3 | 5/6 |
| control 10h · lr 3e-5 · step 150 (best = final) | real | 2/3 | 3/6 |

## 2. Ordinary response by when the request comes

Eval v2 (400 files, 35-85 s long, seed 1): share of ordinary requests answered, by the request's time in the conversation. The eval files are at most 85 s, so this cannot show behavior in sessions of several minutes.

| model | < 20 s | 20-40 s | >= 40 s |
|---|---|---|---|
| base | 70% (335/477) [66%-74%] | 79% (178/225) [74%-84%] | 69% (174/253) [63%-74%] |
| 1h · lr 3e-5 · step 80 (best) | 83% (397/477) [80%-86%] | 74% (166/225) [67%-80%] | 62% (156/253) [56%-67%] |
| 10h · lr 3e-5 · step 125 (best) | 82% (392/477) [79%-85%] | 73% (164/225) [67%-79%] | 57% (144/253) [51%-63%] |
| 10h · lr 3e-5 · onset ×3 · step 150 (best = final) | 95% (453/477) [93%-97%] | 87% (195/225) [82%-91%] | 83% (210/253) [78%-88%] |
| control 10h · lr 3e-5 · step 150 (best = final) | 69% (330/477) [65%-73%] | 67% (151/225) [61%-73%] | 64% (162/253) [58%-69%] |

## 3. Timed returns (think time with a deadline)

First Moshi speech after the 2 s ack grace relative to the deadline (the last deadline for extensions): on time = within 3 s; early = more than 3 s before (an interruption); late = more than 3 s after; none = no speech in the file after the grace.

| model | waits | on time (+-3 s) | early | late | none |
|---|---|---|---|---|---|
| base | deadline | 14% (2/14) [0%-36%] | 57% (8/14) [29%-79%] | 14% (2/14) [0%-36%] | 14% (2/14) [0%-36%] |
| base | deadline + extension | 6% (2/34) [0%-15%] | 82% (28/34) [68%-94%] | 6% (2/34) [0%-15%] | 6% (2/34) [0%-15%] |
| 1h · lr 3e-5 · step 80 (best) | deadline | 14% (2/14) [0%-36%] | 36% (5/14) [14%-64%] | 21% (3/14) [0%-43%] | 29% (4/14) [7%-50%] |
| 1h · lr 3e-5 · step 80 (best) | deadline + extension | 3% (1/34) [0%-9%] | 82% (28/34) [68%-94%] | 3% (1/34) [0%-9%] | 12% (4/34) [3%-24%] |
| 10h · lr 3e-5 · step 125 (best) | deadline | 0% (0/14) [0%-22%]* | 21% (3/14) [0%-43%] | 29% (4/14) [7%-50%] | 50% (7/14) [21%-79%] |
| 10h · lr 3e-5 · step 125 (best) | deadline + extension | 3% (1/34) [0%-9%] | 29% (10/34) [15%-44%] | 26% (9/34) [12%-41%] | 41% (14/34) [24%-59%] |
| 10h · lr 3e-5 · onset ×3 · step 150 (best = final) | deadline | 21% (3/14) [0%-43%] | 14% (2/14) [0%-36%] | 36% (5/14) [14%-64%] | 29% (4/14) [7%-57%] |
| 10h · lr 3e-5 · onset ×3 · step 150 (best = final) | deadline + extension | 6% (2/34) [0%-15%] | 62% (21/34) [47%-76%] | 12% (4/34) [3%-24%] | 21% (7/34) [9%-35%] |
| control 10h · lr 3e-5 · step 150 (best = final) | deadline | 7% (1/14) [0%-21%] | 71% (10/14) [50%-93%] | 14% (2/14) [0%-36%] | 7% (1/14) [0%-21%] |
| control 10h · lr 3e-5 · step 150 (best = final) | deadline + extension | 0% (0/34) [0%-10%]* | 100% (34/34) [90%-100%]* | 0% (0/34) [0%-10%]* | 0% (0/34) [0%-10%]* |

## 4. Responses to questions addressed to another person by name

Share of other-name questions (eval: held-out names) that Moshi responds to (it should stay silent), by the kind of conversation they occur in.

| model | in ordinary conversations | in wait conversations |
|---|---|---|
| base | 100% (12/12) [76%-100%]* | 79% (22/28) [61%-93%] |
| 1h · lr 3e-5 · step 80 (best) | 42% (5/12) [17%-67%] | 36% (10/28) [18%-54%] |
| 10h · lr 3e-5 · step 125 (best) | 17% (2/12) [0%-42%] | 21% (6/28) [7%-39%] |
| 10h · lr 3e-5 · onset ×3 · step 150 (best = final) | 25% (3/12) [0%-50%] | 57% (16/28) [39%-75%] |
| control 10h · lr 3e-5 · step 150 (best = final) | 25% (3/12) [8%-50%] | 50% (14/28) [32%-68%] |
