# Human recordings vs TTS twins

`scripts/make_human_tts.py` (twins: same utterances re-voiced with held-out Fish voices at the original onsets; adjustments in `data/human_tts/report.json`) and `scripts/human_tts_compare.py`. Each model: the 3 files x seeds 1 2 3, unbatched. Rates % (k/n) [95% CI; each file is its own context, so with 3 files the intervals are very wide; * = all files had the same outcome].

| model | audio | runs | wait success | missed resume | interrupt rate | distractor response | ordinary response |
|---|---|---|---|---|---|---|---|
| base | real | 9 | 0% (0/6) [0%-66%]* | 50% (3/6) [33%-67%] | 83% (5/6) [67%-100%] | 67% (2/3) [1 cluster] | 67% (6/9) [1 cluster] |
| base | TTS twin | 9 | 0% (0/6) [0%-66%]* | 83% (5/6) [67%-100%] | 100% (6/6) [34%-100%]* | 67% (2/3) [1 cluster] | 100% (9/9) [1 cluster] |
| 1h · lr 3e-5 · step 80 (best) | real | 9 | 33% (2/6) [0%-67%] | 67% (4/6) [33%-100%] | 0% (0/6) [0%-66%]* | 0% (0/3) [1 cluster] | 89% (8/9) [1 cluster] |
| 1h · lr 3e-5 · step 80 (best) | TTS twin | 9 | 33% (2/6) [0%-67%] | 50% (3/6) [0%-100%] | 50% (3/6) [33%-67%] | 33% (1/3) [1 cluster] | 100% (9/9) [1 cluster] |
| 10h · lr 1e-5 · step 150 (best = final) | real | 9 | 0% (0/6) [0%-66%]* | 100% (6/6) [34%-100%]* | 33% (2/6) [0%-67%] | 0% (0/3) [1 cluster] | 44% (4/9) [1 cluster] |
| 10h · lr 1e-5 · step 150 (best = final) | TTS twin | 9 | 0% (0/6) [0%-66%]* | 100% (6/6) [34%-100%]* | 0% (0/6) [0%-66%]* | 0% (0/3) [1 cluster] | 11% (1/9) [1 cluster] |
| 10h · lr 3e-5 · step 125 (best) | real | 9 | 17% (1/6) [0%-33%] | 83% (5/6) [67%-100%] | 0% (0/6) [0%-66%]* | 0% (0/3) [1 cluster] | 22% (2/9) [1 cluster] |
| 10h · lr 3e-5 · step 125 (best) | TTS twin | 9 | 17% (1/6) [0%-33%] | 83% (5/6) [67%-100%] | 0% (0/6) [0%-66%]* | 0% (0/3) [1 cluster] | 78% (7/9) [1 cluster] |
