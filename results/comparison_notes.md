## Summary: reducing missed responses (2026-11-15)

Two ways to make 10h · lr 3e-5 (best step 125) miss fewer resumes and ordinary requests: **onset
weighting** in training (the loss on Moshi's text and audio tokens weighted ×3 or ×6 in the first
1.0 s of every Moshi turn; otherwise the same run) and a **decoding-time silence penalty** (the
silence token's logit lowered by a fixed bias at every step, a global decoding setting rather than
a mute rule). Details in `results/LOG.md`.

**Onset weighting (400 eval files, seed 1)**

| model | wait success | interrupt rate | missed resume | resume on time | distractor response | ordinary response | talk-over |
|---|---|---|---|---|---|---|---|
| base | 3% (9/280) | 94% (263/280) | 16% (45/280) | 64% (178/280) | 68% (379/557) | 72% (687/955) | 21% |
| 10h · lr 3e-5 · step 125 (best) | 49% (138/280) | 13% (36/280) | 35% (97/280) | 55% (154/280) | 15% (81/557) | 73% (700/955) | 1% |
| 10h · lr 3e-5 · step 150 (final) | 48% (134/280) | 11% (32/280) | 37% (104/280) | 54% (151/280) | 12% (67/557) | 69% (656/955) | 1% |
| 10h · lr 3e-5 · onset ×3 · step 150 (best = final) | **54%** (152/280) | 28% (78/280) | **14%** (40/280) | 75% (210/280) | 22% (125/557) | **90%** (858/955) | 3% |
| 10h · lr 3e-5 · onset ×6 · step 150 (best = final) | 46% (130/280) | 40% (113/280) | 9% (24/280) | 78% (217/280) | 30% (167/557) | 94% (901/955) | 5% |

Paired differences vs 10h · lr 3e-5 · step 125 on the same 400 files, in pp
[95% paired cluster bootstrap] (`scripts/paired_diff.py`):

| | wait success | interrupt rate | missed resume | distractor response | ordinary response |
|---|---|---|---|---|---|
| onset ×3 | +5.0 [-0.4, +10.4] | +15.0 [+10.4, +19.6] | -20.4 [-25.4, -15.4] | +7.9 [+4.9, +10.9] | +16.5 [+13.9, +19.3] |
| onset ×6 | -2.9 [-8.9, +3.2] | +27.5 [+22.5, +32.9] | -26.1 [-31.4, -20.7] | +15.4 [+12.0, +19.0] | +21.0 [+18.2, +23.9] |

Human recordings (3 files × 3 seeds): onset ×3 answers 8/9 ordinary requests (step 125: 2/9) and
succeeds on 3/6 waits (step 125: 1/6). The TTS twins give the same results (`results/human_tts/comparison.md`).

**Silence penalty on 10h · lr 3e-5 · step 125 (40-file diagnostic set, seed 1;
`results/diagnostics/pad_bias.md`)**: bias 0 / 0.5 / 1 / 1.5 / 2 gives ordinary no-reply 30 / 25 / 10 / 8 / 8%,
missed resume 44 / 26 / 18 / 9 / 12%, interrupt rate 12 / 18 / 47 / 62 / 74%, and wait success 44 / 44 / 41 / 29 / 15%
(34 waits, 73 ordinary requests). On the 37 synthetic files of that set, onset ×3 gives no-reply 16%,
missed resume 16%, interrupt rate 31% and wait success 41%.

**Which option works best.** **Onset ×3** cuts missed resumes from 35% to 14% (base: 16%) and
raises ordinary response from 73% to 90% (base: 72%). Wait success holds: 54% vs 49%, and the
+5 pp difference is not significant.
- **Cost of onset ×3.** It interrupts more (13% → 28%; base 94%) and responds to more distractors
  (15% → 22%; base 68%). Most of the increase is in other-name questions (20% → 48%) and self-talk
  (12% → 28%). It also acknowledges the wait instruction more often (15% → 43%); a brief
  acknowledgement within 2 s is allowed.
- **Onset ×6** goes further in the same direction: fewer missed resumes (9%), but more
  interruptions (40%) and distractor responses (30%), and wait success no higher (46%). It doesn't
  improve on ×3.
- **The silence penalty** at bias 0.5 roughly halves missed resumes and keeps wait success on the
  40 files. Its ordinary no-reply (25%) stays close to base, and any bias of 1 or more brings the
  interruptions back. It is cheaper (no training) but weaker than onset ×3.
- **Caveats.** Onset ×3's best step is the final one (wait success 46% → 53% → 54% at steps 100,
  125 and 150), so it may still be improving. Reply relevance of the onset models hasn't been
  judged yet. Onset ×3 combined with a small penalty hasn't been tested.
