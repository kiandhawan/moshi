# Results log

Decisions about how results are scored, newest first. Metric definitions live in
`scripts/score.py` (module docstring); this file records *why* they are the way they are.

## 2026-11-15: 1h · lr 3e-5 · step 100 (tested live) evaluated as an extra checkpoint

The user found step 100 responsive and reasonably good at waiting in a live conversation. Full
evaluation (`results/extra_models.json` adds it, and step 90, to comparison.md; step 110's weights
were pruned and it was never run behaviorally, so only its losses exist): on the 400 eval files it
is a dip, not a peak - wait success 80: 36% -> 90: 32% -> **100: 16% (44/280)** -> 150: 25%,
because it interrupts 80% of waits (80: 44%), while it is the most responsive (ordinary response
92%, missed resumes 11%). On the 3 human files (seeds 1 2 3) it does as well as its neighbours
(wait success 3/6, interrupt 1/6, ordinary 9/9) - matching the live impression - but on their TTS
twins it interrupts 5/6 and answers all 3 distractors. With 3 files this is anecdotal, but it is
the same direction as the eval set: on synthetic (TTS) users step 100 does not wait; on the real
voice it did. Worth a larger real-voice test before trusting either.

## 2026-11-15: TTS twins of the human test set

To separate "real voice / real microphone" from "conversation content", each human recording got
a TTS twin (`scripts/make_human_tts.py` -> `data/human_tts/<name>_tts.wav` (not in git), events,
`data/indexes/human_tts.jsonl`). Whisper medium.en word timestamps (`whisper_words.json`) ->
utterances at pauses >= 0.4 s -> speakers assigned by hand (only "12:28 am" in talktoroommate is
the roommate Ryan) -> each line re-voiced by Fish s2-pro with held-out eval voices (Chloe for the
user, Ruan for Ryan), eval loudness (-21 / -25 dBFS), placed at its original onset, light room
noise (-55 dBFS), same length. Events copied (id `<name>_tts`, group `human_tts`).
- **Adjustments** (`data/human_tts/report.json`): Fish speaks more slowly than the recordings, so
  each line whose end defines an event was sped up (ffmpeg atempo, pitch kept) to end at it, capped
  at 1.35x: talktoroommate instruction 1.35x, still +0.18 s past instruction_end; "Okay, I'm back"
  1.35x, +0.15 s past wait_end; regularconvo requests 1 and 2 1.11x / 1.06x, on time; all other
  lines unchanged. Fish cost < $0.01 (13 short lines).
- **Results** (`results/human_tts/comparison.md`; 3 files x seeds 1 2 3 per model, so very wide
  intervals): wait success is the same on real and twin audio for all four models (base 0/6,
  1h 3e-5 step 80 2/6, 10h 1e-5 step 150 0/6, 10h 3e-5 step 125 1/6). Ordinary response differs
  most: 10h 3e-5 step 125 answers 2/9 real requests but 7/9 on the twin, base 6/9 vs 9/9 - the
  real recording's voice/microphone seems to make Moshi slower to answer within regularconvo's 2 s
  window. 1h 3e-5 and 10h 1e-5 go the other way on interruptions / ordinary answers. With 3 files
  none of these differences is reliable; the twins are mainly for listening
  (`listen_human_tts/`, not committed).

## 2026-11-15: Check of the lr 2e-6 quietness: real (from training), not a pipeline bug

The 2e-6 runs answered only ~34% of ordinary requests, which looked suspicious for the gentlest
rate. Checks (`results/diagnostics/quiet40.md`, `results/plots/quiet_vs_step.png`; 40-file subset
eval_full_subset_v2 incl. the 3 human files, one batch of 40 at seed 1 for every model):
- **Config / loading**: the checkpoint config.json differs from the loader defaults the base is
  built from only in `lora`, `lora_rank` (128), `lora_scaling` (2.0); no lm_gen_config on either
  side, and every output records the same sampling (temp 0.8 / text 0.7, top-k 250 / 25). Every
  checkpoint of every run has the same 337 LoRA A/B pairs, rank 128, scaling 2.0, and the update
  size ||2 B A|| is ordered by learning rate at every step (step 50: 0.82 / 2.88 / 7.02 for
  2e-6 / 1e-5 / 3e-5, 1 h).
- **Decisive controls**: a null adapter (the step-50 checkpoint with every B = 0, same config.json,
  same LoRA path) and the base model built from the checkpoint config.json without LoRA
  (`run_inference.py --config`) are both **bit-identical** to the base: 40/40 identical text-token
  streams, max audio difference 0. So the pipeline adds nothing; any change comes from trained B.
- **Batching**: 1 h 2e-6 step 50 is as quiet at --batch 1 (no-reply 52% vs base 29% at batch 1)
  as at batch 40 (70% vs 23%).
- **Step 10** (pruned, but its full eval_loss is recorded) is ~ base: wait_p_silent_whole 0.0026
  vs 0.0025, P(speak) after the cue 0.404 vs 0.404, ordinary text loss 1.859 vs 1.860.
- **Trajectory**: every run first becomes much quieter than base (no-reply 23% -> 41-75%, missed
  resumes 15% -> 41-94% by step 50), and only at 3e-5 does replying come back (1 h: no-reply 7-8%
  from step 100; 10 h: 30-34% at steps 100-125). 2e-6 and 1e-5 stay in the quiet phase for all 150
  steps; 2e-6 is the slowest version of the same path (its step 50 looks like 3e-5 at ~step 10 in
  the loss metrics). Teacher-forced loss moves much less than behavior (step 50, 2e-6: P(speak)
  after the cue 0.404 -> 0.373) - in free generation a slightly lower per-frame chance of starting
  to speak compounds, and once Moshi's own stream is silent it conditions on that silence.
Conclusion: a real training effect ("be quiet" is learned before "speak again on cue"), not a
bug. No earlier result changes.

## 2026-11-15: Best learning rate per data size: 3e-5 for both (controls use it)

Rule: highest wait success (400 eval v2 files, seed 1) of each run's best checkpoint, as long as
its ordinary response is within 10 pp of the baseline's 72% (>= 62%); if close, whole-wait silence
probability decides. Best checkpoints (results/best_checkpoints.md):

| data | lr | best step | wait success | ordinary response | P(silent whole wait) | meets limit |
|---|---|---|---|---|---|---|
| 1 h | 2e-6 | 140 | 6% (16/280) [3-9%] | 34% (326/955) | 0.128 | no |
| 1 h | 1e-5 | 150 | 22% (62/280) [18-27%] | 50% (474/955) | 0.298 | no |
| 1 h | **3e-5** | **80** | **36% (100/280) [30-41%]** | **75% (719/955)** | 0.469 | yes |
| 10 h | 2e-6 | 100 | 7% (19/280) [4-10%] | 35% (339/955) | 0.128 | no |
| 10 h | 1e-5 | 150 | 12% (35/280) [9-16%] | 40% (385/955) | 0.279 | no |
| 10 h | **3e-5** | **125** | **49% (138/280) [43-55%]** | **73% (700/955)** | 0.529 | yes |

- **1 h -> 3e-5** and **10 h -> 3e-5**: at each size it is the only learning rate whose best
  checkpoint meets the ordinary-response limit, and it also has the highest wait success, with a
  CI that does not overlap the others' (1 h: 30-41% vs <= 27%; 10 h: 43-55% vs <= 16%). Not
  close, so the silence-probability tie-break is not needed (it points the same way).
- The lower rates fail in the same way: Moshi learns to stay silent (interruptions 94% -> 33-42%)
  but not to come back (missed resumes 57-76%), and answers only 34-50% of ordinary requests.
  At 2e-6 the model barely moves in 150 steps (full-eval val_loss 0.78 vs 0.70 at 3e-5 / 10 h).
- So pair 4 = control_1h_v2 and control_10h_v2 at 3e-5 (`make_run_configs.py --control-lr
  1h=3e-5 10h=3e-5`), same 150-step schedule (checkpoints every 10 / 25 steps).
- Caveat: 3e-5 is the highest rate in the grid, so the optimum may lie above it.

## 2026-11-15: Pair 1 (lr 1e-5) evaluated - no checkpoint met the ordinary-response limit

- Baseline ordinary response 72% (687/955), so the limit is >= 62%. Candidates (top 3 by
  wait_p_silent_whole + final): 1h steps 120/130/150 -> ordinary 48-50%, wait success 18-22%;
  10h steps 100/125/150 -> ordinary 32-40%, wait success 10-12%. None qualifies, so per the
  fallback rule the best = the highest ordinary response = step 150 in both (= final), flagged
  in results/best_checkpoints.md.
- What changed: interruptions fell from 94% to 33% (1h) / 19% (10h) and distractor responses from
  68% to 17% / 11%, but Moshi now stays silent too much: missed resumes 16% -> 57% / 76%, ordinary
  response 72% -> 50% / 40%. So the gain in wait success is limited by never resuming.
- Loss curves: the 1h run overfits after ~step 40 (full eval val_loss 0.746 at step 40 -> 1.061 at
  150; ordinary text loss 1.38 -> 2.47, worse than base 1.86), while wait_p_silent_whole keeps
  rising, so the selection rule's top 3 are all late, overfit checkpoints; steps 40-50 (lowest
  val loss) were not run behaviorally (step 50's weights are kept; step 40's were pruned after its
  loss eval). The 10h run does not overfit (val_loss 0.70 from step 75 on).
- Voice: similarity to the base output 0.63 (1h) / 0.62 (10h), close to the base's own seed-to-seed
  floor 0.65; similarity to the training data's Moshi voice rose from 0.48 (base) to 0.66 / 0.65:
  the LoRA moves Moshi's voice toward the Fish-cloned training voice.

## 2026-11-15: New training schedule (150 steps, frequent checkpoints, best-checkpoint selection)

All training restarted with a new schedule. The two pair-1 runs on the 250-step schedule were
stopped at step ~220 and kept as references (`runs/train_{1h,10h}_v2_lr1e-5_250ref/`, NOTE.txt;
logs `logs/*_250ref.log`; 250 steps, 25 warm-up, checkpoint/eval every 50, default
num_ckpt_keep 3). Why: at 16 files per step, 250 steps are 66.7 epochs of the 1 h set, and the
250ref 1 h run was already overfitting early - its eval-subset loss rose 0.71 (step 50) -> 1.01
(100) -> 1.38 (150) while its training loss fell to 0.13 - so checkpoints every 50 steps were
too coarse to catch the best point, and the end of the run was past it.
- **Schedule** (`make_run_configs.py`): max_steps 150, pct_start 0.1 (15 warm-up steps, then
  cosine decay), seed 0, batch 16, duration_sec 100, LoRA as smoke; checkpoint + eval-subset loss
  every 10 steps for 1 h data (15 checkpoints) and every 25 for 10 h (6); keep all checkpoints.
  150 steps = 40 epochs of train_1h_v2 (44 h seen), 4.4 of train_10h_v2 (44 h seen).
- **Per run** (`scripts/eval_run.py`): eval_loss.py on all 403 eval_full_v2 files for every
  checkpoint (47 s each); the 3 checkpoints with the highest whole-wait silence probability
  (wait_p_silent_whole) + the final one get streaming inference (400 eval v2 files at seed 1,
  human files at seeds 1 2 3) and scoring; best = highest wait success among those whose ordinary
  response is at most 10 pp below the baseline's (ties: higher wait_p_silent_whole; if none
  qualifies, the highest ordinary response, flagged) -> `results/best/<run>.json`,
  `results/best_checkpoints.md`. Then prune: only lora.safetensors of checkpoints that are not
  the best, not a multiple of 50 and not the final are deleted, and only after their eval_loss
  rows, runs.csv row, copied training logs (`results/train_logs/<run>/`) and any scores are in
  results/. `results/runs.csv` has one row per checkpoint (steps, epochs, seen hours, losses,
  role, weights present/deleted).
- **Reports**: comparison.md = base + each run's best and final checkpoint; plots of eval loss and
  whole-wait silence probability vs step per run. `scripts/talk.sh RUN [STEP] [GPU]` serves a
  run's best checkpoint (or a given step) with moshi.server on port 8998.
- **Voice similarity floor**: base seed 1 vs seeds 2/3 on the human files = 0.650 (only 4 pairs
  have >= 3 s of speech, so a rough floor); base vs the training data's Moshi voice (Fish clone)
  = 0.482.

## 2026-11-14: Data v2 and eval v2 are final

The v2 training sets (train_1h_v2, train_10h_v2, control_1h_v2, control_10h_v2) and the eval set
(eval_v2 / eval_full_v2 / eval_full_subset_v2 + the 3 human files) are frozen: every result from
here on uses them unchanged.

## 2026-11-14: Training grid and evaluation protocol

- **Configs** (`scripts/make_run_configs.py` -> `configs/<run>.yaml`): smoke.yaml settings (seed 0,
  batch 16, duration_sec 100, LoRA rank 128 / scaling 2, base kyutai/moshiko-pytorch-bf16,
  40-file eval subset) with max_steps 250, ckpt_freq 50, eval_freq 50, pct_start 0.1 (25 warm-up
  steps, then cosine decay) and num_ckpt_keep null. train.py's default num_ckpt_keep is 3, which
  would delete the step-50/100 checkpoints; pair 1 had already started with it, so
  `scripts/archive_ckpts.sh` copied its checkpoints to `runs/<run>/checkpoints_all/` as they
  were written. Pairs run one per GPU (1 h on GPU 0, 10 h on GPU 1), `scripts/pipeline.sh`.
- **Same seen hours, different repetition.** Every file is shorter than duration_sec, so one
  sample = one whole file: 250 x 16 = 4000 file passes per run = 66.7 epochs of train_1h_v2 and
  7.4 of train_10h_v2, ~73 h of real audio either way (controls 50 / 5.7 epochs, ~57 h).
  1 h vs 10 h therefore compares the number of *unique* scripts at equal training compute.
- **Batched inference.** One stream per process ran at ~0.28x real time alone and slowed
  training ~4x when six shared the GPUs, so `run_inference.py --batch 32` streams 32 files in
  lockstep (independent conversations, same model and sampling settings; files sorted by length,
  padded with silence, outputs trimmed; seed set once per batch). 0.05x real time while sharing a
  GPU with training. All models use it for the 400 eval files (seed 1), so seeds are comparable
  across models but not with --batch 1 at the same seed; the human files stay unbatched (seeds
  1 2 3).
- **New metrics** (`score.py` docstring; reported per model by `compare.py`): time to first
  interruption (+ Kaplan-Meier survival curves, `plot_results.py`), wait success by wait length
  (5-15 / 15-30 / 30-60 s), signed resume error for timed waits (think time without an early end;
  first speech >= 0.5 s after the ack grace, so an interruption counts as an early resume), missed
  resumes (nothing within 10 s of the cue / deadline) separate from late ones, distractor
  responses by kind, talk-over (Moshi starting while the user speaks: the events' user turns, VAD
  on the user channel for the human files), ordinary latency and Moshi speech s/min; reply
  relevance (`relevance.py`: fixed sample of 50 resumes + 50 ordinary replies, judged blind by a
  Claude subagent from Moshi's own text stream vs the reference reply; instructions in
  `results/relevance/JUDGE.md`; no API key is available, so the judge is a subagent of this
  session); voice similarity (`voice_similarity.py`: ECAPA speaker embeddings, vs base on the same
  file); whole-wait silence probability (`eval_loss.py` wait_p_silent_whole / _10s). Medians get
  cluster-bootstrap CIs like the rates. "Overall" = the 400 eval files; human files are reported
  only in their own section.

## 2026-11-14: control_1h_v2

A 1 h control so the 1 h runs have an ordinary-only counterpart, like control_10h_v2 for 10 h.
`scripts/make_control_1h_v2.py`: the block-aligned (multiple of 20 files) prefix of
control_10h_v2 whose total is closest to train_1h_v2's 1.103 h -> 80 files, 1.131 h (the 60-file
cut gives 0.836 h, further off). All 80 are ordinary files of train_10h_v2 (control_10h_v2 starts
with them); 24 are also in train_1h_v2 (all of its ordinary files). `configs/control_1h_v2.yaml`
= smoke.yaml settings with this train_data; lr is null until the grid is set, so the config
fails to load instead of silently using train.py's 1e-4 default.

## 2026-11-14: Data v2

v2 of the data is a new set of folders and indexes (`data/v2/`, `data/indexes/*_v2.jsonl`,
`data/manifests/*_v2.jsonl`, `results/data_summary_v2.md`); v1 is unchanged for reference.
The design is the same as v1: content-unique sets, types in blocks of 20 with the exact mix,
train_1h_v2 the first block-aligned prefix of train_10h_v2, control_10h_v2 = the ordinary files
of train_10h_v2 + ordinary-only files, fixed seeds (train 10, control 11, eval v2 13).
Scripts: `generate_scripts_v2.py` (reuses the v1 generator's pools, decks and conditions),
`gen_datasets_v2.sh`, `render_all_v2.sh`, `build_datasets_v2.py`, `verify_augment.py`,
`check_pools_v2.py`, `check_dataloader.py`.

Training sets (train_1h_v2, train_10h_v2, control_10h_v2):
- **Acoustic augmentation, right channel only** (`render_lib.augment_settings`, seeded by file
  id, recorded per file in the manifest and events): per-speaker gain uniform in +-6 dB (user,
  friend/visitor, other person separately); background noise fan / café / street / quiet room
  (a quarter each) at -62 to -38 dBFS depending on the type (café = distant babble mixed from
  other training TTS clips + cup clinks; street = rumble + passing cars; fan = broadband + motor
  hum); mic coloring in a third (laptop: high-pass, low-pass, presence peak, soft saturation;
  phone: 250-3600 Hz band + 8-bit mu-law); light reverb on the user in a quarter. Moshi's
  channel is bit-identical to the clean render. **Why:** v1's user side was studio-clean TTS at
  fixed levels; the model should learn to stay silent from what is said, not from how clean or
  loud it sounds, and real users have rooms, mics and noise. A clean copy of every file is kept
  (`data/v2/clean/`) so the content checks run on audio without noise and
  `verify_augment.py` checks that only the right channel changed.
- **Think-time durations**: any whole number 5-60 s except 15 and 30 (still eval-only), so the
  model can't key on five fixed values; a "long" stride (40%) draws 31-60 s.
- **Long waits**: at least a third of the waits of every type last >= 30 s (think and describe
  via the long stride, with long pauses of 3.5-8 s between options; side conversations are long
  by themselves). Long early ends happen after 30 s. **Why:** v1 had only 15% of describe waits >= 30 s; long silences are
  where a model is most tempted to break in.
- **Instruction position**: early / middle / late cycling per wait type. Early = the wait right
  after the context and an ordinary dialogue afterwards; middle = one dialogue exchange before
  the context, the next one after the wait; late = a whole dialogue before. **Why:** in v1 the
  instruction always came within the first ~15 s, so "a wait instruction comes right after the
  greeting" was a usable shortcut. The summary reports the planned position and the realized
  share of conversation speech before the instruction (from the rendered timeline): the
  context's own Moshi reply puts "early" files at about 0.3-0.45, not below a third.
- **Instruction and return styles**: 279 new phrasings (`data/pools/v2/phrasings_v2.json`,
  every third held out for eval like v1) tagged with style (request, statement, command,
  fragment, mid-sentence break-off, implied, friend-first), tone (polite, curt, apologetic,
  distracted, neutral, cheerful) and whether they say how the user will return; v1 phrasings
  stay in the decks (style tagged automatically). A quarter of side conversations start with
  the friend interrupting ("Hey, can I ask you something?") and the user's friend-first
  instruction ("Hold on, Moshi."). Return cues stay explicit and hand the turn back
  (implicit returns remain eval-only). Every wait still has an instruction addressed to Moshi.
- **Addressing by name**: 310 other-name questions (`name_questions_*.json`: 240 train, 70
  eval) and 130 third-person Moshi mentions, written by parallel helpers on separate themes,
  names from a 271-name list split 2:1 train/eval (v1 friend names keep their split; eval names
  never occur in train text). Ordinary 15% and wait conversations 10% (outside the wait) get an
  other-name question; the person answers in 60%; Moshi stays silent for the exchange (a
  distractor, kind other_name) and the conversation continues. Contrasts: a user request
  addressed to Moshi by name (start, middle or end; Moshi answers) in 10% / 5%, and the user
  mentioning Moshi in the third person to someone else (Moshi silent) in 6% / 4%. **Why:**
  "is this addressed to me?" is the same skill as staying quiet for a side conversation, and the
  name contrasts stop the model from learning "any name -> silence".
- **Voices**: each v2 file keeps the v1 voice of its first building block (and a side
  conversation its v1 friend voice and name), so unchanged lines came from the TTS cache.

Eval v2 (the default and final eval set: `eval_v2.jsonl`, `eval_full_v2.jsonl`,
`eval_full_subset_v2.jsonl`): 40 of the 400 v1 eval files (stratified: 12 ordinary, 6 think, 6
side, 5 describe, 4 phone, 4 door, 3 reading) are replaced by new files of the same types and
conditions, built by the v1 eval generator from blocks no other eval file uses, each with one
other-name question (held-out names and questions) before the instruction or between turns;
group `other_name`. The other 360 files are the v1 files (re-rendered from the cache, identical
audio). No augmentation. score.py and eval_loss.py count other-name questions as distractors
(score.py also reports `other_name_responses` separately), and every distractor window now
ends at Moshi's next ideal turn (`distractor_until` in the events) as well as after 3 s. **Why
the cut:** a correct answer to the user's next request could otherwise fall inside the 3 s
after a question to someone else and be counted as a false response. The training-time subset
(40 files) now holds 5 other_name files.

Fixes made while building v2 (all before the final render):
- Think waits of 49-60 s were silently rejected by the 85 s length limit once a dialogue had
  to surround every wait (the first render's think waits topped out at 48 s). Think waits of
  46 s or more may now drop the surrounding dialogue when nothing else fits; 49-58 s waits are
  in the list. Train and control were regenerated and re-rendered.
- Name questions and Moshi mentions were not recorded in `block_ids`, so control's exclusion
  (blocks of the train list's ordinary files) could not see them; a middle-position file also
  listed its split dialogue twice. Both fixed; the builder's max-reuse check caught it.
- Leakage check: held-out phrasings of 3 words or fewer now match whole turns only (as short
  generic phrases already did): "Okay, I guess." (an eval greeting answer) matched the start
  of an ordinary train dialogue line.

Billing: all v2 render requests (logged byte counts: first train render 214 k bytes incl. the
killed start, control 42 k, eval 20 k, control re-render 29 k, train + control re-render 162 k +
26 k = 493 k bytes = $7.39 TTS, plus about $0.6 ASR on new Moshi lines) add up to about $8.0;
the credit fell from $25.21 to $13.23 = $11.98. The stages measured on their own (control
$0.73, the final re-render $2.98) matched their estimates; the extra ~$4 appeared during the
first train render's requests and is unexplained (Fish bills with a delay of minutes, so it
might include charges from before). Worth checking Fish's usage page.

Checks: `check_pools_v2.py` (format, banned wait/return words, duplicates vs v1), near-duplicate
check over all pools including v2 (0 flagged), leakage check extended to v2 phrasings, eval
names and eval name questions (it caught a held-out friend name in two train questions and a
train side-conversation line copied into an eval question; all reworded), verification of the
clean copies, `verify_augment.py`, and the data-loader check (`results/dataloader_check_v2.md`).

## 2026-11-14: Content-unique datasets

Every set is now content-unique: within train_1h, train_10h, control_10h and eval, no
building block (context, side conversation, option set, ordinary dialogue, distractor
exchange, phone call, door visit) is used in more than one file. Intended sharing stays:
train_1h is a prefix of train_10h, and control_10h starts with the ordinary files of
train_10h. Short phrasings (instructions, cues, greetings, acks, self-talk, murmurs, wait-like
phrases) and voices still come from their fixed split lists and repeat.

**Why:** with the old pools, contexts were reused up to 4 times in the 10 h set, ordinary
dialogues up to 8 times in the control set and eval contexts up to 7 times. Repeated content
inflates the "hours" of a set (the model hears the same conversation several times, which is
closer to more epochs than to more data), lets eval scores depend on a few memorable
contexts, and makes files that share a context correlated, so per-file statistics overstate
the evidence.

**How:**
- Needed blocks per set computed from the script design (train list 720 scripts, control 500
  ordinary, eval 400) plus about 10% spare; 1,820 new blocks (191k words) written by 14
  parallel helpers on separate topics, with explicit variety requirements and a list of
  overused stock phrases to avoid (`data/pools/NEW_BLOCKS.md`).
- New `welcome_back` lines carry no greeting; the generator adds an opener that fits the wait.
- `generate_scripts.py --unique`: content blocks are drawn without replacement and never
  reshuffled; a script that is rebuilt (too long/short) gives its blocks back; a set that
  runs out fails loudly. `--exclude-blocks-from` keeps the control additions off every block
  of the training list. Same seeds (train 10, control 11, eval 12) and block design.
- `check_duplicates.py`: near-duplicate check across all pools (word 3-gram containment >= 0.20
  for long blocks, calibrated on a real pair at 0.24; content-word Jaccard >= 0.5 for short
  exchanges). Flagged pairs: 1 before fixing (gardening.side1 / pets.side1, both old; none of
  the new blocks), 0 after rewriting pets.side1.
- Leakage check: one train near-miss (dn_37) contained a held-out implicit return verbatim
  and was reworded. Eval phone calls / door visits are now compared block-to-block by the
  near-duplicate check instead of line by line (everyday lines like "What should I prepare?"
  matched by coincidence), and held-out phrases of two words or fewer are ignored.
- Bootstrap CIs in score.py / eval_loss.py resample by cluster (the context, or the dialogue
  / option set), so they now match the unit of independent content.

## 2026-11-14: Eval groups

Eval files are tagged with what is new about them (events `groups`, a list): familiar_type,
new_type_phone, new_type_door, new_type_reading, extension, implicit_return, ordinary, human.
Human files: extend10sec = human, familiar_type, extension (the user extends the wait);
talktoroommate = human, familiar_type; regularconvo = human, ordinary. The old `group` field
(wait type) is kept for the existing tables. eval_loss's ordinary text loss now covers every
Moshi turn that answers an ordinary request (reply / followup in rendered files).

## 2026-11-14: Opening greeting

Moshi opens every conversation (train and eval) by greeting the user and asking how they
are, before the user's first words (see CLAUDE.md).

- Events files gain `user_start`: when the user first speaks. If it is missing, score.py uses
  the first VAD speech on the user channel.
- Moshi speech segments that **start** before `user_start` are the greeting. They are reported
  as `greeted_at_start` and excluded from every other metric, so a greeting that runs into the
  user's first words is not counted as speaking during the instruction or as an interruption.
- eval_loss.py: `greet_p_speak` = mean P(not PAD) before `user_start`; the greeting frames are
  excluded from the "should be silent" windows (they already were, via `moshi_turns`).
- talktoroommate: `wait_end` moved from 13.0 to 13.1 s, the end of the cue "Okay, I'm back"
  (Fish ASR: 12.48-13.12 s), and `resume_window` to [13.1, 18.1].

## 2026-11-13: Acknowledgement rule

After the user finishes a wait instruction, Moshi may give **one brief acknowledgement**
(e.g. "Sure, take your time") but nothing more.

- The acknowledgement must **start within 2.0 s after `instruction_end`**
  (`score.py --ack-grace 2.0`, the default). It is optional and reported as `acknowledged`.
- From `instruction_end + 2.0 s` until `wait_end`, Moshi must be silent. More than 0.5 s of
  speech in that window counts as `interrupted`, including an acknowledgement that runs past
  the 2.0 s mark.
- Speech that starts *before* `instruction_end` is not an acknowledgement. If it overlaps
  `[instruction_start, instruction_end]` it counts as `spoke_during_instruction`.

## 2026-11-13: Other scoring decisions

- **`instruction_start` added to the events format.** `spoke_during_instruction` only counts
  speech overlapping the instruction itself. Before this, Moshi's opening greeting (about
  0.5-2 s into every conversation) made this metric 100% for every model.
- **The resume definition stays strict.** `resumed_on_time` and `resume_latency` require a
  speech segment that *starts* inside the resume window or after `wait_end`. Speech already
  in progress at `wait_end` doesn't count as resuming. It is flagged as `spoke_over_cue`
  instead, e.g. Moshi talking over "Moshi, I'm back".
- **Human test set annotations** (`data/human/*.events.json`):
  - extend10sec: instruction 2.0-4.9 s. Audio before 2.0 s is noise, and the user stops
    talking at about 4.9 s.
  - talktoroommate: instruction 2.48-5.0 s. The question to Ryan at 6.24 s is a distractor.
