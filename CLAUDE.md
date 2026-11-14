# Project context: teaching Moshi conditional silence

## Goal
Measure how much synthetic fine-tuning data it takes to teach Moshi (`kyutai/moshiko-pytorch-bf16`) **conditional silence**:
- Stay quiet when the user asks it to wait.
- Resume at the right time.
- Still answer ordinary requests normally.

The model itself must produce the silence. Do not add an external mute rule.

## Behaviors
- **Opening greeting (always, train and eval):** Moshi opens every conversation by greeting the user and asking how they are, starting at 0.3-0.8 s, before the user's first words. The user answers, then the conversation continues. Greetings come from a varied list split train/eval like other phrasings. Scoring treats Moshi's speech before the user's first line (`user_start` in the events file) as the greeting (`greeted_at_start`), never as speaking during the instruction or as an interruption.
- **Trained (all three listed behaviors):**
  - Think time: "Give me N seconds to think, then ask again." Moshi asks again at the deadline. Durations: train 5/10/20/45/60 s, eval 15/30 s.
  - Side conversation: "I'm talking to my friend, stay quiet until I say I'm back." Moshi resumes after the return cue.
  - Describe options: "Let me describe all the options before you comment." Moshi stays quiet through the pauses between options and comments after "that's all".
- **Conditions inside trained types (train and eval):**
  - Think time: in about half, the user is audible during part of the wait (mumbling, thinking aloud, talking to themselves, not addressed to Moshi). In about 15%, the user ends the wait early ("Okay, I've got it") and Moshi responds right away.
  - Ordinary conversations: about 15% contain "wait"-like phrases that are not wait instructions ("Wait, what did you say?", "Hold on, that's not right"); Moshi answers normally.
- **Held out (eval only, never in training):**
  - New wait types: taking a phone call (only the user's side is heard), someone at the door (the visitor is audible), reading something first (silence with occasional murmurs).
  - Extensions before a deadline ("Give me five more seconds").
  - Implicit returns without an explicit cue ("Anyway, where were we?").

## Data
- Stereo WAVs: **left = Moshi**, **right = user and friend mixed together**. Moshi's channel is silent during waits.
- Pipeline: an LLM writes the scripts, multi-voice TTS renders them, Moshi always uses one fixed voice, and we add realistic pauses and room noise.
- Mix in ordinary conversations.
- **Every training file must be shorter than `duration_sec`.** That way a crop can never start in the middle of a wait without the instruction that started it.
- Hold out for eval: voices (20%), phrasings (every third item of each list), topics (12 of 40), friend names and distractor phrases (a third), and the eval-only wait types and conditions above.
- Mix: train 20% think time, 20% side conversation, 20% describe options, 40% ordinary. Eval adds phone call, door and reading (10% each) and keeps 30% ordinary.
- Distractors to include:
  - questions addressed to the friend (in at least half of friend-wait scripts)
  - pauses before the user has finished
  - phrases that resemble the return cue (near-miss cues, in at least a quarter of friend-wait scripts)
  - the user's own self-talk during think-time waits
- **Content-unique sets:** within each set (1h, 10h, control, eval) no building block (context, side conversation, option set, dialogue, distractor exchange, phone call, door visit) appears in more than one file; only short phrasings, greetings and voices repeat. The 1h set is a prefix of the 10h set and control starts with the 10h set's ordinary files. `scripts/gen_datasets.sh` (`--unique`), `scripts/check_duplicates.py` for near-duplicates.
- Generator: `scripts/generate_scripts.py` (pools in `data/pools/`, splits in `data/splits/`); `scripts/check_split_leakage.py` and `scripts/script_stats.py` check the split and the rates.

## Experiments
All runs start from the same checkpoint and use the same seed.
- Baseline: the unmodified model.
- LoRA fine-tunes at **1h** and **10h** of data, each with learning rates **2e-6, 1e-5, 3e-5**.
- Control: trained only on ordinary conversations, 10h.
- The manifest tracks these separately: **unique scripts, unique audio hours, silence hours, seen hours**.

## Evaluation
- Offline streaming inference on fixed user-channel WAVs, several seeds, VAD on Moshi's channel.
- Metrics:
  - interruption rate during the wait window
  - resume latency after the cue
  - false resumes on distractors
  - ordinary-response rate
- **Eval groups:** every eval script and events file has `groups`, a list of what is new about it: `familiar_type` (trained wait type, only held-out voices/phrasings/topic), `new_type_phone`, `new_type_door`, `new_type_reading`, `extension`, `implicit_return`, `ordinary`, `human`. A file can have several. `score.py` (summary "By eval group") and `eval_loss.py` (`results/metrics/loss_eval_groups.csv`) break results down by them.
- Plus a small human-recorded test set: `data/human/*.wav` with `*.events.json`, indexed by `data/indexes/human.jsonl`.
- **Acknowledgement rule:** Moshi may give one brief acknowledgement (e.g. "Sure, take your time") within **2.0 s** after the instruction ends. Silence is required from then until the cue, and any speech after that 2 s window counts as an interruption. Implemented as `scripts/score.py --ack-grace 2.0` (default). Events files use the standardized fields `id, group, user_start, instruction_end, wait_end, resume_window, distractors, ordinary_requests` (+ optional `notes`).

## Environment and constraints
- Hardware: 2× H200 (141 GB each), running inside a Docker container.
- **Only `/` is writable, and the container's storage may not last.** Commit and push code to git regularly.
- Deadline: two days from 2026-11-12, so roughly 2026-11-14.

## Working style
- The user is new to this setup. **Explain each command in plain language before running it.**
