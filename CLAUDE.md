# Project context: teaching Moshi conditional silence

## Goal
Measure how much synthetic fine-tuning data it takes to teach Moshi (`kyutai/moshiko-pytorch-bf16`) **conditional silence**:
- Stay quiet when the user asks it to wait.
- Resume at the right time.
- Still answer ordinary requests normally.

The model itself must produce the silence. Do not add an external mute rule.

## Behaviors
- **Opening greeting (always, train and eval):** Moshi opens every conversation by greeting the user and asking how they are, starting at 0.3-0.8 s, before the user's first words. The user answers, then the conversation continues. Greetings come from a varied list split train/eval like other phrasings. Scoring treats Moshi's speech before the user's first line (`user_start` in the events file) as the greeting (`greeted_at_start`), never as speaking during the instruction or as an interruption.
- **Trained:**
  - "Give me N seconds to think, then ask again."
  - "I'm talking to my friend, stay quiet until I say I'm back."
- **Held out (eval only, never in training):** "Let me describe all the options before you comment."

## Data
- Stereo WAVs: **left = Moshi**, **right = user and friend mixed together**. Moshi's channel is silent during waits.
- Pipeline: an LLM writes the scripts, multi-voice TTS renders them, Moshi always uses one fixed voice, and we add realistic pauses and room noise.
- Mix in ordinary conversations.
- **Every training file must be shorter than `duration_sec`.** That way a crop can never start in the middle of a wait without the instruction that started it.
- Hold out voices, phrasings, and the one wait type above for eval.
- Distractors to include:
  - questions addressed to the friend
  - pauses before the user has finished
  - phrases that resemble the return cue

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
- Plus a small human-recorded test set: `data/human/*.wav` with `*.events.json`, indexed by `data/indexes/human.jsonl`.
- **Acknowledgement rule:** Moshi may give one brief acknowledgement (e.g. "Sure, take your time") within **2.0 s** after the instruction ends. Silence is required from then until the cue, and any speech after that 2 s window counts as an interruption. Implemented as `scripts/score.py --ack-grace 2.0` (default). Events files use the standardized fields `id, group, user_start, instruction_end, wait_end, resume_window, distractors, ordinary_requests` (+ optional `notes`).

## Environment and constraints
- Hardware: 2× H200 (141 GB each), running inside a Docker container.
- **Only `/` is writable, and the container's storage may not last.** Commit and push code to git regularly.
- Deadline: two days from 2026-11-12, so roughly 2026-11-14.

## Working style
- The user is new to this setup. **Explain each command in plain language before running it.**
