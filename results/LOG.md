# Results log

Decisions about how results are scored, newest first. Metric definitions live in
`scripts/score.py` (module docstring); this file records *why* they are the way they are.

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
