# Results log

Decisions about how results are scored, newest first. Metric definitions live in
`scripts/score.py` (module docstring); this file records *why* they are the way they are.

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
