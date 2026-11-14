# v2 content brief (for the parallel writers)

Project: synthetic spoken conversations that teach a voice assistant called **Moshi** to stay
silent when it should, and still answer when it is addressed. The lines you write are spoken
by text-to-speech, so write natural *spoken* English: contractions, short sentences, no emoji,
no stage directions, no brackets, no lists. Every item is used at most once per dataset, so
**variety matters most**: different situations, people, moods, wordings, lengths. Never reuse a
sentence, an opening, or a joke across items.

## Where and how to write
- Write only your own file `data/pools/v2/<your file>.json` (named in your task). Never touch
  other files in `data/pools/`. Put any helper scripts in your own folder
  `/tmp/claude-0/v2writers/<your file stem>/` (create it), never in a shared folder.
- Build the file in batches (load it if it exists, append, `json.dump(..., indent=1,
  ensure_ascii=False)`), so no single write is huge.
- Check with `cd /root/moshi-finetune && .venv/bin/python scripts/check_pools_v2.py data/pools/v2/<your file>.json`
  until it prints `0 problems`. Read the existing pools (`data/pools/generic.json`,
  `data/splits/phrasings.json`) first so you don't repeat them.

## Item formats
**name_questions**: the user, in the middle of talking to Moshi, turns to another person in
the room and asks them something by name. Moshi must stay silent for that exchange.
`{"id": "...", "split": "train"|"eval", "user": "...{name}...", "answer": "...", "name_pos": "start"|"middle"|"end"}`
- `{name}` exactly once (a person's first name is filled in later); `name_pos` says where it
  is: start ("{name}, did you feed the cat?", "Hey {name}, is it raining?"), middle ("The
  milk smells off. {name}, did you buy it Tuesday?", "Can you, um, {name}, grab the charger?"),
  end ("Is that the last bus, {name}?", "Pass me that pen, {name}."). Aim for roughly 40%
  start, 25% middle, 35% end.
- Mostly questions (about 75%), some requests or remarks that clearly expect that person to
  respond. Everyday, self-contained: it must fit into any conversation, so don't refer to what
  was said to Moshi. 3-20 words.
- `answer`: that person's short natural reply (1-14 words): yes/no, information, a joke, a
  grumble, a question back. (Sometimes the line is used without the answer, so the question
  must also work on its own.)
- Not allowed (they'd sound like wait instructions or return cues): wait, hold on, hang on,
  back, where were we, one sec / a second / a minute / a moment, quiet, stay, "I'm done", and
  the word Moshi.
- Vary who it is (partner, kid, parent, flatmate, coworker, guest, sibling), the setting and
  the mood (teasing, tired, urgent, sweet, annoyed, curious).

**moshi_mentions**: the user mentions Moshi *in the third person* to someone else, so Moshi
must NOT answer ("I asked Moshi earlier and it said...", "{name}, Moshi thinks we should leave
at six", "Did you know Moshi can do timers?").
`{"id": "...", "split": "train", "user": "...Moshi...", "answer": "..."|null, "name_pos": "start"|"middle"|"end"|"none"}`
- Must contain "Moshi" but never address it (no "Moshi, ..." at the start, no "..., Moshi?").
  `{name}` optional (about half the items); `name_pos` none if absent. `answer` = the other
  person's reply or null (about a quarter null). 3-20 words; no wait/hold on/hang on/where were
  we/I'm back.

**phrasings** (one file, `phrasings_v2.json`): lists of `{"id", "split", "text", "style",
"tone"}` (+ `"states_return": true|false` for the three instruction lists). Split: every third
item of each list (3rd, 6th, 9th, ...) is `"eval"`, the rest `"train"`.
- `style`: request ("Could you give me...?"), statement ("I need {n} seconds."), command
  ("Give me {n} seconds."), fragment ("{n} seconds, please." / "One minute—friend."), breakoff
  (the user starts a sentence and breaks off mid-way: "So I think the best one is— sorry, give
  me a moment, my friend's here."; write the break as an em dash), implied ("My friend just
  walked in, one minute." — the instruction is clear from the situation), friend_first (only in
  side_instruction: the user's reply to Moshi after the friend has already interrupted, e.g.
  "Hold on, Moshi." / "Sorry Moshi, give me a minute with her.").
- `tone`: polite, curt, apologetic, distracted, neutral, cheerful.
- `states_return`: whether the instruction says how the user will come back ("...then ask me
  again", "...I'll tell you when I'm back", "...I'll say when I'm done").
- Every instruction is addressed to Moshi (it may use the name "Moshi"); the user never just
  wanders off without telling Moshi.
- Return cues (side_cue, describe_cue, early_end) must say explicitly that the user is back /
  done / ready, and hand the turn back to Moshi. Never implicit returns like "Anyway, where were
  we?", "So, as I was saying", "where was I" (those are held out for evaluation).
