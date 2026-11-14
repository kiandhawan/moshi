# Writing new building blocks (content-unique datasets)

Every training and eval file must now use its own building blocks, so the pools need many
more items. Read `data/pools/README.md` first: the JSON format and the text rules there
still apply. This file adds what is different for the new items.

## Where to write
- Topic blocks: `data/pools/new/<topic>.json`, a JSON object like a topic file:
  `{"topic": "<topic>", "contexts": [...], "side_conversations": [...], "ordinary_dialogues": [...], "option_sets": [...]}`
  containing **only the new items**. Never edit `data/pools/topics/*.json`.
- Build the file in several steps (e.g. with a short Python snippet that loads the file if it
  exists, appends a batch of items, and writes it back with `json.dump(..., indent=1,
  ensure_ascii=False)`), so no single write is huge.
- Check with `cd /root/moshi-finetune && .venv/bin/python scripts/check_pools.py --new <topic>`
  until it prints `0 problems`.

## How many and which IDs (per topic)
| | train topic | eval topic |
|---|---|---|
| contexts | 9: `<topic>.ctx4` .. `.ctx12` | 19: `.ctx4` .. `.ctx22` |
| side_conversations | 3: `.side4` .. `.side6` | 3: `.side4` .. `.side6` |
| option_sets | 4: `.opt3` .. `.opt6` | 3: `.opt3` .. `.opt5` |
| ordinary_dialogues | 28: `.dlg5` .. `.dlg32` | 9: `.dlg4` .. `.dlg12` |

## One change to the format
`welcome_back` lines must contain **no greeting** ("Welcome back", "Hi again", "There you
are", "Good to have you back", "You're back"). The generator adds an opener that fits how the
user was away (a phone call, the door, reading, a side conversation). Write only the part
that refers back to the open question, e.g. "So, the blue tiles or the grey ones?" or
"We'd got as far as the two flights. Leaning one way yet?". `think_reask` lines are asked
again after thinking time and also need no greeting.

## Variety (most important)
Each block will be heard once per dataset, so make them genuinely different from each other
and from the existing items in `data/pools/topics/<topic>.json` (read them first):
- **Situations:** vary the concrete problem, place, time of day, stakes, and who else is
  involved (partner, kids, parents, flatmates, coworkers, neighbours, strangers).
- **People:** vary age, life stage, job, confidence, and how much they know about the topic.
- **Moods and relationships:** cheerful, stressed, tired, sarcastic, shy, excited, grumpy,
  polite, blunt; with friends: teasing, caring, impatient, awkward, warm.
- **Speaking styles:** short clipped turns vs. rambling ones, questions vs. statements,
  hesitations, self-corrections, interruptions of one's own thought, casual vs. formal.
- **Lengths:** vary turn counts and lengths within the allowed ranges.
- **Moshi:** helpful and concise, but not formulaic: vary how it answers, asks back, agrees
  or gently disagrees.
- **Avoid stock phrases and repeated jokes.** These are already overused in the pools, do
  not use them: "welcome back", "you were choosing between", "we were talking about",
  "we were planning your", "the first one is", "the last one is", "and the last one",
  "option three is", "have you thought about", "what do you think", "good to have you back",
  "there you are", "hi again". Do not reuse the same opening line, joke, or punchline twice.
- No two of your blocks may share a sentence or the same overall storyline.
