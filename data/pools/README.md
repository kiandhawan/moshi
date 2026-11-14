# Building-block pools

Hand-written content that `scripts/generate_scripts.py` combines into conversation scripts.
Nothing here is audio; TTS happens later (`scripts/render_lib.py`).

- `generic.json`: topic-free items (Moshi greetings and acknowledgements, friend names,
  distractors, extension phrases). Each item has a `split` (`train`, `eval` or `both`).
- `topics/<topic>.json`: one file per topic in `data/splits/topics.json`. Every item in it
  belongs to that topic, and so to that topic's split. The ID prefix is the topic name.

## Speakers
- `user`: the person talking to Moshi.
- `moshi`: the voice assistant, a friendly, concise conversational AI. Moshi's turns are
  1-3 short sentences, spoken style, no lists or markdown.
- `friend`: someone in the room with the user (only in side conversations). In text, the
  friend's name is written `{friend}` and filled in by the generator.

## Text rules (all items)
- Spoken English, the way people actually talk. Contractions are fine.
- No digits: write numbers as words ("twenty minutes", "half past six").
- No symbols other than . , ? ! ' and -. No emojis, quotes, parentheses or ellipses.
- No real people, brands are fine sparingly, no profanity.
- Side conversations never address Moshi and never use return-cue wording
  ("I'm back", "where were we", "okay Moshi", "sorry about that").
- In `ordinary_dialogues`, a user turn may contain one ` / ` marking a natural mid-sentence
  pause (the user hesitates and then continues). Moshi must not reply at that point.
  Use it in about one user turn per dialogue.

## Topic file format (`topics/<topic>.json`)
```json
{
  "topic": "cooking",
  "contexts": [
    {
      "id": "cooking.ctx1",
      "turns": [
        {"speaker": "user", "text": "I want to cook something nice tonight but I'm out of ideas."},
        {"speaker": "moshi", "text": "How about a pasta dish? Would you rather have something creamy or something with tomatoes?"}
      ],
      "pending": "creamy or tomato pasta",
      "think_reask": ["So, have you decided? Creamy pasta or tomato?", "..."],
      "welcome_back": ["Welcome back! You were picking a pasta for tonight, creamy or tomato?", "..."],
      "endings": [
        {"user": "Let's go creamy, maybe a carbonara.", "moshi": "Great choice. You'll need eggs, cheese and some bacon."},
        {"user": "...", "moshi": "..."}
      ]
    }
  ],
  "side_conversations": [
    {"id": "cooking.side1", "turns": [
      {"speaker": "user", "text": "{friend}, do we still have garlic?"},
      {"speaker": "friend", "text": "I think there's one bulb left in the cupboard."}
    ]}
  ],
  "ordinary_dialogues": [
    {"id": "cooking.dlg1", "turns": [
      {"speaker": "user", "text": "..."},
      {"speaker": "moshi", "text": "..."}
    ]}
  ],
  "option_sets": []
}
```

- `contexts`: a short user-Moshi exchange (2-4 turns, about 15-45 words in total) that
  ends with a **Moshi turn asking the user a question or leaving a decision open**. The
  generator follows it with a wait instruction ("give me ten seconds to think, then ask me
  again" or "hold on, I need to talk to my roommate"). `think_reask` (2-3 lines): what
  Moshi says when the thinking time is up, referring back to the open question.
  `welcome_back` (2-3 lines): what Moshi says when the user returns from a side
  conversation, also referring back. `endings` (2): the user's answer to the open question
  and Moshi's brief reply. Each ending must fit after any of the reask/welcome-back lines.
- `side_conversations`: the user and the friend talking to each other, 4-12 turns,
  about 30-130 words (10-45 seconds spoken), on this topic, starting with the user. Use
  `{friend}` in about half of them. Include at least one question from the user to the
  friend.
- `ordinary_dialogues`: a normal user-Moshi conversation on this topic, starting with the
  user, 6-12 turns, about 90-200 words. The user asks for things or chats, Moshi answers
  helpfully. Every Moshi turn follows a user turn.
- `option_sets` (eval topics only): `{"id", "lead_in", "options", "moshi_comment"}`.
  `lead_in`: one user line saying they need help choosing. `options`: 3-4 user lines, each
  describing one option in 10-25 words ("The first one is ..."). `moshi_comment`: 2
  alternative Moshi replies that compare the options, 2-3 sentences, referring to them.
