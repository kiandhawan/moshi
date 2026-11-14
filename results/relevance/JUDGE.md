# Reply relevance: judging instructions

Each line of a `batch_<tag>.jsonl` file is one item: a spoken conversation between a user and
the voice assistant Moshi (text transcribed from speech, so expect missing punctuation,
fragments and small word errors). You get:

- `situation`: the moment being judged (Moshi resuming after a wait, or answering a request);
- `context`: the scripted conversation just before that moment (lines by the user, a friend /
  other person, and the *reference* Moshi turns);
- `reference_reply`: what an ideal Moshi says here;
- `generated_reply`: what the model being tested actually said, starting at that moment.

Decide `fits`: **true** if the generated reply is a sensible, on-topic reply at this point of
this conversation: it responds to what was just said or (for resumes) picks the conversation
back up appropriately (e.g. asking again after a think-time deadline, commenting on the options
after the user said that's all, re-engaging when the user is back). It does not have to match
the reference's wording or content; the reference shows what kind of reply is expected.
**false** if it is off-topic, ignores or contradicts what was just said, is incoherent, is only
filler ("um", "yeah" alone), or continues talking about something unrelated.

Judge each item on its own. Write one JSON line per item to `verdicts_<tag>.jsonl`:
`{"uid": "...", "fits": true, "why": "<= 15 words"}`.
