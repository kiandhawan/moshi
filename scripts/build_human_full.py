"""Build eval_full files from the human recordings: the real user audio on the right
channel, and the ideal Moshi side (Moshi clone voice) on the left.

Every file opens with a Moshi greeting that asks how the user is (CLAUDE.md), placed before
the user's first words. Moshi's lines follow the acknowledgement rule (results/LOG.md): at most one short
acknowledgement starting within 2 s of the instruction end, silence until the cue, then a
reply shortly after the cue. Times were placed in the gaps of the user's speech (Fish ASR
of the recordings, 2026-11-14):
  extend10sec    2.08-4.72 "Um I don't know yet. Ask me again in 10 seconds"
                 11.20-13.04 "Actually wait 10 more seconds"
  talktoroommate 2.56-5.20 "Great thanks. Let me talk to my roommate for a few seconds"
                 6.24-7.36 "Ryan what time is it"  8.48-9.60 "12:28 a.m."
                 10.88-11.52 "Great thanks"  12.48-13.12 "Okay I'm back"
  regularconvo   4.56-6.00 "Hi I'm okay. How are you"  12.08-13.36 "What should I make
                 for dinner today"  20.40-21.68 "Do you have any easier ideas"
                 28.96-29.76 "Okay thanks"

The events (instruction times, cue, distractors, ordinary requests) are copied from
data/human/<id>.events.json. Writes data/eval_full/human_<id>.* and adds them to
data/indexes/eval_full.jsonl.

Usage: python scripts/build_human_full.py   (needs FISH_API_KEY; cached after the first run)
"""

import json
from pathlib import Path

from fishaudio import FishAudio

from render_lib import overlaps_with_user, render_full, update_index

OUT = Path("data/eval_full")
INDEX = Path("data/indexes/eval_full.jsonl")

MOSHI_LINES = {
    "extend10sec": [
        {"kind": "greeting", "start": 0.4, "text": "Hi, how are you?"},
        {"kind": "ack", "start": 5.2, "text": "Sure, take your time."},
        {"kind": "resume", "start": 22.3, "text": "Okay, it's been a little while. Have you decided yet?"},
    ],
    "talktoroommate": [
        {"kind": "greeting", "start": 0.4, "text": "Hey, how are you doing?"},
        {"kind": "ack", "start": 5.35, "text": "Sure, go ahead."},
        {"kind": "resume", "start": 13.4, "text": "Welcome back! What would you like to talk about?"},
    ],
    "regularconvo": [
        {"kind": "greeting", "start": 0.6, "text": "Hey, how's it going?"},
        {"kind": "ordinary", "start": 6.3, "text": "I'm doing great, thanks for asking! What's on your mind?"},
        {"kind": "ordinary", "start": 13.7,
         "text": "How about a chicken stir fry with vegetables? It's quick and tasty."},
        {"kind": "ordinary", "start": 22.0,
         "text": "Sure, you could make pasta with tomato sauce. It only takes fifteen minutes."},
        {"kind": "ordinary", "start": 30.1, "text": "You're welcome, enjoy your dinner!"},
    ],
}


def main() -> None:
    client = FishAudio()
    rows = []
    for name, lines in MOSHI_LINES.items():
        events = json.loads(Path(f"data/human/{name}.events.json").read_text())
        spec = {
            "id": f"human_{name}",
            "group": events["group"],
            "user_audio": f"data/human/{name}.wav",
            "moshi_lines": lines,
            "events": {k: v for k, v in events.items() if k not in ("id", "group")},
        }
        row = render_full(spec, client, OUT, INDEX.parent)
        rows.append(row)
        print(f"{spec['id']}: {row['duration']:.1f}s -> {OUT / (spec['id'] + '.wav')}")
        for w in overlaps_with_user(row, INDEX.parent):
            print(f"  WARNING {w}")
    update_index(INDEX, rows)
    print(f"Updated {INDEX}")


if __name__ == "__main__":
    main()
