"""Split voices_all.json into train/eval voices (about 20% eval, balanced by gender and
accent) and write the think-time durations per split.

Voices are grouped by (gender, accent); walking through the groups in order, every fifth
voice goes to eval (a running counter, so small groups are spread across the split rather
than all landing in train). The Moshi clone voice (data/voices/moshi_voice.json) is used
for every Moshi line in both splits and is not part of this list.

Writes data/splits/voices.json and data/splits/waits.json.
Usage: python scripts/make_splits.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

EVAL_FRACTION = 0.2
THINK_SECONDS = {"train": [5, 10, 20, 45, 60], "eval": [15, 30]}
EXTENSION_SECONDS = [5, 10]  # eval only: "give me five more seconds"


def main() -> None:
    voices = json.loads(Path("data/splits/voices_all.json").read_text())["voices"]
    groups = defaultdict(list)
    for v in voices:
        groups[(v["gender"], v["accent"])].append(v)
    split, acc = {}, 0.0
    for key in sorted(groups):
        for v in groups[key]:
            acc += EVAL_FRACTION
            if acc >= 1.0:
                split[v["id"]], acc = "eval", acc - 1.0
            else:
                split[v["id"]] = "train"
    out = {"note": "Voice split (scripts/make_splits.py). Moshi always uses the clone in "
                   "data/voices/moshi_voice.json.",
           "moshi": json.loads(Path("data/voices/moshi_voice.json").read_text())["id"]}
    for s in ("train", "eval"):
        out[s] = [{k: v[k] for k in ("id", "title", "gender", "accent", "age")}
                  for v in voices if split[v["id"]] == s]
    Path("data/splits/voices.json").write_text(json.dumps(out, indent=1) + "\n")
    Path("data/splits/waits.json").write_text(json.dumps(
        {"think_seconds": THINK_SECONDS, "extension_seconds": EXTENSION_SECONDS}, indent=1) + "\n")
    for s in ("train", "eval"):
        g = Counter(v["gender"] for v in out[s])
        a = Counter(v["accent"] for v in out[s])
        print(f"{s}: {len(out[s])} voices, {dict(g)}, accents {dict(sorted(a.items()))}")


if __name__ == "__main__":
    main()
