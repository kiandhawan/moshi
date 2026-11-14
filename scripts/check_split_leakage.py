"""Check that no eval-only material appears in train scripts (and vice versa for voices).

Eval-only: eval voices, eval topics (and building blocks from them), held-out phrasings,
held-out friend names and distractors. Checks IDs recorded in each script and also the
turn texts (an eval phrasing's text must not occur in any train turn).

Usage: python scripts/check_split_leakage.py --train DIR [--eval DIR]
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def norm(t: str) -> str:
    return re.sub(r"[^a-z' ]", "", t.lower()).strip()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--eval", type=Path)
    args = p.parse_args()
    voices = json.loads((ROOT / "data/splits/voices.json").read_text())
    topics = json.loads((ROOT / "data/splits/topics.json").read_text())
    ph = json.loads((ROOT / "data/splits/phrasings.json").read_text())
    gen = json.loads((ROOT / "data/pools/generic.json").read_text())
    eval_voices = {v["id"] for v in voices["eval"]}
    train_voices = {v["id"] for v in voices["train"]}
    eval_topics = set(topics["eval"])
    eval_ph = [x for k, v in ph.items() if k != "note" for x in v if x["split"] == "eval"]
    eval_gen = [x for k in ("friend_names", "distractors") for x in gen[k] if x["split"] == "eval"]
    eval_ids = {x["id"] for x in eval_ph + eval_gen}
    # texts: phrasings with slots removed become regexes; distractor user lines; names
    eval_text = []
    for x in eval_ph:
        parts = [re.escape(norm(s)) for s in re.split(r"\{n\}|\{friend\}", x["text"])]
        eval_text.append((x["id"], re.compile(r"[a-z\- ]+".join(parts))))
    for x in gen["friend_names"]:
        if x["split"] == "eval":
            eval_text.append((x["id"], re.compile(rf"\b{x['text'].lower()}\b")))
    for x in eval_gen:
        if x.get("user"):
            eval_text.append((x["id"], re.compile(re.escape(norm(x["user"].replace("{friend}", ""))))))

    problems, n = [], 0
    for f in sorted(args.train.glob("*.json")):
        if f.name == "index.jsonl":
            continue
        s = json.loads(f.read_text())
        n += 1
        if s["split"] != "train":
            problems.append(f"{f.name}: split is {s['split']}")
        for role, vid in s["voices"].items():
            if role != "moshi" and vid and vid not in train_voices:
                problems.append(f"{f.name}: {role} voice {vid} is {'an eval' if vid in eval_voices else 'an unknown'} voice")
        for t in s["topics"]:
            if t in eval_topics or t not in topics["train"]:
                problems.append(f"{f.name}: topic {t} is not a train topic")
        for b in s["block_ids"]:
            if b.split(".")[0] in eval_topics:
                problems.append(f"{f.name}: block {b} is from an eval topic")
        for i in s["phrasing_ids"] + [s.get("friend_name_id")]:
            if i in eval_ids:
                problems.append(f"{f.name}: held-out item {i}")
        if s.get("friend_name") in {x["text"] for x in gen["friend_names"] if x["split"] == "eval"}:
            problems.append(f"{f.name}: eval friend name {s['friend_name']}")
        if s["type"] == "describe_options":
            problems.append(f"{f.name}: describe_options is eval only")
        for k, t in enumerate(s["turns"]):
            for iid, rx in eval_text:
                if rx.search(norm(t["text"])):
                    problems.append(f"{f.name}[{k}]: text matches held-out {iid}: {t['text']!r}")
            if t.get("reference_only"):
                problems.append(f"{f.name}[{k}]: reference_only in a train script")
    m = 0
    if args.eval:
        for f in sorted(args.eval.glob("*.json")):
            s = json.loads(f.read_text())
            m += 1
            for role, vid in s["voices"].items():
                if role != "moshi" and vid and vid not in eval_voices:
                    problems.append(f"{f.name}: {role} voice {vid} is not an eval voice")
            for t in s["topics"]:
                if t not in eval_topics:
                    problems.append(f"{f.name}: topic {t} is not an eval topic")
            if any(t["speaker"] == "moshi" and not t.get("reference_only") for t in s["turns"]):
                problems.append(f"{f.name}: Moshi turn without reference_only")
    for x in problems:
        print("LEAK", x)
    print(f"checked {n} train and {m} eval scripts: {len(problems)} problems")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
