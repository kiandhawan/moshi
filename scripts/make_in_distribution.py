"""In-distribution test set: rendered training-list files that no training run used.

The v2 training list has 720 rendered scripts (data/v2/train/, seed 10). train_10h_v2 takes the
first 540, and no training index (any configs/*.yaml train_data) contains the other 180. This picks
60 of those 180, keeping the training mix of 24 ordinary + 12 think time + 12 side conversation
+ 12 describe options. No file shares a building block with train_10h_v2 (which contains
train_1h_v2), the data of the models this set is for (base, 10h lr3e-5 and its onset variants).
Files sharing no block with any set (incl. the controls and the never-run moreord index) are taken
first, in list order; the rest of a type's quota is filled in list order with files whose only
shared blocks are in control_10h_v2 / train_10h_v2_moreord (control files reuse dialogue blocks of
training-list wait files); those are listed in the output and in each events file
(`shares_block_with`). These files share everything else with training: the train split's
phrasings, voices, names, topics and acoustic augmentation.

For each file:
- data/v2/in_dist/<id>.wav: the user-side audio = the right channel of the training render (as
  trained, with its augmentation), mono, 24 kHz, for run_inference.py.
- <id>.events.json: built with render.events_for (the same code as the eval events) from the
  script and the turn timings of the training events file; split "eval", groups ["in_distribution"].
  The rebuilt right_turns are checked against the original ones.
- data/indexes/in_distribution_v2.jsonl (path, duration, events).
Checks: no selected id in any training index; no shared block with any training set.
Usage: python scripts/make_in_distribution.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import sphn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render  # noqa: E402  (events_for: the eval event definitions)

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "data/v2/train"
OUT = REPO / "data/v2/in_dist"
TRAIN_SETS = ["train_1h_v2", "train_10h_v2", "control_1h_v2", "control_10h_v2", "train_10h_v2_moreord"]
QUOTA = {"ordinary": 24, "think_time": 12, "side_conversation": 12, "describe_options": 12}


def jl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def timed_turns(script, ev):
    """Script turns with start/end from the events file (same order per speaker side)."""
    ru, rm = iter(ev["right_turns"]), iter(ev["moshi_turns"])
    out = []
    for t in script["turns"]:
        tt = next(rm) if t["speaker"] == "moshi" else next(ru)
        out.append({**t, "start": tt["start"], "end": tt["end"]})
    return out


def main():
    used_ids, blocks = set(), {}
    for n in TRAIN_SETS:
        used_ids |= {Path(r["path"]).stem for r in jl(REPO / f"data/indexes/{n}.jsonl")}
        blocks[n] = {b for r in jl(REPO / f"data/manifests/{n}.jsonl") for b in r.get("block_ids") or []}
    used_blocks = blocks["train_10h_v2"] | blocks["train_1h_v2"]
    manifest = sorted(jl(TRAIN / "manifest.jsonl"), key=lambda r: int(r["id"].rsplit("_", 1)[1]))
    unused = [r for r in manifest if r["id"] not in used_ids]
    shares = {r["id"]: [n for n in TRAIN_SETS if blocks[n] & set(r.get("block_ids") or [])] for r in unused}
    picked, need = [], dict(QUOTA)
    for strict in (True, False):  # first files sharing no block with any set, then the rest
        for r in unused:
            if need.get(r["type"], 0) == 0 or r in picked or used_blocks & set(r.get("block_ids") or []):
                continue
            if strict and shares[r["id"]]:
                continue
            picked.append(r); need[r["type"]] -= 1
    assert not any(need.values()), f"not enough unused files: still need {need}"
    picked.sort(key=lambda r: int(r["id"].rsplit("_", 1)[1]))
    skipped = [(r["id"], shares[r["id"]]) for r in picked if shares[r["id"]]]

    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    for r in picked:
        i = r["id"]
        script = json.loads((REPO / f"data/v2/scripts/train/{i}.json").read_text())
        ev0 = json.loads((TRAIN / f"{i}.events.json").read_text())
        turns = timed_turns(script, ev0)
        ev = render.events_for({**script, "split": "eval", "groups": ["in_distribution"]}, turns)
        assert ev["right_turns"] == ev0["right_turns"], f"{i}: rebuilt turns differ"
        assert ev["user_start"] == ev0["user_start"] and ev["distractors"] == ev0["distractors"], i
        ev = {"id": i, "group": script["type"], **ev, "moshi_turns": ev0["moshi_turns"],
              "augmentation": ev0.get("augmentation"), "shares_block_with": shares[i],
              "notes": f"In-distribution test file: unused training-list render {i} (data/v2/train/), user side = its "
                       f"right channel as trained (augmented); events rebuilt by scripts/make_in_distribution.py."}
        x, sr = sphn.read(str(TRAIN / f"{i}.wav"))
        assert sr == render.SR and x.shape[0] == 2
        sphn.write_wav(str(OUT / f"{i}.wav"), x[1:2].astype(np.float32), sample_rate=sr)
        (OUT / f"{i}.events.json").write_text(json.dumps(ev, indent=1))
        index.append({"path": f"../v2/in_dist/{i}.wav", "duration": round(x.shape[1] / sr, 3),
                      "events": f"../v2/in_dist/{i}.events.json"})
    (REPO / "data/indexes/in_distribution_v2.jsonl").write_text("".join(json.dumps(r) + "\n" for r in index))

    # checks
    ids = {Path(r["path"]).stem for r in index}
    for n in TRAIN_SETS:
        assert not ids & {Path(r["path"]).stem for r in jl(REPO / f"data/indexes/{n}.jsonl")}, n
    assert not used_blocks & {b for r in picked for b in r["block_ids"]}
    hours = sum(r["duration"] for r in index) / 3600
    waits = sum(1 for r in picked if r["type"] != "ordinary")
    print(f"{len(index)} files ({hours:.2f} h): {dict(Counter(r['type'] for r in picked))}; {waits} waits; "
          f"ids {picked[0]['id']} .. {picked[-1]['id']}")
    print(f"{len(skipped)} picked files share a block only with sets the tested models never trained on: "
          + ", ".join(f"{i} ({', '.join(b)})" for i, b in skipped))
    print("checked: no id in", ", ".join(TRAIN_SETS), "; no building block shared with train_1h_v2 / train_10h_v2")
    print("wrote data/v2/in_dist/*.wav + *.events.json, data/indexes/in_distribution_v2.jsonl")


if __name__ == "__main__":
    main()
