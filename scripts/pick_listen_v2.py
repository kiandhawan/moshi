"""Pick 8 random v2 training files to listen to (fixed seed): 2 per wait type and 2 ordinary,
including 2 with heavy augmentation and 2 with other-name questions. Copies them (augmented
stereo; left = Moshi, right = user side) to listen_sample/v2/ with a README describing each.

Usage: python scripts/pick_listen_v2.py [--index data/indexes/train_10h_v2.jsonl]
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_datasets_v2 import heaviness, realized_position  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TYPES = ["think_time", "side_conversation", "describe_options", "ordinary"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=Path, default=ROOT / "data/indexes/train_10h_v2.jsonl")
    p.add_argument("--out", type=Path, default=ROOT / "listen_sample/v2")
    p.add_argument("--seed", type=int, default=2)
    args = p.parse_args()
    rows = [json.loads(l) for l in args.index.read_text().splitlines() if l.strip()]
    for r in rows:
        r["ev"] = json.loads((args.index.parent / r["events"]).read_text())
        r["id"] = Path(r["path"]).stem
        r["heavy"] = heaviness(r["ev"]["augmentation"]) >= 2
        r["other"] = "other_name" in (r["ev"].get("distractor_kinds") or [])
    rng = random.Random(args.seed)
    for _ in range(10000):  # random draw until the constraints hold (2 heavy, 2 other-name)
        pick = [r for t in TYPES for r in rng.sample([x for x in rows if x["ev"]["type"] == t], 2)]
        if sum(r["heavy"] for r in pick) >= 2 and sum(r["other"] for r in pick) >= 2:
            break
    else:
        raise SystemExit("no draw satisfies the constraints")
    args.out.mkdir(parents=True, exist_ok=True)
    lines = ["# v2 listening sample", "", f"8 random files from `{args.index.relative_to(ROOT)}` (seed {args.seed}): "
             "2 per wait type and 2 ordinary. Stereo: left = Moshi (clean), right = user side (augmented).", ""]
    for k, r in enumerate(pick, 1):
        src = args.index.parent / r["path"]
        dst = args.out / f"{k}_{r['ev']['type']}_{r['id']}.wav"
        shutil.copy(src, dst)
        ev, a = r["ev"], r["ev"]["augmentation"]
        s = json.loads((ROOT / "data/v2/scripts" / ("train" if r["id"].startswith("trainv2") else "control")
                        / f"{r['id']}.json").read_text())
        mic = a["mic"]["type"] if a["mic"] else "none"
        lines += [f"## {k}. {dst.name} ({r['duration']:.1f} s)", "",
                  f"- {ev['type']}, conditions: {', '.join(ev['conditions']) or 'none'}"
                  + (f"; wait {ev['instruction_end']:.1f}-{ev['wait_end']:.1f} s ({ev['wait_sec']:.0f} s); "
                     f"instruction position {s.get('position_plan')} (share of talk before: {realized_position(ev):.2f})"
                     if ev.get("instruction_end") is not None else ""),
                  f"- augmentation{' (heavy)' if r['heavy'] else ''}: noise {a['noise']['type']} {a['noise']['dbfs']} dBFS, "
                  f"mic {mic}, user reverb {'yes' if a['user_reverb'] else 'no'}, gains user {a['gain_db']['user']:+} / "
                  f"friend {a['gain_db']['friend']:+} / other {a['gain_db']['other']:+} dB", "", "```"]
        for t in s["turns"]:
            tag = t.get("kind") or t.get("event") or t.get("role") or ""
            lines.append(f"{t['speaker']:7s} {('[' + tag + ']') if tag else '':20s} {t['text']}")
        lines += ["```", ""]
    (args.out / "README.md").write_text("\n".join(lines) + "\n")
    for k, r in enumerate(pick, 1):
        print(f"{k}. {r['ev']['type']:18s} {r['id']} heavy={r['heavy']} other_name={r['other']}")


if __name__ == "__main__":
    main()
