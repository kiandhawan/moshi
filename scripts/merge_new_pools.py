"""Merge the new blocks in data/pools/new/ into the pools (append; existing items unchanged).

data/pools/new/<topic>.json -> data/pools/topics/<topic>.json (each list extended)
data/pools/new/generic.json -> data/pools/generic.json (distractors, phone_calls, door_visits)
Refuses ID clashes. Moves data/pools/new/ to data/pools/merged_<date>/ afterwards so the blocks
are not counted twice (the near-duplicate check reads data/pools/new/ too).

Usage: python scripts/merge_new_pools.py
"""

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / "data/pools/new"


def main() -> None:
    added = {}
    for f in sorted(NEW.glob("*.json")):
        new = json.loads(f.read_text())
        dst = ROOT / ("data/pools/generic.json" if f.stem == "generic" else f"data/pools/topics/{f.stem}.json")
        old = json.loads(dst.read_text())
        keys = ("distractors", "phone_calls", "door_visits") if f.stem == "generic" else \
            ("contexts", "side_conversations", "ordinary_dialogues", "option_sets")
        have = {it["id"] for k in keys for it in old.get(k, [])}
        for k in keys:
            items = new.get(k, [])
            clash = [it["id"] for it in items if it["id"] in have]
            if clash:
                raise SystemExit(f"{f.name}: IDs already in {dst.name}: {clash[:5]}")
            old.setdefault(k, []).extend(items)
            added[k] = added.get(k, 0) + len(items)
        dst.write_text(json.dumps(old, indent=1, ensure_ascii=False) + "\n")
    out = ROOT / f"data/pools/merged_{time.strftime('%Y%m%d_%H%M')}"
    NEW.rename(out)
    print("added:", added, f"; new-block files moved to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
