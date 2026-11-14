"""Pick a fixed, stratified subset of eval_full for the periodic evaluation during training.

Per script type a fixed number of files (QUOTA), with the eval-only conditions represented
(EXTRA: at least this many with the condition), preferring distinct clusters (contexts /
dialogues / option sets) and voices; plus every human answer key. Deterministic (SEED).
The full 403-file eval_full_v2 set is for eval_loss.py on the saved checkpoints after training.

Writes data/indexes/eval_full_subset_v2.jsonl (from eval_full_v2; v1: eval_full_subset.jsonl). Usage: python scripts/make_eval_subset.py
"""

import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IDX = ROOT / "data/indexes"
SEED = 0
QUOTA = {"think_time": 6, "side_conversation": 6, "describe_options": 5, "phone_call": 5, "door": 5,
         "reading": 5, "ordinary": 5}
OTHER_NAME_MIN = 5  # v2: at least this many other_name files (taken within the type quotas)
EXTRA = {"think_time": ("extension", 3), "side_conversation": ("implicit_return", 2),
         "phone_call": ("implicit_return", 2), "door": ("implicit_return", 2), "reading": ("implicit_return", 1)}


def main() -> None:
    rows = [json.loads(l) for l in open(IDX / "eval_full_v2.jsonl")]  # eval v2 (final)
    for r in rows:
        r["ev"] = json.loads((IDX / r["events"]).read_text())
    human = [r for r in rows if "human" in r["ev"].get("groups", [])]
    rng = random.Random(SEED)
    chosen, used_clusters = [], set()
    for kind, n in QUOTA.items():
        pool = [r for r in rows if r["ev"].get("type") == kind and "human" not in r["ev"].get("groups", [])]
        rng.shuffle(pool)
        pool.sort(key=lambda r: r["ev"]["cluster"] in used_clusters)  # fresh clusters first (stable)
        pick = []
        on = [r for r in pool if "other_name" in r["ev"]["groups"]]
        want_on = OTHER_NAME_MIN - sum("other_name" in r["ev"]["groups"] for r in chosen)
        if want_on > 0 and on and kind in ("ordinary", "think_time", "side_conversation", "describe_options", "door"):
            pick += on[:1 if kind != "ordinary" else 2]
        if kind in EXTRA:
            g, k = EXTRA[kind]
            pick += [r for r in pool if g in r["ev"]["groups"] and r not in pick][:max(0, k - sum(g in r["ev"]["groups"] for r in pick))]
        rest = [r for r in pool if r not in pick and (kind not in EXTRA or g not in r["ev"]["groups"])]
        pick += rest[: n - len(pick)]
        chosen += pick
        used_clusters |= {r["ev"]["cluster"] for r in pick}
    sub = chosen + human
    (IDX / "eval_full_subset_v2.jsonl").write_text("".join(json.dumps({k: r[k] for k in ("path", "duration", "events")}) + "\n"
                                                        for r in sub))
    groups = Counter(g for r in sub for g in r["ev"]["groups"])
    print(f"{len(sub)} files ({len(chosen)} generated + {len(human)} human), "
          f"{len({r['ev']['cluster'] for r in sub})} distinct clusters, {sum(r['duration'] for r in sub) / 60:.1f} min audio")
    print("types:", dict(Counter(r["ev"].get("type", "human") for r in sub)))
    print("groups:", dict(groups.most_common()))


if __name__ == "__main__":
    main()
