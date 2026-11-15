"""Index for run B (more_ordinary): train_10h_v2 with ~60% ordinary conversations at ~the same hours.

Files: the first k files of each wait type of train_10h_v2 (in its order, so the earliest blocks),
all 216 ordinary files of train_10h_v2, and the first control-only ordinary files of
control_10h_v2 (seed 11; generated to avoid the blocks of the training list's ordinary files only)
that share no building block with any selected file - the wait files of train_10h_v2 contain
dialogue blocks that some control files reuse - until ordinary = 60% of files; k is
chosen so the total audio is closest to train_10h_v2's. Writes data/indexes/train_10h_v2_moreord.jsonl,
data/manifests/train_10h_v2_moreord.jsonl and prints unique scripts, hours and the type mix.
Usage: python scripts/make_more_ordinary.py
"""
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
IDX, MAN = REPO / "data/indexes", REPO / "data/manifests"
WAITS = ["think_time", "side_conversation", "describe_options"]
ORD_SHARE = 0.60


def read(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main():
    t10, c10 = read(MAN / "train_10h_v2.jsonl"), read(MAN / "control_10h_v2.jsonl")
    t_idx = {Path(r["path"]).stem: r for r in read(IDX / "train_10h_v2.jsonl")}
    c_idx = {Path(r["path"]).stem: r for r in read(IDX / "control_10h_v2.jsonl")}
    target = sum(r["total_sec"] for r in t10)
    ords = [r for r in t10 if r["type"] == "ordinary"]
    extra = [r for r in c10 if r["id"].startswith("control_")]  # control-only, never in train_10h_v2
    assert not {r["id"] for r in extra} & {r["id"] for r in t10}
    best = None
    for k in range(1, 109):
        waits = [r for w in WAITS for r in [x for x in t10 if x["type"] == w][:k]]
        n_ord = round(ORD_SHARE / (1 - ORD_SHARE) * len(waits))
        if n_ord - len(ords) > len(extra) or n_ord < len(ords):
            continue
        used = {b for r in waits + ords for b in r.get("block_ids") or []}
        free = [r for r in extra if not used & set(r.get("block_ids") or [])]
        if n_ord - len(ords) > len(free):
            continue
        rows = waits + ords + free[: n_ord - len(ords)]
        sec = sum(r["total_sec"] for r in rows)
        if best is None or abs(sec - target) < abs(best[1] - target):
            best = (rows, sec, k)
    rows, sec, k = best
    rows.sort(key=lambda r: (r["id"].startswith("control_"), r.get("order_index", 0)))
    index = [t_idx.get(r["id"]) or c_idx[r["id"]] for r in rows]
    (IDX / "train_10h_v2_moreord.jsonl").write_text("".join(json.dumps(r) + "\n" for r in index))
    (MAN / "train_10h_v2_moreord.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    blocks = Counter(b for r in rows for b in r.get("block_ids") or [])
    types = Counter(r["type"] for r in rows)
    hrs = {t: sum(r["total_sec"] for r in rows if r["type"] == t) / 3600 for t in types}
    print(f"k = {k} wait files per type; {len(rows)} files (unique scripts), {sec / 3600:.3f} h "
          f"(train_10h_v2: {len(t10)} files, {target / 3600:.3f} h); max file {max(r['total_sec'] for r in rows):.1f} s")
    for t in WAITS + ["ordinary"]:
        print(f"  {t:18s} {types[t]:4d} files ({types[t] / len(rows):.1%}), {hrs[t]:.2f} h ({hrs[t] * 3600 / sec:.1%} of audio)")
    print(f"  ordinary from train_10h_v2: {len(ords)}, from control_10h_v2: {types['ordinary'] - len(ords)}")
    assert max(blocks.values()) == 1, "a building block repeats"
    print(f"  silence {sum(r['silence_sec'] for r in rows) / 3600:.2f} h, wait {sum(r['wait_sec'] or 0 for r in rows) / 3600:.2f} h; "
          f"max block reuse {max(blocks.values())}")


if __name__ == "__main__":
    main()
