"""Cut control_1h_v2: the block-aligned prefix of control_10h_v2 closest in audio to train_1h_v2.

Same rule as the other cuts (build_datasets.cut: prefix length a multiple of 20 files, total
closest to the target). Target = train_1h_v2's total audio. control_10h_v2 starts with the
ordinary files of train_10h_v2, so control_1h_v2 shares its first files with train_1h_v2.
Writes data/indexes/control_1h_v2.jsonl and data/manifests/control_1h_v2.jsonl (rows copied
from the control_10h_v2 manifest). Reads only; changes no other file.

Usage: python scripts/make_control_1h_v2.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IDX, MAN = ROOT / "data/indexes", ROOT / "data/manifests"
BLOCK = 20


def read(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.open()]


def cut(rows: list[dict], target: float) -> int:
    """Length of the prefix (a multiple of BLOCK) whose total duration is closest to target."""
    best, cum = None, 0.0
    for i, r in enumerate(rows):
        cum += r["duration"]
        if (i + 1) % BLOCK == 0 and (best is None or abs(cum - target) < best[1]):
            best = (i + 1, abs(cum - target))
    return best[0]


def main() -> None:
    c10, t1 = read(IDX / "control_10h_v2.jsonl"), read(IDX / "train_1h_v2.jsonl")
    target = sum(r["duration"] for r in t1)
    n = cut(c10, target)
    c1 = c10[:n]
    man = {r["id"]: r for r in read(MAN / "control_10h_v2.jsonl")}
    ids = [Path(r["path"]).stem for r in c1]
    with (IDX / "control_1h_v2.jsonl").open("w") as f:
        f.writelines(json.dumps(r) + "\n" for r in c1)
    with (MAN / "control_1h_v2.jsonl").open("w") as f:
        f.writelines(json.dumps(man[i]) + "\n" for i in ids)
    shared = set(ids) & {Path(r["path"]).stem for r in t1}
    hrs = sum(r["duration"] for r in c1) / 3600
    print(f"target {target / 3600:.4f} h (train_1h_v2, {len(t1)} files)")
    print(f"control_1h_v2: {n} files, {hrs:.4f} h, shared with train_1h_v2: {len(shared)}, "
          f"from train_10h_v2 ordinary: {sum(i.startswith('trainv2_') for i in ids)}, "
          f"control-only: {sum(i.startswith('control_') for i in ids)}, "
          f"silence {sum(man[i]['silence_sec'] for i in ids) / 3600:.3f} h")
    for alt in (0.9,):
        m = cut(c10, alt * 3600)
        print(f"(for reference: a {alt} h target would give {m} files, "
              f"{sum(r['duration'] for r in c10[:m]) / 3600:.4f} h)")


if __name__ == "__main__":
    main()
