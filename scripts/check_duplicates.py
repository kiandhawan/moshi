"""Near-duplicate check across all building-block pools.

Every block (context, side conversation, option set, ordinary dialogue, distractor exchange,
phone call, door visit) from data/pools/topics/*.json, data/pools/generic.json and, if present,
data/pools/new/*.json is turned into its words. Two blocks are flagged when
  - long blocks (>= 20 words): the share of the smaller block's word 3-grams that also occur
    in the other (containment) is >= SHINGLE_T, or
  - short blocks (distractor exchanges, < 20 words): the content-word sets (stopwords removed)
    have Jaccard >= SHORT_T and at least 3 content words in common.
Blocks are compared across kinds and topics. Candidate pairs come from an inverted index of
3-grams / content words, so every pair that could pass a threshold is checked.

Usage: python scripts/check_duplicates.py [--show N] [--json OUT]
"""

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHINGLE_T, SHORT_T, SHORT_WORDS = 0.20, 0.5, 20  # calibrated: 0.24 = same storyline beat for beat
STOP = set("""a an the and or but so to of in on at for with from by is are was were be been am i you he she
it we they me my your our their this that these those do does did not no yes it's i'm you're don't can
could would should will just really very too also then there here what which who how when where why
okay oh well um like get got have has had if as about up out""".split())
TOPIC_KEYS = ("contexts", "side_conversations", "option_sets", "ordinary_dialogues")


def text_of(kind: str, it: dict) -> str:
    parts = [t["text"] for t in it.get("turns", [])]
    parts += it.get("think_reask", []) + it.get("welcome_back", [])
    parts += [e["user"] + " " + e["moshi"] for e in it.get("endings", [])]
    parts += [it.get("lead_in", "")] + it.get("options", []) + it.get("moshi_comment", [])
    parts += it.get("lines", [])
    if kind == "distractors":
        parts += [it["user"], it["friend"]]
    return " ".join(p for p in parts if p).replace("{friend}", "")


def blocks(include_new: bool = True) -> list[tuple[str, str, str, str]]:
    out = []
    files = sorted(glob.glob(str(ROOT / "data/pools/topics/*.json")))
    if include_new:
        files += sorted(glob.glob(str(ROOT / "data/pools/new/*.json")))
    for f in files:
        d = json.load(open(f))
        src = "new" if "/new/" in f else "pool"
        if Path(f).stem == "generic":
            keys = ("distractors", "phone_calls", "door_visits")
        else:
            keys = TOPIC_KEYS
        for k in keys:
            for it in d.get(k, []):
                out.append((it["id"], k, text_of(k, it), src))
    gen = json.load(open(ROOT / "data/pools/generic.json"))
    for k in ("distractors", "phone_calls", "door_visits"):
        for it in gen[k]:
            out.append((it["id"], k, text_of(k, it), "pool"))
    return out


def words(t: str) -> list[str]:
    return re.findall(r"[a-z']+", t.lower())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--show", type=int, default=15)
    p.add_argument("--json", type=Path, default=None, help="Write the flagged pairs here.")
    p.add_argument("--no-new", action="store_true", help="Only the existing pools (ignore data/pools/new/).")
    args = p.parse_args()
    bl = blocks(not args.no_new)
    ids = [b[0] for b in bl]
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        print(f"WARNING: duplicate IDs across pools: {dup[:10]}")
    sh, cw = [], []
    for _, _, t, _ in bl:
        w = words(t)
        sh.append({" ".join(w[i:i + 3]) for i in range(len(w) - 2)} if len(w) >= SHORT_WORDS else set())
        cw.append({x for x in w if x not in STOP} if len(w) < SHORT_WORDS else set())
    inv = defaultdict(list)
    for i, s in enumerate(sh):
        for g in s:
            inv[g].append(i)
    cand = defaultdict(int)
    for lst in inv.values():
        if len(lst) > 200:  # very common 3-grams carry no signal and would explode the pairs
            continue
        for a in range(len(lst)):
            for b in range(a + 1, len(lst)):
                cand[(lst[a], lst[b])] += 1
    flagged = []
    for (i, j), c in cand.items():
        m = min(len(sh[i]), len(sh[j]))
        if m and c / m >= SHINGLE_T:
            flagged.append((c / m, i, j, "3-gram containment"))
    inv2 = defaultdict(list)
    for i, s in enumerate(cw):
        for g in s:
            inv2[g].append(i)
    cand2 = defaultdict(int)
    for lst in inv2.values():
        for a in range(len(lst)):
            for b in range(a + 1, len(lst)):
                cand2[(lst[a], lst[b])] += 1
    for (i, j), c in cand2.items():
        jac = c / len(cw[i] | cw[j])
        if c >= 3 and jac >= SHORT_T:
            flagged.append((jac, i, j, "content-word Jaccard"))
    flagged.sort(reverse=True)
    print(f"{len(bl)} blocks ({sum(b[3] == 'new' for b in bl)} new); flagged pairs: {len(flagged)} "
          f"(3-gram containment >= {SHINGLE_T} for >= {SHORT_WORDS} words; content-word Jaccard >= {SHORT_T} for shorter)")
    by = defaultdict(int)
    for _, i, j, _ in flagged:
        by[f"{bl[i][1]} / {bl[j][1]}"] += 1
    for k, v in sorted(by.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    for s, i, j, how in flagged[: args.show]:
        print(f"  {s:.2f} {how:22s} {bl[i][0]} <> {bl[j][0]}")
    if args.json:
        args.json.write_text(json.dumps([{"score": round(s, 3), "how": h, "a": bl[i][0], "b": bl[j][0],
                                          "a_new": bl[i][3] == "new", "b_new": bl[j][3] == "new"}
                                         for s, i, j, h in flagged], indent=1))


if __name__ == "__main__":
    main()
