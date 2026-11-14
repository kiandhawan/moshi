"""Mix of script types, distractor rates per wait type, and condition rates.

For each script directory: share of each type (and wait types vs ordinary); per type, the
share of scripts containing each distractor kind (and the mean count per script); the share
with each condition (self_talk, early_end, extension, implicit, near_miss, wait_like, ...).

Usage: python scripts/script_stats.py DIR [DIR ...]
"""

import glob
import json
import sys
from collections import Counter, defaultdict

KINDS = ["friend_question", "question_to_other", "near_miss", "pause", "self_talk"]


def main() -> None:
    for d in sys.argv[1:]:
        fs = [json.load(open(f)) for f in sorted(glob.glob(f"{d}/*_*.json"))]
        n = len(fs)
        types = Counter(s["type"] for s in fs)
        waits = sum(c for t, c in types.items() if t != "ordinary")
        print(f"== {d}: {n} scripts, wait types {waits / n:.0%}, ordinary {types['ordinary'] / n:.0%}")
        print("   " + ", ".join(f"{t} {c} ({c / n:.0%})" for t, c in types.most_common()))
        by = defaultdict(list)
        for s in fs:
            by[s["type"]].append(s)
        hdr = "".join(f"{k:>19s}" for k in KINDS)
        print(f"   {'distractors: % of scripts with >=1 (mean count)':52s}")
        print(f"   {'type':18s}{hdr}")
        for t, ss in sorted(by.items()):
            cells = []
            for k in KINDS:
                c = [sum(1 for tn in s["turns"] if tn.get("event") == "distractor" and tn.get("kind") == k) for s in ss]
                cells.append(f"{sum(x > 0 for x in c) / len(ss):5.0%} ({sum(c) / len(ss):.1f})")
            print(f"   {t:18s}" + "".join(f"{c:>19s}" for c in cells))
        print("   conditions: % of scripts of the type")
        for t, ss in sorted(by.items()):
            cs = Counter(c for s in ss for c in s.get("conditions", []))
            if cs:
                print(f"   {t:18s} " + ", ".join(f"{c} {v / len(ss):.0%} ({v})" for c, v in sorted(cs.items())))
        print()


if __name__ == "__main__":
    main()
