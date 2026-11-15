"""Paired difference between two models on the same eval files (eval v2, 400 files, human files left out).

For each metric: rate(B) - rate(A) in percentage points with a 95% paired cluster-bootstrap CI (the
clusters are resampled once per draw and both models' counts are taken from the same clusters).
Usage: python scripts/paired_diff.py MODEL_A MODEL_B [MODEL_B2 ...]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_lib import cluster_of, readable, model_roles  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
METRICS = [("wait success", "wait_success", "run"), ("interrupt rate", "interrupted", "run"),
           ("missed resume", "missed_resume", "run"), ("resume on time", "resumed_on_time", "run"),
           ("distractor response", "distractor_responses", "list"), ("ordinary response", "ordinary_answered", "list"),
           ("talk-over", "talk_over", "list")]


def load(m):
    rs = [json.loads(l) for l in (REPO / f"results/metrics/{m}.jsonl").read_text().splitlines() if l.strip()]
    return {r["id"]: r for r in rs if "human" not in r["groups"] and r.get("seed", 1) == 1}


def counts(rows, ids, key, kind):
    k, n = defaultdict(int), defaultdict(int)
    for i in ids:
        r = rows[i]
        vals = ([bool(r[key])] if r.get(key) is not None else []) if kind == "run" else [bool(v) for v in r.get(key) or []]
        c = cluster_of(r)
        k[c] += sum(vals); n[c] += len(vals)
    return k, n


def main():
    a, bs = sys.argv[1], sys.argv[2:]
    roles = model_roles()
    ra = load(a)
    rng = np.random.default_rng(0)
    for b in bs:
        rb = load(b)
        ids = sorted(set(ra) & set(rb))
        print(f"\n{readable(b, roles)}  minus  {readable(a, roles)}  ({len(ids)} paired files)")
        for name, key, kind in METRICS:
            ka, na = counts(ra, ids, key, kind)
            kb, nb = counts(rb, ids, key, kind)
            cs = sorted(c for c in na if na[c] or nb[c])
            KA, NA = np.array([ka[c] for c in cs]), np.array([na[c] for c in cs])
            KB, NB = np.array([kb[c] for c in cs]), np.array([nb[c] for c in cs])
            d = KB.sum() / NB.sum() - KA.sum() / NA.sum()
            idx = rng.integers(0, len(cs), size=(5000, len(cs)))
            ds = KB[idx].sum(1) / np.maximum(NB[idx].sum(1), 1) - KA[idx].sum(1) / np.maximum(NA[idx].sum(1), 1)
            lo, hi = np.percentile(ds, [2.5, 97.5])
            flag = "" if lo <= 0 <= hi else "  *"
            print(f"  {name:20s} {KA.sum() / NA.sum():5.0%} -> {KB.sum() / NB.sum():5.0%}   {d * 100:+5.1f} pp "
                  f"[{lo * 100:+5.1f}, {hi * 100:+5.1f}]{flag}")
    print("\n* = 95% interval excludes 0")


if __name__ == "__main__":
    main()
