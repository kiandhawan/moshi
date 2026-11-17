"""Paired difference between two models on the same eval files (eval v2, 400 files, human files left out).

For each metric: rate(B) - rate(A) in percentage points with a 95% paired cluster-bootstrap CI (the
clusters are resampled once per draw and both models' counts are taken from the same clusters).
Last line: the overall behavior score (report_lib.SCORE_COMPONENTS: mean of silence, return, answer,
addressee on 0-100), B - A, all four components resampled from the same clusters in each draw.
Usage: python scripts/paired_diff.py MODEL_A MODEL_B [MODEL_B2 ...]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_lib import SCORE_COMPONENTS, cluster_of, readable, model_roles  # noqa: E402

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


def score_diff(ra, rb, ids, rng, b=5000):
    cs = sorted({cluster_of(ra[i]) for i in ids})
    arr = lambda d: np.array([d[c] for c in cs], float)  # noqa: E731
    flip = np.array([inv for *_, inv in SCORE_COMPONENTS])[:, None]
    idx = rng.integers(0, len(cs), size=(b, len(cs)))
    out = []
    for rows in (ra, rb):
        K, N = zip(*[(arr(k), arr(n)) for k, n in (counts(rows, ids, key, kind) for _, key, kind, _ in SCORE_COMPONENTS)])
        K, N = np.array(K), np.array(N)  # (4, clusters)
        def sc(ks, ns):
            rate = ks / np.where(ns > 0, ns, np.nan)
            return (100 * np.where(flip if rate.ndim == 2 else flip[:, 0], 1 - rate, rate)).mean(0)
        out.append((sc(K.sum(1), N.sum(1)), sc(K[:, idx].sum(-1), N[:, idx].sum(-1))))
    (pa, da), (pb, db) = out
    ds = db - da
    lo, hi = np.nanpercentile(ds, [2.5, 97.5])
    print(f"  {'overall score':20s} {pa:5.1f} -> {pb:5.1f}   {pb - pa:+5.1f}    [{lo:+5.1f}, {hi:+5.1f}]{'' if lo <= 0 <= hi else '  *'}")


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
        score_diff(ra, rb, ids, rng)
    print("\n* = 95% interval excludes 0")


if __name__ == "__main__":
    main()
