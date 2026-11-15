"""Extra measurements for results/REPORT.md -> results/diagnostics/report_extra.md.

1. Reply to the greeting answer: in how many training / control files does Moshi answer the user's
   reply to its opening greeting (a Moshi turn between the greeting answer and the user's next line),
   and on the human file regularconvo (request 1 = "Hi, I'm okay. How are you?") per seed.
2. Ordinary response by request onset in the conversation (< 20 s, 20-40 s, >= 40 s; eval v2, seed 1):
   does responsiveness fall later in a conversation?
3. Timed returns: think-time waits with a deadline (no early end; eval v2 deadlines 15 / 30 s, plus
   extensions): first speech after the ack grace vs the deadline (score.py resume_error): within
   [-3, +3] s, early (< -3 s, i.e. interrupted before the deadline), late (> +3 s), none.
4. Other-name questions by context: in ordinary conversations vs in wait conversations.
Rates: % (k/n) [95% CI, cluster bootstrap over contexts; * = all contexts had the same outcome, Wilson over contexts]. Usage: python scripts/report_extra.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_lib import bootstrap_ratio, cluster_of, md_table, model_roles, readable, wilson  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MODELS = ["base", "train_1h_v2_lr3e-5__s080", "train_10h_v2_lr3e-5__s125", "train_10h_v2_lr3e-5_onsetw3__s150",
          "control_10h_v2__s150"]


def cell(pairs):
    """pairs: list of (cluster, hit) -> '% (k/n) [lo-hi]'."""
    if not pairs:
        return "-"
    k, n = defaultdict(int), defaultdict(int)
    for c, h in pairs:
        k[c] += bool(h); n[c] += 1
    cs = sorted(n)
    K, N = np.array([k[c] for c in cs], float), np.array([n[c] for c in cs], float)
    k_, n_ = int(K.sum()), int(N.sum())
    if len(cs) < 2:
        ci = "[1 cluster]"
    elif k_ in (0, n_):  # bootstrap collapses: Wilson over clusters, marked * (as report_lib.rate_cell)
        lo, hi = wilson(k_ * len(cs) // n_, len(cs)); ci = f"[{lo:.0%}-{hi:.0%}]*"
    else:
        lo, hi = bootstrap_ratio(K, N); ci = f"[{lo:.0%}-{hi:.0%}]"
    return f"{k_ / n_:.0%} ({k_}/{n_}) {ci}"


def rows(m):
    return [json.loads(l) for l in (REPO / f"results/metrics/{m}.jsonl").read_text().splitlines() if l.strip()]


def events(i):
    return json.loads((REPO / f"data/v2/eval_full/{i}.events.json").read_text())


def main():
    roles = model_roles()
    md = ["# Extra measurements for the report", "", "`scripts/report_extra.py` (docstring for definitions). "
          "Rates % (k/n) [95% CI, cluster bootstrap over contexts; * = every context had the same outcome, Wilson interval over contexts].", ""]

    # 1. greeting answer in the training data
    body = []
    for name in ["train_1h_v2", "train_10h_v2", "control_1h_v2", "control_10h_v2"]:
        n = rep = 0
        gaps = []
        for l in (REPO / f"data/indexes/{name}.jsonl").read_text().splitlines():
            e = json.loads((REPO / "data/indexes" / json.loads(l)["events"]).read_text())
            ga = [t for t in e.get("right_turns") or [] if t.get("event") == "greeting_answer"]
            if not ga:
                continue
            g = ga[0]; n += 1
            nxt = min([t["start"] for t in e["right_turns"] if t["start"] > g["end"]] or [1e9])
            rep += any(m["kind"] != "greeting" and g["end"] - 0.2 <= m["start"] < nxt for m in e["moshi_turns"])
            gaps.append(nxt - g["end"])
        body.append([name, str(n), str(rep), f"{np.median(gaps):.2f} s"])
    md += ["## 1. Reply to the greeting answer", "",
           "Training data: files with a greeting answer, files where Moshi replies to it, median gap from the "
           "greeting answer's end to the user's next line.", "",
           md_table(["set", "files with a greeting answer", "Moshi replies to it", "median gap to the user's next line"], body), ""]
    body = []
    for m in MODELS:
        for src, lab in (("results/metrics", "real"), ("results/human_tts/metrics", "TTS twin")):
            p = REPO / f"{src}/{m}.jsonl"
            if not p.exists():
                continue
            rs = sorted([json.loads(l) for l in p.read_text().splitlines()
                         if l.strip() and json.loads(l)["id"].startswith("regularconvo")], key=lambda r: r["seed"])
            hits = [bool(r["ordinary_answered"][0]) for r in rs]
            body.append([readable(m, roles), lab, f"{sum(hits)}/{len(hits)}",
                         f"{sum(sum(r['ordinary_answered'][1:]) for r in rs)}/{sum(len(r['ordinary_answered'][1:]) for r in rs)}"])
    md += ["Human file regularconvo (seeds 1 2 3; anecdotal, one recording): request 1 is the answer to Moshi's "
           "greeting (\"Hi, I'm okay. How are you?\"), requests 2-3 are ordinary questions.", "",
           md_table(["model", "audio", "answers the greeting answer", "answers requests 2-3"], body), ""]

    # 2. ordinary response by request onset
    bins = [(0, 20, "< 20 s"), (20, 40, "20-40 s"), (40, 1e9, ">= 40 s")]
    body = []
    evc = {}
    for m in MODELS:
        acc = defaultdict(list)
        for r in rows(m):
            if "human" in r["groups"]:
                continue
            e = evc.setdefault(r["id"], events(r["id"]))
            for t, a in zip(e.get("ordinary_requests") or [], r.get("ordinary_answered") or []):
                b = next(lab for lo, hi, lab in bins if lo <= t < hi)
                acc[b].append((cluster_of(r), a))
        body.append([readable(m, roles)] + [cell(acc[lab]) for _, _, lab in bins])
    md += ["## 2. Ordinary response by when the request comes", "",
           "Eval v2 (400 files, 35-85 s long, seed 1): share of ordinary requests answered, by the request's time "
           "in the conversation. The eval files are at most 85 s, so this cannot show behavior in sessions of "
           "several minutes.", "",
           md_table(["model"] + [lab for _, _, lab in bins], body), ""]

    # 3. timed returns
    body = []
    for m in MODELS:
        out = {"plain": defaultdict(list), "extension": defaultdict(list)}
        for r in rows(m):
            if "human" in r["groups"] or not r.get("timed_wait"):
                continue
            g = "extension" if "extension" in (r.get("conditions") or []) else "plain"
            err = r.get("resume_error")
            cat = "none" if err is None else ("on time" if -3 <= err <= 3 else ("early" if err < -3 else "late"))
            for c in ("on time", "early", "late", "none"):
                out[g][c].append((cluster_of(r), cat == c))
        for g in ("plain", "extension"):
            body.append([readable(m, roles), "deadline" if g == "plain" else "deadline + extension"]
                        + [cell(out[g][c]) for c in ("on time", "early", "late", "none")])
    md += ["## 3. Timed returns (think time with a deadline)", "",
           "First Moshi speech after the 2 s ack grace relative to the deadline (the last deadline for extensions): "
           "on time = within 3 s; early = more than 3 s before (an interruption); late = more than 3 s after; "
           "none = no speech in the file after the grace.", "",
           md_table(["model", "waits", "on time (+-3 s)", "early", "late", "none"], body), ""]

    # 4. other-name questions by context
    body = []
    for m in MODELS:
        acc = defaultdict(list)
        for r in rows(m):
            if "human" in r["groups"]:
                continue
            ctx = "ordinary conversation" if r["type"] == "ordinary" else "wait conversation"
            for a in r.get("other_name_responses") or []:
                acc[ctx].append((cluster_of(r), a))
        body.append([readable(m, roles), cell(acc["ordinary conversation"]), cell(acc["wait conversation"])])
    md += ["## 4. Responses to questions addressed to another person by name", "",
           "Share of other-name questions (eval: held-out names) that Moshi responds to (it should stay silent), "
           "by the kind of conversation they occur in.", "",
           md_table(["model", "in ordinary conversations", "in wait conversations"], body), ""]

    out = REPO / "results/diagnostics/report_extra.md"
    out.write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
