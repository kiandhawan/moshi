"""In-distribution vs held-out vs new wait types vs human speech, per model.

Conditions (rows of results/metrics/<model>.jsonl unless noted; eval at seed 1, human at seeds 1 2 3):
- in-distribution: the 60 unused training-list files (results/in_distribution/metrics/<model>.jsonl;
  scripts/make_in_distribution.py): train-split voices, phrasings, names, topics and augmentation,
  unseen conversations; trained wait types + ordinary (36 waits).
- held-out, trained types (only new: voices/phrasings/topics): eval files whose groups are exactly
  ["familiar_type"] or ["ordinary"], i.e. nothing else new about them.
- held-out, trained types (all): groups contain familiar_type or ordinary (also files with
  extensions, implicit returns or other-name questions).
- new wait types: groups contain new_type_phone / new_type_door / new_type_reading.
- human speech: the 3 human recordings (real voice, one speaker).
Rates % (k/n) [95% CI, cluster bootstrap over contexts]. Writes results/in_distribution/comparison.md.
Usage: python scripts/in_dist_compare.py MODEL...
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_lib import md_table, model_roles, ordered, rate_cell, readable, score_cell  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
COLS = [("wait success", "wait_success", "run"), ("interrupt rate", "interrupted", "run"),
        ("missed resume", "missed_resume", "run"), ("distractor response", "distractor_responses", "list"),
        ("ordinary response", "ordinary_answered", "list")]
NEW = {"new_type_phone", "new_type_door", "new_type_reading"}


def rows(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def conditions(m):
    main = rows(REPO / f"results/metrics/{m}.jsonl")
    ev = [r for r in main if "human" not in r["groups"]]
    g = lambda r: set(r["groups"])  # noqa: E731
    return [("in-distribution (60 unused training-list files)", rows(REPO / f"results/in_distribution/metrics/{m}.jsonl")),
            ("held-out, trained types, only voices/phrasings/topics new", [r for r in ev if g(r) in ({"familiar_type"}, {"ordinary"})]),
            ("held-out, trained types, all", [r for r in ev if g(r) & {"familiar_type", "ordinary"}]),
            ("held-out, new wait types (phone, door, reading)", [r for r in ev if g(r) & NEW]),
            ("human speech (3 recordings x 3 seeds)", [r for r in main if "human" in r["groups"]])]


def main():
    roles = model_roles()
    body = []
    for m in ordered(sys.argv[1:]):
        for name, rs in conditions(m):
            if rs:
                body.append([readable(m, roles), name, str(len(rs)), score_cell(rs)]
                            + [rate_cell(rs, k, kind) for _, k, kind in COLS])
    md = ["# In-distribution vs held-out vs human speech", "",
          "`scripts/in_dist_compare.py` (conditions in its docstring). In-distribution = rendered training-list files "
          "no run trained on (`scripts/make_in_distribution.py`): same voices, phrasings, names, topics and acoustic "
          "augmentation as training, new conversations. Held-out eval files have held-out voices/phrasings/topics and "
          "no augmentation (room noise only). Rates % (k/n) [95% CI, cluster bootstrap over contexts; * = every "
          "context had the same outcome]. Overall score: mean of silence, return, answer and addressee (0-100, "
          "`results/checkpoint_table.md`); for human speech it rests on 6 waits, 9 requests and 3 distractors.", "",
          md_table(["model", "condition", "runs", "overall score"] + [c for c, _, _ in COLS], body), ""]
    out = REPO / "results/in_distribution/comparison.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
