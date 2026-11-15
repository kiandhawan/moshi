"""Compact view for chat / notes: key metrics per eval group, one row per (group, model).
Usage: python scripts/brief_table.py MODEL...   (results/metrics/<model>.jsonl; loss columns from
loss_eval_groups.csv when the model has a matching label: base, or RUN/stepN for RUN__sN)"""
import csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_lib import load_rows, rate_cell, md_table, n_files, ordered, readable, model_roles  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
GROUPS = ["ALL", "familiar_type", "new_type_phone", "new_type_door", "new_type_reading", "extension",
          "implicit_return", "other_name", "ordinary", "human"]
COLS = [("wait success", "wait_success", "run"), ("interrupt", "interrupted", "run"),
        ("resume on time", "resumed_on_time", "run"), ("missed resume", "missed_resume", "run"),
        ("distractor resp.", "distractor_responses", "list"), ("ordinary resp.", "ordinary_answered", "list"),
        ("talk-over", "talk_over", "list")]
models = load_rows(ordered(sys.argv[1:]))
roles = model_roles()
loss = {(r["label"], r["eval_group"]): r for r in csv.DictReader(open(REPO / "results/metrics/loss_eval_groups.csv"))}
loss.update({(r["label"], "ALL"): r for r in csv.DictReader(open(REPO / "results/metrics/loss_eval.csv"))})
label = lambda m: "base" if m == "base" else m.replace("__s", "/step").replace("/step0", "/step")  # noqa: E731
body = []
for g in GROUPS:
    for m, rows in models.items():
        rs = [r for r in rows if (g in r["groups"] if g == "human" else ("human" not in r["groups"] and (g == "ALL" or g in r["groups"])))]
        if not rs:
            continue
        lr = loss.get((label(m), g), {})
        p = lr.get("wait_p_silent_whole")
        body.append([g, readable(m, roles), f"`{m}`", str(n_files(rs))] + [rate_cell(rs, k, kind).split(" [")[0] for _, k, kind in COLS]
                    + [f"{float(p):.1e}" if p else "-"])
print(md_table(["eval group", "model", "run (raw)", "files"] + [c for c, _, _ in COLS] + ["P(silent whole wait)"], body))
