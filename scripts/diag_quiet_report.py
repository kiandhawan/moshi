"""Report of the 2e-6 quietness check (scripts/diag_quiet.sh): results/diagnostics/quiet40.md and
results/plots/quiet_vs_step.png.

Behavioral (40-file subset eval_full_subset_v2, one batch of 40 at seed 1 for every model; human
files included): no-reply on ordinary requests (no Moshi speech start within 5 s of the request
end = 1 - ordinary_answered) and missed resumes (no speech start within 10 s of the cue /
deadline), per kept checkpoint; step 0 = base. Loss-based (all 403 eval_full_v2 files, every
checkpoint including pruned ones): wait_p_silent_whole from results/metrics/loss_eval.csv.
Usage: python scripts/diag_quiet_report.py
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_lib import md_table, rate_cell, cluster_counts, model_sort_key, readable  # noqa: E402
from plot_results import LR_COLOR, BASE, style  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
D = REPO / "results/diagnostics/quiet40"
RUNS = [f"train_{s}_v2_lr{lr}" for s in ("1h", "10h") for lr in ("2e-6", "1e-5", "3e-5")]


def load(m):
    p = D / f"{m}.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else None


def rate(rows, key, kind, invert=False):
    k, n = cluster_counts(rows, key, kind)
    k, n = int(k.sum()), int(n.sum())
    return ((n - k) if invert else k), n


def main():
    base = load("base")
    body = []
    def row(name, rs):
        nr_k, nr_n = rate(rs, "ordinary_answered", "list", invert=True)
        body.append([name, f"{nr_k}/{nr_n} ({nr_k / nr_n:.0%})", rate_cell(rs, "missed_resume", "run").split(" [")[0],
                     rate_cell(rs, "interrupted", "run").split(" [")[0], rate_cell(rs, "wait_success", "run").split(" [")[0],
                     f"{sum(r['moshi_speech_sec'] for r in rs) / sum(r['duration_sec'] for r in rs) * 60:.1f}"])
    for m in ["base", "base__b1"]:
        if load(m):
            row(m + (" (batch 1)" if m.endswith("__b1") else " (batch 40)"), load(m))
    series = {}
    for run in RUNS:
        for p in sorted(D.glob(f"{run}__s*.jsonl"), key=lambda p: model_sort_key(p.stem.removesuffix("__b1"))):
            m = p.stem
            rs = load(m)
            row(readable(m.removesuffix("__b1")) + (" (batch 1)" if m.endswith("__b1") else ""), rs)
            if m.endswith("__b1"):
                continue
            step = int(m.rsplit("__s", 1)[1])
            series.setdefault(run, []).append((step, rate(rs, "ordinary_answered", "list", True), rate(rs, "missed_resume", "run")))
    loss = list(csv.DictReader(open(REPO / "results/metrics/loss_eval.csv")))
    lb = next(r for r in loss if r["label"] == "base")
    md = ["# 2e-6 quietness check (40-file subset)", "",
          "`scripts/diag_quiet.sh` + `scripts/diag_quiet_report.py`. Every model: the 40 files of eval_full_subset_v2 "
          "(34 waits, 73 ordinary requests; 3 human files included) in one batch of 40 at seed 1, so batching and "
          "sampling are the same for all; `(batch 1)` = the same files one at a time. no-reply = ordinary requests with "
          "no Moshi speech start within 5 s.", "",
          md_table(["model", "ordinary no-reply", "missed resume", "interrupt", "wait success", "Moshi speech s/min"], body), ""]
    (REPO / "results/diagnostics/quiet40.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    fig, axes = plt.subplots(3, 2, figsize=(10, 9.5), dpi=140, sharex="col")
    b_nr, b_mr = rate(base, "ordinary_answered", "list", True), rate(base, "missed_resume", "run")
    for j, size in enumerate(("1h", "10h")):
        for i, (title, ylab) in enumerate((("No reply to ordinary requests (40 files)", "rate"),
                                           ("Missed resumes (40 files)", "rate"),
                                           ("P(silent through the entire wait), 403 files", "probability (log)"))):
            ax = axes[i, j]
            style(ax, f"{size}: {title}", "training step" if i == 2 else "", ylab)
            for run in RUNS:
                if f"_{size}_" not in run:
                    continue
                lr = run.split("_lr")[1]
                if i < 2:
                    pts = [(0, (b_nr if i == 0 else b_mr))] + [(s, (nr if i == 0 else mr)) for s, nr, mr in sorted(series.get(run, []))]
                    x = [s for s, _ in pts]; y = [k / n for _, (k, n) in pts]
                else:
                    pts = sorted((int(r["step"]), float(r["wait_p_silent_whole"])) for r in loss
                                 if r["label"].startswith(run + "/") and r.get("wait_p_silent_whole"))
                    x = [0] + [s for s, _ in pts]; y = [float(lb["wait_p_silent_whole"])] + [v for _, v in pts]
                ax.plot(x, y, color=LR_COLOR[lr], linewidth=2, marker="o", markersize=5, markeredgecolor="white", label=f"lr {lr}")
            if i < 2:
                ax.axhline((b_nr if i == 0 else b_mr)[0] / (b_nr if i == 0 else b_mr)[1], color=BASE, linestyle="--", linewidth=1.5, label="base")
                ax.set_ylim(0, 1); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
            else:
                ax.set_yscale("log")
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(REPO / "results/plots/quiet_vs_step.png"); plt.close(fig)
    print("Wrote results/diagnostics/quiet40.md, results/plots/quiet_vs_step.png")


if __name__ == "__main__":
    main()
