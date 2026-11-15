"""Plots in results/plots/ from results/metrics/ (rerun after each evaluation).

  data_size_wait_success.png, data_size_ordinary_response.png
      metric vs training data size (1 h, 10 h), one line per learning rate; baseline (gray dashed)
      and the ordinary-only controls (yellow) as references. 95% cluster-bootstrap CIs as bars.
  survival.png
      per model: fraction of waits still silent vs seconds into the wait (Kaplan-Meier: an
      interruption = score.py time_to_interruption; waits without one are censored at their
      length). Eval v2 waits, seed 1. Speech in the first 2 s (acknowledgement grace) never counts.
  eval_loss_vs_step.png, silence_prob_vs_step.png
      eval_loss.py val_loss and wait_p_silent_whole (P(silent through the entire wait), teacher-forced,
      averaged over waits) on all 403 eval_full_v2 files vs training step, per run; step 0 = base.
The data-size and survival plots use each run's best checkpoint (results/best/<run>.json).

Colors: fixed per entity (lr 2e-6 blue, 1e-5 orange, 3e-5 aqua, control yellow, base gray);
line style = data size (solid 10 h, dashed 1 h).
Usage: python scripts/plot_results.py
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_lib import cluster_counts, bootstrap_ratio, model_roles, ordered, readable, run_sort_key  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MET, OUT = REPO / "results/metrics", REPO / "results/plots"
LR_COLOR = {"2e-6": "#2a78d6", "1e-5": "#eb6834", "3e-5": "#1baf7a"}
CONTROL, BASE, INK, GRID = "#eda100", "#8a8a85", "#2b2b29", "#e4e3df"
SIZES = {"1h": 1.1, "10h": 9.9}


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.set_xlabel(xlabel, color=INK); ax.set_ylabel(ylabel, color=INK)
    ax.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b5b4ae")
    ax.tick_params(colors="#5a5955")


def rows(model):
    p = MET / f"{model}.jsonl"
    if not p.exists():
        return None
    return [r for r in map(json.loads, p.read_text().splitlines()) if "human" not in (r.get("groups") or [])]


def rate(rs, key, kind):
    k, n = cluster_counts(rs, key, kind)
    if n.sum() == 0:
        return None
    lo, hi = bootstrap_ratio(k, n)
    return k.sum() / n.sum(), lo, hi


def model_info(name):
    """(size, lr or 'control') for a run name or RUN__sNNN, None for base."""
    name = name.rsplit("__s", 1)[0]
    if name.startswith("control_"):
        return name.split("_")[1], "control"
    if name.startswith("train_"):
        return name.split("_")[1], name.split("_lr")[1]
    return None


def data_size_plot(models, key, kind, title, fname):
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    style(ax, title, "unique training audio (hours, log scale)", "rate")
    ax.set_xscale("log"); ax.set_xticks(list(SIZES.values()), ["1 h", "10 h"]); ax.minorticks_off()
    b = rate(rows("base"), key, kind) if rows("base") else None
    if b:
        ax.axhline(b[0], color=BASE, linestyle="--", linewidth=2, label=f"baseline {b[0]:.0%}")
        ax.axhspan(b[1], b[2], color=BASE, alpha=0.12, linewidth=0)
    series = {}
    for m in models:
        info = model_info(m)
        if info and rows(m):
            r = rate(rows(m), key, kind)
            if r:
                series.setdefault(info[1], []).append((SIZES[info[0]], *r))
    for lr in ["2e-6", "1e-5", "3e-5", "control"]:
        if lr not in series:
            continue
        pts = sorted(series[lr])
        x, y, lo, hi = map(np.array, zip(*pts))
        c = CONTROL if lr == "control" else LR_COLOR[lr]
        ax.errorbar(x, y, yerr=[y - lo, hi - y], color=c, linewidth=2, marker="o", markersize=8, capsize=4,
                    markeredgecolor="white", markeredgewidth=1.5, linestyle=":" if lr == "control" else "-",
                    label="control (ordinary only)" if lr == "control" else f"lr {lr}")
    ax.set_ylim(0, 1); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / fname); plt.close(fig)


def km(times, events):
    """Kaplan-Meier survival: step function points (t, S)."""
    order = np.argsort(times)
    t, e = np.asarray(times)[order], np.asarray(events)[order]
    n, s, xs, ys = len(t), 1.0, [0.0], [1.0]
    for i, (ti, ei) in enumerate(zip(t, e)):
        if ei:
            s *= 1 - 1 / (n - i)
            xs += [ti, ti]; ys += [ys[-1], s]
    xs.append(t.max()); ys.append(ys[-1])
    return xs, ys


def survival_plot(models):
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    style(ax, "Waits still silent", "seconds into the wait (from the end of the instruction)", "fraction of waits silent")
    for m in models:
        rs = rows(m)
        if not rs:
            continue
        w = [r for r in rs if r.get("wait_len") is not None]
        if not w:
            continue
        t = [r["time_to_interruption"] if r["time_to_interruption"] is not None else r["wait_len"] for r in w]
        e = [r["time_to_interruption"] is not None for r in w]
        xs, ys = km(t, e)
        info = model_info(m)
        if info is None:
            c, ls = BASE, "-"
        else:
            c = CONTROL if info[1] == "control" else LR_COLOR[info[1]]
            ls = "--" if info[0] == "1h" else "-"
        ax.plot(xs, ys, color=c, linestyle=ls, linewidth=2, label=f"{readable(m, model_roles())} (n={len(w)})")
    ax.set_ylim(0, 1.02); ax.set_xlim(0, 52)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(OUT / "survival.png"); plt.close(fig)


def step_plot(metric, title, ylabel, fname, log):
    p = MET / "loss_eval.csv"
    if not p.exists():
        return
    lr_rows = list(csv.DictReader(p.open()))
    base = next((r for r in lr_rows if r["label"] == "base" and r.get(metric)), None)
    fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=150)
    style(ax, title, "training step", ylabel)
    if log:
        ax.set_yscale("log")
    runs = sorted({r["label"].split("/")[0] for r in lr_rows if "/" in r["label"] and not r["label"].split("/")[0].endswith("_250ref")},
                  key=run_sort_key)
    for run in runs:
        info = model_info(run)
        if not info:
            continue
        pts = sorted((int(r["step"]), float(r[metric])) for r in lr_rows
                     if r["label"].startswith(run + "/") and r.get(metric))
        if not pts:
            continue
        if base:
            pts = [(0, float(base[metric]))] + pts
        x, y = zip(*pts)
        c = CONTROL if info[1] == "control" else LR_COLOR[info[1]]
        ax.plot(x, y, color=c, linestyle="--" if info[0] == "1h" else "-", linewidth=2, marker="o", markersize=6,
                markeredgecolor="white", label=readable(run))
    if base:
        ax.axhline(float(base[metric]), color=BASE, linestyle=":", linewidth=1.5, label="baseline")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / fname); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    models = ordered(["base"] + [json.loads(p.read_text())["model"] for p in (REPO / "results/best").glob("*.json")])
    data_size_plot(models, "wait_success", "run", "Wait success vs data size", "data_size_wait_success.png")
    data_size_plot(models, "ordinary_answered", "list", "Ordinary response vs data size", "data_size_ordinary_response.png")
    survival_plot(models)
    step_plot("val_loss", "Eval loss (all 403 eval_full_v2 files)", "val_loss (text + audio)", "eval_loss_vs_step.png", False)
    step_plot("wait_p_silent_whole", "P(silent through the entire wait), teacher-forced", "probability (log scale)",
              "silence_prob_vs_step.png", True)
    print("Wrote", ", ".join(p.name for p in sorted(OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
