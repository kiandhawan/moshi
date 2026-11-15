"""Overall behavior score of every checkpoint with behavioral results, and a plot per run.

Score (report_lib.behavior_score): mean of four 0-100 components, higher is better -
silence = 100 - interrupt rate, return = 100 - missed-resume rate, answer = ordinary response rate,
addressee = 100 - distractor response rate - on the 400 eval v2 files (seed 1; human files left out),
with a 95% bootstrap interval over contexts.
Writes results/checkpoint_table.md (one row per checkpoint: score, components, wait success, role)
and results/plots/overall_score_<run>.png: score vs step for the measured checkpoints only (markers,
95% error bars), the checkpoint the original selection rule picked (results/best/<run>.json:
highest wait success with ordinary response >= base - 10 pp) circled, and base and the two
controls' selected checkpoints as reference lines. Usage: python scripts/checkpoint_table.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_lib import (METRICS_DIR, behavior_score, md_table, model_roles, model_sort_key, rate_cell,  # noqa: E402
                        readable, run_sort_key)
from plot_results import BASE, CONTROL, INK, LR_COLOR, VARIANT_COLOR, style  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/plots"


def eval_rows(m):
    rs = [json.loads(l) for l in (METRICS_DIR / f"{m}.jsonl").read_text().splitlines() if l.strip()]
    return [r for r in rs if "human" not in (r.get("groups") or [])]


def color(run):
    var = next((v for v in VARIANT_COLOR if run.endswith("_" + v)), None)
    if var:
        return VARIANT_COLOR[var]
    if run.startswith("control"):
        return CONTROL
    lr = run.split("_lr")[1].split("_")[0] if "_lr" in run else None
    return LR_COLOR.get(lr, INK)


def main():
    roles = model_roles()
    best = {json.loads(p.read_text())["run"]: json.loads(p.read_text()) for p in (REPO / "results/best").glob("*.json")}
    models = sorted((p.stem for p in METRICS_DIR.glob("*.jsonl") if p.stem == "base" or "__s" in p.stem), key=model_sort_key)
    res = {}
    body = []
    for m in models:
        rs = eval_rows(m)
        s = behavior_score(rs)
        res[m] = s
        run, step = (m.rsplit("__s", 1)[0], int(m.rsplit("__s", 1)[1])) if "__s" in m else ("base", 0)
        b = best.get(run)
        role = []
        if b and step == b["step"]:
            role.append("selected by the rule" + (" (fallback)" if b.get("reason", "").lower().startswith("no candidate") else ""))
        if b and m == b["final_model"]:
            role.append("final")
        c = s["components"]
        body.append([readable(m, roles), f"`{m}`", str(step) if run != "base" else "-", str(len(rs)),
                     f"**{s['score']:.1f}** [{s['lo']:.1f}-{s['hi']:.1f}]",
                     f"{c['silence']:.1f}", f"{c['return']:.1f}", f"{c['answer']:.1f}", f"{c['addressee']:.1f}",
                     rate_cell(rs, "wait_success", "run"), ", ".join(role) or "-"])
    md = ["# Overall behavior score per checkpoint", "",
          "`scripts/checkpoint_table.py`. Every checkpoint with behavioral results (400 eval v2 files, seed 1; human "
          "files left out). **Overall score** = mean of four 0-100 components, higher is better: silence = 100 - "
          "interrupt rate, return = 100 - missed-resume rate, answer = ordinary response rate, addressee = 100 - "
          "distractor response rate; [95% CI, bootstrap over contexts, all four components resampled together]. "
          "Only checkpoints that were run behaviorally appear: per run, the 3 with the highest teacher-forced "
          "whole-wait silence probability + the final one (+ extra models). Role: the checkpoint the original rule "
          "picked (highest wait success with ordinary response >= base - 10 pp; fallback = highest ordinary response "
          "when none qualified) and the final one. Plots: `results/plots/overall_score_<run>.png`.", "",
          md_table(["checkpoint", "run (raw)", "step", "files", "overall score", "silence", "return", "answer",
                    "addressee", "wait success", "role"], body), ""]
    (REPO / "results/checkpoint_table.md").write_text("\n".join(md))
    print("\n".join(md))

    # plots, one per run
    refs = [("base", "base", BASE, ":")]
    for crun in ("control_1h_v2", "control_10h_v2"):
        if crun in best:
            refs.append((best[crun]["model"], f"control {crun.split('_')[1]} (selected)", CONTROL, "--" if "1h" in crun else "-."))
    runs = sorted({m.rsplit("__s", 1)[0] for m in models if "__s" in m}, key=run_sort_key)
    for run in runs:
        pts = sorted((int(m.rsplit("__s", 1)[1]), res[m]) for m in models if m.rsplit("__s", 1)[0] == run)
        fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=150)
        style(ax, f"Overall behavior score: {readable(run)}", "training step", "overall score (0-100)")
        for m, lab, c, ls in refs:
            if m in res:
                ax.axhline(res[m]["score"], color=c, linestyle=ls, linewidth=2.2 if m == "base" else 1.5,
                           zorder=3 if m == "base" else 2, label=f"{lab}: {res[m]['score']:.1f}")
        col = color(run)
        x = [p for p, _ in pts]; y = [s["score"] for _, s in pts]
        err = [[s["score"] - s["lo"] for _, s in pts], [s["hi"] - s["score"] for _, s in pts]]
        ax.errorbar(x, y, yerr=err, color=col, linewidth=2, marker="o", markersize=7, markeredgecolor="white",
                    capsize=3, label="measured checkpoints (95% CI)")
        b = best.get(run)
        if b and b["step"] in x:
            sel = res[f"{run}__s{b['step']:03d}"]
            ax.plot([b["step"]], [sel["score"]], marker="o", markersize=17, markerfacecolor="none", markeredgecolor=INK,
                    markeredgewidth=2, linestyle="none", label=f"selected by the rule: step {b['step']}")
            ax.annotate(f"selected\n{sel['score']:.1f}", (b["step"], sel["score"]), textcoords="offset points",
                        xytext=(0, -34), ha="center", fontsize=8, color=INK)
        ax.set_xlim(0, 155); ax.set_ylim(30, 100)
        ax.set_xticks(sorted(set([0, 50, 100, 150] + x)))
        ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        fig.tight_layout(); fig.savefig(OUT / f"overall_score_{run}.png"); plt.close(fig)
    print(f"Wrote results/checkpoint_table.md and {len(runs)} plots results/plots/overall_score_<run>.png")


if __name__ == "__main__":
    main()
