"""Report of the decoding-time silence penalty sweep (scripts/pad_bias_sweep.sh):
results/diagnostics/pad_bias.md and results/plots/pad_bias_tradeoff.png.

40-file diagnostic set (eval_full_subset_v2: 34 waits, 73 ordinary requests, 3 human files
included), one batch of 40 at seed 1. Bias b = the text padding token's (PAD) logit lowered by b at
every decoding step (run_inference.py --pad-bias). Bias 0 = the existing diagnostic run of the
model (results/diagnostics/quiet40/). Rates % (k/n) [95% CI, cluster bootstrap].
Usage: python scripts/pad_bias_report.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_lib import cluster_counts, md_table, rate_cell, readable, model_roles  # noqa: E402
from plot_results import INK, style  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MODELS = ["train_1h_v2_lr3e-5__s080", "train_10h_v2_lr3e-5__s125"]
BIASES = [0.0, 0.5, 1.0, 1.5, 2.0]
COLS = [("ordinary no-reply", "ordinary_answered", "list", True), ("missed resume", "missed_resume", "run", False),
        ("interrupt rate", "interrupted", "run", False), ("wait success", "wait_success", "run", False)]


def load(m, b):
    p = (REPO / f"results/diagnostics/quiet40/{m}.jsonl") if b == 0 else \
        (REPO / f"results/diagnostics/pad_bias/{m}__pb{b:g}.jsonl")
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else None


def cell(rs, key, kind, invert):
    if not invert:
        return rate_cell(rs, key, kind)
    k, n = cluster_counts(rs, key, kind)  # no-reply = 1 - answered: count the unanswered
    import numpy as np
    from report_lib import bootstrap_ratio
    nk = n - k
    lo, hi = bootstrap_ratio(nk, n)
    return f"{nk.sum() / n.sum():.0%} ({int(nk.sum())}/{int(n.sum())}) [{lo:.0%}-{hi:.0%}]"


def rate(rs, key, kind, invert):
    k, n = cluster_counts(rs, key, kind)
    return ((n.sum() - k.sum()) if invert else k.sum()) / n.sum()


def main():
    roles = model_roles()
    body, curves = [], {}
    for m in MODELS:
        for b in BIASES:
            rs = load(m, b)
            if rs is None:
                body.append([readable(m, roles), f"{b:g}"] + ["(missing)"] * len(COLS))
                continue
            body.append([readable(m, roles), f"{b:g}"] + [cell(rs, k, kind, inv) for _, k, kind, inv in COLS]
                        + [f"{sum(r['moshi_speech_sec'] for r in rs) / sum(r['duration_sec'] for r in rs) * 60:.1f}"])
            curves.setdefault(m, []).append((b, [rate(rs, k, kind, inv) for _, k, kind, inv in COLS]))
    base = json.loads("[" + ",".join((REPO / "results/diagnostics/quiet40/base.jsonl").read_text().split("\n")[:-1]) + "]")
    body.insert(0, ["base", "0"] + [cell(base, k, kind, inv) for _, k, kind, inv in COLS]
                + [f"{sum(r['moshi_speech_sec'] for r in base) / sum(r['duration_sec'] for r in base) * 60:.1f}"])
    md = ["# Decoding-time silence penalty (PAD logit bias)", "",
          "`scripts/pad_bias_sweep.sh` + `scripts/pad_bias_report.py`. The 40-file diagnostic set (eval_full_subset_v2: 34 "
          "waits, 73 ordinary requests, 3 human files included), one batch of 40 at seed 1. At every decoding step the text "
          "padding token's logit is lowered by the bias before sampling (`run_inference.py --pad-bias`): a global decoding "
          "setting applied identically everywhere, not a mute rule - nothing looks at the conversation. Bias 0 = the "
          "existing diagnostic run. no-reply = ordinary requests with no Moshi speech start within 5 s.", "",
          md_table(["model", "PAD bias", "ordinary no-reply", "missed resume", "interrupt rate", "wait success",
                    "Moshi speech s/min"], body), ""]
    (REPO / "results/diagnostics/pad_bias.md").write_text("\n".join(md))
    print("\n".join(md))
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), dpi=140, sharex=True)
    colors = {"train_1h_v2_lr3e-5__s080": "#1baf7a", "train_10h_v2_lr3e-5__s125": "#2a78d6"}
    styles = {"train_1h_v2_lr3e-5__s080": "--", "train_10h_v2_lr3e-5__s125": "-"}
    for j, (title, *_r) in enumerate(COLS):
        ax = axes[j]
        style(ax, title, "PAD logit bias", "rate" if j == 0 else "")
        for m, pts in curves.items():
            x = [b for b, _ in pts]; y = [v[j] for _, v in pts]
            ax.plot(x, y, color=colors[m], linestyle=styles[m], linewidth=2, marker="o", markersize=6,
                    markeredgecolor="white", label=readable(m, roles))
        ax.axhline(rate(base, COLS[j][1], COLS[j][2], COLS[j][3]), color="#8a8a85", linestyle=":", linewidth=1.5, label="base")
        ax.set_ylim(0, 1); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Decoding-time silence penalty: trade-off (40 files, seed 1)", x=0.01, ha="left", color=INK)
    fig.tight_layout(); fig.savefig(REPO / "results/plots/pad_bias_tradeoff.png"); plt.close(fig)
    print("Wrote results/diagnostics/pad_bias.md, results/plots/pad_bias_tradeoff.png")


if __name__ == "__main__":
    main()
