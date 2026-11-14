"""Aggregation and table formatting shared by score.py and compare.py.

Rows are the per-run dicts written to results/metrics/<model>.jsonl (one per file and seed).
Run-level metrics (booleans) are counted over runs; list metrics (one boolean per distractor
or ordinary request) are counted over those events. All rates come with a 95% Wilson
confidence interval.
"""

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
METRICS_DIR = REPO / "results" / "metrics"

# (label, row key, kind): kind "run" = one boolean per run, "list" = list of booleans per run.
PRIMARY = [
    ("wait success", "wait_success", "run"),
    ("interrupt rate", "interrupted", "run"),
    ("resume on time", "resumed_on_time", "run"),
    ("distractor response", "distractor_responses", "list"),
    ("ordinary response", "ordinary_answered", "list"),
]
SECONDARY = [
    ("greeted at start", "greeted_at_start", "run"),
    ("spoke during instr.", "spoke_during_instruction", "run"),
    ("acknowledged", "acknowledged", "run"),
    ("spoke over cue", "spoke_over_cue", "run"),
]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def count(rows: list[dict], key: str, kind: str) -> tuple[int, int]:
    if kind == "run":
        vals = [bool(r[key]) for r in rows if r.get(key) is not None]
    else:
        vals = [bool(v) for r in rows for v in r.get(key) or []]
    return sum(vals), len(vals)


def fmt_rate(k: int, n: int) -> str:
    if n == 0:
        return "-"
    lo, hi = wilson(k, n)
    return f"{k / n:.0%} ({k}/{n}) [{lo:.0%}-{hi:.0%}]"


def n_files(rows: list[dict]) -> int:
    return len({r["id"] for r in rows})


def latency_cell(rows: list[dict]) -> str:
    lat = [r["resume_latency"] for r in rows if r.get("resume_latency") is not None]
    return f"{statistics.median(lat):.2f}s (n={len(lat)})" if lat else "-"


def iou_cell(rows: list[dict]) -> str:
    ious = [r["vad_vs_text"]["iou"] for r in rows if r.get("vad_vs_text", {}).get("iou") is not None]
    return f"{statistics.mean(ious):.2f}" if ious else "-"


def load_rows(models: list[str] | None = None) -> dict[str, list[dict]]:
    """{model: rows} from results/metrics/<model>.jsonl (all models if None), in the given order."""
    paths = ([METRICS_DIR / f"{m}.jsonl" for m in models] if models
             else sorted(METRICS_DIR.glob("*.jsonl")))
    out = {}
    for p in paths:
        if not p.exists():
            raise SystemExit(f"No results for model '{p.stem}': {p} not found")
        out[p.stem] = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return out


def by_group(rows: list[dict]) -> dict[str, list[dict]]:
    g: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        g[r["group"]].append(r)
    return dict(sorted(g.items()))


def md_table(header: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def summary_markdown(models: dict[str, list[dict]]) -> str:
    """Primary and secondary tables with one row per (model, group)."""
    prim, sec = [], []
    for model, rows in models.items():
        for group, rs in by_group(rows).items():
            prim.append([model, group, str(n_files(rs)), str(len(rs))]
                        + [fmt_rate(*count(rs, key, kind)) for _, key, kind in PRIMARY])
            sec.append([model, group]
                       + [fmt_rate(*count(rs, key, kind)) for _, key, kind in SECONDARY]
                       + [latency_cell(rs), iou_cell(rs)])
    return ("## Primary metrics\n\n"
            "Rates: % (k/n) [95% Wilson CI]. Run-level rates are over runs (files x seeds); "
            "distractor and ordinary rates are over individual events x seeds.\n\n"
            + md_table(["model", "group", "files", "runs"] + [lbl for lbl, _, _ in PRIMARY], prim)
            + "\n\n## Secondary metrics\n\n"
            + md_table(["model", "group"] + [lbl for lbl, _, _ in SECONDARY]
                       + ["resume latency median", "VAD/text IoU"], sec) + "\n")


def comparison_markdown(models: dict[str, list[dict]]) -> str:
    """One table per group: metrics as rows, models as columns."""
    names = list(models)
    groups = sorted({r["group"] for rows in models.values() for r in rows})
    parts = ["Rates: % (k/n) [95% Wilson CI]. Run-level rates are over runs (files x seeds); "
             "distractor and ordinary rates are over individual events x seeds.\n"]
    for group in groups:
        per = {m: [r for r in models[m] if r["group"] == group] for m in names}
        body = [["files"] + [str(n_files(per[m])) if per[m] else "-" for m in names],
                ["runs"] + [str(len(per[m])) if per[m] else "-" for m in names]]
        for lbl, key, kind in PRIMARY:  # always shown, "-" when nothing to count
            body.append([lbl] + [fmt_rate(*count(per[m], key, kind)) for m in names])
        for lbl, key, kind in SECONDARY:  # shown only when some model has data
            cells = [fmt_rate(*count(per[m], key, kind)) for m in names]
            if any(c != "-" for c in cells):
                body.append([lbl] + cells)
        body.append(["resume latency median"] + [latency_cell(per[m]) for m in names])
        parts.append(f"## {group}\n\n" + md_table(["metric"] + names, body) + "\n")
    return "\n".join(parts)
