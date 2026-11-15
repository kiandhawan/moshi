"""Aggregation and table formatting shared by score.py and compare.py.

Rows are the per-run dicts written to results/metrics/<model>.jsonl (one per file and seed).
Run-level metrics (booleans) are counted over runs; list metrics (one boolean per distractor
or ordinary request) are counted over those events. All rates come with a 95% confidence
interval from a cluster bootstrap: whole clusters (events "cluster": the context of a wait,
the dialogue of an ordinary file, the option set of describe-options, each human file on its
own) are resampled with replacement, so files and seeds that share a context move together.
"""

import json
import math
import statistics

import numpy as np
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
METRICS_DIR = REPO / "results" / "metrics"


def model_dir(model: str) -> Path:
    """Inference outputs of a model name: base -> runs/base/infer; RUN__sNNN (checkpoint at step NNN
    of RUN) -> runs/RUN/infer/sNNN; anything else -> runs/<model>/infer. Holds eval/ and human/."""
    if "__s" in model:
        run, step = model.rsplit("__s", 1)
        return REPO / "runs" / run / "infer" / f"s{step}"
    return REPO / "runs" / model / "infer"


def model_label(model: str) -> str:
    """The eval_loss.py label of a model name: base, or RUN/stepN."""
    if "__s" in model:
        run, step = model.rsplit("__s", 1)
        return f"{run}/step{int(step)}"
    return model

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
    ("other-name response", "other_name_responses", "list"),
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


BOOT_B, BOOT_SEED = 2000, 0


def cluster_of(r: dict) -> str:
    return r.get("cluster") or r["id"]


def bootstrap_ratio(num: np.ndarray, den: np.ndarray, b: int = BOOT_B, seed: int = BOOT_SEED) -> tuple[float, float]:
    """95% percentile CI of sum(num) / sum(den), resampling clusters (one entry per cluster)
    with replacement. NaN if fewer than 2 clusters."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    if len(num) < 2:
        return (math.nan, math.nan)
    idx = np.random.default_rng(seed).integers(0, len(num), size=(b, len(num)))
    d = den[idx].sum(1)
    est = num[idx].sum(1)[d > 0] / d[d > 0]
    return (float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5)))


def cluster_counts(rows: list[dict], key: str, kind: str) -> tuple[np.ndarray, np.ndarray]:
    k: dict[str, int] = defaultdict(int)
    n: dict[str, int] = defaultdict(int)
    for r in rows:
        if kind == "run":
            vals = [bool(r[key])] if r.get(key) is not None else []
        else:
            vals = [bool(v) for v in r.get(key) or []]
        c = cluster_of(r)
        k[c] += sum(vals)
        n[c] += len(vals)
    cs = [c for c in n if n[c]]
    return np.array([k[c] for c in cs]), np.array([n[c] for c in cs])


def rate_cell(rows: list[dict], key: str, kind: str) -> str:
    """'% (k/n) [95% CI]' with the cluster bootstrap; '[1 cluster]' when no CI is possible."""
    kc, nc = cluster_counts(rows, key, kind)
    k, n = int(kc.sum()), int(nc.sum())
    if n == 0:
        return "-"
    if len(nc) < 2:
        ci = "[1 cluster]"
    elif k in (0, n):
        # Every cluster has the same outcome, so the bootstrap collapses to a single point:
        # use a Wilson interval over the number of clusters instead (conservative), marked *.
        lo, hi = wilson(k * len(nc) // n, len(nc))
        ci = f"[{lo:.0%}-{hi:.0%}]*"
    else:
        lo, hi = bootstrap_ratio(kc, nc)
        ci = f"[{lo:.0%}-{hi:.0%}]"
    return f"{k / n:.0%} ({k}/{n}) {ci}"


def n_clusters(rows: list[dict]) -> int:
    return len({cluster_of(r) for r in rows})


def fmt_rate(k: int, n: int) -> str:
    """Wilson interval (kept for reference; the tables use rate_cell's cluster bootstrap)."""
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


def by_tag(rows: list[dict]) -> dict[str, list[dict]]:
    """Rows per eval group tag (events "groups"); a row with several tags is in each."""
    g: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        for t in r.get("groups") or []:
            g[t].append(r)
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
            prim.append([model, group, str(n_files(rs)), str(n_clusters(rs)), str(len(rs))]
                        + [rate_cell(rs, key, kind) for _, key, kind in PRIMARY])
            sec.append([model, group]
                       + [rate_cell(rs, key, kind) for _, key, kind in SECONDARY]
                       + [latency_cell(rs), iou_cell(rs)])
    tag_rows = []
    for model, rows in models.items():
        for tag, rs in by_tag(rows).items():
            tag_rows.append([model, tag, str(n_files(rs)), str(n_clusters(rs)), str(len(rs))]
                            + [rate_cell(rs, key, kind) for _, key, kind in PRIMARY]
                            + [rate_cell(rs, "greeted_at_start", "run"), latency_cell(rs)])
    tag_part = ("" if not tag_rows else
                "\n## By eval group\n\n"
                "What is new about each file (events `groups`): familiar_type, new_type_phone, new_type_door, other_name, "
                "new_type_reading, extension, implicit_return, ordinary, human. A file with several tags is "
                "counted in each.\n\n"
                + md_table(["model", "eval group", "files", "contexts", "runs"] + [lbl for lbl, _, _ in PRIMARY]
                           + ["greeted at start", "resume latency median"], tag_rows) + "\n")
    return ("## Primary metrics\n\n"
            "Rates: % (k/n) [95% CI, cluster bootstrap over contexts; * = all clusters had the same outcome, so a Wilson interval "
            "over the number of clusters]. Run-level rates are over runs (files x seeds); "
            "distractor and ordinary rates are over individual events x seeds.\n\n"
            + md_table(["model", "group", "files", "contexts", "runs"] + [lbl for lbl, _, _ in PRIMARY], prim)
            + "\n\n## Secondary metrics\n\n"
            + md_table(["model", "group"] + [lbl for lbl, _, _ in SECONDARY]
                       + ["resume latency median", "VAD/text IoU"], sec) + "\n" + tag_part)


def comparison_markdown(models: dict[str, list[dict]]) -> str:
    """One table per group: metrics as rows, models as columns."""
    names = list(models)
    groups = sorted({r["group"] for rows in models.values() for r in rows})
    parts = ["Rates: % (k/n) [95% CI, cluster bootstrap over contexts; * = all clusters had the same outcome, so a Wilson interval "
            "over the number of clusters]. Run-level rates are over runs (files x seeds); "
             "distractor and ordinary rates are over individual events x seeds.\n"]
    tags = sorted({t for rows in models.values() for r in rows for t in r.get("groups") or []})
    sections = [(g, lambda r, g=g: r["group"] == g) for g in groups] + \
               [(f"eval group: {t}", lambda r, t=t: t in (r.get("groups") or [])) for t in tags]
    for group, keep in sections:
        per = {m: [r for r in models[m] if keep(r)] for m in names}
        body = [["files"] + [str(n_files(per[m])) if per[m] else "-" for m in names],
                ["contexts"] + [str(n_clusters(per[m])) if per[m] else "-" for m in names],
                ["runs"] + [str(len(per[m])) if per[m] else "-" for m in names]]
        for lbl, key, kind in PRIMARY:  # always shown, "-" when nothing to count
            body.append([lbl] + [rate_cell(per[m], key, kind) for m in names])
        for lbl, key, kind in SECONDARY:  # shown only when some model has data
            cells = [rate_cell(per[m], key, kind) for m in names]
            if any(c != "-" for c in cells):
                body.append([lbl] + cells)
        body.append(["resume latency median"] + [latency_cell(per[m]) for m in names])
        parts.append(f"## {group}\n\n" + md_table(["metric"] + names, body) + "\n")
    return "\n".join(parts)


# ---------------------------------------------------------------- model-per-row report (comparison.md)
# kind: "run" / "list" = rates as above; "med" = median of a per-row value (or of the values of a
# per-row list), with a cluster-bootstrap CI; "permin" = sum(key) / sum(duration) * 60.

WAIT_COLS = [
    ("wait success", "wait_success", "run"),
    ("wait success 5-15 s", "wait_success_5_15", "run"),
    ("wait success 15-30 s", "wait_success_15_30", "run"),
    ("wait success 30-60 s", "wait_success_30_60", "run"),
    ("interrupt rate", "interrupted", "run"),
    ("time to 1st interruption (interrupted waits)", "time_to_interruption", "med"),
    ("resume on time", "resumed_on_time", "run"),
    ("late resume", "late_resume", "run"),
    ("missed resume", "missed_resume", "run"),
    ("resume latency", "resume_latency", "med"),
    ("resume error, timed waits (signed)", "resume_error", "med"),
    ("spoke over cue", "spoke_over_cue", "run"),
    ("acknowledged", "acknowledged", "run"),
    ("spoke during instr.", "spoke_during_instruction", "run"),
]
DISTRACTOR_COLS = [
    ("distractor response", "distractor_responses", "list"),
    ("question to friend / other", "distractor_friend_or_other", "list"),
    ("other-name question", "distractor_other_name", "list"),
    ("near-miss cue", "distractor_near_miss", "list"),
    ("self-talk", "distractor_self_talk", "list"),
    ("mid-sentence pause", "distractor_pause", "list"),
]
CONV_COLS = [
    ("ordinary response", "ordinary_answered", "list"),
    ("ordinary latency", "ordinary_latency", "med"),
    ("Moshi speech s/min", "moshi_speech_sec", "permin"),
    ("talk-over", "talk_over", "list"),
    ("greeted at start", "greeted_at_start", "run"),
]


def row_values(r: dict, key: str) -> list[float]:
    v = r.get(key)
    if v is None:
        return []
    return [x for x in v if x is not None] if isinstance(v, list) else [v]


def median_cell(rows: list[dict], key: str) -> str:
    """'median s [95% CI] (n)', CI from resampling clusters and taking the pooled median."""
    per: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        per[cluster_of(r)] += row_values(r, key)
    cl = [c for c in per if per[c]]
    vals = [x for c in cl for x in per[c]]
    if not vals:
        return "-"
    med = float(np.median(vals))
    if len(cl) < 2:
        return f"{med:+.2f}s (n={len(vals)})" if key == "resume_error" else f"{med:.2f}s (n={len(vals)})"
    rng = np.random.default_rng(BOOT_SEED)
    arrs = [np.array(per[c]) for c in cl]
    meds = [float(np.median(np.concatenate([arrs[i] for i in rng.integers(0, len(cl), len(cl))])))
            for _ in range(500)]
    lo, hi = np.percentile(meds, [2.5, 97.5])
    f = "+.2f" if key == "resume_error" else ".2f"
    return f"{med:{f}}s [{lo:{f}}, {hi:{f}}] (n={len(vals)})"


def permin_cell(rows: list[dict], key: str) -> str:
    num = defaultdict(float); den = defaultdict(float)
    for r in rows:
        if r.get(key) is not None:
            num[cluster_of(r)] += r[key]; den[cluster_of(r)] += r["duration_sec"] / 60
    if not den:
        return "-"
    cl = list(den)
    est = sum(num.values()) / sum(den.values())
    if len(cl) < 2:
        return f"{est:.1f}"
    lo, hi = bootstrap_ratio(np.array([num[c] for c in cl]), np.array([den[c] for c in cl]))
    return f"{est:.1f} [{lo:.1f}-{hi:.1f}]"


def any_cell(rows: list[dict], key: str, kind: str) -> str:
    if kind in ("run", "list"):
        return rate_cell(rows, key, kind)
    if kind == "med":
        return median_cell(rows, key)
    return permin_cell(rows, key)


def model_tables(models: dict[str, list[dict]], extra_cols: dict[str, dict[str, str]] | None = None,
                 keep=lambda r: True) -> str:
    """Three tables (waits, distractors, conversation), one row per model, for rows passing keep."""
    out = []
    for title, cols in (("Waits", WAIT_COLS), ("Distractors (response rate; lower is better)", DISTRACTOR_COLS),
                        ("Conversation", CONV_COLS)):
        body = []
        for m, rows in models.items():
            rs = [r for r in rows if keep(r)]
            if not rs:
                continue
            cells = [any_cell(rs, k, kind) for _, k, kind in cols]
            body.append([m, str(n_files(rs)), str(n_clusters(rs)), str(len(rs))] + cells)
        if body:
            keepcol = [i for i in range(4, len(body[0])) if any(b[i] != "-" for b in body)]
            hdr = ["model", "files", "contexts", "runs"] + [cols[i - 4][0] for i in keepcol]
            out.append(f"**{title}**\n\n" + md_table(hdr, [b[:4] + [b[i] for i in keepcol] for b in body]))
    return "\n\n".join(out)
