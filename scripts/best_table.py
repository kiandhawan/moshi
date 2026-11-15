"""results/best_checkpoints.md from results/best/<run>.json (written by scripts/eval_run.py)."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_lib import model_roles, readable, run_sort_key  # noqa: E402
REPO = Path(__file__).resolve().parents[1]
recs = sorted((json.loads(p.read_text()) for p in (REPO / "results/best").glob("*.json")), key=lambda r: run_sort_key(r["run"]))
roles = model_roles()
lines = ["# Best checkpoints", "",
         "One row per run (scripts/eval_run.py). Candidates = the 3 checkpoints with the highest whole-wait silence "
         "probability (eval_loss.py, all 403 eval_full_v2 files) + the final one, each run on the 400 eval v2 files (seed 1) "
         "and the human files (seeds 1 2 3). Best = highest wait success among candidates whose ordinary response is at most "
         "10 percentage points below the baseline's. Rates on the 400 eval v2 files: % (k/n) [95% CI, cluster bootstrap].", "",
         "| run | run (raw) | best step | epochs | seen h | checkpoint | wait success | ordinary response | interrupt rate "
         "| distractor response | P(silent whole wait) | val loss | reason |", "|" + "---|" * 13]
for r in recs:
    c = r["candidates"][str(r["step"])]
    lines.append(f"| {readable(r['run'])} | `{r['run']}` | {r['step']} | {r['epochs']} | {r['seen_hours']} | `{r['checkpoint']}` | {c['wait_success']} | "
                 f"{c['ordinary_response']} | {c['interrupt_rate']} | {c['distractor_response']} | {c['wait_p_silent_whole']:.2e} | "
                 f"{c['val_loss']:.3f} | {r['reason']} |")
lines += ["", "## All candidates", "", "| checkpoint | run (raw) | step | wait success | ordinary response | interrupt rate "
          "| P(silent whole wait) | val loss |", "|---|---|---|---|---|---|---|---|"]
for r in recs:
    for s, c in sorted(r["candidates"].items(), key=lambda x: int(x[0])):
        m = f"{r['run']}__s{int(s):03d}"
        lines.append(f"| {readable(m, roles)} | `{r['run']}` | {s} | {c['wait_success']} | {c['ordinary_response']} | {c['interrupt_rate']} | "
                     f"{c['wait_p_silent_whole']:.2e} | {c['val_loss']:.3f} |")
(REPO / "results/best_checkpoints.md").write_text("\n".join(lines) + "\n")
print(f"Wrote results/best_checkpoints.md ({len(recs)} runs)")
