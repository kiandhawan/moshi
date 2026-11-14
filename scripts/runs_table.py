"""Write results/runs.csv: one row per training run (configs/<run>.yaml of the grid).

Columns: run, data set, unique scripts (files), unique hours, silence hours (manifest silence_sec:
time with no speech on either channel... as recorded by render.py), wait hours (manifest wait_sec),
steps, batch, samples seen (steps x batch x world size), epochs (samples / files), seen hours of
real audio (epochs x unique hours: every file is shorter than duration_sec, so one sample = one
whole file, padding not counted; the last partial epoch is counted at the mean file length), lr,
status (not started / running step N / done), checkpoint paths.

Usage: python scripts/runs_table.py
"""

import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_run_configs import runs  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    import yaml
    rows = []
    for name in runs({}):
        cfg = yaml.safe_load((REPO / f"configs/{name}.yaml").read_text())
        ds = Path(cfg["data"]["train_data"]).stem
        man = [json.loads(l) for l in open(REPO / f"data/manifests/{ds}.jsonl")]
        n = len(man)
        hours = sum(r["total_sec"] for r in man) / 3600
        steps, batch = cfg["max_steps"], cfg["batch_size"]
        samples = steps * batch
        run_dir = REPO / cfg["run_dir"]
        ckpts = sorted(run_dir.glob("checkpoints/checkpoint_*/consolidated"))
        log = REPO / f"logs/{name}.log"
        status = "not started"
        if log.exists():
            st = re.findall(r"step: (\d+)", log.read_text())
            status = f"running, step {int(st[-1])}" if st else "starting"
            if any(c.parent.name == f"checkpoint_{steps:06d}" for c in ckpts):
                status = "done"
        rows.append({
            "run": name, "data_set": ds, "unique_scripts": n, "unique_hours": round(hours, 3),
            "silence_hours": round(sum(r["silence_sec"] for r in man) / 3600, 3),
            "wait_hours": round(sum(r["wait_sec"] or 0 for r in man) / 3600, 3),
            "steps": steps, "batch": batch, "samples_seen": samples, "epochs": round(samples / n, 2),
            "seen_hours": round(samples / n * hours, 2), "lr": cfg["optim"]["lr"] if cfg["optim"]["lr"] else "TBD",
            "seed": cfg["seed"], "status": status,
            "checkpoints": " ".join(str(c.relative_to(REPO)) for c in ckpts)})
    out = REPO / "results/runs.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    for r in rows:
        print({k: v for k, v in r.items() if k != "checkpoints"})


if __name__ == "__main__":
    main()
