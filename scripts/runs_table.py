"""Write results/runs.csv: one row per checkpoint of every training run (grid runs and the
250-step references runs/*_250ref).

Columns: run, data_set, lr, schedule (max_steps), step, epochs (step x batch / files: every file
is shorter than duration_sec, so one sample = one whole file), seen_hours of real audio
(epochs x unique hours; padding not counted), unique_scripts, unique_hours, silence_hours and
wait_hours of the data set (manifest silence_sec / wait_sec), train_loss and eval_subset_loss
at that step (trainer logs: 40-file eval_full_subset_v2), full eval_loss val_loss and
wait_p_silent_whole (eval_loss.py, 403 files), scored (behavioral inference + score.py done:
model RUN__sNNN), role (best / final / every-50 / pruned / kept; from results/best/RUN.json),
weights (present / deleted), checkpoint path.

Usage: python scripts/runs_table.py [--keep RUN=s1,s2 ...]   (--keep is informational; the roles
come from results/best/RUN.json)
"""

import argparse
import csv
import json
import re
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_lib import model_roles, model_sort_key, readable  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MET = REPO / "results/metrics"


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def main() -> None:
    import yaml
    argparse.ArgumentParser().add_argument("--keep", nargs="*")
    loss = {r["label"]: r for r in csv.DictReader((MET / "loss_eval.csv").open())} if (MET / "loss_eval.csv").exists() else {}
    rows = []
    roles = model_roles()
    for run_dir in sorted((REPO / "runs").iterdir()):
        args = run_dir / "args.yaml"
        if not args.exists() or not (run_dir.name.startswith("train_") or run_dir.name.startswith("control_")):
            continue
        run = run_dir.name
        cfg = yaml.safe_load(args.read_text())
        ds = Path(cfg["data"]["train_data"]).stem
        man = jsonl(REPO / f"data/manifests/{ds}.jsonl")
        n, hours = len(man), sum(r["total_sec"] for r in man) / 3600
        batch = cfg["batch_size"] * cfg.get("world_size", 1)
        tr = {r["step"]: r for r in jsonl(run_dir / "metrics.train.jsonl")}
        ev = {r["step"]: r for r in jsonl(run_dir / "metrics.eval.jsonl")}
        best = json.loads((REPO / f"results/best/{run}.json").read_text()) if (REPO / f"results/best/{run}.json").exists() else None
        dirs = {}
        for sub in ("checkpoints", "checkpoints_all"):
            for c in sorted((run_dir / sub).glob("checkpoint_*")):
                dirs.setdefault(int(c.name.split("_")[1]), c / "consolidated")
        for step, c in sorted(dirs.items()):
            label = f"{run}/step{step}"
            has_w = (c / "lora.safetensors").exists()
            final = step == cfg["max_steps"]
            if best is None:
                role = "final" if final else ""
            elif step == best["step"]:
                role = "best" + (" + final" if final else "")
            elif final:
                role = "final"
            elif step % 50 == 0:
                role = "every-50"
            else:
                role = "pruned" if not has_w else "to prune"
            rows.append({
                "label": readable(f"{run}__s{step:03d}", roles), "run": run, "data_set": ds, "lr": cfg["optim"]["lr"], "schedule_steps": cfg["max_steps"], "step": step,
                "epochs": round(step * batch / n, 2), "seen_hours": round(step * batch / n * hours, 2),
                "unique_scripts": n, "unique_hours": round(hours, 3),
                "silence_hours": round(sum(r["silence_sec"] for r in man) / 3600, 3),
                "wait_hours": round(sum(r["wait_sec"] or 0 for r in man) / 3600, 3),
                "train_loss": round(tr[step]["loss"], 4) if step in tr else "",
                "eval_subset_loss": round(ev[step]["eval_loss"], 4) if step in ev else "",
                "val_loss": loss.get(label, {}).get("val_loss", ""),
                "wait_p_silent_whole": loss.get(label, {}).get("wait_p_silent_whole", ""),
                "scored": (MET / f"{run}__s{step:03d}.jsonl").exists(),
                "role": role + (" (250-step reference, stopped early)" if run.endswith("_250ref") else ""),
                "weights": "present" if has_w else "deleted", "checkpoint": str(c.relative_to(REPO))})
    rows.sort(key=lambda r: model_sort_key(f"{r['run']}__s{int(r['step']):03d}"))
    out = REPO / "results/runs.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {out} ({len(rows)} checkpoints)")


if __name__ == "__main__":
    main()
