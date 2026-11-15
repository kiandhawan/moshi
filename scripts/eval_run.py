"""Evaluate one finished training run, pick its best checkpoint, record everything, then prune.

Steps (each skips work already done, so the script can be rerun after an interruption):
 1. copy the training / eval-subset loss logs (runs/RUN/metrics.{train,eval}.jsonl) to
    results/train_logs/RUN/
 2. eval_loss.py on all 403 eval_full_v2 files for every checkpoint (results/metrics/loss_eval*.csv)
 3. candidates = the 3 checkpoints with the highest whole-wait silence probability
    (wait_p_silent_whole, all files) + the final checkpoint; for each: streaming inference
    (400 eval v2 files, seed 1, batches of 32; human files, seeds 1 2 3) into
    runs/RUN/infer/sNNN/ and score.py as model RUN__sNNN
 4. best = the candidate with the highest wait success (eval v2, 400 files) among those whose
    ordinary response is at least the baseline's minus 10 percentage points (ties: higher
    wait_p_silent_whole). If none qualifies, the candidate with the highest ordinary response,
    flagged. -> results/best/RUN.json, results/best_checkpoints.md
 5. voice similarity (best, final), listening examples for the best (listen/RUN/, not committed),
    runs.csv, comparison.md and plots
 6. prune: delete lora.safetensors of every checkpoint that is not the best, not a multiple of
    50 and not the final - only after checking its eval_loss rows, its runs.csv row and the
    copied training logs are in results/ (and, for candidates, its scores). config.json stays.
Reply relevance needs a judge (relevance.py prepare / a subagent / collect) and is done outside.

Usage: python scripts/eval_run.py RUN GPU [--no-prune]
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_lib import load_rows, model_dir, rate_cell, cluster_counts  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PY = str(REPO / ".venv/bin/python")
MET = REPO / "results/metrics"


def sh(cmd: list[str], log: Path) -> None:
    print(f"$ {' '.join(cmd)}  > {log.relative_to(REPO)}", flush=True)
    with log.open("w") as f:
        r = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise SystemExit(f"FAILED ({r.returncode}): {' '.join(cmd)}; see {log}")


def ckpts(run: str) -> dict[int, Path]:
    return {int(p.parent.name.split("_")[1]): p
            for p in sorted((REPO / f"runs/{run}/checkpoints").glob("checkpoint_*/consolidated"))}


def loss_rows() -> dict[str, dict]:
    p = MET / "loss_eval.csv"
    return {r["label"]: r for r in csv.DictReader(p.open())} if p.exists() else {}


def overall(model: str) -> dict:
    rows = [r for r in load_rows([model])[model] if "human" not in r["groups"]]
    out = {}
    for key, kind, name in (("wait_success", "run", "wait_success"), ("ordinary_answered", "list", "ordinary_response"),
                            ("interrupted", "run", "interrupt_rate"), ("resumed_on_time", "run", "resume_on_time"),
                            ("distractor_responses", "list", "distractor_response")):
        k, n = cluster_counts(rows, key, kind)
        out[name] = {"rate": float(k.sum() / n.sum()) if n.sum() else None, "k": int(k.sum()), "n": int(n.sum()),
                     "cell": rate_cell(rows, key, kind)}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run"); p.add_argument("gpu")
    p.add_argument("--no-prune", action="store_true")
    a = p.parse_args()
    run, gpu = a.run, a.gpu
    logs = REPO / "logs"
    cfg = (REPO / f"configs/{run}.yaml").read_text()
    max_steps = int(next(l.split()[1] for l in cfg.splitlines() if l.startswith("max_steps")))
    ck = ckpts(run)
    assert max_steps in ck, f"{run}: no final checkpoint (step {max_steps}) yet"

    # 1. training logs
    tl = REPO / f"results/train_logs/{run}"
    tl.mkdir(parents=True, exist_ok=True)
    for f in ("metrics.train.jsonl", "metrics.eval.jsonl", "args.yaml"):
        if (REPO / f"runs/{run}/{f}").exists():
            shutil.copy2(REPO / f"runs/{run}/{f}", tl / f)

    # 2. eval_loss on every checkpoint
    for step, c in ck.items():
        if loss_rows().get(f"{run}/step{step}", {}).get("wait_p_silent_whole") in (None, ""):  # missing or pre-v2 row
            sh([PY, "scripts/eval_loss.py", "--checkpoint", str(c), "--gpu", gpu], logs / f"eval_loss_{run}_s{step:03d}.log")
    lr = loss_rows()
    whole = {s: float(lr[f"{run}/step{s}"]["wait_p_silent_whole"]) for s in ck}
    top3 = sorted(ck, key=lambda s: -whole[s])[:3]
    cands = sorted(set(top3) | {max_steps})
    print(f"top 3 by whole-wait silence probability: {top3}; candidates {cands}", flush=True)

    # 3. inference + scoring for the candidates (two at a time on this GPU)
    def infer_cmds(step):
        out = model_dir(f"{run}__s{step:03d}")
        lora = ["--lora", str(ck[step])]
        return [([PY, "scripts/run_inference.py", "data/indexes/eval_v2.jsonl", *lora, "--seeds", "1", "--batch", "32",
                  "--gpu", gpu, "--out-dir", str(out / "eval")], logs / f"infer_{run}_s{step:03d}_eval.log"),
                ([PY, "scripts/run_inference.py", "data/indexes/human.jsonl", *lora, "--seeds", "1", "2", "3",
                  "--gpu", gpu, "--out-dir", str(out / "human")], logs / f"infer_{run}_s{step:03d}_human.log")]
    todo = [s for s in cands if not (MET / f"{run}__s{s:03d}.jsonl").exists()]
    for i in range(0, len(todo), 2):
        procs = []
        for s in todo[i:i + 2]:
            for cmd, log in infer_cmds(s):
                print(f"$ {' '.join(cmd)}", flush=True)
                procs.append((subprocess.Popen(cmd, cwd=REPO, stdout=log.open("w"), stderr=subprocess.STDOUT), cmd))
        for pr, cmd in procs:
            if pr.wait():
                raise SystemExit(f"FAILED: {' '.join(cmd)}")
        for s in todo[i:i + 2]:
            d = model_dir(f"{run}__s{s:03d}")
            sh([PY, "scripts/score.py", "--outputs", str(d / "eval"), str(d / "human"), "--events",
                "data/indexes/eval_full_v2.jsonl", "data/indexes/human.jsonl", "--name", f"{run}__s{s:03d}", "--quiet"],
               logs / f"score_{run}_s{s:03d}.log")

    # 4. best checkpoint
    base = overall("base")
    floor = base["ordinary_response"]["rate"] - 0.10
    res = {s: overall(f"{run}__s{s:03d}") for s in cands}
    ok = [s for s in cands if res[s]["ordinary_response"]["rate"] >= floor]
    if ok:
        best = max(ok, key=lambda s: (res[s]["wait_success"]["rate"], whole[s]))
        reason = (f"highest wait success ({res[best]['wait_success']['cell']}) among the candidates whose ordinary "
                  f"response is within 10 pp of the baseline's {base['ordinary_response']['rate']:.0%} "
                  f"(>= {floor:.0%}: {', '.join(f's{s}' for s in ok)})")
        tied = [s for s in ok if s != best and res[s]["wait_success"]["rate"] == res[best]["wait_success"]["rate"]]
        if tied:
            reason += f"; tied on wait success with {tied}, higher whole-wait silence probability"
    else:
        best = max(cands, key=lambda s: res[s]["ordinary_response"]["rate"])
        reason = (f"NO candidate kept ordinary response within 10 pp of the baseline's "
                  f"{base['ordinary_response']['rate']:.0%}; this is the one with the highest ordinary response")
    import yaml
    ycfg = yaml.safe_load(cfg)
    idx = [json.loads(l) for l in open(ycfg["data"]["train_data"]) if l.strip()]
    n_files, hours = len(idx), sum(r["duration"] for r in idx) / 3600
    epochs = best * ycfg["batch_size"] / n_files
    rec = {"run": run, "step": best, "checkpoint": str(ck[best].relative_to(REPO)), "model": f"{run}__s{best:03d}",
           "final_model": f"{run}__s{max_steps:03d}", "max_steps": max_steps, "epochs": round(epochs, 2),
           "seen_hours": round(epochs * hours, 2), "reason": reason, "baseline_ordinary": base["ordinary_response"],
           "candidates": {str(s): {**{k: v["cell"] for k, v in res[s].items()},
                                   "wait_p_silent_whole": whole[s], "val_loss": float(lr[f"{run}/step{s}"]["val_loss"])}
                          for s in cands},
           "top3_by_silence": top3}
    (REPO / "results/best").mkdir(exist_ok=True)
    (REPO / f"results/best/{run}.json").write_text(json.dumps(rec, indent=1))
    print(f"BEST {run}: step {best} - {reason}", flush=True)
    sh([PY, "scripts/best_table.py"], logs / "best_table.log")

    # 5. voice, listening examples, records, report
    sh([PY, "scripts/voice_similarity.py", rec["model"], rec["final_model"], "--gpu", gpu], logs / f"voice_{run}.log")
    sh([PY, "scripts/pick_listen.py", rec["model"], run], logs / f"listen_{run}.log")
    keep = {best, max_steps} | {s for s in ck if s % 50 == 0}
    sh([PY, "scripts/runs_table.py", "--keep", f"{run}=" + ",".join(map(str, sorted(keep)))], logs / "runs_table.log")
    sh([PY, "scripts/compare.py"], logs / "compare.log")
    sh([PY, "scripts/plot_results.py"], logs / "plots.log")

    # 6. prune
    if a.no_prune:
        return
    runs_csv = {(r["run"], int(r["step"])) for r in csv.DictReader(open(REPO / "results/runs.csv"))}
    lr = loss_rows()
    groups = {r["label"] for r in csv.DictReader(open(MET / "loss_eval_groups.csv"))}
    for s, c in ck.items():
        if s in keep:
            continue
        label = f"{run}/step{s}"
        checks = {"eval_loss row": label in lr, "eval_loss group rows": label in groups,
                  "per-file loss": (MET / "loss_eval" / f"{label.replace('/', '__')}.jsonl").exists(),
                  "runs.csv row": (run, s) in runs_csv, "training logs copied": (tl / "metrics.train.jsonl").exists(),
                  "scores (if a candidate)": s not in cands or (MET / f"{run}__s{s:03d}.jsonl").exists()}
        if not all(checks.values()):
            print(f"NOT pruning step {s}: missing {[k for k, v in checks.items() if not v]}", flush=True)
            continue
        w = c / "lora.safetensors"
        if w.exists():
            w.unlink()
            print(f"pruned {w.relative_to(REPO)}", flush=True)
    sh([PY, "scripts/runs_table.py", "--keep", f"{run}=" + ",".join(map(str, sorted(keep)))], logs / "runs_table.log")
    free = shutil.disk_usage("/").free / 1e9
    print(f"DONE {run}; disk free {free:.0f} GB" + ("  WARNING: below 50 GB" if free < 50 else ""), flush=True)


if __name__ == "__main__":
    main()
