"""Real human recordings vs their TTS twins (scripts/make_human_tts.py), per model.

Real: rows of results/metrics/<model>.jsonl whose groups contain "human" (data/human, seeds 1 2 3).
Twins: results/human_tts/metrics/<model>.jsonl (data/human_tts, seeds 1 2 3, scored with the copied
events). 3 files x 3 seeds each; rates are % (k/n), with each file its own context, so CIs are wide.
Writes results/human_tts/comparison.md and copies the twins and every model's outputs into
listen_human_tts/ (not committed). Usage: python scripts/human_tts_compare.py MODEL...
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_lib import md_table, model_dir, model_roles, ordered, rate_cell, readable  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
COLS = [("wait success", "wait_success", "run"), ("missed resume", "missed_resume", "run"),
        ("interrupt rate", "interrupted", "run"), ("distractor response", "distractor_responses", "list"),
        ("ordinary response", "ordinary_answered", "list")]


def rows(p: Path, human_only: bool) -> list[dict]:
    rs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return [r for r in rs if "human" in r["groups"]] if human_only else rs


def main() -> None:
    models = ordered(sys.argv[1:])
    roles = model_roles()
    body = []
    for m in models:
        for label, rs in (("real", rows(REPO / f"results/metrics/{m}.jsonl", True)),
                          ("TTS twin", rows(REPO / f"results/human_tts/metrics/{m}.jsonl", False))):
            body.append([readable(m, roles), label, str(len(rs))] + [rate_cell(rs, k, kind) for _, k, kind in COLS])
    md = ["# Human recordings vs TTS twins", "",
          "`scripts/make_human_tts.py` (twins: same utterances re-voiced with held-out Fish voices at the original "
          "onsets; adjustments in `data/human_tts/report.json`) and `scripts/human_tts_compare.py`. Each model: the 3 "
          "files x seeds 1 2 3, unbatched. Rates % (k/n) [95% CI; each file is its own context, so with 3 files the "
          "intervals are very wide; * = all files had the same outcome].", "",
          md_table(["model", "audio", "runs"] + [c for c, _, _ in COLS], body), ""]
    out = REPO / "results/human_tts/comparison.md"
    out.write_text("\n".join(md))
    print("\n".join(md))

    dst = REPO / "listen_human_tts"
    (dst / "twins").mkdir(parents=True, exist_ok=True)
    for f in sorted((REPO / "data/human_tts").glob("*_tts.wav")):
        shutil.copy2(f, dst / "twins" / f.name)
    for m in models:
        d = dst / readable(m, roles)
        d.mkdir(parents=True, exist_ok=True)
        for f in sorted((model_dir(m) / "human_tts").glob("*_seed*.*")):
            shutil.copy2(f, d / f.name)
    print(f"Copied the twins and {len(models)} models' outputs to {dst}")


if __name__ == "__main__":
    main()
