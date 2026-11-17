"""Build the listening site for the human test set and its TTS twins (every model output so far).

For every runs/**/infer/{human,human_tts}/ directory (base and every evaluated checkpoint):
- Moshi's channel (left) of each output -> mono MP3 (20 kbps); the 9 clips of a model + set
  (3 recordings x seeds 1 2 3) are concatenated into one file audio/<model>__<set>.mp3, with byte
  offsets in data.json, so the page fetches one file per model and slices it.
- The user inputs (data/human/*.wav, data/human_tts/*.wav) -> audio/input_<id>.mp3, once each.
- data.json: recordings (events, notes, Whisper words of the user), models (readable label, run,
  step, best/final), clips (score.py row: outcome fields, Moshi speech segments, Moshi words with
  times, a 0.1 s loudness envelope of Moshi's channel).
Usage: python scripts/build_listen_site.py OUT_DIR   (then copy the page next to it; audio is not committed)
"""
import ast
import io
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import sphn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_lib import model_roles, model_sort_key, readable  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SETS = {"human": ("data/human", "results/metrics"), "human_tts": ("data/human_tts", "results/human_tts/metrics")}
RECS = ["extend10sec", "talktoroommate", "regularconvo"]
KBPS = "20k"
ENV_STEP = 0.1


def mp3(x: np.ndarray, sr: int) -> bytes:
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    r = subprocess.run(["ffmpeg", "-v", "error", "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
                        "-c:a", "libmp3lame", "-b:a", KBPS, "-f", "mp3", "pipe:1"], input=pcm, capture_output=True, check=True)
    return r.stdout


def envelope(x: np.ndarray, sr: int) -> list[int]:
    n = int(sr * ENV_STEP)
    k = len(x) // n
    rms = np.sqrt((x[: k * n].reshape(k, n) ** 2).mean(1))
    db = 20 * np.log10(rms + 1e-5)  # -100 .. 0
    return [int(v) for v in np.clip((db + 60) / 60 * 99, 0, 99)]  # -60 dB -> 0, 0 dB -> 99


def moshi_words(tokens: str) -> list[list]:
    """SentencePiece pieces with times -> [[word, start, end], ...]."""
    out = []
    for t in (ast.literal_eval(tokens) if isinstance(tokens, str) else tokens) if tokens else []:
        p = t["piece"]
        if p.startswith("▁") or not out:
            out.append([p.lstrip("▁"), t["start"], t["end"]])
        else:
            out[-1][0] += p; out[-1][2] = t["end"]
    return [w for w in out if w[0]]


def rows(path: Path) -> dict:
    if not path.exists():
        return {}
    return {(r["id"], r["seed"]): r for r in map(json.loads, path.read_text().splitlines())}


def main() -> None:
    out = Path(sys.argv[1]); (out / "audio").mkdir(parents=True, exist_ok=True)
    roles = model_roles()
    best = {}
    for p in (REPO / "results/best").glob("*.json"):
        b = json.loads(p.read_text()); best[b["model"]] = "best"
        best.setdefault(b["final_model"], "final")
    words = json.loads((REPO / "data/human_tts/whisper_words.json").read_text())

    recordings = []
    for s, (d, _) in SETS.items():
        for r in RECS:
            rid = r if s == "human" else f"{r}_tts"
            ev = json.loads((REPO / f"{d}/{rid}.events.json").read_text())
            x, sr = sphn.read(str(REPO / f"{d}/{rid}.wav"))
            (out / f"audio/input_{rid}.mp3").write_bytes(mp3(x[0], sr))
            recordings.append({"id": rid, "base": r, "set": s, "events": ev, "words": words[r],
                               "duration": round(x.shape[1] / sr, 2), "env": envelope(x[0], sr)})

    dirs = sorted(p for p in REPO.glob("runs/*/infer/**/human*") if p.is_dir() and p.name in SETS and any(p.glob("*.wav")))
    jobs = []
    for p in dirs:
        run = p.relative_to(REPO / "runs").parts[0]
        sub = p.parent.name
        model = "base" if run == "base" else f"{run}__{sub}"
        jobs.append((model, run, p.name, p))

    def build(job):
        model, run, s, p = job
        sc = rows(REPO / SETS[s][1] / f"{model}.jsonl")
        blob, clips = io.BytesIO(), []
        for r in RECS:
            rid = r if s == "human" else f"{r}_tts"
            for seed in (1, 2, 3):
                w = p / f"{rid}_seed{seed}.wav"
                if not w.exists():
                    continue
                x, sr = sphn.read(str(w))
                meta = json.loads(w.with_suffix(".json").read_text())
                b = mp3(x[0], sr)
                row = sc.get((rid, seed), {})
                keep = ["wait_success", "interrupted", "first_interruption", "missed_resume", "resumed_on_time",
                        "late_resume", "resume_latency", "ordinary_answered", "ordinary_latency",
                        "distractor_responses", "greeted_at_start", "acknowledged", "speech_segments", "timed_wait"]
                clips.append({"rec": rid, "seed": seed, "off": blob.tell(), "len": len(b),
                              "dur": round(x.shape[1] / sr, 2), "text": meta.get("text", ""),
                              "words": moshi_words(meta.get("text_tokens", "")), "env": envelope(x[0], sr),
                              "score": {k: row.get(k) for k in keep}, "scored": bool(row)})
                blob.write(b)
        f = f"{model}__{s}.mp3"
        (out / "audio" / f).write_bytes(blob.getvalue())
        return {"model": model, "set": s, "file": f"audio/{f}", "clips": clips}

    with ThreadPoolExecutor(16) as ex:
        bundles = list(ex.map(build, jobs))

    models = {}
    for model, run, _, _ in jobs:
        step = 0 if model == "base" else int(model.rsplit("__s", 1)[1])
        models[model] = {"id": model, "run": run, "step": step, "label": readable(model, roles),
                         "runLabel": "base" if model == "base" else readable(run),
                         "role": "base" if model == "base" else best.get(model, "")}
    order = sorted(models, key=model_sort_key)
    data = {"recordings": recordings, "models": [models[m] for m in order], "bundles": bundles,
            "kbps": KBPS, "envStep": ENV_STEP}
    (out / "data.json").write_text(json.dumps(data, separators=(",", ":")))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"{len(models)} models, {len(bundles)} bundles, {sum(len(b['clips']) for b in bundles)} clips, "
          f"{size / 1e6:.1f} MB in {out}")


if __name__ == "__main__":
    main()
