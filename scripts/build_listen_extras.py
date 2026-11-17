"""Extra sections for the listening site: data samples and demo videos (run after build_listen_site.py).

Samples: listen_sample/*.wav (data v1: train_10_*, eval_12_*) and listen_sample/v2/*.wav (data v2 training).
Each -> stereo MP3 (left = Moshi, right = user side, 48 kbps) at audio/sample_<id>.mp3, plus a timed
transcript (Moshi turns from the events file; user/friend/other turn times from the events file, text from
the script, matched in order), the wait/resume windows and both channels' loudness envelopes.
Demos: every video in demos/ (optional demos/demos.json: [{"file", "title", "note"}] sets order and titles)
-> H.264/AAC MP4, 720p max, at video/<stem>.mp4 with a poster video/<stem>.jpg. Kept under 15 MB each.
Writes extras.json next to data.json. Usage: python scripts/build_listen_extras.py OUT_DIR
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import sphn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_listen_site import envelope  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
MAX_BYTES = 14_500_000


def v1_first(paths, stem):
    """v1 files (train_10_*, eval_12_*) also exist under data/v2/ (eval v2 kept most v1 eval files); use v1's."""
    return sorted(paths, key=lambda p: ("/v2/" in str(p)) != stem.startswith("trainv2"))


def events(stem: str) -> dict:
    hits = [p for p in REPO.glob(f"data/**/{stem}.events.json") if "/clean/" not in str(p)]
    return next(e for e in (json.loads(p.read_text()) for p in v1_first(hits, stem)) if "moshi_turns" in e)


def transcript(stem: str, ev: dict) -> list:
    script = v1_first(REPO.glob(f"data/**/scripts/**/{stem}.json"), stem)[0]
    others = [t for t in json.loads(script.read_text())["turns"] if t["speaker"] != "moshi"]
    lines = [{"who": "moshi", "s": t["start"], "e": t["end"], "tag": t["kind"], "text": t["text"]} for t in ev["moshi_turns"]]
    for rt, st in zip(ev["right_turns"], others):
        lines.append({"who": rt["speaker"], "s": rt["start"], "e": rt["end"], "tag": rt.get("kind") or rt.get("event") or "",
                      "text": st["text"]})
    return sorted(lines, key=lambda l: l["s"])


def samples(out: Path) -> list:
    res = []
    for w in sorted((REPO / "listen_sample").glob("*.wav")) + sorted((REPO / "listen_sample/v2").glob("*.wav")):
        stem = re.sub(r"^\d+_[a-z_]+?_(?=(train|eval))", "", w.stem)
        ev = events(stem)
        x, sr = sphn.read(str(w))
        dst = out / f"audio/sample_{stem}.mp3"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(w), "-c:a", "libmp3lame", "-b:a", "48k", str(dst)], check=True)
        v2 = "/v2/" in str(w)
        res.append({"id": stem, "src": f"audio/sample_{stem}.mp3", "dur": round(x.shape[1] / sr, 2),
                    "set": "v2 training" if v2 else ("v1 eval" if stem.startswith("eval") else "v1 training"),
                    "type": ev["type"], "conditions": ev.get("conditions", []), "groups": ev.get("groups", []),
                    "augmentation": ev.get("augmentation"), "events": {k: ev.get(k) for k in
                    ("instruction_start", "instruction_end", "wait_end", "resume_window", "distractors", "wait_sec")},
                    "env": [envelope(x[0], sr), envelope(x[1], sr)], "lines": transcript(stem, ev)})
    return res


def demos(out: Path) -> list:
    d = REPO / "demos"
    if not d.exists():
        return []
    meta = json.loads((d / "demos.json").read_text()) if (d / "demos.json").exists() else []
    files = [d / m["file"] for m in meta] + sorted(p for p in d.iterdir() if p.suffix.lower() in VIDEO_EXT
                                                   and p.name not in {m["file"] for m in meta})
    info = {m["file"]: m for m in meta}
    res = []
    (out / "video").mkdir(exist_ok=True)
    for f in files:
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(f)],
                                   capture_output=True, text=True, check=True).stdout.strip())
        dst, poster = out / f"video/{f.stem}.mp4", out / f"video/{f.stem}.jpg"
        for vk, ak in ((1400, 96), (900, 64), (550, 48), (300, 32)):  # step the bitrate down until it fits
            if (vk + ak) * 1000 / 8 * dur * 1.05 > MAX_BYTES:
                continue
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(f), "-vf", "scale='min(1280,iw)':-2", "-c:v", "libx264",
                            "-preset", "medium", "-b:v", f"{vk}k", "-maxrate", f"{int(vk * 1.3)}k", "-bufsize", f"{vk * 2}k",
                            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", f"{ak}k", "-movflags", "+faststart", str(dst)], check=True)
            if dst.stat().st_size <= MAX_BYTES:
                break
        else:
            raise SystemExit(f"{f.name}: too long to fit in 15 MB ({dur:.0f} s)")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(min(1.0, dur / 2)), "-i", str(dst), "-frames:v", "1",
                        "-q:v", "4", str(poster)], check=True)
        m = info.get(f.name, {})
        res.append({"src": f"video/{f.stem}.mp4", "poster": f"video/{f.stem}.jpg", "dur": round(dur, 1),
                    "title": m.get("title") or f.stem.replace("_", " ").replace("-", " "), "note": m.get("note", "")})
        print(f"  {f.name}: {dur:.0f} s -> {dst.stat().st_size / 1e6:.1f} MB")
    return res


def main() -> None:
    out = Path(sys.argv[1])
    data = {"samples": samples(out), "demos": demos(out)}
    (out / "extras.json").write_text(json.dumps(data, separators=(",", ":")))
    print(f"{len(data['samples'])} samples, {len(data['demos'])} demos")


if __name__ == "__main__":
    main()
