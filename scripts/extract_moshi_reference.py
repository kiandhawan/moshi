"""Extract clean Moshi speech from inference outputs to use as a voice-cloning reference.

Scans <outputs...>/*_seed*.wav (left = Moshi, right = user) and keeps stretches where Moshi
speaks and the user channel is quiet:
  - Silero VAD speech segment on the left channel (same rules as score.py), >= MIN_CLIP s
  - no right-channel VAD speech within MARGIN s before/after it
  - right channel RMS below QUIET_DBFS over the whole stretch (+ margin)
Longest clips first (at most MAX_PER_CONV per conversation, for variety) until TARGET_SEC is
reached (never more than MAX_SEC). Clips are joined with GAP_SEC of silence.

Writes <out>.wav (mono 24 kHz) and <out>.json (where each clip came from + Moshi's text).

Usage: python scripts/extract_moshi_reference.py eval_outputs/base_human eval_outputs/moshi_ref_pool \
           --out data/voices/moshi_reference
"""

import argparse
import json
from pathlib import Path

import numpy as np
import sphn
import torch
from silero_vad import load_silero_vad

from score import vad_segments  # same VAD settings as scoring

MIN_CLIP = 1.5
MARGIN = 0.3
QUIET_DBFS = -45.0
TARGET_SEC = 45.0
MAX_SEC = 60.0
MAX_PER_CONV = 3
GAP_SEC = 0.25
PEAK_DBFS = -1.0


def rms_dbfs(x: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("outputs", type=Path, nargs="+", help="One or more run_inference.py output folders.")
    p.add_argument("--out", type=Path, required=True, help="Output path without extension.")
    args = p.parse_args()

    torch.set_num_threads(1)
    model = load_silero_vad()
    candidates = []
    wav_paths = sorted(w for d in args.outputs for w in d.glob("*_seed*.wav"))
    for wav_path in wav_paths:
        wav, sr = sphn.read(str(wav_path))
        left, right = wav[0], wav[1]
        meta_path = wav_path.with_suffix(".json")
        tokens = json.loads(meta_path.read_text())["text_tokens"] if meta_path.exists() else []
        user_segs = vad_segments(model, right, sr)
        for s, e in vad_segments(model, left, sr):
            if e - s < MIN_CLIP:
                continue
            if any(us < e + MARGIN and ue > s - MARGIN for us, ue in user_segs):
                continue
            a, b = max(0, int((s - MARGIN) * sr)), min(len(right), int((e + MARGIN) * sr))
            if rms_dbfs(right[a:b]) > QUIET_DBFS:
                continue
            text = "".join(t["piece"] for t in tokens if s - 0.2 <= t["start"] <= e)
            candidates.append({"source": f"{wav_path.parent.name}/{wav_path.name}", "start": round(s, 2), "end": round(e, 2),
                               "dur": e - s, "text": text.replace("▁", " ").strip(),
                               "audio": left[int(s * sr):int(e * sr)], "sr": sr})

    candidates.sort(key=lambda c: -c["dur"])
    chosen, per_conv, total = [], {}, 0.0
    for c in candidates:
        conv = c["source"]  # one output file (conversation x seed)
        if per_conv.get(conv, 0) >= MAX_PER_CONV or total + c["dur"] > MAX_SEC:
            continue
        chosen.append(c)
        per_conv[conv] = per_conv.get(conv, 0) + 1
        total += c["dur"]
        if total >= TARGET_SEC:
            break
    if not chosen:
        raise SystemExit("No clean Moshi stretches found.")

    sr = chosen[0]["sr"]
    gap = np.zeros(int(GAP_SEC * sr), dtype=np.float32)
    parts = []
    for c in chosen:
        parts += [c["audio"].astype(np.float32), gap]
    audio = np.concatenate(parts[:-1])
    audio *= 10 ** (PEAK_DBFS / 20) / max(np.abs(audio).max(), 1e-9)  # peak-normalize

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sphn.write_wav(str(args.out.with_suffix(".wav")), audio[None], sample_rate=sr)
    manifest = {"total_speech_sec": round(total, 2), "duration_sec": round(len(audio) / sr, 2),
                "sample_rate": sr, "candidates_found": len(candidates),
                "clips": [{k: c[k] for k in ("source", "start", "end", "text")} for c in chosen]}
    args.out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "clips"}))
    print(f"  from {len(wav_paths)} output files")
    for c in manifest["clips"]:
        print(f"  {c['source']:42s} {c['start']:6.2f}-{c['end']:6.2f}  {c['text'][:70]!r}")


if __name__ == "__main__":
    main()
