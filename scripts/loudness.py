"""Loudness of speech in stereo conversation WAVs (RMS dBFS over VAD speech, and peak).

  python scripts/loudness.py turns  [--index data/indexes/eval_full.jsonl]
      per Moshi turn of each eval_full answer key (RMS over the VAD speech inside the turn)
  python scripts/loudness.py moshi  DIR
      per left-channel VAD speech segment of run_inference.py outputs (real Moshi output)
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import sphn
import torch
from silero_vad import load_silero_vad

from score import vad_segments


def db(x: float) -> float:
    return 20 * np.log10(max(x, 1e-9))


def level(x: np.ndarray, sr: int, segs) -> tuple[float, float]:
    """(RMS dBFS over the given speech segments, peak dBFS)."""
    parts = [x[int(s * sr):int(e * sr)] for s, e in segs]
    a = np.concatenate(parts) if parts else x[:0]
    if not len(a):
        return float("nan"), float("nan")
    return db(np.sqrt(np.mean(a.astype(np.float64) ** 2))), db(np.abs(a).max())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["turns", "moshi"])
    p.add_argument("dir", nargs="?", type=Path)
    p.add_argument("--index", type=Path, default=Path("data/indexes/eval_full.jsonl"))
    args = p.parse_args()
    torch.set_num_threads(1)
    model = load_silero_vad()
    if args.mode == "turns":
        for line in args.index.read_text().splitlines():
            row = json.loads(line)
            wav = args.index.parent / row["path"]
            ev = json.loads((args.index.parent / row["events"]).read_text())
            x, sr = sphn.read(str(wav))
            left = vad_segments(model, x[0], sr)
            right = vad_segments(model, x[1], sr)
            print(f"{wav.stem}")
            for t in ev["moshi_turns"]:
                segs = [(max(s, t["start"]), min(e, t["end"])) for s, e in left if s < t["end"] and e > t["start"]]
                r, pk = level(x[0], sr, segs)
                print(f"  moshi {t['kind']:9s} {t['start']:6.2f}-{t['end']:6.2f}  RMS {r:6.1f} dBFS  peak {pk:6.1f} dBFS  {t['text'][:45]!r}")
            r, pk = level(x[1], sr, right)
            print(f"  user channel (all speech)          RMS {r:6.1f} dBFS  peak {pk:6.1f} dBFS")
            for spk in sorted({u["speaker"] for u in ev.get("right_turns", [])}):
                segs = [(max(s, u["start"]), min(e, u["end"])) for u in ev["right_turns"] if u["speaker"] == spk
                        for s, e in right if s < u["end"] and e > u["start"]]
                r, pk = level(x[1], sr, segs)
                print(f"  {spk:8s} (right channel)            RMS {r:6.1f} dBFS  peak {pk:6.1f} dBFS")
    else:
        vals = []
        for f in sorted(glob.glob(str(args.dir / "*_seed*.wav"))):
            x, sr = sphn.read(f)
            for s, e in vad_segments(model, x[0], sr):
                if e - s >= 0.5:
                    vals.append(level(x[0], sr, [(s, e)]))
        r = np.array([v[0] for v in vals]); pk = np.array([v[1] for v in vals])
        print(f"{args.dir}: {len(vals)} Moshi speech segments >= 0.5 s")
        print(f"  RMS dBFS  median {np.median(r):.1f}  p10 {np.percentile(r, 10):.1f}  p90 {np.percentile(r, 90):.1f}")
        print(f"  peak dBFS median {np.median(pk):.1f}  p10 {np.percentile(pk, 10):.1f}  p90 {np.percentile(pk, 90):.1f}")


if __name__ == "__main__":
    main()
