"""Check v2 augmented training files against their clean copies (scripts/render.py --augment).

The content checks (turn edges, tail, overlaps, transcript) run on the clean copies with
verify_eval_full.py. This script checks, per file of --index (augmented) and --clean (clean):
  - same length; the left (Moshi) channel is identical (Moshi stays clean);
  - transcript (<id>.json) identical; events identical except the augmentation record;
  - the right channel differs from the clean one (augmentation applied) but is not clipped
    (peak <= 0.99) and still carries the speech: correlation with the clean right channel
    over the user's turns >= MIN_CORR (0.3; phone band / reverb / noise lower it), and the
    speech-to-noise ratio (RMS in turns vs. outside turns) >= MIN_SNR_DB;
  - the recorded settings match render_lib.augment_settings(id) (fixed seed).
Prints the augmentation mix (noise types and levels, mic coloring, user reverb, gains).

Usage: python scripts/verify_augment.py --index data/indexes/train_all_v2.jsonl --clean data/indexes/train_all_v2_clean.jsonl
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import sphn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_lib import augment_settings  # noqa: E402

MIN_CORR, MIN_SNR_DB = 0.3, 6.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--clean", type=Path, required=True)
    args = p.parse_args()
    rows = [json.loads(l) for l in args.index.read_text().splitlines() if l.strip()]
    clean = {Path(r["path"]).stem: r for r in (json.loads(l) for l in args.clean.read_text().splitlines() if l.strip())}
    problems, mix, snrs, corrs = [], Counter(), [], []
    for r in rows:
        name = Path(r["path"]).stem
        c = clean.get(name)
        if c is None:
            problems.append(f"{name}: no clean copy")
            continue
        a_p, c_p = args.index.parent / r["path"], args.clean.parent / c["path"]
        x, sr = sphn.read(str(a_p))
        y, _ = sphn.read(str(c_p))
        ev = json.loads((args.index.parent / r["events"]).read_text())
        evc = json.loads((args.clean.parent / c["events"]).read_text())
        aug = ev.get("augmentation")
        if x.shape != y.shape:
            problems.append(f"{name}: shape {x.shape} vs clean {y.shape}")
            continue
        if np.abs(x[0] - y[0]).max() > 0:
            problems.append(f"{name}: left (Moshi) channel differs from the clean copy")
        if a_p.with_suffix(".json").read_text() != c_p.with_suffix(".json").read_text():
            problems.append(f"{name}: transcript differs")
        strip = lambda e: {k: v for k, v in e.items() if k not in ("augmentation", "noise_dbfs")}  # noqa: E731
        if strip(ev) != strip(evc):
            problems.append(f"{name}: events differ beyond the augmentation record")
        if aug != augment_settings(name):
            problems.append(f"{name}: recorded augmentation is not augment_settings({name})")
        if np.abs(x[1] - y[1]).max() < 1e-3:
            problems.append(f"{name}: right channel not augmented")
        if np.abs(x[1]).max() > 0.995:
            problems.append(f"{name}: right channel clipped (peak {np.abs(x[1]).max():.3f})")
        m = np.zeros(x.shape[1], dtype=bool)
        for t in ev["right_turns"]:
            m[int(t["start"] * sr):int(t["end"] * sr)] = True
        if m.any() and (~m).sum() > sr:
            sig, noi = x[1][m], x[1][~m]
            snr = 10 * np.log10(np.mean(sig.astype(np.float64) ** 2) / max(np.mean(noi.astype(np.float64) ** 2), 1e-12))
            corr = float(np.corrcoef(sig, y[1][m])[0, 1])
            snrs.append(snr)
            corrs.append(corr)
            if snr < MIN_SNR_DB:
                problems.append(f"{name}: speech-to-noise {snr:.1f} dB < {MIN_SNR_DB}")
            if corr < MIN_CORR:
                problems.append(f"{name}: correlation with the clean user side {corr:.2f} < {MIN_CORR}")
        mix[f"noise {aug['noise']['type']}"] += 1
        mix[f"mic {aug['mic']['type'] if aug['mic'] else 'none'}"] += 1
        mix[f"user reverb {'yes' if aug['user_reverb'] else 'no'}"] += 1
    n = len(rows)
    print(f"{n} augmented files checked against their clean copies")
    for k, v in sorted(mix.items()):
        print(f"  {k}: {v} ({v / n:.0%})")
    if snrs:
        print(f"  speech-to-noise (RMS in user turns vs outside): median {np.median(snrs):.1f} dB, "
              f"5th pct {np.percentile(snrs, 5):.1f} dB; correlation with clean: median {np.median(corrs):.2f}, "
              f"min {min(corrs):.2f}")
    for pr in problems:
        print("PROBLEM", pr)
    print(f"{len(problems)} problems")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
