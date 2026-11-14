"""Check that eval_full WAVs match their transcripts and events (the ideal Moshi side).

For each file in the index:
  - greeting: the left (Moshi) channel has VAD speech inside GREETING_WINDOW (0.3-1.5 s),
    and no speech starts before 0.3 s;
  - tail: the file ends TAIL_SEC (1.5 s, +-TAIL_TOL) after the last VAD speech on either
    channel, and that tail is silent (RMS below -60 dBFS on both channels);
  - every moshi_turn in the events file has a VAD segment on the left channel that starts
    within START_TOL of the turn start and ends within END_TOL of the turn end;
  - no left-channel speech outside the turns (beyond SLACK);
  - every transcript word (<id>.json alignments) lies inside one of the turns;
  - for human files (user_audio from data/human/<name>.wav), the right channel equals the
    original recording sample for sample (up to 16-bit rounding).
VAD settings are the same as score.py. Exit code 1 on any problem.

Usage: python scripts/verify_eval_full.py [--index data/indexes/eval_full.jsonl]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import sphn
import torch
from silero_vad import load_silero_vad

from score import vad_segments

GREETING_WINDOW = (0.3, 1.5)
START_TOL, END_TOL, SLACK = 0.15, 0.25, 0.2
TAIL_SEC, TAIL_TOL, SILENT_DBFS = 1.5, 0.1, -60.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=Path, default=Path("data/indexes/eval_full.jsonl"))
    args = p.parse_args()
    torch.set_num_threads(1)
    model = load_silero_vad()
    problems = []
    for line in args.index.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        wav_p = args.index.parent / row["path"]
        ev = json.loads((args.index.parent / row["events"]).read_text())
        ali = json.loads(wav_p.with_suffix(".json").read_text())["alignments"]
        x, sr = sphn.read(str(wav_p))
        name = wav_p.stem
        segs = vad_segments(model, x[0], sr)
        turns = ev["moshi_turns"]
        print(f"{name}: {x.shape[1] / sr:.2f} s, written {wav_p.stat().st_mtime_ns // 10**9}")
        print(f"  left VAD : {', '.join(f'{s:.2f}-{e:.2f}' for s, e in segs)}")
        print(f"  turns    : {', '.join(f'{t['kind']} {t['start']:.2f}-{t['end']:.2f}' for t in turns)}")
        lo, hi = GREETING_WINDOW
        if segs and segs[0][0] < lo - 0.02:
            problems.append(f"{name}: left-channel speech starts at {segs[0][0]:.2f} s, before {lo} s")
        right_segs = vad_segments(model, x[1], sr)
        last = max([e for _, e in segs + right_segs] or [0.0])
        dur = x.shape[1] / sr
        tail = x[:, int(round(last * sr)) + int(0.05 * sr):]
        tail_db = 20 * np.log10(max(float(np.sqrt(np.mean(tail.astype(np.float64) ** 2))) if tail.size else 0.0, 1e-9))
        print(f"  tail     : last speech ends {last:.2f} s, file ends {dur:.2f} s -> {dur - last:.2f} s tail, "
              f"tail RMS {tail_db:.0f} dBFS")
        if abs(dur - last - TAIL_SEC) > TAIL_TOL:
            problems.append(f"{name}: tail is {dur - last:.2f} s, want {TAIL_SEC} s")
        if tail_db > SILENT_DBFS:
            problems.append(f"{name}: tail not silent ({tail_db:.0f} dBFS)")
        if not any(s < hi and e > lo for s, e in segs):
            problems.append(f"{name}: no left-channel speech in the greeting window {lo}-{hi} s")
        for t in turns:
            match = [(s, e) for s, e in segs if abs(s - t["start"]) <= START_TOL]
            if not match:
                problems.append(f"{name}: {t['kind']} at {t['start']} s has no VAD segment starting within {START_TOL} s")
                continue
            # the turn may be split by VAD at a pause: take the last segment inside the turn
            end = max(e for s, e in segs if s < t["end"] + SLACK and e > t["start"])
            if abs(end - t["end"]) > END_TOL:
                problems.append(f"{name}: {t['kind']} ends at {t['end']} s, VAD speech ends at {end:.2f} s")
        for s, e in segs:
            if not any(s >= t["start"] - SLACK and e <= t["end"] + SLACK for t in turns):
                problems.append(f"{name}: left-channel speech {s:.2f}-{e:.2f} s outside every Moshi turn")
        for w, (s, e), _ in ali:
            if not any(s >= t["start"] - 0.01 and e <= t["end"] + 0.01 for t in turns):
                problems.append(f"{name}: transcript word {w!r} {s}-{e} s outside every Moshi turn")
        greet = [t for t in turns if t["kind"] == "greeting"]
        if not greet or not (lo <= greet[0]["start"] <= hi):
            problems.append(f"{name}: events have no greeting turn starting in {lo}-{hi} s")
        if ev.get("user_start") is not None and greet and greet[0]["end"] > ev["user_start"]:
            problems.append(f"{name}: greeting ends at {greet[0]['end']} s, after user_start {ev['user_start']} s")
        human = Path("data/human") / f"{name.removeprefix('human_')}.wav"
        if name.startswith("human_") and human.exists():
            u, _ = sphn.read(str(human), sample_rate=sr)
            u = u[1] if u.shape[0] == 2 else u[0]
            diff = np.abs(x[1, :len(u)] - u).max()
            rest = np.abs(x[1, len(u):]).max() if x.shape[1] > len(u) else 0.0
            print(f"  right ch : max |diff| vs {human} = {diff:.5f}, after recording {rest:.5f}")
            if diff > 2e-4 or rest > 0:
                problems.append(f"{name}: right channel differs from {human}")
    for pr in problems:
        print("PROBLEM", pr)
    print(f"{len(problems)} problems")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
