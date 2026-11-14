"""Check that eval_full WAVs match their transcripts and events (the ideal Moshi side).

For each file in the index:
  - greeting: the left (Moshi) channel has VAD speech inside GREETING_WINDOW (0.3-1.5 s),
    and no speech starts before 0.3 s;
  - tail: the file ends TAIL_SEC (1.5 s, +-TAIL_TOL) after the last speech on either channel
    (Moshi: the end of its last turn, itself checked against the audio; user side: the
    scripted end of the last turn, or VAD for a recording), and that tail is silent (no VAD speech): below -60 dBFS on the left; on the right too, or, for
    rendered files with room noise (events noise_dbfs), below -45 dBFS with no speech;
  - length under 90 s;
  - rendered files (events with right_turns): Moshi never overlaps a user/friend/visitor
    turn, and there is no right-channel speech outside those turns (beyond SLACK_RIGHT);
  - every moshi_turn in the events file starts and ends where its audio does: the first /
    last 20 ms frame above ONSET_DBFS on the left channel (digital silence elsewhere) within
    0.15 s after the start (a short breath before the first word) / 0.35 s before the end
    (the quiet decay of the last word), and never outside the turn; and Silero VAD sees speech in the turn (coarse: VAD end within END_TOL; a lone
    short word under VAD_MIN_TURN may go undetected);
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
END_TOL, SLACK, VAD_MIN_TURN = 0.5, 0.2, 0.8
ONSET_DBFS, ONSET_TOL, OFFSET_TOL, EDGE_EPS = -45.0, 0.15, 0.35, 0.02
TAIL_SEC, TAIL_TOL, SILENT_DBFS, NOISY_TAIL_DBFS = 1.5, 0.1, -60.0, -45.0
MAX_SEC, SLACK_RIGHT, OVERLAP_TOL = 90.0, 0.35, 0.05


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
        dur = x.shape[1] / sr
        hop = int(0.02 * sr)
        # Precise edges of each Moshi turn: the left channel is digital silence outside the
        # turns, so its first / last 20 ms frame above ONSET_DBFS is where the audio is.
        edges = []
        for t in turns:
            a0 = max(0.0, t["start"] - 0.3)
            a = x[0, int(a0 * sr):int(min(dur, t["end"] + 0.3) * sr)]
            fr = 20 * np.log10(np.sqrt((a[: len(a) // hop * hop].reshape(-1, hop).astype(np.float64) ** 2).mean(1)) + 1e-12)
            loud = np.nonzero(fr > ONSET_DBFS)[0]
            edges.append((a0 + loud[0] * hop / sr, a0 + (loud[-1] + 1) * hop / sr) if len(loud) else (None, None))
        # Last speech: Moshi's precise end, and the user side's scripted end (rendered files)
        # or its VAD end (recordings, which have no scripted turns).
        rt = ev.get("right_turns")
        right_end = max([u["end"] for u in rt] or [0.0]) if rt is not None else max([e for _, e in right_segs] or [0.0])
        # (Moshi turn ends are the clip ends, checked against the audio below.)
        last = max([t["end"] for t in turns] + [right_end, 0.0])
        tail = x[:, int(round(last * sr)) + int(0.05 * sr):]
        rms_db = lambda a: 20 * np.log10(max(float(np.sqrt(np.mean(a.astype(np.float64) ** 2))) if a.size else 0.0, 1e-9))  # noqa: E731
        tl, tr = rms_db(tail[0]), rms_db(tail[1])
        tail_speech = [(s_, e_) for s_, e_ in right_segs if e_ > last + 0.2]  # VAD overruns turn ends a little
        print(f"  tail     : last speech ends {last:.2f} s, file ends {dur:.2f} s -> {dur - last:.2f} s tail, "
              f"tail RMS left {tl:.0f} / right {tr:.0f} dBFS")
        if abs(dur - last - TAIL_SEC) > TAIL_TOL:
            problems.append(f"{name}: tail is {dur - last:.2f} s, want {TAIL_SEC} s")
        if tl > SILENT_DBFS or tr > (NOISY_TAIL_DBFS if ev.get("noise_dbfs") is not None else SILENT_DBFS) or tail_speech:
            problems.append(f"{name}: tail not silent (left {tl:.0f}, right {tr:.0f} dBFS, speech {tail_speech})")
        if dur >= MAX_SEC:
            problems.append(f"{name}: {dur:.1f} s is not under {MAX_SEC} s")
        if rt is not None:
            for t in turns:
                for u in rt:
                    ov = min(t["end"], u["end"]) - max(t["start"], u["start"])
                    if ov > OVERLAP_TOL:
                        problems.append(f"{name}: Moshi {t['kind']} {t['start']:.2f}-{t['end']:.2f} overlaps "
                                        f"{u['speaker']} {u['start']:.2f}-{u['end']:.2f} by {ov:.2f} s")
            for s_, e_ in right_segs:
                if not any(s_ >= u["start"] - SLACK_RIGHT and e_ <= u["end"] + SLACK_RIGHT for u in rt) \
                        and not any(u["start"] < e_ and u["end"] > s_ for u in rt):
                    problems.append(f"{name}: right-channel speech {s_:.2f}-{e_:.2f} s outside every user-side turn")
        if not any(s < hi and e > lo for s, e in segs):
            problems.append(f"{name}: no left-channel speech in the greeting window {lo}-{hi} s")
        for t, (on, off) in zip(turns, edges):
            # one-sided: never before the turn starts / after it ends; quiet edges inside it
            # (a breath before the first word, the decay of the last) may be below the threshold
            if on is None or not (t["start"] - EDGE_EPS <= on <= t["start"] + ONSET_TOL):
                problems.append(f"{name}: {t['kind']} at {t['start']} s: audio onset at "
                                f"{'none' if on is None else f'{on:.2f} s'}")
            if off is None or not (t["end"] - OFFSET_TOL <= off <= t["end"] + EDGE_EPS):
                problems.append(f"{name}: {t['kind']} ends at {t['end']} s: audio ends at "
                                f"{'none' if off is None else f'{off:.2f} s'}")
            # Coarse VAD check: Silero must see speech in the turn (it can miss a lone short word)
            over = [(s_, e_) for s_, e_ in segs if s_ < t["end"] + SLACK and e_ > t["start"] - SLACK]
            if not over and t["end"] - t["start"] >= VAD_MIN_TURN:
                problems.append(f"{name}: {t['kind']} {t['start']}-{t['end']} s: no VAD speech in the turn")
            elif over and abs(max(e_ for _, e_ in over) - t["end"]) > END_TOL:
                problems.append(f"{name}: {t['kind']} ends at {t['end']} s, VAD speech ends at "
                                f"{max(e_ for _, e_ in over):.2f} s")
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
