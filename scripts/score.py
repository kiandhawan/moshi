"""Score Moshi inference outputs against events files.

Inputs
  --outputs DIR   folder written by scripts/run_inference.py: <id>_seed<N>.wav (left = Moshi,
                  right = user) and <id>_seed<N>.json (text tokens with timestamps)
  --events INDEX  jsonl whose lines have an "events" field: path to an events JSON (relative
                  paths are resolved against the index's folder)
  --name NAME     model name used in the results files and summary table

Events format (one JSON per test file; times in seconds on the user-audio timeline):
  {"id": "extend10sec",            # matches the output file stem <id>_seed<N>
   "group": "timed_wait",          # timed_wait | friend_wait | describe_options | ordinary
   "user_start": 2.0,              # when the user first speaks (default: first VAD speech on
                                   # the user channel); Moshi speech before it is the greeting
   "instruction_start": 2.0,       # when the user starts the wait instruction; null if none
   "instruction_end": 4.9,         # null for ordinary conversations
   "wait_end": 22.0,               # when Moshi may speak again; null if no wait
   "resume_window": [22.0, 26.0],  # Moshi's speech should start inside this window
   "distractors": [8.1, 15.0],     # times of things Moshi should NOT respond to
   "ordinary_requests": [30.2],    # times (end of request) Moshi SHOULD answer
   "ordinary_window": 5.0,         # optional: answer must start within this many s (default 5)
   "notes": "..."}                 # optional free text
Only id and group are required. Missing wait fields mean "no wait" (e.g. ordinary
conversations); missing distractors / ordinary_requests mean none.

Speech detection: Silero VAD on the left channel only (resampled to 16 kHz), segments merged
across gaps < 0.3 s, then segments < 0.2 s dropped. Speech is also derived from the non-padding
text tokens (each token = its 80 ms slot, tokens < 0.5 s apart merged) and the disagreement
between the two is reported. All metrics use the VAD segments.

Metrics per file and seed (VAD-based; times straight from the WAV, no extra offset):
  greeted_at_start          Moshi speech starts before user_start: the opening greeting
                            (CLAUDE.md: Moshi always opens by greeting and asking how the user
                            is). Those greeting segments are excluded from all metrics below,
                            even if they run past user_start.
  spoke_during_instruction  any speech overlapping [instruction_start, instruction_end]
  acknowledged              speech starting in [instruction_end, instruction_end + ack_grace)
  interrupted               > 0.5 s of speech in [instruction_end + ack_grace, wait_end]
                            (an acknowledgement that runs past the grace period counts)
  first_interruption        start of the first speech inside that window
  resumed_on_time           a speech segment starts inside resume_window (strict: speech that
                            began before wait_end and runs into the window doesn't count)
  spoke_over_cue            Moshi was already speaking at wait_end (a segment starts before
                            wait_end and is still going at it), e.g. talking over the return cue
  resume_latency            wait_end -> start of the first segment starting at/after wait_end
  distractor_responses      per distractor: speech starts within 3 s after it
  ordinary_answered         per ordinary request: speech starts within 5 s after it
  wait_success (PRIMARY)    not interrupted AND resumed_on_time AND no distractor response;
                            one per wait (each test file has one wait), null if no wait

Outputs
  results/metrics/<NAME>.jsonl   one row per file and seed (overwritten on each run)
  results/summary.md             tables per model and group with 95% Wilson CIs, rebuilt from
                                 all results/metrics/*.jsonl (see report_lib.py)
To compare chosen models side by side, use scripts/compare.py.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import sphn
import torch
import torchaudio
from silero_vad import get_speech_timestamps, load_silero_vad

from report_lib import load_rows, summary_markdown

VAD_SR = 16000
MERGE_GAP = 0.3          # merge VAD segments separated by less than this
MIN_SEGMENT = 0.2        # then drop VAD segments shorter than this
TEXT_MERGE_GAP = 0.5     # merge text-token slots separated by less than this
INTERRUPT_MIN_SPEECH = 0.5
DISTRACTOR_WINDOW = 3.0
ORDINARY_WINDOW = 5.0
GRID = 0.01              # resolution for comparing VAD vs text speech

REPO = Path(__file__).resolve().parent.parent
METRICS_DIR = REPO / "results" / "metrics"
SUMMARY_PATH = REPO / "results" / "summary.md"
REQUIRED_FIELDS = ("id", "group")
OPTIONAL_DEFAULTS = {"user_start": None, "instruction_start": None, "instruction_end": None, "wait_end": None,
                     "resume_window": None, "distractors": [], "ordinary_requests": [],
                     "ordinary_window": ORDINARY_WINDOW}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outputs", type=Path, required=True)
    p.add_argument("--events", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--ack-grace", type=float, default=2.0,
                   help="Seconds after instruction_end in which a brief acknowledgement is allowed.")
    p.add_argument("--quiet", action="store_true", help="Don't print per-run segment details.")
    return p.parse_args()


# ---------------------------------------------------------------- segments

Seg = tuple[float, float]


def merge(segs: list[Seg], gap: float) -> list[Seg]:
    out: list[list[float]] = []
    for s, e in sorted(segs):
        if out and s - out[-1][1] < gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def vad_segments(model, left: np.ndarray, sr: int) -> list[Seg]:
    wav = torch.from_numpy(np.ascontiguousarray(left, dtype=np.float32))
    if sr != VAD_SR:
        wav = torchaudio.functional.resample(wav, sr, VAD_SR)
    # Disable Silero's own merging/padding/min-length so only our rules apply.
    ts = get_speech_timestamps(wav, model, sampling_rate=VAD_SR, min_speech_duration_ms=0,
                               min_silence_duration_ms=0, speech_pad_ms=0)
    raw = [(t["start"] / VAD_SR, t["end"] / VAD_SR) for t in ts]
    return [(s, e) for s, e in merge(raw, MERGE_GAP) if e - s >= MIN_SEGMENT]


def text_segments(meta: dict) -> list[Seg]:
    return merge([(t["start"], t["end"]) for t in meta["text_tokens"]], TEXT_MERGE_GAP)


def to_mask(segs: list[Seg], dur: float) -> np.ndarray:
    m = np.zeros(int(np.ceil(dur / GRID)), dtype=bool)
    for s, e in segs:
        m[int(s / GRID):int(np.ceil(e / GRID))] = True
    return m


def disagreement(vad: list[Seg], text: list[Seg], dur: float) -> dict:
    a, b = to_mask(vad, dur), to_mask(text, dur)
    sec = lambda m: round(float(m.sum()) * GRID, 2)  # noqa: E731
    both, union = sec(a & b), sec(a | b)
    return {"vad_speech_sec": sec(a), "text_speech_sec": sec(b),
            "vad_only_sec": sec(a & ~b), "text_only_sec": sec(b & ~a),
            "iou": round(both / union, 3) if union else None}


def speech_in(segs: list[Seg], lo: float, hi: float) -> float:
    return sum(max(0.0, min(e, hi) - max(s, lo)) for s, e in segs)


def first_start_in(segs: list[Seg], lo: float, hi: float) -> float | None:
    return next((s for s, _ in segs if lo <= s <= hi), None)


# ---------------------------------------------------------------- metrics

def score_run(ev: dict, segs: list[Seg], text_segs: list[Seg], ack_grace: float,
              user_start: float | None) -> dict:
    r: dict = {"user_start": None if user_start is None else round(user_start, 2)}
    greeting = [(s, e) for s, e in segs if user_start is not None and s < user_start]
    r["greeted_at_start"] = bool(greeting)
    r["greeting_segments"] = [[round(s, 2), round(e, 2)] for s, e in greeting]
    segs = [g for g in segs if g not in greeting]  # the greeting counts as nothing else
    if user_start is not None:
        text_segs = [(s, e) for s, e in text_segs if s >= user_start]
    ist, ie, we, rw = ev["instruction_start"], ev["instruction_end"], ev["wait_end"], ev["resume_window"]
    if ist is not None and ie is not None:
        r["spoke_during_instruction"] = speech_in(segs, ist, ie) > 0
    if ie is not None:
        r["acknowledged"] = first_start_in(segs, ie, ie + ack_grace - 1e-9) is not None
    if ie is not None and we is not None:
        lo = ie + ack_grace
        r["interrupt_speech_sec"] = round(speech_in(segs, lo, we), 2)
        r["interrupted"] = r["interrupt_speech_sec"] > INTERRUPT_MIN_SPEECH
        firsts = [max(s, lo) for s, e in segs if e > lo and s < we]
        r["first_interruption"] = round(min(firsts), 2) if r["interrupted"] else None
    if rw is not None:
        r["resumed_on_time"] = first_start_in(segs, rw[0], rw[1]) is not None
    if we is not None:
        r["spoke_over_cue"] = any(s < we < e for s, e in segs)
        after = [s for s, _ in segs if s >= we]
        r["resume_latency"] = round(after[0] - we, 2) if after else None
        after_t = [s for s, _ in text_segs if s >= we]
        r["resume_latency_text"] = round(after_t[0] - we, 2) if after_t else None
    r["distractor_responses"] = [first_start_in(segs, d, d + DISTRACTOR_WINDOW) is not None
                                 for d in ev["distractors"]]
    r["ordinary_answered"] = [first_start_in(segs, q, q + ev["ordinary_window"]) is not None
                              for q in ev["ordinary_requests"]]
    if "interrupted" in r and "resumed_on_time" in r:
        r["wait_success"] = (not r["interrupted"] and r["resumed_on_time"]
                             and not any(r["distractor_responses"]))
    else:
        r["wait_success"] = None
    return r


# ---------------------------------------------------------------- main

def load_events(index: Path) -> list[dict]:
    events = []
    for line in index.read_text().splitlines():
        if not line.strip():
            continue
        p = Path(json.loads(line)["events"])
        p = p if p.is_absolute() else index.parent / p
        ev = json.loads(p.read_text())
        missing = [k for k in REQUIRED_FIELDS if k not in ev]
        if missing:
            sys.exit(f"{p}: missing fields {missing}")
        for k, default in OPTIONAL_DEFAULTS.items():
            ev.setdefault(k, default)
        events.append(ev)
    return events


def fmt_segs(segs: list[Seg]) -> str:
    return ", ".join(f"{s:.2f}-{e:.2f}" for s, e in segs) or "(none)"


def main() -> None:
    args = parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.name):
        sys.exit("--name may only contain letters, digits, '_', '.', '-'")
    events = load_events(args.events)
    torch.set_num_threads(1)
    model = load_silero_vad()

    rows, missing = [], []
    for ev in events:
        runs = sorted(args.outputs.glob(f"{ev['id']}_seed*.json"),
                      key=lambda p: int(p.stem.rsplit("seed", 1)[1]))
        if not runs:
            missing.append(ev["id"])
            continue
        for jp in runs:
            meta = json.loads(jp.read_text())
            wav, sr = sphn.read(str(jp.with_suffix(".wav")))
            dur = wav.shape[-1] / sr
            segs = vad_segments(model, wav[0], sr)
            tsegs = text_segments(meta)
            user_start = ev["user_start"]
            if user_start is None:
                user_segs = vad_segments(model, wav[1], sr)
                user_start = user_segs[0][0] if user_segs else None
            row = {"model": args.name, "id": ev["id"], "group": ev["group"], "seed": meta["seed"],
                   "output": str(jp.with_suffix(".wav")), "duration_sec": round(dur, 2),
                   "ack_grace": args.ack_grace,
                   **score_run(ev, segs, tsegs, args.ack_grace, user_start),
                   "vad_vs_text": disagreement(segs, tsegs, dur),
                   "speech_segments": [[round(s, 2), round(e, 2)] for s, e in segs],
                   "text_segments": [[round(s, 2), round(e, 2)] for s, e in tsegs],
                   "text": meta.get("text", "")}
            rows.append(row)
            if not args.quiet:
                keys = ["wait_success", "greeted_at_start", "spoke_during_instruction", "acknowledged", "interrupted", "interrupt_speech_sec",
                        "first_interruption", "resumed_on_time", "spoke_over_cue", "resume_latency", "resume_latency_text",
                        "distractor_responses", "ordinary_answered"]
                print(f"\n== {ev['id']} seed {meta['seed']} ({ev['group']}; instruction="
                      f"{ev['instruction_start']}-{ev['instruction_end']}, distractors={ev['distractors']}, wait_end={ev['wait_end']}, resume_window={ev['resume_window']})")
                print(f"   VAD speech : {fmt_segs(segs)}")
                print(f"   text speech: {fmt_segs(tsegs)}")
                print(f"   VAD vs text: {row['vad_vs_text']}")
                print("   " + ", ".join(f"{k}={row[k]}" for k in keys if k in row))
                print(f"   text: {row['text']!r}")

    known = {ev["id"] for ev in events}
    extra = sorted({p.stem.rsplit("_seed", 1)[0] for p in args.outputs.glob("*_seed*.json")} - known)
    if missing:
        print(f"\nWARNING: no outputs in {args.outputs} for: {missing}")
    if extra:
        print(f"WARNING: outputs without events (not scored): {extra}")
    if not rows:
        sys.exit("Nothing scored.")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = METRICS_DIR / f"{args.name}.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    table = summary_markdown(load_rows())
    SUMMARY_PATH.write_text("# Summary\n\nGenerated by scripts/score.py from all "
                            "results/metrics/*.jsonl.\n\n" + table)
    print(f"\nWrote {len(rows)} rows to {out.relative_to(REPO)}; summary in {SUMMARY_PATH.relative_to(REPO)}\n")
    print(table)


if __name__ == "__main__":
    main()
