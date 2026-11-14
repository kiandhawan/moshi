"""Render conversation scripts (scripts/generate_scripts.py) into training and eval files.

  python scripts/render.py --scripts data/scripts_pilot/train --out data/pilot/train --name pilot_train
  python scripts/render.py --scripts data/scripts_pilot/eval  --out data/pilot/eval  --name pilot_eval

Steps for each run:
 1. TTS every unique (voice, text) line with Fish Audio, 24 kHz, up to --workers (12)
    requests at once with retries; Moshi's lines use the cloned Moshi voice, user / friend /
    visitor lines the script's voice IDs. Clips are cached in data/tts_cache/ by (voice, text),
    trimmed of silence at both ends; Moshi's lines also get word timings (Fish ASR matched to
    the script words). A rerun makes no requests for clips already cached.
 2. Timeline: each turn starts gap_before after the previous speech ends, or, for anchored
    turns (deadlines, self-talk, extensions), offset seconds after the end of the anchor
    turn, using the real clip lengths. A user-side anchored turn that would overlap earlier
    user-side speech is pushed back (reported as a shift); Moshi's anchored turns keep their
    time (the deadline).
 3. render_lib.render_full mixes the file: right = user + friend/visitor (light reverb) +
    faint room noise, left = Moshi only; loudness normalization per speaker, greeting never
    before 0.3 s, exactly 1.5 s after the last speech, and it fails if a file is 90 s or more.

Outputs
  train: <out>/<id>.wav (stereo, left = Moshi), <id>.json (DailyTalk alignments,
         SPEAKER_MAIN only), <id>.events.json; data/indexes/<name>.jsonl (path, duration,
         events); <out>/manifest.jsonl (one row per file, see manifest_row).
  eval:  <out>/<id>.wav (user side only, mono, for run_inference.py) + <id>.events.json and
         data/indexes/<name>.jsonl for score.py; the full ideal stereo file + transcript +
         events in data/eval_full/ (indexed in data/indexes/eval_full.jsonl and
         data/indexes/<name>_full.jsonl) for eval_loss.py; <out>/manifest.jsonl.

Event times (seconds, file time) written to every events file:
  user_start, instruction_start/end, wait_end (deadline after any extension, the end of the
  early-end line, or the end of the return cue), resume_window [wait_end, wait_end + 4 s for
  think time, + 5 s otherwise], distractors (end times; Moshi should not respond) with
  distractor_windows [end, next user speech] and kinds, ordinary_requests (end times),
  moshi_turns and right_turns (who speaks when).

Reads FISH_API_KEY from the environment (never printed).
"""

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import sphn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_lib import (MAX_SEC, SR, cache_key, CACHE, moshi_voice_id, render_full,  # noqa: E402
                        synth, update_index)

ROOT = Path(__file__).resolve().parents[1]
INDEXES = ROOT / "data/indexes"
EVAL_FULL = ROOT / "data/eval_full"
NOISE_DBFS = (-58.0, -50.0)   # per-file background noise level range (right channel)
PUSH_GAP = 0.15               # min gap when an anchored user-side turn must be pushed back
RESUME_WINDOW = {"think_time": 4.0}
RESUME_WINDOW_DEFAULT = 5.0


def voice_of(turn: dict, script: dict, moshi: str) -> str:
    if turn["speaker"] == "moshi":
        return moshi
    if turn["speaker"] in ("friend", "visitor"):
        return script["voices"]["friend"]
    return script["voices"]["user"]


def prefetch(scripts: list[dict], client, moshi: str, workers: int) -> dict:
    """TTS (and, for Moshi, ASR) every unique line not yet cached, in parallel."""
    jobs = {}
    for s in scripts:
        for t in s["turns"]:
            v = voice_of(t, s, moshi)
            jobs[(v, t["text"])] = jobs.get((v, t["text"]), False) or t["speaker"] == "moshi"
    todo = {k: a for k, a in jobs.items()
            if not (CACHE / f"{cache_key(k[1], k[0])}.wav").exists()
            or (a and not (CACHE / f"{cache_key(k[1], k[0])}.json").exists())}
    stats = {"lines": len(jobs), "requested": len(todo), "tts_bytes": sum(len(k[1].encode()) for k in todo)}
    t0 = time.perf_counter()
    if todo:
        print(f"TTS: {len(jobs)} unique lines, {len(todo)} not cached ({stats['tts_bytes']} bytes); "
              f"{workers} at once", flush=True)
        done = 0
        with ThreadPoolExecutor(workers) as ex:
            futs = {ex.submit(synth, client, text, voice, align): (voice, text) for (voice, text), align in todo.items()}
            for f in as_completed(futs):
                f.result()  # raises after the retries are used up: fail loudly
                done += 1
                if done % 50 == 0 or done == len(todo):
                    print(f"  {done}/{len(todo)} clips ({time.perf_counter() - t0:.0f} s)", flush=True)
    stats["sec"] = round(time.perf_counter() - t0, 1)
    return stats


def timeline(script: dict, moshi: str) -> tuple[list[dict], list[str]]:
    """Absolute start/end of every turn from the real (cached) clip lengths."""
    out, notes = [], []
    cur = 0.0           # end of all speech so far
    user_end = 0.0      # end of right-channel speech so far
    for k, t in enumerate(script["turns"]):
        x, _ = synth(None, t["text"], voice_of(t, script, moshi), align=False)  # cached: no request
        dur = len(x) / SR
        if "anchor" in t:
            start = out[t["anchor"]["turn"]]["end"] + t["anchor"]["offset"]
            if t["speaker"] != "moshi" and start < user_end + PUSH_GAP:
                notes.append(f"turn {k} pushed {user_end + PUSH_GAP - start:.2f} s later (overlapped user speech)")
                start = user_end + PUSH_GAP
        else:
            start = cur + t["gap_before"]
        end = start + dur
        out.append({**t, "start": round(start, 3), "end": round(end, 3)})
        cur = max(cur, end)
        if t["speaker"] != "moshi":
            user_end = max(user_end, end)
    return out, notes


def events_for(script: dict, turns: list[dict]) -> dict:
    user = [t for t in turns if t["speaker"] != "moshi"]
    ev = {"user_start": user[0]["start"] if user else None, "type": script["type"],
          "conditions": script.get("conditions", [])}
    instr = next((t for t in turns if t.get("event") == "instruction"), None)
    if instr:
        ev["instruction_start"], ev["instruction_end"] = instr["start"], instr["end"]
        early = next((t for t in turns if t.get("kind") == "early_end"), None)
        cue = next((t for t in turns if t.get("event") == "return_cue"), None)
        if script["type"] == "think_time" and not early:
            d = script["deadline"]
            wait_end = turns[d["after_turn"]]["end"] + d["sec"]
        else:
            wait_end = (early or cue)["end"]
        ev["wait_end"] = round(wait_end, 3)
        ev["resume_window"] = [round(wait_end, 3),
                               round(wait_end + RESUME_WINDOW.get(script["type"], RESUME_WINDOW_DEFAULT), 3)]
        ev["wait_sec"] = round(wait_end - instr["end"], 3)
    dis = [t for t in turns if t.get("event") == "distractor"]
    ev["distractors"] = [t["end"] for t in dis]
    ev["distractor_kinds"] = [t.get("kind") for t in dis]
    ev["distractor_windows"] = []
    for t in dis:  # until the next user-side speech starts (e.g. the rest of a paused sentence)
        nxt = [u["start"] for u in user if u["start"] > t["end"]]
        ev["distractor_windows"].append([t["end"], round(nxt[0], 3) if nxt else None])
    ev["ordinary_requests"] = [t["end"] for t in turns if t.get("event") == "ordinary_request"]
    ev["right_turns"] = [{"speaker": t["speaker"], "start": t["start"], "end": t["end"],
                          **({"event": t["event"]} if t.get("event") else {}),
                          **({"kind": t["kind"]} if t.get("kind") else {})} for t in user]
    ev["notes"] = f"Rendered from script {script['id']} by scripts/render.py."
    return ev


def manifest_row(script: dict, row: dict, ev: dict, turns: list[dict], path: str) -> dict:
    spans = sorted((t["start"], t["end"]) for t in turns)
    speech, s0, e0 = 0.0, None, None
    for s, e in spans:  # union of speech on both channels
        if e0 is None or s > e0:
            if e0 is not None:
                speech += e0 - s0
            s0, e0 = s, e
        else:
            e0 = max(e0, e)
    if e0 is not None:
        speech += e0 - s0
    total = row["duration"]
    return {
        "id": script["id"], "split": script["split"], "type": script["type"],
        "wait_type": script.get("wait_type"), "conditions": script.get("conditions", []),
        "topic": script["topic"], "topics": script["topics"], "voices": script["voices"],
        "friend_name": script.get("friend_name"), "phrasing_ids": script["phrasing_ids"],
        "block_ids": script["block_ids"], "path": path,
        "total_sec": total, "speech_sec": round(speech, 3), "silence_sec": round(total - speech, 3),
        "moshi_speech_sec": round(sum(t["end"] - t["start"] for t in turns if t["speaker"] == "moshi"), 3),
        "wait_sec": ev.get("wait_sec"),
        "events": {k: ev[k] for k in ("user_start", "instruction_start", "instruction_end", "wait_end",
                                      "resume_window", "distractors", "distractor_kinds",
                                      "ordinary_requests") if k in ev},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scripts", type=Path, required=True, help="Folder of script JSON files.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--name", required=True, help="Index name: data/indexes/<name>.jsonl")
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()

    scripts = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(args.scripts / "*_*.json")))]
    if not scripts:
        sys.exit(f"No scripts in {args.scripts}")
    splits = {s["split"] for s in scripts}
    if len(splits) != 1:
        sys.exit(f"Mixed splits in {args.scripts}: {splits}")
    split = splits.pop()
    from fishaudio import FishAudio
    client = FishAudio()
    moshi = moshi_voice_id()
    stats = prefetch(scripts, client, moshi, args.workers)

    args.out.mkdir(parents=True, exist_ok=True)
    full_dir = args.out if split == "train" else EVAL_FULL
    rows, full_rows, manifest, problems = [], [], [], []
    for s in scripts:
        turns, notes = timeline(s, moshi)
        ev = events_for(s, turns)
        noise = float(np.random.default_rng(int(hashlib.sha1(s["id"].encode()).hexdigest()[:8], 16))
                      .uniform(*NOISE_DBFS))
        spec = {
            "id": s["id"], "group": s["type"], "noise_dbfs": round(noise, 1),
            "user_lines": [{"text": t["text"], "start": t["start"], "speaker": t["speaker"],
                            "voice": voice_of(t, s, moshi)} for t in turns if t["speaker"] != "moshi"],
            "moshi_lines": [{"text": t["text"], "start": t["start"], "kind": t.get("role", "reply")}
                            for t in turns if t["speaker"] == "moshi"],
            "events": {**ev, "noise_dbfs": round(noise, 1)},
        }
        try:
            row = render_full(spec, None, full_dir, INDEXES)  # all clips are cached: no requests
        except ValueError as e:
            raise SystemExit(f"FAILED {s['id']}: {e}")
        if row["duration"] >= MAX_SEC:  # render_full already refuses this; belt and braces
            raise SystemExit(f"FAILED {s['id']}: {row['duration']} s is not under {MAX_SEC} s")
        problems += [f"{s['id']}: {n}" for n in notes]
        if split == "train":
            rows.append(row)
            manifest.append(manifest_row(s, row, ev, turns, row["path"]))
        else:
            full_rows.append(row)
            x, _ = sphn.read(str(INDEXES / row["path"]))
            user_wav = args.out / f"{s['id']}.wav"
            sphn.write_wav(str(user_wav), x[1:2], sample_rate=SR)  # user side only, as the model hears it
            full_ev = json.loads((INDEXES / row["events"]).read_text())
            ev_p = args.out / f"{s['id']}.events.json"
            ev_p.write_text(json.dumps({k: v for k, v in full_ev.items() if k != "moshi_turns"}, indent=2) + "\n")
            rel = lambda q: os.path.relpath(q.resolve(), INDEXES.resolve())  # noqa: E731
            rows.append({"path": rel(user_wav), "duration": row["duration"], "events": rel(ev_p)})
            manifest.append(manifest_row(s, row, ev, turns, rel(user_wav)))
        print(f"{s['id']} [{s['type']}] {row['duration']:.1f} s" + (f"  ({'; '.join(notes)})" if notes else ""),
              flush=True)

    update_index(INDEXES / f"{args.name}.jsonl", rows)
    if split == "eval":
        update_index(INDEXES / "eval_full.jsonl", full_rows)
        update_index(INDEXES / f"{args.name}_full.jsonl", full_rows)
    (args.out / "manifest.jsonl").write_text("".join(json.dumps(m) + "\n" for m in manifest))
    tot = sum(m["total_sec"] for m in manifest)
    sp = sum(m["speech_sec"] for m in manifest)
    print(f"\n{len(manifest)} {split} files -> {args.out}; index data/indexes/{args.name}.jsonl"
          + (f" (+ {args.name}_full.jsonl, eval_full.jsonl)" if split == "eval" else ""))
    print(f"  audio {tot / 60:.1f} min, speech {sp / 60:.1f} min, silence {(tot - sp) / 60:.1f} min; "
          f"length {min(m['total_sec'] for m in manifest):.1f}-{max(m['total_sec'] for m in manifest):.1f} s")
    print(f"  TTS: {stats['lines']} unique lines, {stats['requested']} requested "
          f"({stats['tts_bytes']} bytes, ~${stats['tts_bytes'] / 1e6 * 15:.2f}) in {stats['sec']} s")
    for pr in problems:
        print("  NOTE", pr)


if __name__ == "__main__":
    main()
