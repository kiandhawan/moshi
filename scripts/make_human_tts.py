"""TTS twins of the human test set: data/human_tts/<name>_tts.wav (+ events, index human_tts).

Each human recording was transcribed with Whisper medium.en (word timestamps,
data/human_tts/whisper_words.json), split into utterances at pauses >= 0.4 s, and each utterance
was assigned a speaker by hand from its content (UTTERANCES below; only "12:28 am" in
talktoroommate is the roommate, Ryan). Every utterance is re-voiced with Fish Audio (s2-pro,
render_lib.synth, cached and trimmed) with held-out eval voices - one for the user, a different
one for the other person - normalized to the eval loudness (user -21 dBFS, other -25 dBFS), and
placed at its original onset (Whisper start of its first word), so the original events files stay
valid. A line is sped up with ffmpeg atempo (pitch kept) when it would run into the next utterance
(leaving a 0.15 s gap) or past the end of the file, or past the event time its end defines
(instruction_end, wait_end = end of the return cue, ordinary request ends), capped at
MAX_SPEEDUP; any remaining overrun is reported. Later lines never move. Light
pink room noise at -55 dBFS (as in the rendered eval files). Same length and format as the
original (mono, 24 kHz).

Events: copied from data/human/<name>.events.json with id <name>_tts, cluster human_tts_<name>
and the group tag "human" replaced by "human_tts". Adjustments and the TTS line ends next to the
event times they define are written to data/human_tts/report.json and printed.
Needs FISH_API_KEY (set -a && . ~/.config/fish_audio/env && set +a). Usage: python scripts/make_human_tts.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import sphn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_lib import LEVEL_DBFS, SR, normalize, room_noise, synth  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data/human_tts"
GAP = 0.15
MAX_SPEEDUP = 1.35   # beyond this TTS speech sounds unnatural; the rest is reported as overrun
NOISE_DBFS = -55.0
# (start s = Whisper onset, speaker, text) - speakers assigned by hand from the transcripts
UTTERANCES = {
    "extend10sec": [(2.04, "user", "Um, I don't know yet. Ask me again in 10 seconds."),
                    (11.12, "user", "Actually, wait 10 more seconds.")],
    "talktoroommate": [(2.36, "user", "Great, thanks."),
                       (3.60, "user", "Let me talk to my roommate for a few seconds."),
                       (6.40, "user", "Ryan, what time is it?"),
                       (8.48, "other", "It's 12:28 am."),
                       (10.74, "user", "Great, thanks."),
                       (12.54, "user", "Okay, I'm back.")],
    "regularconvo": [(4.22, "user", "Hi, I'm okay. How are you?"),
                     (11.74, "user", "What should I make for dinner today?"),
                     (19.88, "user", "Do you have any easier ideas?"),
                     (28.12, "user", "Okay, thanks.")],
}


def voices() -> dict[str, str]:
    ev = json.loads((REPO / "data/splits/voices.json").read_text())["eval"]
    by = {v["title"]: v["id"] for v in ev}
    return {"user": by["Chloe"], "other": by["Ruan"]}  # two held-out eval voices, chosen to sound different


def atempo(x: np.ndarray, factor: float) -> np.ndarray:
    with tempfile.TemporaryDirectory() as d:
        a, b = Path(d) / "in.wav", Path(d) / "out.wav"
        sphn.write_wav(str(a), x[None], sample_rate=SR)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(a), "-filter:a", f"atempo={factor:.4f}",
                        "-ar", str(SR), "-ac", "1", str(b)], check=True)
        return sphn.read(str(b), sample_rate=SR)[0][0]


def main() -> None:
    from fishaudio import FishAudio
    client = FishAudio()
    v = voices()
    OUT.mkdir(parents=True, exist_ok=True)
    report, index = {}, []
    for name, utts in UTTERANCES.items():
        orig, sr = sphn.read(str(REPO / f"data/human/{name}.wav"), sample_rate=SR)
        n = orig.shape[-1]
        track = room_noise(n, NOISE_DBFS, seed=hash(name) % 2**31)
        ev0 = json.loads((REPO / f"data/human/{name}.events.json").read_text())
        ends = {"instruction_end": ev0.get("instruction_end"), "wait_end": ev0.get("wait_end")}
        ends.update({f"ordinary_request_{j + 1}": q for j, q in enumerate(ev0.get("ordinary_requests", []))})
        lines = []
        for i, (start, spk, text) in enumerate(utts):
            nxt = utts[i + 1][0] if i + 1 < len(utts) else n / SR
            # event times defined by this line's end: inside (start, next onset]
            evs = {k: t for k, t in ends.items() if t is not None and start < t <= nxt}
            x, _ = synth(client, text, v[spk], align=False)
            x = normalize(x, LEVEL_DBFS[spk])
            limit = nxt - GAP - start
            if evs:
                limit = min(limit, min(evs.values()) - start)
            dur, factor = len(x) / SR, 1.0
            if dur > limit:
                factor = min(dur / limit, MAX_SPEEDUP)
                x = atempo(x, factor)
            a = int(round(start * SR))
            b = min(n, a + len(x))
            track[a:b] += x[: b - a]
            lines.append({"start": start, "speaker": spk, "text": text, "tts_sec": round(dur, 2),
                          "end": round(start + (b - a) / SR, 2), "speedup": round(factor, 3),
                          "event_ends": {k: round(start + (b - a) / SR - t, 2) for k, t in evs.items()}})
        peak = np.abs(track).max()
        if peak > 0.99:
            track *= 0.99 / peak
        sphn.write_wav(str(OUT / f"{name}_tts.wav"), track[None].astype(np.float32), sample_rate=SR)
        ev = json.loads((REPO / f"data/human/{name}.events.json").read_text())
        ev.update(id=f"{name}_tts", cluster=f"human_tts_{name}",
                  groups=["human_tts" if g == "human" else g for g in ev["groups"]],
                  notes=f"TTS twin of data/human/{name}.wav (scripts/make_human_tts.py); event times copied from the "
                        f"original. Original notes: {ev.get('notes', '')}")
        (OUT / f"{name}_tts.events.json").write_text(json.dumps(ev, indent=1))
        index.append({"path": f"../human_tts/{name}_tts.wav", "duration": round(n / SR, 3),
                      "events": f"../human_tts/{name}_tts.events.json"})
        keys = {k: ev.get(k) for k in ("user_start", "instruction_end", "wait_end", "distractors", "ordinary_requests")
                if ev.get(k) not in (None, [])}
        report[name] = {"lines": lines, "events": keys}
        print(f"\n{name} ({n / SR:.1f} s): events {keys}")
        for ln in lines:
            ov = "  " + ", ".join(f"ends {d:+.2f} s vs {k}" for k, d in ln["event_ends"].items()) if ln["event_ends"] else ""
            print(f"  {ln['start']:6.2f}-{ln['end']:6.2f}  {ln['speaker']:5s} x{ln['speedup']:<5} {ln['text']}{ov}")
    (OUT / "report.json").write_text(json.dumps({"voices": {"user": "Chloe (eval)", "other": "Ruan (eval)"},
                                                 "files": report}, indent=1))
    (REPO / "data/indexes/human_tts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in index))
    print("\nWrote data/human_tts/*_tts.wav, *.events.json, report.json and data/indexes/human_tts.jsonl")


if __name__ == "__main__":
    main()
