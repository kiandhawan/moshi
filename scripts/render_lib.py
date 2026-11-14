"""Render a timed conversation into a training-format stereo file.

This is the rendering step of the data generator. Every conversation (train or eval) goes
through `render_full`, which writes:

  <out_dir>/<id>.wav          stereo 24 kHz, left = ideal Moshi, right = user + friend
  <out_dir>/<id>.json         {"alignments": [[word, [start, end], "SPEAKER_MAIN"], ...]}
                              (Moshi's words only, the format train.py reads)
  <out_dir>/<id>.events.json  the standard events fields (see score.py) plus
                              "moshi_turns": [{kind, start, end, text}] for the ideal side

and returns the index row {"path", "duration", "events"} with paths relative to the index
folder, the same layout as data/indexes/human.jsonl. Because the user is on the right
channel, the same .wav also works directly as input to run_inference.py, which reads the
right channel of stereo files.

Conversation spec (a dict):
  id, group
  user_audio   optional path to a recorded user side (mono, or right channel of stereo)
  user_lines   [{text, voice, start}]  TTS lines for the user/friend side
  moshi_lines  [{text, start, kind}]   kind: greeting | ack | resume | ordinary
  events       the standard events fields; "id" and "group" are filled in, and
               "user_start" (the user's first words) from user_lines if not given

TTS (Fish Audio, 24 kHz) and word timings (Fish ASR) are cached in data/tts_cache/, keyed
by model + voice + text, so re-rendering costs nothing. Reads FISH_API_KEY from the
environment; it is never printed.
"""

import difflib
import hashlib
import io
import json
import os
import re
import wave
from pathlib import Path

import numpy as np
import sphn

SR = 24000
MODEL = "s2-pro"
CACHE = Path("data/tts_cache")
MAIN = "SPEAKER_MAIN"
TAIL_SEC = 1.5          # digital silence after the last speech on either channel
GREETING_MIN_START = 0.3  # the opening greeting never starts earlier (CLAUDE.md)
MAX_SEC = 100.0         # must stay below training duration_sec (see CLAUDE.md)


def moshi_voice_id() -> str:
    return json.loads(Path("data/voices/moshi_voice.json").read_text())["id"]


def parse_wav(data: bytes) -> tuple[bytes, int, int, int]:
    """Return (pcm, sample_rate, channels, sample_width). Streamed WAVs from Fish carry a
    placeholder length in the header, so read the fmt chunk and take everything after the
    'data' chunk header as audio instead of trusting the declared sizes."""
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE", "not a WAV"
    pos, fmt = 12, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], int.from_bytes(data[pos + 4:pos + 8], "little")
        if cid == b"fmt ":
            fmt = data[pos + 8:pos + 8 + 16]
        if cid == b"data":
            pcm = data[pos + 8:]
            ch = int.from_bytes(fmt[2:4], "little")
            sr = int.from_bytes(fmt[4:8], "little")
            sw = int.from_bytes(fmt[14:16], "little") // 8
            return pcm[: len(pcm) - len(pcm) % (ch * sw)], sr, ch, sw
        pos += 8 + size + (size % 2)
    raise ValueError("no data chunk")


def to_wav_bytes(x: np.ndarray, sr: int = SR) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def trim(x: np.ndarray, rel_db: float = -35.0, margin: float = 0.03) -> np.ndarray:
    """Cut leading/trailing silence so a line's `start` is its speech onset."""
    hop = SR // 100
    n = len(x) // hop
    if n == 0:
        return x
    rms = np.sqrt((x[: n * hop].reshape(n, hop).astype(np.float64) ** 2).mean(1))
    on = np.nonzero(rms > rms.max() * 10 ** (rel_db / 20))[0]
    a = max(0, on[0] * hop - int(margin * SR))
    b = min(len(x), (on[-1] + 1) * hop + int(margin * SR))
    return x[a:b]


def _norm(w: str) -> str:
    return re.sub(r"[^a-z0-9']", "", w.lower())


def map_words(script: str, asr: list[tuple[str, float, float]], dur: float):
    """Give each script word (punctuation kept, as in DailyTalk) a time span. Words the ASR
    recognised take its timestamps; the rest are interpolated between their neighbours,
    proportionally to their length."""
    words = script.split()
    spans: list[list[float] | None] = [None] * len(words)
    sm = difflib.SequenceMatcher(a=[_norm(w) for w in words], b=[_norm(t) for t, _, _ in asr],
                                 autojunk=False)
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            _, s, e = asr[blk.b + k]
            spans[blk.a + k] = [s, e]
    i = 0
    while i < len(words):  # fill runs of unmatched words
        if spans[i] is not None:
            i += 1
            continue
        j = i
        while j < len(words) and spans[j] is None:
            j += 1
        lo = spans[i - 1][1] if i > 0 else 0.0
        hi = spans[j][0] if j < len(words) else dur
        lens = np.array([max(len(w), 1) for w in words[i:j]], dtype=float)
        edges = lo + (hi - lo) * np.concatenate([[0], np.cumsum(lens) / lens.sum()])
        for k in range(i, j):
            spans[k] = [float(edges[k - i]), float(edges[k - i + 1])]
        i = j
    return [(w, round(s, 3), round(max(e, s + 0.01), 3)) for w, (s, e) in zip(words, spans)]


def synth(client, text: str, voice: str) -> tuple[np.ndarray, list[tuple[str, float, float]]]:
    """TTS one line (cached). Returns trimmed 24 kHz mono audio and word timings relative
    to the start of that audio."""
    key = hashlib.sha1(f"{MODEL}|{voice}|{text}".encode()).hexdigest()[:16]
    wav_p, ali_p = CACHE / f"{key}.wav", CACHE / f"{key}.json"
    if wav_p.exists() and ali_p.exists():
        x, _ = sphn.read(str(wav_p), sample_rate=SR)
        return x[0], clamp_words(json.loads(ali_p.read_text())["words"], len(x[0]) / SR)
    from fishaudio.types import TTSConfig
    raw = client.tts.convert(text=text, reference_id=voice, format="wav", model=MODEL,
                             config=TTSConfig(format="wav", sample_rate=SR))
    pcm, sr, ch, sw = parse_wav(raw)
    assert sr == SR and sw == 2, (sr, sw)
    x = np.frombuffer(pcm, dtype="<i2").reshape(-1, ch)[:, 0].astype(np.float32) / 32768
    x = trim(x)
    asr = client.asr.transcribe(audio=to_wav_bytes(x), language="en", include_timestamps=True)
    asr_words = [(s.text.strip(), s.start, s.end) for s in asr.segments if s.text.strip()]
    words = map_words(text, asr_words, len(x) / SR)
    CACHE.mkdir(parents=True, exist_ok=True)
    sphn.write_wav(str(wav_p), x[None], sample_rate=SR)
    ali_p.write_text(json.dumps({"model": MODEL, "voice": voice, "text": text,
                                 "asr_text": asr.text, "words": words}, indent=1))
    return x, clamp_words(words, len(x) / SR)


def clamp_words(words, dur: float) -> list[tuple[str, float, float]]:
    """Keep word times inside the clip: the ASR's last timestamp can run a few tens of ms
    past the end of the (trimmed) audio."""
    return [(w, round(min(s, dur - 0.01), 3), round(min(e, dur), 3)) for w, s, e in words]


def _place(track: np.ndarray, x: np.ndarray, start: float) -> np.ndarray:
    a = int(round(start * SR))
    if a + len(x) > len(track):
        track = np.pad(track, (0, a + len(x) - len(track)))
    track[a:a + len(x)] += x
    return track


def render_full(spec: dict, client, out_dir: Path, index_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    right = np.zeros(0, dtype=np.float32)
    speech_ends = [0.0]  # end of every piece of speech on either channel
    if spec.get("user_audio"):
        u, _ = sphn.read(spec["user_audio"], sample_rate=SR)
        right = (u[1] if u.shape[0] == 2 else u[0]).astype(np.float32)
        speech_ends.append(last_speech_end(right))
    for ln in spec.get("user_lines", []):
        x, _ = synth(client, ln["text"], ln["voice"])
        right = _place(right, x, ln["start"])
        speech_ends.append(ln["start"] + len(x) / SR)  # clips are trimmed to their speech

    left = np.zeros(0, dtype=np.float32)
    voice = moshi_voice_id()
    alignments, turns = [], []
    for ln in sorted(spec["moshi_lines"], key=lambda l: l["start"]):
        x, words = synth(client, ln["text"], voice)
        start, end = ln["start"], ln["start"] + len(x) / SR
        if turns and start < turns[-1]["end"]:
            raise ValueError(f"{spec['id']}: Moshi line at {start}s overlaps the previous one")
        if ln["kind"] == "greeting" and start < GREETING_MIN_START:
            raise ValueError(f"{spec['id']}: greeting starts at {start}s, before {GREETING_MIN_START}s")
        speech_ends.append(end)
        left = _place(left, x, start)
        alignments += [[w, [round(start + s, 3), round(start + e, 3)], MAIN] for w, s, e in words]
        turns.append({"kind": ln["kind"], "start": round(start, 3), "end": round(end, 3),
                      "text": ln["text"]})

    # The file ends exactly TAIL_SEC after the last speech on either channel. Anything a
    # recording has after that is non-speech (room noise) and is cut, so the tail is silent.
    n = int(round((max(speech_ends) + TAIL_SEC) * SR))
    left, right = left[:n], right[:n]
    dur = n / SR
    if dur >= MAX_SEC:
        raise ValueError(f"{spec['id']}: {dur:.1f}s is not shorter than duration_sec={MAX_SEC}")
    left, right = np.pad(left, (0, n - len(left))), np.pad(right, (0, n - len(right)))
    # Avoid clipping, per channel, so a recorded user side is never altered by a loud TTS line.
    left, right = (ch * (0.99 / max(np.abs(ch).max(), 1e-9)) if np.abs(ch).max() > 0.99 else ch
                   for ch in (left, right))

    cid = spec["id"]
    wav_p = out_dir / f"{cid}.wav"
    sphn.write_wav(str(wav_p), np.stack([left, right]), sample_rate=SR)
    wav_p.with_suffix(".json").write_text(json.dumps({"alignments": alignments}, indent=1))
    events = {"id": cid, "group": spec["group"], **spec.get("events", {}), "moshi_turns": turns}
    if events.get("user_start") is None and spec.get("user_lines"):
        events["user_start"] = round(min(l["start"] for l in spec["user_lines"]), 3)
    ev_p = out_dir / f"{cid}.events.json"
    ev_p.write_text(json.dumps(events, indent=2) + "\n")
    rel = lambda p: os.path.relpath(p.resolve(), index_dir.resolve())  # noqa: E731
    return {"path": rel(wav_p), "duration": round(dur, 3), "events": rel(ev_p)}


def last_speech_end(x: np.ndarray) -> float:
    """End (s) of the last VAD speech segment in a 24 kHz mono signal (0 if none); same VAD
    settings as score.py."""
    import torch
    from silero_vad import load_silero_vad
    from score import vad_segments
    torch.set_num_threads(1)
    segs = vad_segments(load_silero_vad(), x, SR)
    return segs[-1][1] if segs else 0.0


def overlaps_with_user(row: dict, index_dir: Path, min_sec: float = 0.1) -> list[str]:
    """List Moshi turns that overlap user-side speech (Silero VAD on the right channel)."""
    import torch
    from silero_vad import load_silero_vad
    from score import vad_segments
    torch.set_num_threads(1)
    x, sr = sphn.read(str(index_dir / row["path"]))
    user = vad_segments(load_silero_vad(), x[1], sr)
    turns = json.loads((index_dir / row["events"]).read_text())["moshi_turns"]
    out = []
    for t in turns:
        ov = sum(max(0.0, min(t["end"], e) - max(t["start"], s)) for s, e in user)
        if ov > min_sec:
            out.append(f"{t['kind']} {t['start']:.2f}-{t['end']:.2f} overlaps user speech by {ov:.2f}s")
    return out


def update_index(index: Path, rows: list[dict]) -> None:
    """Add rows to an index jsonl, replacing any existing row with the same path."""
    old = [json.loads(l) for l in index.read_text().splitlines() if l.strip()] if index.exists() else []
    new_paths = {r["path"] for r in rows}
    keep = [r for r in old if r["path"] not in new_paths]
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text("".join(json.dumps(r) + "\n" for r in keep + rows))
