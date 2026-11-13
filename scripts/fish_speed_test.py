"""Fish Audio TTS speed/concurrency test, timestamp check, and listening samples.

1. Sequential: 10 ten-word lines, each with a different voice (9 library voices + the Moshi
   clone). Reports time per line and audio duration. Saves the audio to data/tts_samples/.
2. Concurrency: batches of 2/4/8/16 simultaneous requests; reports errors (e.g. HTTP 429) and
   throughput (seconds of audio produced per wall-clock second).
3. Estimates the time to generate 1 h and 10 h of speech.
4. Timestamp check: one raw /v1/tts request (response headers + content type), and one ASR
   request to see whether Fish's timestamps are per word or per segment.

Reads FISH_API_KEY from the environment (never printed).
Usage: python scripts/fish_speed_test.py
"""

import io
import json
import os
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from fishaudio import FishAudio

SR = 24000
MODEL = "s2-pro"
OUT = Path("data/tts_samples")
LINES = [
    "Could you give me a minute to think about that?",
    "I'm talking to my friend, so please stay quiet.",
    "Okay, I'm back now. Where were we just before?",
    "What's the weather going to be like this weekend?",
    "Let me check my calendar and get back to you.",
    "Sure, take your time, I'll wait right here for you.",
    "Hey Sam, did you remember to buy the milk today?",
    "Can you recommend a good book about ancient history?",
    "Hold on, my sister is asking me something right now.",
    "Alright, I've decided. Let's go with the second option.",
]
MOSHI_EXTRA = [
    "Of course. Just let me know when you're ready to continue.",
    "Welcome back! So, have you decided which option you prefer?",
]


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


def wav_seconds(data: bytes) -> float:
    pcm, sr, ch, sw = parse_wav(data)
    return len(pcm) / (sr * ch * sw)


def wav_info(data: bytes) -> tuple[int, int]:
    _, sr, ch, _ = parse_wav(data)
    return sr, ch


def save_wav(data: bytes, path: Path) -> None:
    """Rewrite with a correct header (Fish's streamed header has a placeholder length)."""
    pcm, sr, ch, sw = parse_wav(data)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(ch); w.setsampwidth(sw); w.setframerate(sr); w.writeframes(pcm)


def main() -> None:
    client = FishAudio()
    voices = json.loads(Path("data/splits/voices_all.json").read_text())["voices"]
    moshi_id = json.loads(Path("data/voices/moshi_voice.json").read_text())["id"]
    # 9 varied library voices (every 5th, spreading across the round-robin order) + Moshi.
    picks = [(v["title"], v["id"]) for v in voices[::5][:9]] + [("Moshi", moshi_id)]
    OUT.mkdir(parents=True, exist_ok=True)

    def tts(text: str, ref: str) -> tuple[bytes, float]:
        t0 = time.perf_counter()
        audio = client.tts.convert(text=text, reference_id=ref, format="wav", model=MODEL)
        return audio, time.perf_counter() - t0

    # ---- 1. sequential
    print(f"== Sequential ({MODEL}, wav)")
    seq = []
    for i, ((title, ref), text) in enumerate(zip(picks, LINES)):
        audio, dt = tts(text, ref)
        dur = wav_seconds(audio)
        sr, ch = wav_info(audio)
        name = f"{i:02d}_{title.replace(' ', '_')}.wav"
        save_wav(audio, OUT / name)
        seq.append({"voice": title, "text": text, "sec": dt, "audio_sec": dur})
        print(f"  {title:10s} {dt:5.2f}s for {dur:4.1f}s audio ({sr} Hz, {ch} ch) -> {name}")
    for j, text in enumerate(MOSHI_EXTRA):
        audio, _ = tts(text, moshi_id)
        save_wav(audio, OUT / f"moshi_extra_{j}.wav")
    avg = sum(s["sec"] for s in seq) / len(seq)
    avg_audio = sum(s["audio_sec"] for s in seq) / len(seq)
    print(f"  average {avg:.2f}s per line, {avg_audio:.2f}s audio per line "
          f"(x{avg / avg_audio:.2f} real time, sequential)")

    # ---- 2. concurrency
    print("== Concurrency")
    conc = {}
    for n in (2, 4, 8, 16):
        jobs = [(LINES[k % len(LINES)], picks[k % 9][1]) for k in range(n)]
        t0 = time.perf_counter()
        errors, audio_sec = [], 0.0
        with ThreadPoolExecutor(n) as ex:
            futs = [ex.submit(tts, t, r) for t, r in jobs]
            for f in futs:
                try:
                    a, _ = f.result()
                    audio_sec += wav_seconds(a)
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{type(e).__name__}: {str(e)[:120]}")
        wall = time.perf_counter() - t0
        conc[n] = {"wall": wall, "ok": n - len(errors), "errors": errors,
                   "throughput": audio_sec / wall}
        print(f"  {n:2d} parallel: {wall:5.2f}s wall, {n - len(errors)}/{n} ok, "
              f"{audio_sec / wall:5.2f}s audio per second" + (f", errors: {errors[:2]}" if errors else ""))

    best_n = max((n for n in conc if not conc[n]["errors"]), key=lambda n: conc[n]["throughput"],
                 default=None)
    if best_n:
        tp = conc[best_n]["throughput"]
        print(f"== Estimate at {best_n} parallel ({tp:.2f}s audio/s): "
              f"1h of speech ~ {3600 / tp / 60:.1f} min, 10h ~ {36000 / tp / 60:.0f} min")
    print(f"   (sequential: 1h ~ {3600 * avg / avg_audio / 60:.0f} min)")

    # ---- 4. timestamp check
    print("== Timestamp check")
    key = os.environ["FISH_API_KEY"]
    r = httpx.post("https://api.fish.audio/v1/tts",
                   headers={"Authorization": f"Bearer {key}", "model": MODEL},
                   json={"text": LINES[0], "reference_id": picks[0][1], "format": "wav"},
                   timeout=120)
    hdrs = {k: v for k, v in r.headers.items()
            if k.lower() not in ("set-cookie", "authorization")}
    print(f"  /v1/tts status {r.status_code}, content-type {r.headers.get('content-type')}, "
          f"{len(r.content)} bytes")
    print(f"  response headers: {sorted(hdrs)}")
    print(f"  concurrency limit: {hdrs.get('ratelimit-limit-concurrency')}, "
          f"current: {hdrs.get('ratelimit-current-concurrency')}")
    print(f"  any timestamp/alignment header: "
          f"{[k for k in hdrs if any(w in k.lower() for w in ('time', 'align', 'word', 'mark'))]}")
    sample = next(OUT.glob("00_*.wav"))
    asr = client.asr.transcribe(audio=sample.read_bytes(), language="en", include_timestamps=True)
    seg = asr.segments[:3] if getattr(asr, "segments", None) else []
    print(f"  ASR text: {asr.text!r}")
    print(f"  ASR segments ({len(asr.segments or [])}): "
          f"{[s.model_dump() if hasattr(s, 'model_dump') else s for s in seg]}")


if __name__ == "__main__":
    main()
