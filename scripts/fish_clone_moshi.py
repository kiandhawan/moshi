"""Create (or reuse) a private Fish Audio voice cloned from the Moshi reference audio.

Reads FISH_API_KEY from the environment. Reuses an existing voice of yours with the same
title instead of creating a duplicate. Writes the voice ID to data/voices/moshi_voice.json.

Usage: python scripts/fish_clone_moshi.py [--ref data/voices/moshi_reference.wav]
"""

import argparse
import json
import time
from pathlib import Path

from fishaudio import FishAudio

TITLE = "Moshi reference clone"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ref", type=Path, default=Path("data/voices/moshi_reference.wav"))
    p.add_argument("--out", type=Path, default=Path("data/voices/moshi_voice.json"))
    args = p.parse_args()

    client = FishAudio()
    mine = client.voices.list(page_size=100, self_only=True, title=TITLE).items
    existing = [v for v in mine if v.title == TITLE]
    if existing:
        voice = existing[0]
        print(f"Reusing existing voice {voice.id} (state={voice.state})")
    else:
        voice = client.voices.create(
            title=TITLE,
            voices=[args.ref.read_bytes()],
            description="Cloned from Moshi (kyutai/moshiko) base-model outputs, for synthetic "
                        "training data. Private.",
            visibility="private",
            enhance_audio_quality=False,  # keep Moshi's own timbre/codec character
        )
        print(f"Created voice {voice.id} (state={voice.state})")

    for _ in range(60):  # wait until it's ready
        if voice.state == "trained":
            break
        time.sleep(5)
        voice = client.voices.get(voice.id)
    print(f"state={voice.state}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "id": voice.id, "title": voice.title, "state": voice.state,
        "visibility": voice.visibility, "reference": str(args.ref),
        "reference_manifest": str(args.ref.with_suffix(".json")),
    }, indent=2) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
