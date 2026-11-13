"""Select ~50 varied English voices from Fish Audio's official public library.

Only voices published by the "Fish Official" account are used: these are stock voices, so no
clones of real people or fictional characters end up in the training data. Character, effect,
ASMR and announcer voices, children/teens and multilingual voices are excluded, since the
users in our conversations should sound like ordinary adults.

Selection is a round-robin over (gender, accent) buckets so accents and genders are spread
out, taking the most-used (best-tested) voices first within each bucket.

Reads FISH_API_KEY from the environment.
Usage: python scripts/fish_voices.py [--n 50] [--out data/splits/voices_all.json]
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from fishaudio import FishAudio

FISH_OFFICIAL = "d8b0991f96b44e489422ca2ddf0bd31d"
EXCLUDE_TAGS = {"character-voice", "character", "asmr", "announcer", "child", "teen", "kid",
                "horror", "gaming", "anime", "robotic", "singing", "meditation"}
EXCLUDE_TITLE = re.compile(r"doll|demon|zombie|witch|ghost|monster|vampire|robot|radio|broadcast|"
                           r"horror|narrator", re.I)
OTHER_LANG_TAGS = {"arabic", "french", "russian", "spanish", "german", "japanese", "chinese",
                   "korean", "italian", "portuguese", "hindi"}
ACCENT_TAGS = {"en-us": "american", "en-gb": "british", "en-ie": "irish", "en-au": "australian",
               "en-ca": "canadian", "en-za": "south_african", "en-ng": "nigerian",
               "en-in": "indian", "en-sg": "singaporean"}
ACCENT_WORDS = {"new york": "american", "american": "american", "london": "british",
                "british": "british", "irish": "irish", "australian": "australian",
                "canadian": "canadian", "south african": "south_african", "nigerian": "nigerian",
                "indian": "indian", "singaporean": "singaporean", "scottish": "scottish"}


def labels(v: dict) -> dict | None:
    tags = {t.lower() for t in v["tags"]}
    desc = (v.get("description") or "").lower()
    if tags & EXCLUDE_TAGS or tags & OTHER_LANG_TAGS or EXCLUDE_TITLE.search(v["title"]):
        return None
    if re.search(r"\b(boy|girl|child|teen|teenage)\b", desc):
        return None
    if "female" in tags or re.search(r"\bwoman'?s?\b|\bfemale\b", desc):
        gender = "female"
    elif "male" in tags or re.search(r"\bman'?s?\b|\bmale\b", desc):
        gender = "male"
    else:
        return None
    accent = next((ACCENT_TAGS[t] for t in tags if t in ACCENT_TAGS), None)
    if accent is None:
        accent = next((a for w, a in ACCENT_WORDS.items() if w in desc or w in tags), "unspecified")
    age = next((a for a in ("young", "middle-aged", "old") if a in tags), "unspecified")
    return {"gender": gender, "accent": accent, "age": age}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--out", type=Path, default=Path("data/splits/voices_all.json"))
    args = p.parse_args()

    client = FishAudio()
    voices, page = [], 1
    while True:
        r = client.voices.list(page_size=100, page_number=page, author_id=FISH_OFFICIAL,
                               language="en", sort_by="task_count")
        voices += [v.model_dump(mode="json", by_alias=True) for v in r.items]
        if not r.items or len(voices) >= r.total:
            break
        page += 1

    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for v in voices:  # already sorted by task_count, descending
        if v["type"] != "tts" or v["state"] != "trained":
            continue
        lab = labels(v)
        if lab is None:
            continue
        buckets[(lab["gender"], lab["accent"])].append({
            "id": v["_id"], "title": v["title"], **lab, "tags": v["tags"],
            "description": (v.get("description") or "").strip(), "task_count": v["task_count"]})

    chosen, queues = [], {k: list(q) for k, q in sorted(buckets.items())}
    while len(chosen) < args.n and any(queues.values()):
        for k in queues:
            if queues[k] and len(chosen) < args.n:
                chosen.append(queues[k].pop(0))

    out = {"source": "Fish Audio public library, author 'Fish Official'",
           "total_official_english": len(voices),
           "eligible": sum(len(q) for q in buckets.values()), "n": len(chosen),
           "voices": chosen}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"{len(voices)} official English voices, {out['eligible']} eligible, {len(chosen)} chosen")


if __name__ == "__main__":
    main()
