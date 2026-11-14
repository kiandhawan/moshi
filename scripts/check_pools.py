"""Validate the building-block pools in data/pools/ against data/pools/README.md.

Checks the format, ID prefixes, counts and text rules (no digits or odd symbols, no
return-cue wording in side conversations, lengths). Prints problems and a summary.
Exit code 1 if any problem is found.

Usage: python scripts/check_pools.py [topic ...]   (default: every topic in topics.json)
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOPICS = json.loads((ROOT / "data/splits/topics.json").read_text())
BAD_CHARS = re.compile(r"[^A-Za-z ,.?!'\-/{}]")
CUE_WORDS = re.compile(r"\b(i'm back|i am back|where were we|okay moshi|ok moshi|moshi|sorry about that)\b", re.I)
WANT = {"train": {"contexts": 3, "side_conversations": 3, "ordinary_dialogues": 4, "option_sets": 0},
        "eval": {"contexts": 3, "side_conversations": 3, "ordinary_dialogues": 3, "option_sets": 2}}


def words(t: str) -> int:
    return len(t.replace(" / ", " ").split())


def check_text(where: str, t: str, problems: list[str]) -> None:
    if not isinstance(t, str) or not t.strip():
        problems.append(f"{where}: empty text")
        return
    if re.search(r"\d", t):
        problems.append(f"{where}: digits in {t!r}")
    bad = set(BAD_CHARS.findall(t.replace("{friend}", "")))
    if bad:
        problems.append(f"{where}: characters {sorted(bad)} in {t!r}")
    if "{" in t.replace("{friend}", ""):
        problems.append(f"{where}: unknown slot in {t!r}")


def check_topic(topic: str, split: str) -> tuple[list[str], dict]:
    p = ROOT / f"data/pools/topics/{topic}.json"
    if not p.exists():
        return [f"{topic}: missing {p.relative_to(ROOT)}"], {}
    problems: list[str] = []
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        return [f"{topic}: invalid JSON: {e}"], {}
    if d.get("topic") != topic:
        problems.append(f"{topic}: 'topic' field is {d.get('topic')!r}")
    stats = {"words": 0}
    ids = set()
    for key, n in WANT[split].items():
        items = d.get(key, [])
        stats[key] = len(items)
        if len(items) < n:
            problems.append(f"{topic}: {key} has {len(items)}, want {n}")
        if n == 0 and items:
            problems.append(f"{topic}: {key} only for eval topics")
        for it in items:
            iid = it.get("id", "?")
            w = f"{topic}/{iid}"
            if not iid.startswith(topic + "."):
                problems.append(f"{w}: id must start with '{topic}.'")
            if iid in ids:
                problems.append(f"{w}: duplicate id")
            ids.add(iid)
            if key == "option_sets":
                texts = [it.get("lead_in", "")] + it.get("options", []) + it.get("moshi_comment", [])
                for t in texts:
                    check_text(w, t, problems)
                if not 3 <= len(it.get("options", [])) <= 4:
                    problems.append(f"{w}: want 3-4 options")
                if len(it.get("moshi_comment", [])) < 2:
                    problems.append(f"{w}: want 2 moshi_comment")
                stats["words"] += sum(words(t) for t in texts if isinstance(t, str))
                continue
            turns = it.get("turns", [])
            for k, tn in enumerate(turns):
                check_text(f"{w}[{k}]", tn.get("text", ""), problems)
            speakers = [tn.get("speaker") for tn in turns]
            nw = sum(words(tn.get("text", "")) for tn in turns)
            stats["words"] += nw
            if not turns or speakers[0] != "user":
                problems.append(f"{w}: must start with the user")
            if any(speakers[i] == speakers[i + 1] for i in range(len(speakers) - 1)):
                problems.append(f"{w}: two consecutive turns by the same speaker")
            if key == "contexts":
                if speakers and speakers[-1] != "moshi":
                    problems.append(f"{w}: context must end with a Moshi turn")
                if set(speakers) - {"user", "moshi"} or not 2 <= len(turns) <= 4:
                    problems.append(f"{w}: context wants 2-4 user/moshi turns")
                if not 12 <= nw <= 55:
                    problems.append(f"{w}: context has {nw} words, want about 15-45")
                for f, lo in (("think_reask", 2), ("welcome_back", 2), ("endings", 2)):
                    if len(it.get(f, [])) < lo:
                        problems.append(f"{w}: want at least {lo} {f}")
                for t in it.get("think_reask", []) + it.get("welcome_back", []):
                    check_text(w, t, problems)
                for e in it.get("endings", []):
                    check_text(w, e.get("user", ""), problems)
                    check_text(w, e.get("moshi", ""), problems)
                stats["words"] += sum(words(t) for t in it.get("think_reask", []) + it.get("welcome_back", []))
                stats["words"] += sum(words(e.get("user", "")) + words(e.get("moshi", "")) for e in it.get("endings", []))
            elif key == "side_conversations":
                if set(speakers) - {"user", "friend"} or not 4 <= len(turns) <= 12:
                    problems.append(f"{w}: side conversation wants 4-12 user/friend turns")
                if not 25 <= nw <= 140:
                    problems.append(f"{w}: side conversation has {nw} words, want about 30-130")
                for k, tn in enumerate(turns):
                    if CUE_WORDS.search(tn.get("text", "")):
                        problems.append(f"{w}[{k}]: return-cue wording in a side conversation")
                    if " / " in tn.get("text", ""):
                        problems.append(f"{w}[{k}]: ' / ' only in ordinary dialogues")
                if not any(tn["speaker"] == "user" and "?" in tn.get("text", "") for tn in turns):
                    problems.append(f"{w}: no question from the user to the friend")
            elif key == "ordinary_dialogues":
                if set(speakers) - {"user", "moshi"} or not 6 <= len(turns) <= 12:
                    problems.append(f"{w}: dialogue wants 6-12 user/moshi turns")
                if not 80 <= nw <= 220:
                    problems.append(f"{w}: dialogue has {nw} words, want about 90-200")
                for k, tn in enumerate(turns):
                    if tn["speaker"] == "moshi" and " / " in tn["text"]:
                        problems.append(f"{w}[{k}]: ' / ' only in user turns")
                    if tn["text"].count(" / ") > 1:
                        problems.append(f"{w}[{k}]: at most one ' / ' per turn")
    return problems, stats


def main() -> None:
    wanted = sys.argv[1:]
    all_topics = {t: s for s in ("train", "eval") for t in TOPICS[s]}
    problems, total = [], 0
    for topic in wanted or all_topics:
        if topic not in all_topics:
            problems.append(f"{topic}: not in topics.json")
            continue
        pr, st = check_topic(topic, all_topics[topic])
        problems += pr
        total += st.get("words", 0)
        if st:
            print(f"{topic:18s} {all_topics[topic]:5s} " + " ".join(f"{k}={v}" for k, v in st.items()))
    for p in problems:
        print("PROBLEM", p)
    print(f"{len(problems)} problems, {total} words")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
