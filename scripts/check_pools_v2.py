"""Check the v2 content files in data/pools/v2/ (format, splits, banned words, duplicates).

  name_questions   {id, split, user, answer, name_pos}: a question/request the user addresses
                   to another person in the room by name ({name} exactly once, at the start,
                   middle or end of the line); `answer` is that person's short reply.
  moshi_mentions   {id, split, user, answer, name_pos}: the user mentions Moshi in the third
                   person to someone else ("I asked Moshi earlier..."); {name} optional
                   (name_pos none when absent); answer may be null.
  phrasings (phrasings_v2.json): lists of {id, split, text, style, tone[, states_return]}.

Usage: python scripts/check_pools_v2.py [FILE ...]   (default: every data/pools/v2/*.json)
"""

import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANNED_NQ = re.compile(r"\b(wait|hold on|hang on|back|where were we|one sec|a sec|a second|a minute|a moment|"
                       r"quiet|stay|i'm done|moshi)\b", re.I)
STYLES = {"request", "statement", "command", "fragment", "breakoff", "implied", "friend_first"}
TONES = {"polite", "curt", "apologetic", "distracted", "neutral", "cheerful"}
PHRASING_LISTS = {"think_instruction", "side_instruction", "describe_instruction", "friend_interrupt",
                  "side_cue", "describe_cue", "early_end"}
EXPLICIT_RETURN = re.compile(r"\b(back|done|finished|sorted|all yours|gone|left|over|that's all|that's it|"
                             r"that's everything|ready|got it|figured|decided|i know|okay, so|all set)\b", re.I)
IMPLICIT = re.compile(r"where were we|as i was saying|anyway,|so anyway|where was i", re.I)


def words(t: str) -> int:
    return len(t.replace("{name}", "X").split())


def name_pos(t: str) -> str:
    w = [x for x in t.replace("{name}", " {name} ").split() if re.search(r"[\w{}]", x)]  # punctuation-only tokens don't count
    i = next((k for k, x in enumerate(w) if "{name}" in x), None)
    if i is None:
        return "none"
    return "start" if i == 0 or (i == 1 and w[0].lower().strip(",") in {"hey", "oh", "so", "okay", "um", "hi"}) \
        else "end" if i == len(w) - 1 else "middle"


def v1_texts() -> set[str]:
    out = set()
    ph = json.loads((ROOT / "data/splits/phrasings.json").read_text())
    for k, v in ph.items():
        if k != "note":
            out |= {norm(x["text"]) for x in v}
    gen = json.loads((ROOT / "data/pools/generic.json").read_text())
    for x in gen["distractors"]:
        out.add(norm(x["user"].replace("{friend}", "{name}")))
    return out


def norm(t: str) -> str:
    return re.sub(r"[^a-z{}' ]", "", t.lower()).strip()


def check_file(f: str, old: set[str], seen_ids: set[str], seen_text: dict) -> list[str]:
    pr = []
    d = json.loads(Path(f).read_text())
    for kind, items in d.items():
        if kind == "note":
            continue
        if not isinstance(items, list):
            pr.append(f"{kind}: not a list")
            continue
        for it in items:
            iid = it.get("id")
            where = f"{Path(f).name}:{iid}"
            if not iid or iid in seen_ids:
                pr.append(f"{where}: missing or duplicate id")
            seen_ids.add(iid)
            if it.get("split") not in ("train", "eval"):
                pr.append(f"{where}: split must be train or eval")
            if kind in ("name_questions", "moshi_mentions"):
                u, a = it.get("user", ""), it.get("answer")
                if kind == "name_questions":
                    if u.count("{name}") != 1:
                        pr.append(f"{where}: user line needs {{name}} exactly once")
                    if BANNED_NQ.search(u) or (a and BANNED_NQ.search(a)):
                        pr.append(f"{where}: banned word ({BANNED_NQ.search(u + ' ' + (a or '')).group(0)!r})")
                    if not a:
                        pr.append(f"{where}: answer required")
                else:
                    if u.count("{name}") > 1:
                        pr.append(f"{where}: {{name}} at most once")
                    if "Moshi" not in u:
                        pr.append(f"{where}: must mention Moshi (third person)")
                    if re.match(r"^(hey |okay |so )?moshi\b", u, re.I) or re.search(r",\s*moshi[?.!]?$", u, re.I):
                        pr.append(f"{where}: addresses Moshi directly (must be third person)")
                    if re.search(r"\b(wait|hold on|hang on|where were we|be right back|i'm back)\b", u, re.I):
                        pr.append(f"{where}: wait-like or return-like wording")
                if it.get("name_pos") != name_pos(u):
                    pr.append(f"{where}: name_pos {it.get('name_pos')!r} but {{name}} is at the {name_pos(u)}")
                if not 3 <= words(u) <= 20:
                    pr.append(f"{where}: user line has {words(u)} words (3-20)")
                if a is not None and not 1 <= words(a) <= 14:
                    pr.append(f"{where}: answer has {words(a)} words (1-14)")
                key = norm(u)
                texts = [key] + ([norm(a)] if a else [])
            elif kind in PHRASING_LISTS:
                t = it.get("text", "")
                if it.get("style") not in STYLES:
                    pr.append(f"{where}: style {it.get('style')!r} not in {sorted(STYLES)}")
                if it.get("tone") not in TONES:
                    pr.append(f"{where}: tone {it.get('tone')!r} not in {sorted(TONES)}")
                if kind == "think_instruction" and t.count("{n}") != 1:
                    pr.append(f"{where}: needs {{n}} exactly once")
                if kind != "think_instruction" and "{n}" in t:
                    pr.append(f"{where}: {{n}} only in think_instruction")
                if "{name}" in t:
                    pr.append(f"{where}: use {{friend}}, not {{name}}")
                if kind.endswith("instruction") and not isinstance(it.get("states_return"), bool):
                    pr.append(f"{where}: states_return (true/false) required")
                if kind in ("side_cue", "describe_cue", "early_end"):
                    if IMPLICIT.search(t):
                        pr.append(f"{where}: implicit-return wording is eval-only material")
                    if not EXPLICIT_RETURN.search(t):
                        pr.append(f"{where}: return cue must say explicitly that the user is back/done")
                if kind == "friend_interrupt" and "moshi" in t.lower():
                    pr.append(f"{where}: the friend does not talk to Moshi")
                if not 1 <= words(t) <= 30:
                    pr.append(f"{where}: {words(t)} words (1-30)")
                key = norm(t)
                texts = [key]
            else:
                pr.append(f"{Path(f).name}: unknown list {kind!r}")
                break
            for k in texts:
                if k in old:
                    pr.append(f"{where}: same text as an existing (v1) item: {k!r}")
                if k in seen_text and seen_text[k] != iid:
                    pr.append(f"{where}: same text as {seen_text[k]}")
                seen_text.setdefault(k, iid)
        if kind in PHRASING_LISTS:
            c = Counter(x.get("split") for x in items)
            if items and abs(c["eval"] - len(items) / 3) > 1.5:
                pr.append(f"{kind}: {c['eval']} of {len(items)} eval items (want every third)")
    return pr


def main() -> None:
    files = sys.argv[1:] or [f for f in sorted(glob.glob(str(ROOT / "data/pools/v2/*.json"))) if not f.endswith("names.json")]
    old, ids, texts, problems = v1_texts(), set(), {}, []
    for f in files:
        p = check_file(f, old, ids, texts)
        d = json.loads(Path(f).read_text())
        print(f"{Path(f).name}: " + ", ".join(f"{k} {len(v)}" for k, v in d.items() if isinstance(v, list))
              + f"; {len(p)} problems")
        problems += p
    for x in problems:
        print("  PROBLEM", x)
    print(f"{len(problems)} problems")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
