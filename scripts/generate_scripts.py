"""Generate conversation scripts (text + planned timing, no audio) from the pools.

Combines building blocks from data/pools/ and phrasings from data/splits/ at random, drawing
only from the requested split, so eval voices, held-out phrasings, eval topics and held-out
distractors/friend names never appear in train scripts (scripts/check_split_leakage.py
verifies this on the output).

Script types (fractions per split; trained wait types = the three behaviours in CLAUDE.md):
  train: think_time 0.2, side_conversation 0.2, describe_options 0.2, ordinary 0.4
  eval:  think_time 0.14, side_conversation 0.14, describe_options 0.12,
         phone_call 0.1, door 0.1, reading 0.1 (eval-only wait types), ordinary 0.3
  think_time         context -> "give me N seconds to think, then ask me again" -> [ack] ->
                     silence -> Moshi asks again at the deadline -> [answer + Moshi reply].
  side_conversation  context -> "I'm talking to my friend, stay quiet until I'm back" -> [ack]
                     -> user and friend talk (with distractors) -> return cue -> Moshi
                     welcomes the user back, referring to the context -> [answer + reply].
  describe_options   lead-in -> "let me describe all the options before you comment" ->
                     [ack] -> options with pauses -> "that's all, what do you think?" -> Moshi.
  phone_call (eval)  context -> "I have to take this call" -> the user's side of a call (the
                     caller is not heard) -> "I'm off the phone" -> Moshi welcomes back.
  door (eval)        context -> "someone's at the door" -> user and visitor talk -> back.
  reading (eval)     context -> "let me read this first" -> silence with murmurs -> "done".
  ordinary           a user-Moshi dialogue (two if the first is short).
Conditions (exact fractions of the scripts of a type, at least one if the type occurs):
  think_time: self_talk 50% (the user mumbles or thinks aloud during the wait, not to Moshi),
              early_end 15% ("Okay, I've got it" before the deadline; Moshi answers at once),
              extension 40% (eval only, "give me five more seconds").
  side_conversation: near_miss 35% (inserted near-miss cue said to the friend),
              extra_question 30% (an extra question to the friend; the side conversations
              themselves already contain at least one).
  side/phone/door/reading, eval only: implicit 40% (return without "I'm back", e.g.
              "Anyway, where were we?").
  ordinary: wait_like 15% ("Wait, what did you say?", "Hold on, that's not right" or a
              "Hold on," prefix; not a wait instruction, Moshi answers normally).
Every script opens with Moshi greeting the user and asking how they are (CLAUDE.md), starting
at 0.3-0.8 s, then a short user answer, then the rest of the conversation.

Each script is one JSON file with: id, split, type, topic(s), voices (user, friend, moshi),
friend_name, phrasing_ids, block_ids, wait_sec / extension, and an ordered list of turns.
Each turn has speaker (user | friend | moshi), text, est_sec (estimated spoken length), and
either gap_before (silence after the previous turn ends) or anchor {turn, offset} (start
offset seconds after the end of turn number `turn`, used for deadlines). Optional: event
(greeting_answer | instruction | extension | distractor | return_cue | ordinary_request) with kind
(distractor: friend_question | question_to_other | near_miss | pause | self_talk; return_cue:
explicit | implicit | early_end; ordinary_request: wait_like), role for
Moshi turns (greeting | ack | resume | reply | followup), block/phrasing IDs.
In eval scripts every Moshi turn has reference_only: true. These are the ideal replies, used
for eval_full and loss evaluation only, never rendered into the model's input.

Planned length is 30-85 s (estimated), so rendered files stay under 90 s.

Usage:
  python scripts/generate_scripts.py --split train --n 20 --seed 1 --out data/scripts_pilot/train
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIX = {"train": {"think_time": 0.2, "side_conversation": 0.2, "describe_options": 0.2, "ordinary": 0.4},
       "eval": {"think_time": 0.14, "side_conversation": 0.14, "describe_options": 0.12,
                "phone_call": 0.1, "door": 0.1, "reading": 0.1, "ordinary": 0.3}}
MIN_SEC, MAX_SEC = 30.0, 85.0
ACK_P, ENDING_P = 0.5, 0.7
# Conditions, as exact fractions of the scripts of the type (at least one if the type occurs)
SELF_TALK_P = 0.5        # think_time: the user mumbles / thinks aloud during the wait
EARLY_END_P = 0.15       # think_time: "Okay, I've got it" before the deadline, Moshi replies at once
EXTENSION_P = 0.4        # think_time, eval only: "give me five more seconds"
NEAR_MISS_P = 0.35       # side_conversation: an inserted near-miss return cue said to the friend
EXTRA_QUESTION_P = 0.3   # side_conversation: an extra inserted question to the friend
IMPLICIT_P = 0.4         # eval only (side/phone/door/reading): "Anyway, where were we?"
WAIT_LIKE_P = 0.15       # ordinary: "Wait, what did you say?" etc., not a wait instruction
TAIL_SEC = 1.5  # same as render_lib: silence after the last speech
NUM_WORDS = {5: "five", 10: "ten", 15: "fifteen", 20: "twenty", 30: "thirty", 45: "forty-five", 60: "sixty"}


def est_sec(text: str) -> float:
    """Spoken length estimate for Fish s2-pro: ~0.31 s per word (fit on the Moshi clone
    lines rendered so far: 3-13 words took 0.7-4.2 s)."""
    return round(0.1 + 0.31 * len(text.split()), 2)


def gap(rng: random.Random) -> float:
    """Planned silence between turns: mostly 0.3-1.5 s, sometimes a longer pause."""
    return round(rng.uniform(0.3, 1.5) if rng.random() < 0.85 else rng.uniform(1.5, 3.5), 2)


def moshi_gap(rng: random.Random) -> float:
    return round(rng.uniform(0.3, 1.2) if rng.random() < 0.9 else rng.uniform(1.2, 2.0), 2)


class Deck:
    """Draw items without replacement, reshuffling when exhausted, so reuse is spread
    evenly over a pool (no item is used twice before every item was used once)."""

    def __init__(self, items: list, rng: random.Random):
        self.items, self.rng, self.queue = list(items), rng, []
        assert self.items, "empty pool"

    def draw(self, avoid=lambda x: False):
        for _ in range(2 * len(self.items) + 1):
            if not self.queue:
                self.queue = self.rng.sample(self.items, len(self.items))
            x = self.queue.pop()
            if not avoid(x):
                return x
            self.queue.insert(0, x)
        return x


class Pools:
    def __init__(self, split: str, rng: random.Random):
        self.split, self.rng = split, rng
        topics = json.loads((ROOT / "data/splits/topics.json").read_text())[split]
        blocks = {k: [] for k in ("contexts", "side_conversations", "ordinary_dialogues", "option_sets")}
        for t in topics:
            d = json.loads((ROOT / f"data/pools/topics/{t}.json").read_text())
            for k in blocks:
                blocks[k] += [{**it, "topic": t} for it in d.get(k, [])]
        self.decks = {k: Deck(v, rng) for k, v in blocks.items() if v}
        ok = lambda it: it["split"] in (split, "both")  # noqa: E731
        ph = json.loads((ROOT / "data/splits/phrasings.json").read_text())
        for k, v in ph.items():
            if k != "note" and any(ok(x) for x in v):
                self.decks[k] = Deck([x for x in v if ok(x)], rng)
        gen = json.loads((ROOT / "data/pools/generic.json").read_text())
        for k in ("acks", "friend_names"):
            self.decks[k] = Deck([x for x in gen[k] if ok(x)], rng)
        for kind in ("friend_question", "near_miss"):
            self.decks[kind] = Deck([x for x in gen["distractors"] if x["kind"] == kind and ok(x)], rng)
        for k in ("self_talk", "wait_like", "phone_calls", "door_visits", "reading_murmurs"):
            items = [x for x in gen[k] if ok(x)]
            if items:
                self.decks[k] = Deck(items, rng)
        self.wait_like_moshi = gen["wait_like_moshi"]
        v = json.loads((ROOT / "data/splits/voices.json").read_text())
        self.voices, self.moshi_voice = v[split], v["moshi"]
        self.think_seconds = json.loads((ROOT / "data/splits/waits.json").read_text())["think_seconds"][split]
        self.extension_seconds = json.loads((ROOT / "data/splits/waits.json").read_text())["extension_seconds"]

    def draw(self, k: str, **kw):
        return self.decks[k].draw(**kw)


class Builder:
    """Accumulates the turns of one script."""

    def __init__(self, split: str, rng: random.Random, friend: str):
        self.split, self.rng, self.friend = split, rng, friend
        self.turns: list[dict] = []
        self.phrasing_ids: list[str] = []
        self.block_ids: list[str] = []

    def add(self, speaker: str, text: str, gap_before: float | None = None, **extra) -> int:
        text = text.replace("{friend}", self.friend)
        t = {"speaker": speaker, "text": text, "est_sec": est_sec(text)}
        if "anchor" not in extra:
            t["gap_before"] = round(gap_before if gap_before is not None else
                                    (moshi_gap(self.rng) if speaker == "moshi" else gap(self.rng)), 2)
        t.update({k: v for k, v in extra.items() if v is not None})
        if speaker == "moshi" and self.split == "eval":
            t["reference_only"] = True
        self.turns.append(t)
        return len(self.turns) - 1

    def dialogue(self, turns: list[dict], block: str, first_gap: float | None = None) -> None:
        """Add user/moshi turns, splitting ' / ' hesitations into a pause distractor and
        tagging user turns that Moshi answers as ordinary requests."""
        self.block_ids.append(block)
        for k, tn in enumerate(turns):
            g = first_gap if k == 0 else None
            if tn["speaker"] == "moshi":
                self.add("moshi", tn["text"], g, role="reply", block=block)
                continue
            parts = [p.strip() for p in tn["text"].split(" / ")]
            if len(parts) == 2:
                self.add("user", parts[0], g, event="distractor", kind="pause", block=block)
                g = round(self.rng.uniform(1.5, 3.0), 2)
            answered = k + 1 < len(turns) and turns[k + 1]["speaker"] == "moshi"
            self.add("user", parts[-1], g, event="ordinary_request" if answered else None,
                     kind=tn.get("kind"), phrasing=tn.get("phrasing"), block=block)

    def timeline(self) -> tuple[float, float]:
        """(estimated total length, estimated speech) following gaps and anchors."""
        ends, cur, speech = [], 0.0, 0.0
        for t in self.turns:
            start = ends[t["anchor"]["turn"]] + t["anchor"]["offset"] if "anchor" in t else cur + t["gap_before"]
            ends.append(start + t["est_sec"])
            cur = max(cur, ends[-1])
            speech += t["est_sec"]
        return round(cur + TAIL_SEC, 2), round(speech, 2)


def num(n: int) -> str:
    return NUM_WORDS[n]


NEAR_MISS_RX = re.compile(r"\b(back|where were we)\b", re.I)


def build(kind: str, split: str, P: Pools, rng: random.Random, cond: set[str]) -> dict:
    """Build one script of the given type. `cond` holds the conditions assigned to it:
    self_talk, early_end, extension (think_time); near_miss, extra_question (side);
    implicit (side/phone/door/reading, eval); wait_like (ordinary)."""
    user_v = rng.choice(P.voices)
    friend_v = rng.choice([v for v in P.voices if v["id"] != user_v["id"]])
    friend = P.draw("friend_names")
    b = Builder(split, rng, friend["text"])
    meta: dict = {"type": kind, "wait_type": None if kind == "ordinary" else kind,
                  "conditions": sorted(cond)}
    uses_friend = kind in ("side_conversation", "door")

    def phr(deck: str) -> dict:
        x = P.draw(deck)
        b.phrasing_ids.append(x["id"])
        return x

    # Opening (always): Moshi greets and asks how the user is, the user answers, and then
    # goes straight on with the rest of the conversation.
    g = phr("greeting")
    b.add("moshi", g["text"], round(rng.uniform(0.3, 0.8), 2), role="greeting", phrasing=g["id"])
    a = phr("greeting_answer")
    b.add("user", a["text"], round(rng.uniform(0.3, 1.0), 2), event="greeting_answer", phrasing=a["id"])
    first_gap = round(rng.uniform(0.2, 0.6), 2)

    def ack() -> None:
        if rng.random() < ACK_P:
            x = phr("acks")
            b.add("moshi", x["text"], round(rng.uniform(0.2, 0.7), 2), role="ack", phrasing=x["id"])

    def ending(ctx: dict, force: bool = False, moshi_gap_=None) -> None:
        if force or rng.random() < ENDING_P:
            e = rng.choice(ctx["endings"])
            b.add("user", e["user"], round(rng.uniform(0.3, 0.6), 2) if force else None,
                  event="ordinary_request", block=ctx["id"])
            b.add("moshi", e["moshi"], moshi_gap_, role="followup", block=ctx["id"])

    def cue(deck: str) -> None:
        """The return signal: an explicit cue, or (eval condition) an implicit return."""
        if "implicit" in cond:
            x = phr("implicit_return")
            b.add("user", x["text"], round(rng.uniform(0.5, 2.0), 2), event="return_cue", kind="implicit", phrasing=x["id"])
        else:
            x = phr(deck)
            b.add("user", x["text"], round(rng.uniform(0.5, 2.0), 2), event="return_cue", kind="explicit", phrasing=x["id"])

    if kind == "ordinary":
        dlg = P.draw("ordinary_dialogues")
        turns = [dict(t) for t in dlg["turns"]]
        if "wait_like" in cond:
            turns = add_wait_like(turns, P, rng, b)
        b.dialogue(turns, dlg["id"], first_gap)
        meta["topics"] = [dlg["topic"]]
        if b.timeline()[0] < MIN_SEC:
            d2 = P.draw("ordinary_dialogues", avoid=lambda x: x["id"] == dlg["id"])
            b.dialogue(d2["turns"], d2["id"], round(rng.uniform(1.0, 2.5), 2))
            meta["topics"].append(d2["topic"])
        while b.timeline()[0] > MAX_SEC and len(b.turns) > 6:  # cut at a Moshi turn
            b.turns.pop()
            while b.turns and b.turns[-1]["speaker"] != "moshi":
                b.turns.pop()
        if "wait_like" in cond and not any(t.get("kind") == "wait_like" for t in b.turns):
            raise ValueError("wait-like turn was cut")
        if b.turns[-1]["speaker"] == "moshi":
            last_user = max(i for i, t in enumerate(b.turns) if t["speaker"] == "user")
            b.turns[last_user].setdefault("event", "ordinary_request")

    elif kind == "describe_options":
        opt = P.draw("option_sets")
        b.block_ids.append(opt["id"])
        meta["topics"] = [opt["topic"]]
        b.add("user", opt["lead_in"], first_gap, block=opt["id"])
        x = phr("describe_instruction")
        b.add("user", x["text"], round(rng.uniform(0.4, 0.9), 2), event="instruction",
              kind="describe_options", phrasing=x["id"])
        ack()
        for k, o in enumerate(opt["options"]):
            gp = round(rng.uniform(1.0, 3.0), 2) if k else round(rng.uniform(0.6, 1.5), 2)
            if k and gp > 1.5:  # a long pause after the previous option: a pause distractor
                b.turns[-1].update(event="distractor", kind="pause")
            b.add("user", o, gp, block=opt["id"])
        x = phr("describe_cue")
        b.add("user", x["text"], round(rng.uniform(0.5, 1.5), 2), event="return_cue", kind="explicit", phrasing=x["id"])
        b.add("moshi", rng.choice(opt["moshi_comment"]), role="resume", block=opt["id"])

    else:  # waits that follow a context exchange
        ctx = P.draw("contexts")
        b.dialogue(ctx["turns"], ctx["id"], first_gap)
        meta["topics"] = [ctx["topic"]]

        if kind == "think_time":
            secs = [s for s in P.think_seconds if s >= 10] if cond & {"self_talk", "early_end"} else P.think_seconds
            n = rng.choice(secs or P.think_seconds)
            x = phr("think_instruction")
            instr = b.add("user", x["text"].replace("{n}", num(n)), event="instruction",
                          kind="think_time", wait_sec=n, phrasing=x["id"])
            ack()
            meta["wait_sec"] = n
            # times (s after the instruction ends) that the self-talk / extension / early end use
            stop = n - 1.0
            if "early_end" in cond:
                early_at = round(rng.uniform(max(4.0, 0.4 * n), n - 1.5), 2)
                stop = early_at - 1.0
            lead = round(rng.uniform(1.0, 3.0), 2) if "extension" in cond else None
            if lead:
                stop = n - lead - 1.0
            if "self_talk" in cond:  # the user mumbles or thinks aloud, not to Moshi
                t0 = 3.0
                for _ in range(rng.choice([1, 2])):
                    m = P.draw("self_talk")
                    if t0 + est_sec(m["text"]) > stop:
                        break
                    off = round(rng.uniform(t0, max(t0, stop - est_sec(m["text"]))), 2)
                    if len([t for t in b.turns if t.get("kind") == "self_talk"]):
                        off = max(off, t0)
                    b.phrasing_ids.append(m["id"])
                    b.add("user", m["text"], anchor={"turn": instr, "offset": off},
                          event="distractor", kind="self_talk", phrasing=m["id"])
                    t0 = off + est_sec(m["text"]) + 1.5
                if not any(t.get("kind") == "self_talk" for t in b.turns):
                    raise ValueError("no room for self-talk")
            if "early_end" in cond:
                x = phr("early_end")
                b.add("user", x["text"], anchor={"turn": instr, "offset": early_at},
                      event="return_cue", kind="early_end", phrasing=x["id"])
                meta["early_end"] = {"after_sec": early_at}
                ending(ctx, force=True, moshi_gap_=round(rng.uniform(0.3, 0.8), 2))  # Moshi answers right away
            else:
                anchor, wait = instr, n
                if lead:
                    m_ = rng.choice(P.extension_seconds)
                    x = phr("extension")
                    anchor = b.add("user", x["text"].replace("{n}", num(m_)), event="extension", extra_sec=m_,
                                   phrasing=x["id"], anchor={"turn": instr, "offset": round(n - lead, 2)})
                    wait = m_
                    meta["extension"] = {"extra_sec": m_, "said_before_deadline_sec": lead}
                b.add("moshi", rng.choice(ctx["think_reask"]), role="resume", block=ctx["id"],
                      anchor={"turn": anchor, "offset": round(wait + rng.uniform(0.0, 0.6), 2)})
                meta["deadline"] = {"after_turn": anchor, "sec": wait}
                ending(ctx)

        elif kind == "side_conversation":
            x = phr("side_instruction")
            b.add("user", x["text"], event="instruction", kind="side_conversation", phrasing=x["id"])
            ack()
            side = P.draw("side_conversations")
            b.block_ids.append(side["id"])
            meta["topics"].append(side["topic"])
            turns = [dict(t, block=side["id"]) for t in side["turns"]]
            inserts = (["near_miss"] if "near_miss" in cond else []) + \
                      (["friend_question"] if "extra_question" in cond else [])
            for dk in inserts:
                d = P.draw(dk)
                b.phrasing_ids.append(d["id"])
                # only at the start or the end, so an inserted exchange never splits a
                # question from its answer inside the side conversation
                pos = rng.choice([0, len(turns)])
                turns[pos:pos] = [{"speaker": "user", "text": d["user"], "event": "distractor",
                                   "kind": d["kind"], "phrasing": d["id"]},
                                  {"speaker": "friend", "text": d["friend"], "phrasing": d["id"]}]
            for k, t in enumerate(turns):
                if t["speaker"] == "user" and "event" not in t and t["text"].rstrip().endswith("?"):
                    t.update(event="distractor", kind="friend_question")
                b.add(t["speaker"], t["text"], round(rng.uniform(0.6, 1.5), 2) if k == 0 else None,
                      **{x_: t[x_] for x_ in ("event", "kind", "block", "phrasing") if x_ in t})
            cue("side_cue")
            b.add("moshi", rng.choice(ctx["welcome_back"]), role="resume", block=ctx["id"])
            ending(ctx)

        elif kind == "phone_call":  # eval only: one side of a call; the caller is not heard
            x = phr("phone_instruction")
            b.add("user", x["text"], event="instruction", kind="phone_call", phrasing=x["id"])
            ack()
            call = phr("phone_calls")
            for k, line in enumerate(call["lines"]):
                gp = round(rng.uniform(1.0, 2.0), 2) if k == 0 else round(rng.uniform(1.5, 4.0), 2)
                ev = {}
                if NEAR_MISS_RX.search(line):
                    ev = {"event": "distractor", "kind": "near_miss"}
                elif line.rstrip().endswith("?"):
                    ev = {"event": "distractor", "kind": "question_to_other"}
                b.add("user", line, gp, phrasing=call["id"], **ev)
            cue("phone_cue")
            b.add("moshi", rng.choice(ctx["welcome_back"]), role="resume", block=ctx["id"])
            ending(ctx)

        elif kind == "door":  # eval only: the user answers the door; the visitor is audible
            x = phr("door_instruction")
            b.add("user", x["text"], event="instruction", kind="door", phrasing=x["id"])
            ack()
            visit = phr("door_visits")
            for k, t in enumerate(visit["turns"]):
                gp = round(rng.uniform(2.0, 5.0), 2) if k == 0 else None  # walking to the door
                ev = {}
                if NEAR_MISS_RX.search(t["text"]):
                    ev = {"event": "distractor", "kind": "near_miss"}
                elif t["speaker"] == "user" and t["text"].rstrip().endswith("?"):
                    ev = {"event": "distractor", "kind": "question_to_other"}
                b.add("visitor" if t["speaker"] == "visitor" else "user", t["text"], gp, phrasing=visit["id"], **ev)
            b.turns[-1]["after_gap_hint"] = "walking back"
            cue("door_cue")
            b.turns[-1]["gap_before"] = round(rng.uniform(2.0, 4.0), 2)
            b.add("moshi", rng.choice(ctx["welcome_back"]), role="resume", block=ctx["id"])
            ending(ctx)

        elif kind == "reading":  # eval only: the user reads silently, with occasional murmurs
            x = phr("reading_instruction")
            instr = b.add("user", x["text"], event="instruction", kind="reading", phrasing=x["id"])
            ack()
            read_sec = round(rng.uniform(8.0, 25.0), 1)
            t0 = 3.0
            for _ in range(rng.choice([0, 1, 2])):
                m = phr("reading_murmurs")
                if t0 + est_sec(m["text"]) > read_sec - 1.0:
                    break
                off = round(rng.uniform(t0, read_sec - 1.0 - est_sec(m["text"])), 2)
                b.add("user", m["text"], anchor={"turn": instr, "offset": off},
                      event="distractor", kind="self_talk", phrasing=m["id"])
                t0 = off + est_sec(m["text"]) + 1.5
            meta["reading_sec"] = read_sec
            if "implicit" in cond:
                y = phr("implicit_return")
                b.add("user", y["text"], anchor={"turn": instr, "offset": read_sec},
                      event="return_cue", kind="implicit", phrasing=y["id"])
            else:
                y = phr("reading_cue")
                b.add("user", y["text"], anchor={"turn": instr, "offset": read_sec},
                      event="return_cue", kind="explicit", phrasing=y["id"])
            b.add("moshi", rng.choice(ctx["welcome_back"]), role="resume", block=ctx["id"])
            ending(ctx)

        # Fit the length by dropping the optional ending; if still too long (or too short),
        # the caller retries with other blocks. (Dropping earlier turns would shift the turn
        # numbers that anchors refer to.)
        while (b.timeline()[0] > MAX_SEC and b.turns[-1].get("role") == "followup"
               and "early_end" not in cond):
            b.turns = b.turns[:-2]

    for t in b.turns:
        t.pop("after_gap_hint", None)
    total, speech = b.timeline()
    if not MIN_SEC <= total <= MAX_SEC:
        raise ValueError(f"length {total}")
    return {
        **meta, "topic": meta["topics"][0],
        "voices": {"user": user_v["id"], "friend": friend_v["id"] if uses_friend else None,
                   "moshi": P.moshi_voice},
        "friend_name": friend["text"] if kind == "side_conversation" else None,
        "friend_name_id": friend["id"] if kind == "side_conversation" else None,
        "phrasing_ids": b.phrasing_ids, "block_ids": b.block_ids,
        "est_total_sec": total, "est_speech_sec": speech, "turns": b.turns,
    }


def add_wait_like(turns: list[dict], P: Pools, rng: random.Random, b: Builder) -> list[dict]:
    """Insert a 'wait'-like phrase that is NOT a wait instruction; Moshi answers normally.
    repeat/check: a new user turn right after a Moshi turn; Moshi repeats that turn (repeat)
    or stands by it (check).
    prefix: 'Hold on,' etc. in front of the user's next follow-up."""
    spots = [k for k in range(2, len(turns)) if turns[k]["speaker"] == "user" and turns[k - 1]["speaker"] == "moshi"]
    if not spots:
        raise ValueError("no place for a wait-like phrase")
    k = rng.choice(spots)
    w = P.draw("wait_like")
    b.phrasing_ids.append(w["id"])
    if w["mode"] == "prefix":
        t = turns[k]["text"]
        t = t if t.startswith("I ") or t.startswith("I'") else t[0].lower() + t[1:]
        turns[k] = dict(turns[k], text=f"{w['text']} {t}", kind="wait_like", phrasing=w["id"])
        return turns
    lead = rng.choice(P.wait_like_moshi[w["mode"]])
    # repeat: Moshi repeats its previous turn; check: Moshi stands by it in a line of its own
    reply = f"{lead} {turns[k - 1]['text'].replace(' / ', ' ')}" if w["mode"] == "repeat" else lead
    return turns[:k] + [{"speaker": "user", "text": w["text"], "kind": "wait_like", "phrasing": w["id"]},
                        {"speaker": "moshi", "text": reply}] + turns[k:]


def assign_conditions(kinds: list[str], split: str, rng: random.Random) -> list[set[str]]:
    """Exact-count condition assignment (at least one where the type is present), so even a
    small pilot contains every case."""
    conds: list[set[str]] = [set() for _ in kinds]

    def pick(idx: list[int], frac: float, name: str, of: int | None = None) -> list[int]:
        """Give `name` to round(frac * of) of the scripts in idx (of defaults to len(idx))."""
        k = min(len(idx), max(1, round(frac * (of or len(idx))))) if idx else 0
        chosen = rng.sample(idx, k)
        for i in chosen:
            conds[i].add(name)
        return chosen

    think = [i for i, k in enumerate(kinds) if k == "think_time"]
    if split == "eval":
        ext = pick(think, EXTENSION_P, "extension")
        think_rest = [i for i in think if i not in ext]
    else:
        think_rest = think
    if think_rest:  # 15% of all think-time scripts, among those without an extension
        pick(think_rest, EARLY_END_P, "early_end", of=len(think))
    pick(think, SELF_TALK_P, "self_talk")
    side = [i for i, k in enumerate(kinds) if k == "side_conversation"]
    pick(side, NEAR_MISS_P, "near_miss")
    pick(side, EXTRA_QUESTION_P, "extra_question")
    if split == "eval":
        for t in ("side_conversation", "phone_call", "door", "reading"):  # 40% within each type
            pick([i for i, k in enumerate(kinds) if k == t], IMPLICIT_P, "implicit")
    pick([i for i, k in enumerate(kinds) if k == "ordinary"], WAIT_LIKE_P, "wait_like")
    return conds


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["train", "eval"], required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    rng = random.Random(f"{args.split}-{args.seed}")
    P = Pools(args.split, rng)
    mix = MIX[args.split]
    counts = {k: round(f * args.n) for k, f in mix.items()}
    counts["ordinary"] += args.n - sum(counts.values())
    kinds = [k for k, c in counts.items() for _ in range(c)]
    rng.shuffle(kinds)
    conds = assign_conditions(kinds, args.split, rng)

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, kind in enumerate(kinds):
        for attempt in range(100):
            try:
                s = build(kind, args.split, P, rng, conds[i])
                break
            except ValueError:
                continue
        else:
            raise SystemExit(f"could not build a {kind} script {sorted(conds[i])} within {MIN_SEC}-{MAX_SEC} s")
        sid = f"{args.split}_{args.seed:02d}_{i:05d}"
        script = {"id": sid, "split": args.split, "seed": args.seed, **s}
        if args.split == "eval":
            script["note"] = ("Moshi turns are reference_only: the ideal replies, for eval_full and "
                              "loss evaluation. Never render them into the model's input.")
        (args.out / f"{sid}.json").write_text(json.dumps(script, indent=1, ensure_ascii=False) + "\n")
        rows.append({"id": sid, "type": kind, "conditions": s["conditions"], "topic": s["topic"],
                     "est_total_sec": s["est_total_sec"], "est_speech_sec": s["est_speech_sec"],
                     "wait_sec": s.get("wait_sec")})
    (args.out / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    tot = sum(r["est_total_sec"] for r in rows)
    sp = sum(r["est_speech_sec"] for r in rows)
    print(f"{len(rows)} {args.split} scripts -> {args.out}: {dict(Counter(r['type'] for r in rows))}")
    print(f"  estimated {tot / 60:.1f} min total, {sp / 60:.1f} min speech; "
          f"length {min(r['est_total_sec'] for r in rows):.0f}-{max(r['est_total_sec'] for r in rows):.0f} s")


if __name__ == "__main__":
    main()
