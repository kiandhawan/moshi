"""Generate the v2 conversation scripts (train / control) and eval v2 (see results/LOG.md).

Same design as scripts/generate_scripts.py (v1, unchanged; its pools, Deck, conditions,
content-unique decks and ordered blocks are reused): content-unique sets, types in blocks of
20 with the exact mix, prefix-stable ordered lists, fixed seeds. v2 changes for training:

  think time      any whole number of seconds 5-60 except 15 and 30 (eval only); a "long"
                  stride (LONG_P = 0.4 of the type) draws 31-60 s, the rest 5-29 s.
  describe        the long stride uses 3.5-8 s pauses between options and needs >= 30 s of
                  wait (estimated); side conversations are long by themselves (v1: 74% >= 30 s).
  position        early / middle / late, cycling per wait type: early = the wait comes right
                  after the context and an ordinary dialogue follows the resume; middle = the
                  dialogue's first exchange before the context, its next one after the wait;
                  late = a whole dialogue before the context and only the resume after it.
                  Think waits >= LEAN_SEC (46 s) may drop the dialogue to fit (else 49-60 s
                  waits would never fit under 85 s). The realized position (share of
                  conversation speech before the instruction) is written as instruction_position.
  instructions    v1 train phrasings + data/pools/v2/phrasings_v2.json (train items) with styles
                  request / statement / command / fragment / breakoff / implied / friend_first,
                  tones, with or without saying how the user will return. Side conversations:
                  FRIEND_FIRST_P start with the friend interrupting ("Hey, can I ask you
                  something?") and the user's friend_first instruction ("Hold on, Moshi.").
                  Return cues (side_cue, describe_cue, early_end) the same, explicit only.
  other_name      the user asks another person in the room something by name (v2
                  name_questions, a name from data/pools/v2/names.json, train split); the person
                  answers in ANSWER_P of them; Moshi stays silent, then the conversation goes on.
                  ordinary 15%, wait conversations 10% (outside the wait). Distractor kind
                  other_name; the person is speaker "other" (a third voice).
  moshi_name      contrast: a user request to Moshi addressed by name ("Moshi, ...", "...,
                  Moshi?", or before the last sentence); Moshi answers. ordinary 10%, waits 5%.
  moshi_mention   contrast: the user talks about Moshi in the third person to someone else
                  ("I asked Moshi earlier..."); Moshi stays silent (distractor kind
                  moshi_mention). ordinary 6%, waits 4%.
Voices: a file's user voice is the v1 voice of the file that used its first block (and a side
conversation keeps its v1 friend voice and name), so unchanged lines come from the TTS cache.

Eval v2 (--eval-v2): the 400 v1 eval scripts minus 40 (stratified by type), plus 40 new
scripts of the same types and conditions, built by the v1 eval generator from blocks no other
eval file uses, each with one other_name exchange (eval names, eval name_questions) before
the instruction (wait types) or between turns (ordinary); groups += other_name.

Usage:
  python scripts/generate_scripts_v2.py --split train --n 720 --seed 10 --out data/v2/scripts/train
  python scripts/generate_scripts_v2.py --split train --only ordinary --n 500 --seed 11 \
      --id-prefix control_11 --exclude-ordinary-from data/v2/scripts/train --out data/v2/scripts/control
  python scripts/generate_scripts_v2.py --eval-v2 --v1-eval data/scripts/eval --out data/v2/scripts/eval
"""

import argparse
import glob
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path

import generate_scripts as g1
from generate_scripts import (BLOCK, MIX, PHASE, Deck, PoolExhausted, Pools, add_wait_like, est_sec, gap,
                              moshi_gap, resume_line, stride_hit)

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "data/pools/v2"
MIN_SEC, MAX_SEC = 30.0, 85.0
ACK_P, ENDING_P, TAIL_SEC = 0.5, 0.7, 1.5
LONG_P = 0.4            # think_time / describe_options: wait >= 30 s (planned; early ends keep >= 30 s)
LONG_SEC = 30.0
LEAN_SEC = 46            # think waits this long may lose the surrounding dialogue to fit (see build_v2)
FRIEND_FIRST_P = 0.25   # side_conversation: the friend interrupts first
OTHER_NAME_P = {"ordinary": 0.15, "wait": 0.10}
MOSHI_NAME_P = {"ordinary": 0.10, "wait": 0.05}
MENTION_P = {"ordinary": 0.06, "wait": 0.04}
ANSWER_P = 0.6          # the named person answers
POSITIONS = ["early", "middle", "late"]
PROTECTED = {"other_name", "moshi_name", "moshi_mention"}
PHASE2 = {**PHASE, "long": 0.3, "friend_first": 0.6, "other_name": 0.4, "moshi_name": 0.7, "moshi_mention": 0.2}
TRANSITIONS = ["Oh, also, ", "Different question. ", "Okay, something else. ", "Can I ask about something else? ",
               "Changing the subject, ", "Oh, and another thing. ", "Right, next thing. ", "Unrelated, but ",
               "", "", ""]
EVAL_V2_REPLACE = {"think_time": 6, "side_conversation": 6, "describe_options": 5, "phone_call": 4, "door": 4,
                   "reading": 3, "ordinary": 12}


def num_words(n: int) -> str:
    ones = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
            "sixteen seventeen eighteen nineteen").split()
    if n < 20:
        return ones[n]
    tens = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty"}[n // 10]
    return tens + (f"-{ones[n % 10]}" if n % 10 else "")


def auto_style(text: str) -> tuple[str, bool]:
    """Style and states_return for a v1 phrasing (v1 items carry no tags)."""
    t = text.lower()
    style = "request" if "?" in t or t.startswith(("can ", "could ", "would ", "mind ", "do you mind")) else \
        "statement" if t.startswith(("i ", "i'm", "i've", "i'll", "my ", "there's", "that's", "okay, i", "ok, i")) \
        else "command"
    ret = bool(re.search(r"ask me again|until i|till i|when i|i'll (tell|say|let)|then ", t))
    return style, ret


def other_name_exchange(nq: dict, name: str, answered: bool, kind: str = "other_name") -> list[dict]:
    q = {"speaker": "user", "text": nq["user"].replace("{name}", name), "event": "distractor", "kind": kind,
         "phrasing": nq["id"], "block": nq["id"], "name_pos": nq["name_pos"], "addressee": name}
    out = [q]
    if answered and nq.get("answer"):
        out.append({"speaker": "other", "text": nq["answer"], "block": nq["id"]})
    return out


def add_moshi_name(text: str, rng: random.Random) -> tuple[str, str]:
    """Address a user turn to Moshi by name: start, end, or before the last sentence (middle)."""
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    opts = ["start", "end"] + (["middle"] if len(sents) >= 2 else [])
    pos = rng.choice(opts)

    def lower1(t: str) -> str:
        return t if t.startswith(("I ", "I'")) else t[0].lower() + t[1:]
    if pos == "start":
        return f"{rng.choice(['Moshi, ', 'Hey Moshi, ', 'Okay Moshi, ', 'So Moshi, '])}{lower1(text)}", pos
    if pos == "middle":
        return " ".join(sents[:-1] + [f"Moshi, {lower1(sents[-1])}"]), pos
    m = re.match(r"^(.*?)([.!?]*)$", text.strip())
    body, punct = m.group(1), m.group(2) or "."
    return f"{body}, Moshi{punct[-1]}", pos


class PoolsV2(Pools):
    """v1 pools + v2 phrasings, name questions, Moshi mentions and person names."""

    def __init__(self, split: str, rng: random.Random, unique: bool = True, exclude: set[str] | None = None):
        super().__init__(split, rng, unique=unique, exclude=exclude)
        exclude = exclude or set()
        ok = lambda it: it["split"] in (split, "both")  # noqa: E731
        ph1 = json.loads((ROOT / "data/splits/phrasings.json").read_text())
        ph2 = json.loads((V2 / "phrasings_v2.json").read_text())
        self.style = {}
        for k in ("think_instruction", "side_instruction", "describe_instruction", "side_cue", "describe_cue",
                  "early_end", "friend_interrupt"):
            items = []
            for x in ph1.get(k, []):
                if ok(x):
                    st, ret = auto_style(x["text"])
                    items.append({**x, "style": st, "tone": "neutral", "states_return": ret, "source": "v1"})
            items += [{**x, "source": "v2"} for x in ph2.get(k, []) if ok(x)]
            for x in items:
                self.style[x["id"]] = x
            if k == "side_instruction":
                self.decks["side_instruction"] = Deck([x for x in items if x["style"] != "friend_first"], rng)
                self.decks["side_instruction_ff"] = Deck([x for x in items if x["style"] == "friend_first"], rng)
            else:
                self.decks[k] = Deck(items, rng)
        nq = [x for f in sorted(glob.glob(str(V2 / "name_questions_*.json")))
              for x in json.loads(Path(f).read_text())["name_questions"] if ok(x)]
        mm = [x for x in json.loads((V2 / "moshi_mentions.json").read_text())["moshi_mentions"] if ok(x)]
        self.decks["name_questions"] = Deck([x for x in nq if x["id"] not in exclude], rng, "name_questions", unique)
        if mm:
            self.decks["moshi_mentions"] = Deck([x for x in mm if x["id"] not in exclude], rng, "moshi_mentions", unique)
        names = json.loads((V2 / "names.json").read_text())["person_names"]
        self.decks["person_names"] = Deck([x for x in names if ok(x)], rng)
        if split == "train":
            w = [s for s in range(5, 61) if s not in (15, 30)]
            self.think_short, self.think_long = [s for s in w if s < LONG_SEC], [s for s in w if s > LONG_SEC]


class B2:
    """Turns with uid anchors ({"uid", "offset"}), so turns can be removed without breaking anchors."""

    def __init__(self, rng: random.Random, friend: str):
        self.rng, self.friend = rng, friend
        self.turns: list[dict] = []
        self.phrasing_ids: list[str] = []
        self.block_ids: list[str] = []
        self._n = 0

    def add(self, speaker: str, text: str, gap_before: float | None = None, seg: str = "main", **extra) -> int:
        text = text.replace("{friend}", self.friend)
        self._n += 1
        t = {"_uid": self._n, "_seg": seg, "speaker": speaker, "text": text, "est_sec": est_sec(text)}
        if "anchor" not in extra:
            t["gap_before"] = round(gap_before if gap_before is not None else
                                    (moshi_gap(self.rng) if speaker == "moshi" else gap(self.rng)), 2)
        t.update({k: v for k, v in extra.items() if v is not None})
        self.turns.append(t)
        return self._n

    def dialogue(self, turns: list[dict], block: str | None, first_gap: float | None = None, seg: str = "main") -> None:
        """User/Moshi(/other) turns; ' / ' hesitations become a pause distractor; user turns
        that Moshi answers are ordinary requests; inserted exchanges keep their events."""
        if block:
            self.block_ids.append(block)
        for k, tn in enumerate(turns):
            g = first_gap if k == 0 else tn.get("gap")
            keep = {x: tn[x] for x in ("event", "kind", "phrasing", "name_pos", "addressee") if x in tn}
            blk = tn.get("block", block)
            if tn["speaker"] == "moshi":
                self.add("moshi", tn["text"], g, seg=seg, role="reply", block=blk)
                continue
            if tn["speaker"] == "other":
                self.add("other", tn["text"], g if g is not None else round(self.rng.uniform(0.3, 1.0), 2),
                         seg=seg, block=blk)
                continue
            if "event" in keep and keep["event"] == "distractor":  # inserted exchange
                self.add("user", tn["text"], g, seg=seg, block=blk, **keep)
                continue
            parts = [p.strip() for p in tn["text"].split(" / ")]
            if len(parts) == 2:
                self.add("user", parts[0], g, seg=seg, event="distractor", kind="pause", block=blk)
                g = round(self.rng.uniform(1.5, 3.0), 2)
            answered = k + 1 < len(turns) and turns[k + 1]["speaker"] == "moshi"
            if answered:
                keep.setdefault("event", "ordinary_request")
            self.add("user", parts[-1], g, seg=seg, block=blk, **keep)

    def times(self) -> dict[int, tuple[float, float]]:
        out, cur = {}, 0.0
        for t in self.turns:
            s = out[t["anchor"]["uid"]][1] + t["anchor"]["offset"] if "anchor" in t else cur + t["gap_before"]
            out[t["_uid"]] = (s, s + t["est_sec"])
            cur = max(cur, s + t["est_sec"])
        return out

    def total(self) -> float:
        return round(max([e for _, e in self.times().values()] or [0.0]) + TAIL_SEC, 2)

    def trim_seg(self, seg: str, min_turns: int) -> bool:
        """Drop the last user..Moshi block of a segment (a block ends on a Moshi turn). A name
        exchange at the very end of the segment (after its last Moshi turn) stays, and the block
        before it goes. Never drops a protected turn or an anchor target. False if it can't."""
        idx = [i for i, t in enumerate(self.turns) if t["_seg"] == seg]
        moshi = [i for i in idx if self.turns[i]["speaker"] == "moshi"]
        if not moshi:
            if idx and min_turns == 0 and not any(self.turns[i].get("kind") in PROTECTED for i in idx):
                self.turns = [t for i, t in enumerate(self.turns) if i not in set(idx)]
                return True
            return False
        last = moshi[-1]
        prev = moshi[-2] if len(moshi) > 1 else None
        drop = [i for i in idx if (prev is None or i > prev) and i <= last]
        if len(idx) - len(drop) < min_turns:
            return False
        if any(self.turns[i].get("kind") in PROTECTED for i in drop):
            return False  # never cut a planned name exchange / Moshi-by-name turn
        uids = {self.turns[i]["_uid"] for i in drop}
        if any("anchor" in t and t["anchor"]["uid"] in uids for t in self.turns):
            return False
        self.turns = [t for i, t in enumerate(self.turns) if i not in set(drop)]
        return True

    def finalize(self, meta: dict) -> list[dict]:
        pos = {t["_uid"]: i for i, t in enumerate(self.turns)}
        out = []
        for t in self.turns:
            t = dict(t)
            if "anchor" in t:
                t["anchor"] = {"turn": pos[t["anchor"]["uid"]], "offset": t["anchor"]["offset"]}
            t.pop("_uid")
            t.pop("_seg")
            out.append(t)
        if "deadline" in meta:
            meta["deadline"] = {"after_turn": pos[meta["deadline"].pop("after_uid")], "sec": meta["deadline"]["sec"]}
        return out


def spots(turns: list[dict], allow_end: bool = False) -> list[int]:
    """Indices k where an exchange can go between a Moshi turn (k-1) and the next user turn."""
    return [k for k in range(1, len(turns) + (1 if allow_end else 0))
            if turns[k - 1]["speaker"] == "moshi" and (k == len(turns) or turns[k]["speaker"] == "user")]


def build_v2(kind: str, P: PoolsV2, rng: random.Random, cond: set[str], vmap: dict) -> dict:
    meta: dict = {"type": kind, "wait_type": None if kind == "ordinary" else kind, "conditions": sorted(cond)}
    position = next((c.split(":")[1] for c in cond if c.startswith("position:")), None)
    cond = {c for c in cond if not c.startswith("position:")}
    meta["conditions"] = sorted(cond)
    # content blocks first (the voices follow the first block)
    if kind == "ordinary":
        D = P.draw("ordinary_dialogues")
        primary = D
    elif kind == "describe_options":
        O = P.draw("option_sets")
        D = P.draw("ordinary_dialogues")
        primary = O
    else:
        C = P.draw("contexts")
        D = P.draw("ordinary_dialogues")
        primary = C
    S = P.draw("side_conversations") if kind == "side_conversation" else None
    train_v = {v["id"]: v for v in P.voices}
    vm = vmap.get(primary["id"], {})
    user_v = vm.get("user") if vm.get("user") in train_v else rng.choice(P.voices)["id"]
    friend_v, friend_name, friend_id = None, None, None
    if S is not None:
        vs = vmap.get(S["id"], {})
        friend_v = vs.get("friend") if vs.get("friend") in train_v and vs.get("friend") != user_v else \
            rng.choice([v for v in P.voices if v["id"] != user_v])["id"]
        names = {x["text"]: x for x in P.decks["friend_names"].items}
        if vs.get("friend_name") in names:
            friend_name, friend_id = vs["friend_name"], names[vs["friend_name"]]["id"]
        else:
            fx = P.draw("friend_names")
            friend_name, friend_id = fx["text"], fx["id"]
    b = B2(rng, friend_name or "")
    wk = "ordinary" if kind == "ordinary" else "wait"

    def phr(deck: str) -> dict:
        x = P.draw(deck)
        b.phrasing_ids.append(x["id"])
        return x

    # name exchanges, prepared as raw turns and inserted into the dialogue parts below
    inserts = []
    if "other_name" in cond:
        nq = P.draw("name_questions")
        nm = P.draw("person_names", avoid=lambda x: x["text"] == friend_name)
        inserts.append(other_name_exchange(nq, nm["text"], rng.random() < ANSWER_P))
        b.phrasing_ids.append(nm["id"])
        b.block_ids.append(nq["id"])  # a content block: used once per set
    if "moshi_mention" in cond:
        mm = P.draw("moshi_mentions")
        nm = P.draw("person_names", avoid=lambda x: x["text"] == friend_name)
        ex = other_name_exchange(mm, nm["text"], rng.random() < 0.75, kind="moshi_mention")
        ex[0]["addressee"] = nm["text"] if "{name}" in mm["user"] else None
        inserts.append(ex)
        b.block_ids.append(mm["id"])
    other_v = None
    if inserts:
        other_v = rng.choice([v for v in P.voices if v["id"] not in (user_v, friend_v)])["id"]

    # Opening (always)
    gr = phr("greeting")
    b.add("moshi", gr["text"], round(rng.uniform(0.3, 0.8), 2), role="greeting", phrasing=gr["id"])
    a = phr("greeting_answer")
    b.add("user", a["text"], round(rng.uniform(0.3, 1.0), 2), event="greeting_answer", phrasing=a["id"])
    first_gap = round(rng.uniform(0.2, 0.6), 2)

    dturns = [dict(t) for t in D["turns"]]
    if kind == "ordinary" and "wait_like" in cond:
        dturns = add_wait_like(dturns, P, rng, b)
    # where the dialogue goes for wait types
    if kind != "ordinary":
        if position == "middle":  # the dialogue's first exchange before, its next one after the wait
            pre, post = dturns[:2], dturns[2:4]
        elif position == "late":
            pre, post = dturns, []
        else:
            pre, post = [], dturns
        meta["position_plan"] = position
    else:
        pre, post = dturns, []
    # Moshi by name: one user request in the dialogue part (or the context)
    ctx_turns = [dict(t) for t in (C["turns"] if kind not in ("ordinary", "describe_options") else [])]
    if "moshi_name" in cond:
        cands = [(lst, k) for lst in (pre, post, ctx_turns) for k in range(len(lst) - 1)
                 if lst[k]["speaker"] == "user" and lst[k + 1]["speaker"] == "moshi" and " / " not in lst[k]["text"]
                 and not lst[k].get("kind") and (lst is ctx_turns or k < 3)]
        if not cands:
            raise ValueError("no turn to address Moshi by name")
        lst, k = rng.choice(cands)
        lst[k]["text"], np_ = add_moshi_name(lst[k]["text"], rng)
        lst[k]["kind"], lst[k]["name_pos"], lst[k]["addressee"] = "moshi_name", np_, "Moshi"
    # insert the exchanges between a Moshi turn and the next user turn
    for ex in inserts:
        targets = [(lst, k) for lst in (pre, post)  # early in the segment: survives trimming
                   for k in spots(lst)[:1 if (lst is pre and position == "late") else 2]]
        if pre and pre[-1]["speaker"] == "moshi":
            targets.append((pre, len(pre)))  # after the dialogue, before the context / lead-in
        if post and post[0]["speaker"] == "user":
            targets.append((post, 0))  # right after the resume, before the next dialogue
        if kind not in ("ordinary", "describe_options"):
            targets += [(ctx_turns, len(ctx_turns))] if ctx_turns and ctx_turns[-1]["speaker"] == "moshi" else []
        if not targets:
            raise ValueError("no place for a name exchange")
        lst, k = rng.choice(targets)
        ex = [dict(t) for t in ex]
        ex[0]["gap"] = None
        nxt_gap = round(rng.uniform(0.8, 2.0) if len(ex) == 2 else rng.uniform(1.8, 3.5), 2)
        if k < len(lst):
            lst[k] = dict(lst[k], gap=nxt_gap)
        else:
            meta["_gap_pre_end" if lst is pre else "_gap_after_insert"] = nxt_gap
        lst[k:k] = ex

    def ack() -> None:
        if rng.random() < ACK_P:
            x = phr("acks")
            b.add("moshi", x["text"], round(rng.uniform(0.2, 0.7), 2), role="ack", phrasing=x["id"])

    def ending(ctx: dict, force: bool = False, moshi_gap_=None) -> None:
        p_end = {"late": 0.25, "middle": 1.0, "early": 1.0}.get(position, ENDING_P)
        if force or rng.random() < p_end:
            e = rng.choice(ctx["endings"])
            b.add("user", e["user"], round(rng.uniform(0.3, 0.6), 2) if force else None,
                  event="ordinary_request", block=ctx["id"], seg="ending")
            b.add("moshi", e["moshi"], moshi_gap_, role="followup", block=ctx["id"], seg="ending")

    if kind == "ordinary":
        b.dialogue(pre, D["id"], first_gap, seg="main")
        meta["topics"] = [D["topic"]]
        if b.total() < MIN_SEC:
            d2 = P.draw("ordinary_dialogues", avoid=lambda x: x["id"] == D["id"])
            b.dialogue(d2["turns"], d2["id"], round(rng.uniform(1.0, 2.5), 2), seg="main2")
            meta["topics"].append(d2["topic"])
        while b.total() > MAX_SEC and (b.trim_seg("main2", 2) or b.trim_seg("main", 4)):
            pass
        for c, k_ in (("wait_like", "wait_like"), ("other_name", "other_name"), ("moshi_name", "moshi_name"),
                      ("moshi_mention", "moshi_mention")):
            if c in cond and not any(t.get("kind") == k_ for t in b.turns):
                raise ValueError(f"{c} turn was cut")
        if b.turns[-1]["speaker"] != "moshi":
            raise ValueError("ends without Moshi")
    else:
        if pre:
            b.dialogue(pre, D["id"], first_gap, seg="pre")
        if kind == "describe_options":
            b.block_ids.append(O["id"])
            meta["topics"] = [O["topic"]]
            lead = O["lead_in"]
            if pre:
                lead = rng.choice(TRANSITIONS) + lead
            b.add("user", lead, first_gap if not pre else meta.pop("_gap_pre_end", round(rng.uniform(0.6, 1.5), 2)),
                  block=O["id"])
            x = phr("describe_instruction")
            instr = b.add("user", x["text"], round(rng.uniform(0.4, 0.9), 2), event="instruction",
                          kind="describe_options", phrasing=x["id"])
            ack()
            long_ = "long" in cond
            for k, o in enumerate(O["options"]):
                gp = (round(rng.uniform(3.5, 8.0), 2) if long_ else round(rng.uniform(1.0, 3.0), 2)) if k \
                    else round(rng.uniform(0.6, 1.5), 2)
                if k and gp > 1.5:
                    b.turns[-1].update(event="distractor", kind="pause")
                b.add("user", o, gp, block=O["id"])
            x = phr("describe_cue")
            cue_uid = b.add("user", x["text"], round(rng.uniform(0.5, 1.5), 2), event="return_cue", kind="explicit",
                            phrasing=x["id"])
            b.add("moshi", rng.choice(O["moshi_comment"]), role="resume", block=O["id"])
            tm = b.times()
            if long_ and tm[cue_uid][1] - tm[instr][1] < LONG_SEC + 1.5:
                raise ValueError("describe wait too short for the long condition")
        else:
            ctx = C
            if pre and ctx_turns and ctx_turns[0]["speaker"] == "user":
                t0 = ctx_turns[0]["text"]
                tr = rng.choice(TRANSITIONS)
                ctx_turns[0] = dict(ctx_turns[0], text=tr + (t0[0].lower() + t0[1:] if tr.endswith(", ") and
                                                             not t0.startswith(("I ", "I'")) else t0))
            b.dialogue(ctx_turns, ctx["id"], first_gap if not pre else
                       meta.pop("_gap_pre_end", round(rng.uniform(0.6, 1.5), 2)), seg="ctx")
            meta["topics"] = [ctx["topic"]]
            g_ins = meta.pop("_gap_after_insert", None)
            if kind == "think_time":
                secs = [x for x in P.think_long if x >= 33 or "early_end" not in cond] if "long" in cond else [s for s in P.think_short
                                                            if s >= 10 or not cond & {"self_talk", "early_end"}]
                n = rng.choice(secs)
                x = phr("think_instruction")
                instr = b.add("user", x["text"].replace("{n}", num_words(n)), g_ins, event="instruction",
                              kind="think_time", wait_sec=n, phrasing=x["id"])
                ack()
                meta["wait_sec"] = n
                stop = n - 1.0
                if "early_end" in cond:  # a long wait stays >= 30 s when it ends early
                    lo = max(4.0, 0.4 * n, LONG_SEC + 0.5 if "long" in cond else 0.0)
                    early_at = round(rng.uniform(lo, n - 1.5), 2)
                    stop = early_at - 1.0
                if "self_talk" in cond:
                    t0 = 3.0
                    for _ in range(rng.choice([1, 2, 3] if n >= 30 else [1, 2])):
                        m = P.draw("self_talk")
                        if t0 + est_sec(m["text"]) > stop:
                            break
                        off = round(rng.uniform(t0, max(t0, stop - est_sec(m["text"]))), 2)
                        b.phrasing_ids.append(m["id"])
                        b.add("user", m["text"], anchor={"uid": instr, "offset": off},
                              event="distractor", kind="self_talk", phrasing=m["id"])
                        t0 = off + est_sec(m["text"]) + 1.5
                    if not any(t.get("kind") == "self_talk" for t in b.turns):
                        raise ValueError("no room for self-talk")
                if "early_end" in cond:
                    x = phr("early_end")
                    b.add("user", x["text"], anchor={"uid": instr, "offset": early_at},
                          event="return_cue", kind="early_end", phrasing=x["id"])
                    meta["early_end"] = {"after_sec": early_at, "direct": "early_direct" in cond}
                    if "early_direct" in cond:
                        b.add("moshi", rng.choice(ctx["think_reask"]), round(rng.uniform(0.3, 1.0), 2),
                              role="resume", block=ctx["id"])
                        e = rng.choice(ctx["endings"])
                        b.add("user", e["user"], event="ordinary_request", block=ctx["id"])
                        b.add("moshi", e["moshi"], role="followup", block=ctx["id"])
                    else:
                        ending(ctx, force=True, moshi_gap_=round(rng.uniform(0.3, 0.8), 2))
                else:
                    b.add("moshi", rng.choice(ctx["think_reask"]), role="resume", block=ctx["id"],
                          anchor={"uid": instr, "offset": round(n + rng.uniform(0.0, 0.6), 2)})
                    meta["deadline"] = {"after_uid": instr, "sec": n}
                    ending(ctx)
            else:  # side_conversation
                if "friend_first" in cond:
                    fi = phr("friend_interrupt")
                    b.add("friend", fi["text"], g_ins, event="distractor", kind="friend_interrupt", phrasing=fi["id"])
                    x = phr("side_instruction_ff")
                    g_ins = round(rng.uniform(0.3, 0.9), 2)
                else:
                    x = phr("side_instruction")
                b.add("user", x["text"], g_ins, event="instruction", kind="side_conversation", phrasing=x["id"])
                ack()
                b.block_ids.append(S["id"])
                meta["topics"].append(S["topic"])
                turns = [dict(t, block=S["id"]) for t in S["turns"]]
                ins = (["near_miss"] if "near_miss" in cond else []) + \
                      (["friend_question"] if "extra_question" in cond else [])
                for dk in ins:
                    d = P.draw(dk)
                    b.phrasing_ids.append(d["id"])
                    b.block_ids.append(d["id"])
                    pos_ = rng.choice([0, len(turns)])
                    turns[pos_:pos_] = [{"speaker": "user", "text": d["user"], "event": "distractor",
                                         "kind": d["kind"], "phrasing": d["id"]},
                                        {"speaker": "friend", "text": d["friend"], "phrasing": d["id"]}]
                for k, t in enumerate(turns):
                    if t["speaker"] == "user" and "event" not in t and t["text"].rstrip().endswith("?"):
                        t.update(event="distractor", kind="friend_question")
                    b.add(t["speaker"], t["text"], round(rng.uniform(0.6, 1.5), 2) if k == 0 else None,
                          **{x_: t[x_] for x_ in ("event", "kind", "block", "phrasing") if x_ in t})
                x = phr("side_cue")
                b.add("user", x["text"], round(rng.uniform(0.5, 2.0), 2), event="return_cue", kind="explicit",
                      phrasing=x["id"])
                b.add("moshi", resume_line(ctx, kind, False, rng, P), role="resume", block=ctx["id"])
                ending(ctx)
        if post:
            p0 = post[0]
            if p0["speaker"] == "user" and p0.get("kind") is None and "event" not in p0 and position == "early":
                post[0] = dict(p0, text=rng.choice(TRANSITIONS[:8]) + p0["text"])
            b.dialogue(post, None if pre else D["id"], round(rng.uniform(0.8, 2.0), 2), seg="post")
            for t in b.turns:  # the post turns of a split dialogue still name their block
                if t["_seg"] == "post" and "block" not in t:
                    t["block"] = D["id"]
        # fit: drop the optional ending, then trim the dialogue (pre for late/middle, post for early)
        def drop_ending() -> bool:
            return (any(t["_seg"] == "ending" for t in b.turns) and "early_end" not in cond
                    and b.trim_seg("ending", 0))
        order = {"early": [lambda: b.trim_seg("post", 4), drop_ending, lambda: b.trim_seg("post", 2)],
                 "middle": [lambda: b.trim_seg("post", 2), lambda: b.trim_seg("pre", 2), drop_ending,
                            lambda: b.trim_seg("post", 0)],
                 "late": [lambda: b.trim_seg("pre", 2), drop_ending]}[position]
        # a very long think wait (>= LEAN_SEC) may drop the dialogue around it when nothing else
        # fits, so durations up to 60 s survive the length limit (else they'd be rejected)
        lean = kind == "think_time" and meta.get("wait_sec", 0) >= LEAN_SEC
        if lean:
            order = order + [lambda: b.trim_seg("post", 0), lambda: b.trim_seg("pre", 0)]
        while b.total() > MAX_SEC and any(f() for f in order):
            pass
        for c in ("other_name", "moshi_name", "moshi_mention"):
            if c in cond and not any(t.get("kind") == c for t in b.turns):
                raise ValueError(f"{c} turn was cut")
        if position == "early" and not any(t["_seg"] == "post" for t in b.turns) and not lean:
            raise ValueError("early position lost its dialogue")
        if position in ("middle", "late") and not any(t["_seg"] == "pre" for t in b.turns) and not lean:
            raise ValueError("position lost its dialogue")
    total = b.total()
    if not MIN_SEC <= total <= MAX_SEC:
        raise ValueError(f"length {total}")
    # realized position: share of conversation speech (greeting exchange excluded) before the instruction
    if kind != "ordinary":
        tm = b.times()
        ins = next(t for t in b.turns if t.get("event") == "instruction")
        talk = [(tm[t["_uid"]], t) for t in b.turns[2:] if t.get("event") not in ("instruction",)
                and t["_seg"] in ("pre", "ctx", "post", "ending", "main") and t["speaker"] in ("user", "moshi", "other")
                and t.get("kind") not in ("self_talk",)]
        before = sum(e - s for (s, e), t in talk if s < tm[ins["_uid"]][0])
        after = sum(e - s for (s, e), t in talk if s > tm[ins["_uid"]][0]
                    and (t["_seg"] in ("post", "ending") or t.get("role") in ("resume", "followup")
                         or t.get("event") == "ordinary_request"))
        meta["instruction_position"] = round(before / max(before + after, 1e-6), 3)
    _, speech = 0, sum(t["est_sec"] for t in b.turns)
    turns = b.finalize(meta)
    phr_meta = {i: {k: P.style[i][k] for k in ("style", "tone", "states_return", "source") if k in P.style[i]}
                for i in b.phrasing_ids if i in P.style}
    return {
        **meta, "topic": meta["topics"][0],
        "voices": {"user": user_v, "friend": friend_v, "other": other_v, "moshi": P.moshi_voice},
        "friend_name": friend_name, "friend_name_id": friend_id,
        "phrasing_ids": b.phrasing_ids, "phrasing_styles": phr_meta, "block_ids": b.block_ids,
        "est_total_sec": total, "est_speech_sec": round(speech, 2), "turns": turns,
    }


class OrderedPlanV2(g1.OrderedPlan):
    """v1 plan (blocks with the exact mix, v1 conditions on strides) + v2 conditions."""

    def next(self) -> tuple[str, set[str]]:
        kind, cond = super().next()
        if kind != "ordinary":
            k = self.sub.get(f"pos_{kind}", 0)
            self.sub[f"pos_{kind}"] = k + 1
            cond.add("position:" + POSITIONS[(k + hash_off(kind)) % 3])
        if kind in ("think_time", "describe_options") and self._hit2("long", f"long_{kind}", LONG_P):
            cond.add("long")
        if kind == "side_conversation" and self._hit2("friend_first", "side_ff", FRIEND_FIRST_P):
            cond.add("friend_first")
        wk = "ordinary" if kind == "ordinary" else "wait"
        for c, p in (("other_name", OTHER_NAME_P), ("moshi_name", MOSHI_NAME_P), ("moshi_mention", MENTION_P)):
            if self._hit2(c, f"{c}_{wk}", p[wk]):
                cond.add(c)
        return kind, cond

    def _hit2(self, name: str, key: str, frac: float) -> bool:
        k = self.sub.get(key, 0)
        self.sub[key] = k + 1
        return stride_hit(k, frac, PHASE2[name])


def hash_off(kind: str) -> int:
    return {"think_time": 0, "side_conversation": 1, "describe_options": 2}.get(kind, 0)


def voice_map(dirs: list[Path]) -> dict:
    """block id -> v1 voices of the file that used it (first block -> user voice; side conversation -> friend)."""
    out = {}
    for d in dirs:
        for f in sorted(d.glob("*_*.json")):
            s = json.loads(f.read_text())
            for b_ in s["block_ids"]:
                e = out.setdefault(b_, {})
                e.setdefault("user", s["voices"]["user"])
                if ".side" in b_ and s["voices"].get("friend"):
                    e.setdefault("friend", s["voices"]["friend"])
                    e.setdefault("friend_name", s.get("friend_name"))
    return out


def insert_turns(s: dict, pos: int, new: list[dict]) -> None:
    """Insert turns into a v1 script at index pos, shifting anchors and the deadline."""
    for t in s["turns"]:
        if "anchor" in t and t["anchor"]["turn"] >= pos:
            t["anchor"]["turn"] += len(new)
    if s.get("deadline") and s["deadline"]["after_turn"] >= pos:
        s["deadline"]["after_turn"] += len(new)
    s["turns"][pos:pos] = new


def timeline_total(turns: list[dict]) -> float:
    ends, cur = [], 0.0
    for t in turns:
        st = ends[t["anchor"]["turn"]] + t["anchor"]["offset"] if "anchor" in t else cur + t["gap_before"]
        ends.append(st + t["est_sec"])
        cur = max(cur, ends[-1])
    return round(cur + TAIL_SEC, 2)


def eval_v2(args) -> None:
    """Eval v2: drop 40 v1 eval scripts (stratified), add 40 other_name scripts of the same types/conditions."""
    v1 = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(args.v1_eval / "*_*.json")))]
    rng = random.Random("eval-v2-other-name-13")
    drop = []
    for kind, k in EVAL_V2_REPLACE.items():
        cands = [s for s in v1 if s["type"] == kind]
        drop += rng.sample(cands, k)
    drop_ids = {s["id"] for s in drop}
    keep = [s for s in v1 if s["id"] not in drop_ids]
    used = {b_ for s in keep for b_ in s["block_ids"]}
    P = Pools("eval", rng, unique=True, exclude=used)  # the v1 eval pools and phrasings, like the other 360
    nq_eval = [x for x in json.loads((V2 / "name_questions_eval.json").read_text())["name_questions"]]
    P.decks["name_questions"] = Deck(nq_eval, rng, "name_questions", unique=True)
    names = [x for x in json.loads((V2 / "names.json").read_text())["person_names"] if x["split"] == "eval"]
    name_deck = Deck(names, rng)
    args.out.mkdir(parents=True, exist_ok=True)
    for f in args.out.glob("*.json"):
        f.unlink()
    for s in keep:
        shutil.copy(args.v1_eval / f"{s['id']}.json", args.out / f"{s['id']}.json")
    new_rows = []
    for i, old in enumerate(sorted(drop, key=lambda s: s["order_index"])):
        cond = set(old["conditions"])
        for attempt in range(200):
            try:
                s = g1.build(old["type"], "eval", P, rng, cond)
                nq = P.draw("name_questions")
                nm = name_deck.draw(avoid=lambda x: x["text"] == s.get("friend_name"))
                ex = other_name_exchange(nq, nm["text"], rng.random() < ANSWER_P)
                turns = s["turns"]
                if s["type"] == "ordinary":
                    cands = [k for k in range(3, len(turns)) if turns[k - 1]["speaker"] == "moshi"
                             and turns[k]["speaker"] == "user" and k < len(turns) - 1]
                else:
                    ins = next(k for k, t in enumerate(turns) if t.get("event") == "instruction")
                    cands = [ins] if turns[ins - 1]["speaker"] == "moshi" else [2]
                if not cands:
                    raise ValueError("no spot")
                k = rng.choice(cands)
                new = []
                for j, t in enumerate(ex):
                    t = {x: v for x, v in t.items() if x != "gap"}
                    sp = t["speaker"]
                    t["est_sec"] = est_sec(t["text"])
                    t["gap_before"] = round(rng.uniform(0.5, 1.5) if j == 0 else rng.uniform(0.3, 1.0), 2)
                    new.append(t)
                insert_turns(s, k, new)
                if "anchor" not in s["turns"][k + len(new)]:
                    s["turns"][k + len(new)]["gap_before"] = round(
                        rng.uniform(0.8, 2.0) if len(new) == 2 else rng.uniform(1.8, 3.5), 2)
                tot = timeline_total(s["turns"])
                if tot > MAX_SEC:
                    raise ValueError("too long")
                s["est_total_sec"] = tot
                s["block_ids"].append(nq["id"])
                s["phrasing_ids"] += [nq["id"], nm["id"]]
                vs = [v["id"] for v in P.voices if v["id"] not in (s["voices"]["user"], s["voices"].get("friend"))]
                s["voices"]["other"] = rng.choice(vs)
                P.commit()
                break
            except ValueError:
                P.rollback()
        else:
            raise SystemExit(f"could not build other_name script for {old['id']}")
        sid = f"eval_13_on_{i:05d}"
        script = {"id": sid, "split": "eval", "seed": 13, "order_index": 400 + i,
                  "generator": {"script": "generate_scripts_v2.py --eval-v2", "replaces": old["id"], "seed": 13},
                  **s, "other_name": {"replaces": old["id"], "name_pos": nq["name_pos"], "answered": len(ex) == 2}}
        script["groups"] = g1.eval_groups(s["type"], s["conditions"]) + ["other_name"]
        script["note"] = ("Moshi turns are reference_only: the ideal replies, for eval_full and loss evaluation. "
                          "Never render them into the model's input.")
        for t in script["turns"]:
            if t["speaker"] == "moshi":
                t["reference_only"] = True
        (args.out / f"{sid}.json").write_text(json.dumps(script, indent=1, ensure_ascii=False) + "\n")
        new_rows.append(script)
    (args.out / "replaced.json").write_text(json.dumps(
        {"dropped_v1": sorted(drop_ids), "added": [s["id"] for s in new_rows]}, indent=1) + "\n")
    allv = keep + new_rows
    print(f"eval v2: {len(keep)} v1 scripts kept + {len(new_rows)} other_name scripts -> {args.out}")
    print("  types:", dict(Counter(s["type"] for s in allv)))
    print("  groups:", dict(Counter(gg for s in allv for gg in s["groups"]).most_common()))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["train"], default="train")
    p.add_argument("--n", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--only", choices=list(MIX["train"]), default=None)
    p.add_argument("--id-prefix", default=None)
    p.add_argument("--exclude-ordinary-from", type=Path, action="append", default=[],
                   help="Don't use blocks of the ordinary-type scripts in this folder (control additions).")
    p.add_argument("--voices-from", type=Path, action="append",
                   default=[ROOT / "data/scripts/train", ROOT / "data/scripts/control"],
                   help="v1 script folders: reuse their voice for the same block (TTS cache hits).")
    p.add_argument("--eval-v2", action="store_true")
    p.add_argument("--v1-eval", type=Path, default=ROOT / "data/scripts/eval")
    args = p.parse_args()
    if args.eval_v2:
        return eval_v2(args)
    exclude = set()
    for d in args.exclude_ordinary_from:
        for f in d.glob("*_*.json"):
            s = json.loads(f.read_text())
            if s["type"] == "ordinary":
                exclude |= set(s["block_ids"])
    rng = random.Random(f"v2-{args.split}-{args.seed}" + (f"-only-{args.only}" if args.only else ""))
    P = PoolsV2(args.split, rng, unique=True, exclude=exclude)
    vmap = voice_map(args.voices_from)
    plan = OrderedPlanV2(args.split, rng, args.only)
    args.out.mkdir(parents=True, exist_ok=True)
    for f in args.out.glob("*.json*"):
        f.unlink()
    prefix = args.id_prefix or f"{args.split}v2_{args.seed:02d}"
    rows = []
    for i in range(args.n):
        kind, cond_i = plan.next()
        for attempt in range(200):
            try:
                s = build_v2(kind, P, rng, set(cond_i), vmap)
                P.commit()
                break
            except ValueError:
                P.rollback()
            except PoolExhausted as e:
                raise SystemExit(f"script {i} ({kind}): {e}")
        else:
            raise SystemExit(f"could not build a {kind} script {sorted(cond_i)}")
        sid = f"{prefix}_{i:05d}"
        script = {"id": sid, "split": args.split, "seed": args.seed, "order_index": i, "data_version": 2,
                  "generator": {"script": "generate_scripts_v2.py", "ordered": True, "unique": True, "only": args.only,
                                "seed": args.seed, "block": 1 if args.only else BLOCK[args.split],
                                "exclude_ordinary_from": [str(d) for d in args.exclude_ordinary_from]},
                  **s}
        (args.out / f"{sid}.json").write_text(json.dumps(script, indent=1, ensure_ascii=False) + "\n")
        rows.append({"id": sid, "type": kind, "conditions": s["conditions"], "position": s.get("position_plan"),
                     "instruction_position": s.get("instruction_position"), "est_total_sec": s["est_total_sec"],
                     "wait_sec": s.get("wait_sec")})
    (args.out / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    tot = sum(r["est_total_sec"] for r in rows)
    print(f"{len(rows)} scripts -> {args.out}: {dict(Counter(r['type'] for r in rows))}; est {tot / 3600:.2f} h")
    print("  conditions:", dict(Counter(c for r in rows for c in r["conditions"]).most_common()))


if __name__ == "__main__":
    main()
