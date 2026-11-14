"""Generate conversation scripts (text + planned timing, no audio) from the pools.

Combines building blocks from data/pools/ and phrasings from data/splits/ at random, drawing
only from the requested split, so eval voices, held-out phrasings, eval topics and held-out
distractors/friend names never appear in train scripts (scripts/check_split_leakage.py
verifies this on the output).

Script types (fractions per split):
  train: think_time 0.3, side_conversation 0.3, ordinary 0.4
  eval:  think_time 0.25, side_conversation 0.25, describe_options 0.2 (eval only), ordinary 0.3
  think_time         context -> "give me N seconds to think, then ask me again" -> [ack] ->
                     silence -> Moshi asks again at the deadline -> [answer + Moshi reply].
                     Eval: in 40% (at least one) the user extends before the deadline.
  side_conversation  context -> "I'm talking to my friend, stay quiet until I'm back" -> [ack]
                     -> user and friend talk (with distractors) -> return cue -> Moshi
                     welcomes the user back, referring to the context -> [answer + reply].
  describe_options   lead-in -> "let me describe all the options before you comment" ->
                     [ack] -> options with pauses -> "that's all, what do you think?" -> Moshi.
  ordinary           a user-Moshi dialogue (two if the first is short).
About half the scripts start with a short Moshi greeting.

Each script is one JSON file with: id, split, type, topic(s), voices (user, friend, moshi),
friend_name, phrasing_ids, block_ids, wait_sec / extension, and an ordered list of turns.
Each turn has speaker (user | friend | moshi), text, est_sec (estimated spoken length), and
either gap_before (silence after the previous turn ends) or anchor {turn, offset} (start
offset seconds after the end of turn number `turn`, used for deadlines). Optional: event
(instruction | extension | distractor | return_cue | ordinary_request) with kind, role for
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
MIX = {"train": {"think_time": 0.3, "side_conversation": 0.3, "ordinary": 0.4},
       "eval": {"think_time": 0.25, "side_conversation": 0.25, "describe_options": 0.2, "ordinary": 0.3}}
MIN_SEC, MAX_SEC = 30.0, 85.0
GREETING_P, ACK_P, ENDING_P, EXTENSION_P = 0.5, 0.5, 0.7, 0.4
TAIL_SEC = 1.0
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
        for k in ("greetings", "acks", "friend_names"):
            self.decks[k] = Deck([x for x in gen[k] if ok(x)], rng)
        for kind in ("friend_question", "near_miss"):
            self.decks[kind] = Deck([x for x in gen["distractors"] if x["kind"] == kind and ok(x)], rng)
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
            self.add("user", parts[-1], g, event="ordinary_request" if answered else None, block=block)

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


def build(kind: str, split: str, P: Pools, rng: random.Random, extension: bool = False) -> dict:
    user_v = rng.choice(P.voices)
    friend_v = rng.choice([v for v in P.voices if v["id"] != user_v["id"]])
    friend = P.draw("friend_names")
    b = Builder(split, rng, friend["text"])
    meta: dict = {"type": kind, "wait_type": None if kind == "ordinary" else kind}
    if rng.random() < GREETING_P:
        g = P.draw("greetings")
        b.add("moshi", g["text"], round(rng.uniform(0.3, 1.0), 2), role="greeting", phrasing=g["id"])
        b.phrasing_ids.append(g["id"])
    first_gap = round(rng.uniform(0.3, 1.2), 2)

    def ack(after: int) -> int | None:
        if rng.random() >= ACK_P:
            return None
        a = P.draw("acks")
        b.phrasing_ids.append(a["id"])
        return b.add("moshi", a["text"], round(rng.uniform(0.2, 0.7), 2), role="ack", phrasing=a["id"])

    def ending(ctx: dict) -> None:
        if rng.random() < ENDING_P:
            e = rng.choice(ctx["endings"])
            b.add("user", e["user"], event="ordinary_request", block=ctx["id"])
            b.add("moshi", e["moshi"], role="followup", block=ctx["id"])

    if kind == "ordinary":
        dlg = P.draw("ordinary_dialogues")
        b.dialogue(dlg["turns"], dlg["id"], first_gap)
        meta["topics"] = [dlg["topic"]]
        if b.timeline()[0] < MIN_SEC:
            d2 = P.draw("ordinary_dialogues", avoid=lambda x: x["id"] == dlg["id"])
            b.dialogue(d2["turns"], d2["id"], round(rng.uniform(1.0, 2.5), 2))
            meta["topics"].append(d2["topic"])
        while b.timeline()[0] > MAX_SEC and len(b.turns) > 4:  # cut at a Moshi turn
            b.turns.pop()
            while b.turns and b.turns[-1]["speaker"] != "moshi":
                b.turns.pop()
        if b.turns[-1]["speaker"] == "moshi":
            last_user = max(i for i, t in enumerate(b.turns) if t["speaker"] == "user")
            b.turns[last_user].setdefault("event", "ordinary_request")

    elif kind in ("think_time", "side_conversation"):
        ctx = P.draw("contexts")
        b.dialogue(ctx["turns"], ctx["id"], first_gap)
        meta["topics"] = [ctx["topic"]]
        if kind == "think_time":
            n = rng.choice(P.think_seconds)
            ph = P.draw("think_instruction")
            b.phrasing_ids.append(ph["id"])
            instr = b.add("user", ph["text"].replace("{n}", num(n)), event="instruction",
                          kind="think_time", wait_sec=n, phrasing=ph["id"])
            ack(instr)
            anchor, wait = instr, n
            meta["wait_sec"] = n
            if extension:
                m = rng.choice(P.extension_seconds)
                ex = P.draw("extension")
                b.phrasing_ids.append(ex["id"])
                lead = round(rng.uniform(1.0, 3.0), 2)  # said this long before the deadline
                anchor = b.add("user", ex["text"].replace("{n}", num(m)), event="extension",
                               extra_sec=m, phrasing=ex["id"], anchor={"turn": instr, "offset": round(n - lead, 2)})
                wait = m
                meta["extension"] = {"extra_sec": m, "said_before_deadline_sec": lead}
            reask = rng.choice(ctx["think_reask"])
            b.add("moshi", reask, role="resume", block=ctx["id"],
                  anchor={"turn": anchor, "offset": round(wait + rng.uniform(0.0, 0.6), 2)})
            meta["deadline"] = {"after_turn": anchor, "sec": wait}
        else:
            ph = P.draw("side_instruction")
            b.phrasing_ids.append(ph["id"])
            instr = b.add("user", ph["text"], event="instruction", kind="side_conversation", phrasing=ph["id"])
            ack(instr)
            side = P.draw("side_conversations")
            b.block_ids.append(side["id"])
            meta["topics"].append(side["topic"])
            turns = [dict(t, block=side["id"]) for t in side["turns"]]
            for _ in range(rng.choice([0, 1, 1, 2])):  # insert distractor exchanges
                d = P.draw(rng.choice(["friend_question", "near_miss"]))
                b.phrasing_ids.append(d["id"])
                pos = rng.randrange(1, len(turns) + 1)
                turns[pos:pos] = [{"speaker": "user", "text": d["user"], "event": "distractor",
                                   "kind": d["kind"], "phrasing": d["id"]},
                                  {"speaker": "friend", "text": d["friend"], "phrasing": d["id"]}]
            for k, t in enumerate(turns):
                if t["speaker"] == "user" and "event" not in t and t["text"].rstrip().endswith("?"):
                    t.update(event="distractor", kind="friend_question")
                b.add(t["speaker"], t["text"], round(rng.uniform(0.6, 1.5), 2) if k == 0 else None,
                      **{x: t[x] for x in ("event", "kind", "block", "phrasing") if x in t})
            cue = P.draw("side_cue")
            b.phrasing_ids.append(cue["id"])
            b.add("user", cue["text"], round(rng.uniform(0.5, 2.0), 2), event="return_cue", phrasing=cue["id"])
            b.add("moshi", rng.choice(ctx["welcome_back"]), role="resume", block=ctx["id"])
        ending(ctx)
        # Fit the length by dropping the ending; if still too long (or too short), the
        # caller retries with other blocks. (Dropping earlier turns would shift the turn
        # numbers that deadline anchors refer to.)
        while b.timeline()[0] > MAX_SEC and b.turns[-1].get("role") == "followup":
            b.turns = b.turns[:-2]

    elif kind == "describe_options":
        opt = P.draw("option_sets")
        b.block_ids.append(opt["id"])
        meta["topics"] = [opt["topic"]]
        b.add("user", opt["lead_in"], first_gap, block=opt["id"])
        ph = P.draw("describe_instruction")
        b.phrasing_ids.append(ph["id"])
        b.add("user", ph["text"], round(rng.uniform(0.4, 0.9), 2), event="instruction",
              kind="describe_options", phrasing=ph["id"])
        ack(len(b.turns) - 1)
        for k, o in enumerate(opt["options"]):
            g = round(rng.uniform(1.0, 3.0), 2) if k else round(rng.uniform(0.6, 1.5), 2)
            if k and g > 1.5:  # a long pause after the previous option: a pause distractor
                b.turns[-1].update(event="distractor", kind="pause")
            b.add("user", o, g, block=opt["id"])
        cue = P.draw("describe_cue")
        b.phrasing_ids.append(cue["id"])
        b.add("user", cue["text"], round(rng.uniform(0.5, 1.5), 2), event="return_cue", phrasing=cue["id"])
        b.add("moshi", rng.choice(opt["moshi_comment"]), role="resume", block=opt["id"])
        if b.timeline()[0] < MIN_SEC:
            raise ValueError("too short")

    total, speech = b.timeline()
    if not MIN_SEC <= total <= MAX_SEC:
        raise ValueError(f"length {total}")
    return {
        **meta, "topic": meta["topics"][0],
        "voices": {"user": user_v["id"], "friend": friend_v["id"] if kind == "side_conversation" else None,
                   "moshi": P.moshi_voice},
        "friend_name": friend["text"] if kind == "side_conversation" else None,
        "friend_name_id": friend["id"] if kind == "side_conversation" else None,
        "phrasing_ids": b.phrasing_ids, "block_ids": b.block_ids,
        "est_total_sec": total, "est_speech_sec": speech, "turns": b.turns,
    }


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
    # Eval: exactly EXTENSION_P of the think-time scripts (at least one) get an extension.
    think_idx = [i for i, k in enumerate(kinds) if k == "think_time"]
    n_ext = max(1, round(EXTENSION_P * len(think_idx))) if args.split == "eval" and think_idx else 0
    ext_idx = set(rng.sample(think_idx, n_ext))

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, kind in enumerate(kinds):
        for attempt in range(50):
            try:
                s = build(kind, args.split, P, rng, extension=i in ext_idx)
                break
            except ValueError:
                continue
        else:
            raise SystemExit(f"could not build a {kind} script within {MIN_SEC}-{MAX_SEC} s")
        sid = f"{args.split}_{args.seed:02d}_{i:05d}"
        script = {"id": sid, "split": args.split, "seed": args.seed, **s}
        if args.split == "eval":
            script["note"] = ("Moshi turns are reference_only: the ideal replies, for eval_full and "
                              "loss evaluation. Never render them into the model's input.")
        (args.out / f"{sid}.json").write_text(json.dumps(script, indent=1, ensure_ascii=False) + "\n")
        rows.append({"id": sid, "type": kind, "topic": s["topic"], "est_total_sec": s["est_total_sec"],
                     "est_speech_sec": s["est_speech_sec"], "wait_sec": s.get("wait_sec"),
                     "extension": bool(s.get("extension"))})
    (args.out / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    tot = sum(r["est_total_sec"] for r in rows)
    sp = sum(r["est_speech_sec"] for r in rows)
    print(f"{len(rows)} {args.split} scripts -> {args.out}: {dict(Counter(r['type'] for r in rows))}")
    print(f"  estimated {tot / 60:.1f} min total, {sp / 60:.1f} min speech; "
          f"length {min(r['est_total_sec'] for r in rows):.0f}-{max(r['est_total_sec'] for r in rows):.0f} s")
    uses = Counter(b for f in args.out.glob(f"{args.split}_*.json")
                   for b in json.loads(f.read_text())["block_ids"])
    print(f"  max uses of one building block: {max(uses.values()) if uses else 0}")


if __name__ == "__main__":
    main()
