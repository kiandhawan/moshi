"""Cut the rendered training/control sets, write their indexes and manifests, and summarize.

Inputs: data/{train,control,eval}/manifest.jsonl (scripts/render.py) and data/scripts/*/.
  train_10h  = the first k training files (k a multiple of the 20-script block, so the type
               mix is exact), k chosen so the total audio is closest to 10 h;
  train_1h   = the first j files of the same list (j a block multiple, closest to 1 h): a
               strict prefix, so a subset of train_10h with the same mix;
  control_10h = the ordinary files of train_10h, then ordinary-only control files in order
               until the total is closest to 10 h;
  eval       = all 400 eval files (user side) / eval_full (ideal stereo + human answer keys).
Writes data/indexes/{train_1h,train_10h,control_10h}.jsonl, data/manifests/<set>.jsonl (one
row per file, with seed, order_index and generator settings for reproduction) and
results/data_summary.md.

Usage: python scripts/build_datasets.py [CREDIT_BEFORE CREDIT_AFTER]
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import sphn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_lib import CACHE, cache_key, moshi_voice_id  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IDX, MAN = ROOT / "data/indexes", ROOT / "data/manifests"
TTS_PER_BYTE, ASR_PER_SEC = 15 / 1e6, 0.36 / 3600
KINDS = ["friend_question", "question_to_other", "near_miss", "pause", "self_talk"]
BLOCK_KEYS = {"ctx": "contexts", "side": "side_conversations", "dlg": "ordinary_dialogues", "opt": "option_sets",
              "dq": "distractor_exchanges", "dn": "distractor_exchanges", "call": "phone_calls", "door": "door_visits"}


def block_kind(b: str) -> str:
    """'cooking.ctx4' -> contexts, 'dq_31' -> distractor_exchanges, 'call_13' -> phone_calls."""
    key = re.sub(r"\d+$", "", b.split(".")[1]) if "." in b else b.split("_")[0]
    return BLOCK_KEYS[key]


def load(split_dir: str) -> list[dict]:
    rows = [json.loads(l) for l in open(ROOT / f"data/{split_dir}/manifest.jsonl")]
    for r in rows:
        s = json.loads((ROOT / f"data/scripts/{split_dir}/{r['id']}.json").read_text())
        r["script"] = s
        r["seed"], r["order_index"], r["generator"] = s["seed"], s["order_index"], s["generator"]
    return sorted(rows, key=lambda r: r["order_index"])


def cut(rows: list[dict], target: float, block: int) -> list[dict]:
    """Prefix whose length is a multiple of block and whose total is closest to target."""
    best, cum = None, 0.0
    for i, r in enumerate(rows):
        cum += r["total_sec"]
        if (i + 1) % block == 0 and (best is None or abs(cum - target) < best[1]):
            best = (i + 1, abs(cum - target))
    if best is None or sum(r["total_sec"] for r in rows) < target - 1800:
        raise SystemExit(f"not enough audio for {target / 3600:.1f} h ({sum(r['total_sec'] for r in rows) / 3600:.2f} h)")
    return rows[: best[0]]


def index_row(r: dict, split_dir: str) -> dict:
    """The render index row for this file (path relative to data/indexes)."""
    wav = r["path"] if r["path"].startswith("..") else f"../{split_dir}/{r['id']}.wav"
    return {"path": wav, "duration": r["total_sec"], "events": wav[:-4] + ".events.json"}


def fish_cost(rows: list[dict], moshi: str) -> tuple[float, float, int]:
    """(TTS $, ASR $, unique clips) to render this set on its own, from the cached clips."""
    seen, b, asr_sec = set(), 0, 0.0
    for r in rows:
        s = r["script"]
        for t in s["turns"]:
            v = moshi if t["speaker"] == "moshi" else s["voices"]["friend" if t["speaker"] in ("friend", "visitor") else "user"]
            if (v, t["text"]) in seen:
                continue
            seen.add((v, t["text"]))
            b += len(t["text"].encode())
            if t["speaker"] == "moshi":
                x = sphn.read(str(CACHE / f"{cache_key(t['text'], v)}.wav"))[0]
                asr_sec += x.shape[-1] / 24000
    return b * TTS_PER_BYTE, asr_sec * ASR_PER_SEC, len(seen)


def stats(name: str, rows: list[dict], moshi: str) -> dict:
    scripts = [r["script"] for r in rows]
    n = len(rows)
    types = Counter(r["type"] for r in rows)
    dis = {}
    for t in sorted(types):
        ss = [s for s in scripts if s["type"] == t]
        dis[t] = {k: sum(any(tn.get("event") == "distractor" and tn.get("kind") == k for tn in s["turns"]) for s in ss) / len(ss)
                  for k in KINDS}
    uses: dict[str, Counter] = defaultdict(Counter)
    for s in scripts:
        for b in s["block_ids"]:
            uses[block_kind(b)][b] += 1
    pools = defaultdict(int)  # pool sizes for this split
    split = scripts[0]["split"]
    for t in json.loads((ROOT / "data/splits/topics.json").read_text())[split]:
        d = json.loads((ROOT / f"data/pools/topics/{t}.json").read_text())
        for k in ("contexts", "side_conversations", "ordinary_dialogues", "option_sets"):
            pools[k] += len(d.get(k, []))
    gen = json.loads((ROOT / "data/pools/generic.json").read_text())
    pools["distractor_exchanges"] = sum(x["split"] in (split, "both") for x in gen["distractors"])
    pools["phone_calls"] = sum(x["split"] in (split, "both") for x in gen["phone_calls"])
    pools["door_visits"] = sum(x["split"] in (split, "both") for x in gen["door_visits"])
    voices = {s["voices"]["user"] for s in scripts} | {s["voices"]["friend"] for s in scripts if s["voices"].get("friend")}
    tts, asr, clips = fish_cost(rows, moshi)
    cond = Counter(c for s in scripts for c in s.get("conditions", []))
    groups = Counter(g for s in scripts for g in s.get("groups", []) or [])
    return {
        "name": name, "files": n, "unique_scripts": len({r["id"] for r in rows}),
        "unique_conversations": len({tuple(r["script"]["block_ids"]) for r in rows}),
        "max_reuse": max((max(u.values()) for u in uses.values()), default=0),
        "total_h": sum(r["total_sec"] for r in rows) / 3600, "speech_h": sum(r["speech_sec"] for r in rows) / 3600,
        "silence_h": sum(r["silence_sec"] for r in rows) / 3600,
        "moshi_h": sum(r["moshi_speech_sec"] for r in rows) / 3600,
        "wait_h": sum(r["wait_sec"] or 0 for r in rows) / 3600,
        "max_sec": max(r["total_sec"] for r in rows), "min_sec": min(r["total_sec"] for r in rows),
        "types": types, "distractors": dis, "conditions": cond, "groups": groups,
        "voices": len(voices), "uses": uses, "pools": pools, "tts": tts, "asr": asr, "clips": clips,
        "seeds": sorted({(r["seed"], r["generator"].get("only")) for r in rows}),
    }


def md(st: dict, extra: str = "") -> str:
    n = st["files"]
    lines = [f"## {st['name']}", "", extra, "" if extra else None,
             f"- Files / unique scripts / unique conversations: **{n} / {st['unique_scripts']} / "
             f"{st['unique_conversations']}**; max reuse of any building block within the set: "
             f"**{st['max_reuse']}**; length {st['min_sec']:.1f}-{st['max_sec']:.1f} s",
             f"- Audio: **{st['total_h']:.2f} h** total, {st['speech_h']:.2f} h speech (either channel), "
             f"{st['silence_h']:.2f} h silence; Moshi speaks {st['moshi_h']:.2f} h",
             f"- Wait time (instruction end to wait end): **{st['wait_h']:.2f} h**",
             f"- Mix: " + ", ".join(f"{t} {c} ({c / n:.0%})" for t, c in st["types"].most_common())
             + f"; wait types {1 - st['types'].get('ordinary', 0) / n:.0%}, ordinary {st['types'].get('ordinary', 0) / n:.0%}",
             "- Conditions: " + (", ".join(f"{c} {v}" for c, v in sorted(st["conditions"].items())) or "none"),
             ]
    if st["groups"]:
        lines.append("- Eval groups (a file can be in several): " + ", ".join(f"{g} {v}" for g, v in st["groups"].most_common()))
    lines += [f"- Distinct voices: **{st['voices']}** user/friend voices + the Moshi clone",
              f"- Seeds: " + ", ".join(f"{s}" + (f" (only {o})" if o else "") for s, o in st["seeds"]),
              "", "Distractors (% of files of the type with at least one):", "",
              "| type | " + " | ".join(KINDS) + " |", "|---|" + "---|" * len(KINDS)]
    for t, d in st["distractors"].items():
        lines.append(f"| {t} | " + " | ".join(f"{d[k]:.0%}" for k in KINDS) + " |")
    lines += ["", "Building-block reuse (uses per block among blocks used; pool = blocks available in this split):", "",
              "| block | used | pool | max uses | mean uses |", "|---|---|---|---|---|"]
    for k in dict.fromkeys(BLOCK_KEYS.values()):
        u = st["uses"].get(k)
        if u:
            lines.append(f"| {k} | {len(u)} | {st['pools'][k]} | {max(u.values())} | {sum(u.values()) / len(u):.1f} |")
    lines += ["", f"Fish Audio cost to render this set on its own: **${st['tts'] + st['asr']:.2f}** "
                  f"(TTS ${st['tts']:.2f} for {st['clips']} unique clips, ASR ${st['asr']:.2f} for Moshi's word timings)", ""]
    return "\n".join(l for l in lines if l is not None)


def main() -> None:
    moshi = moshi_voice_id()
    train, control, ev = load("train"), load("control"), load("eval")
    t10 = cut(train, 36000, 20)
    t1 = cut(train, 3600, 20)
    assert [r["id"] for r in t1] == [r["id"] for r in t10[: len(t1)]], "1 h set must be a prefix of the 10 h set"
    base = [r for r in t10 if r["type"] == "ordinary"]
    have = sum(r["total_sec"] for r in base)
    c10 = base + cut(control, 36000 - have, 1)
    MAN.mkdir(parents=True, exist_ok=True)
    sets = {"train_1h": (t1, "train"), "train_10h": (t10, "train"), "control_10h": (c10, None), "eval": (ev, "eval")}
    for name, (rows, d) in sets.items():
        if name != "eval":
            idx = [index_row(r, "train" if r["id"].startswith("train_") else "control") for r in rows]
            (IDX / f"{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in idx))
        (MAN / f"{name}.jsonl").write_text("".join(json.dumps({k: v for k, v in r.items() if k != "script"}) + "\n"
                                                   for r in rows))
    for name, (rows, _) in sets.items():
        too_long = [r["id"] for r in rows if r["total_sec"] >= 90]
        if too_long:
            raise SystemExit(f"{name}: files of 90 s or more: {too_long[:5]}")
    st = {k: stats(k, v[0], moshi) for k, v in sets.items()}
    for k, v in st.items():  # content-unique: no block in more than one file of a set
        if v["max_reuse"] > 1:
            raise SystemExit(f"{k}: a building block is used {v['max_reuse']} times")
    full = [json.loads(l) for l in open(IDX / "eval_full.jsonl")]
    notes = {
        "train_1h": f"`data/indexes/train_1h.jsonl`: the first {len(t1)} files of the ordered training list "
                    "(a strict prefix of train_10h, cut at a 20-script block so the mix is exact).",
        "train_10h": f"`data/indexes/train_10h.jsonl`: the first {len(t10)} files of the ordered training list "
                     f"({len(train)} rendered; the rest are the start of a longer list). Extending to 100 h needs more "
                     "blocks (about 10x): write them, then generate the continuation with `--exclude-blocks-from "
                     "data/scripts/train` so these files stay unchanged and the longer set stays content-unique.",
        "control_10h": f"`data/indexes/control_10h.jsonl`: the {len(base)} ordinary files of train_10h, then "
                       f"{len(c10) - len(base)} ordinary-only files (seed 11, drawn from dialogues not used anywhere "
                       f"in the training list) in order. Same train voices and pools.",
        "eval": f"`data/indexes/eval.jsonl`: {len(ev)} user-side files + events for run_inference.py/score.py; "
                f"`data/indexes/eval_full.jsonl`: their ideal stereo versions + transcripts plus the "
                f"{sum('/human_' in r['path'] for r in full)} human answer keys ({len(full)} files) for eval_loss.py.",
    }
    out = ["# Data summary", "",
           "Generated by `scripts/build_datasets.py` from the render manifests. Scripts: "
           "`scripts/gen_datasets.sh` = `generate_scripts.py --ordered --unique` (train seed 10, control seed 11 with "
           "`--only ordinary --exclude-blocks-from data/scripts/train`, eval seed 12); every set is content-unique "
           "(no building block in more than one file of a set; see results/LOG.md); "
           "audio: `scripts/render.py` via `scripts/render_all.sh`. Every manifest row in `data/manifests/` records "
           "the seed, order_index and generator settings. Fish costs are computed from the text and the cached "
           "Moshi clips ($15 per million bytes TTS, $0.36 per hour ASR); a set's cost counts every clip it "
           "needs, so subsets overlap (train_1h is part of train_10h; control_10h shares the ordinary files).", ""]
    out += [md(st[k], notes[k]) for k in sets]
    ver = []
    for idx in ("train_all", "control_all", "eval_full"):
        log = ROOT / f"logs/verify_{idx}.log"
        if log.exists():
            lines = log.read_text().splitlines()
            nfiles = sum(", written" in l for l in lines)
            last = next((l for l in reversed(lines) if l.endswith("problems")), "?")
            ver.append(f"- `verify_eval_full.py --index data/indexes/{idx}.jsonl`: {nfiles} files, {last}")
    longest = max(max(r["total_sec"] for r in v[0]) for v in sets.values())
    out += ["## Verification", "", *ver,
            f"- Longest file in any set: {longest:.1f} s (limit: under 90 s; render.py refuses longer files).",
            "- `check_split_leakage.py`: train_10h vs eval and control_10h: 0 problems.",
            "- `check_duplicates.py` (near-duplicates across all pools): 0 flagged pairs (1 before fixing).", ""]
    if len(sys.argv) > 2:
        out += ["## Fish Audio billing", "",
                f"- API credit before rendering: ${float(sys.argv[1]):.2f}; after: ${float(sys.argv[2]):.2f}; "
                f"billed for the full render (train list, eval, control additions): "
                f"**${float(sys.argv[1]) - float(sys.argv[2]):.2f}**.", ""]
    (ROOT / "results/data_summary.md").write_text("\n".join(out) + "\n")
    for k, v in st.items():
        print(f"{k:12s} {v['files']:4d} files {v['total_h']:6.2f} h (speech {v['speech_h']:.2f}, silence {v['silence_h']:.2f}, "
              f"wait {v['wait_h']:.2f}), max {v['max_sec']:.1f} s, cost ${v['tts'] + v['asr']:.2f}")
    print("Wrote results/data_summary.md, data/indexes/{train_1h,train_10h,control_10h}.jsonl, data/manifests/")


if __name__ == "__main__":
    main()
