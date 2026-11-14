"""Cut the rendered v2 sets, write their indexes and manifests, and summarize (results/data_summary_v2.md).

Same cuts as scripts/build_datasets.py (v1, unchanged): train_10h_v2 = the block-aligned prefix
of the ordered v2 training list closest to 10 h; train_1h_v2 = the block-aligned prefix
closest to 1 h (a prefix of train_10h_v2); control_10h_v2 = the ordinary files of
train_10h_v2, then v2 control files in order until closest to 10 h; eval v2 = all 400 eval v2
files (+ the 3 human answer keys in eval_full_v2).
Inputs: data/v2/{train,control,eval}/manifest.jsonl and data/v2/scripts/*/.
Writes data/indexes/{train_1h_v2,train_10h_v2,control_10h_v2}.jsonl,
data/manifests/<set>_v2.jsonl and results/data_summary_v2.md, with the v2 statistics: duration
distribution, waits >= 30 s per type, instruction positions, instruction / return-cue styles
and phrasings per wait type, other-name / Moshi-by-name / Moshi-mention counts, augmentation mix.

Usage: python scripts/build_datasets_v2.py [CREDIT_BEFORE CREDIT_AFTER]
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_datasets as bd  # noqa: E402
from render_lib import CACHE, cache_key, moshi_voice_id  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IDX, MAN = ROOT / "data/indexes", ROOT / "data/manifests"
bd.BLOCK_KEYS.update({"nq": "name_questions", "mm": "moshi_mentions"})
bd.KINDS[:] = ["friend_question", "question_to_other", "near_miss", "pause", "self_talk", "friend_interrupt",
               "other_name", "moshi_mention"]
WAIT_TYPES = ["think_time", "side_conversation", "describe_options", "phone_call", "door", "reading"]


def voice(t: dict, s: dict, moshi: str) -> str:
    sp = t["speaker"]
    return moshi if sp == "moshi" else s["voices"]["other"] if sp == "other" else \
        s["voices"]["friend"] if sp in ("friend", "visitor") else s["voices"]["user"]


def fish_cost(rows: list[dict], moshi: str) -> tuple[float, float, int]:
    seen, b, asr_sec = set(), 0, 0.0
    for r in rows:
        s = r["script"]
        for t in s["turns"]:
            v = voice(t, s, moshi)
            if (v, t["text"]) in seen:
                continue
            seen.add((v, t["text"]))
            b += len(t["text"].encode())
            if t["speaker"] == "moshi":
                import sphn
                asr_sec += sphn.read(str(CACHE / f"{cache_key(t['text'], v)}.wav"))[0].shape[-1] / 24000
    return b * bd.TTS_PER_BYTE, asr_sec * bd.ASR_PER_SEC, len(seen)


bd.fish_cost = fish_cost


def load(d: str) -> list[dict]:
    rows = [json.loads(l) for l in open(ROOT / f"data/v2/{d}/manifest.jsonl")]
    for r in rows:
        s = json.loads((ROOT / f"data/v2/scripts/{d}/{r['id']}.json").read_text())
        r["script"] = s
        r["block_ids"] = s["block_ids"]  # from the script (name questions / mentions included)
        r["seed"], r["order_index"], r["generator"] = s["seed"], s["order_index"], s["generator"]
        ev_path = ROOT / "data/indexes" / (r["path"][:-4] + ".events.json") if d != "eval" else \
            ROOT / f"data/v2/eval_full/{r['id']}.events.json"
        r["_ev"] = json.loads(ev_path.read_text())
    return sorted(rows, key=lambda r: r["order_index"])


def realized_position(ev: dict) -> float | None:
    """Share of conversation speech (greeting exchange and the wait itself excluded) that comes
    before the instruction: speech ending before instruction_start vs. starting after wait_end."""
    if ev.get("instruction_start") is None:
        return None
    turns = [(t["start"], t["end"]) for t in ev["right_turns"] if t.get("event") != "greeting_answer"] + \
            [(t["start"], t["end"]) for t in ev.get("moshi_turns", []) if t["kind"] != "greeting"]
    before = sum(e - s for s, e in turns if e <= ev["instruction_start"] + 1e-6)
    after = sum(e - s for s, e in turns if s >= ev["wait_end"] - 1e-6)
    return before / max(before + after, 1e-6)


def pct(a, b):
    return f"{a / b:.0%}" if b else "-"


def v2_section(name: str, rows: list[dict]) -> str:
    n = len(rows)
    L = [f"### v2 details: {name}", ""]
    durs = np.array([r["total_sec"] for r in rows])
    edges = [30, 40, 50, 60, 70, 80, 90]
    h = np.histogram(durs, bins=edges)[0]
    L += ["File length (s): " + ", ".join(f"{a}-{b}: {c}" for a, b, c in zip(edges, edges[1:], h))
          + f" (median {np.median(durs):.1f} s)", ""]
    waits = [r for r in rows if r["type"] in WAIT_TYPES]
    if waits:
        L += ["| wait type | files | wait s: min / median / max | >= 30 s | distinct durations (think) | "
              "position planned e/m/l | realized share before (median) | realized e/m/l |", "|---|---|---|---|---|---|---|---|"]
        for t in WAIT_TYPES:
            rr = [r for r in waits if r["type"] == t]
            if not rr:
                continue
            w = np.array([r["wait_sec"] for r in rr])
            plan = Counter(r.get("position_plan") for r in rr)
            rp = [realized_position(r["_ev"]) for r in rr]
            lab = Counter("early" if x < 1 / 3 else "middle" if x < 2 / 3 else "late" for x in rp)
            think = len({r["script"].get("wait_sec") for r in rr}) if t == "think_time" else "-"
            L.append(f"| {t} | {len(rr)} | {w.min():.1f} / {np.median(w):.1f} / {w.max():.1f} | "
                     f"**{pct((w >= 30).sum(), len(rr))}** | {think} | "
                     f"{plan.get('early', 0)}/{plan.get('middle', 0)}/{plan.get('late', 0)} | {np.median(rp):.2f} | "
                     f"{lab.get('early', 0)}/{lab.get('middle', 0)}/{lab.get('late', 0)} |")
        allw = np.array([r["wait_sec"] for r in waits])
        L += ["", f"All waits: {pct((allw >= 30).sum(), len(allw))} last 30 s or longer. Positions: planned = the "
              "generator's structure (early: the wait right after the context, a dialogue after; middle: one exchange "
              "before, one after; late: a whole dialogue before); realized = share of the conversation's speech "
              "(greeting exchange and the wait excluded) before the instruction, from the rendered timeline; "
              "e/m/l = under 1/3, 1/3-2/3, over 2/3.", ""]
        if name.startswith("eval"):  # eval keeps the v1 design: no planned positions or style tags
            L[-1] += (" Eval v2 keeps the v1 eval design (instruction right after the context, v1 phrasings "
                      "without style tags), so the planned positions and the style table don't apply.")
            L += [""]
        # instruction and return-cue phrasings
        L += [] if name.startswith("eval") else ["| wait type | instruction phrasings | instruction styles | tones | states how to return | "
              "return-cue phrasings | cue styles | friend interrupts first |", "|---|---|---|---|---|---|---|---|"]
        for t in (WAIT_TYPES if not name.startswith("eval") else []):
            rr = [r for r in waits if r["type"] == t]
            if not rr:
                continue
            ins, cues = [], []
            for r in rr:
                s = r["script"]
                st = s.get("phrasing_styles") or {}
                for tn in s["turns"]:
                    if tn.get("event") == "instruction":
                        ins.append((tn.get("phrasing"), st.get(tn.get("phrasing"), {})))
                    if tn.get("event") == "return_cue":
                        cues.append((tn.get("phrasing"), st.get(tn.get("phrasing"), {})))
            sty = Counter(x[1].get("style", "?") for x in ins)
            L.append(f"| {t} | {len({x[0] for x in ins})} | " + ", ".join(f"{k} {v}" for k, v in sty.most_common())
                     + f" | {len({x[1].get('tone') for x in ins})} | {pct(sum(bool(x[1].get('states_return')) for x in ins), len(ins))} | "
                     f"{len({x[0] for x in cues})} | {len({x[1].get('style') for x in cues if x[1]})} | "
                     f"{sum('friend_first' in r['conditions'] for r in rr)} |")
        L += [""]
    # name addressing
    add = [(r, a) for r in rows for a in r.get("addressed") or []]
    if add:
        L += ["Addressing by name (exchanges; files in brackets):", ""]
        for kind, label in (("other_name", "other-name questions (Moshi silent)"),
                            ("moshi_name", "Moshi addressed by name (Moshi answers)"),
                            ("moshi_mention", "Moshi mentioned in the third person (Moshi silent)")):
            xs = [(r, a) for r, a in add if a["kind"] == kind]
            if not xs:
                continue
            files = {r["id"] for r, _ in xs}
            ordf = {r["id"] for r, _ in xs if r["type"] == "ordinary"}
            pos = Counter(a.get("name_pos") for _, a in xs)
            L.append(f"- {label}: {len(xs)} ({len(files)} files: {len(ordf)} ordinary = "
                     f"{pct(len(ordf), sum(r['type'] == 'ordinary' for r in rows))} of ordinary, "
                     f"{len(files) - len(ordf)} wait conversations); name position "
                     + ", ".join(f"{k} {v}" for k, v in pos.most_common())
                     + (f"; answered {pct(sum(a['answered'] for _, a in xs), len(xs))}; "
                        f"{len({a.get('addressee') for _, a in xs if a.get('addressee')})} distinct names"
                        if kind != "moshi_name" else ""))
        L += [""]
    # augmentation
    aug = [r.get("augmentation") for r in rows if r.get("augmentation")]
    if aug:
        nt = Counter(a["noise"]["type"] for a in aug)
        lv = {k: [a["noise"]["dbfs"] for a in aug if a["noise"]["type"] == k] for k in nt}
        mic = Counter(a["mic"]["type"] if a["mic"] else "none" for a in aug)
        rv = sum(bool(a["user_reverb"]) for a in aug)
        gu = [a["gain_db"]["user"] for a in aug]
        heavy = sum(heaviness(a) >= 2 for a in aug)
        L += ["Augmentation (right channel only; Moshi's channel is clean):", "",
              "- Background noise: " + ", ".join(f"{k} {v} ({pct(v, len(aug))}, {min(lv[k]):.0f} to {max(lv[k]):.0f} dBFS)"
                                                 for k, v in nt.most_common()),
              "- Mic coloring: " + ", ".join(f"{k} {v} ({pct(v, len(aug))})" for k, v in mic.most_common()),
              f"- Light reverb on the user: {rv} ({pct(rv, len(aug))})",
              f"- Per-speaker gain: user {min(gu):+.1f} to {max(gu):+.1f} dB (friend and other person drawn separately, "
              f"each uniform in +-6 dB)",
              f"- Heavy (at least two of: loud noise >= -42 dBFS, mic coloring, user reverb): {heavy} ({pct(heavy, len(aug))})", ""]
    return "\n".join(L)


def heaviness(a: dict) -> int:
    return int(a["noise"]["dbfs"] >= -42 and a["noise"]["type"] != "quiet_room") + int(bool(a["mic"])) + \
        int(bool(a["user_reverb"]))


def main() -> None:
    moshi = moshi_voice_id()
    train, control, ev = load("train"), load("control"), load("eval")
    t10 = bd.cut(train, 36000, 20)
    t1 = bd.cut(train, 3600, 20)
    assert [r["id"] for r in t1] == [r["id"] for r in t10[: len(t1)]], "1 h set must be a prefix of the 10 h set"
    base = [r for r in t10 if r["type"] == "ordinary"]
    c10 = base + bd.cut(control, 36000 - sum(r["total_sec"] for r in base), 1)
    sets = {"train_1h_v2": t1, "train_10h_v2": t10, "control_10h_v2": c10, "eval_v2": ev}
    for name, rows in sets.items():
        if name != "eval_v2":
            idx = [{"path": r["path"], "duration": r["total_sec"], "events": r["path"][:-4] + ".events.json"} for r in rows]
            (IDX / f"{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in idx))
        (MAN / f"{name}.jsonl").write_text("".join(json.dumps({k: v for k, v in r.items() if k not in ("script", "_ev")}) + "\n"
                                                   for r in rows))
        too_long = [r["id"] for r in rows if r["total_sec"] >= 90]
        if too_long:
            raise SystemExit(f"{name}: files of 90 s or more: {too_long[:5]}")
    st = {}
    for k, rows in sets.items():
        st[k] = bd.stats(k, rows, moshi)
        split = rows[0]["script"]["split"]
        pools = st[k]["pools"]
        for f in sorted((ROOT / "data/pools/v2").glob("*.json")):
            d = json.loads(f.read_text())
            for kk in ("name_questions", "moshi_mentions"):
                pools[kk] += sum(x["split"] == split for x in d.get(kk, []))
        if st[k]["max_reuse"] > 1:
            raise SystemExit(f"{k}: a building block is used {st[k]['max_reuse']} times")
    full = [json.loads(l) for l in open(IDX / "eval_full_v2.jsonl")]
    rep = json.loads((ROOT / "data/v2/scripts/eval/replaced.json").read_text())
    notes = {
        "train_1h_v2": f"`data/indexes/train_1h_v2.jsonl`: the first {len(t1)} files of the ordered v2 training list "
                       "(a strict prefix of train_10h_v2, cut at a 20-script block so the mix is exact).",
        "train_10h_v2": f"`data/indexes/train_10h_v2.jsonl`: the first {len(t10)} files of the ordered v2 training list "
                        f"({len(train)} rendered).",
        "control_10h_v2": f"`data/indexes/control_10h_v2.jsonl`: the {len(base)} ordinary files of train_10h_v2, then "
                          f"{len(c10) - len(base)} ordinary-only v2 files (seed 11, from dialogues and name questions not "
                          "used by any ordinary file of the training list) in order. Augmented like the training set.",
        "eval_v2": f"`data/indexes/eval_v2.jsonl`: {len(ev)} user-side files + events (the default eval set; final); "
                   f"`data/indexes/eval_full_v2.jsonl`: ideal stereo versions + the human answer keys ({len(full)} files); "
                   f"`data/indexes/eval_full_subset_v2.jsonl`: the 40-file training-time subset. {len(rep['added'])} "
                   f"v1 eval files were replaced by other_name files of the same types and conditions "
                   f"(`data/v2/scripts/eval/replaced.json`); the other {len(ev) - len(rep['added'])} are the v1 files. "
                   "No augmentation.",
    }
    out = ["# Data summary, v2", "",
           "Generated by `scripts/build_datasets_v2.py` from the v2 render manifests. v1 (`results/data_summary.md`, "
           "`data/{train,control,eval}`, `data/indexes/*.jsonl` without `_v2`) is unchanged, for reference. Scripts: "
           "`scripts/gen_datasets_v2.sh` (`generate_scripts_v2.py`: train seed 10, control seed 11, eval v2 seed 13); "
           "audio: `scripts/render_all_v2.sh` (train and control with right-channel augmentation). Every set is "
           "content-unique (no building block, name question or Moshi mention in more than one file of a set). "
           "Changes and reasons: results/LOG.md. Fish costs as in v1 (from the text and the cached clips; subsets overlap).", ""]
    for k, rows in sets.items():
        out += [bd.md(st[k], notes[k]).replace(f"## {k}", f"## {k}"), v2_section(k, rows), ""]
    ver = []
    for log in sorted((ROOT / "logs").glob("verify_*_v2*.log")):
        lines = log.read_text().splitlines()
        last = next((l for l in reversed(lines) if l.endswith("problems")), "?")
        head = next((l for l in lines if l.startswith("checked") or "augmented files checked" in l), None)
        nfiles = sum(", written" in l for l in lines)
        ver.append(f"- `{log.stem.removeprefix('verify_')}`: " + (f"{nfiles} files, " if nfiles else "")
                   + (f"{head}; " if head else "") + last)
    longest = max(max(r["total_sec"] for r in v) for v in sets.values())
    out += ["## Verification", "", *ver,
            f"- Longest file in any set: {longest:.1f} s (limit: under 90 s).", ""]
    if len(sys.argv) > 2:
        out += ["## Fish Audio billing", "",
                f"- API credit before the v2 render: ${float(sys.argv[1]):.2f}; after: ${float(sys.argv[2]):.2f}; "
                f"billed: **${float(sys.argv[1]) - float(sys.argv[2]):.2f}** (most lines came from the TTS cache: "
                "each v2 file keeps the v1 voice of its first block).", ""]
    (ROOT / "results/data_summary_v2.md").write_text("\n".join(out) + "\n")
    for k, v in st.items():
        print(f"{k:15s} {v['files']:4d} files {v['total_h']:6.2f} h (speech {v['speech_h']:.2f}, silence {v['silence_h']:.2f}, "
              f"wait {v['wait_h']:.2f}), max {v['max_sec']:.1f} s, max reuse {v['max_reuse']}")
    print("Wrote results/data_summary_v2.md, data/indexes/*_v2.jsonl, data/manifests/*_v2.jsonl")


if __name__ == "__main__":
    main()
