"""Data-loader sanity check: load training files through the training pipeline exactly as a
batch (train.py's Mimi + Interleaver + InterleavedTokenizer + build_data_loader, duration_sec
from the config) and print, per 80 ms step, Moshi's text stream (word pieces or padding) next to
the scripted events. Confirms that every step of the wait (instruction end + ack grace to the
end of the wait) is padding, and that the steps after each distractor (other-name questions
included) up to Moshi's next turn are padding too. No model weights are loaded (Mimi and the
text tokenizer only), nothing is trained.

Usage: python scripts/check_dataloader.py [--index data/indexes/train_10h_v2.jsonl] [--ids ID ...]
       (default: the first think_time, side_conversation and ordinary file of the index)
Writes results/dataloader_check_v2.md.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finetune.data.args import DataArgs  # noqa: E402
from finetune.data.data_loader import build_data_loader  # noqa: E402
from finetune.data.interleaver import InterleavedTokenizer, Interleaver  # noqa: E402

PAD, EPAD, ZERO = 3, 0, -1   # moshi LMModel: existing_text_padding_id, existing_text_end_padding_id, zero_token_id
ACK_GRACE = 2.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=Path, default=ROOT / "data/indexes/train_10h_v2.jsonl")
    p.add_argument("--ids", nargs="*", default=None)
    p.add_argument("--config", type=Path, default=ROOT / "configs/smoke.yaml")
    p.add_argument("--out", type=Path, default=ROOT / "results/dataloader_check_v2.md")
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    rows = [json.loads(l) for l in args.index.read_text().splitlines() if l.strip()]
    for r in rows:
        r["ev"] = json.loads((args.index.parent / r["events"]).read_text())
    if args.ids:
        pick = [next(r for r in rows if Path(r["path"]).stem == i) for i in args.ids]
    else:
        pick = [next(r for r in rows if r["ev"]["type"] == t) for t in ("think_time", "side_conversation")]
        pick.append(next((r for r in rows if r["ev"]["type"] == "ordinary" and "other_name" in r["ev"].get("distractor_kinds", [])),
                         next(r for r in rows if r["ev"]["type"] == "ordinary")))
    tmp = Path(tempfile.mkdtemp()) / "check.jsonl"
    tmp.write_text("".join(json.dumps({"path": str((args.index.parent / r["path"]).resolve()), "duration": r["duration"]}) + "\n"
                           for r in pick))

    from moshi.models import loaders
    info = loaders.CheckpointInfo.from_hf_repo(hf_repo=cfg["moshi_paths"]["hf_repo_id"])
    mimi = info.get_mimi(device="cuda")
    mimi.eval()
    spm = info.get_text_tokenizer()
    interleaver = Interleaver(spm, mimi.frame_rate, PAD, EPAD, ZERO, keep_main_only=True)
    tok = InterleavedTokenizer(mimi, interleaver, duration_sec=cfg["duration_sec"])
    loader = build_data_loader(instruct_tokenizer=tok, args=DataArgs(train_data=str(tmp), eval_data=str(tmp)),
                               batch_size=len(pick), seed=0, rank=0, world_size=1, is_eval=True)
    batch = next(loader)
    codes = batch.codes  # (B, 1 + audio codebooks, T): row 0 = Moshi's text stream
    fps = mimi.frame_rate
    out = [f"# Data-loader check (v2)", "",
           f"Files loaded through the training pipeline as one batch (`build_data_loader`, Mimi, Interleaver, "
           f"duration_sec {cfg['duration_sec']} from {args.config.name}): batch codes shape {tuple(codes.shape)} "
           f"(batch, text + audio streams, steps at {fps:.1f} Hz = {1000 / fps:.0f} ms). Text stream per step: "
           f"`.` = padding ({PAD}), `,` = end-of-padding ({EPAD}, the step before a word), `_` = after the file end; "
           "word pieces are printed. Only Mimi and the text tokenizer are loaded; nothing is trained.", ""]
    all_ok = True
    for b, r in enumerate(pick):
        ev = r["ev"]
        text = codes[b, 0].tolist()
        T = len(text)
        name = Path(r["path"]).stem
        checks = []

        def steps(a: float, z: float) -> range:
            return range(max(0, int(round(a * fps))), min(T, int(round(z * fps))))

        def is_pad(i: int) -> bool:
            return text[i] in (PAD, EPAD, ZERO)
        if ev.get("instruction_end") is not None:
            acks = [t for t in ev["moshi_turns"] if t["kind"] == "ack"]
            lo = ev["instruction_end"] + (ACK_GRACE if acks else 0.0)
            lo = max([lo] + [t["end"] for t in acks])
            st = steps(lo, ev["wait_end"])
            bad = [i for i in st if not is_pad(i)]
            checks.append(f"wait {lo:.2f}-{ev['wait_end']:.2f} s ({len(st)} steps): "
                          + ("**all padding**" if not bad else f"**{len(bad)} non-padding steps** {bad[:5]}"))
            all_ok &= not bad
            st = steps(ev["instruction_start"], ev["instruction_end"])
            bad = [i for i in st if not is_pad(i)]
            checks.append(f"instruction {ev['instruction_start']:.2f}-{ev['instruction_end']:.2f} s: "
                          + ("all padding" if not bad else f"{len(bad)} non-padding steps"))
            all_ok &= not bad
        mo = sorted(t["start"] for t in ev["moshi_turns"])
        for d, k, u in zip(ev["distractors"], ev["distractor_kinds"], ev.get("distractor_until") or []):
            z = min(u if u is not None else d + 3.0, d + 3.0)
            st = steps(d, z)
            bad = [i for i in st if not is_pad(i)]
            checks.append(f"distractor {k} ending {d:.2f} s, window to {z:.2f} s: " + ("all padding" if not bad else
                                                                                     f"**{len(bad)} non-padding**"))
            all_ok &= not bad
        # every word step lies inside a Moshi turn
        inside = lambda i: any(t["start"] - 0.2 <= i / fps <= t["end"] + 0.1 for t in ev["moshi_turns"])  # noqa: E731
        stray = [i for i in range(T) if not is_pad(i) and not inside(i)]
        checks.append(f"text outside Moshi's turns: {len(stray)} steps")
        all_ok &= not stray
        out += [f"## {name} ({ev['type']}, {r['duration']:.1f} s, conditions {', '.join(ev['conditions']) or 'none'})", ""]
        out += [f"- {c}" for c in checks] + ["", "```"]
        # timeline: one line per scripted event / turn, with the text steps in that span
        marks = []
        for t in ev["right_turns"]:
            marks.append((t["start"], t["end"], f"RIGHT {t['speaker']}" + (f" [{t.get('event')}:{t.get('kind', '')}]" if t.get("event") else "")))
        for t in ev["moshi_turns"]:
            marks.append((t["start"], t["end"], f"MOSHI {t['kind']}"))
        if ev.get("instruction_end") is not None:
            marks.append((ev["instruction_end"], ev["wait_end"], "== WAIT (instruction end -> wait end)"))
        marks.sort()
        prev = 0.0
        for a, z, lab in marks:
            if a > prev + 0.25:  # a gap with no scripted speech
                seg = text[int(round(prev * fps)):int(round(a * fps))]
                out.append(f"{prev:6.2f}-{a:6.2f}  (no speech){'':22s} | {render(seg, spm)}")
            seg = text[int(round(a * fps)):int(round(z * fps))]
            out.append(f"{a:6.2f}-{z:6.2f}  {lab:34s} | {render(seg, spm)}")
            prev = max(prev, z)
        out.append(f"{prev:6.2f}-{T / fps:6.2f}  (rest, incl. padding to duration_sec) | "
                   f"{render(text[int(round(prev * fps)):], spm)}")
        out += ["```", ""]
    out.insert(3, f"**Result: {'all checks pass' if all_ok else 'PROBLEMS, see below'}.**\n")
    args.out.write_text("\n".join(out) + "\n")
    print("\n".join(out))
    sys.exit(0 if all_ok else 1)


def render(seg: list[int], spm) -> str:
    """Compact text stream: runs of padding as dots (one per step, long runs as '.xN'), words decoded."""
    s, run = [], 0
    for x in seg + [None]:
        if x in (PAD, EPAD, ZERO):
            run += 1
            continue
        if run:
            s.append("." * run if run <= 12 else f".x{run}")
            run = 0
        if x is not None:
            s.append(spm.id_to_piece(int(x)).replace("▁", " ").strip() or "_")
    return " ".join(s)


if __name__ == "__main__":
    main()
