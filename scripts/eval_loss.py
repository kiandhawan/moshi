"""Teacher-forced (loss-based) evaluation of a checkpoint on the eval_full files.

Each eval_full file is a complete conversation with the *ideal* Moshi side on the left
channel and its transcript (scripts/render_lib.py). The model sees the whole file at once,
like a training step without the backward pass, and we read off how likely it finds the
ideal behaviour. This is cheap, deterministic (no sampling) and needs no VAD, so it can be
run on every checkpoint; the streaming evaluation (run_inference.py + score.py) remains the
measure of what the model actually does.

Reuses the training code, so the numbers are computed on exactly what training sees:
  - data:  finetune.data.dataset.get_dataset_iterator (sphn crops of duration_sec, Mimi
           encoding, Interleaver/InterleavedTokenizer built with the same arguments as
           train.py), so text/audio tokens are identical to a training batch;
  - model: the same LMModel from moshi's loaders with the run's LoRA config, LoRA layers
           not fused (as during training); no checkpoint = the base model, which is what
           training starts from (LoRA B matrices start at zero);
  - loss:  finetune.loss.compute_loss_with_mask with the run's text_padding_weight and
           first_codebook_weight_multiplier, as in train.py and finetune/eval.py.

Metrics (one CSV row per checkpoint, pooled over all frames of all files; a frame is one
80 ms step of the text stream, frame t covers [t, t+1) x 80 ms):
  val_loss, text_loss, audio_loss
      Mean over files of the training losses (finetune/eval.py averages over batches; here
      a batch is one file).
  greet_p_speak (greeted_at_start)
      Mean P(not PAD) over the frames before user_start (the user's first words), where
      Moshi should greet the user and ask how they are (CLAUDE.md). greet_target_speak is the
      same fraction in the ideal transcript (reference ceiling, as for cue_p_speak).
  wait_p_pad
      Mean P(PAD) over frames from instruction_end + ack_grace (2.0 s, the ack rule) to
      wait_end. PAD (id 3) is the "say nothing" token.
  cue_p_speak
      Mean P(not PAD) over the first --cue-sec (2.0) s after wait_end (the return cue or the
      deadline). Not-PAD includes EPAD (id 0), the token Moshi emits right before a word
      starts, so this is the probability of starting or continuing speech.
      cue_target_speak is the same fraction in the ideal transcript, a ceiling for
      reference: even while speaking most frames are PAD (one text token per word, words
      last several frames), so a perfect model sits well below 1.
  distractor_p_pad
      Mean P(PAD) over the --distractor-sec (3.0) s after each distractor.
  ordinary_text_loss
      Training text loss restricted to the frames of the ideal Moshi turns that answer an
      ordinary request (kind ordinary in the human answer keys, reply / followup in rendered
      files; plus the frame before each, where the EPAD goes), pooled over frames.
In the "should be silent" windows, frames where the ideal Moshi side is speaking (the
opening greeting, or an acknowledgement running past the 2 s mark; see moshi_turns) are left
out, so the greeting never counts as speaking during the instruction or as an interruption.

Usage:
  python scripts/eval_loss.py                                  # base model
  python scripts/eval_loss.py --checkpoint runs/X/checkpoints/checkpoint_000100/consolidated
Options: --data (default data/indexes/eval_full.jsonl), --label, --gpu, --config.
Writes results/metrics/loss_eval.csv (a row with the same label is replaced), one row per
(checkpoint, eval group) in results/metrics/loss_eval_groups.csv (groups from the events
files: familiar_type, new_type_phone/door/reading, extension, implicit_return, ordinary,
human; a file with several tags counts in each), and results/metrics/loss_eval/<label>.jsonl
(per-file numbers).
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # see run_inference.py
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # the training package `finetune`

ORDINARY_KINDS = {"ordinary", "reply", "followup"}  # Moshi turns that answer an ordinary request
COLUMNS = ["label", "checkpoint", "step", "files", "val_loss", "text_loss", "audio_loss",
           "greet_p_speak", "greet_target_speak", "greet_frames", "wait_p_pad", "wait_frames", "cue_p_speak", "cue_target_speak", "cue_frames",
           "distractor_p_pad", "distractor_frames", "ordinary_text_loss", "ordinary_frames",
           "date"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Folder with lora.safetensors + config.json. Omit for the base model.")
    p.add_argument("--config", type=Path, default=None,
                   help="Training config for loss weights/duration_sec. Default: the run's "
                        "args.yaml for a checkpoint, configs/smoke.yaml for the base model.")
    p.add_argument("--data", type=Path, default=REPO / "data/indexes/eval_full.jsonl")
    p.add_argument("--label", default=None, help="Row name (default: base, or <run>/step<N>).")
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--ack-grace", type=float, default=2.0)
    p.add_argument("--cue-sec", type=float, default=2.0)
    p.add_argument("--distractor-sec", type=float, default=3.0)
    p.add_argument("--out", type=Path, default=REPO / "results/metrics/loss_eval.csv")
    return p.parse_args()


def checkpoint_meta(ckpt: Path | None) -> tuple[Path | None, str, int | None]:
    """(run args.yaml, default label, step) for .../<run>/checkpoints/checkpoint_N/consolidated."""
    if ckpt is None:
        return None, "base", 0
    ckpt = ckpt.resolve()
    step_dir = ckpt.parent if ckpt.name == "consolidated" else ckpt
    run_dir = step_dir.parent.parent
    step = int(step_dir.name.split("_")[-1]) if step_dir.name.startswith("checkpoint_") else None
    return run_dir / "args.yaml", f"{run_dir.name}/step{step}", step


class RecordingTokenizer:
    """Wraps the training InterleavedTokenizer to remember which file/crop each Sample came
    from (the training iterator only yields the tokens)."""

    def __init__(self, tok):
        self.tok, self.duration_sec, self.mimi, self.last = tok, tok.duration_sec, tok.mimi, None

    def __call__(self, wav, start_sec, path):
        self.last = (path, float(start_sec))
        return self.tok(wav, start_sec, path)


def main() -> None:
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from moshi.models import loaders

    from finetune.args import TrainArgs
    from finetune.data.dataset import DataFile, get_dataset_iterator
    from finetune.data.interleaver import InterleavedTokenizer, Interleaver
    from finetune.loss import compute_loss_with_mask

    torch.set_grad_enabled(False)
    run_args, default_label, step = checkpoint_meta(args.checkpoint)
    config = args.config or (run_args if run_args and run_args.exists() else REPO / "configs/smoke.yaml")
    targs = TrainArgs.load(str(config), drop_extra_fields=False)
    label = args.label or default_label
    print(f"Label {label!r}, config {config}", flush=True)

    # ---- model: same loaders as train.py (get_fsdp_model), LoRA not fused
    lora = config_path = None
    if args.checkpoint is not None:
        lora, config_path = args.checkpoint / "lora.safetensors", args.checkpoint / "config.json"
    mp = targs.moshi_paths
    ci = loaders.CheckpointInfo.from_hf_repo(
        hf_repo=mp.hf_repo_id, moshi_weights=mp.moshi_path, mimi_weights=mp.mimi_path,
        tokenizer=mp.tokenizer_path, config_path=config_path or mp.config_path, lora_weights=lora)
    mimi = ci.get_mimi(device="cuda")
    mimi.eval()
    model = ci.get_moshi(device="cuda", dtype=getattr(torch, targs.param_dtype), fuse_lora=False)
    model.eval()
    n_lora = sum(p.numel() for n, p in model.named_parameters() if "lora" in n)
    print(f"Model loaded ({'LoRA params: %d' % n_lora if lora else 'base'})", flush=True)

    # ---- data: same Interleaver arguments and dataset iterator as train.py
    interleaver = Interleaver(ci.get_text_tokenizer(), mimi.frame_rate, model.text_padding_token_id,
                              model.end_of_text_padding_id, model.zero_token_id, keep_main_only=True)
    tok = RecordingTokenizer(InterleavedTokenizer(mimi, interleaver, duration_sec=targs.duration_sec))
    samples = get_dataset_iterator(DataFile(args.data), instruct_tokenizer=tok, rank=0, world_size=1,
                                   is_finite=True, seed=None, shuffle_at_epoch=False)

    events_by_wav = {}
    for line in args.data.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("events"):
                events_by_wav[str((args.data.parent / row["path"]).resolve())] = \
                    json.loads((args.data.parent / row["events"]).read_text())

    PAD, EPAD = model.text_padding_token_id, model.end_of_text_padding_id
    fps = mimi.frame_rate
    SUM_KEYS, CNT_KEYS = ("greet", "greet_t", "wait", "cue", "cue_t", "dis", "ord"), ("greet", "wait", "cue", "dis", "ord")
    files, per_file = 0, []
    t0 = time.perf_counter()
    for sample in samples:
        path, start = tok.last
        codes = sample.codes  # [1, 1 + 16, T]: text, Moshi audio (8), user audio (8)
        out = model(codes=codes, condition_tensors=None)
        text_target = codes[:, : model.audio_offset]
        loss_kw = dict(mode="text", text_padding_weight=targs.text_padding_weight,
                       text_padding_ids={PAD, EPAD})
        text_loss = compute_loss_with_mask(out.text_logits, text_target, out.text_mask, **loss_kw).item()
        audio_loss = compute_loss_with_mask(
            out.logits, codes[:, model.audio_offset: model.audio_offset + model.dep_q], out.mask,
            mode="audio", first_codebook_weight_multiplier=targs.first_codebook_weight_multiplier).item()
        files += 1
        rec = {"path": path, "crop_start": start, "text_loss": text_loss, "audio_loss": audio_loss}
        tot, cnt = {k: 0.0 for k in SUM_KEYS}, {k: 0 for k in CNT_KEYS}  # this file's sums / frame counts

        ev = events_by_wav.get(str(Path(path).resolve()))
        if ev is not None:
            T = codes.shape[-1]
            valid = out.text_mask[0, 0]  # real (non-padding-of-the-crop) frames
            p_pad = out.text_logits[0, 0].float().softmax(-1)[:, PAD]
            target = text_target[0, 0]

            def frames(a: float, b: float) -> torch.Tensor:
                """Bool mask of frames whose start lies in [a, b) seconds (file time)."""
                m = torch.zeros(T, dtype=torch.bool, device=valid.device)
                lo, hi = max(0, math.ceil((a - start) * fps)), min(T, math.ceil((b - start) * fps))
                if hi > lo:
                    m[lo:hi] = True
                return m & valid

            speaking = torch.zeros(T, dtype=torch.bool, device=valid.device)
            for turn in ev.get("moshi_turns", []):
                speaking |= frames(turn["start"] - 1 / fps, turn["end"])
            if ev.get("user_start") is not None:
                m = frames(0.0, ev["user_start"])
                tot["greet"] += (1 - p_pad[m]).sum().item(); cnt["greet"] += int(m.sum())
                tot["greet_t"] += (target[m] != PAD).sum().item()
                rec["greet_p_speak"] = (1 - p_pad[m]).mean().item() if m.any() else None
            if ev.get("instruction_end") is not None and ev.get("wait_end") is not None:
                m = frames(ev["instruction_end"] + args.ack_grace, ev["wait_end"]) & ~speaking
                tot["wait"] += p_pad[m].sum().item(); cnt["wait"] += int(m.sum())
                rec["wait_p_pad"] = p_pad[m].mean().item() if m.any() else None
                m = frames(ev["wait_end"], ev["wait_end"] + args.cue_sec)
                tot["cue"] += (1 - p_pad[m]).sum().item(); cnt["cue"] += int(m.sum())
                tot["cue_t"] += (target[m] != PAD).sum().item()
                rec["cue_p_speak"] = (1 - p_pad[m]).mean().item() if m.any() else None
            for d in ev.get("distractors", []):
                m = frames(d, d + args.distractor_sec) & ~speaking
                tot["dis"] += p_pad[m].sum().item(); cnt["dis"] += int(m.sum())
            m = torch.zeros(T, dtype=torch.bool, device=valid.device)
            for turn in ev.get("moshi_turns", []):
                if turn["kind"] in ORDINARY_KINDS:
                    m |= frames(turn["start"] - 1 / fps, turn["end"])
            if m.any():
                ol = compute_loss_with_mask(out.text_logits, text_target, out.text_mask & m[None, None],
                                            **loss_kw).item()
                tot["ord"] += ol * int(m.sum()); cnt["ord"] += int(m.sum())
                rec["ordinary_text_loss"] = ol
        rec.update(groups=(ev or {}).get("groups", []), sums=tot, counts=cnt)
        per_file.append(rec)
        print(f"  {Path(path).name}: text {text_loss:.3f} audio {audio_loss:.3f} "
              + " ".join(f"{k} {v:.3f}" for k, v in rec.items()
                         if k in ("greet_p_speak", "wait_p_pad", "cue_p_speak", "ordinary_text_loss") and v is not None),
              flush=True)
    if not files:
        sys.exit(f"No samples in {args.data}")

    def aggregate(recs: list[dict]) -> dict:
        """The metric columns for a set of files (frame-pooled, as described above)."""
        tot = {k: sum(r["sums"][k] for r in recs) for k in SUM_KEYS}
        cnt = {k: sum(r["counts"][k] for r in recs) for k in CNT_KEYS}
        mean = lambda k: sum(r[k] for r in recs) / len(recs)  # noqa: E731
        ratio = lambda a, n: round(a / n, 4) if n else None  # noqa: E731
        return {
            "files": len(recs),
            "val_loss": round(mean("text_loss") + mean("audio_loss"), 4),
            "text_loss": round(mean("text_loss"), 4), "audio_loss": round(mean("audio_loss"), 4),
            "greet_p_speak": ratio(tot["greet"], cnt["greet"]),
            "greet_target_speak": ratio(tot["greet_t"], cnt["greet"]), "greet_frames": cnt["greet"],
            "wait_p_pad": ratio(tot["wait"], cnt["wait"]), "wait_frames": cnt["wait"],
            "cue_p_speak": ratio(tot["cue"], cnt["cue"]), "cue_target_speak": ratio(tot["cue_t"], cnt["cue"]),
            "cue_frames": cnt["cue"],
            "distractor_p_pad": ratio(tot["dis"], cnt["dis"]), "distractor_frames": cnt["dis"],
            "ordinary_text_loss": ratio(tot["ord"], cnt["ord"]), "ordinary_frames": cnt["ord"],
        }

    head = {"label": label, "checkpoint": str(args.checkpoint) if args.checkpoint else "",
            "step": step if step is not None else ""}
    date = time.strftime("%Y-%m-%d %H:%M")
    row = {**head, **aggregate(per_file), "date": date}
    tags = sorted({t for r in per_file for t in r["groups"]})
    group_rows = [{**head, "eval_group": t, **aggregate([r for r in per_file if t in r["groups"]]), "date": date}
                  for t in tags]
    untagged = [r for r in per_file if not r["groups"]]
    if untagged:
        print(f"WARNING: {len(untagged)} files have no eval groups (events 'groups'), e.g. {untagged[0]['path']}")
    print(f"{files} files in {time.perf_counter() - t0:.1f}s")
    print(json.dumps(row, indent=1))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.out.exists():
        with args.out.open() as f:
            rows = [r for r in csv.DictReader(f) if r["label"] != label]
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows + [row])
    gout = args.out.with_name(args.out.stem + "_groups.csv")
    gcols = COLUMNS[:3] + ["eval_group"] + COLUMNS[3:]
    grows = []
    if gout.exists():
        with gout.open() as f:
            grows = [r for r in csv.DictReader(f) if r["label"] != label]
    with gout.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=gcols)
        w.writeheader()
        w.writerows(grows + group_rows)
    print(f"{'eval group':18s} {'files':>5s} {'val_loss':>8s} {'greet':>6s} {'wait':>6s} {'cue':>6s} {'distr':>6s} {'ord':>6s}")
    for g in group_rows:
        f_ = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
        print(f"{g['eval_group']:18s} {g['files']:5d} {g['val_loss']:8.4f} {f_(g['greet_p_speak']):>6s} "
              f"{f_(g['wait_p_pad']):>6s} {f_(g['cue_p_speak']):>6s} {f_(g['distractor_p_pad']):>6s} "
              f"{f_(g['ordinary_text_loss']):>6s}")
    detail = args.out.parent / "loss_eval" / f"{label.replace('/', '__')}.jsonl"
    detail.parent.mkdir(parents=True, exist_ok=True)
    detail.write_text("".join(json.dumps(r) + "\n" for r in per_file))
    print(f"Wrote {args.out} and {detail}")


if __name__ == "__main__":
    main()
