"""Offline streaming inference for Moshi, simulating a live conversation.

The user-side audio is fed to Moshi one 80 ms frame (1920 samples at 24 kHz) at a time,
in order, exactly like `moshi.server` does for a websocket client. Whatever Moshi
generates is recorded. Frames are processed as fast as the GPU allows (no wall-clock
sleeping); because the model only ever sees past input, the result is the same as a
live session that keeps up with real time.

For each input file and seed we write:
  <out-dir>/<stem>_seed<seed>.wav   stereo, 24 kHz: left = Moshi, right = user input
  <out-dir>/<stem>_seed<seed>.json  text tokens with timestamps + run metadata

Timeline convention: output slot k covers [k*80ms, (k+1)*80ms). The frame Moshi returns
after receiving input frame i is placed in slot i + OUTPUT_DELAY_FRAMES (= i + 1): input
frame i has fully arrived only at (i+1)*80ms, so that is the earliest a live user could
hear the reply (assuming zero compute/network latency).

Usage:
  python scripts/run_inference.py INPUT [--lora CKPT_DIR] [--seeds 1 2 3] [--gpu 1] --out-dir DIR
INPUT is a .wav file or a .jsonl whose lines have a "path" field (relative paths are
resolved against the jsonl's folder, as in training data).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Runpod's environment sets HF_HUB_ENABLE_HF_TRANSFER=1 but the hf_transfer package is
# not installed, which makes every Hugging Face download fail. Must be set before
# huggingface_hub is imported (moshi imports it).
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

SAMPLE_RATE = 24000
TAIL_SILENCE_SEC = 8.0
OUTPUT_DELAY_FRAMES = 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path, help="A .wav file, or a .jsonl listing wav paths under 'path'.")
    p.add_argument("--out-dir", type=Path, required=True, help="Where to write outputs.")
    p.add_argument("--lora", type=Path, default=None,
                   help="Checkpoint folder with lora.safetensors and config.json. Omit for the base model.")
    p.add_argument("--seeds", type=int, nargs="+", default=[1], help="Run each file once per seed.")
    p.add_argument("--gpu", type=int, default=None, help="GPU index (sets CUDA_VISIBLE_DEVICES).")
    p.add_argument("--tail-sec", type=float, default=TAIL_SILENCE_SEC,
                   help="Seconds of silence fed after the input ends (default 8).")
    p.add_argument("--hf-repo", default="kyutai/moshiko-pytorch-bf16", help="Base model repository.")
    return p.parse_args()


def list_inputs(path: Path) -> list[Path]:
    if path.suffix == ".jsonl":
        files = []
        for line in path.read_text().splitlines():
            if line.strip():
                p = Path(json.loads(line)["path"])
                files.append(p if p.is_absolute() else path.parent / p)
    else:
        files = [path]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        sys.exit(f"Input files not found: {missing[:5]}")
    stems = [f.stem for f in files]
    dupes = sorted({s for s in stems if stems.count(s) > 1})
    if dupes:
        sys.exit(f"Input files must have unique names (outputs are named after them): {dupes[:5]}")
    return files


def main() -> None:
    args = parse_args()
    if args.gpu is not None:
        # Must happen before torch initialises CUDA; afterwards the chosen GPU is "cuda".
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import numpy as np
    import sphn
    import torch
    torch.set_grad_enabled(False)  # inference only, like `with torch.no_grad(): main()` in moshi.server
    from moshi.models import LMGen, loaders
    from moshi.run_inference import get_condition_tensors, seed_all
    # Note: moshi.server can't be imported - it calls main() at module level and starts
    # the web server - so the few lines of ServerState setup we need are copied below.

    inputs = list_inputs(args.input)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    # ---- Load the model once (same calls as moshi.server.main) ----
    lora_weights = config_path = None
    if args.lora is not None:
        lora_weights = args.lora / "lora.safetensors"
        config_path = args.lora / "config.json"
        for f in (lora_weights, config_path):
            if not f.exists():
                sys.exit(f"--lora folder is missing {f.name}: {f}")
    print(f"Loading {args.hf_repo}" + (f" + LoRA {args.lora}" if args.lora else " (base model)"), flush=True)
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        args.hf_repo, lora_weights=lora_weights, config_path=config_path)
    mimi = checkpoint_info.get_mimi(device=device)
    text_tokenizer = checkpoint_info.get_text_tokenizer()
    lm = checkpoint_info.get_moshi(device=device, dtype=torch.bfloat16, fuse_lora=True)
    # Copied from moshi.server.ServerState.__init__: LMGen with the checkpoint's default
    # sampling settings (lm_gen_config; empty -> LMGen defaults), mimi/lm_gen in streaming mode.
    condition_tensors = get_condition_tensors(checkpoint_info.model_type, lm, batch_size=1, cfg_coef=1.0)
    lm_gen = LMGen(lm, cfg_coef=1.0, condition_tensors=condition_tensors, **checkpoint_info.lm_gen_config)
    frame_size = int(mimi.sample_rate / mimi.frame_rate)  # 1920 samples = 80 ms
    mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)
    # Copied from moshi.server.ServerState.warmup: a few silent frames so CUDA graphs and
    # kernels are built before timing. State is reset (and RNG re-seeded) before each run.
    for _ in range(4):
        codes = mimi.encode(torch.zeros(1, 1, frame_size, dtype=torch.float32, device=device))
        for c in range(codes.shape[-1]):
            tokens = lm_gen.step(codes[:, :, c:c + 1])
            if tokens is not None:
                mimi.decode(tokens[:, 1:])
    torch.cuda.synchronize()
    frame_sec = frame_size / mimi.sample_rate
    assert mimi.sample_rate == SAMPLE_RATE
    sampling = {k: getattr(lm_gen, k) for k in ("use_sampling", "temp", "temp_text", "top_k", "top_k_text")}
    print(f"Sampling settings: {sampling}", flush=True)
    padding_ids = {lm.text_padding_token_id, lm.end_of_text_padding_id}  # 3 and 0, as the server skips

    def run_one(user_pcm: np.ndarray, seed: int) -> tuple[np.ndarray, list[int | None]]:
        """Stream `user_pcm` (float32, 24 kHz, length a multiple of frame_size) through
        Moshi. Returns Moshi's audio (same length) and one text token id per output slot."""
        n_frames = len(user_pcm) // frame_size
        moshi_pcm = np.zeros(n_frames * frame_size, dtype=np.float32)
        text_ids: list[int | None] = [None] * n_frames
        # Fresh conversation: reset streaming state (as handle_chat does), then seed so
        # that sampling is reproducible regardless of what ran before.
        mimi.reset_streaming()
        lm_gen.reset_streaming()
        seed_all(seed)
        skip_frames = 1
        for i in range(n_frames):
            chunk = torch.from_numpy(user_pcm[i * frame_size:(i + 1) * frame_size])
            chunk = chunk.to(device=device)[None, None]
            codes = mimi.encode(chunk)
            if skip_frames:
                # Same as moshi.server: the first encoded frame has a special structure
                # (left padding), so the encoder state is reset to reapply it next call.
                mimi.reset_streaming()
                skip_frames -= 1
            for c in range(codes.shape[-1]):
                tokens = lm_gen.step(codes[:, :, c:c + 1])
                if tokens is None:
                    continue
                out = mimi.decode(tokens[:, 1:])[0, 0].float().cpu().numpy()
                slot = i + OUTPUT_DELAY_FRAMES
                if slot < n_frames:
                    moshi_pcm[slot * frame_size:(slot + 1) * frame_size] = out
                    text_ids[slot] = int(tokens[0, 0, 0].item())
        return moshi_pcm, text_ids

    total_audio = total_wall = 0.0
    with torch.no_grad():
        for path in inputs:
            pcm, _ = sphn.read(str(path), sample_rate=SAMPLE_RATE)  # (channels, samples), resampled
            if pcm.ndim == 1:
                pcm = pcm[None]
            if pcm.shape[0] == 1:
                user = pcm[0]
            elif pcm.shape[0] == 2:
                user = pcm[1]  # stereo: right channel is the user side
            else:
                sys.exit(f"{path}: expected mono or stereo, got {pcm.shape[0]} channels")
            input_sec = len(user) / SAMPLE_RATE
            # Append the tail silence, then zero-pad to a whole number of 80 ms frames.
            n_total = len(user) + int(round(args.tail_sec * SAMPLE_RATE))
            n_total = -(-n_total // frame_size) * frame_size
            user_padded = np.zeros(n_total, dtype=np.float32)
            user_padded[:len(user)] = user
            stream_sec = n_total / SAMPLE_RATE

            for seed in args.seeds:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                moshi_pcm, text_ids = run_one(user_padded, seed)
                torch.cuda.synchronize()
                wall = time.perf_counter() - t0
                total_audio += stream_sec
                total_wall += wall

                stem = f"{path.stem}_seed{seed}"
                sphn.write_wav(str(args.out_dir / f"{stem}.wav"),
                               np.stack([moshi_pcm, user_padded]), sample_rate=SAMPLE_RATE)
                words = [
                    {"start": round(k * frame_sec, 3), "end": round((k + 1) * frame_sec, 3), "id": tid,
                     "piece": text_tokenizer.id_to_piece(tid)}  # type: ignore
                    for k, tid in enumerate(text_ids) if tid is not None and tid not in padding_ids
                ]
                meta = {
                    "input": str(path),
                    "seed": seed,
                    "lora": str(args.lora) if args.lora else None,
                    "hf_repo": args.hf_repo,
                    "sampling": sampling,
                    "sample_rate": SAMPLE_RATE,
                    "frame_sec": frame_sec,
                    "output_delay_frames": OUTPUT_DELAY_FRAMES,
                    "input_sec": input_sec,
                    "tail_silence_sec": args.tail_sec,
                    "stream_sec": stream_sec,
                    "wall_sec": wall,
                    "realtime_factor": wall / stream_sec,
                    "text": "".join(w["piece"] for w in words).replace("▁", " ").strip(),
                    # Non-padding text tokens; start/end is the output slot they were emitted in.
                    "text_tokens": words,
                    # Raw text token per 80 ms output slot (null = no output yet).
                    "text_token_ids_per_frame": text_ids,
                }
                (args.out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
                print(f"{stem}: {stream_sec:.1f}s audio in {wall:.1f}s "
                      f"(x{wall / stream_sec:.2f} real time) -> {meta['text'][:80]!r}", flush=True)

    if total_audio:
        print(f"Total: {total_audio:.1f}s audio in {total_wall:.1f}s (x{total_wall / total_audio:.2f} real time)")


if __name__ == "__main__":
    main()
