"""Voice similarity of each model's Moshi output to the base model's (speaker embeddings).

Embedding: SpeechBrain ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb, 192-d) of Moshi's channel
(left) of an inference output, restricted to its VAD speech segments (results/metrics/<model>.jsonl,
greeting included) resampled to 16 kHz; files with < 3 s of speech are skipped.
  cos_vs_base       per eval v2 file (seed 1): cosine(model, base on the same file), averaged over
                    files where both have >= 3 s; 95% CI by cluster bootstrap over contexts. For the
                    base row: base seed 1 vs seeds 2 and 3 on the human files (sampling noise floor).
  cos_vs_reference  mean cosine to the training data's Moshi voice (mean embedding of the left
                    channel of the first 40 train_10h_v2 files).
Embeddings are cached in runs/<model>/voice_emb.npz. Writes results/metrics/voice_similarity.csv
(a row per model, replaced on rerun).

Usage: python scripts/voice_similarity.py MODEL... [--gpu 0]
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_lib import bootstrap_ratio, model_dir  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/metrics/voice_similarity.csv"
MIN_SPEECH = 3.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="+")
    p.add_argument("--gpu", type=int, default=None)
    a = p.parse_args()
    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    import sphn
    import torch
    import torchaudio
    from speechbrain.inference.speaker import EncoderClassifier
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                         savedir=str(Path.home() / ".cache/speechbrain/ecapa"), run_opts={"device": dev})

    def embed(wav_path: str, segs: list) -> np.ndarray | None:
        x, sr = sphn.read(wav_path)
        left = torch.from_numpy(np.ascontiguousarray(x[0], dtype=np.float32))
        if segs is not None:
            left = torch.cat([left[int(s * sr):int(e * sr)] for s, e in segs]) if segs else left[:0]
        if len(left) < MIN_SPEECH * sr:
            return None
        left = torchaudio.functional.resample(left, sr, 16000)
        with torch.no_grad():
            e = enc.encode_batch(left[None].to(dev))[0, 0].float().cpu().numpy()
        return e / np.linalg.norm(e)

    def model_embs(m: str) -> dict:
        cache = model_dir(m) / "voice_emb.npz"
        if cache.exists():
            z = np.load(cache, allow_pickle=True)
            return z["embs"].item()
        rows = [json.loads(l) for l in open(REPO / f"results/metrics/{m}.jsonl")]
        embs = {}
        for r in rows:
            e = embed(r["output"], r["speech_segments"])
            if e is not None:
                embs[(r["id"], r["seed"])] = e
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, embs=np.array(embs, dtype=object))
        return embs

    ref_path = REPO / "results/metrics/voice_reference_emb.npy"
    if ref_path.exists():
        ref = np.load(ref_path)
    else:
        idx = [json.loads(l) for l in open(REPO / "data/indexes/train_10h_v2.jsonl")][:40]
        es = [embed(str((REPO / "data/indexes" / r["path"]).resolve()), None) for r in idx]
        ref = np.mean([e for e in es if e is not None], 0); ref /= np.linalg.norm(ref)
        np.save(ref_path, ref)

    clusters = {}
    for l in open(REPO / "results/metrics/base.jsonl"):
        r = json.loads(l); clusters[r["id"]] = r["cluster"]
    base = model_embs("base")
    rows = {r["model"]: r for r in csv.DictReader(OUT.open())} if OUT.exists() else {}
    for m in a.models:
        embs = base if m == "base" else model_embs(m)
        if m == "base":
            pairs = [(k, (k[0], s)) for k in embs if k[1] == 1 and not k[0].startswith("eval_")  # human files
                     for s in (2, 3) if (k[0], s) in embs]
            sims = [(clusters.get(k[0], k[0]), float(embs[k] @ embs[o])) for k, o in pairs]
        else:
            sims = [(clusters.get(k[0], k[0]), float(e @ base[k])) for k, e in embs.items()
                    if k[1] == 1 and k[0].startswith("eval_") and k in base]
        cl = sorted({c for c, _ in sims})
        num = np.array([sum(v for c2, v in sims if c2 == c) for c in cl]); den = np.array([sum(1 for c2, _ in sims if c2 == c) for c in cl], float)
        lo, hi = bootstrap_ratio(num, den) if len(cl) > 1 else (float("nan"),) * 2
        vref = float(np.mean([e @ ref for k, e in embs.items() if k[0].startswith("eval_")])) if embs else float("nan")
        rows[m] = {"model": m, "n": len(sims), "cos_vs_base": float(num.sum() / den.sum()) if len(sims) else "",
                   "lo": lo, "hi": hi, "cos_vs_reference": vref}
        print(rows[m])
    import fcntl
    with open(OUT.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        old = {r["model"]: r for r in csv.DictReader(OUT.open())} if OUT.exists() else {}
        old.update({m: rows[m] for m in a.models})
        with OUT.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["model", "n", "cos_vs_base", "lo", "hi", "cos_vs_reference"])
            w.writeheader(); w.writerows(old.values())


if __name__ == "__main__":
    main()
