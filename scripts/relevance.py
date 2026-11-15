"""Reply relevance: does Moshi's generated reply fit the conversation, compared with the reference?

A fixed random sample (seed 7) of 50 resumes (eval v2 files with a wait: Moshi's first ideal turn
at/after wait_end) and 50 ordinary replies (ordinary requests: Moshi's ideal answer), the same
items for every model. For each model, the reply is Moshi's own text stream (inference json,
non-padding text tokens) starting in [t, t + 12 s), where t = wait_end or the request end, cut at
the user's next turn that starts >= 1 s after t. Empty = no reply (not judged; a failure in the strict rate_all).

  prepare MODEL...  writes results/relevance/batch_<tag>.jsonl: blind, shuffled items (context =
                    the scripted conversation before t, with the reference Moshi turns; the
                    reference reply; the generated reply; no model names) and the key
                    results/relevance/key_<tag>.json (uid -> model, item)
  (a judge - here a Claude subagent with the instructions in results/relevance/JUDGE.md - writes
   results/relevance/verdicts_<tag>.jsonl: {"uid", "fits": true/false, "why"})
  collect           joins all verdicts with their keys -> results/metrics/relevance.csv: per model
                    and kind (resume / ordinary) rate = fits / judged, k, n, no_reply, 95% CI
                    (cluster bootstrap over contexts), and the strict rate_all = fits / all sampled
                    items (no reply counts as not fitting), n_all, lo_all, hi_all

Usage: python scripts/relevance.py prepare --tag pair1 base train_1h_v2_lr1e-5 ...
       python scripts/relevance.py collect
"""

import argparse
import csv
import glob
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_lib import bootstrap_ratio, model_dir, ordered, readable, model_roles  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DIR = REPO / "results/relevance"
N_EACH, SEED, WINDOW, CONTEXT_TURNS = 50, 7, 12.0, 8


def load_eval() -> list[dict]:
    out = []
    for ep in sorted(glob.glob(str(REPO / "data/v2/eval_full/eval_*.events.json"))):
        e = json.load(open(ep))
        s = json.load(open(REPO / f"data/v2/scripts/eval/{e['id']}.json"))
        su = iter(e["right_turns"]); sm = iter(e["moshi_turns"])
        turns = []
        for t in s["turns"]:  # scripts and events list the turns in the same order (checked)
            tt = next(sm) if t["speaker"] == "moshi" else next(su)
            turns.append({"speaker": t["speaker"], "text": t["text"], "start": tt["start"], "end": tt["end"],
                          **({"kind": tt.get("kind")} if t["speaker"] == "moshi" else {})})
        out.append({"ev": e, "turns": turns})
    return out


def items() -> list[dict]:
    """The fixed sample (same for every model)."""
    files = load_eval()
    res, ordn = [], []
    for f in files:
        e, turns = f["ev"], f["turns"]
        moshi = [t for t in turns if t["speaker"] == "moshi"]
        if e.get("wait_end") is not None:
            ref = next((t for t in moshi if t["start"] >= e["wait_end"] - 0.1), None)
            if ref:
                res.append(dict(kind="resume", id=e["id"], t=e["wait_end"], ref=ref["text"], cluster=e["cluster"]))
        for q in e.get("ordinary_requests", []):
            ref = next((t for t in moshi if t["start"] >= q - 0.1), None)
            if ref:
                ordn.append(dict(kind="ordinary", id=e["id"], t=q, ref=ref["text"], cluster=e["cluster"]))
    rng = random.Random(SEED)
    picked = rng.sample(res, N_EACH) + rng.sample(ordn, N_EACH)
    turns = {f["ev"]["id"]: f["turns"] for f in files}
    for it in picked:
        before = [t for t in turns[it["id"]] if t["start"] < it["t"] - 0.05][-CONTEXT_TURNS:]
        it["context"] = [f"{'Moshi' if t['speaker'] == 'moshi' else t['speaker'].capitalize()}: {t['text']}" for t in before]
        nxt = [t["start"] for t in turns[it["id"]] if t["speaker"] != "moshi" and t["start"] >= it["t"] + 1.0]
        it["cut"] = min([it["t"] + WINDOW] + nxt)
    return picked


def generated(model: str, it: dict) -> str:
    j = model_dir(model) / f"eval/{it['id']}_seed1.json"
    toks = json.load(open(j))["text_tokens"]
    return "".join(w["piece"] for w in toks if it["t"] <= w["start"] < it["cut"]).replace("▁", " ").strip()


def prepare(tag: str, models: list[str]) -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    its = items()
    (DIR / "items.json").write_text(json.dumps(its, indent=1))
    batch, key = [], {}
    rng = random.Random(f"{tag}")
    for m in models:
        for i, it in enumerate(its):
            g = generated(m, it)
            uid = f"{tag}-{rng.randrange(16 ** 8):08x}"
            key[uid] = {"model": m, "item": i, "kind": it["kind"], "cluster": it["cluster"], "empty": not g}
            if g:
                batch.append({"uid": uid, "situation": ("Moshi had been asked to wait; this is the moment it "
                              "should speak again (deadline reached or the user is back)." if it["kind"] == "resume"
                              else "The user just made a request or remark that Moshi should answer."),
                              "context": it["context"], "reference_reply": it["ref"], "generated_reply": g})
    rng.shuffle(batch)
    (DIR / f"batch_{tag}.jsonl").write_text("".join(json.dumps(b) + "\n" for b in batch))
    (DIR / f"key_{tag}.json").write_text(json.dumps(key, indent=1))
    print(f"{len(batch)} items to judge ({sum(k['empty'] for k in key.values())} empty) -> {DIR / f'batch_{tag}.jsonl'}")


def collect() -> None:
    key, verdict = {}, {}
    for kp in glob.glob(str(DIR / "key_*.json")):
        key.update(json.load(open(kp)))
    for vp in glob.glob(str(DIR / "verdicts_*.jsonl")):
        for line in open(vp):
            if line.strip():
                v = json.loads(line); verdict[v["uid"]] = bool(v["fits"])
    rows = []
    roles = model_roles()
    for m in ordered({k["model"] for k in key.values()}):
        for kind in ("resume", "ordinary"):
            ks = {u: k for u, k in key.items() if k["model"] == m and k["kind"] == kind}
            judged = {u: k for u, k in ks.items() if not k["empty"]}
            missing = [u for u in judged if u not in verdict]
            if missing:
                print(f"WARNING {m} {kind}: {len(missing)} items not judged yet")
                continue
            cl = sorted({k["cluster"] for k in judged.values()})
            num = np.array([sum(verdict[u] for u, k in judged.items() if k["cluster"] == c) for c in cl], float)
            den = np.array([sum(1 for k in judged.values() if k["cluster"] == c) for c in cl], float)
            kk, n = int(num.sum()), int(den.sum())
            lo, hi = bootstrap_ratio(num, den) if n else (float("nan"),) * 2
            # strict rate: every sampled item counts, no reply = does not fit
            cla = sorted({k["cluster"] for k in ks.values()})
            num_a = np.array([sum(verdict.get(u, False) for u, k in ks.items() if k["cluster"] == c) for c in cla], float)
            den_a = np.array([sum(1 for k in ks.values() if k["cluster"] == c) for c in cla], float)
            lo_a, hi_a = bootstrap_ratio(num_a, den_a)
            rows.append({"label": readable(m, roles), "model": m, "kind": kind, "k": kk, "n": n, "rate": kk / n if n else "",
                         "lo": lo, "hi": hi, "no_reply": len(ks) - len(judged), "n_all": int(den_a.sum()),
                         "rate_all": num_a.sum() / den_a.sum(), "lo_all": lo_a, "hi_all": hi_a})
    out = REPO / "results/metrics/relevance.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "model", "kind", "k", "n", "rate", "lo", "hi", "no_reply",
                                              "n_all", "rate_all", "lo_all", "hi_all"])
        w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["prepare", "collect"])
    p.add_argument("models", nargs="*")
    p.add_argument("--tag", default="x")
    a = p.parse_args()
    prepare(a.tag, a.models) if a.cmd == "prepare" else collect()
