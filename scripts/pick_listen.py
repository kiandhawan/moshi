"""Copy 4 example outputs of a model into listen/<model>/ (not committed): a wait success and a
wait failure (different wait types where possible) and an ordinary conversation answered in full
and one with a missed request, from results/metrics/<model>.jsonl (eval v2, seed 1), picked with
a fixed seed. Names: <eval group>__<outcome>__<id>.wav (stereo: left = Moshi, right = user).
Usage: python scripts/pick_listen.py MODEL [FOLDER]   (default folder = MODEL)
"""
import json, random, shutil, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
m = sys.argv[1]
name = sys.argv[2] if len(sys.argv) > 2 else m
rows = [json.loads(l) for l in open(REPO / f"results/metrics/{m}.jsonl")]
rows = [r for r in rows if "human" not in r["groups"]]
rng = random.Random(0)
out = REPO / "listen" / name
out.mkdir(parents=True, exist_ok=True)
for f in out.glob("*.wav"):
    f.unlink()
group = lambda r: next((g for g in r["groups"] if g != "familiar_type"), r["groups"][0])  # noqa: E731
waits = [r for r in rows if r.get("wait_success") is not None]
ok = [r for r in waits if r["wait_success"]]
bad = [r for r in waits if not r["wait_success"]]
picks = []
if ok:
    picks.append((rng.choice(ok), "wait_success"))
if bad:
    other = [r for r in bad if not ok or r["type"] != picks[0][0]["type"]] or bad
    picks.append((rng.choice(other), "wait_failure"))
ordn = [r for r in rows if r["type"] == "ordinary" and r["ordinary_answered"]]
full = [r for r in ordn if all(r["ordinary_answered"])]
miss = [r for r in ordn if not all(r["ordinary_answered"])]
if full:
    picks.append((rng.choice(full), "ordinary_all_answered"))
if miss:
    picks.append((rng.choice(miss), "ordinary_missed_request"))
for r, outcome in picks:
    why = ""
    if outcome == "wait_failure":
        why = "+".join(k for k, v in (("interrupted", r["interrupted"]), ("not_on_time", not r["resumed_on_time"]),
                                       ("distractor", any(r["distractor_responses"]))) if v)
    g = group(r) if group(r) == r["type"] else f"{r['type']}-{group(r)}"
    name = f"{g}__{outcome}{'_' + why if why else ''}__{r['id']}.wav"
    shutil.copy2(r["output"], out / name)
    print(name)
