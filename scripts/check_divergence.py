"""Training-instability check for a running (or finished) run: reads runs/RUN/metrics.train.jsonl.

DIVERGED (exit 2) when: a non-finite loss; or after warm-up (step > 15) the loss is above the step-1
loss; or 3 consecutive steps each more than 2x the median of the 10 steps before them (and > 0.2
above it, so the tiny late losses of 1 h runs cannot trigger it). A single such step = SPIKE
(printed, exit 0). The finished lr 2e-6..3e-5 runs never exceeded 1.15x that median.
Usage: python scripts/check_divergence.py RUN
"""
import json
import math
import statistics
import sys
from pathlib import Path

run = sys.argv[1]
p = Path(__file__).resolve().parents[1] / f"runs/{run}/metrics.train.jsonl"
L = [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
loss = [(r["step"], r["loss"]) for r in L]
if not loss:
    print(f"{run}: no steps yet"); sys.exit(0)
for s, v in loss:
    if not math.isfinite(v):
        print(f"DIVERGED {run}: non-finite loss {v} at step {s}"); sys.exit(2)
first = loss[0][1]
for s, v in loss:
    if s > 15 and v > first:
        print(f"DIVERGED {run}: loss {v:.3f} at step {s} above the step-1 loss {first:.3f}"); sys.exit(2)
run_len, spikes = 0, []
for i in range(10, len(loss)):
    med = statistics.median(v for _, v in loss[i - 10:i])
    s, v = loss[i]
    if v > 2 * med and v - med > 0.2:
        run_len += 1; spikes.append(f"step {s}: {v:.3f} vs median {med:.3f}")
        if run_len >= 3:
            print(f"DIVERGED {run}: 3 consecutive steps > 2x the previous-10 median ({'; '.join(spikes[-3:])})"); sys.exit(2)
    else:
        run_len = 0
print(f"{run}: ok, step {loss[-1][0]}, loss {loss[-1][1]:.3f}" + (f"; SPIKES: {'; '.join(spikes)}" if spikes else ""))
