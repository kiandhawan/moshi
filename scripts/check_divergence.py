"""Training-instability check for a running (or finished) run: reads runs/RUN/metrics.train.jsonl.

DIVERGED (exit 2) when: a non-finite loss; or after warm-up (step > 15) 3 consecutive steps above
the step-1 loss; or 3 consecutive steps each more than 2x the median of the 10 steps before them
(and > 0.2 above it, so the tiny late losses of 1 h runs cannot trigger it). A single such step =
SPIKE (printed, exit 0). The finished lr 2e-6..1e-4 runs never exceeded 1.15x that median.
(Until 2026-11-17 04:05 one step above the step-1 loss counted as divergence; that stopped
train_10h_v2_lr3e-4 at step 18 on a single spike it was recovering from - see results/LOG.md.)
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
above, spikes = [], []
for s, v in loss:
    if s > 15 and v > first:
        above.append((s, v)); spikes.append(f"step {s}: {v:.3f} above the step-1 loss {first:.3f}")
        if len(above) >= 3 and [x for x, _ in above[-3:]] == list(range(s - 2, s + 1)):
            print(f"DIVERGED {run}: 3 consecutive steps above the step-1 loss {first:.3f} "
                  f"({', '.join(f'step {x}: {y:.3f}' for x, y in above[-3:])})"); sys.exit(2)
run_len = 0
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
