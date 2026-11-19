# Results index — teaching Moshi conditional silence

LoRA fine-tunes of `kyutai/moshiko-pytorch-bf16` on synthetic conversations (data v2) that teach Moshi to stay
silent when asked to wait, come back at the right time, and still answer ordinary requests. All numbers: 400
held-out eval files (seed 1), best checkpoint per run (highest wait success with ordinary response ≥ 62%).
Overall score = mean of silence (100 − interrupt), return (100 − missed), answer (ordinary response),
addressee (100 − distractor response).

## Headline

| model | overall score | wait success | interrupt | missed return | ordinary | distractor |
|---|---|---|---|---|---|---|
| base | 48.5 | 3% | 94% | 16% | 72% | 68% |
| 1 h · lr 1e-4 · step 50 | 73.3 | 41% | 32% | 35% | 75% | 16% |
| 1 h · lr 1e-4 · onset ×3 · step 150 | 68.4 | 27% | 66% | 10% | 95% | 45% |
| 10 h · lr 3e-5 · step 125 | 77.8 | 49% | 13% | 35% | 73% | 15% |
| 10 h · lr 1e-4 · step 125 | 85.1 | 62% | 11% | 19% | 82% | 12% |
| 10 h · lr 1e-4 · onset ×3 · step 150 | 86.5 | 65% | 19% | 11% | 91% | 15% |
| 10 h · lr 3e-4 · onset ×3 · step 100 | **87.7** | **66%** | 16% | 11% | 90% | 13% |
| control 10 h · lr 1e-4 (ordinary data only) | – | 7% | – | – | – | – |

**Findings**
- **Data size matters most:** 10 h beats 1 h at every learning rate; ordinary-only controls don't learn to wait,
  so the gains come from the wait conversations.
- **Learning rate:** 1e-4 ≫ 3e-5 (+13 pp wait success at 10 h); 3e-4 ties 1e-4.
- **Onset weighting ×3** (3× loss in the first second of every Moshi turn) trades readiness to speak against
  waiting: fewer missed returns, more answers, more interruptions. It helps only with enough data
  (interaction +6.2 / +6.7 score points at lr 1e-4 / 3e-5; [onset_x_size.md](onset_x_size.md)).
- **Still failing:** timed returns ("ask me again in 30 seconds") and extensions — the models don't track time.

## Where things are

| file | contents |
|---|---|
| [REPORT.md](REPORT.md), [REPORT_APPENDIX.md](REPORT_APPENDIX.md) | write-up (covers runs up to lr 3e-5 + onset) |
| [LOG.md](LOG.md) | every decision and run, newest first |
| [comparison.md](comparison.md) | every evaluated model, every metric, overall and by eval group |
| [best_checkpoints.md](best_checkpoints.md), [best/](best/) | best checkpoint per run and why |
| [checkpoint_table.md](checkpoint_table.md), [runs.csv](runs.csv) | overall score of every checkpoint; every checkpoint's losses and data hours |
| [onset_x_size.md](onset_x_size.md) | 2×2 data size × onset weighting, with interactions |
| [diagnostics/paired_diffs.txt](diagnostics/paired_diffs.txt) | paired cluster-bootstrap comparisons |
| [data_summary_v2.md](data_summary_v2.md) | datasets: hours, unique content, Fish cost |
| [human_tts/](human_tts/) | human recordings vs TTS twins |
| [plots/](plots/) | data size, learning rate, survival and per-run score plots |
| [metrics/](metrics/) | per-file scores (`<model>.jsonl`), loss evaluations (`loss_eval*.csv`) |
| `../runs/<run>/` | configs, training logs, per-output transcripts (audio and weights not in git) |
| `../logs/` | raw job logs |
