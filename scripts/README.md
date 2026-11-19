# Scripts — reproduction pipeline

Run from the repo root with `.venv/bin/python`. Fish Audio steps need `FISH_API_KEY`
(`set -a && . ~/.config/fish_audio/env && set +a`). Details and reasons: `results/LOG.md`.

**1. Splits, voices, Moshi's voice** (one-time; outputs are in the repo)
`make_splits.py` (train/eval splits of topics, phrasings, names) · `fish_voices.py` (stock Fish voices) ·
`extract_moshi_reference.py` + `fish_clone_moshi.py` (the fixed Moshi voice)

**2. Data v2**
`gen_datasets_v2.sh` (scripts: `generate_scripts_v2.py`, built on `generate_scripts.py`) →
`render_all_v2.sh` (audio: `render.py`, `render_lib.py`, `loudness.py`) →
`build_datasets_v2.py` (indexes, manifests, `results/data_summary_v2.md`; uses `build_datasets.py`) ·
`make_control_1h_v2.py` · `make_eval_subset.py`
Human test set: `build_human_full.py` (ideal-reply versions) · `make_human_tts.py` (TTS twins) ·
In-distribution check set: `make_in_distribution.py`

**3. Training and evaluation**
`make_run_configs.py` (grid configs) → `run.sh RUN GPU` (train in tmux, `check_divergence.py`, then
`eval_run.py`: `eval_loss.py` on every checkpoint, top 3 + final through `run_inference.py` and `score.py`,
best checkpoint, pruning, `voice_similarity.py`, `pick_listen.py`, tables) · `extra_ckpts.sh` (test more checkpoints)

**4. Reports**
`compare.py` (comparison.md) · `best_table.py` · `checkpoint_table.py` · `runs_table.py` · `plot_results.py` ·
`paired_diff.py` · `onset_x_size.py` · `human_tts_compare.py` · `in_dist_compare.py` · `relevance.py` ·
`report_extra.py` · shared code `report_lib.py`

**5. Diagnostics reported in REPORT.md**
`diag_quiet.sh` + `diag_quiet_report.py` (40-file quiet diagnostics) · `pad_bias_sweep.sh` + `pad_bias_report.py`
(decoding-time silence penalty)

**6. Listening and demos**
`talk.sh` (live Moshi server) · `build_listen_site.py` + `build_listen_extras.py` (listening site)

Removed in the cleanup (data checkers, one-off launchers for specific nights, v1-only pipeline scripts):
recover any of them with `git show b67627d8:scripts/<name>`.
