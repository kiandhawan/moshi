"""Write one training config per run of the grid, all derived from configs/smoke.yaml.

Changed from smoke.yaml: train_data, run_dir, optim.lr, max_steps 250, ckpt_freq 50, num_ckpt_keep
null (keep all; pair 1 ran with the default 3 and scripts/archive_ckpts.sh copied its checkpoints),
eval_freq 50 (on the 40-file eval_full_subset_v2, already smoke's eval_data), optim.pct_start
0.1 (25 warm-up steps out of 250, then cosine decay). Unchanged: seed 0, batch 16,
duration_sec 100, LoRA (rank 128, scaling 2), base kyutai/moshiko-pytorch-bf16, loss weights.
Controls get lr null until the best learning rate per data size is chosen (pass
--control-lr 1h=3e-5 10h=1e-5 to fill them in); a null lr makes train.py refuse to load.

Usage: python scripts/make_run_configs.py [--control-lr 1h=LR 10h=LR]
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IDX = "/root/moshi-finetune/data/indexes"
LRS = {"2e-6": "2e-6", "1e-5": "1e-5", "3e-5": "3e-5"}
SIZES = {"1h": "train_1h_v2", "10h": "train_10h_v2"}
CONTROLS = {"1h": "control_1h_v2", "10h": "control_10h_v2"}


def runs(control_lr: dict[str, str | None]) -> dict[str, tuple[str, str | None]]:
    """run name -> (index name, lr)."""
    out = {f"{ix}_lr{lr}": (ix, lr) for lr in LRS for ix in SIZES.values()}
    out.update({ix: (ix, control_lr.get(size)) for size, ix in CONTROLS.items()})
    return out


def config(name: str, index: str, lr: str | None) -> str:
    s = (ROOT / "configs/smoke.yaml").read_text()
    body = s[s.index("# data"):]
    subs = [
        ("'/root/moshi-finetune/daily-talk-contiguous/smoke.jsonl'", f"'{IDX}/{index}.jsonl'"),
        ("  lr: 2e-6\n", f"  lr: {lr}\n" if lr else "  lr: null  # TO SET: best learning rate for this data size (results/LOG.md)\n"),
        ("max_steps: 20\n", "max_steps: 250\n"),
        ("  pct_start: 0.1  # example uses 0.05; with max_steps 20 that gives a 0-length warmup and OneCycleLR divides by zero\n",
         "  pct_start: 0.1  # 25 warm-up steps of 250, then cosine decay\n"),
        ("eval_freq: 50\n", "eval_freq: 50\n"),
        ("ckpt_freq: 20\n", "ckpt_freq: 50\nnum_ckpt_keep: null  # keep every checkpoint (default 3 deletes older ones)\n"),
        ('run_dir: "runs/smoke"', f'run_dir: "runs/{name}"'),
    ]
    for old, new in subs:
        assert old in body, old
        body = body.replace(old, new)
    head = (f"# {name}: LoRA on data/indexes/{index}.jsonl"
            + (f", lr {lr}" if lr else ", lr NOT SET YET (config refuses to load until it is)") + ".\n"
            "# Written by scripts/make_run_configs.py from configs/smoke.yaml; only train_data, run_dir, lr,\n"
            "# max_steps (250), ckpt_freq (50), num_ckpt_keep (all), eval_freq (50) and pct_start (0.1) differ.\n")
    return head + "\n" + body


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--control-lr", nargs="*", default=[], help="e.g. 1h=3e-5 10h=1e-5")
    a = p.parse_args()
    control_lr = dict(x.split("=") for x in a.control_lr)
    for name, (index, lr) in runs(control_lr).items():
        (ROOT / f"configs/{name}.yaml").write_text(config(name, index, lr))
        print(f"configs/{name}.yaml  data={index}  lr={lr}")


if __name__ == "__main__":
    main()
