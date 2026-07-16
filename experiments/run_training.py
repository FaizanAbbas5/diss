#!/usr/bin/env python
"""Train the Arm-2 encoder + projector for a config.

    python experiments/run_training.py --config configs/arm2_e2e_smoke_cpu.yaml

Resumable: re-running the same config picks up from the last checkpoint in
the hashed run dir.
"""
from __future__ import annotations

import argparse

from t2t.config import load_config
from t2t.train import train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = train(cfg, args.results_dir)
    print(f"Run dir: {out}")


if __name__ == "__main__":
    main()
