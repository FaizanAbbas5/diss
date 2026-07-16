#!/usr/bin/env python
"""Generate summaries for a config.

    python experiments/run_generation.py --config configs/e2e_smoke_cpu.yaml
"""
from __future__ import annotations

import argparse

from t2t import data
from t2t.config import load_config, run_dir, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 13)))
    examples = data.load(cfg["dataset"], cfg["split"], limit=cfg.get("limit"))
    out = run_dir(cfg, args.results_dir)
    print(f"{len(examples)} examples -> {out}")

    backend = cfg.get("backend", "hf")
    if backend == "groq":
        from t2t.generate.groq import run_generation_groq as runner
    elif backend == "hf":
        from t2t.generate import run_generation as runner
    elif backend == "soft":
        from t2t.generate.soft import run_generation_soft as runner
    else:
        raise SystemExit(
            f"Unknown backend: {backend!r} (expected 'hf', 'groq', or 'soft')"
        )
    path = runner(cfg, examples, out)
    print(f"Generations written to {path}")


if __name__ == "__main__":
    main()
