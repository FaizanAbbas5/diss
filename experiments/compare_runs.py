#!/usr/bin/env python
"""Compare metrics across finished runs, e.g. the prompt battery:

    python experiments/compare_runs.py --configs configs/battery/e2e_*.yaml

Reads each run's metrics.json (run run_eval.py first, ideally with --claims).
"""
from __future__ import annotations

import argparse
import json

from t2t.config import load_config, run_dir


def _fmt(x, pct: bool = False) -> str:
    if x is None:
        return "-"
    if isinstance(x, int):
        return str(x)
    return f"{x * 100:.1f}%" if pct else f"{x:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    header = [
        ("run", 22), ("prompt", 24), ("n", 5),
        ("num-support", 12), ("unsup-items", 12),
        ("rel-wrong", 10),
        ("claims-ent", 11), ("contra-items", 13),
        ("parent-f1", 10), ("logprob", 9),
    ]
    print("  ".join(name.ljust(w) for name, w in header))
    print("-" * (sum(w for _, w in header) + 2 * (len(header) - 1)))

    for path in args.configs:
        cfg = load_config(path)
        mpath = run_dir(cfg, args.results_dir) / "metrics.json"
        m = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
        num = m.get("numbers", {})
        claims = m.get("claims", {})
        cells = [
            cfg.get("name", "?"),
            cfg.get("prompt", "?"),
            _fmt(num.get("items")),
            _fmt(num.get("number_support_rate_micro"), pct=True),
            _fmt(num.get("pct_items_with_unsupported_number"), pct=True),
            _fmt(m.get("relations", {}).get("relational_wrong_rate"), pct=True),
            _fmt(claims.get("claim_entailed_rate_micro"), pct=True),
            _fmt(claims.get("pct_items_with_contradiction"), pct=True),
            _fmt(m.get("parent", {}).get("parent_f1")),
            _fmt(m.get("uncertainty", {}).get("mean_seq_logprob")),
        ]
        if not m:
            cells[2] = "(no metrics.json yet)"
        print("  ".join(str(c).ljust(w) for c, (_, w) in zip(cells, header)))


if __name__ == "__main__":
    main()
