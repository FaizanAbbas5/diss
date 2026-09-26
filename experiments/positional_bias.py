"""Positional-bias check for wrong-winner claims (dissertation Ch6, qualitative
analysis).

For each evaluated RotoWire run, split the games by whether the FIRST-listed
team (first row of the `teams` section, i.e. the home team in the RotoWire
JSON) or the SECOND-listed team actually won, and report how often the output
contains a wrong-winner claim in each case, with a bootstrap CI on the
difference. Also counts which team the wrong claims name.

    python experiments/positional_bias.py --configs configs/battery_full/rotowire_faithful.yaml \
        configs/final/rotowire_baseline.yaml configs/arms/rotowire_arm2_aug_filtered_gen.yaml

Reads results/<run>/metrics_per_item.jsonl (evaluator output) and the dataset;
writes nothing.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path

from t2t.config import load_config, run_dir
from t2t.data import load
from t2t.facts import GameFacts


def _ci_diff(k_a: int, n_a: int, k_b: int, n_b: int, n_boot: int = 10_000, seed: int = 13):
    """Percentile bootstrap CI for rate_a - rate_b of two independent proportions."""
    rng = random.Random(seed)
    a = [1] * k_a + [0] * (n_a - k_a)
    b = [1] * k_b + [0] * (n_b - k_b)
    diffs = []
    for _ in range(n_boot):
        ra = sum(rng.choice(a) for _ in range(n_a)) / n_a
        rb = sum(rng.choice(b) for _ in range(n_b)) / n_b
        diffs.append(ra - rb)
    diffs.sort()
    return diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]


def compute(cfg_path: str, results_dir: str) -> dict | None:
    """Return the positional-bias statistics for one evaluated run."""
    cfg = load_config(cfg_path)
    d = run_dir(cfg, results_dir)
    per_item = d / "metrics_per_item.jsonl"
    if not per_item.exists():
        return None
    examples = {
        ex.id: ex
        for ex in load(
            cfg["dataset"], cfg["split"], limit=cfg.get("limit"), sample_seed=cfg.get("sample_seed")
        )
    }
    n = {"first": 0, "second": 0}
    err = {"first": 0, "second": 0}
    named = {"first": 0, "second": 0}
    for line in open(per_item, encoding="utf-8"):
        m = json.loads(line)
        ex = examples.get(m["id"])
        if ex is None:
            continue
        gf = GameFacts(ex.table)
        if not gf.ok or gf.tie:
            continue
        first, second = gf.team_names[0], gf.team_names[1]
        key = "first" if gf.winner == first else "second"
        n[key] += 1
        wrong = [w for w in m["relations"]["wrong"] if w["type"] == "winner"]
        if wrong:
            err[key] += 1
        for w in wrong:
            if first in w["claimed"] and second not in w["claimed"]:
                named["first"] += 1
            elif second in w["claimed"] and first not in w["claimed"]:
                named["second"] += 1
    r_first = err["first"] / n["first"] if n["first"] else float("nan")
    r_second = err["second"] / n["second"] if n["second"] else float("nan")
    lo, hi = _ci_diff(err["first"], n["first"], err["second"], n["second"])
    return {
        "name": cfg.get("name"), "dataset": cfg["dataset"], "split": cfg["split"],
        "n": n, "err": err, "named": named,
        "r_first": r_first, "r_second": r_second, "ci": (lo, hi),
    }


def analyse(cfg_path: str, results_dir: str) -> None:
    s = compute(cfg_path, results_dir)
    if s is None:
        print(f"{cfg_path}: no metrics_per_item.jsonl (run run_eval.py first)")
        return
    n, err, named = s["n"], s["err"], s["named"]
    lo, hi = s["ci"]
    print(f"{s['name']}  ({s['dataset']}/{s['split']}, {n['first'] + n['second']} decided games)")
    print(f"  first-listed team won : {n['first']:4d} games, wrong-winner claim in {err['first']:4d} ({100 * s['r_first']:5.1f}%)")
    print(f"  second-listed team won: {n['second']:4d} games, wrong-winner claim in {err['second']:4d} ({100 * s['r_second']:5.1f}%)")
    print(f"  difference (first - second): {100 * (s['r_first'] - s['r_second']):+.1f} pp, 95% CI [{100 * lo:+.1f}, {100 * hi:+.1f}]")
    print(f"  wrong claims naming the first-listed team: {named['first']}, the second-listed team: {named['second']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", required=True, help="config paths (globs ok)")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()
    paths = [p for pat in args.configs for p in (sorted(glob.glob(pat)) or [pat])]
    for p in paths:
        analyse(p, args.results_dir)


if __name__ == "__main__":
    main()
