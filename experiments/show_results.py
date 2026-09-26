#!/usr/bin/env python
"""Print per-item results of a finished run: reference, generated text, metrics.

    python experiments/show_results.py --config configs/e2e_smoke_cpu.yaml [--n 10] [--no-parent]
"""
from __future__ import annotations

import argparse
import json

from t2t import data
from t2t.config import load_config, run_dir
from t2t.eval.numbers import score_text
from t2t.eval.relations import score_text_relations


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--n", type=int, default=10, help="max items to show")
    ap.add_argument("--no-parent", action="store_true", help="skip per-item PARENT")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rd = run_dir(cfg, args.results_dir)
    gen_path = rd / "generations.jsonl"
    if not gen_path.exists():
        raise SystemExit(f"No generations at {gen_path}; run run_generation.py first")

    outputs: dict[str, str] = {}
    logprobs: dict[str, float] = {}
    with open(gen_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                outputs[rec["id"]] = rec["output"]
                if rec.get("seq_logprob") is not None:
                    logprobs[rec["id"]] = rec["seq_logprob"]

    # Claim-verification results are expensive to compute, so they are read
    # from run_eval.py --claims output rather than recomputed here.
    claims: dict[str, dict] = {}
    per_item_path = rd / "metrics_per_item.jsonl"
    if per_item_path.exists():
        with open(per_item_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if "claims" in rec:
                        claims[rec["id"]] = rec["claims"]

    examples = {
        ex.id: ex
        for ex in data.load(
            cfg["dataset"], cfg["split"],
            limit=cfg.get("limit"), sample_seed=cfg.get("sample_seed"),
        )
    }
    ids = [i for i in outputs if i in examples][: args.n]

    parent_f1: dict[str, float] = {}
    if not args.no_parent:
        from t2t.eval.parent_metric import compute_parent

        with_refs = [i for i in ids if examples[i].references]
        if with_refs:
            res = compute_parent(
                [outputs[i] for i in with_refs],
                [examples[i].references for i in with_refs],
                [examples[i].table for i in with_refs],
            )
            parent_f1 = dict(zip(with_refs, res["per_item_f1"]))

    for i in ids:
        ex = examples[i]
        s = score_text(outputs[i], ex.table)
        print("=" * 78)
        print(f"[{i}]")
        if "mr" in ex.meta:
            print(f"TABLE     : {ex.meta['mr']}")
        elif ex.meta:
            print(f"META      : {ex.meta}")
        print(f"REFERENCE : {ex.references[0] if ex.references else '(no reference)'}")
        if len(ex.references) > 1:
            print(f"            (+{len(ex.references) - 1} more reference(s))")
        print(f"GENERATED : {outputs[i]}")
        nums = (
            f"{s['n_numbers']} number(s), {s['n_derived']} via derivation, "
            f"{len(s['unsupported'])} unsupported"
        )
        if s["unsupported"]:
            nums += f" -> {s['unsupported']}"
        metrics_line = f"METRICS   : number-check: {nums}"
        if i in parent_f1:
            metrics_line += f" | PARENT F1: {parent_f1[i]:.3f}"
        if i in logprobs:
            metrics_line += f" | seq-logprob: {logprobs[i]:.3f}"
        rel = score_text_relations(outputs[i], ex.table)
        if rel["n_claims"]:
            metrics_line += f" | relations: {rel['n_wrong']}/{rel['n_claims']} wrong"
        if i in claims:
            c = claims[i]
            metrics_line += f" | claims entailed: {c['n_entailed']}/{c['n_claims']}"
        print(metrics_line)
        for c in rel["wrong"]:
            print(
                f"  REL-{c['type'].upper()}: {c['sentence'][:110]}"
                f"  [claimed: {c['claimed']} | table: {c['expected']}]"
            )
        for claim in claims.get(i, {}).get("claims", []):
            if claim["label"] != "entailment":
                print(f"  {claim['label'].upper():<13}: {claim['sentence']}")
    print("=" * 78)
    print(f"Shown {len(ids)} of {len(outputs)} generated items from {gen_path}")


if __name__ == "__main__":
    main()
