#!/usr/bin/env python
"""Evaluate a finished generation run (number-accuracy always; PARENT opt-in).

    python experiments/run_eval.py --config configs/e2e_smoke_cpu.yaml [--parent]
"""
from __future__ import annotations

import argparse
import json

from t2t import data
from t2t.config import load_config, run_dir
from t2t.eval.numbers import score_run
from t2t.eval.relations import score_run_relations


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--parent", action="store_true", help="also compute PARENT")
    ap.add_argument(
        "--claims", action="store_true", help="also run NLI claim verification"
    )
    ap.add_argument(
        "--nli-model",
        default=None,
        help="override the NLI model for --claims",
    )
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

    examples = data.load(cfg["dataset"], cfg["split"], limit=cfg.get("limit"))
    tables = {ex.id: ex.table for ex in examples}
    refs = {ex.id: ex.references for ex in examples}
    ids = [i for i in outputs if i in tables]
    skipped = len(outputs) - len(ids)
    if skipped:
        print(f"Warning: {skipped} generations have no matching example; skipped")

    result = score_run({i: outputs[i] for i in ids}, tables)
    relations = score_run_relations({i: outputs[i] for i in ids}, tables)
    metrics: dict = {"numbers": result["aggregate"], "relations": relations["aggregate"]}
    if logprobs:
        vals = [logprobs[i] for i in ids if i in logprobs]
        metrics["uncertainty"] = {
            "mean_seq_logprob": sum(vals) / len(vals),
            "items": len(vals),
        }

    if args.parent:
        from t2t.eval.parent_metric import compute_parent

        with_refs = [i for i in ids if refs[i]]
        metrics["parent"] = compute_parent(
            [outputs[i] for i in with_refs],
            [refs[i] for i in with_refs],
            [tables[i] for i in with_refs],
        )

    claims_per_item: dict = {}
    if args.claims:
        from t2t.eval.claims import DEFAULT_NLI_MODEL, score_run_claims

        cres = score_run_claims(
            {i: outputs[i] for i in ids},
            tables,
            model_name=args.nli_model or DEFAULT_NLI_MODEL,
        )
        metrics["claims"] = cres["aggregate"]
        claims_per_item = cres["per_item"]

    (rd / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with open(rd / "metrics_per_item.jsonl", "w", encoding="utf-8") as f:
        for i in ids:
            line = {"id": i, **result["per_item"][i]}
            line["relations"] = relations["per_item"][i]
            if i in claims_per_item:
                line["claims"] = claims_per_item[i]
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(json.dumps(metrics, indent=2))
    print(f"Written to {rd / 'metrics.json'}")


if __name__ == "__main__":
    main()
