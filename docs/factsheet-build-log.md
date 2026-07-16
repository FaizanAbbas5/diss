# Fact-sheet arm — build log (2026-07-07)

Build session for [factsheet-arm-design.md](factsheet-arm-design.md). All
four verification gates passed. Sheet policy version: **v1**.

## What was built

- `src/t2t/facts.py` — derivation core (winner/margin/leaders + aliases)
  moved verbatim out of `t2t.eval.relations`; the checker imports it, so
  sheet and checker share one derivation **by construction**. Only
  addition: FGM/FGA in the per-player stats dict (sheet shooting lines;
  the checker never reads them). All 15 pre-existing relational-checker
  tests pass unchanged.
- `src/t2t/factsheet/` — `render.py` (selection + phrase building) +
  Jinja templates `rotowire.j2` / `e2e.j2`. Dispatch is structural:
  GameFacts-derivable tables get the box-score sheet; everything else
  (E2E, pathological box scores) gets labelled attribute/value lines.
- `t2t.serialise.get_serialiser(name)` registry (`markdown` | `factsheet`);
  config key `serialisation`, absent = markdown so **every existing config
  hash is untouched**. `hf.py`/`groq.py` resolve the serialiser from
  config. Results UI shows the run's serialisation and uses it in the
  "prompt sent to the model" view.
- Configs: `factsheet_{rotowire,e2e}_groq_pilot.yaml`,
  `factsheet_{rotowire,e2e}_colab.yaml` (prompt = `*/faithful_v1`
  placeholder until the battery freeze), `factsheet_e2e_smoke_cpu.yaml`.
- Tests: `test_facts.py` (13 property tests: tie/argmax/margin,
  missing/None columns, leader ties, row-order invariance, shared-city
  ambiguity) and `test_factsheet.py` (goldens for both templates, tie
  branch, fallback, registry dispatch, **number invariant**). Suite:
  **84 passed** (59 before).

## Gate results

1. **Unit tests** — green (84, including untouched `test_relations.py`).
   Invariant test: every number in a rendered sheet passes the tiered
   number check on crafted games incl. tie / all-None / empty-roster
   variants.
2. **Eyeball** — 5 RotoWire + 5 E2E sheets in
   [factsheet-samples.md](factsheet-samples.md); read and sane (incl. a
   correctly derived Harden triple-double). Bulk invariant: **50/50**
   RotoWire and **100/100** E2E validation sheets have every number
   table-supported.
3. **CPU smoke** — `fs-e2e-smoke-cpu` (Qwen2.5-0.5B,
   `serialisation: factsheet`, 3 items): generation → eval →
   show_results end to end. Residual qualitative hallucination visible
   ("sophisticated dining experience") — exactly the generation-stage
   residual the arm measures; sheet numbers all supported.
4. **Groq paired pilot** — `fs-rw-pilot-groq-16fda6037b` vs the existing
   raw-table run `rw-pilot-groq-520beca7c2`: same 50 validation games,
   same prompt (`rotowire/faithful_v1`), same model
   (llama-3.1-8b-instant), same decoding; only the `{table}` slot
   content differs. Numbers below.

## Paired pilot: raw markdown vs fact sheet (n=50, RotoWire validation)

| metric | baseline (markdown) | fact-sheet v1 |
|---|---|---|
| relational claims extracted | 559 | 683 |
| relational wrong rate | 4.8% (27) | **0.4% (3)** |
| items with ≥1 relational error | 42% | **6%** |
| wrong by type | 11 leader, 9 winner, 6 attribution, 1 team | 3 leader |
| number support (exact tier) | 99.8% | **100%** |
| items with unsupported number | 0% | 0% |
| NLI claim entailment | _pending_ | _pending_ |
| PARENT F1 | _pending_ | _pending_ |

Readings:

- The dominant relational classes from the W2 pilot — wrong winner,
  misattribution — go to **zero**; the 3 residual errors are all leader
  claims. Deterministic derivation removes exactly the failure mode it
  was designed to remove.
- Not claim-avoidance: the fact-sheet outputs make **more** checkable
  relational claims (683 vs 559), so the lower error rate is not the
  model saying less.
- **Coupling caveat** (methods chapter): the relational checker shares
  its derivation with the sheet, so part of this comparison is
  checker-aligned by construction. The NLI/PARENT rows (no shared code)
  are the arbiters; manual annotation of ~20 rendered sheets happens in
  the W3 pilot.
- **Cross-family caveat**: Groq-served Llama-8B, not the frozen LLM —
  pilot signal only. The dissertation number comes from
  `factsheet_rotowire_colab.yaml` vs the markdown baseline after the
  prompt freeze (one config-line update each).

## Resume (if the pending evals were interrupted)

Generations are cached; only the eval needs re-running. Fills the two
_pending_ rows above (NLI on CPU takes ~45 min per run):

```
.venv\Scripts\python experiments\run_eval.py --config configs/factsheet_rotowire_groq_pilot.yaml --claims --parent
.venv\Scripts\python experiments\run_eval.py --config configs/rotowire_groq_pilot.yaml --claims --parent
python experiments\compare_runs.py --configs configs/rotowire_groq_pilot.yaml configs/factsheet_rotowire_groq_pilot.yaml
```

## Open items

- Prompt placeholder `*/faithful_v1` in the colab configs until the
  battery freezes the baseline prompt.
- E2E Groq pilot (`factsheet_e2e_groq_pilot.yaml`) not yet run —
  near-null transformation, expected effect ≈ 0; run alongside the next
  Groq session if wanted.
- W3 manual pilot: annotate ~20 rendered sheets directly (derivation-bug
  check, per the design doc's mitigation #2).
