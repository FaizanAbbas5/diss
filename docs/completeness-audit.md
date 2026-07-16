# Completeness audit — what's missing for the final implementation

Audited 2026-07-08 against [plan.md](plan.md), [arm2-design.md](arm2-design.md),
[factsheet-arm-design.md](factsheet-arm-design.md). Suite state: **123 tests
passing**. Verdict: ~85% of the *code* exists; the genuinely missing pieces
are concentrated in **experimental validity** (sampling, test split, stats)
and **week-3 tooling** (annotation), not in the arms themselves.

## A. Complete and verified

- **Arm 1 pipeline**: loaders (E2E, RotoWire), frozen markdown serialisation,
  prompt registry (4 candidates × 2 datasets), HF + Groq backends, greedy
  decoding, resumable JSONL, config-hash provenance.
- **Eval suite**: tiered number check; RG-lite relational checker (always-on);
  NLI claims with derived-result premises; vendored PARENT; seq-logprob
  capture. All pilot-calibrated on 150 real outputs.
- **Fact-sheet arm**: all 4 gates passed ([build log](factsheet-build-log.md)).
  Paired Groq pilot (same 50 games): relational wrong rate 4.8% → **0.4%**,
  items with relational error 42% → **6%**, while making *more* checkable
  claims (683 vs 559). Cross-family + coupling caveats logged.
- **Dashboard** (run/browse/compare), docs suite, memory of decisions.

## B. Built but not finished

| # | Item | State | What's left |
|---|------|-------|-------------|
| B1 | **Arm-2 overfit gate (gate 3)** | Training code, tests, smoke gates done; overfit run checkpointed at step 110/1500, lm-loss 2.16 | Finish the run (T4 ~1–2 h, or CPU overnight; resumes from checkpoint), confirm loss → ~0 + near-verbatim generations; then gate 4 dry-run and `arm2_e2e_t4.yaml`. Write `docs/arm2-build-log.md` recording gates |
| B2 | **Pending pilot evals** | Fact-sheet vs baseline NLI + PARENT rows "pending" | Two `run_eval --claims --parent` commands (GPU-minutes on Colab; ~45 min each on CPU) — commands in the build log |
| B3 | **Prompt battery / freeze** | 8 configs ready; 1 stray partial run | The Colab session: run battery, `compare_runs`, freeze winner, update `*/faithful_v1` placeholders in factsheet + arm2 configs |

## C. Missing and blocking for dissertation-final numbers

| # | Gap | Why it blocks | Effort |
|---|-----|---------------|--------|
| C1 | **Seeded random subsampling** — `data.load(limit=N)` takes the *first* N items | Plan commits to "n≈300 (seeded)"; first-N on RotoWire is temporally biased (season order). Must land **before any final run** — changes the item set, so never mix with first-N runs | ~1 h + tests |
| C2 | **Test-split configs** — every config uses train/validation | Selection happens on validation; final reported numbers must come from the held-out test split | ~30 min |
| C3 | **Paired statistics** — no bootstrap/Wilcoxon anywhere | "Arm X beats baseline" needs per-item paired tests + CIs (plan §Evaluation). New `experiments/compare_paired.py`: align ids across two run dirs, per-item deltas, Wilcoxon + bootstrap CI | ~½ day |
| C4 | **Manual-annotation tooling (W3 critical path)** — nothing exists | Sampler (paired across arms, seeded, 50/arm/dataset, ~15% duplicates for self-consistency, **blinded arm labels**), per-sentence CSV export with the 6-category taxonomy (incl. extrinsic), import + agreement stats, auto-vs-manual correlation **with the relational subset reported separately**, ~20 rendered fact sheets in the sample | 1–2 days |
| C5 | **BERTScore** (secondary quality metric per plan) | Quality dimension currently PARENT-only | ~2 h (`--bertscore` flag) |
| C6 | **Extrinsic dual-reporting** (blocked on supervisor sign-off) | Metrics must emit both *intrinsic faithfulness* and *strict groundedness* aggregates once the stance is agreed; NLI neutral/contradiction split already provides the raw signal | ~2 h once decided |

## D. Missing for execution logistics (code-light, friction-heavy)

| # | Gap | Notes | Effort |
|---|-----|-------|--------|
| D1 | **HPC onboarding pack** | Slurm script for `arm2_rotowire_hpc.yaml`, env bootstrap (venv + `pip install -e .`), data staging, resume flags. Written now = submit-and-go on day 1 of access | ~2 h |
| D2 | **Colab notebooks** beyond `baseline_e2e.ipynb` | Battery runner (8 configs + evals + compare table), arm-2 T4 training (overfit → full, Drive checkpoints), final-runs notebook. CLI works; notebooks cut session friction | ~½ day |
| D3 | **JSON serialiser + format mini-battery** | Open decision from the markdown-vs-JSON discussion; registry makes it ~20 lines + 2 configs. Settles the examiner question with data | ~2 h |
| D4 | **Repro freeze at final runs** | `pip freeze` lockfile + pin HF model revisions in configs when final runs start | ~1 h, scheduled not now |

## E. Explicitly deferred / stretch (non-blocking)

- Claims clause-level splitting + sentence-splitter initials fix — decide
  *after* W3 calibration quantifies the FP cost.
- **Extraction-recall upgrades from harvardnlp/data2text** (Wiseman 2017
  code release): port the `data_utils.py` alignment heuristics (spelled-out
  number words, pronouns, name variants) into `align.py` and the
  number/relations extractors — addresses the documented digits-only v0
  limitation. Decide after W3 calibration measures what v0 misses.
- **CS/CO metrics over RG-lite tuples**: adapt `non_rg_metrics.py` (pure
  Python) to our extractor's tuples; adds the missing omission/coverage
  dimension, with the repo's released gold tuples as a published target.
  Never present RG-lite numbers as comparable to published RG (different
  extractor). Do NOT attempt their LuaTorch IE models — dead toolchain.
- seq-logprob ↔ manual-label correlation analysis — part of the W3 analysis,
  fold into C4's reporting.
- Stretch dataset loader (numericNLG or MLB) — only if W7–8 slack exists.
- Cross-family LLM-judge — optional W8 extra.
- E2E fact-sheet Groq pilot — near-null transformation, run opportunistically.

## Recommended order

1. **C1 + C2 now** (small, and every final run depends on them).
2. **B1** — finish the overfit gate (background/overnight), unblocking the
   whole neural arm.
3. **B3** — the Colab session: battery → freeze → placeholders (+ B2 evals
   and D3's format check in the same sitting).
4. **C3 + C4** — stats and annotation tooling before W3 annotation starts.
5. **D1** — HPC pack while waiting on approval; **C5, C6, D2** alongside.

Rough remaining code effort: **3–4 focused days**, plus compute sessions.
