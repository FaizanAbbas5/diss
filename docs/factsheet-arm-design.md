# Fact-sheet arm build spec — deterministic preparation → frozen LLM

Contract for implementing the template/fact-sheet arm. Context assumed:
[plan.md](plan.md) (§Insurance arm) and [pilot-w2-groq.md](pilot-w2-groq.md).
This is the plan's fallback realisation of Arm 2, pulled forward because
(a) it needs **zero training compute** — full dissertation-grade runs fit
on the free T4, so it is immune to the HPC risk; (b) the pilot showed the
dominant hallucination class is relational (wrong winner / false leader /
misattribution), which is precisely what deterministic derivation fixes;
(c) most of the derivation machinery already exists in `relations.py`.

Framing for the write-up: **fallback realisation of Arm 2 and a free
engineered-representation reference point** (arm-3-lite question pending
with supervisor). The neural encoder remains the thesis's novel
contribution; this arm de-risks, it does not de-scope.

## Pipeline

```
Table ──deterministic selection + derivation──► fact sheet (plain text)
      ──fills the {table} slot of the SAME frozen prompt──► frozen LLM
      ──identical decoding──► output ──identical eval suite──► metrics
```

No LLM anywhere in sheet construction; the sheet cannot contain
fabrications. The LLM can still hallucinate at generation — that residual
rate is exactly what the arm measures.

## What exists and must be reused

- `t2t.eval.relations._GameFacts` — winner/margin/per-team-leaders (tie-
  aware) + roster/alias machinery. **Refactor required**: move the
  derivation core to a new shared module `t2t/facts.py`; `relations.py`
  imports it (claim-extraction patterns stay in `relations.py`). The
  relational checker's behaviour must not change — its 15 tests pin it.
- `t2t.serialise` — grows a registry (see Harness integration).
- `t2t.prompts` — unchanged. The prompt is held verbatim; only the slot
  content varies. The baseline prompt's phrase "box-score statistics" is
  slightly off for a fact sheet — **keep it anyway**: prompt constancy is
  the fair-comparison rule, the slot is the manipulated variable.
  State this explicitly in the write-up; examiners will ask.
- `t2t.generate.hf` / `groq` — unchanged apart from serialiser lookup.
- Full eval suite + `compare_runs.py` — runs unchanged on this arm.

## On sharing derivation code with the metric (decided, with rationale)

The fact sheet asserts facts the relational checker verifies. Sharing one
derivation module makes them identical **by construction** — a shared bug
would inject a wrong fact into the sheet *and* score the model's faithful
copy of it as correct, silently. Independent implementations of
`argmax(TEAM-PTS)` would however be near-identical anyway; divergence risk
buys little. Decision: **share `t2t/facts.py`**, and mitigate the
shared-bug risk three ways:
1. the module gets its own exhaustive property tests (ties, all-None
   columns, missing TEAM-PTS, negative/zero margins impossible, leaders on
   every category, deterministic ordering);
2. the **week-3 manual pilot annotates ~20 rendered sheets directly** for
   factual correctness — a sheet error is a derivation bug, fix and rerun;
3. NLI, PARENT, and manual labels remain arbiter metrics that share no
   code with the sheet.
Document this openly in the methods chapter (supervisor's coupling
caution).

## Content-selection policy (this IS the engineered representation)

Deterministic, tie-aware, ordering fixed (value desc, then name asc).
Versioned: config carries `factsheet_version: v1`; any policy change bumps
the version and therefore the config hash — old runs stay comparable.

RotoWire v1 sheet, in order:
1. **Result sentence**: "The {winner} defeated the {loser} {W} to {L}
   (margin {M})." plus season records "{team} are now {wins}-{losses}."
2. **Team lines**: PTS, FG_PCT, FG3_PCT, FT_PCT, REB, AST, TOV per team,
   as labelled lines.
3. **Leaders per team**: top scorer (PTS plus FGM/FGA shooting line), top
   rebounder, top assist player. Ties: list all tied players.
4. **Notable performances**: every player with PTS ≥ 20, REB ≥ 10, or
   AST ≥ 10; derived double-doubles / triple-doubles named as such.
5. **Secondary scorers**: next two scorers per team with PTS only.
Omitted by design: quarter-by-quarter story, minutes, steals/blocks/fouls
(unless they cross the notable threshold via category leaders). The sheet
is selection + derivation, not a table dump — that asymmetry vs the raw
markdown baseline is the experimental manipulation.

E2E v1 sheet: one labelled line per attribute, nothing derived. This is a
near-null transformation (8 attributes, nothing to select or compute) —
included for protocol symmetry; expected effect ≈ 0 and the write-up says
so. RotoWire is where this arm can win.

## Rendering

Jinja2 (already a dependency), templates in
`src/t2t/factsheet/templates/{rotowire,e2e}.j2`. Plain text; result and
notable facts as full sentences (the NLI premise work showed explicit
sentences verify best), stats as `label: value` lines. Deterministic
output — golden tests pin exact rendering for a crafted game.

## Harness integration

`t2t/serialise/__init__.py` gains a registry:
`get_serialiser(name) -> Callable[[Table], str]` with entries
`markdown` (default) and `factsheet`. Config key `serialisation`,
**absent = markdown** so every existing config hash is untouched.
`hf.py`, `groq.py` (and later `soft.py`) resolve the serialiser from
config instead of importing `to_markdown` directly. Note the Arm-2
interplay for later: the augmentation variant's visible table stays
`markdown`; `factsheet` is this arm's setting, not a global default.

## File layout to create

```
src/t2t/facts.py                          # shared derivations (moved core)
src/t2t/factsheet/{__init__,render}.py
src/t2t/factsheet/templates/{rotowire,e2e}.j2
configs/factsheet_rotowire_groq_pilot.yaml   # n=50, SAME items as rw-pilot
configs/factsheet_e2e_groq_pilot.yaml        # n=100, pipeline validation
configs/factsheet_rotowire_colab.yaml        # n=300, Qwen 4-bit, real run
configs/factsheet_e2e_colab.yaml
tests/test_facts.py                        # property tests on derivations
tests/test_factsheet.py                    # renderer goldens + invariants
```

## Verification gates, in order (none need HPC)

1. Unit tests green: `facts.py` properties; renderer goldens; registry
   dispatch; **invariant test — every number in a rendered sheet passes
   the tiered number check** (nothing unsupported can enter the sheet);
   `relations.py` tests unchanged after the refactor.
2. Eyeball gate: render 5 RotoWire + 5 E2E sheets and read them.
3. CPU smoke: 3 E2E items, Qwen2.5-0.5B, `serialisation: factsheet`,
   generation → eval → show_results end to end.
4. **Groq paired pilot**: `factsheet_rotowire_groq_pilot.yaml` uses the
   same 50 validation games as the existing raw-table pilot run
   (`rw-pilot-groq-520beca7c2`) — `compare_runs.py` gives the first
   prepared-vs-raw comparison on identical items. Cross-family caveat
   applies: pilot signal, not a dissertation number.
5. Colab, after the battery freezes the baseline prompt: baseline vs
   fact-sheet with Qwen2.5-7B-4bit, n=300 — the first real arm comparison
   of the dissertation.

## Dependencies and open items

- **Battery/prompt freeze** (pending Colab session): configs take the
  prompt by name; placeholder `*/faithful_v1` until frozen, then a
  one-line config update.
- **Extrinsic scoring stance** (plan.md, supervisor sign-off pending):
  affects how this arm's results are *reported*, not how it is built —
  expect it to shine on intrinsic faithfulness and reduce extrinsic
  content as a side effect (the sheet contains none to imitate).
- Not in scope: LLM polishing of the sheet, learned selection, any change
  to prompts or decoding, arm-3 proper.

## Success definition for the build session

`pytest` green including new tests and untouched `relations.py` tests;
gates 1–4 passed with artefacts (rendered sample sheets committed under
`docs/` or referenced, Groq paired comparison table saved); Colab configs
ready so the real comparison is one command after the prompt freeze; a
short `docs/factsheet-build-log.md` recording the gate results and the
paired-pilot numbers.
