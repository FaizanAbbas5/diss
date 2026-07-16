# Week-2 Groq pilot — findings (rev. 2 after supervisor feedback)

Pilot runs via Groq free API (`llama-3.1-8b-instant`, temperature 0):
E2E cleaned n=100, RotoWire n=50 validation items, `faithful_v1` prompts.
Purpose: exercise the metric suite on real outputs before the week-3
baseline runs. Model differs from the dissertation's frozen LLM (Qwen2.5-7B);
findings are about metrics and hallucination *classes*, not baseline numbers.

## Headline finding: hallucination is relational, and existence-checking masks it

- Every number in all 50 RotoWire summaries passes the *existence* check —
  each digit string appears somewhere in the table. **This is not the same
  as digit accuracy.** The deterministic relational checker found 6 claims
  pairing a real table number with the wrong player or stat ("Ty Lawson had
  12 rebounds" — Lawson has 4; the 12 exists elsewhere). The earlier
  "zero digit hallucination" phrasing (rev. 1) overstated the result.
- Full relational audit of 559 verifiable relational claims across 50
  summaries: **27 errors — 11 false leader claims, 9 wrong winners, 6
  misattributed stats, 1 wrong team label — touching 42% of summaries.**
- So the model is digit-*copying* reliable but relationally unreliable:
  wrong winner ("The Bucks secured a 102-107 victory" — Washington scored
  107), false leaders ("Harden led the Rockets with 24" — Howard had 26),
  wins/losses swaps, misattributed stats.

Implication for the thesis: the failure mode of a modern 8B on box scores
is **comparative/relational**, exactly the class that deterministic
derivation (fact-sheet arm) or a learned representation with pre-computed
comparisons (who won, by how much, who led each category) can guarantee.
A number-existence metric alone would call this model near-perfect.

**Model-transfer caveat (supervisor feedback):** all of the above is
`llama-3.1-8b-instant` on Groq's serving stack. Number-copying behaviour
can shift with model family and quantisation. Nothing here hardens into a
design decision until re-confirmed on the frozen Qwen2.5-7B-4bit — the
week-2 battery run `configs/battery/rotowire_faithful.yaml` (n=30 on the
Colab T4) doubles as that re-confirmation gate.

## Metric suite changes driven by the pilot

1. **Number checker → tiered** (exact / derived / unsupported): same-column
   pairwise derivations blanket most small integers on dense tables, so a
   single "supported" rate is misleadingly lenient. Attribution errors are
   invisible to this metric *by design*; see (3).
2. **NLI premise enrichment**: premises now lead with a derived game-result
   sentence. Before: 68/335 claims flagged contradiction at ~75% FP rate
   (NLI cannot compare 122 > 95, so true "X defeated Y" sentences were
   flagged). After re-scoring: 30 flipped to entailed/neutral; all
   genuinely-wrong winner claims stayed flagged.
3. **RG-lite relational checker added** (`t2t.eval.relations`, always-on in
   `run_eval`) — the supervisor's point: extending derived premises to
   leaders/attribution is rebuilding RG, so build it deterministically
   instead of hoping NLI learns to compare integers. Derives winner,
   margin, and per-team leaders (tie-aware); pattern-extracts winner /
   margin / leader / stat-attribution / team-label claims (precision-first:
   ambiguous clauses yield no claim); verifies against the table. It found
   every error class the NLI missed, including the 6 numeric attribution
   errors and 2 wrong winners the NLI scored leniently.
4. **NLI residual weaknesses** (deferred to week-3 calibration): verified-
   correct conjunctive two-player sentences still draw high-confidence
   contradictions; the regex sentence splitter breaks on initials ("J.J.").
   With RG-lite covering the relational classes deterministically, NLI's
   remaining role is qualitative/unsupported-context claims.

## Metric–arm coupling, and how it's mitigated

The relational checker derives facts to *measure* faithfulness; the
fact-sheet and encoder arms derive facts to *improve* it. The metric could
therefore flatter exactly the arms designed to beat it. Mitigations:
- the derivation/extraction code is independently unit-tested (15 cases:
  ties, inversions like "fell to", margin, clause splitting, word-boundary
  regression — which caught a real bug where "Lin" matched "line");
- flagged values were spot-checked directly against tables;
- the **week-3 manual calibration reports auto-vs-manual agreement
  separately on the relational subset**, so any checker bias is measured
  rather than assumed.

## Extrinsic content: a definition and a training decision (open)

45% of RotoWire claims are NEUTRAL — venue, dates, streaks, standings that
are genuinely absent from the box score.
- **Definition (proposed, needs supervisor sign-off):** add an explicit
  `extrinsic` category to the annotation taxonomy, and report two rates per
  arm: *intrinsic faithfulness* (contradictions + misattributions vs the
  table) and *strict groundedness* (extrinsic also counts against). One
  number would hide a 45%-of-claims judgement call.
- **Training:** RotoWire references contain this extrinsic content, so
  naive LM-loss training of Arm 2 teaches the model to emit ungroundable
  claims. Decision for W4/5: filter training references to grounded
  sentences (the claims + relations checkers can produce the filter) or
  down-weight extrinsic sentences in the loss.

## E2E is near-ceiling for a competent 8B

100 restaurants: 99% claims entailed, zero contradictions, all numbers
exact. E2E serves as fast-iteration and complexity-contrast dataset;
RotoWire carries the headroom.

## Process notes

- Groq free tier handled n=150 generations with automatic 429 pacing;
  resumable JSONL meant zero loss across rate-limit waits.
- CPU NLI (~350 pairs) took tens of minutes — week-3 claim evals run on the
  Colab GPU. The relational checker is model-free and instant.
- Stored claims labels in `results/rw-pilot-groq-520beca7c2/` predate the
  premise fix; re-evaluate before quoting per-item NLI labels from that run.
