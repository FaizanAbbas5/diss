# Minimising Hallucinations in Table2Text — project bootstrap & experimental design

## Context

Dissertation (due ~early Sept 2026, **~9 weeks**, solo) testing whether preparing tabular data into a learned representation before a **frozen LLM** sees it reduces hallucination versus feeding the raw serialised table. Committed arms: (1) strong prompting baseline on the raw table, (2) neural encoder → adapter → frozen LLM, with a deterministic-template fallback. Arm 3 (hand-engineered features) is out of scope. Faithfulness is the primary metric, quality secondary.

Compute: **free Colab (T4 16GB) now; university HPC access to be requested**. Everything must therefore be checkpointable, cacheable, and runnable on a T4. Groq's free API tier is additionally available for metric-calibration pilots and cross-family robustness checks — never for the arm comparison itself (no embedding access, different serving stack, no logprobs).

Decisions made:
- Same frozen open-weights LLM, prompt, and decoding across all arms (adapter needs embedding-layer access, so no API models anywhere in the comparison).
- Arm 2 input design: **pilot both** — soft tokens *replacing* the table vs *augmenting* it — pick the winner on automatic metrics, manual-eval only the winner.
- Timeline forces scope: **2 datasets deep + 1 stretch**, not the original 5.

## Design

### Frozen LLM
**Qwen2.5-7B-Instruct, 4-bit NF4** (fits T4 for both inference and projector training; ungated on HF; strong enough that the baseline is credible). Swappable via config; `Qwen2.5-0.5B-Instruct` for local CPU smoke tests. Greedy decoding, fixed seeds, identical generation config across arms.

### Datasets (2 deep + 1 stretch)
1. **E2E cleaned** (restaurants, 8 fixed attributes) — *development* dataset: fast iteration, cheap annotation. Possible near-ceiling faithfulness for a 7B model becomes part of the story (effect vs table complexity), not a failure.
2. **RotoWire** (NBA box scores) — *headline* dataset: canonical table-to-text hallucination benchmark, numeric-heavy, real headroom.
3. **Stretch (automatic metrics only): numericNLG or MLB** — only if weeks 7–8 allow.

Test subsample n≈300/dataset (seeded) for automatic metrics; n=50 per arm per deep dataset for manual annotation, paired (same tables across arms).

### Arm 1 — baseline
Markdown serialisation (frozen across arms). Prompt battery (plain; faithfulness-constrained; chain-of-density adaptation), best on dev automatic metrics is frozen as *the* baseline prompt — that selection is what makes "strong baseline" defensible.

### Arm 2 — learned representation
- **Encoder: small permutation-invariant table encoder** (cell = column-ID embedding + value encoding; pool → row vectors → 1–2 cross-row attention layers → k≈16 learned queries → linear projector into LLM embedding space). A few million params; trains on a T4. Numeric value encoding gets first-class attention — exact-number recovery is the failure mode.
  - Not a flat MLP (row-order-sensitive, needs hand-designed features that would smuggle arm-3 engineering into arm 2) and not a NAM (additive structure blocks cross-player interactions). For the write-up: this is the BLIP-2/LLaVA-style *projector* pattern — not "LoRA", which edits LLM weights and would break the frozen-LLM claim.
- **Training:** end-to-end encoder+projector, LM loss on references through the frozen 4-bit LLM; auxiliary BCE on per-cell "mentioned in reference" labels (derived by number/name alignment). Replacement-mode and augmentation-mode variants. E2E first, then RotoWire (RotoWire training wants HPC or Colab Pro). **Training-target caveat (W4/5 decision):** ~45% of RotoWire reference claims are extrinsic (venue, dates, streaks) — naive LM loss teaches the model to emit ungroundable content. Plan: filter references to grounded sentences using the claims + relational checkers, or down-weight extrinsic sentences.
- **Pilot gate (end of week 6):** replacement vs augmentation on automatic metrics; winner gets manual eval. Augmentation gets a "random soft tokens" control.

### Insurance arm — fact-sheet templates (the fallback, scheduled early)
Deterministic rules select and derive content → Jinja renders a **textual fact sheet** → same frozen LLM, same prompt slot. Input can't contain fabrications; the LLM can still hallucinate at generation, which is what gets measured. Guarantees a substantive "prepared vs raw input" comparison and doubles as an engineered-representation reference point. **Pulled forward from W4 (decision 2026-07-06): needs no training compute, so it de-risks the HPC dependency and delivers the first arm comparison early. Full build spec: [factsheet-arm-design.md](factsheet-arm-design.md).**

### Evaluation suite (built before arm 2)
- **Number-accuracy checker** (custom, unit-tested, tiered exact/derived/unsupported): every output number must be in the table or derivable by simple ops. Checks existence only — attribution is the relational checker's job.
- **Relational checker "RG-lite"** (`t2t.eval.relations`, deterministic, always-on): derives winner/margin/per-team leaders from the table, pattern-extracts winner, margin, leader, stat-attribution, and team-label claims, and verifies them. Added after the W2 pilot showed relational errors are the dominant class and NLI cannot compare integers (this replaces the earlier "RG only if time" position — supervisor feedback, empirically vindicated).
- **Claim verification**: sentence-split → entity-match row retrieval + derived-result premise facts → NLI (DeBERTa-v3-MNLI) → % entailed claims. Covers qualitative/unsupported-context claims; relational classes are handled deterministically above.
- **PARENT** + BERTScore.
- **Sequence log-probability** (uncertainty signal): length-normalised logprob captured at generation time for every run; correlate with manual faithfulness labels in the W3 pilot (precedent: Xiao & Wang 2021 for data-to-text).
- **Manual protocol**: per-sentence taxonomy (made-up number / misattributed value / unsupported claim / contradiction / **extrinsic — plausible but unverifiable from the table** / omission), 50 outputs per arm per deep dataset, ~15% re-annotated for self-consistency. Automatic metrics validated against manual labels on the week-3 pilot, **with auto-vs-manual agreement reported separately on the relational subset** (the relational checker and the prepared-input arms derive the same facts, so checker bias must be measured, not assumed).
- **Scoring stance (proposed, confirm with supervisor)**: report two faithfulness rates per arm — *intrinsic* (contradictions/misattributions vs the table) and *strict groundedness* (extrinsic content also counts against). The W2 pilot measured extrinsic at ~45% of claims, too large to bury in one number.
- Paired stats: bootstrap CIs + Wilcoxon.

Suite framing for the write-up, per the Huang et al. (2025, ACM TOIS) survey taxonomy (§4.1.2): fact-based (number check, PARENT, optional RG), classifier-based (NLI claim verification, designed sentence-level to answer the granularity-mismatch critique), uncertainty-based (seq-logprob). QA-based metrics excluded (question generation over tables is the weak link; NLI covers the role with one model); LLM-as-judge excluded from the primary suite (self-preference risk judging same-family outputs) — optional W8 extra with a cross-family judge.

## Schedule (gates in bold)

- **W1**: scaffold, env, E2E loader + serialiser, generation harness, number-metric v0, Colab smoke run (7B-4bit, n=20). *Request HPC access.*
- **W2**: RotoWire loader, prompt battery, PARENT + NLI metrics, **freeze baseline prompt**.
- **W3**: full baseline runs (n=300 × 2 datasets), pilot manual annotation, **baseline numbers locked**.
- **W4**: fact-sheet template arm built + evaluated. Encoder/projector implementation starts.
- **W5–6**: projector training E2E → RotoWire; both variants. **Gate: if training isn't converging by end of W6, template arm is promoted to the primary realisation of arm 2.**
- **W7**: full arm-2 runs, manual annotation of winner (~200 summaries).
- **W8**: analysis, paired stats, stretch dataset if slack.
- **W9**: buffer + results chapter support.

## Verification practices

- Metric unit tests with adversarial hand-crafted cases.
- CPU smoke test end-to-end with Qwen2.5-0.5B before any GPU run.
- Projector training must overfit 100 E2E items before scaling.
- Every result dir carries config + hash; greedy decoding makes reruns bit-reproducible.

## Risks & mitigations

- **Free-Colab limits** → 4-bit model, resumable generation, Drive checkpoints; Colab Pro as cheap unblock; HPC requested W1.
- **Replacement mode loses exact numbers** → expected and reportable; augmentation + fact-sheet arms carry the positive-result chances.
- **Projector training fails** → W6 gate promotes template arm (sanctioned fallback).
- **E2E near-ceiling** → complexity-contrast framing; RotoWire is the headline.
- **Annotation overruns** → 50/arm/dataset cap, winner-only annotation for arm-2 variants.

## Open questions for supervisor

- Is 2-deep + 1-stretch datasets acceptable against the summary's "about five"?
- Does the fact-sheet arm's rule-based selection count as a lightweight arm-3 data point?

## Anchor papers

Wiseman et al. 2017 (RotoWire, RG/CS/CO) · Puduppully & Lapata 2019 (content selection/planning) · Dhingra et al. 2019 (PARENT) · Dušek et al. 2019 (E2E cleaned) · Tsimpoukelli et al. 2021 (Frozen), Li et al. 2023 (BLIP-2), Liu et al. 2023 (LLaVA) — frozen-LLM + projector precedent · Suadaa et al. 2021 (numericNLG) · Huang et al. 2025 (hallucination survey; §4.1.2 organises the eval chapter — PDF in project_info/papers/) · Xiao & Wang 2021 (uncertainty ↔ hallucination in data-to-text).
