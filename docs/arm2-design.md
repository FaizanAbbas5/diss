# Arm 2 build spec — learned representation → projector → frozen LLM

Contract for the Arm-2 implementation session. Context assumed: read
[plan.md](plan.md) (§Arm 2) and [pilot-w2-groq.md](pilot-w2-groq.md) first.
Goal: **fully built and CPU/T4-verified before HPC access arrives**, so the
only HPC step is scaling the RotoWire training run.

## What exists and must be reused (do not rebuild)

- `t2t.data` — canonical `Table`/`Section`/`Example`; E2E + RotoWire loaders.
- `t2t.serialise.to_markdown` — the frozen serialisation (augmentation
  variant still shows the table as markdown).
- `t2t.prompts` — registry; the `{table}` placeholder in each prompt is the
  **insertion point** for soft tokens (see Generation below). Prompt name
  comes from config — the baseline prompt gets frozen after the week-2
  battery, so nothing here may hardcode `faithful_v1`.
- `t2t.config` — config hash / run dirs / seeds; every training run gets a
  hashed run dir like generation runs do.
- `t2t.eval.*` — the full metric suite runs unchanged on Arm-2 outputs.
- `t2t.eval.numbers.extract_numbers` + `t2t.eval.relations` alias machinery —
  reuse for reference⇄cell alignment (aux labels + reference filtering).

## Architecture (decided; deviations need a written reason)

Table → **encoder** → k soft vectors → **projector** → frozen LLM embeddings.

- **Cell embedding**: column-ID embedding (per-dataset vocab) + value
  encoding. Numeric: small MLP over `[z_scored_value, log1p(|v|), sign,
  is_int]`, z-stats computed per column on the train split and saved with
  the checkpoint. String: hash-bucket embedding (2048 buckets). None: a
  learned null vector.
- **Rows**: mean-pool a row's cell embeddings + row-entity handling via the
  key column's value encoding → row vector. Permutation-invariant across
  rows by construction (no positional encoding over rows).
- **Cross-row interaction**: 1–2 transformer layers over row vectors
  (d≈256). This is where comparisons (who won / who led) can be computed —
  the pilot showed relational facts are exactly what the LLM gets wrong.
- **Mini Q-Former**: k=16 learned query vectors cross-attend over row
  vectors → k output vectors.
- **Projector**: linear (or 2-layer MLP) from d=256 to the LLM's hidden
  size — **read from `model.config.hidden_size` at runtime** (Qwen2.5-0.5B:
  896, 7B: 3584); never hardcode.
- Total trainable params: a few million. The LLM stays 100% frozen — no
  LoRA inside the LLM; call the module a *projector* in all writing.

## Training (decided)

- **Loss**: LM cross-entropy on the reference summary through the frozen
  LLM. Label mask: prompt tokens and soft-token positions = -100; only
  reference tokens contribute.
- **Aux loss**: BCE on per-cell "mentioned in reference" labels, weight
  ~0.3 (tune later). Labels via alignment: a cell counts as mentioned if
  its numeric value appears in the reference (reuse `extract_numbers`) or
  its string value / entity name word-boundary-matches (reuse the alias
  approach from `relations.py`). Write `t2t/encode/align.py` + tests.
- **Two conditioning variants, separate checkpoints**:
  - `replacement`: prompt's `{table}` slot ← soft tokens only.
  - `augmentation`: `{table}` slot ← markdown table, soft tokens prepended
    to it. Control condition for eval: `random`: same k random-but-fixed
    vectors (proves gains aren't just extra tokens).
- **Reference filtering** (the 45%-extrinsic problem, plan.md): config flag
  `filter_references: true` → keep only reference sentences that contain at
  least one aligned fact and no relational error (reuse claims/relations
  code). Train both filtered and unfiltered on E2E; decide for RotoWire
  from that ablation. This is a *dissertation decision point* — keep both
  paths runnable.
- **Optimisation**: AdamW on encoder+projector only, lr 1e-4, cosine,
  ~5% warmup, grad accumulation to an effective batch of 16–32.
  Checkpoint every N steps to the run dir (Drive/HPC-safe), resumable —
  Colab sessions die; assume any run can be killed at any step.
- **Frozen LLM on GPU**: 4-bit NF4 (bitsandbytes Linear4bit backprops
  gradients w.r.t. *inputs*, which is all we need), gradient checkpointing
  on, `use_cache=False` during training. On CPU smoke: fp32 0.5B, no
  quantisation.

## Generation with soft tokens (`t2t/generate/soft.py`)

Render the chat prompt with a **sentinel string** in the `{table}` slot,
tokenise, split ids at the sentinel, then build
`inputs_embeds = [embed(prefix_ids), soft_or_table_embeds, embed(suffix_ids)]`
and call `model.generate(inputs_embeds=..., attention_mask=...)`.
Gotchas that cost time if ignored:
- soft tokens must be cast to the embedding dtype (fp16 under 4-bit);
- left padding + attention masks must cover the soft positions;
- greedy decoding, identical generation config to Arm 1;
- output records identical schema to `hf.py` (id / output / seq_logprob)
  so the entire eval + dashboard stack works unchanged.

## File layout to create

```
src/t2t/encode/{__init__,numeric,table_encoder,align}.py
src/t2t/train/{__init__,dataset,collate,loop}.py
src/t2t/generate/soft.py
experiments/run_training.py            # --config configs/arm2_*.yaml
configs/arm2_e2e_smoke_cpu.yaml        # 0.5B fp32, 20 items, few steps
configs/arm2_e2e_overfit.yaml          # THE gate: 100 items -> near-zero loss
configs/arm2_e2e_t4.yaml               # full E2E training, 7B 4-bit
configs/arm2_rotowire_hpc.yaml         # the HPC run
tests/test_{numeric,encoder,align,collate,soft_split}.py
```

## Verification gates, in order (none need HPC)

1. Unit tests: encoder shape/permutation-invariance (shuffling player rows
   must not change output), numeric encoding, alignment labels on crafted
   references, sentinel prompt splitting, collate masks.
2. CPU smoke (`arm2_e2e_smoke_cpu.yaml`, Qwen2.5-0.5B fp32): forward +
   backward run, loss visibly decreases over ~50 steps, checkpoint saves
   and resumes.
3. **Overfit gate** (plan.md): 100 E2E items → LM loss near zero, and
   generation on those items reproduces references near-verbatim. If this
   fails, nothing downstream is trustworthy. CPU-overnight or free-T4.
4. End-to-end dry run: trained smoke checkpoint → `soft.py` generation →
   `run_eval` → `show_results` all work on both variants + random control.
5. Only then: `arm2_e2e_t4.yaml` on Colab, `arm2_rotowire_hpc.yaml` on HPC.

## Dependencies on other workstreams

- **Baseline prompt freeze** (battery on Colab, pending): configs reference
  the prompt by name; when the battery picks a winner, update configs, no
  code change.
- **Frozen-Qwen re-confirmation** (pilot doc): if Qwen shows digit-copying
  *failures* on RotoWire (unlike Llama), the numeric-encoding design gains
  importance — note it in the training ablation plan but build as specced.
- Fact-sheet arm is NOT part of this spec — it has its own build contract,
  [factsheet-arm-design.md](factsheet-arm-design.md), sequenced *before*
  this build. Coordination point: that build refactors the `_GameFacts`
  derivation core into a shared `t2t/facts.py`; if it has already landed,
  import derivations from there (e.g. for reference filtering) instead of
  from `t2t.eval.relations`.

## Success definition for the build session

`pytest` green including new tests; gates 1–4 passed and their artefacts
(smoke run dir, overfit curve, dry-run metrics) committed under `results/`
notes or referenced in a short `docs/arm2-build-log.md`; HPC config ready
so the RotoWire run is a one-command start.
