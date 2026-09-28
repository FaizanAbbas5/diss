# How to run and test the pipeline

Everything below runs from the repo root. Steps 1 and 2 are one-time
setup; after that, testing is the three commands in step 3, or the
dashboard (step 0), which wraps the same commands in a web UI.

## 0. Dashboard (optional, replaces the terminal loop)

```powershell
pip install -e ".[ui]"              # one-time
python -m streamlit run ui/app.py   # opens http://localhost:8501
```

Three pages:

- **Run experiments**: pick a config, see its status (generated n /
  limit, metrics present), and launch *Generate*, *Evaluate* (with
  `--claims` / `--parent` toggles), or both chained, with a live log
  tail and a Stop button. Jobs are the same resumable scripts as below.
- **Results browser**: aggregate metric tiles per run, then per-item
  cards with the table, references, generated text (unsupported numbers
  highlighted), NLI claim verdicts, the exact prompt, and seq-logprob.
  Items can be filtered to flagged ones, and runs without metrics can
  be evaluated from this page.
- **Compare runs**: the `compare_runs.py` table as a sortable dataframe
  with CSV download, plus per-metric bar charts.

## 1. Setup (one-time)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

(On Linux: `source .venv/bin/activate` instead of the second line.)

## 2. Sanity check (one-time, ~1 min)

```powershell
pytest
```

Expected: `159 passed`, the full unit-test suite (evaluators, fact
sheet, prompts, encoder, loaders).

## 3. Run the test pipeline (CPU, small model)

Three commands: generate, aggregate metrics, per-item view.

```powershell
python experiments/run_generation.py --config configs/e2e_smoke_cpu.yaml
python experiments/run_eval.py       --config configs/e2e_smoke_cpu.yaml --claims
python experiments/show_results.py   --config configs/e2e_smoke_cpu.yaml
```

(`--claims` runs NLI claim verification; first use downloads a ~370 MB
model. Drop the flag for a faster metrics-only pass.)

- `run_generation.py` sends 3 E2E validation tables through
  Qwen2.5-0.5B-Instruct on CPU (~1 GB model download on first run, then
  a few seconds per item). Output is cached: rerunning skips finished
  items, so interrupting is safe.
- `run_eval.py` writes aggregate metrics (add `--parent` to include
  PARENT).
- `show_results.py` prints one block per item: reference text,
  generated text, then the metrics:

```
==============================================================================
[e2e-validation-0001]
TABLE     : name[Alimentum], area[city centre], familyFriendly[no], near[Burger King]
REFERENCE : Alimentum is not family-friendly, and is near the Burger King in the city centre.
            (+6 more reference(s))
GENERATED : Alimentum, an adult-oriented eatery located in the city centre, offers a
            sophisticated dining experience with its focus on fine cuisine and a welcoming
            atmosphere for families seeking a more upscale culinary adventure.
METRICS   : number-check: 0 number(s), 0 unsupported | PARENT F1: 0.051
==============================================================================
[e2e-validation-0002]
TABLE     : name[Alimentum], area[city centre], familyFriendly[yes]
REFERENCE : Alimentum is a family-friendly place in the city centre.
            (+3 more reference(s))
GENERATED : Alimentum is a family-friendly restaurant located in the city centre, offering
            delicious cuisine at reasonable prices.
METRICS   : number-check: 0 number(s), 0 unsupported | PARENT F1: 0.376
==============================================================================
```

How to read the metrics line:

- **number-check**: every number in the generated text is checked
  against the table, including simple derivations such as margins and
  sums. Unsupported numbers are listed explicitly. E2E outputs rarely
  contain digits, so this metric matters mainly on RotoWire.
- **PARENT F1**: table-aware overlap with the references; low values
  flag text unsupported by table and reference. Above, the hallucinated
  description ("adult-oriented ... welcoming atmosphere for families")
  scores 0.051 while the faithful one scores 0.376.
- **seq-logprob**: length-normalised log-probability of the generation
  (closer to 0 means more confident), recorded for every run.
- **claims entailed** (with `--claims`): each clause unit is verified
  against the table rows it mentions with an NLI model; units judged
  NEUTRAL (unsupported) or CONTRADICTION are printed under the item:

  ```
  METRICS   : ... | claims entailed: 0/1
    CONTRADICTION: Alimentum, an adult-oriented eatery ... welcoming atmosphere for families ...
  ```

Useful flags for `show_results.py`: `--n 20` (how many items) and
`--no-parent` (skip PARENT for a quicker look).

## Comparing runs

After running several battery configs and their evaluations:

```powershell
python experiments/compare_runs.py --configs configs/battery_full/rotowire_*.yaml
```

prints one row per run (number support, claim entailment,
contradictions, PARENT F1, seq-logprob).

## 4. Where the files live

Each config gets one directory: `results/<name>-<confighash>/`

| file | contents |
| --- | --- |
| `meta.json` | full config + hash (provenance) |
| `generations.jsonl` | one line per item: id, output, seq-logprob |
| `metrics.json` | aggregates from `run_eval.py` |
| `metrics_per_item.jsonl` | per-item metric results |

## 5. Real model (GPU)

The same three commands with a GPU config, for example
`--config configs/final/e2e_baseline.yaml`, run Qwen2.5-7B-Instruct in
4-bit. Without a local GPU, open `colab/baseline_e2e.ipynb` in Colab
(T4 runtime), set `REPO_URL`, and run all cells; on a SLURM cluster see
`hpc/README.md`.

## 6. Groq backend (optional, no GPU needed)

With a key from console.groq.com (`$env:GROQ_API_KEY = "gsk_..."`), a
config with `backend: groq` generates through Groq's API. This was used
for metric-calibration pilots only; it is not the dissertation baseline
(see README). `seq-logprob` shows `-` for Groq runs because the API
does not return log-probabilities.

## 7. Trying variations

Copy a YAML in `configs/` and edit it, for example `limit`, `prompt`
(registered names are in `src/t2t/prompts/__init__.py`), `model`, or
`dataset: rotowire` with `split: validation`. Each edited config hashes
to its own results directory, so runs never overwrite each other.
