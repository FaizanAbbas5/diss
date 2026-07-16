# How to run and test the pipeline

Everything below runs from the repo root. Steps 1–2 are one-time setup;
after that, testing is the three commands in step 3 — or the dashboard
(step 0), which wraps the same commands in a web UI.

## 0. Dashboard (optional, replaces the terminal loop)

```powershell
pip install -e ".[ui]"              # one-time
python -m streamlit run ui/app.py   # opens http://localhost:8501
```

Three pages:

- **Run experiments** — pick a config, see its status (generated n / limit,
  metrics present), launch *Generate*, *Evaluate* (with `--claims` /
  `--parent` toggles), or both chained, with a live log tail and a Stop
  button. Jobs are the same resumable scripts as below.
- **Results browser** — aggregate metric tiles per run, then per-item cards:
  table, reference(s), generated text with unsupported numbers highlighted,
  NLI claim verdicts, the exact prompt, seq-logprob. Filter to flagged items,
  sort by confidence, search, compute per-item PARENT on demand. Runs without
  metrics can be evaluated right from this page (works for Colab-synced runs
  too).
- **Compare runs** — the `compare_runs.py` table as a sortable dataframe with
  CSV download, plus per-metric bar charts and seq-logprob distributions.

## 1. Setup (one-time)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

(On Linux/HPC: `source .venv/bin/activate` instead of the second line.)

## 2. Sanity check (one-time, ~1 s)

```powershell
pytest
```

Expected: `20 passed`. These are the metric and serialisation unit tests —
if they pass, the number-checker and PARENT are behaving.

## 3. Run the test pipeline (CPU, small model)

Three commands: **generate → aggregate metrics → per-item view**.

```powershell
python experiments/run_generation.py --config configs/e2e_smoke_cpu.yaml
python experiments/run_eval.py       --config configs/e2e_smoke_cpu.yaml --claims
python experiments/show_results.py   --config configs/e2e_smoke_cpu.yaml
```

(`--claims` runs NLI claim verification — first use downloads a ~370 MB
model; drop the flag for a faster metrics-only pass.)

- `run_generation.py` sends 3 E2E validation tables through
  Qwen2.5-0.5B-Instruct on CPU (~1 GB model download on first run, then
  ~2–3 s per item). Output is cached: rerunning skips finished items, so
  it's safe to interrupt.
- `run_eval.py` writes aggregate metrics (add `--parent` to include PARENT).
- `show_results.py` prints what you asked to see — reference text, generated
  text, then the metrics — one block per item:

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

- **number-check** — every number in the generated text is checked against
  the table (including simple derivations like margins and sums). Unsupported
  numbers are listed explicitly. E2E outputs rarely contain digits, so this
  metric earns its keep on RotoWire.
- **PARENT F1** — table-aware overlap with the reference(s); low values flag
  text unsupported by table+reference. Above, the hallucinated description
  ("adult-oriented … welcoming atmosphere for families") scores 0.051 while
  the faithful one scores 0.376.
- **seq-logprob** — length-normalised log-probability of the generation
  (closer to 0 = more confident). An uncertainty-based hallucination signal;
  recorded automatically for every run.
- **claims entailed** (when evaluated with `--claims`) — each sentence is
  verified against the table rows it mentions with an NLI model; sentences
  judged NEUTRAL (unsupported) or CONTRADICTION are printed under the item:

  ```
  METRICS   : ... | claims entailed: 0/1
    CONTRADICTION: Alimentum, an adult-oriented eatery ... welcoming atmosphere for families ...
  ```

## Comparing runs (prompt battery)

After running several battery configs (`configs/battery/`) and their evals:

```powershell
python experiments/compare_runs.py --configs configs/battery/e2e_*.yaml
```

prints one row per run (number support, claim entailment, contradictions,
PARENT F1, seq-logprob) so the strongest baseline prompt is easy to pick.

Useful flags for `show_results.py`: `--n 20` (how many items), `--no-parent`
(skip PARENT, e.g. for a quick look).

## 4. Where the files live

Each config gets one directory: `results/<name>-<confighash>/`

| file | contents |
| --- | --- |
| `meta.json` | full config + hash (provenance) |
| `generations.jsonl` | one `{"id", "output"}` per item |
| `metrics.json` | aggregates from `run_eval.py` |
| `metrics_per_item.jsonl` | per-item number-check results |

## 5. Real model (Colab GPU)

The same three commands with `--config configs/e2e_baseline_colab20.yaml`
run n=20 through Qwen2.5-7B-Instruct (4-bit) — that needs a GPU. Easiest
path: push the repo to GitHub, open `colab/baseline_e2e.ipynb` in Colab
(T4 runtime), set `REPO_URL`, run all cells. See README for the git commands.

## 6. Groq pilot (optional, no GPU needed)

With a free key from console.groq.com you can generate through Groq's API —
useful for testing the metrics on lots of outputs quickly (it is not the
dissertation baseline; see README):

```powershell
$env:GROQ_API_KEY = "gsk_..."
python experiments/run_generation.py --config configs/e2e_groq_pilot.yaml
python experiments/run_eval.py       --config configs/e2e_groq_pilot.yaml --claims
python experiments/show_results.py   --config configs/e2e_groq_pilot.yaml
```

Note: `seq-logprob` shows `-` for Groq runs (APIs don't return logprobs).

## 7. Trying variations

Copy a YAML in `configs/` and edit — e.g. `limit`, `prompt`
(see `src/t2t/prompts/__init__.py` for registered names), `model`, or
`dataset: rotowire` with `split: validation`. Each edited config hashes to
its own results directory, so runs never overwrite each other.
