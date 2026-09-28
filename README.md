# Minimising Hallucinations in Table2Text

Dissertation project testing whether preparing tabular data before a
frozen LLM sees it reduces hallucination, compared with feeding the raw
serialised table. All arms share the same frozen open-weights LLM
(Qwen2.5-7B-Instruct, 4-bit), the same prompt, and greedy decoding; only
the table preparation differs. The full design and results are in the
dissertation report, and its appendices A and B are the user and
maintenance manuals for this code.

## Layout

```
src/t2t/            the library
  data/             E2E cleaned + RotoWire loaders -> canonical Table/Example types
  serialise/        deterministic Table -> markdown
  factsheet/        deterministic Table -> fact sheet (Jinja templates)
  prompts/          prompt registry (8 prompts)
  generate/         backends: local HF (greedy, resumable), soft-token, Groq API
  encode/           table encoder for the soft-token arm
  train/            training loop for the table encoder
  eval/             number check, relational checker, NLI claims, PARENT, BLEU, bootstrap
configs/            one YAML per experiment; runs are keyed by config hash
experiments/        command-line entry points (generation, evaluation, training, reports)
ui/                 Streamlit dashboard (run experiments, browse + compare results)
hpc/                SLURM scripts and cluster setup
colab/              notebook entry point for GPU runs
tests/              unit tests (159)
results/            run outputs, created at runtime, one dir per config hash
data/               datasets, auto-downloaded on first run
```

## Setup (Windows, PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

On Linux replace the activation line with `source .venv/bin/activate`.
Optional extras: `gpu` (bitsandbytes for 4-bit loading), `analysis`
(reports and figures), `ui` (dashboard).

## Quick start

```powershell
# CPU smoke test: 3 E2E examples through Qwen2.5-0.5B, then metrics
python experiments/run_generation.py --config configs/e2e_smoke_cpu.yaml
python experiments/run_eval.py --config configs/e2e_smoke_cpu.yaml
python experiments/show_results.py --config configs/e2e_smoke_cpu.yaml
```

A step-by-step guide with expected output is in [run.md](run.md).

## Dashboard

A local web UI wraps the same scripts: launch runs with live logs,
browse per-item results (table, references, generation, flagged claims,
unsupported numbers highlighted), and compare metrics across runs with
charts and CSV export.

```powershell
pip install -e ".[ui]"                    # one-time
python -m streamlit run ui/app.py         # opens http://localhost:8501
```

Runs started in the UI are the same resumable subprocesses as the CLI,
so the two are interchangeable, and results synced into `results/` from
elsewhere appear in the browser and compare pages automatically.

Generation is resumable: rerunning a config skips already-generated
items, so an interrupted run continues where it stopped. Outputs land in
`results/<name>-<confighash>/` (`meta.json`, `generations.jsonl`, and
after evaluation `metrics.json`, `metrics_per_item.jsonl`).

## Colab (real model, GPU)

Open `colab/baseline_e2e.ipynb` in Colab (T4 runtime), set `REPO_URL`,
and run all cells. The notebook clones the repository from a Git host,
so the code must be pushed to one first. It runs the same commands as
the CLI with the 7B model in 4-bit.

## Groq backend (pilots only)

`backend: groq` in a config routes generation through Groq's API (no
GPU needed), with the same JSONL output and the same evaluation
tooling; set `GROQ_API_KEY` (console.groq.com). It was used for metric
calibration pilots only. It cannot substitute for the local baseline:
the arm comparison needs embedding access and log-probabilities, which
the API does not provide.

## Metrics

- **Number check** (`t2t.eval.numbers`): every number in the output
  must be in the table or derivable by same-column pairwise sums or
  differences.
- **Relational checker "RG-lite"** (`t2t.eval.relations`): derives
  winner, margin, and leaders from the table and deterministically
  verifies winner, leader, and attribution claims extracted from the
  output. Catches misattributed-but-existing numbers and wrong winners,
  which the number check cannot see.
- **NLI claim verification** (`t2t.eval.claims`, `run_eval.py
  --claims`): splits the output into clause units, retrieves the table
  rows each unit mentions, and classifies entailed, neutral or
  contradiction with an MNLI model (default
  `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, ~370 MB download on
  first use).
- **PARENT** (Dhingra et al. 2019, `run_eval.py --parent`): table-aware
  overlap with the references, using the reference implementation
  vendored at `src/t2t/eval/_parent_impl.py` (Apache-2.0, from
  google-research/language); the PyPI package does not install on
  Windows.
- **BLEU** (`t2t.eval.bleu`): corpus BLEU with per-item counts so the
  report can bootstrap intervals.
- **Sequence log-probability**: recorded at generation time as an
  uncertainty signal.

## Prompt battery

Four prompts per dataset (`plain`, `faithful`, `strict`, `structured`;
see `src/t2t/prompts/__init__.py`). The battery configs live in
`configs/battery_v2/` (pilot sample) and `configs/battery_full/` (full
validation split); `configs/battery/` holds the earlier E2E versions.
Compare evaluated runs with:

```powershell
python experiments/compare_runs.py --configs configs/battery_full/rotowire_*.yaml
```

The `structured` prompts emit a fact list and then a `SUMMARY:` line;
the harness strips the scratch work automatically and keeps the raw
text as `raw_output`.
