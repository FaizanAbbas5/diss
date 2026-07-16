# Minimising Hallucinations in Table2Text

Dissertation project testing whether preparing tabular data into a learned
representation before a **frozen LLM** sees it reduces hallucination versus
feeding the raw serialised table. Full experimental design, schedule, and
risk register: [docs/plan.md](docs/plan.md).

All arms share the same frozen open-weights LLM (Qwen2.5-7B-Instruct, 4-bit),
the same prompt, and greedy decoding — only the table preparation differs.

## Layout

```
src/t2t/            library: data loaders, serialisation, prompts, generation, metrics
  data/             E2E cleaned + RotoWire loaders -> canonical Table/Example types
  serialise/        deterministic Table -> markdown (frozen across arms)
  prompts/          prompt registry (baseline prompt frozen after week-2 selection)
  generate/         HF generation harness (greedy, resumable JSONL)
  eval/             number-accuracy checker, PARENT wrapper
configs/            one YAML per experiment; runs are keyed by config hash
experiments/        run_generation.py / run_eval.py
ui/                 Streamlit dashboard (run experiments, browse + compare results)
colab/              notebook entrypoints for GPU runs
tests/              metric + serialisation unit tests
results/            run outputs (gitignored), one dir per config hash
data/               downloaded datasets (gitignored)
```

## Setup (Windows, PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

On Linux/HPC replace the activation line with `source .venv/bin/activate`.
GPU extras (Colab/HPC only, bitsandbytes for 4-bit): `pip install -e ".[gpu]"`.

## Quick start

```powershell
# CPU smoke test: 3 E2E examples through Qwen2.5-0.5B, then metrics
python experiments/run_generation.py --config configs/e2e_smoke_cpu.yaml
python experiments/run_eval.py --config configs/e2e_smoke_cpu.yaml
python experiments/show_results.py --config configs/e2e_smoke_cpu.yaml
```

Step-by-step testing guide with expected output: [run.md](run.md).

## Dashboard

A local web UI wraps the same scripts — launch runs with live logs, browse
per-item results (table, references, generation, flagged claims, unsupported
numbers highlighted), and compare metrics across runs with charts and CSV
export:

```powershell
pip install -e ".[ui]"                    # one-time
python -m streamlit run ui/app.py         # opens http://localhost:8501
```

Runs started in the UI are the same resumable subprocesses as the CLI, so
the two are interchangeable; Colab results synced into `results/` show up in
the browser and compare pages automatically.

Generation is resumable: rerunning a config skips already-generated ids, so
interrupted Colab sessions just pick up where they left off. Outputs land in
`results/<name>-<confighash>/` (`meta.json`, `generations.jsonl`, `metrics.json`).

## Colab (real model, GPU)

Open `colab/baseline_e2e.ipynb` in Colab (T4 runtime), set `REPO_URL` to this
repository, run all cells. It reproduces the n=20 E2E baseline with
Qwen2.5-7B-Instruct in 4-bit.

## Groq backend (pilots only)

`backend: groq` in a config routes generation through Groq's free API
(fast, free, no GPU needed) — same JSONL output, same eval tooling:

```powershell
$env:GROQ_API_KEY = "gsk_..."   # console.groq.com
python experiments/run_generation.py --config configs/e2e_groq_pilot.yaml
python experiments/run_eval.py --config configs/e2e_groq_pilot.yaml --claims
```

Use it for metric calibration, annotation pilots, and cross-family
robustness checks. It is **not** a substitute for the baseline: the arm
comparison requires the same local frozen LLM everywhere (embedding access
for the adapter arm; identical serving stack; logprobs).

## Git (one-time setup)

```powershell
git init -b main
git add -A
git commit -m "Bootstrap Table2Text hallucination project"
# create a private GitHub repo, then:
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

The Colab notebook clones the GitHub repo, so push before running it.

## Metrics

- **Number accuracy** (built-in, `t2t.eval.numbers`): every number in the
  output must be in the table or derivable by same-column pairwise sums or
  differences. Digit numerals only in v0 — validate against the manual pilot.
- **Relational checker "RG-lite"** (`t2t.eval.relations`, always-on): derives
  winner, margin, and per-team leaders from the table and deterministically
  verifies winner / leader / stat-attribution / team-label claims extracted
  from the output. Catches misattributed-but-existing numbers and wrong
  winners that both the number checker and NLI miss.
- **NLI claim verification** (`t2t.eval.claims`): `run_eval.py --claims`.
  Splits the output into sentences, retrieves the table rows each sentence
  mentions, and classifies entailed/neutral/contradiction with an MNLI
  model (default `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, ~370 MB
  download on first use). Catches unsupported qualitative claims and
  contradictions that the number checker cannot see.
- **PARENT** (Dhingra et al. 2019): `experiments/run_eval.py --parent`. Uses
  the reference implementation vendored at `src/t2t/eval/_parent_impl.py`
  (Apache-2.0, from google-research/language) — the PyPI package does not
  install on Windows.
- **Sequence log-probability**: recorded automatically at generation time
  (uncertainty-based hallucination signal).

## Prompt selection (week 2)

Four baseline prompt candidates per dataset (`plain`, `faithful`, `strict`,
`structured` — see `src/t2t/prompts/__init__.py`). Run the battery configs in
`configs/battery/` on Colab, evaluate each with `--claims`, then compare:

```powershell
python experiments/compare_runs.py --configs configs/battery/e2e_*.yaml
```

The winner is frozen as *the* baseline prompt for all arms. The `structured`
prompts emit a fact list then a `SUMMARY:` line; the harness strips the
scratch work automatically (raw text kept as `raw_output`).
