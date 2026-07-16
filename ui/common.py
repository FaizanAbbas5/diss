"""Shared helpers: paths, run/config scanning, cached data loading, job panel."""
from __future__ import annotations

import html
import json
import re
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from t2t import data as t2t_data
from t2t.eval.numbers import score_text

import jobs

ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = ROOT / "configs"
RESULTS_DIR = ROOT / "results"
LOGS_DIR = RESULTS_DIR / "_ui_logs"

_NUM_RE = re.compile(r"\d{1,3}(?:,\d{3})+|\d+\.\d+|\d+")  # keep in sync with t2t.eval.numbers


# ---------------------------------------------------------------- formatting

def pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def f3(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"


# ------------------------------------------------------------------ configs

def list_configs() -> list[Path]:
    return sorted(set(CONFIGS_DIR.rglob("*.yaml")) | set(CONFIGS_DIR.rglob("*.yml")))


# -------------------------------------------------------------------- runs

def scan_runs() -> list[dict]:
    """One entry per results/<run>/meta.json, newest first."""
    runs: list[dict] = []
    if not RESULTS_DIR.exists():
        return runs
    for meta_path in RESULTS_DIR.glob("*/meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            cfg = meta["config"]
        except (OSError, json.JSONDecodeError, KeyError):
            continue
        d = meta_path.parent
        gen_path = d / "generations.jsonl"
        n_gen = 0
        if gen_path.exists():
            with open(gen_path, encoding="utf-8") as f:
                n_gen = sum(1 for line in f if line.strip())
        mtimes = [p.stat().st_mtime for p in d.iterdir() if p.is_file()]
        runs.append(
            {
                "dir": d,
                "name": cfg.get("name", d.name),
                "hash": meta.get("config_hash", ""),
                "config": cfg,
                "n_gen": n_gen,
                "has_metrics": (d / "metrics.json").exists(),
                "mtime": max(mtimes) if mtimes else 0.0,
            }
        )
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return runs


def run_label(run: dict) -> str:
    cfg = run["config"]
    model = str(cfg.get("model", "?")).split("/")[-1]
    return (
        f'{run["name"]} · {cfg.get("dataset", "?")}/{cfg.get("split", "?")}'
        f' · {model} · {cfg.get("prompt", "?")} · {run["n_gen"]} gen'
    )


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.exists():
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


@st.cache_data(show_spinner=False)
def load_generations(dir_str: str, mtime: float) -> dict[str, dict]:
    """id -> generation record; mtime busts the cache when the file grows."""
    return {rec["id"]: rec for rec in read_jsonl(Path(dir_str) / "generations.jsonl")}


def gen_mtime(rd: Path) -> float:
    p = rd / "generations.jsonl"
    return p.stat().st_mtime if p.exists() else 0.0


def load_metrics(rd: Path) -> dict | None:
    p = rd / "metrics.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_per_item(rd: Path) -> dict[str, dict]:
    return {rec["id"]: rec for rec in read_jsonl(rd / "metrics_per_item.jsonl")}


# ----------------------------------------------------------------- datasets

@st.cache_data(show_spinner="Loading dataset…")
def load_examples(dataset: str, split: str, limit: int | None):
    """Cached t2t.data.load with an absolute data dir (cwd-independent)."""
    sub = "e2e" if dataset.startswith("e2e") else dataset
    return t2t_data.load(dataset, split, data_dir=str(ROOT / "data" / sub), limit=limit)


@st.cache_data(show_spinner="Computing per-item PARENT…")
def compute_parent_per_item(dir_str: str, mtime: float) -> dict[str, float]:
    from t2t.eval.parent_metric import compute_parent

    rd = Path(dir_str)
    cfg = json.loads((rd / "meta.json").read_text(encoding="utf-8"))["config"]
    gens = load_generations(dir_str, mtime)
    examples = {
        ex.id: ex for ex in load_examples(cfg["dataset"], cfg["split"], cfg.get("limit"))
    }
    ids = [i for i in gens if i in examples and examples[i].references]
    if not ids:
        return {}
    res = compute_parent(
        [gens[i]["output"] for i in ids],
        [examples[i].references for i in ids],
        [examples[i].table for i in ids],
    )
    return dict(zip(ids, res["per_item_f1"]))


def items_summary(gens: dict[str, dict], examples_by_id: dict, per_item: dict[str, dict]) -> pd.DataFrame:
    """One row per generated item with the fields the filters/sorts need."""
    rows = []
    for id_, rec in gens.items():
        pi = per_item.get(id_, {})
        if "n_numbers" in pi:
            nums = pi
        elif id_ in examples_by_id:
            nums = score_text(rec["output"], examples_by_id[id_].table)
        else:
            nums = {"n_numbers": 0, "unsupported": []}
        claims = pi.get("claims") or {}
        rows.append(
            {
                "id": id_,
                "output": rec.get("output", ""),
                "seq_logprob": rec.get("seq_logprob"),
                "n_numbers": nums.get("n_numbers", 0),
                "unsupported": nums.get("unsupported") or [],
                "n_unsupported": len(nums.get("unsupported") or []),
                "n_claims": claims.get("n_claims"),
                "n_entailed": claims.get("n_entailed"),
                "n_contradicted": claims.get("n_contradicted"),
            }
        )
    return pd.DataFrame(rows)


def highlight_unsupported(text: str, unsupported: list[float]) -> str:
    """HTML-escape text, marking numbers flagged by the number checker."""
    bad = {round(float(u), 6) for u in unsupported}
    out: list[str] = []
    last = 0
    for m in _NUM_RE.finditer(text):
        out.append(html.escape(text[last : m.start()]))
        tok = m.group(0)
        if round(float(tok.replace(",", "")), 6) in bad:
            out.append(
                '<span style="background:rgba(208,59,59,0.16);'
                "border-bottom:2px solid #d03b3b;border-radius:2px;"
                'padding:0 2px;font-weight:600" '
                'title="number not supported by the table">'
                f"{html.escape(tok)}</span>"
            )
        else:
            out.append(html.escape(tok))
        last = m.end()
    out.append(html.escape(text[last:]))
    return "".join(out)


# ---------------------------------------------------------------- job panel

_STATUS_ICON = {"running": "⏳", "done": "✅", "failed": "❌", "cancelled": "⚠️"}


def job_panel(key: str) -> None:
    """Status line + live-tailing log for the job registered under `key`.

    Polls every 2s while the job runs, then triggers one full rerun so the
    rest of the page picks up the new run state.
    """
    job = jobs.get_job(key)
    if job is None:
        return
    was_running = job.status == "running"

    @st.fragment(run_every=2.0 if was_running else None)
    def _panel() -> None:
        j = jobs.get_job(key)
        if j is None:
            return
        head_l, head_r = st.columns([5, 1])
        head_l.markdown(
            f"{_STATUS_ICON.get(j.status, '•')} **{j.label}** — {j.status}"
            f" · {j.elapsed():.0f}s"
        )
        if j.status == "running":
            if head_r.button("⏹ Stop", key=f"stop-{key}"):
                jobs.cancel(key)
        with st.container(height=360):
            st.code(jobs.tail(j.log_path) or "(no output yet)", language=None)
        if was_running and j.status != "running":
            time.sleep(0.2)  # let the worker thread finish writing status
            st.rerun(scope="app")

    _panel()
