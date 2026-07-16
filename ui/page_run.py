"""Run page: pick a config, launch generation/eval, watch the live log."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import dotenv_values

from t2t.config import load_config, run_dir

import common
import jobs


def _log_file(stem: str) -> Path:
    return common.LOGS_DIR / f"{stem}-{time.strftime('%Y%m%d-%H%M%S')}.log"


def _start(key: str, label: str, steps: list[list[str]], stem: str) -> None:
    jobs.start_job(key, label, steps, _log_file(stem), cwd=common.ROOT)
    st.rerun()


def _groq_key_present() -> bool:
    if os.environ.get("GROQ_API_KEY"):
        return True
    env_file = common.ROOT / ".env"
    return bool(env_file.exists() and dotenv_values(env_file).get("GROQ_API_KEY"))


def page() -> None:
    st.title("🚀 Run experiments")

    configs = common.list_configs()
    if not configs:
        st.info("No YAML configs found under configs/.")
        return

    options = [str(p) for p in configs]
    sel = st.selectbox(
        "Config",
        options,
        format_func=lambda p: Path(p).relative_to(common.ROOT).as_posix(),
        key="run-config",
    )
    sel_path = Path(sel)

    try:
        cfg = load_config(sel_path)
    except Exception as e:  # noqa: BLE001 — show any parse problem in the UI
        st.error(f"Could not parse config: {e}")
        return

    rd = run_dir(cfg, common.RESULTS_DIR)
    gen_path = rd / "generations.jsonl"
    n_done = len(common.read_jsonl(gen_path))
    limit = cfg.get("limit")
    backend = cfg.get("backend", "hf")

    left, right = st.columns([1, 1.4], gap="large")

    with left:
        st.subheader("Config", divider="gray")
        summary = {
            "dataset": f'{cfg.get("dataset", "?")} / {cfg.get("split", "?")}',
            "model": cfg.get("model", "?"),
            "backend": backend,
            "prompt": cfg.get("prompt", "?"),
            "limit": limit if limit is not None else "all",
            "results dir": rd.name,
        }
        st.table({"setting": list(summary.keys()), "value": [str(v) for v in summary.values()]})
        with st.expander("Raw YAML"):
            st.code(sel_path.read_text(encoding="utf-8"), language="yaml")

        st.subheader("Status", divider="gray")
        if limit:
            st.progress(min(1.0, n_done / limit), text=f"{n_done} / {limit} generated")
        else:
            st.caption(f"{n_done} generated (no limit set)")
        st.caption(
            ("✅" if (rd / "metrics.json").exists() else "—")
            + " metrics.json"
            + ("  ·  generation is resumable: rerunning skips finished items" if n_done else "")
        )
        if backend == "groq":
            if _groq_key_present():
                st.caption("🔑 GROQ_API_KEY found (env or .env)")
            else:
                st.error("GROQ_API_KEY is not set — add it to .env or the environment before generating.")
        elif "7B" in str(cfg.get("model", "")) and cfg.get("quantization") != "none":
            st.caption("⚠️ Large model — meant for Colab/GPU, will be very slow locally.")

    with right:
        st.subheader("Launch", divider="gray")
        opt_claims = st.checkbox(
            "NLI claim verification (--claims)", value=True,
            help="First use downloads a ~370 MB MNLI model.",
        )
        opt_parent = st.checkbox("PARENT (--parent)", value=True)

        gen_cmd = [sys.executable, str(common.ROOT / "experiments" / "run_generation.py"), "--config", str(sel_path)]
        eval_cmd = [sys.executable, str(common.ROOT / "experiments" / "run_eval.py"), "--config", str(sel_path)]
        if opt_claims:
            eval_cmd.append("--claims")
        if opt_parent:
            eval_cmd.append("--parent")

        job = jobs.get_job(sel)
        running = job is not None and job.status == "running"
        name = cfg.get("name", sel_path.stem)

        b1, b2, b3 = st.columns(3)
        if b1.button("▶ Generate", disabled=running, width="stretch"):
            _start(sel, f"generate · {name}", [gen_cmd], sel_path.stem)
        if b2.button("▶ Evaluate", disabled=running or n_done == 0, width="stretch",
                     help="Needs generations first." if n_done == 0 else None):
            _start(sel, f"evaluate · {name}", [eval_cmd], sel_path.stem)
        if b3.button("▶ Generate + Evaluate", disabled=running, width="stretch"):
            _start(sel, f"generate+evaluate · {name}", [gen_cmd, eval_cmd], sel_path.stem)

        common.job_panel(sel)
        if job is not None and job.status == "done":
            st.success("Finished — see the **Results browser** page for per-item output.")

    others = [j for j in jobs.running_jobs() if j.key != sel]
    if others:
        st.divider()
        st.caption("Also running: " + " · ".join(j.label for j in others))

    with st.expander("Maintenance"):
        if st.button("Run unit tests (pytest)", disabled=jobs.get_job("pytest") is not None
                     and jobs.get_job("pytest").status == "running"):
            _start("pytest", "pytest -q", [[sys.executable, "-m", "pytest", "-q"]], "pytest")
        common.job_panel("pytest")
