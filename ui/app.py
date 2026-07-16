"""T2T Hallucination Lab — Streamlit dashboard.

Launch from the repo root:

    .venv\\Scripts\\streamlit run ui/app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
os.chdir(APP_DIR.parent)  # scripts and loaders assume repo-root cwd

import streamlit as st

st.set_page_config(
    page_title="T2T Hallucination Lab",
    page_icon="🧪",
    layout="wide",
)

import jobs
from page_compare import page as compare_page
from page_results import page as results_page
from page_run import page as run_page

nav = st.navigation(
    [
        st.Page(run_page, title="Run experiments", icon="🚀", url_path="run", default=True),
        st.Page(results_page, title="Results browser", icon="🔍", url_path="results"),
        st.Page(compare_page, title="Compare runs", icon="📊", url_path="compare"),
    ]
)

running = jobs.running_jobs()
if running:
    st.sidebar.info("⏳ running: " + " · ".join(j.label for j in running))
st.sidebar.caption(f"repo: {APP_DIR.parent}")

nav.run()
