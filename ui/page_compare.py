"""Compare page: aggregate metrics across finished runs, as table + charts."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import common
import theme

# key, chart title, value format, subtitle
_CHART_METRICS = [
    ("num_support", "Number support (micro)", ".1%", "share of generated numbers supported by the table · ↑ better"),
    ("unsup_items", "Items with an unsupported number", ".1%", "↓ better"),
    ("claims_ent", "Claims entailed (micro)", ".1%", "NLI-verified sentence rate · ↑ better"),
    ("contra_items", "Items with a contradiction", ".1%", "↓ better"),
    ("parent_f1", "PARENT F1", ".3f", "table-aware overlap with references · ↑ better"),
    ("logprob", "Mean seq-logprob", ".3f", "closer to 0 = more confident"),
]


def _rows(selected: list[dict]) -> pd.DataFrame:
    rows = []
    for run in selected:
        m = common.load_metrics(run["dir"]) or {}
        cfg = run["config"]
        num, claims = m.get("numbers", {}), m.get("claims", {})
        rows.append(
            {
                "dir": str(run["dir"]),
                "run": run["name"],
                "prompt": cfg.get("prompt", "?"),
                "model": str(cfg.get("model", "?")).split("/")[-1],
                "n": num.get("items", run["n_gen"]),
                "num_support": num.get("number_support_rate_micro"),
                "unsup_items": num.get("pct_items_with_unsupported_number"),
                "claims_ent": claims.get("claim_entailed_rate_micro"),
                "contra_items": claims.get("pct_items_with_contradiction"),
                "parent_f1": m.get("parent", {}).get("parent_f1"),
                "logprob": m.get("uncertainty", {}).get("mean_seq_logprob"),
            }
        )
    df = pd.DataFrame(rows)
    # Disambiguate battery runs that share a name but differ in prompt.
    if df["run"].duplicated().any():
        df["run"] = df["run"] + " · " + df["prompt"].str.split("/").str[-1]
    return df


def page() -> None:
    st.title("📊 Compare runs")

    runs = [r for r in common.scan_runs() if r["has_metrics"]]
    if not runs:
        st.info("No evaluated runs yet — run an evaluation first (Run page or Results page).")
        return

    by_dir = {str(r["dir"]): r for r in runs}
    sel = st.multiselect(
        "Runs",
        list(by_dir),
        default=list(by_dir)[:10],
        format_func=lambda d: common.run_label(by_dir[d]),
        key="compare-runs",
    )
    if not sel:
        st.caption("Select at least one run.")
        return

    df = _rows([by_dir[d] for d in sel])
    label_of = dict(zip(df["dir"], df["run"]))
    df = df.drop(columns=["dir"])

    display = df.copy()
    for col in ("num_support", "unsup_items", "claims_ent", "contra_items"):
        display[col] = (display[col] * 100).round(1)
    display = display.rename(
        columns={
            "num_support": "num support %",
            "unsup_items": "unsup items %",
            "claims_ent": "claims entailed %",
            "contra_items": "contra items %",
            "parent_f1": "PARENT F1",
            "logprob": "seq-logprob",
        }
    )
    st.dataframe(display, width="stretch", hide_index=True)
    st.download_button(
        "⬇ Download CSV",
        display.to_csv(index=False).encode("utf-8"),
        file_name="run_comparison.csv",
        mime="text/csv",
    )

    st.subheader("Charts", divider="gray")
    cols = st.columns(2)
    slot = 0
    for key, title, fmt, subtitle in _CHART_METRICS:
        sub = df[["run", key]].dropna().rename(columns={key: "value"})
        if sub.empty:
            continue
        with cols[slot % 2]:
            st.altair_chart(
                theme.metric_bar(sub, title=title, fmt=fmt, subtitle=subtitle),
                width="stretch",
            )
        slot += 1
    if slot == 0:
        st.caption("Selected runs have no comparable aggregate metrics yet.")

    with st.expander("Per-item seq-logprob distribution"):
        frames = []
        for d in sel:
            run = by_dir[d]
            gens = common.load_generations(d, common.gen_mtime(run["dir"]))
            vals = [r["seq_logprob"] for r in gens.values() if r.get("seq_logprob") is not None]
            frames.extend({"run": label_of[d], "value": v} for v in vals)
        if frames:
            st.altair_chart(theme.logprob_box(pd.DataFrame(frames)), width="stretch")
        else:
            st.caption("No seq-logprob values in the selected runs (API backends do not return logprobs).")
