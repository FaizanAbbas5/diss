"""Results browser: aggregate metrics + per-item table/reference/generation view."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import streamlit as st
import yaml

from t2t.prompts import get_messages
from t2t.serialise import get_serialiser, to_markdown

import common
import jobs
import theme

_PAGE_SIZE = 10

_FILTERS = {
    "All items": lambda df: df,
    "Any issue": lambda df: df[
        (df["n_unsupported"] > 0)
        | (df["n_contradicted"].fillna(0) > 0)
        | (df["n_claims"].notna() & (df["n_entailed"] < df["n_claims"]))
    ],
    "Unsupported numbers": lambda df: df[df["n_unsupported"] > 0],
    "Contradicted claims": lambda df: df[df["n_contradicted"].fillna(0) > 0],
    "Not fully entailed": lambda df: df[df["n_claims"].notna() & (df["n_entailed"] < df["n_claims"])],
}


def _kpi_row(metrics: dict | None, n_gen: int) -> None:
    num = (metrics or {}).get("numbers", {})
    claims = (metrics or {}).get("claims", {})
    parent = (metrics or {}).get("parent", {})
    unc = (metrics or {}).get("uncertainty", {})

    row1 = st.columns(4)
    row1[0].metric("Items generated", n_gen)
    row1[1].metric(
        "Number support (micro)", common.pct(num.get("number_support_rate_micro")),
        help="Share of generated numbers found in the table or derivable from it. Shows — when no item contains a number.",
    )
    row1[2].metric(
        "Items w/ unsupported number", common.pct(num.get("pct_items_with_unsupported_number")),
        help="Lower is better.",
    )
    row1[3].metric(
        "Mean seq-logprob", common.f3(unc.get("mean_seq_logprob")),
        help="Length-normalised generation log-probability; closer to 0 = more confident. Unavailable for API (Groq) runs.",
    )
    row2 = st.columns(4)
    row2[0].metric(
        "Claims entailed (micro)", common.pct(claims.get("claim_entailed_rate_micro")),
        help="Sentence-level NLI verification against retrieved table rows (run eval with --claims).",
    )
    row2[1].metric(
        "Items w/ contradiction", common.pct(claims.get("pct_items_with_contradiction")),
        help="Lower is better.",
    )
    row2[2].metric(
        "PARENT F1", common.f3(parent.get("parent_f1")),
        help="Table-aware overlap with references (run eval with --parent).",
    )
    row2[3].metric("Total claims", claims.get("total_claims", "—"))


def _eval_panel(run: dict) -> None:
    """Launch run_eval.py for an existing run via a config snapshot (same hash)."""
    rd: Path = run["dir"]
    key = f"{rd}::eval"
    job = jobs.get_job(key)
    running = job is not None and job.status == "running"
    c1, c2, c3 = st.columns([1.2, 1.2, 2])
    opt_claims = c1.checkbox("--claims", value=True, key=f"ev-claims-{rd.name}")
    opt_parent = c2.checkbox("--parent", value=True, key=f"ev-parent-{rd.name}")
    if c3.button("▶ Run evaluation", disabled=running or run["n_gen"] == 0, key=f"ev-btn-{rd.name}"):
        snap = rd / "config.snapshot.yaml"
        snap.write_text(yaml.safe_dump(run["config"], sort_keys=False), encoding="utf-8")
        cmd = [sys.executable, str(common.ROOT / "experiments" / "run_eval.py"), "--config", str(snap)]
        if opt_claims:
            cmd.append("--claims")
        if opt_parent:
            cmd.append("--parent")
        log = common.LOGS_DIR / f"eval-{rd.name}.log"
        jobs.start_job(key, f"evaluate · {run['name']}", [cmd], log, cwd=common.ROOT)
        st.rerun()
    common.job_panel(key)


def _render_item(ex, rec: dict, row: dict, claims: dict | None, parent_f1: float | None) -> None:
    with st.container(border=True):
        badges = []
        if rec.get("seq_logprob") is not None:
            badges.append(f"seq-logprob `{rec['seq_logprob']:.3f}`")
        if parent_f1 is not None:
            badges.append(f"PARENT F1 `{parent_f1:.3f}`")
        if row["n_numbers"]:
            badges.append(f"numbers `{row['n_numbers'] - row['n_unsupported']}/{row['n_numbers']} supported`")
        if claims:
            badges.append(f"claims `{claims.get('n_entailed', 0)}/{claims.get('n_claims', 0)} entailed`")
        st.markdown(f"**`{rec['id']}`** &nbsp; " + " · ".join(badges))

        col_t, col_g = st.columns([1, 1.3], gap="medium")
        with col_t:
            table_md = to_markdown(ex.table) if ex else "_example not found in dataset_"
            if ex and sum(len(s.rows) * len(s.columns) for s in ex.table.sections) > 60:
                with st.expander("Table (large)", expanded=False):
                    st.markdown(table_md)
            else:
                st.markdown(table_md)
            if ex:
                with st.expander("Prompt sent to the model"):
                    cfg_prompt = st.session_state.get("_current_prompt")
                    serialise = get_serialiser(
                        st.session_state.get("_current_serialisation", "markdown")
                    )
                    if cfg_prompt:
                        for msg in get_messages(cfg_prompt, serialise(ex.table)):
                            st.caption(msg["role"])
                            st.code(msg["content"], language=None)

        with col_g:
            st.caption("Generated")
            st.markdown(
                '<div style="line-height:1.55">'
                + common.highlight_unsupported(rec["output"], row["unsupported"])
                + "</div>",
                unsafe_allow_html=True,
            )
            if row["unsupported"]:
                st.markdown(
                    f'<small style="color:{theme.STATUS["critical"]};font-weight:600">'
                    f"✗ unsupported number(s): {row['unsupported']}</small>",
                    unsafe_allow_html=True,
                )
            if rec.get("raw_output"):
                with st.expander("Raw output (before SUMMARY extraction)"):
                    st.code(rec["raw_output"], language=None)

            if claims and claims.get("claims"):
                st.caption("Claim verification (NLI)")
                for cl in claims["claims"]:
                    st.markdown(
                        f'{theme.claim_chip(cl["label"])} &nbsp;'
                        f'<span>{cl["sentence"]}</span> '
                        f'<small style="color:{theme.MUTED}">p={cl["probs"][cl["label"]]:.2f}</small>',
                        unsafe_allow_html=True,
                    )

            if ex and ex.references:
                st.caption("Reference")
                st.markdown(ex.references[0])
                if len(ex.references) > 1:
                    with st.expander(f"+{len(ex.references) - 1} more reference(s)"):
                        for ref in ex.references[1:]:
                            st.markdown(f"- {ref}")


def page() -> None:
    st.title("🔍 Results browser")

    runs = common.scan_runs()
    if not runs:
        st.info("No runs found under results/. Launch one from the **Run experiments** page.")
        return

    by_dir = {str(r["dir"]): r for r in runs}
    sel = st.selectbox(
        "Run", list(by_dir), format_func=lambda d: common.run_label(by_dir[d]), key="results-run"
    )
    run = by_dir[sel]
    cfg = run["config"]
    rd = run["dir"]
    st.session_state["_current_prompt"] = cfg.get("prompt")
    st.session_state["_current_serialisation"] = cfg.get("serialisation", "markdown")
    st.caption(
        f"`{rd.name}` · backend **{cfg.get('backend', 'hf')}** · model `{cfg.get('model', '?')}`"
        f" · prompt `{cfg.get('prompt', '?')}` · serialisation"
        f" **{cfg.get('serialisation', 'markdown')}** · limit {cfg.get('limit', 'all')}"
    )

    gens = common.load_generations(sel, common.gen_mtime(rd))
    metrics = common.load_metrics(rd)
    per_item = common.load_per_item(rd)

    _kpi_row(metrics, len(gens))
    if metrics is None:
        st.warning("No metrics.json yet — run the evaluation to fill in the aggregate metrics and claim labels.")
    with st.expander("Evaluate this run", expanded=metrics is None):
        _eval_panel(run)

    if not gens:
        st.info("No generations yet.")
        return

    try:
        examples_by_id = {
            ex.id: ex for ex in common.load_examples(cfg["dataset"], cfg["split"], cfg.get("limit"))
        }
    except Exception as e:  # noqa: BLE001 — dataset files may be absent locally
        st.error(f"Could not load dataset {cfg.get('dataset')}/{cfg.get('split')}: {e}")
        return

    df = common.items_summary(gens, examples_by_id, per_item)

    st.subheader("Per-item results", divider="gray")
    f1, f2, f3_, f4 = st.columns([1.2, 1.4, 1.4, 1])
    flt = f1.selectbox("Show", list(_FILTERS), key=f"filter-{rd.name}")
    sort_options = ["Dataset order", "Least confident first", "Most confident first"]
    want_parent = f4.toggle("Per-item PARENT", key=f"parent-{rd.name}",
                            help="Computed on demand and cached.")
    parent_scores: dict[str, float] = {}
    if want_parent:
        parent_scores = common.compute_parent_per_item(sel, common.gen_mtime(rd))
        sort_options.append("PARENT F1 (low first)")
    sort = f2.selectbox("Sort", sort_options, key=f"sort-{rd.name}")
    query = f3_.text_input("Search id / text", key=f"q-{rd.name}")

    view = _FILTERS[flt](df)
    if query:
        q = query.lower()
        view = view[
            view["id"].str.lower().str.contains(q, regex=False)
            | view["output"].str.lower().str.contains(q, regex=False)
        ]
    if sort == "Least confident first":
        view = view.sort_values("seq_logprob", na_position="last")
    elif sort == "Most confident first":
        view = view.sort_values("seq_logprob", ascending=False, na_position="last")
    elif sort == "PARENT F1 (low first)":
        view = view.assign(_pf=view["id"].map(parent_scores)).sort_values("_pf", na_position="last")

    n_pages = max(1, math.ceil(len(view) / _PAGE_SIZE))
    top = st.columns([3, 1])
    top[0].caption(f"{len(view)} of {len(df)} items match")
    page_no = top[1].number_input("Page", 1, n_pages, 1, key=f"page-{rd.name}-{flt}") if n_pages > 1 else 1

    for _, row in view.iloc[(page_no - 1) * _PAGE_SIZE : page_no * _PAGE_SIZE].iterrows():
        id_ = row["id"]
        _render_item(
            examples_by_id.get(id_),
            gens[id_],
            row,
            (per_item.get(id_) or {}).get("claims"),
            parent_scores.get(id_),
        )
