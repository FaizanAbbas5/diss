"""Chart palette and Altair helpers.

Colors are the validated reference palette from the dataviz method:
single-hue (blue) marks for magnitude comparisons, reserved status colors
(always paired with an icon + text label, never color alone) for claim
verdicts.
"""
from __future__ import annotations

import altair as alt
import pandas as pd

BLUE = "#2a78d6"          # categorical slot 1 / sequential anchor
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"

STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

# claim label -> (icon + text chip, chip color). The warning slot uses the
# darker yellow step so the text stays legible on a light surface.
CLAIM_STYLE = {
    "entailment": ("✓ entailed", STATUS["good"]),
    "neutral": ("? unsupported", "#c98500"),
    "contradiction": ("✗ contradiction", STATUS["critical"]),
}


def claim_chip(label: str) -> str:
    text, color = CLAIM_STYLE.get(label, (label, MUTED))
    return (
        f'<span style="color:{color};font-weight:700;white-space:nowrap">{text}</span>'
    )


def metric_bar(
    df: pd.DataFrame,
    *,
    title: str,
    fmt: str = ".1%",
    subtitle: str | None = None,
) -> alt.LayerChart:
    """Horizontal single-hue bars, one per run, with direct value labels.

    Expects columns: run (str), value (float). Bars keep dataframe order.
    """
    negative = bool((df["value"] < 0).any())
    x = alt.X("value:Q", title=None, axis=alt.Axis(format=fmt, tickCount=4))
    y = alt.Y("run:N", sort=None, title=None)
    base = alt.Chart(df)
    bars = base.mark_bar(color=BLUE, cornerRadiusEnd=3, size=16).encode(
        x=x,
        y=y,
        tooltip=[
            alt.Tooltip("run:N", title="run"),
            alt.Tooltip("value:Q", title=title, format=fmt),
        ],
    )
    labels = base.mark_text(
        align="right" if negative else "left",
        dx=-5 if negative else 5,
        fontWeight=600,
        clip=False,
    ).encode(x=x, y=y, text=alt.Text("value:Q", format=fmt))
    title_params = alt.TitleParams(
        text=title, subtitle=subtitle or "", anchor="start",
        fontSize=13, subtitleColor=MUTED, subtitleFontSize=11,
    )
    return (bars + labels).properties(
        title=title_params, height=max(70, 34 * len(df) + 20)
    )


def logprob_box(df: pd.DataFrame) -> alt.Chart:
    """Per-item seq-logprob distribution, one box per run (single hue)."""
    n_runs = df["run"].nunique()
    return (
        alt.Chart(df)
        .mark_boxplot(color=BLUE, size=18)
        .encode(
            x=alt.X("value:Q", title="per-item seq-logprob (closer to 0 = more confident)"),
            y=alt.Y("run:N", sort=None, title=None),
        )
        .properties(height=max(80, 40 * n_runs + 20))
    )
