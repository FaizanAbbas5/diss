#!/usr/bin/env python
"""Battery comparison report: paired bootstrap CIs, pairwise differences vs
the selected prompt, fold-stability ranking, and dissertation figures.

    python experiments/battery_report.py \
        --configs configs/battery_v2/rotowire_*.yaml --out results/rw_battery_v2_report

Every config must have a finished run: generations.jsonl plus
metrics_per_item.jsonl produced by run_eval.py with --claims --parent.
Figures need matplotlib:  pip install -e ".[analysis]"

Selection follows a PRE-SPECIFIED decision rule (documented in summary.md):
primary = micro rate of wrong relational claims (lower is better);
tie-breaks in order: unsupported-number rate, claim entailment, PARENT F1.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path

import numpy as np

from t2t.config import load_config, run_dir
from t2t.eval.paired_stats import (
    bootstrap_ci,
    fold_split,
    fold_stats,
    mean,
    paired_bootstrap_diff,
    prop_positive,
    rank,
    ratio_of_sums,
)


def corpus_bleu(*cols: np.ndarray) -> float:
    """Corpus BLEU from summed per-item n-gram counts, so that resampling
    items yields BLEU confidence intervals (Koehn, EMNLP 2004). Column
    order: match_1..4, total_1..4, hyp_len, ref_len."""
    from t2t.eval.bleu import bleu_from_counts

    s = [float(np.sum(c)) for c in cols]
    return bleu_from_counts(s[0:4], s[4:8], s[8], s[9])


# ---------------------------------------------------------------- palette ---
# Reference dataviz palette (validated, colourblind-safe adjacent order).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",  # blue orange aqua yellow
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]  # magenta green violet red
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# ---------------------------------------------------------------- metrics ---
# (key, label, stat, columns, higher_better, is_rate)
METRICS = [
    ("rel_wrong", "Wrong relational claims", ratio_of_sums,
     ("rel_n_wrong", "rel_n_claims"), False, True),
    ("rel_items", "Items with relational error", prop_positive,
     ("rel_n_wrong",), False, True),
    ("num_unsupported", "Unsupported numbers", ratio_of_sums,
     ("num_n_unsupported", "num_n_numbers"), False, True),
    ("entailed", "Claims entailed (NLI)", ratio_of_sums,
     ("cl_n_entailed", "cl_n_claims"), True, True),
    # per-claim analogue of contra_items: comparable across prompts that
    # differ in how much they say (see fig_battery_contradiction_decomp)
    ("contra_claims", "Contradictions per claim", ratio_of_sums,
     ("cl_n_contradicted", "cl_n_claims"), False, True),
    ("contra_items", "Items with contradiction", prop_positive,
     ("cl_n_contradicted",), False, True),
    ("parent", "PARENT F1", mean, ("parent_f1",), True, False),
    # style/literature anchor, not a faithfulness metric (see t2t.eval.bleu)
    ("bleu", "BLEU", corpus_bleu,
     ("bleu_match_1", "bleu_match_2", "bleu_match_3", "bleu_match_4",
      "bleu_total_1", "bleu_total_2", "bleu_total_3", "bleu_total_4",
      "bleu_hyp_len", "bleu_ref_len"), True, False),
    ("seq_logprob", "Seq-logprob", mean, ("seq_logprob",), True, False),
]
PRIMARY = "rel_wrong"
TIE_BREAKS = ["num_unsupported", "entailed", "parent"]
# The forest plot keeps rate metrics only, on one shared pp axis; PARENT
# differences are in the summary tables (0-1 units would squash a pp axis).
FOREST_METRICS = ["rel_wrong", "rel_items", "entailed", "contra_claims"]


def load_run(cfg_path: str, results_root: str):
    cfg = load_config(cfg_path)
    rd = run_dir(cfg, results_root)
    if "prompt" in cfg:
        label = cfg["prompt"].split("/")[-1].split("_v")[0]
        if cfg.get("serialisation", "markdown") != "markdown":
            label += "+fs"  # arm run: same prompt, fact-sheet serialisation
    else:
        label = cfg["name"].replace("rw-full-", "")  # e.g. arm2 soft-backend runs
    per_item: dict[str, dict] = {}
    with open(rd / "metrics_per_item.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                per_item[json.loads(line)["id"]] = json.loads(line)
    with open(rd / "generations.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec["id"] in per_item:
                    per_item[rec["id"]]["seq_logprob"] = rec.get("seq_logprob")
    return label, cfg, rd, per_item


def columns_for(per_item: dict[str, dict], ids: list[str]) -> dict[str, np.ndarray]:
    """Flatten the nested per-item records into aligned numeric columns."""
    def col(fn):
        return np.array([fn(per_item[i]) for i in ids], dtype=float)

    cols = {
        "num_n_numbers": col(lambda r: r["n_numbers"]),
        "num_n_unsupported": col(lambda r: r["n_numbers"] - r["n_supported"]),
        "rel_n_claims": col(lambda r: r["relations"]["n_claims"]),
        "rel_n_wrong": col(lambda r: r["relations"]["n_wrong"]),
    }
    if all("claims" in per_item[i] for i in ids):
        cols["cl_n_claims"] = col(lambda r: r["claims"]["n_claims"])
        cols["cl_n_entailed"] = col(lambda r: r["claims"]["n_entailed"])
        cols["cl_n_contradicted"] = col(lambda r: r["claims"]["n_contradicted"])
    if all("parent_f1" in per_item[i] for i in ids):
        cols["parent_f1"] = col(lambda r: r["parent_f1"])
    if all("bleu_hyp_len" in per_item[i] for i in ids):
        for key in [f"bleu_match_{n}" for n in range(1, 5)] + [
            f"bleu_total_{n}" for n in range(1, 5)
        ] + ["bleu_hyp_len", "bleu_ref_len"]:
            cols[key] = col(lambda r, k=key: r[k])
    if all(per_item[i].get("seq_logprob") is not None for i in ids):
        cols["seq_logprob"] = col(lambda r: r["seq_logprob"])
    return cols


def fmt(x: float, is_rate: bool) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{100 * x:.1f}%" if is_rate else f"{x:.3f}"


# ---------------------------------------------------------------- figures ---

def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def save(fig, out: Path, name: str):
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight",
                    facecolor=fig.get_facecolor())


def fig_metric_panels(plt, results, labels, out: Path):
    present = [m for m in METRICS if m[0] in results[labels[0]]["point"]]
    ncols = min(4, len(present))
    nrows = math.ceil(len(present) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.6 * nrows))
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(present):]:
        ax.set_visible(False)
    for ax, (key, label, _s, _c, higher, is_rate) in zip(axes, present):
        style_axes(ax)
        for k, lb in enumerate(labels):
            pt, lo, hi = (results[lb][w][key] for w in ("point", "lo", "hi"))
            v, l, h = (100 * x if is_rate else x for x in (pt, lo, hi))
            ax.bar(k, v, width=0.55, color=SERIES[k], zorder=3)
            ax.errorbar(k, v, yerr=[[v - l], [h - v]], fmt="none",
                        ecolor=INK_2, elinewidth=1.2, capsize=3, zorder=4)
            anchor = h if v >= 0 else l  # clear of the CI whisker cap
            va, off = ("bottom", 3) if v >= 0 else ("top", -3)
            ax.annotate(fmt(pt, is_rate), (k, anchor),
                        textcoords="offset points", xytext=(0, off),
                        ha="center", va=va, fontsize=7.5, color=INK)
        ax.set_title(f"{label}\n({'higher' if higher else 'lower'} is better)",
                     fontsize=8.5, color=INK_2)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.margins(y=0.18)
    fig.suptitle("Prompt battery: point estimates with 95% bootstrap CIs",
                 fontsize=11, color=INK, y=1.0)
    fig.tight_layout()
    save(fig, out, "fig_battery_metrics")
    plt.close(fig)


def fig_forest(plt, diffs, labels, winner, out: Path):
    rows = []  # (metric_label, rival, oriented diff, lo, hi, colour, is_rate)
    for key, label, _s, _c, higher, is_rate in METRICS:
        if key not in FOREST_METRICS or (winner, key) not in diffs.get("_keys", set()):
            continue
        for lb in labels:
            if lb == winner:
                continue
            d, lo, hi = diffs[(lb, key)]
            flip = 1.0 if not higher else -1.0  # positive => rival worse
            d, lo, hi = flip * d, *(sorted((flip * lo, flip * hi)))
            rows.append((label, lb, d, lo, hi, SERIES[labels.index(lb)], is_rate))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(rows) + 1.6))
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax.axvline(0, color=BASELINE, linewidth=1.2, zorder=2)
    ys, ylabels, seen = [], [], set()
    for r, (mlabel, rival, d, lo, hi, colour, is_rate) in enumerate(rows):
        y = len(rows) - r
        scale = 100 if is_rate else 1
        ax.plot([lo * scale, hi * scale], [y, y], color=colour, linewidth=1.8,
                solid_capstyle="round", zorder=3)
        ax.plot(d * scale, y, "o", color=colour, markersize=6, zorder=4,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ys.append(y)
        head = mlabel if mlabel not in seen else ""
        seen.add(mlabel)
        ylabels.append(f"{head}   {rival}" if head else f"   {rival}")
    ax.set_yticks(ys)
    ax.set_yticklabels(ylabels, fontsize=8, color=INK)
    ax.set_xlabel(f"difference vs '{winner}' in percentage points "
                  f"(positive = worse than {winner})", fontsize=8, color=INK_2)
    ax.set_title(f"Paired differences vs selected prompt '{winner}' "
                 "(95% paired-bootstrap CIs)", fontsize=10, color=INK)
    fig.tight_layout()
    save(fig, out, "fig_battery_forest")
    plt.close(fig)


def fig_folds(plt, fold_ranks, labels, n_folds, out: Path):
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    xs = np.arange(1, n_folds + 1)
    for k, lb in enumerate(labels):
        ranks = [fold_ranks[f][k] for f in range(n_folds)]
        ax.plot(xs, ranks, "-o", color=SERIES[k], linewidth=2, markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=lb, zorder=3)
        ax.annotate(lb, (xs[-1], ranks[-1]), textcoords="offset points",
                    xytext=(8, 0), va="center", fontsize=8, color=INK)
    ax.set_ylim(len(labels) + 0.5, 0.5)  # rank 1 on top
    ax.set_yticks(range(1, len(labels) + 1))
    ax.set_xticks(xs)
    ax.set_xlabel("disjoint fold", fontsize=8.5, color=INK_2)
    ax.set_ylabel("rank (1 = best)", fontsize=8.5, color=INK_2)
    ax.set_title("Ranking stability across folds (primary metric)",
                 fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.28), ncol=len(labels))
    ax.margins(x=0.18)
    fig.tight_layout()
    save(fig, out, "fig_battery_folds")
    plt.close(fig)


def fig_contradiction_decomp(plt, results, per_run_cols, labels, out: Path,
                             n_boot: int, seed: int):
    """Why an item-level rate is not a like-for-like comparison: it is the
    per-claim rate compounded over how many claims each prompt makes."""
    if any("cl_n_claims" not in per_run_cols[lb] for lb in labels):
        return
    volume = {}
    for lb in labels:
        volume[lb] = bootstrap_ci(mean, [per_run_cols[lb]["cl_n_claims"]],
                                  n_boot, seed)

    panels = [
        ("How much is said", "NLI claims per item", volume, False),
        ("What you observe", "Items with ≥1 contradiction (%)",
         {lb: (results[lb]["point"]["contra_items"],
               results[lb]["lo"]["contra_items"],
               results[lb]["hi"]["contra_items"]) for lb in labels}, True),
        ("Like-for-like", "Contradictions per claim (%)",
         {lb: (results[lb]["point"]["contra_claims"],
               results[lb]["lo"]["contra_claims"],
               results[lb]["hi"]["contra_claims"]) for lb in labels}, True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
    fig.patch.set_facecolor(SURFACE)
    for ax, (title, ylab, vals, is_rate) in zip(axes, panels):
        style_axes(ax)
        for k, lb in enumerate(labels):
            pt, lo, hi = vals[lb]
            v, l, h = (100 * x if is_rate else x for x in (pt, lo, hi))
            ax.bar(k, v, width=0.55, color=SERIES[k], zorder=3)
            ax.errorbar(k, v, yerr=[[v - l], [h - v]], fmt="none",
                        ecolor=INK_2, elinewidth=1.2, capsize=3, zorder=4)
            txt = f"{v:.1f}%" if is_rate else f"{v:.2f}"
            ax.annotate(txt, (k, h), textcoords="offset points",
                        xytext=(0, 3), ha="center", va="bottom",
                        fontsize=7.5, color=INK)
        ax.set_title(title, fontsize=9.5, color=INK, pad=8)
        ax.set_ylabel(ylab, fontsize=8, color=INK_2)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.margins(y=0.20)
    fig.suptitle("Item-level contradiction rates are confounded by how much "
                 "each prompt says", fontsize=10.5, color=INK, y=1.02)
    fig.tight_layout()
    save(fig, out, "fig_battery_contradiction_decomp")
    plt.close(fig)


def fig_rolling(plt, per_run_cols, labels, ids, out: Path, window: int = 75):
    """Rolling primary-metric rate across the (temporally ordered) split:
    one line per prompt, window of `window` games; shows whether prompt
    differences are stable across the season or driven by a stretch."""
    order = np.argsort([int(i.rsplit("-", 1)[-1]) for i in ids])
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    kern = np.ones(window)
    ends = []
    for k, lb in enumerate(labels):
        wrong = per_run_cols[lb]["rel_n_wrong"][order]
        claims = per_run_cols[lb]["rel_n_claims"][order]
        rate = 100 * np.convolve(wrong, kern, "valid") / np.convolve(claims, kern, "valid")
        xs = np.arange(len(rate)) + window / 2  # centred window
        ax.plot(xs, rate, color=SERIES[k], linewidth=2, label=lb, zorder=3)
        ends.append([xs[-1], rate[-1], lb])
    # dodge overlapping end labels: enforce a minimum vertical gap
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.055
    for prev, cur in zip(sorted(ends, key=lambda e: e[1]),
                         sorted(ends, key=lambda e: e[1])[1:]):
        if cur[1] - prev[1] < gap:
            cur[1] = prev[1] + gap
    for x, y, lb in ends:
        ax.annotate(lb, (x, y), textcoords="offset points", xytext=(6, 0),
                    va="center", fontsize=8, color=INK)
    ax.set_xlabel("validation game index (split order ≈ season time)",
                  fontsize=8.5, color=INK_2)
    ax.set_ylabel(f"wrong relational claims (%, rolling {window} games)",
                  fontsize=8.5, color=INK_2)
    ax.set_title("Primary metric across the split: rolling window per prompt",
                 fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=min(len(labels), 6))
    ax.margins(x=0.14)
    fig.tight_layout()
    save(fig, out, "fig_battery_rolling")
    plt.close(fig)


def fig_parent_violin(plt, per_run_cols, labels, out: Path):
    if any("parent_f1" not in per_run_cols[lb] for lb in labels):
        return
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    rng = np.random.default_rng(13)
    for k, lb in enumerate(labels):
        vals = per_run_cols[lb]["parent_f1"]
        parts = ax.violinplot(vals, positions=[k], widths=0.7,
                              showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(SERIES[k])
            body.set_alpha(0.25)
            body.set_edgecolor(SERIES[k])
            body.set_linewidth(1.2)
        ax.scatter(k + rng.uniform(-0.09, 0.09, len(vals)), vals, s=7,
                   color=SERIES[k], alpha=0.55, linewidths=0, zorder=3)
        m = float(np.mean(vals))
        ax.plot([k - 0.2, k + 0.2], [m, m], color=INK, linewidth=1.6, zorder=4)
        ax.annotate(f"{m:.3f}", (k + 0.24, m), fontsize=7.5, color=INK,
                    va="center")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("PARENT F1 (per item)", fontsize=8.5, color=INK_2)
    ax.set_title("Per-item PARENT F1 distribution (bar = mean)",
                 fontsize=10, color=INK)
    fig.tight_layout()
    save(fig, out, "fig_battery_parent_violin")
    plt.close(fig)


# ------------------------------------------------------------------- main ---

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True,
                    help="battery config paths (globs ok)")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default=None,
                    help="report dir (default results/<first-name>-report)")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--folds", type=int, default=0, help="0 = auto (3 if n<400 else 5)")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    paths: list[str] = []
    for pat in args.configs:
        hits = sorted(glob.glob(pat)) or [pat]
        paths.extend(hits)
    runs = [load_run(p, args.results_dir) for p in paths]
    labels = [r[0] for r in runs]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"duplicate prompt labels: {labels}")

    id_sets = [set(r[3]) for r in runs]
    ids = sorted(set.intersection(*id_sets))
    dropped = len(set.union(*id_sets)) - len(ids)
    if dropped:
        print(f"Warning: {dropped} item(s) missing from some runs; "
              f"paired analysis uses the {len(ids)} common items")
    if not ids:
        raise SystemExit("no common items across runs")

    per_run_cols = {lb: columns_for(pi, ids) for lb, _c, _rd, pi in runs}
    common = set.intersection(*(set(c) for c in per_run_cols.values()))
    active = [m for m in METRICS if all(c in common for c in m[3])
              and (m[2] is not ratio_of_sums
                   or any(np.sum(per_run_cols[lb][m[3][-1]]) > 0 for lb in labels))]
    print(f"{len(ids)} paired items; metrics: {[m[0] for m in active]}")

    # point estimates + CIs
    results = {lb: {"point": {}, "lo": {}, "hi": {}} for lb in labels}
    for key, _l, stat, colkeys, _h, _r in active:
        for lb in labels:
            cols = [per_run_cols[lb][c] for c in colkeys]
            pt, lo, hi = bootstrap_ci(stat, cols, args.n_boot, args.seed)
            results[lb]["point"][key] = pt
            results[lb]["lo"][key] = lo
            results[lb]["hi"][key] = hi

    # pre-specified selection rule
    active_keys = {m[0]: m for m in active}
    def sort_key(lb):
        parts = []
        for key in [PRIMARY, *TIE_BREAKS]:
            if key in active_keys:
                v = results[lb]["point"][key]
                v = 0.0 if math.isnan(v) else v
                parts.append(round(v, 4) * (1 if not active_keys[key][4] else -1))
        return tuple(parts)
    winner = min(labels, key=sort_key)
    print(f"Selected prompt by pre-specified rule: {winner}")

    # paired diffs vs winner
    diffs: dict = {"_keys": set()}
    for key, _l, stat, colkeys, _h, _r in active:
        for lb in labels:
            if lb == winner:
                continue
            d = paired_bootstrap_diff(
                stat,
                [per_run_cols[lb][c] for c in colkeys],
                [per_run_cols[winner][c] for c in colkeys],
                args.n_boot, args.seed)
            diffs[(lb, key)] = d
            diffs["_keys"].add((winner, key))

    # folds on the primary metric
    n_folds = args.folds or (3 if len(ids) < 400 else 5)
    folds = fold_split(len(ids), n_folds, args.seed)
    pm = active_keys[PRIMARY]
    fold_ranks = []
    fold_points = {lb: fold_stats(pm[2], [per_run_cols[lb][c] for c in pm[3]], folds)
                   for lb in labels}
    for f in range(n_folds):
        fold_ranks.append(rank([fold_points[lb][f] for lb in labels],
                               higher_better=pm[4]))

    out = Path(args.out or Path(args.results_dir) / "battery_report")
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- tables ---
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "higher_better", *labels])
        for key, label, _s, _c, higher, is_rate in active:
            w.writerow([label, higher, *(
                f"{results[lb]['point'][key]:.6f}" for lb in labels)])

    lines = [
        "# Prompt battery report", "",
        "> Seq-logprob caveat: computed over the full generation, so for "
        "two-stage prompts (structured) it includes scratch tokens and is "
        "not comparable with single-stage prompts. Not part of the decision "
        "rule.", "",
        f"- Paired items: **{len(ids)}** | bootstrap resamples: {args.n_boot} "
        f"| seed: {args.seed} | folds: {n_folds}",
        "- Decision rule (pre-specified): minimise wrong-relational-claim rate; "
        "ties broken by unsupported-number rate, claim entailment, PARENT F1.",
        f"- **Selected prompt: `{winner}`**", "",
        "## Point estimates (95% bootstrap CI)", "",
        "| metric | " + " | ".join(labels) + " |",
        "|---" * (len(labels) + 1) + "|",
    ]
    for key, label, _s, _c, higher, is_rate in active:
        cells = []
        for lb in labels:
            pt = fmt(results[lb]["point"][key], is_rate)
            lo = fmt(results[lb]["lo"][key], is_rate)
            hi = fmt(results[lb]["hi"][key], is_rate)
            cells.append(f"{pt} [{lo}, {hi}]")
        arrow = "↑" if higher else "↓"
        lines.append(f"| {label} {arrow} | " + " | ".join(cells) + " |")
    lines += ["", f"## Paired differences vs `{winner}`", "",
              "Positive oriented difference = rival worse. CI excluding 0 marks "
              "a robust difference.", "",
              "| metric | rival | Δ (oriented) | 95% CI | robust? |",
              "|---|---|---|---|---|"]
    for key, label, _s, _c, higher, is_rate in active:
        for lb in labels:
            if lb == winner:
                continue
            d, lo, hi = diffs[(lb, key)]
            flip = 1.0 if not higher else -1.0
            d2, lo2, hi2 = flip * d, *sorted((flip * lo, flip * hi))
            robust = "yes" if lo2 > 0 or hi2 < 0 else "no"
            lines.append(
                f"| {label} | {lb} | {fmt(d2, is_rate)} | "
                f"[{fmt(lo2, is_rate)}, {fmt(hi2, is_rate)}] | {robust} |")
    lines += ["", "## Fold ranking (primary metric)", "",
              "| fold | " + " | ".join(labels) + " |",
              "|---" * (len(labels) + 1) + "|"]
    for f in range(n_folds):
        lines.append(f"| {f + 1} | " + " | ".join(
            str(fold_ranks[f][k]) for k in range(len(labels))) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ------------------------------------------------------------ figures ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial"]
        plt.rcParams["svg.fonttype"] = "none"
    except ImportError:
        print("matplotlib not installed: tables written, figures skipped. "
              "Install with: pip install -e \".[analysis]\"")
        return

    fig_metric_panels(plt, results, labels, out)
    fig_forest(plt, diffs, labels, winner, out)
    fig_folds(plt, fold_ranks, labels, n_folds, out)
    fig_contradiction_decomp(plt, results, per_run_cols, labels, out,
                             args.n_boot, args.seed)
    if len(ids) >= 300:
        fig_rolling(plt, per_run_cols, labels, ids, out)
    fig_parent_violin(plt, per_run_cols, labels, out)
    print(f"Report written to {out}")


if __name__ == "__main__":
    main()
