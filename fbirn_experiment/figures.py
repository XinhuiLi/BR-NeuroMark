"""Figures for H1 (nested CV / ROC), H2 (permutation + |d|), H3 (loadings)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from fbirn_experiment.fnc import symmetric_matrix_from_upper_vec
from fbirn_experiment.h1_cv import NestedCVResult


# ── H1 ───────────────────────────────────────────────────────────────────────


def plot_h1_fold_aucs(
    res_edges: NestedCVResult,
    res_fa: NestedCVResult,
    res_ica: NestedCVResult,
    out_path: Path,
    *,
    dpi: int = 150,
) -> None:
    folds = np.arange(1, len(res_edges.outer_aucs) + 1)
    w = 0.22
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(folds - w, res_edges.outer_aucs, width=w, label="Edges (elastic net)")
    ax.bar(folds, res_fa.outer_aucs, width=w, label="FA + logistic")
    ax.bar(folds + w, res_ica.outer_aucs, width=w, label="ICA + logistic")
    ax.set_xlabel("Outer CV fold")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("H1: Nested CV performance by fold")
    ax.set_xticks(folds)
    ax.legend(fontsize=8)
    ax.set_ylim(0.0, 1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_h1_oof_roc(
    res_edges: NestedCVResult,
    res_fa: NestedCVResult,
    res_ica: NestedCVResult,
    out_path: Path,
    *,
    dpi: int = 150,
) -> None:
    from sklearn.metrics import roc_auc_score, roc_curve

    y = np.asarray(res_edges.y_true_oof).astype(int)
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for name, p, color in (
        ("Edges (OOF)", res_edges.proba_oof, "#1f77b4"),
        ("FA (OOF)", res_fa.proba_oof, "#ff7f0e"),
        ("ICA (OOF)", res_ica.proba_oof, "#2ca02c"),
    ):
        fpr, tpr, _ = roc_curve(y, p)
        a = roc_auc_score(y, p)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {a:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("H1: Out-of-fold ROC (nested CV)")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# ── H2 ───────────────────────────────────────────────────────────────────────


def plot_h2_permutation_null(
    null: np.ndarray,
    observed: float,
    out_path: Path,
    *,
    dpi: int = 150,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(null, bins=40, density=True, color="#7f7f7f", alpha=0.85, edgecolor="white")
    ax.axvline(observed, color="crimson", lw=2, label=f"Observed Δ = {observed:.4f}")
    ax.set_xlabel(r"$\Delta$ = mean(|d|) between − mean(|d|) within (permuted domains)")
    ax.set_ylabel("Density")
    ax.set_title("H2: Permutation null for domain-label randomization")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_h2_abs_d_violin(
    edge_d: np.ndarray,
    mask_between: np.ndarray,
    mask_within: np.ndarray,
    out_path: Path,
    *,
    dpi: int = 150,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.violinplot(
        [np.abs(edge_d[mask_within]), np.abs(edge_d[mask_between])],
        positions=[1, 2],
        showmeans=True,
        showmedians=True,
    )
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Within-domain edges", "Between-domain edges"])
    ax.set_ylabel(r"$|$Cohen's $d|$")
    ax.set_title("H2: Effect sizes by edge type (univariate)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# ── H3 ───────────────────────────────────────────────────────────────────────


def _significance_bracket(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    p: float,
    dy: float = 0.003,
) -> None:
    """Draw a horizontal bracket with significance annotation."""
    if p < 0.001:
        txt = "***"
    elif p < 0.01:
        txt = "**"
    elif p < 0.05:
        txt = "*"
    else:
        txt = "n.s."
    ax.plot([x1, x1, x2, x2], [y, y + dy, y + dy, y], lw=1, color="k")
    ax.text(
        (x1 + x2) / 2,
        y + dy,
        f"{txt}\np={p:.3f}",
        ha="center",
        va="bottom",
        fontsize=6,
    )


def plot_h3_summary_bars(
    summary: pd.DataFrame,
    out_path: Path,
    *,
    dpi: int = 150,
) -> None:
    """
    Grouped bar chart: between-domain vs within-domain mean |loading| per factor.
    Paired Wilcoxon across factors tests whether the two distributions differ.
    """
    df = summary.copy()
    n = len(df)
    x = np.arange(n)
    w = 0.30
    labels = [str(int(r) + 1) for r in df["factor"].values]

    between_vals = df["mean_abs_loading_between"].values
    within_vals = df["mean_abs_loading_within"].values

    fig, ax = plt.subplots(figsize=(max(8, 0.7 * n), 4.8))
    bars_b = ax.bar(x - w / 2, between_vals, width=w, label="Between-domain", color="#1f77b4")
    bars_w = ax.bar(x + w / 2, within_vals, width=w, label="Within-domain", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Factor (1-based)")
    ax.set_ylabel("Mean |loading|")
    ax.set_title("H3: Mean absolute loadings by edge class (exploratory FA)")

    valid = np.isfinite(between_vals) & np.isfinite(within_vals)
    if valid.sum() >= 5:
        try:
            stat_res = sp_stats.wilcoxon(
                between_vals[valid], within_vals[valid], alternative="two-sided"
            )
            p_val = float(stat_res.pvalue) if hasattr(stat_res, "pvalue") else float(stat_res[1])
        except Exception:
            p_val = float("nan")
    else:
        _, p_val = sp_stats.mannwhitneyu(
            between_vals[valid], within_vals[valid], alternative="two-sided"
        )

    ymax = float(np.nanmax(np.concatenate([between_vals, within_vals])))
    if np.isfinite(p_val):
        _significance_bracket(ax, -w / 2, w / 2, ymax * 1.04, p_val, dy=ymax * 0.02)

    ax.set_ylim(0, ymax * 1.18)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_h3_domain_pair_heatmap(
    domain_pair_summary: pd.DataFrame,
    out_path: Path,
    *,
    dpi: int = 150,
) -> None:
    """
    Heatmap: factors (rows) x hypothesis domain pairs (columns), cell value =
    mean |loading| for edges in that pair.
    """
    if domain_pair_summary.empty:
        return
    pivot = domain_pair_summary.pivot_table(
        index="factor", columns="pair_label", values="mean_abs_loading"
    )
    pivot.index = [f"F{int(i) + 1}" for i in pivot.index]

    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(pivot.columns)), max(5, 0.35 * len(pivot))))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xlabel("Hypothesis domain pair")
    ax.set_ylabel("Factor")
    ax.set_title("H3: Mean |loading| by factor and hypothesis domain pair")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean |loading|")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_h3_domain_pair_bars(
    domain_pair_summary: pd.DataFrame,
    between_baseline: float,
    within_baseline: float,
    out_path: Path,
    *,
    dpi: int = 150,
) -> None:
    """
    Bar chart: average mean |loading| across all factors for each hypothesis
    domain pair, with horizontal reference lines for overall between- and
    within-domain baselines.
    """
    if domain_pair_summary.empty:
        return
    agg = (
        domain_pair_summary
        .groupby("pair_label", sort=False)["mean_abs_loading"]
        .mean()
    )
    pair_order = list(dict.fromkeys(domain_pair_summary["pair_label"]))
    agg = agg.reindex(pair_order)

    x = np.arange(len(agg))
    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(agg)), 4.8))
    ax.bar(x, agg.values, color="#d62728", edgecolor="white", width=0.6)
    ax.axhline(between_baseline, ls="--", lw=1, color="#1f77b4", label="Between-domain avg")
    ax.axhline(within_baseline, ls="--", lw=1, color="#ff7f0e", label="Within-domain avg")
    ax.set_xticks(x)
    ax.set_xticklabels(agg.index, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Mean |loading| (averaged across factors)")
    ax.set_title("H3: Hypothesis domain-pair loadings vs. baselines")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _domain_tick_labels(
    icn_domain: np.ndarray,
) -> tuple[list[str], list[float], list[float]]:
    """
    Produce tick labels + positions + separator lines for domains sorted by index.
    Returns (labels, tick_positions, separator_positions).
    """
    unique_doms = []
    seen: set = set()
    for d in icn_domain:
        key = str(d)
        if key not in seen:
            seen.add(key)
            unique_doms.append(key)

    ticks: list[float] = []
    seps: list[float] = []
    names: list[str] = []
    start = 0
    for dom in unique_doms:
        count = int(np.sum(np.array([str(x) for x in icn_domain]) == dom))
        ticks.append(start + count / 2.0 - 0.5)
        names.append(dom)
        start += count
        seps.append(start - 0.5)
    return names, ticks, seps[:-1]


def plot_h3_loadings_symmetric_matrices(
    loadings: np.ndarray,
    ii: np.ndarray,
    jj: np.ndarray,
    n_icns: int,
    out_dir: Path,
    icn_domain: np.ndarray | None = None,
    *,
    dpi: int = 150,
    prefix: str = "h3_fnc_matrix_factor",
) -> None:
    """
    One n_icns x n_icns symmetric heatmap per factor. If icn_domain is provided
    the axes are labelled with domain names and thin separator lines are drawn.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_f = loadings.shape[0]
    vmax = float(np.percentile(np.abs(loadings), 99.5)) if loadings.size else 1.0
    if vmax <= 0:
        vmax = 1.0

    if icn_domain is not None:
        dom_names, dom_ticks, dom_seps = _domain_tick_labels(icn_domain)
    else:
        dom_names = dom_ticks = dom_seps = None

    for f in range(n_f):
        mat = symmetric_matrix_from_upper_vec(
            np.abs(loadings[f]), ii, jj, n_icns
        )
        fig, ax = plt.subplots(figsize=(7.2, 6))
        im = ax.imshow(mat, cmap="Reds", vmin=0.0, vmax=vmax, interpolation="nearest")
        ax.set_title(f"H3: |FA loadings| as FNC matrix — factor {f + 1}", fontsize=14)

        if dom_names is not None:
            ax.set_xticks(dom_ticks)
            ax.set_xticklabels(dom_names, rotation=90, ha="right", fontsize=12)
            ax.set_yticks(dom_ticks)
            ax.set_yticklabels(dom_names, fontsize=12)
            ax.set_xlabel("Domain", fontsize=12)
            ax.set_ylabel("Domain", fontsize=12)
            for s in dom_seps:
                ax.axhline(s, color="white", lw=0.6)
                ax.axvline(s, color="white", lw=0.6)
        else:
            ax.set_xlabel("ICN index", fontsize=12)
            ax.set_ylabel("ICN index", fontsize=12)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_{f + 1:02d}.png", dpi=dpi)
        plt.close(fig)


# ── Orchestrator ─────────────────────────────────────────────────────────────


def generate_all_figures(
    out_dir: Path,
    res_edges: NestedCVResult,
    res_fa: NestedCVResult,
    res_ica: NestedCVResult,
    h2: dict,
    h3_summary: pd.DataFrame,
    h3_loadings: np.ndarray,
    ii: np.ndarray,
    jj: np.ndarray,
    n_icns: int,
    icn_domain: np.ndarray | None = None,
    h3_domain_pair_summary: pd.DataFrame | None = None,
) -> None:
    out_dir = Path(out_dir) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_h1_fold_aucs(res_edges, res_fa, res_ica, out_dir / "h1_nestedcv_auc_by_fold.png")
    plot_h1_oof_roc(res_edges, res_fa, res_ica, out_dir / "h1_oof_roc.png")
    plot_h2_permutation_null(
        h2["null_delta_mean_abs_d"],
        float(h2["observed_delta_mean_abs_d"]),
        out_dir / "h2_permutation_null.png",
    )
    plot_h2_abs_d_violin(
        h2["edge_cohens_d"],
        h2["mask_between"],
        h2["mask_within"],
        out_dir / "h2_abs_cohen_d_by_edge_type.png",
    )
    plot_h3_summary_bars(h3_summary, out_dir / "h3_mean_abs_loading_by_class.png")
    plot_h3_loadings_symmetric_matrices(
        h3_loadings,
        ii,
        jj,
        n_icns,
        out_dir,
        icn_domain=icn_domain,
        prefix="h3_fnc_matrix_factor",
    )
    if h3_domain_pair_summary is not None and not h3_domain_pair_summary.empty:
        plot_h3_domain_pair_heatmap(
            h3_domain_pair_summary,
            out_dir / "h3_domain_pair_heatmap.png",
        )
        between_base = float(h3_summary["mean_abs_loading_between"].mean())
        within_base = float(h3_summary["mean_abs_loading_within"].mean())
        plot_h3_domain_pair_bars(
            h3_domain_pair_summary,
            between_base,
            within_base,
            out_dir / "h3_domain_pair_bars.png",
        )
