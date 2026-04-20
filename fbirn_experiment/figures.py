"""Figures for H1 (nested CV / ROC), H2 (permutation + |d|), H3 (loadings)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from fbirn_experiment.fnc import symmetric_matrix_from_upper_vec
from fbirn_experiment.h1_cv import NestedCVResult


def _truncate_icn_label(s: str, max_len: int = 26) -> str:
    s = str(s).strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _h1_latent_axis_prefix(method: str) -> str:
    """FA → *Factor*; ICA → *Component* (sklearn ordering)."""
    m = str(method).strip().upper()
    return "Component" if m == "ICA" else "Factor"


def _h1_icn_pair_label(
    i: int,
    j: int,
    icn_labels: np.ndarray | Sequence[str] | None,
) -> str:
    """One line: indices plus optional NeuroMark-style ICN names."""
    if icn_labels is None:
        return f"ICN {i}–{j}"
    n = len(icn_labels)
    if i < 0 or j < 0 or i >= n or j >= n:
        return f"ICN {i}–{j}"
    li = _truncate_icn_label(str(icn_labels[i]))
    lj = _truncate_icn_label(str(icn_labels[j]))
    return f"ICN {i} ({li}) – ICN {j} ({lj})"


# ── H1 ───────────────────────────────────────────────────────────────────────


def plot_h1_fold_aucs(
    res_edges: NestedCVResult,
    res_fa: NestedCVResult | None,
    res_ica: NestedCVResult,
    out_path: Path,
    *,
    dpi: int = 150,
) -> None:
    folds = np.arange(1, len(res_edges.outer_aucs) + 1)
    w = 0.22
    fig, ax = plt.subplots(figsize=(9, 4))
    if res_fa is None:
        ax.bar(folds - w / 2, res_edges.outer_aucs, width=w, label="Edges (L2 logistic)")
        ax.bar(folds + w / 2, res_ica.outer_aucs, width=w, label="ICA + logistic")
    else:
        ax.bar(folds - w, res_edges.outer_aucs, width=w, label="Edges (L2 logistic)")
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
    res_fa: NestedCVResult | None,
    res_ica: NestedCVResult,
    out_path: Path,
    *,
    dpi: int = 150,
) -> None:
    from sklearn.metrics import roc_auc_score, roc_curve

    y = np.asarray(res_edges.y_true_oof).astype(int)
    fig, ax = plt.subplots(figsize=(5.2, 5))
    curves: list[tuple[str, np.ndarray, str]] = [
        ("Edges (OOF)", res_edges.proba_oof, "#1f77b4"),
        ("ICA (OOF)", res_ica.proba_oof, "#2ca02c"),
    ]
    if res_fa is not None:
        curves.insert(1, ("FA (OOF)", res_fa.proba_oof, "#ff7f0e"))
    for name, p, color in curves:
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


def plot_h1_stability_violin(
    res_edges: NestedCVResult,
    res_fa: NestedCVResult | None,
    res_ica: NestedCVResult,
    stability: dict,
    out_path: Path,
    *,
    dpi: int = 150,
) -> None:
    """Violin plot of outer-fold AUCs; annotate Levene test for equal spread."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = [np.asarray(res_edges.outer_aucs, dtype=np.float64)]
    labels = ["Edges\n(L2 logistic)"]
    if res_fa is not None:
        data.append(np.asarray(res_fa.outer_aucs, dtype=np.float64))
        labels.append("FA +\nlogistic")
    data.append(np.asarray(res_ica.outer_aucs, dtype=np.float64))
    labels.append("ICA +\nlogistic")
    n_v = len(data)
    positions = list(range(1, n_v + 1))
    parts = ax.violinplot(
        data,
        positions=positions,
        widths=0.55,
        showmeans=True,
        showmedians=False,
        showextrema=True,
    )
    for b in parts["bodies"]:
        b.set_alpha(0.55)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("ROC-AUC (outer test fold)")
    ax.set_title("H1: Stability of AUC across outer folds")
    ax.set_ylim(0.0, 1.02)
    p_lev = float(stability.get("levene_pvalue", 1.0))
    p_fl = float(stability.get("fligner_pvalue", 1.0))
    ax.text(
        0.02,
        0.02,
        f"Levene test (equal spread): p = {p_lev:.3g}\n"
        f"Fligner–Killeen: p = {p_fl:.3g}",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.35),
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_h1_edge_top_coefficients(
    ii: np.ndarray,
    jj: np.ndarray,
    coef_edges: np.ndarray,
    out_path: Path,
    *,
    top_k: int = 45,
    dpi: int = 150,
    icn_labels: np.ndarray | Sequence[str] | None = None,
) -> None:
    """Top |logistic coefficients| on edge features (FNC upper triangle)."""
    coef_edges = np.asarray(coef_edges, dtype=np.float64).ravel()
    n = min(top_k, coef_edges.shape[0])
    order = np.argsort(-np.abs(coef_edges))[:n]
    vals = coef_edges[order]
    labs = [
        _h1_icn_pair_label(int(ii[idx]), int(jj[idx]), icn_labels)
        for idx in order
    ]
    fig, ax = plt.subplots(figsize=(9, max(4.0, 0.14 * n)))
    y = np.arange(n)
    colors = np.where(vals >= 0, "#1f77b4", "#d62728")
    ax.barh(y, vals, color=colors, height=0.75)
    ax.set_yticks(y)
    tick_fs = 6 if icn_labels is not None else 7
    ax.set_yticklabels(labs, fontsize=tick_fs)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("Coefficient (L2 logistic, full-sample refit)")
    ax.set_title(f"H1 interpretability: top {n} edge coefficients (|value|)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_h1_latent_coefficients_and_loadings(
    method: str,
    coef_latent: np.ndarray,
    components: np.ndarray,
    ii: np.ndarray,
    jj: np.ndarray,
    out_path: Path,
    *,
    n_icns: int | None = None,
    dpi: int = 1000,
    icn_domain: np.ndarray | Sequence[str] | None = None,
    matrix_col_inches: float = 2.12,
    matrix_row_inches: float = 1.58,
    matrix_hspace: float = 0.09,
    matrix_wspace: float = 0.09,
) -> None:
    """Logistic coefficients per latent dim + symmetric ICN×ICN loading matrices.

    Renders **every** row of *components* as an ``n_icns``×``n_icns`` matrix (same order
    as sklearn factors/components). Each row is mapped from the FNC upper-triangle
    vector via ``symmetric_matrix_from_upper_vec``.

    If *icn_domain* has length ``n_icns`` (one label per ICN, same as H3), matrix axes
    use **domain** tick positions and labels like ``plot_h3_loadings_symmetric_matrices``.

    *matrix_col_inches* / *matrix_row_inches* scale the matplotlib figure size per matrix
    column/row (larger values → bigger 105×105 panels on screen and in the saved PNG at
    the same ``dpi``). *matrix_hspace* / *matrix_wspace* set subplot gaps in the matrix
    grid (fraction of average subplot height/width; smaller → tighter packing).
    """
    coef_latent = np.asarray(coef_latent, dtype=np.float64).ravel()
    comp = np.asarray(components, dtype=np.float64)
    dim_prefix = _h1_latent_axis_prefix(method)
    k = coef_latent.shape[0]
    if comp.shape[0] != k:
        comp = comp[:k]
    ii = np.asarray(ii, dtype=np.int64)
    jj = np.asarray(jj, dtype=np.int64)
    if n_icns is None:
        n_icns = int(max(ii.max(), jj.max()) + 1)

    dom_names: list[str] | None = None
    dom_ticks: list[float] | None = None
    dom_seps: list[float] | None = None
    if icn_domain is not None:
        dom_arr = np.asarray(icn_domain).reshape(-1)
        if dom_arr.shape[0] == n_icns:
            if dom_arr.dtype == object:
                dom_arr = dom_arr.astype(str)
            dom_names, dom_ticks, dom_seps = _domain_tick_labels(dom_arr)

    show = np.arange(k, dtype=np.int64)
    n_show = k

    matrices: list[np.ndarray] = []
    for j in range(n_show):
        vec = comp[int(show[j]), :]
        matrices.append(
            symmetric_matrix_from_upper_vec(vec, ii, jj, n_icns),
        )
    vmax = float(np.max(np.abs(np.stack(matrices, axis=0)))) if matrices else 1.0
    if not np.isfinite(vmax) or vmax < 1e-15:
        vmax = 1.0

    # Grid layout: ICA uses up to 10 matrices per row; FA keeps a compact sqrt-ish column count.
    if dim_prefix == "Component":
        ncol = min(10, n_show)
    else:
        ncol = int(min(8, max(3, int(np.ceil(np.sqrt(n_show * 1.2))))))
        ncol = min(ncol, n_show)
    nrow = int(np.ceil(n_show / ncol))

    # Figure size (inches): larger panels → each 105×105 matrix uses more canvas area
    # at the same DPI (or combine with higher ``dpi`` for more pixels).
    fig_w = 1.15 + float(matrix_col_inches) * ncol
    fig_h = 1.02 + float(matrix_row_inches) * nrow + 0.55
    fig = plt.figure(figsize=(fig_w, fig_h), layout="constrained")
    # Taller top row so the coefficient bar chart is easier to read.
    gs = fig.add_gridspec(2, 1, height_ratios=[0.52, nrow * 1.08])
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.bar(np.arange(k), coef_latent, color="#555555", edgecolor="k", linewidth=0.25)
    ax0.axhline(0, color="k", lw=0.55)
    ax0.set_xticks(np.arange(k))
    xtick_fs = 8 if k > 18 else 9
    ax0.set_xticklabels(
        [f"{dim_prefix} {d + 1}" for d in range(k)],
        rotation=90,
        ha="center",
        va="top",
        fontsize=xtick_fs,
    )
    ax0.set_ylabel("Logistic coef.", fontsize=10)
    ax0.set_title(f"{method}: weight per {dim_prefix.lower()}", fontsize=11)
    ax0.tick_params(axis="y", labelsize=9)

    gs_mat = gs[1, 0].subgridspec(
        nrow,
        ncol,
        hspace=float(matrix_hspace),
        wspace=float(matrix_wspace),
    )
    tick_stride = max(1, n_icns // 5)
    tick_idx = np.arange(0, n_icns, tick_stride)
    dom_lab_fs = 3.5 if (dom_names is not None and len(dom_names) > 12) else 5
    dom_axis_fs = 6.5
    axes_mat: list = []
    last_im = None
    for idx in range(n_show):
        axm = fig.add_subplot(gs_mat[idx // ncol, idx % ncol])
        axes_mat.append(axm)
        last_im = axm.imshow(
            matrices[idx],
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            aspect="equal",
            origin="upper",
            interpolation="nearest",
        )
        axm.set_title(f"{dim_prefix} {int(show[idx]) + 1}", fontsize=8, pad=2)
        row, col = idx // ncol, idx % ncol
        if dom_names is not None:
            axm.set_xticks(dom_ticks)
            axm.set_xticklabels(
                dom_names, rotation=90, ha="right", fontsize=dom_lab_fs,
            )
            axm.set_yticks(dom_ticks)
            axm.set_yticklabels(dom_names, fontsize=dom_lab_fs)
            for s in dom_seps or []:
                axm.axhline(s, color="white", lw=0.45)
                axm.axvline(s, color="white", lw=0.45)
            axm.set_xlabel("Domain", fontsize=dom_axis_fs)
            axm.set_ylabel("Domain", fontsize=dom_axis_fs)
        else:
            axm.set_xticks(tick_idx)
            axm.set_yticks(tick_idx)
            axm.tick_params(axis="both", labelsize=7)
            if row == nrow - 1:
                axm.set_xlabel("ICN index", fontsize=8)
            else:
                axm.set_xlabel("")
                axm.tick_params(axis="x", labelbottom=False)
            if col == 0:
                axm.set_ylabel("ICN index", fontsize=8)
            else:
                axm.set_ylabel("")
                axm.tick_params(axis="y", labelleft=False)

    cbar = fig.colorbar(
        last_im, ax=axes_mat, shrink=0.45, fraction=0.046,
    )
    cbar.set_label("Loading", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle(
        f"H1 interpretability: {method} — all {dim_prefix.lower()}s as "
        f"{n_icns}×{n_icns} FNC matrices (median-CV refit)",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def regenerate_h1_latent_interpretability_figures(
    artifacts_dir: Path | str,
    figures_dir: Path | str | None = None,
    *,
    dpi: int = 150,
    **plot_kw: Any,
) -> None:
    """Redraw latent interpretability PNGs from artifacts (ICA always; FA if saved).

    Reads:

    - ``artifacts_dir / "fnc_edges_bundle.npz"`` — ``triu_i``, ``triu_j``, ``icn_domain``
    - ``artifacts_dir / "h1_interpretability_coefs.npz"`` — coefs and ``components_``

    No time courses or nested CV re-run required. Optional ``plot_kw`` is forwarded to
    ``plot_h1_latent_coefficients_and_loadings`` (e.g. ``matrix_col_inches=2.3``).
    """
    artifacts_dir = Path(artifacts_dir)
    fig_dir = Path(figures_dir) if figures_dir is not None else artifacts_dir.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = artifacts_dir / "fnc_edges_bundle.npz"
    coefs_path = artifacts_dir / "h1_interpretability_coefs.npz"
    if not bundle_path.is_file():
        raise FileNotFoundError(
            f"Expected {bundle_path} (save from a full run with artifacts enabled).",
        )
    if not coefs_path.is_file():
        raise FileNotFoundError(
            f"Expected {coefs_path} (H1 interpretability refit artifacts).",
        )

    b = np.load(bundle_path, allow_pickle=True)
    ii = np.asarray(b["triu_i"], dtype=np.int64)
    jj = np.asarray(b["triu_j"], dtype=np.int64)
    icn_domain = np.asarray(b["icn_domain"])

    z = np.load(coefs_path, allow_pickle=True)
    fa_keys = {"coef_fa_latent", "fa_components"}.issubset(z.files)
    if fa_keys and np.asarray(z["coef_fa_latent"]).size > 0:
        plot_h1_latent_coefficients_and_loadings(
            "FA",
            np.asarray(z["coef_fa_latent"]),
            np.asarray(z["fa_components"]),
            ii,
            jj,
            fig_dir / "h1_fa_latent_interpretability.png",
            icn_domain=icn_domain,
            dpi=dpi,
            **plot_kw,
        )
        print(f"Wrote {fig_dir / 'h1_fa_latent_interpretability.png'}")
    plot_h1_latent_coefficients_and_loadings(
        "ICA",
        np.asarray(z["coef_ica_latent"]),
        np.asarray(z["ica_components"]),
        ii,
        jj,
        fig_dir / "h1_ica_latent_interpretability.png",
        icn_domain=icn_domain,
        dpi=dpi,
        **plot_kw,
    )
    print(f"Wrote {fig_dir / 'h1_ica_latent_interpretability.png'}")


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
    res_fa: NestedCVResult | None,
    res_ica: NestedCVResult,
    h2: dict,
    h3_summary: pd.DataFrame,
    h3_loadings: np.ndarray,
    ii: np.ndarray,
    jj: np.ndarray,
    n_icns: int,
    icn_domain: np.ndarray | None = None,
    h3_domain_pair_summary: pd.DataFrame | None = None,
    *,
    h1_stability: dict | None = None,
    h1_interpretability: dict | None = None,
) -> None:
    out_dir = Path(out_dir) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_h1_fold_aucs(res_edges, res_fa, res_ica, out_dir / "h1_nestedcv_auc_by_fold.png")
    plot_h1_oof_roc(res_edges, res_fa, res_ica, out_dir / "h1_oof_roc.png")
    if h1_stability is not None:
        plot_h1_stability_violin(
            res_edges,
            res_fa,
            res_ica,
            h1_stability,
            out_dir / "h1_auc_stability_violin.png",
        )
    if h1_interpretability is not None:
        plot_h1_edge_top_coefficients(
            ii,
            jj,
            h1_interpretability["coef_edges"],
            out_dir / "h1_edge_top_coefficients.png",
            icn_labels=icn_domain,
        )
        if "coef_fa_latent" in h1_interpretability:
            plot_h1_latent_coefficients_and_loadings(
                "FA",
                h1_interpretability["coef_fa_latent"],
                h1_interpretability["fa_components"],
                ii,
                jj,
                out_dir / "h1_fa_latent_interpretability.png",
                n_icns=n_icns,
                icn_domain=icn_domain,
            )
        plot_h1_latent_coefficients_and_loadings(
            "ICA",
            h1_interpretability["coef_ica_latent"],
            h1_interpretability["ica_components"],
            ii,
            jj,
            out_dir / "h1_ica_latent_interpretability.png",
            n_icns=n_icns,
            icn_domain=icn_domain,
        )
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
