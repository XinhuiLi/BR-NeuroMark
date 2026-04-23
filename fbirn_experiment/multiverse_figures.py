"""Visualization for multiverse analysis results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy import stats as sp_stats


FORK_COLS = ["connectivity", "confound", "reduction", "classifier", "domain_granularity"]
FORK_LABELS = {
    "connectivity": "D1: Connectivity",
    "confound": "D2: Confound",
    "reduction": "D3: Reduction",
    "classifier": "D4: Classifier",
    "domain_granularity": "D5: Domain",
}

# Human-readable option labels (raw multiverse codes → plot / table text).
_FORK_LEVEL_DISPLAY: dict[str, dict[str, str]] = {
    "connectivity": {
        "pearson_z": "Pearson (Fisher z)",
        "spearman": "Spearman",
        "partial_corr": "Partial correlation",
        "mutual_info": "Mutual information",
    },
    "confound": {
        "none": "None",
        "ols": "OLS confound removal",
        "combat": "ComBat harmonization",
    },
    "reduction": {
        "none": "No reduction (edges only)",
        "fa": "Factor analysis",
        "ica": "ICA",
        "pca": "PCA",
        "nmf": "NMF",
    },
    "classifier": {
        "elasticnet": "Elastic net",
        "logistic_l2": "Logistic regression (L2)",
        "svm_linear": "Linear SVM",
        "rf": "Random forest",
    },
    "domain_granularity": {
        "subdomain_14": "14 subdomains",
        "domain_7": "7 domains",
    },
}

_OUTCOME_COL_DISPLAY: dict[str, str] = {
    "h1_delta_auc": "ΔAUC (latent − edges)",
    "h2_delta_d": "Δ mean |Cohen's d|",
    "h3_wilcoxon_p": "Wilcoxon p-value",
}


def format_fork_level(fork_col: str, raw: object) -> str:
    """Map stored fork level codes to short, reader-friendly labels."""
    s = str(raw).strip()
    per_fork = _FORK_LEVEL_DISPLAY.get(fork_col)
    if per_fork is not None and s in per_fork:
        return per_fork[s]
    if not s:
        return s
    # Fallback: snake_case → Title Case (handles ad-hoc / future levels).
    return " ".join(part.capitalize() for part in s.split("_"))


def format_outcome_axis_label(outcome_col: str) -> str:
    """Axis label for forest plots (column is still the raw key in data)."""
    return _OUTCOME_COL_DISPLAY.get(outcome_col, outcome_col.replace("_", " "))

_FONT_TITLE = 15
_FONT_LABEL = 13
_FONT_TICK = 11
_FONT_LEGEND = 11

# tab20c: 20 colors as 5 groups × 4 shades — same hue family per domain (fork),
# distinct shades for each option within the domain.
_TAB20C = plt.cm.tab20c
_N_TAB20C = 20
_N_FORKS = len(FORK_COLS)
_SLOTS_PER_DOMAIN = _N_TAB20C // _N_FORKS  # 4


def _tab20c_domain_level_color(fork_idx: int, level_idx: int) -> tuple:
    """Pick a tab20c color: domain ``fork_idx`` (0–4), option index within domain."""
    slot = fork_idx * _SLOTS_PER_DOMAIN + min(int(level_idx), _SLOTS_PER_DOMAIN - 1)
    slot = min(max(slot, 0), _N_TAB20C - 1)
    return _TAB20C(slot / (_N_TAB20C - 1))


_COLOR_GREEN = "#2ca02c"
_COLOR_RED = "#d62728"


def _sort_df(df: pd.DataFrame, outcome_col: str) -> pd.DataFrame:
    """Sort by outcome, dropping NaN rows."""
    dfc = df.dropna(subset=[outcome_col]).copy()
    dfc = dfc.sort_values(outcome_col).reset_index(drop=True)
    return dfc


def _tile_ax(
    ax: plt.Axes,
    dfc: pd.DataFrame,
    fork_cols: list[str] | None = None,
) -> None:
    """Draw colored-block tile plot showing which fork levels were used.

    Each domain (fork) uses one hue family from ``tab20c``; options within
    a domain use different shades.  Active specifications are filled
    rectangles instead of dots.  Light gray grid lines separate rows
    (conditions) and columns (specifications).
    """
    cols = [c for c in (fork_cols or FORK_COLS) if c in dfc.columns]
    n_specs = len(dfc)

    all_levels: dict[str, list[str]] = {}
    for col in cols:
        all_levels[col] = sorted(dfc[col].dropna().unique().tolist())

    y_labels: list[str] = []
    row_idx = 0
    _sep_gray = "#c8c8c8"
    for fork_idx, col in enumerate(cols):
        levels = all_levels[col]
        for level_idx, lev in enumerate(levels):
            fork_color = _tab20c_domain_level_color(fork_idx, level_idx)
            mask = dfc[col].values == lev
            xs = np.where(mask)[0]
            for x in xs:
                ax.barh(
                    row_idx, width=1.0, left=x - 0.5, height=0.8,
                    color=fork_color, linewidth=0, rasterized=True,
                )
            y_labels.append(
                f"{FORK_LABELS.get(col, col)}: {format_fork_level(col, lev)}",
            )
            row_idx += 1

    n_rows = len(y_labels)
    for k in range(n_rows - 1):
        ax.axhline(k + 0.5, color=_sep_gray, lw=0.85, zorder=5)
    if n_specs > 1:
        for j in range(n_specs - 1):
            ax.axvline(j + 0.5, color=_sep_gray, lw=0.85, zorder=5)

    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=_FONT_TICK)
    ax.set_xlim(-0.5, n_specs - 0.5)
    ax.set_ylim(len(y_labels) - 0.5, -0.5)
    ax.set_xlabel("Specification (sorted)", fontsize=_FONT_LABEL)
    ax.tick_params(axis="x", labelsize=_FONT_TICK)
    ax.set_facecolor("#f7f7f7")


# ── Specification curve plots ────────────────────────────────────────────


def plot_spec_curve_h1(df: pd.DataFrame, out_path: Path, *, dpi: int = 200) -> None:
    """Specification curve for H1: ΔAUC (latent − edges)."""
    dfc = _sort_df(df, "h1_delta_auc")
    if dfc.empty:
        return

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 3], "hspace": 0.12},
    )
    xs = np.arange(len(dfc))

    colors = np.where(dfc["h1_delta_auc"].values > 0, _COLOR_GREEN, _COLOR_RED)
    ax_top.bar(xs, dfc["h1_delta_auc"].values, color=colors, width=1.0, linewidth=0)
    ax_top.axhline(0, color="k", lw=0.7)
    ax_top.set_ylabel("ΔAUC (latent − edges)", fontsize=_FONT_LABEL)
    ax_top.set_title("H1 Specification Curve: Classification Performance", fontsize=_FONT_TITLE)
    med = dfc["h1_delta_auc"].median()
    ax_top.axhline(med, color="#e78ac3", ls="--", lw=1.2, label=f"median = {med:.3f}")
    ax_top.legend(fontsize=_FONT_LEGEND, loc="upper left")
    ax_top.tick_params(labelsize=_FONT_TICK)

    _tile_ax(ax_bot, dfc)

    fig.subplots_adjust(hspace=0.12)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_spec_curve_h2(df: pd.DataFrame, out_path: Path, *, dpi: int = 200) -> None:
    """Specification curve for H2: Δ mean|d| and p-value."""
    cols_needed = ["h2_delta_d", "h2_p"]
    for c in cols_needed:
        if c not in df.columns:
            return
    dfc = _sort_df(df, "h2_delta_d")
    if dfc.empty:
        return

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={"height_ratios": [2.5, 3], "hspace": 0.08},
    )
    xs = np.arange(len(dfc))

    h2_colors = np.where(dfc["h2_delta_d"].values > 0, _COLOR_GREEN, _COLOR_RED)
    ax_top.bar(xs, dfc["h2_delta_d"].values, color=h2_colors, width=1.0, linewidth=0, zorder=1)
    ax_top.axhline(0, color="k", lw=0.7)
    ax_top.set_ylabel("Δ mean|d|", fontsize=_FONT_LABEL)
    ax_top.set_title("H2 Specification Curve: Between- vs. Within-Domain", fontsize=_FONT_TITLE)
    ax_top.tick_params(labelsize=_FONT_TICK)

    ax_p = ax_top.twinx()
    ax_p.scatter(
        xs, dfc["h2_p"].values, s=18, color="#e78ac3", alpha=0.85, zorder=3,
        label="Permutation p",
    )
    ax_p.axhline(0.05, color="red", ls="--", lw=1.0, label="p = 0.05")
    ax_p.set_ylabel("p-value", fontsize=_FONT_LABEL)
    ax_p.set_yscale("log")
    ax_p.tick_params(labelsize=_FONT_TICK)
    ax_p.legend(fontsize=_FONT_LEGEND, loc="upper left")

    _tile_ax(ax_bot, dfc)

    fig.subplots_adjust(hspace=0.12)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_spec_curve_h3(df: pd.DataFrame, out_path: Path, *, dpi: int = 200) -> None:
    """Specification curve for H3: Wilcoxon p-value for between > within loadings."""
    dfc = df.dropna(subset=["h3_wilcoxon_p"]).copy()
    dfc = dfc[dfc["reduction"] != "none"].copy()
    if dfc.empty:
        return
    dfc = dfc.sort_values("h3_wilcoxon_p").reset_index(drop=True)

    fig, (ax_p, ax_bot) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 3], "hspace": 0.08},
    )
    xs = np.arange(len(dfc))

    colors = np.where(dfc["h3_wilcoxon_p"].values < 0.05, _COLOR_GREEN, _COLOR_RED)
    ax_p.bar(xs, -np.log10(dfc["h3_wilcoxon_p"].values), color=colors, width=1.0, linewidth=0)
    ax_p.axhline(-np.log10(0.05), color="red", ls="--", lw=1.0, label="p = 0.05")
    ax_p.set_ylabel("−log₁₀(p)", fontsize=_FONT_LABEL)
    ax_p.set_title("H3 Specification Curve: Factor Loading Structure", fontsize=_FONT_TITLE)
    ax_p.legend(fontsize=_FONT_LEGEND, loc="upper right")
    ax_p.tick_params(labelsize=_FONT_TICK)

    _tile_ax(ax_bot, dfc)

    fig.subplots_adjust(hspace=0.12)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Scatter-violin-box plot (marginal influence) ─────────────────────────


def plot_forest(
    df: pd.DataFrame,
    outcome_col: str,
    title: str,
    out_path: Path,
    *,
    dpi: int = 200,
    n_boot: int = 2000,
) -> None:
    """Scatter-violin-box plot of marginal influence per fork level.

    For each fork level, shows a half-violin distribution, a compact box
    plot (median + IQR), and jittered individual data points — commonly
    referred to as a "raincloud" plot.
    """
    dfc = df.dropna(subset=[outcome_col]).copy()
    if dfc.empty:
        return

    entries: list[dict] = []
    rng = np.random.default_rng(42)

    fork_boundaries: list[int] = []
    for fork_idx, fork in enumerate(FORK_COLS):
        if fork not in dfc.columns:
            continue
        levels = sorted(dfc[fork].dropna().unique().tolist())
        if not levels:
            continue
        fork_boundaries.append(len(entries))
        for level_idx, lev in enumerate(levels):
            vals = dfc.loc[dfc[fork] == lev, outcome_col].values
            if len(vals) == 0:
                continue
            fork_color = _tab20c_domain_level_color(fork_idx, level_idx)
            entries.append({
                "fork_col": fork,
                "fork": FORK_LABELS.get(fork, fork),
                "level": str(lev),
                "vals": vals.copy(),
                "mean": float(np.mean(vals)),
                "color": fork_color,
            })

    if not entries:
        return

    n_rows = len(entries)
    fig, ax = plt.subplots(figsize=(8, 0.5 * n_rows + 1.5))
    grand_mean = dfc[outcome_col].mean()

    for i, e in enumerate(entries):
        y = n_rows - 1 - i
        vals = e["vals"]
        c = e["color"]
        c_dark = mcolors.to_rgba(c, alpha=1.0)

        if len(vals) >= 4:
            parts = ax.violinplot(
                vals, positions=[y], vert=False, showextrema=False,
                widths=0.7,
            )
            for body in parts["bodies"]:
                m = np.mean(body.get_paths()[0].vertices[:, 1])
                body.get_paths()[0].vertices[:, 1] = np.clip(
                    body.get_paths()[0].vertices[:, 1], m, None,
                )
                body.set_facecolor(mcolors.to_rgba(c, alpha=0.35))
                body.set_edgecolor(c_dark)
                body.set_linewidth(0.6)

        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        box_h = 0.18
        ax.barh(
            y - box_h / 2, width=q3 - q1, left=q1, height=box_h,
            color=mcolors.to_rgba(c, alpha=0.6), edgecolor=c_dark, linewidth=0.8,
        )
        ax.plot(med, y, "s", color="white", markersize=5, markeredgecolor=c_dark,
                markeredgewidth=1.0, zorder=5)

        jitter = rng.uniform(-0.22, -0.05, len(vals))
        ax.scatter(
            vals, y + jitter, s=12, alpha=0.55, color=c_dark,
            edgecolors="none", zorder=4, rasterized=True,
        )

    ax.axvline(grand_mean, color="gray", ls="--", lw=0.8, alpha=0.5, zorder=1)

    for b in fork_boundaries[1:]:
        y_sep = n_rows - b - 0.5
        ax.axhline(y_sep, color="#cccccc", lw=0.8, ls="-")

    y_labels = [
        f"{e['fork']}: {format_fork_level(e['fork_col'], e['level'])}"
        for e in entries
    ]
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(list(reversed(y_labels)), fontsize=_FONT_TICK)
    ax.set_xlabel(format_outcome_axis_label(outcome_col), fontsize=_FONT_LABEL)
    ax.set_title(title, fontsize=_FONT_TITLE)
    ax.tick_params(axis="x", labelsize=_FONT_TICK)
    ax.set_ylim(-0.7, n_rows - 0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Robustness summary ──────────────────────────────────────────────────


def robustness_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute robustness summary for each hypothesis."""
    rows = []

    # H1
    h1_df = df.dropna(subset=["h1_delta_auc"])
    h1_df = h1_df[h1_df["reduction"] != "none"]
    if not h1_df.empty:
        n_total = len(h1_df)
        n_latent_better = int((h1_df["h1_delta_auc"] > 0).sum())
        rows.append({
            "Hypothesis": "H1: Latent > Edges",
            "Total specs": n_total,
            "# Favourable": n_latent_better,
            "% Robust": f"{100 * n_latent_better / n_total:.1f}",
            "Median effect": f"{h1_df['h1_delta_auc'].median():.4f}",
        })

    # H2
    h2_df = df.dropna(subset=["h2_p"])
    if not h2_df.empty:
        n_total = len(h2_df)
        n_sig = int((h2_df["h2_p"] < 0.05).sum())
        rows.append({
            "Hypothesis": "H2: Between > Within",
            "Total specs": n_total,
            "# Favourable": n_sig,
            "% Robust": f"{100 * n_sig / n_total:.1f}",
            "Median effect": f"{h2_df['h2_delta_d'].median():.4f}",
        })

    # H3
    h3_df = df.dropna(subset=["h3_wilcoxon_p"])
    h3_df = h3_df[h3_df["reduction"] != "none"]
    if not h3_df.empty:
        n_total = len(h3_df)
        n_sig = int((h3_df["h3_wilcoxon_p"] < 0.05).sum())
        rows.append({
            "Hypothesis": "H3: Between loading advantage",
            "Total specs": n_total,
            "# Favourable": n_sig,
            "% Robust": f"{100 * n_sig / n_total:.1f}",
            "Median effect": f"{h3_df['h3_wilcoxon_p'].median():.4f}",
        })

    return pd.DataFrame(rows)


def conditional_robustness(
    df: pd.DataFrame,
    hypothesis: str,
    outcome_col: str,
    threshold: float = 0.05,
    direction: str = "less",
) -> pd.DataFrame:
    """Report robustness conditional on each fork level.

    Parameters
    ----------
    direction : ``"less"`` (p < threshold) or ``"greater"`` (effect > threshold)
    """
    dfc = df.dropna(subset=[outcome_col]).copy()
    if hypothesis.startswith("H3") or hypothesis.startswith("H1"):
        dfc = dfc[dfc["reduction"] != "none"]

    rows = []
    for fork in FORK_COLS:
        if fork not in dfc.columns:
            continue
        for lev in sorted(dfc[fork].dropna().unique()):
            sub = dfc[dfc[fork] == lev]
            if sub.empty:
                continue
            if direction == "less":
                n_sig = int((sub[outcome_col] < threshold).sum())
            else:
                n_sig = int((sub[outcome_col] > threshold).sum())
            rows.append({
                "Hypothesis": hypothesis,
                "Fork": FORK_LABELS.get(fork, fork),
                "Level": format_fork_level(fork, lev),
                "Total": len(sub),
                "# Significant": n_sig,
                "% Significant": f"{100 * n_sig / len(sub):.1f}",
            })
    return pd.DataFrame(rows)


# ── Joint permutation test (Simonsohn et al., 2020) ─────────────────────


def joint_permutation_test(
    df: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Test whether the proportion of significant specs exceeds chance.

    Under the global null, the count of significant specifications follows
    Binomial(n, alpha).  We report a one-sided p-value for each hypothesis
    testing whether the observed count exceeds this null expectation.
    """
    rows: list[dict[str, Any]] = []

    # H1: latent > edges (favourable = ΔAUC > 0)
    h1 = df.dropna(subset=["h1_delta_auc"])
    h1 = h1[h1["reduction"] != "none"]
    if not h1.empty:
        n = len(h1)
        k = int((h1["h1_delta_auc"] > 0).sum())
        p_binom = float(sp_stats.binomtest(k, n, alpha, alternative="greater").pvalue)
        rows.append({
            "Hypothesis": "H1: Latent > Edges",
            "n_specs": n,
            "n_favourable": k,
            "pct_favourable": round(100 * k / n, 1),
            "null_alpha": alpha,
            "null_expected": round(n * alpha, 1),
            "binom_p": p_binom,
            "exceeds_chance": p_binom < 0.05,
        })

    # H2: between > within (significant = p < alpha)
    h2 = df.dropna(subset=["h2_p"])
    if not h2.empty:
        n = len(h2)
        k = int((h2["h2_p"] < alpha).sum())
        p_binom = float(sp_stats.binomtest(k, n, alpha, alternative="greater").pvalue)
        rows.append({
            "Hypothesis": "H2: Between > Within",
            "n_specs": n,
            "n_favourable": k,
            "pct_favourable": round(100 * k / n, 1),
            "null_alpha": alpha,
            "null_expected": round(n * alpha, 1),
            "binom_p": p_binom,
            "exceeds_chance": p_binom < 0.05,
        })

    # H3: between loading advantage (significant = p < alpha)
    h3 = df.dropna(subset=["h3_wilcoxon_p"])
    h3 = h3[h3["reduction"] != "none"]
    if not h3.empty:
        n = len(h3)
        k = int((h3["h3_wilcoxon_p"] < alpha).sum())
        p_binom = float(sp_stats.binomtest(k, n, alpha, alternative="greater").pvalue)
        rows.append({
            "Hypothesis": "H3: Between loading advantage",
            "n_specs": n,
            "n_favourable": k,
            "pct_favourable": round(100 * k / n, 1),
            "null_alpha": alpha,
            "null_expected": round(n * alpha, 1),
            "binom_p": p_binom,
            "exceeds_chance": p_binom < 0.05,
        })

    return pd.DataFrame(rows)


# ── Master figure generator ─────────────────────────────────────────────


def generate_multiverse_figures(
    df: pd.DataFrame,
    fig_dir: Path | str,
    *,
    dpi: int = 200,
) -> None:
    """Generate all multiverse visualizations."""
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_spec_curve_h1(df, fig_dir / "mv_spec_curve_h1.png", dpi=dpi)
    plot_spec_curve_h2(df, fig_dir / "mv_spec_curve_h2.png", dpi=dpi)
    plot_spec_curve_h3(df, fig_dir / "mv_spec_curve_h3.png", dpi=dpi)

    plot_forest(
        df, "h1_delta_auc", "H1 Marginal Influence (ΔAUC)", fig_dir / "mv_forest_h1.png", dpi=dpi,
    )
    plot_forest(
        df, "h2_delta_d", "H2 Marginal Influence (Δ|d|)", fig_dir / "mv_forest_h2.png", dpi=dpi,
    )

    h3_df = df[df["reduction"] != "none"].copy()
    if not h3_df.empty:
        plot_forest(
            h3_df,
            "h3_wilcoxon_p",
            "H3 Marginal Influence (Wilcoxon p)",
            fig_dir / "mv_forest_h3.png",
            dpi=dpi,
        )

    rob = robustness_table(df)
    if not rob.empty:
        rob.to_csv(fig_dir / "mv_robustness_summary.csv", index=False)
        print(f"Saved {fig_dir / 'mv_robustness_summary.csv'}")
        print("\nRobustness summary:\n", rob.to_string(index=False))

    cond_rows = []
    cond_rows.append(conditional_robustness(df, "H2", "h2_p", 0.05, "less"))
    h3_sub = df[df["reduction"] != "none"]
    cond_rows.append(conditional_robustness(h3_sub, "H3", "h3_wilcoxon_p", 0.05, "less"))
    h1_sub = df[df["reduction"] != "none"]
    cond_rows.append(conditional_robustness(h1_sub, "H1", "h1_delta_auc", 0.0, "greater"))
    cond = pd.concat([r for r in cond_rows if not r.empty], ignore_index=True)
    if not cond.empty:
        cond.to_csv(fig_dir / "mv_conditional_robustness.csv", index=False)
        print(f"Saved {fig_dir / 'mv_conditional_robustness.csv'}")

    # Joint permutation test (Simonsohn et al., 2020 §6.5)
    joint = joint_permutation_test(df)
    if not joint.empty:
        joint.to_csv(fig_dir / "mv_joint_permutation_test.csv", index=False)
        print(f"Saved {fig_dir / 'mv_joint_permutation_test.csv'}")
        print("\nJoint permutation test:\n", joint.to_string(index=False))


def regenerate_multiverse_figures(
    results_csv: Path | str,
    *,
    figures_dir: Path | str | None = None,
    dpi: int = 200,
) -> None:
    """Rebuild all multiverse figures and summary CSVs from a saved results table.

    Reads ``multiverse_results.csv`` (same format as ``run_multiverse`` writes under
    ``output_dir``). Drops rows with a non-empty ``error`` column so plots match a
    successful run only.
    """
    results_csv = Path(results_csv)
    if not results_csv.is_file():
        raise FileNotFoundError(
            f"Multiverse results not found: {results_csv}. "
            "Pass the path to multiverse_results.csv from a completed multiverse run.",
        )
    df = pd.read_csv(results_csv)
    if "error" in df.columns:
        err = df["error"]
        ok = err.isna() | (err.astype(str).str.strip().isin(("", "nan")))
        n_drop = int((~ok).sum())
        if n_drop:
            print(f"Dropping {n_drop} row(s) with recorded errors before plotting.")
        df = df.loc[ok].copy()
    fig_dir = Path(figures_dir) if figures_dir is not None else results_csv.parent / "figures"
    generate_multiverse_figures(df, fig_dir, dpi=dpi)
    print(f"Multiverse figures → {fig_dir}")
