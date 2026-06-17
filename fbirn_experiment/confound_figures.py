"""Exploratory plots for confound columns (distributions by diagnostic group)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fbirn_experiment.confounds import DEFAULT_CONFOUND_COLS

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

_COL_TITLE: dict[str, str] = {
    "age": "Age",
    "sex": "Sex",
    "race": "Race",
    "site": "Site",
    "hm": "Head motion",
}

# Continuous panels: (panel title, y-axis label) — raincloud / reference style
_COL_CONTINUOUS_LABELS: dict[str, tuple[str, str]] = {
    "age": ("Age", "Age"),
    "hm": ("Head Motion", "Head Motion (Mean FD)"),
}

_HC_COLOR = "#1f77b4"
_SZ_COLOR = "#ff7f0e"
# Shared by raincloud (age / head motion) and categorical bars (sex / race / site)
_HC_SZ_COLORS = (_HC_COLOR, _SZ_COLOR)


def _column_title(col: str) -> str:
    """Human-readable panel / axis title (first letter capitalized; special cases)."""
    if col in _COL_TITLE:
        return _COL_TITLE[col]
    base = col.replace("_", " ").strip()
    if not base:
        return col
    return base[0].upper() + base[1:].lower()


def _site_category_label(v: Any) -> str:
    """Map raw site codes to display labels (fbirn_* → Site *; missing → Site 8)."""
    if pd.isna(v):
        return "Site 8"
    s = str(v).strip()
    if s.lower() in ("nan", "none", ""):
        return "Site 8"
    low = s.lower()
    key = "fbirn_"
    if key in low:
        i = int(low.index(key))
        suffix = s[i + len(key) :].strip()
        return f"Site {suffix}" if suffix else "Site"
    return s[0].upper() + s[1:].lower() if len(s) > 1 else s.upper()


def _other_category_label(v: Any) -> str:
    if pd.isna(v):
        return "(Missing)"
    s = str(v).strip()
    if s.lower() in ("nan", "none", ""):
        return "(Missing)"
    return s[0].upper() + s[1:].lower() if len(s) > 1 else s.upper()


_SEX_ORDER = ("Female", "Male")
_RACE_ORDER = ("Asian", "Black", "White", "Other")


def _site_sort_key(label: str) -> int:
    s = str(label).strip()
    if s.startswith("Site "):
        tail = s[5:].strip()
        try:
            return int(tail)
        except ValueError:
            return 10_000
    return 10_001


def _ordered_crosstab_rows(ct: pd.DataFrame, col: str) -> pd.DataFrame:
    """Reorder crosstab index: sex, race, site as requested; else lexical."""
    idx_set = set(ct.index.astype(str))
    if col == "sex":
        rows = [x for x in _SEX_ORDER if x in idx_set]
        rows.extend(sorted(x for x in idx_set if x not in rows))
    elif col == "race":
        rows = [x for x in _RACE_ORDER if x in idx_set]
        rows.extend(sorted(x for x in idx_set if x not in rows))
    elif col == "site":
        rows = sorted(idx_set, key=_site_sort_key)
    else:
        rows = sorted(idx_set)
    return ct.reindex(rows).fillna(0.0)


def _is_effectively_continuous(s: pd.Series) -> bool:
    s2 = s.dropna()
    if s2.size == 0:
        return False
    return bool(pd.api.types.is_numeric_dtype(s) and s2.nunique() > 10)


def _swarm_x_left(y: np.ndarray, x0: float, *, x_step: float = 0.048) -> np.ndarray:
    """Beeswarm-style x to the left of *x0* (same y), avoiding overlap in y–x space."""
    y = np.asarray(y, dtype=np.float64)
    n = int(y.size)
    if n == 0:
        return np.array([], dtype=np.float64)
    y_ptp = float(np.ptp(y))
    y_gap = max(y_ptp / max(120.0, 3.0 * n), y_ptp * 1e-9 if y_ptp > 0 else 1e-6)
    order = np.argsort(y, kind="mergesort")
    xs = np.zeros(n, dtype=np.float64)
    placed_y: list[float] = []
    placed_x: list[float] = []

    for k in range(n):
        i = int(order[k])
        yi = float(y[i])
        xi = float(x0 - x_step * 0.4)
        while any(
            abs(py - yi) < y_gap * 1.08 and abs(px - xi) < x_step * 0.92
            for py, px in zip(placed_y, placed_x)
        ):
            xi -= x_step * 0.44
        xs[i] = xi
        placed_y.append(yi)
        placed_x.append(xi)

    return xs


def _continuous_raincloud_labels(col: str) -> tuple[str, str]:
    """(panel title, y-axis label) for continuous confound columns."""
    if col in _COL_CONTINUOUS_LABELS:
        return _COL_CONTINUOUS_LABELS[col]
    disp = _column_title(col)
    return (disp, disp)


def _violin_box_scatter_by_group(
    ax: plt.Axes,
    values: np.ndarray,
    y: np.ndarray,
    *,
    col: str,
    fs_title: int,
    fs_axis: int,
    fs_tick: int,
    fs_legend: int,
) -> None:
    """Raincloud per group: **swarm left**, **box centered** on tick, **half-violin right** (flat edge at box).

    Matches common “raincloud” layout (see reference figure): y = measurement,
    horizontal dashed y-grid, HC blue / SZ orange.
    """
    mask = np.isfinite(values)
    v, yy = values[mask], y[mask]
    data_hc = v[yy == 0]
    data_sz = v[yy == 1]
    title, value_label = _continuous_raincloud_labels(col)

    if data_hc.size == 0 or data_sz.size == 0:
        ax.text(0.5, 0.5, "Insufficient data per group", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=fs_title)
        return

    centers = np.array([1.0, 2.0], dtype=np.float64)
    datasets = [data_hc, data_sz]
    colors = _HC_SZ_COLORS
    swarm_anchor = 0.12

    vp_kw = dict(
        vert=True,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=0.52,
    )
    try:
        vp = ax.violinplot(
            datasets,
            positions=list(centers),
            side="high",
            **vp_kw,
        )
    except TypeError:
        kw_fb = dict(vp_kw)
        kw_fb["positions"] = list(centers + 0.08)
        kw_fb["widths"] = 0.34
        vp = ax.violinplot(datasets, **kw_fb)

    for body, color in zip(vp["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.38)
        body.set_edgecolor("k")
        body.set_linewidth(0.55)
        body.set_zorder(1)

    bp = ax.boxplot(
        datasets,
        positions=list(centers),
        vert=True,
        widths=0.095,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "k", "linewidth": 2.2},
        whiskerprops={"color": "k", "linewidth": 1.0},
        capprops={"color": "k", "linewidth": 1.0},
        boxprops={"linewidth": 1.1, "edgecolor": "k"},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("white")
        patch.set_alpha(1.0)
        patch.set_edgecolor("k")
        patch.set_linewidth(1.1)
        patch.set_zorder(3)

    for d, cx, color, lab in zip(
        datasets,
        centers,
        colors,
        ("HC", "SZ"),
    ):
        x_sw = _swarm_x_left(d, cx - swarm_anchor)
        ax.scatter(
            x_sw,
            d,
            s=20,
            alpha=0.72,
            c=color,
            edgecolors="k",
            linewidths=0.28,
            label=lab,
            zorder=4,
        )

    ax.set_xticks(list(centers))
    ax.set_xticklabels(["HC", "SZ"], fontsize=fs_tick)
    ax.set_xlabel("Group", fontsize=fs_axis)
    ax.set_ylabel(value_label, fontsize=fs_axis)
    ax.set_title(title, fontsize=fs_title)
    ax.tick_params(axis="y", labelsize=fs_tick)
    ax.tick_params(axis="x", labelsize=fs_tick, rotation=0)
    ax.grid(True, axis="y", linestyle="--", alpha=0.55, color="#b0b0b0", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=fs_legend, loc="upper right", framealpha=0.92)

    lo = float(np.min(v))
    hi = float(np.max(v))
    pad = 0.04 * (hi - lo if hi > lo else 1.0)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(0.25, 2.88)


def _categorical_by_group(
    ax: plt.Axes,
    series: pd.Series,
    y: np.ndarray,
    *,
    col: str,
    title: str,
    fs_title: int,
    fs_axis: int,
    fs_tick: int,
    fs_legend: int,
) -> None:
    grp = np.where(y == 0, "HC", "SZ")
    if col == "site":
        mapped = series.map(_site_category_label)
    else:
        mapped = series.map(_other_category_label)
    tmp = pd.DataFrame({"_v": mapped, "_g": grp})
    ct = pd.crosstab(tmp["_v"], tmp["_g"])
    for gcol in ("HC", "SZ"):
        if gcol not in ct.columns:
            ct[gcol] = 0
    ct = ct[["HC", "SZ"]]
    ct = _ordered_crosstab_rows(ct, col)

    # Horizontal bar thickness (pandas ``width`` = bar height for barh); wider for race/site.
    bar_height = 0.82 if col in ("race", "site") else 0.64

    # Horizontal bars: category names on y (no rotated x tick labels).
    ct.plot(
        kind="barh",
        ax=ax,
        stacked=False,
        width=bar_height,
        color=list(_HC_SZ_COLORS),
        edgecolor="k",
        linewidth=0.35,
        legend=False,
    )
    for container, c in zip(ax.containers, _HC_SZ_COLORS):
        for patch in container.patches:
            patch.set_facecolor(c)
    # First row in *ct* at top (Female before Male; Asian…; Site 3, 7, …).
    ax.invert_yaxis()
    ax.set_title(title, fontsize=fs_title)
    ax.set_xlabel("Count", fontsize=fs_axis)
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=fs_tick, rotation=0)
    ax.tick_params(axis="y", labelsize=fs_tick)
    ax.legend(
        handles=[
            mpatches.Patch(facecolor=_HC_COLOR, edgecolor="k", linewidth=0.6, label="HC"),
            mpatches.Patch(facecolor=_SZ_COLOR, edgecolor="k", linewidth=0.6, label="SZ"),
        ],
        title="",
        fontsize=fs_legend,
    )

    if col in ("sex", "race", "site"):
        fs_ann = max(int(fs_tick) - 5, 10)
        for container in ax.containers:
            for rect in container.patches:
                w = float(rect.get_width())
                if w <= 0 or not np.isfinite(w):
                    continue
                y_c = rect.get_y() + 0.5 * rect.get_height()
                x_c = rect.get_x() + 0.5 * w
                ax.text(
                    x_c,
                    y_c,
                    str(int(round(w))),
                    va="center",
                    ha="center",
                    fontsize=fs_ann,
                    color="white",
                    clip_on=True,
                )


def plot_confound_distributions(
    confound_df: pd.DataFrame,
    y: np.ndarray,
    out_path: str | Path,
    *,
    confound_cols: Sequence[str] | None = None,
    dpi: int = 400,
) -> None:
    """
    One row of panels: each confound’s distribution, split by label (0=HC, 1=SZ).

    Numeric columns with more than 10 unique values use a raincloud layout per group
    (beeswarm left, box centered on the tick, right half-violin; value on y);
    other columns use grouped horizontal bar counts (HC vs SZ).
    """
    cols = list(confound_cols or DEFAULT_CONFOUND_COLS)
    out_path = Path(out_path)
    y = np.asarray(y).astype(int).ravel()
    if len(confound_df) != len(y):
        raise ValueError(
            f"confound_df rows ({len(confound_df)}) must match y length ({len(y)})."
        )
    missing = [c for c in cols if c not in confound_df.columns]
    if missing:
        raise ValueError(f"Columns missing from confound_df: {missing}")

    n = len(cols)
    fs_suptitle = 26
    fs_title = 24
    fs_axis = 22
    fs_tick = 21
    fs_legend = 20

    fig_w = min(36.0, 3.8 * n + 1.8)
    fig_h = 6.5
    fig, axes_row = plt.subplots(1, n, figsize=(fig_w, fig_h), squeeze=False)
    axes_flat = np.atleast_1d(axes_row).ravel()

    for k, c in enumerate(cols):
        ax = axes_flat[k]
        s = confound_df[c]
        disp = _column_title(c)
        if _is_effectively_continuous(s):
            vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
            _violin_box_scatter_by_group(
                ax,
                vals,
                y,
                col=c,
                fs_title=fs_title,
                fs_axis=fs_axis,
                fs_tick=fs_tick,
                fs_legend=fs_legend,
            )
        else:
            _categorical_by_group(
                ax,
                s,
                y,
                col=c,
                title=disp,
                fs_title=fs_title,
                fs_axis=fs_axis,
                fs_tick=fs_tick,
                fs_legend=fs_legend,
            )

    # fig.suptitle(
    #     "Distributions of confounding variables by group (HC vs SZ)",
    #     fontsize=fs_suptitle,
    #     y=0.94,
    # )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
