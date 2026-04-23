"""Exploratory plots for confound columns (distributions by diagnostic group)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fbirn_experiment.confounds import DEFAULT_CONFOUND_COLS

_COL_TITLE: dict[str, str] = {
    "age": "Age",
    "sex": "Sex",
    "race": "Race",
    "site": "Site",
    "hm": "Head motion",
}


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


def _hist_by_group(
    ax: plt.Axes,
    values: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    xlabel: str,
    fs_title: int,
    fs_axis: int,
    fs_tick: int,
    fs_legend: int,
    bins: int = 28,
) -> None:
    mask = np.isfinite(values)
    v, yy = values[mask], y[mask]
    lo, hi = float(np.min(v)), float(np.max(v))
    if lo == hi:
        lo, hi = lo - 1.0, hi + 1.0
    edges = np.linspace(lo, hi, bins + 1)
    ax.hist(
        v[yy == 0],
        bins=edges,
        alpha=0.55,
        label="HC",
        color="C0",
        edgecolor="k",
        linewidth=0.35,
    )
    ax.hist(
        v[yy == 1],
        bins=edges,
        alpha=0.55,
        label="SZ",
        color="C1",
        edgecolor="k",
        linewidth=0.35,
    )
    ax.set_title(title, fontsize=fs_title)
    ax.set_xlabel(xlabel, fontsize=fs_axis)
    ax.set_ylabel("Count", fontsize=fs_axis)
    ax.tick_params(axis="both", labelsize=fs_tick)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(fontsize=fs_legend, loc="upper right")


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

    # Horizontal bars: category names on y (no rotated x tick labels).
    ct.plot(
        kind="barh",
        ax=ax,
        stacked=False,
        color=["C0", "C1"],
        edgecolor="k",
        linewidth=0.35,
        legend=True,
    )
    # First row in *ct* at top (Female before Male; Asian…; Site 3, 7, …).
    ax.invert_yaxis()
    ax.set_title(title, fontsize=fs_title)
    ax.set_xlabel("Count", fontsize=fs_axis)
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=fs_tick, rotation=0)
    ax.tick_params(axis="y", labelsize=fs_tick)
    ax.legend(title="", fontsize=fs_legend)


def plot_confound_distributions(
    confound_df: pd.DataFrame,
    y: np.ndarray,
    out_path: str | Path,
    *,
    confound_cols: Sequence[str] | None = None,
    dpi: int = 200,
) -> None:
    """
    One row of panels: each confound’s distribution, split by label (0=HC, 1=SZ).

    Numeric columns with more than 10 unique values use overlaid histograms;
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
    fs_title = 22
    fs_axis = 20
    fs_tick = 17
    fs_legend = 17

    fig_w = min(36.0, 3.8 * n + 1.8)
    fig_h = 6.0
    fig, axes_row = plt.subplots(1, n, figsize=(fig_w, fig_h), squeeze=False)
    axes_flat = np.atleast_1d(axes_row).ravel()

    for k, c in enumerate(cols):
        ax = axes_flat[k]
        s = confound_df[c]
        disp = _column_title(c)
        if _is_effectively_continuous(s):
            vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
            _hist_by_group(
                ax,
                vals,
                y,
                title=disp,
                xlabel=disp,
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

    fig.suptitle(
        "Confound distributions by group (HC vs SZ)",
        fontsize=fs_suptitle,
        y=0.94,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
