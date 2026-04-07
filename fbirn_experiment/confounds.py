"""Load confounding variables and regress them out of FNC edges."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


DEFAULT_CONFOUND_COLS = ("age", "sex", "race", "site", "hm")


def load_confounds(
    csv_path: str | Path,
    confound_cols: Sequence[str] = DEFAULT_CONFOUND_COLS,
) -> pd.DataFrame:
    """Read the CSV and return only the requested confound columns."""
    df = pd.read_csv(csv_path)
    missing = [c for c in confound_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Confound columns not found in {csv_path}: {missing}. "
            f"Available: {list(df.columns[:20])}..."
        )
    return df[list(confound_cols)].copy()


def build_design_matrix(
    df: pd.DataFrame,
    confound_cols: Sequence[str] = DEFAULT_CONFOUND_COLS,
) -> tuple[np.ndarray, list[str]]:
    """
    Encode confounds into a numeric design matrix (with intercept).

    Continuous columns are kept as-is; categorical/string columns are
    one-hot encoded (first category dropped to avoid collinearity).
    Returns (Z, col_names) where Z has shape (n_subjects, n_regressors).
    """
    parts: list[pd.DataFrame] = []
    col_names: list[str] = []

    for col in confound_cols:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 10:
            vals = s.astype(np.float64).values
            if np.any(np.isnan(vals)):
                vals = np.where(np.isnan(vals), np.nanmean(vals), vals)
            parts.append(pd.DataFrame({col: vals}))
            col_names.append(col)
        else:
            dummies = pd.get_dummies(s, prefix=col, drop_first=True, dtype=np.float64)
            parts.append(dummies)
            col_names.extend(dummies.columns.tolist())

    Z = pd.concat(parts, axis=1).values.astype(np.float64)
    intercept = np.ones((Z.shape[0], 1), dtype=np.float64)
    Z = np.hstack([Z, intercept])
    col_names.append("intercept")
    return Z, col_names


def regress_confounds(
    edges: np.ndarray,
    Z: np.ndarray,
) -> np.ndarray:
    """
    OLS residualization: for each edge, regress out the design matrix Z.

    edges: (n_subjects, n_edges)
    Z:     (n_subjects, n_regressors)  — should include intercept
    Returns residuals with the same shape.
    """
    beta, _, _, _ = np.linalg.lstsq(Z, edges, rcond=None)
    return edges - Z @ beta


def regress_confounds_cv(
    edges_train: np.ndarray,
    edges_test: np.ndarray,
    Z_train: np.ndarray,
    Z_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit confound regression on *training* data; apply to both train and test.

    This avoids information leakage from test confound statistics into the
    training residuals.
    """
    beta, _, _, _ = np.linalg.lstsq(Z_train, edges_train, rcond=None)
    return edges_train - Z_train @ beta, edges_test - Z_test @ beta


def confound_summary(
    confound_df: pd.DataFrame,
    Z: np.ndarray,
    col_names: list[str],
) -> dict[str, Any]:
    """Metadata dict for saving alongside artifacts."""
    return {
        "confound_columns_raw": list(confound_df.columns),
        "design_matrix_columns": col_names,
        "n_regressors": len(col_names),
        "n_subjects": int(Z.shape[0]),
    }
