"""Effect sizes, univariate edge tests, FDR."""

from __future__ import annotations

import numpy as np
from scipy import stats


def cohens_d_two_group(x0: np.ndarray, x1: np.ndarray) -> float:
    x0 = np.asarray(x0, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    n0, n1 = len(x0), len(x1)
    if n0 < 2 or n1 < 2:
        return float("nan")
    v0, v1 = np.var(x0, ddof=1), np.var(x1, ddof=1)
    pooled = np.sqrt(((n0 - 1) * v0 + (n1 - 1) * v1) / (n0 + n1 - 2))
    if pooled == 0:
        return 0.0
    return (np.mean(x1) - np.mean(x0)) / pooled


def edge_cohens_d_and_pvalues(
    edges: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y).astype(int)
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    n_edges = edges.shape[1]
    d = np.empty(n_edges)
    p = np.empty(n_edges)
    for e in range(n_edges):
        a, b = edges[i0, e], edges[i1, e]
        d[e] = cohens_d_two_group(a, b)
        _, pv = stats.ttest_ind(a, b, equal_var=False)
        p[e] = pv
    return d, p


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * np.arange(1, m + 1) / m
    passed = ranked <= thresh
    if not passed.any():
        return np.zeros(m, dtype=bool)
    cut = np.max(np.where(passed)[0])
    cutoff = ranked[cut]
    return p <= cutoff
