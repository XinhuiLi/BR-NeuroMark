"""Generalized FNC computation: Pearson, Spearman, partial correlation, MI."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import stats as sp_stats

from fbirn_experiment.fnc import fisher_z, triu_indices

ConnectivityMethod = Literal["pearson_z", "spearman", "partial_corr", "mutual_info"]

CONNECTIVITY_METHODS: list[ConnectivityMethod] = [
    "pearson_z",
    "spearman",
    "partial_corr",
    "mutual_info",
]


def _pairwise_spearman(tc: np.ndarray) -> np.ndarray:
    """tc: (n_timepoints, n_icns) → (n_icns, n_icns) Spearman correlation."""
    n_icns = tc.shape[1]
    ranked = np.empty_like(tc)
    for c in range(n_icns):
        ranked[:, c] = sp_stats.rankdata(tc[:, c])
    r = np.corrcoef(ranked, rowvar=False)
    return np.clip(r, -1.0, 1.0)


def _pairwise_partial_corr(tc: np.ndarray) -> np.ndarray:
    """tc: (n_timepoints, n_icns) → (n_icns, n_icns) partial correlation.

    Uses Ledoit-Wolf shrinkage for stable precision matrix estimation,
    then converts precision to partial correlation:
        ρ_ij = −P_ij / √(P_ii · P_jj)
    """
    from sklearn.covariance import LedoitWolf  # noqa: PLC0415

    lw = LedoitWolf()
    lw.fit(tc)
    prec = lw.precision_
    d = np.sqrt(np.diag(prec))
    pcorr = -prec / np.outer(d, d)
    np.fill_diagonal(pcorr, 0.0)
    return np.clip(pcorr, -1.0, 1.0)


def _pairwise_mutual_info(tc: np.ndarray, n_neighbors: int = 5) -> np.ndarray:
    """tc: (n_timepoints, n_icns) → (n_icns, n_icns) MI (KSG estimator).

    Uses sklearn's ``mutual_info_regression`` with KSG k-NN estimator.
    The matrix is symmetric with zeros on the diagonal.
    """
    from sklearn.feature_selection import mutual_info_regression  # noqa: PLC0415

    n_icns = tc.shape[1]
    mi = np.zeros((n_icns, n_icns), dtype=np.float64)
    for i in range(n_icns):
        vals = mutual_info_regression(
            tc, tc[:, i], n_neighbors=n_neighbors, random_state=0
        )
        mi[i, :] = vals
    mi = (mi + mi.T) / 2.0
    np.fill_diagonal(mi, 0.0)
    return mi


def fnc_edges(
    time_courses: np.ndarray,
    method: ConnectivityMethod = "pearson_z",
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Compute FNC edges for all subjects.

    Parameters
    ----------
    time_courses : (n_subj, n_t, n_icns)
    method : one of ``"pearson_z"``, ``"spearman"``, ``"partial_corr"``, ``"mutual_info"``

    Returns
    -------
    edges : (n_subj, n_edges)
    (ii, jj) : upper-triangle index arrays
    """
    n_subj, _, n_icns = time_courses.shape
    ii, jj = triu_indices(n_icns)
    n_edges = ii.shape[0]
    edges = np.empty((n_subj, n_edges), dtype=np.float64)

    for s in range(n_subj):
        tc = time_courses[s]
        if method == "pearson_z":
            r = np.corrcoef(tc, rowvar=False)
            r = np.clip(r, -1.0, 1.0)
            m = fisher_z(r)
        elif method == "spearman":
            r = _pairwise_spearman(tc)
            m = fisher_z(r)
        elif method == "partial_corr":
            r = _pairwise_partial_corr(tc)
            m = fisher_z(r)
        elif method == "mutual_info":
            m = _pairwise_mutual_info(tc)
        else:
            raise ValueError(f"Unknown connectivity method: {method}")
        edges[s] = m[ii, jj]

    return edges, (ii, jj)
