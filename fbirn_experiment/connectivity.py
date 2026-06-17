"""Functional network connectivity computation and group summaries."""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np

ConnectivityMethod = Literal["pearson_z", "spearman", "partial_corr", "mutual_info"]

CONNECTIVITY_METHODS: list[ConnectivityMethod] = [
    "pearson_z",
    "spearman",
    "partial_corr",
    "mutual_info",
]


def pairwise_correlation_matrix(tc: np.ndarray) -> np.ndarray:
    """tc: (n_timepoints, n_icns) -> (n_icns, n_icns) Pearson correlation."""
    if tc.shape[0] < 3:
        raise ValueError("Need at least 3 time points for stable correlation.")
    c = np.corrcoef(tc, rowvar=False)
    return np.clip(c, -1.0, 1.0)


def fisher_z(r: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    r = np.clip(r, -1.0 + eps, 1.0 - eps)
    return np.arctanh(r)


def triu_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n, k=1)


def symmetric_matrix_from_upper_vec(
    vec: np.ndarray,
    ii: np.ndarray,
    jj: np.ndarray,
    n: int,
) -> np.ndarray:
    """Map strict upper-triangle edge values to an n x n symmetric matrix."""
    m = np.zeros((n, n), dtype=np.float64)
    m[ii, jj] = vec
    m[jj, ii] = vec
    return m


def _pairwise_spearman(tc: np.ndarray) -> np.ndarray:
    """tc: (n_timepoints, n_icns) → (n_icns, n_icns) Spearman correlation."""
    from scipy import stats as sp_stats  # noqa: PLC0415

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


def pairwise_connectivity_matrix(
    tc: np.ndarray,
    method: ConnectivityMethod = "pearson_z",
) -> np.ndarray:
    """Full ICN×ICN connectivity for one subject (same scaling as :func:`fnc_edges`).

    Parameters
    ----------
    tc : (n_timepoints, n_icns)
    method : connectivity fork code

    Returns
    -------
    (n_icns, n_icns) symmetric matrix. Pearson, Spearman, and partial correlation
    are returned in Fisher *z*; mutual information is in sklearn's nat-like units.
    """
    if method == "pearson_z":
        return fisher_z(pairwise_correlation_matrix(tc))
    if method == "spearman":
        r = _pairwise_spearman(tc)
        return fisher_z(r)
    if method == "partial_corr":
        r = _pairwise_partial_corr(tc)
        return fisher_z(r)
    if method == "mutual_info":
        return _pairwise_mutual_info(tc)
    raise ValueError(f"Unknown connectivity method: {method}")


def pearson_fisher_z_connectivity_matrix(tc: np.ndarray) -> np.ndarray:
    """Full ICN x ICN Pearson correlation transformed to Fisher z."""
    return pairwise_connectivity_matrix(tc, "pearson_z")


def group_mean_connectivity_matrices(
    time_courses: np.ndarray,
    y: np.ndarray,
    method: ConnectivityMethod,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Group-mean full connectivity matrices (HC vs SZ) for one method.

    Parameters
    ----------
    time_courses : (n_subj, n_time, n_icns)
    y : (n_subj,) ``0`` = HC, ``1`` = SZ
    method : connectivity fork code

    Returns
    -------
    mean_hc, mean_sz : (n_icns, n_icns)
    n_hc, n_sz : int
    """
    y = np.asarray(y).astype(int).ravel()
    if time_courses.shape[0] != y.shape[0]:
        raise ValueError("time_courses and y must have the same n_subj dimension.")
    mask_hc = y == 0
    mask_sz = y == 1
    n_hc = int(mask_hc.sum())
    n_sz = int(mask_sz.sum())
    if n_hc < 1 or n_sz < 1:
        raise ValueError(f"Need at least one HC and one SZ subject; got HC={n_hc}, SZ={n_sz}.")

    n_icns = int(time_courses.shape[2])
    sum_hc = np.zeros((n_icns, n_icns), dtype=np.float64)
    sum_sz = np.zeros((n_icns, n_icns), dtype=np.float64)
    for s in range(time_courses.shape[0]):
        m = pairwise_connectivity_matrix(time_courses[s], method)
        if mask_hc[s]:
            sum_hc += m
        else:
            sum_sz += m
    return sum_hc / n_hc, sum_sz / n_sz, n_hc, n_sz


def compute_group_mean_connectivity_rows(
    time_courses: np.ndarray,
    y: np.ndarray,
    measures: Sequence[ConnectivityMethod] | None = None,
) -> tuple[list[tuple[str, np.ndarray, np.ndarray]], int, int, int]:
    """Group-mean full matrices for each measure (HC and SZ)."""
    order: list[ConnectivityMethod] = (
        list(measures) if measures is not None else list(CONNECTIVITY_METHODS)
    )
    rows: list[tuple[str, np.ndarray, np.ndarray]] = []
    n_hc = n_sz = 0
    for m in order:
        mean_hc, mean_sz, n_hc, n_sz = group_mean_connectivity_matrices(
            time_courses, y, m
        )
        rows.append((m, mean_hc, mean_sz))
    return rows, n_hc, n_sz, int(time_courses.shape[2])


def group_mean_pearson_fisher_z_matrices(
    time_courses: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Mean Fisher-z Pearson connectivity in z-space per group (HC vs SZ)."""
    return group_mean_connectivity_matrices(time_courses, y, "pearson_z")


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
        m = pairwise_connectivity_matrix(time_courses[s], method)
        edges[s] = m[ii, jj]

    return edges, (ii, jj)


def fnc_edges_from_tc(
    time_courses: np.ndarray,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Backward-compatible alias for Pearson Fisher-z FNC edges."""
    return fnc_edges(time_courses, "pearson_z")


def edge_domain_mask(
    icn_domain: np.ndarray, ii: np.ndarray, jj: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    same = np.array([icn_domain[i] == icn_domain[j] for i, j in zip(ii, jj)])
    return same, ~same


def edge_pair_mask_for_domains(
    icn_domain: np.ndarray,
    ii: np.ndarray,
    jj: np.ndarray,
    domain_a: Any,
    domain_b: Any,
) -> np.ndarray:
    mask = np.zeros(len(ii), dtype=bool)
    for k, (i, j) in enumerate(zip(ii, jj)):
        di, dj = icn_domain[i], icn_domain[j]
        if (di == domain_a and dj == domain_b) or (di == domain_b and dj == domain_a):
            mask[k] = True
    return mask
