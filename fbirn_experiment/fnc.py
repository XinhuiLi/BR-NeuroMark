"""Functional network connectivity from ICN time courses."""

from __future__ import annotations

from typing import Any

import numpy as np


def pairwise_correlation_matrix(tc: np.ndarray) -> np.ndarray:
    """tc: (n_timepoints, n_icns) → (n_icns, n_icns) Pearson correlation."""
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
    """
    Map upper-triangle edge values (strict upper) to an n x n symmetric matrix
    with zeros on the diagonal.
    """
    m = np.zeros((n, n), dtype=np.float64)
    m[ii, jj] = vec
    m[jj, ii] = vec
    return m


def fnc_edges_from_tc(
    time_courses: np.ndarray,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    time_courses: (n_subj, n_t, n_icns)
    Returns edges (n_subj, n_edges) Fisher-z upper triangle, and (ii, jj).
    """
    n_subj, _, n_icns = time_courses.shape
    ii, jj = triu_indices(n_icns)
    n_edges = ii.shape[0]
    edges = np.empty((n_subj, n_edges), dtype=np.float64)
    for s in range(n_subj):
        r = pairwise_correlation_matrix(time_courses[s])
        z = fisher_z(r)
        edges[s] = z[ii, jj]
    return edges, (ii, jj)


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
