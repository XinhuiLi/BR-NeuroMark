"""H2: between- vs within-domain Cohen's d summary with ICN-label permutation.

Estimand
--------
Cross-domain FNC edges show different HC-vs-SZ effect-size magnitude
(mean |Cohen's d|) than within-domain edges. The null permutes ICN domain
labels among ICNs, preserving domain counts (exchangeability under a random
ontology assignment, not a spatial-topology-preserving null).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from fbirn_experiment.connectivity import edge_domain_mask


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
    return float((np.mean(x1) - np.mean(x0)) / pooled)


def edge_cohens_d_and_pvalues(
    edges: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    from scipy import stats  # noqa: PLC0415

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


def _delta_mean_abs_d(
    d: np.ndarray, icn_domain: np.ndarray, ii: np.ndarray, jj: np.ndarray
) -> tuple[float, float, float]:
    """Return (delta, mean_abs_d_within, mean_abs_d_between) for domain labels."""
    within, between = edge_domain_mask(icn_domain, ii, jj)
    if not within.any() or not between.any():
        return float("nan"), float("nan"), float("nan")
    mw = float(np.mean(np.abs(d[within])))
    mb = float(np.mean(np.abs(d[between])))
    return mb - mw, mw, mb


def _validate_h2_inputs(
    edges: np.ndarray,
    y: np.ndarray,
    icn_domain: np.ndarray,
    ii: np.ndarray,
    jj: np.ndarray,
    n_perm: int,
    alternative: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    edges = np.asarray(edges, dtype=np.float64)
    y = np.asarray(y).astype(int).ravel()
    icn_domain = np.asarray(icn_domain).ravel()
    ii = np.asarray(ii, dtype=int).ravel()
    jj = np.asarray(jj, dtype=int).ravel()

    if edges.ndim != 2:
        raise ValueError(f"edges must be 2D (subjects × edges); got shape {edges.shape}.")
    if edges.shape[0] != y.shape[0]:
        raise ValueError(f"edges rows ({edges.shape[0]}) must match y length ({y.shape[0]}).")
    if not np.all(np.isfinite(edges)):
        raise ValueError("edges contains non-finite values.")
    classes, counts = np.unique(y, return_counts=True)
    if not np.array_equal(classes, np.array([0, 1])):
        raise ValueError(f"y must be binary with labels {{0, 1}}; got {classes.tolist()}.")
    if int(counts.min()) < 2:
        raise ValueError("H2 requires at least two subjects in each class.")
    if n_perm < 1:
        raise ValueError("n_perm must be at least 1.")
    if alternative not in {"greater", "two-sided"}:
        raise ValueError("alternative must be 'greater' or 'two-sided'.")
    if ii.shape != jj.shape:
        raise ValueError("ii and jj must have the same shape.")
    if ii.shape[0] != edges.shape[1]:
        raise ValueError(
            f"ii/jj length ({ii.shape[0]}) must match edge count ({edges.shape[1]})."
        )
    if icn_domain.ndim != 1 or icn_domain.shape[0] == 0:
        raise ValueError("icn_domain must be a non-empty 1D array.")
    if ii.size and (ii.min() < 0 or jj.min() < 0 or ii.max() >= len(icn_domain) or jj.max() >= len(icn_domain)):
        raise ValueError("ii/jj contain indices outside icn_domain.")
    return edges, y, icn_domain, ii, jj


def h2_domain_label_permutation_test(
    edges: np.ndarray,
    y: np.ndarray,
    icn_domain: np.ndarray,
    ii: np.ndarray,
    jj: np.ndarray,
    n_perm: int = 5000,
    random_state: int = 0,
    alternative: Literal["greater", "two-sided"] = "two-sided",
) -> dict[str, Any]:
    """Permutation test for between- vs within-domain mean |Cohen's d|.

    Parameters
    ----------
    alternative
        ``"two-sided"`` (default) uses |Δ| against the permuted null.
        ``"greater"`` tests H2: mean(|d|)_between > mean(|d|)_within.
    """
    edges, y, icn_domain, ii, jj = _validate_h2_inputs(
        edges, y, icn_domain, ii, jj, n_perm, alternative
    )
    rng = np.random.default_rng(random_state)
    d, _ = edge_cohens_d_and_pvalues(edges, y)
    within, between = edge_domain_mask(icn_domain, ii, jj)
    if not within.any() or not between.any():
        raise ValueError(
            "H2 requires at least one within-domain and one between-domain edge."
        )

    obs, mean_within, mean_between = _delta_mean_abs_d(d, icn_domain, ii, jj)

    null = np.empty(n_perm)
    labels = np.asarray(icn_domain).copy()
    for p in range(n_perm):
        rng.shuffle(labels)
        null[p], _, _ = _delta_mean_abs_d(d, labels, ii, jj)

    p_one_sided = (1 + np.sum(null >= obs)) / (n_perm + 1)
    p_two_sided = (1 + np.sum(np.abs(null) >= abs(obs))) / (n_perm + 1)
    p_primary = p_one_sided if alternative == "greater" else p_two_sided

    return {
        "observed_delta_mean_abs_d": obs,
        "p_value": float(p_primary),
        "p_value_one_sided": float(p_one_sided),
        "p_value_two_sided": float(p_two_sided),
        "alternative": alternative,
        "n_perm": n_perm,
        "permutation_null": (
            "ICN domain labels are shuffled while preserving label counts; "
            "this tests random ontology assignment and does not preserve spatial topology."
        ),
        "n_edges_between": int(between.sum()),
        "n_edges_within": int(within.sum()),
        "mean_abs_d_between": mean_between,
        "mean_abs_d_within": mean_within,
        "null_delta_mean_abs_d": null,
        "edge_cohens_d": d,
        "mask_between": between,
        "mask_within": within,
    }
