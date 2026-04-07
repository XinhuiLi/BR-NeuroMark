"""H2: between- vs within-domain Cohen's d summary with ICN-label permutation."""

from __future__ import annotations

from typing import Any

import numpy as np

from fbirn_experiment.fnc import edge_domain_mask
from fbirn_experiment.stats_utils import edge_cohens_d_and_pvalues


def h2_domain_label_permutation_test(
    edges: np.ndarray,
    y: np.ndarray,
    icn_domain: np.ndarray,
    ii: np.ndarray,
    jj: np.ndarray,
    n_perm: int = 5000,
    random_state: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(random_state)
    d, _ = edge_cohens_d_and_pvalues(edges, y)
    within, between = edge_domain_mask(icn_domain, ii, jj)

    def stat(dom: np.ndarray) -> float:
        w, b = edge_domain_mask(dom, ii, jj)
        mb = np.mean(np.abs(d[b]))
        mw = np.mean(np.abs(d[w]))
        return float(mb - mw)

    obs = stat(icn_domain)
    null = np.empty(n_perm)
    labels = np.array(icn_domain, copy=True)
    for p in range(n_perm):
        rng.shuffle(labels)
        null[p] = stat(labels)
    pval = (1 + np.sum(np.abs(null) >= abs(obs))) / (n_perm + 1)
    return {
        "observed_delta_mean_abs_d": obs,
        "p_value_two_sided": float(pval),
        "n_perm": n_perm,
        "mean_abs_d_between": float(np.mean(np.abs(d[between]))),
        "mean_abs_d_within": float(np.mean(np.abs(d[within]))),
        "null_delta_mean_abs_d": null,
        "edge_cohens_d": d,
        "mask_between": between,
        "mask_within": within,
    }
