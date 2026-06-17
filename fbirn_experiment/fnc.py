"""Compatibility exports for FNC helpers.

New code should import from :mod:`fbirn_experiment.connectivity`.
"""

from fbirn_experiment.connectivity import (
    edge_domain_mask,
    edge_pair_mask_for_domains,
    fisher_z,
    fnc_edges_from_tc,
    group_mean_pearson_fisher_z_matrices,
    pairwise_correlation_matrix,
    pearson_fisher_z_connectivity_matrix,
    symmetric_matrix_from_upper_vec,
    triu_indices,
)

__all__ = [
    "edge_domain_mask",
    "edge_pair_mask_for_domains",
    "fisher_z",
    "fnc_edges_from_tc",
    "group_mean_pearson_fisher_z_matrices",
    "pairwise_correlation_matrix",
    "pearson_fisher_z_connectivity_matrix",
    "symmetric_matrix_from_upper_vec",
    "triu_indices",
]
