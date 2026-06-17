"""Compatibility exports for H2 effect-size helpers.

New code should import from :mod:`fbirn_experiment.h2_test`.
"""

from fbirn_experiment.h2_test import cohens_d_two_group, edge_cohens_d_and_pvalues

__all__ = ["cohens_d_two_group", "edge_cohens_d_and_pvalues"]
