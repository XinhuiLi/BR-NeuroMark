"""Compatibility exports for group-mean connectivity cache helpers.

New code should import from :mod:`fbirn_experiment.connectivity_cache`.
"""

from fbirn_experiment.connectivity_cache import (
    read_cache_meta,
    save_group_mean_connectivity_cache,
    try_load_group_mean_connectivity_cache,
    validate_cache_matches_data,
)

__all__ = [
    "read_cache_meta",
    "save_group_mean_connectivity_cache",
    "try_load_group_mean_connectivity_cache",
    "validate_cache_matches_data",
]
