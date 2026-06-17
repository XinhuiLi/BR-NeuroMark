"""Modular FBIRN ICN -> FNC -> H1/H2/H3 analysis pipeline."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from fbirn_experiment.config import (
    DEFAULT_CONFOUND_CSV_PATH,
    DEFAULT_DATA_DIR,
    DEFAULT_ICN_DOMAIN_PATH,
    DEFAULT_LABEL_PATH,
    DEFAULT_NEUROMARK_XLSX_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TC_PATH,
    PROJECT_ROOT,
)
from fbirn_experiment.connectivity import (
    CONNECTIVITY_METHODS,
    ConnectivityMethod,
    compute_group_mean_connectivity_rows,
    fnc_edges,
    fnc_edges_from_tc,
    group_mean_connectivity_matrices,
    pairwise_connectivity_matrix,
)
from fbirn_experiment.domain_labels import (
    DOMAIN_GRANULARITIES,
    DomainGranularity,
    aggregate_domains,
)
from fbirn_experiment.io_data import (
    load_fbirn_tc_and_labels,
    load_neuromark_labels,
    load_npz,
    synthetic_dataset,
)

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "build_design_matrix": ("fbirn_experiment.confounds", "build_design_matrix"),
    "CLASSIFIER_CHOICES": ("fbirn_experiment.multiverse", "CLASSIFIER_CHOICES"),
    "compute_h1_stability_tests": ("fbirn_experiment.h1_cv", "compute_h1_stability_tests"),
    "CONFOUND_STRATEGIES": ("fbirn_experiment.multiverse", "CONFOUND_STRATEGIES"),
    "fit_h1_interpretability_refits": ("fbirn_experiment.h1_cv", "fit_h1_interpretability_refits"),
    "H3Result": ("fbirn_experiment.h3_test", "H3Result"),
    "h2_domain_label_permutation_test": (
        "fbirn_experiment.h2_test",
        "h2_domain_label_permutation_test",
    ),
    "h3_factor_loadings_between_within": (
        "fbirn_experiment.h3_test",
        "h3_factor_loadings_between_within",
    ),
    "load_confounds": ("fbirn_experiment.confounds", "load_confounds"),
    "MultispecConfig": ("fbirn_experiment.multiverse", "MultispecConfig"),
    "NestedCVResult": ("fbirn_experiment.h1_cv", "NestedCVResult"),
    "nested_cv_classifiers": ("fbirn_experiment.h1_cv", "nested_cv_classifiers"),
    "REDUCTION_METHODS": ("fbirn_experiment.multiverse", "REDUCTION_METHODS"),
    "regress_confounds": ("fbirn_experiment.confounds", "regress_confounds"),
    "regress_confounds_cv": ("fbirn_experiment.confounds", "regress_confounds_cv"),
    "run_experiment": ("fbirn_experiment.pipeline", "run_experiment"),
    "run_multiverse": ("fbirn_experiment.multiverse", "run_multiverse"),
    "run_single_spec": ("fbirn_experiment.multiverse", "run_single_spec"),
    "SpecResult": ("fbirn_experiment.multiverse", "SpecResult"),
    "summarize_h1": ("fbirn_experiment.h1_cv", "summarize_h1"),
    "SZ_HYPOTHESIS_PAIRS": ("fbirn_experiment.h3_test", "SZ_HYPOTHESIS_PAIRS"),
    "enumerate_multiverse": ("fbirn_experiment.multiverse", "enumerate_multiverse"),
}

__all__ = [
    "CLASSIFIER_CHOICES",
    "compute_group_mean_connectivity_rows",
    "CONFOUND_STRATEGIES",
    "CONNECTIVITY_METHODS",
    "ConnectivityMethod",
    "DEFAULT_CONFOUND_CSV_PATH",
    "DEFAULT_DATA_DIR",
    "DEFAULT_ICN_DOMAIN_PATH",
    "DEFAULT_LABEL_PATH",
    "DEFAULT_NEUROMARK_XLSX_PATH",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_TC_PATH",
    "DOMAIN_GRANULARITIES",
    "DomainGranularity",
    "H3Result",
    "MultispecConfig",
    "NestedCVResult",
    "PROJECT_ROOT",
    "REDUCTION_METHODS",
    "SZ_HYPOTHESIS_PAIRS",
    "SpecResult",
    "aggregate_domains",
    "build_design_matrix",
    "compute_h1_stability_tests",
    "enumerate_multiverse",
    "fit_h1_interpretability_refits",
    "fnc_edges",
    "fnc_edges_from_tc",
    "group_mean_connectivity_matrices",
    "h2_domain_label_permutation_test",
    "h3_factor_loadings_between_within",
    "load_confounds",
    "load_fbirn_tc_and_labels",
    "load_neuromark_labels",
    "load_npz",
    "nested_cv_classifiers",
    "pairwise_connectivity_matrix",
    "regress_confounds",
    "regress_confounds_cv",
    "run_experiment",
    "run_multiverse",
    "run_single_spec",
    "summarize_h1",
    "synthetic_dataset",
]


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
