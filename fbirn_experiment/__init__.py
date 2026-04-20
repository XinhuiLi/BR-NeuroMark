"""Modular FBIRN ICN → FNC → H1/H2/H3 analysis pipeline."""

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
from fbirn_experiment.confounds import (
    build_design_matrix,
    load_confounds,
    regress_confounds,
    regress_confounds_cv,
)
from fbirn_experiment.connectivity import (
    CONNECTIVITY_METHODS,
    ConnectivityMethod,
    fnc_edges,
)
from fbirn_experiment.domain_labels import (
    DOMAIN_GRANULARITIES,
    DomainGranularity,
    aggregate_domains,
)
from fbirn_experiment.fnc import fnc_edges_from_tc
from fbirn_experiment.h1_cv import (
    NestedCVResult,
    compute_h1_stability_tests,
    fit_h1_interpretability_refits,
    nested_cv_classifiers,
    summarize_h1,
)
from fbirn_experiment.h2_test import h2_domain_label_permutation_test
from fbirn_experiment.h3_test import H3Result, SZ_HYPOTHESIS_PAIRS, h3_factor_loadings_between_within
from fbirn_experiment.io_data import (
    load_fbirn_tc_and_labels,
    load_neuromark_labels,
    load_npz,
    synthetic_dataset,
)
from fbirn_experiment.multiverse import (
    CLASSIFIER_CHOICES,
    CONFOUND_STRATEGIES,
    REDUCTION_METHODS,
    MultispecConfig,
    SpecResult,
    enumerate_multiverse,
    run_multiverse,
    run_single_spec,
)
from fbirn_experiment.pipeline import run_experiment

__all__ = [
    "CLASSIFIER_CHOICES",
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
    "compute_h1_stability_tests",
    "fit_h1_interpretability_refits",
    "REDUCTION_METHODS",
    "SZ_HYPOTHESIS_PAIRS",
    "SpecResult",
    "PROJECT_ROOT",
    "aggregate_domains",
    "build_design_matrix",
    "enumerate_multiverse",
    "fnc_edges",
    "fnc_edges_from_tc",
    "h2_domain_label_permutation_test",
    "h3_factor_loadings_between_within",
    "load_confounds",
    "load_fbirn_tc_and_labels",
    "load_neuromark_labels",
    "load_npz",
    "nested_cv_classifiers",
    "regress_confounds",
    "regress_confounds_cv",
    "run_experiment",
    "run_multiverse",
    "run_single_spec",
    "summarize_h1",
    "synthetic_dataset",
]
