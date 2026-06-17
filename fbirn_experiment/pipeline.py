"""End-to-end experiment: FNC → H1/H2/H3 → save artifacts → figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from fbirn_experiment.config import DEFAULT_OUTPUT_DIR
from fbirn_experiment.confounds import (
    build_design_matrix,
    confound_summary,
    load_confounds,
    regress_confounds,
)
from fbirn_experiment.figures import generate_all_figures
from fbirn_experiment.connectivity import fnc_edges_from_tc
from fbirn_experiment.h1_cv import (
    all_pairwise_auc_tests,
    compute_h1_stability_tests,
    fit_h1_interpretability_refits,
    nested_cv_classifiers,
    summarize_h1,
)
from fbirn_experiment.h2_test import h2_domain_label_permutation_test
from fbirn_experiment.h3_test import h3_factor_loadings_between_within
from fbirn_experiment.persistence import (
    ensure_dir,
    save_fnc_bundle,
    save_h1_outputs,
    save_h2_outputs,
    save_h3_outputs,
    save_json,
)


def run_experiment(
    time_courses: np.ndarray,
    y: np.ndarray,
    icn_domain: np.ndarray,
    *,
    output_dir: Path | str | None = None,
    confound_csv: Path | str | None = None,
    confound_cols: Sequence[str] = ("age", "sex", "race", "site", "hm"),
    outer_splits: int = 5,
    inner_splits: int = 3,
    n_jobs: int = 1,
    full_hyperparameter_search: bool = False,
    k_min: int = 5,
    k_max: int = 50,
    k_step: int = 5,
    fa_criterion: str = "bic",
    h2_n_perm: int = 500,
    h3_use_bic: bool = True,
    h3_decomposition: Literal["fa", "ica"] = "ica",
    h3_k_min: int = 5,
    h3_k_max: int = 50,
    h3_k_step: int = 5,
    h3_n_components_fixed: int = 10,
    auc_bootstrap: int = 10000,
    auc_perm_y: int = 5000,
    save_artifacts: bool = True,
    make_figures: bool = True,
    h1_include_fa: bool = False,
    random_state_h2: int = 1,
    random_state_h3: int = 0,
    auc_test_seed: int = 42,
) -> dict[str, Any]:
    out = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    artifacts = ensure_dir(out / "artifacts")

    edges, (ii, jj) = fnc_edges_from_tc(time_courses)
    y = np.asarray(y).astype(int)
    n_icns = int(time_courses.shape[2])

    # ── Confound regression ──────────────────────────────────────────────
    confound_matrix: np.ndarray | None = None
    confound_meta: dict[str, Any] = {"confounds_applied": False}

    if confound_csv is not None:
        csv_path = Path(confound_csv)
        if not csv_path.is_file():
            raise FileNotFoundError(f"Confound CSV not found: {csv_path}")
        confound_df = load_confounds(csv_path, confound_cols)
        if len(confound_df) != edges.shape[0]:
            raise ValueError(
                f"Confound CSV has {len(confound_df)} rows but data has "
                f"{edges.shape[0]} subjects."
            )
        confound_matrix, col_names = build_design_matrix(confound_df, confound_cols)
        confound_meta = {
            "confounds_applied": True,
            **confound_summary(confound_df, confound_matrix, col_names),
        }
        print(
            f"Confound regression: {confound_matrix.shape[1]} regressors "
            f"({col_names}) from {csv_path.name}"
        )
        # Full-sample residualization for H2 / H3
        edges_clean = regress_confounds(edges, confound_matrix)
    else:
        edges_clean = edges

    if save_artifacts:
        save_fnc_bundle(artifacts, edges, ii, jj, y, icn_domain)
        run_meta: dict[str, Any] = {
            "n_subjects": int(time_courses.shape[0]),
            "n_timepoints": int(time_courses.shape[1]),
            "n_icns": n_icns,
            "n_fnc_edges": int(edges.shape[1]),
            "outer_splits": outer_splits,
            "inner_splits": inner_splits,
            "k_min": k_min,
            "k_max": k_max,
            "k_step": k_step,
            "fa_criterion": fa_criterion,
            "h1_include_fa": bool(h1_include_fa),
            "h1_edges_classifier": "logistic_l2",
            "h3_decomposition": h3_decomposition,
            "h3_auto_k": bool(h3_use_bic),
            **confound_meta,
        }
        save_json(out / "run_meta.json", run_meta)

    # H1: fold-wise confound regression inside nested CV
    res_e, res_fa, res_ica = nested_cv_classifiers(
        edges,
        y,
        outer_splits=outer_splits,
        inner_splits=inner_splits,
        n_jobs=n_jobs,
        full_hyperparameter_search=full_hyperparameter_search,
        k_min=k_min,
        k_max=k_max,
        k_step=k_step,
        fa_criterion=fa_criterion if fa_criterion in ("bic", "aic") else "bic",
        confound_matrix=confound_matrix,
        include_fa=h1_include_fa,
    )
    h1_summary = summarize_h1(res_e, res_fa, res_ica)
    auc_tests = all_pairwise_auc_tests(
        res_e,
        res_fa,
        res_ica,
        n_bootstrap=auc_bootstrap,
        n_perm_y=auc_perm_y,
        random_state=auc_test_seed,
    )
    h1_stability = compute_h1_stability_tests(res_e, res_fa, res_ica)
    h1_interp = fit_h1_interpretability_refits(edges, y, res_e, res_fa, res_ica)

    if save_artifacts:
        save_h1_outputs(
            artifacts,
            res_e,
            res_fa,
            res_ica,
            h1_summary,
            auc_tests,
            stability_tests=h1_stability,
            interpretability=h1_interp,
        )

    # H2 / H3: use full-sample residualized edges
    h2 = h2_domain_label_permutation_test(
        edges_clean,
        y,
        icn_domain,
        ii,
        jj,
        n_perm=h2_n_perm,
        random_state=random_state_h2,
    )
    if save_artifacts:
        save_h2_outputs(artifacts, h2)

    h3 = h3_factor_loadings_between_within(
        edges_clean,
        y,
        icn_domain,
        ii,
        jj,
        decomposition=h3_decomposition,
        use_bic_selection=h3_use_bic,
        k_min=h3_k_min,
        k_max=h3_k_max,
        k_step=h3_k_step,
        fa_criterion=fa_criterion if fa_criterion in ("bic", "aic") else "bic",
        n_components_fixed=h3_n_components_fixed,
        random_state=random_state_h3,
    )
    if save_artifacts:
        save_h3_outputs(
            artifacts,
            h3.summary,
            h3.loadings,
            h3.selection,
            h3.domain_pair_summary,
        )

    if make_figures:
        generate_all_figures(
            out,
            res_e,
            res_fa,
            res_ica,
            h2,
            h3.summary,
            h3.loadings,
            ii,
            jj,
            n_icns,
            icn_domain=icn_domain,
            h3_domain_pair_summary=h3.domain_pair_summary,
            h1_stability=h1_stability,
            h1_interpretability=h1_interp,
            h3_decomposition=str(h3.selection.get("decomposition", h3_decomposition)),
        )

    return {
        "output_dir": str(out),
        "edges_shape": edges.shape,
        "confound_meta": confound_meta,
        "h1_summary": h1_summary,
        "h1_auc_tests": auc_tests,
        "h1_stability": h1_stability,
        "h1_interpretability": {
            "hyperparams_edges": h1_interp["hyperparams_edges"],
            **(
                {"hyperparams_fa": h1_interp["hyperparams_fa"]}
                if "hyperparams_fa" in h1_interp
                else {}
            ),
            "hyperparams_ica": h1_interp["hyperparams_ica"],
        },
        "h2": {k: v for k, v in h2.items() if k not in ("null_delta_mean_abs_d", "edge_cohens_d", "mask_between", "mask_within")},
        "h3_summary": h3.summary,
        "h3_domain_pair_summary": h3.domain_pair_summary,
        "h3_selection": h3.selection,
        "res_edges": res_e,
        "res_fa": res_fa,
        "res_ica": res_ica,
    }
