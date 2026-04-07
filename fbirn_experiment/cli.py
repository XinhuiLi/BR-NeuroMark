"""Command-line entry for the FBIRN ICN experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fbirn_experiment.config import (
    DEFAULT_CONFOUND_CSV_PATH,
    DEFAULT_ICN_DOMAIN_PATH,
    DEFAULT_LABEL_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TC_PATH,
)
from fbirn_experiment.io_data import load_fbirn_tc_and_labels, load_npz, synthetic_dataset
from fbirn_experiment.pipeline import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FBIRN ICN FNC experiment (H1: edges vs FA vs ICA; H2; H3 FA)."
    )
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument(
        "--tc",
        type=Path,
        default=DEFAULT_TC_PATH,
        help=f"ICN time courses .npy (default: {DEFAULT_TC_PATH})",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABEL_PATH,
        help=f"Binary labels 0=HC 1=SZ (default: {DEFAULT_LABEL_PATH})",
    )
    parser.add_argument(
        "--icn-domain",
        type=Path,
        default=None,
        help=f"Optional ICN domain labels (default tries {DEFAULT_ICN_DOMAIN_PATH})",
    )
    parser.add_argument(
        "--confounds-csv",
        type=Path,
        default=None,
        help=(
            "CSV with confounding variables to regress out. "
            f"Default: {DEFAULT_CONFOUND_CSV_PATH} (used if --confounds is set)."
        ),
    )
    parser.add_argument(
        "--confound-cols",
        type=str,
        nargs="+",
        default=["age", "sex", "race", "site", "hm"],
        help="Column names in --confounds-csv to regress out (default: age sex race site hm).",
    )
    parser.add_argument(
        "--no-confounds",
        action="store_true",
        help="Skip confound regression even when --confounds-csv is provided.",
    )
    parser.add_argument(
        "--npz",
        type=str,
        default="",
        help="Single .npz with time_courses, y, icn_domain (overrides --tc/--labels)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="GridSearchCV parallel jobs (-1 = all cores).",
    )
    parser.add_argument("--full-grid", action="store_true")
    parser.add_argument("--k-min", type=int, default=5, help="Min latent factors (FA BIC / ICA MSE).")
    parser.add_argument("--k-max", type=int, default=50, help="Max latent factors (FA BIC / ICA MSE).")
    parser.add_argument("--k-step", type=int, default=5, help="Step size for component search grid.")
    parser.add_argument(
        "--fa-criterion",
        choices=("bic", "aic"),
        default="bic",
        help="Information criterion for FA component count on each outer-training fold.",
    )
    parser.add_argument("--h2-perm", type=int, default=2000)
    parser.add_argument(
        "--h3-no-bic",
        action="store_true",
        help="Use fixed H3 factor count (--h3-components) instead of BIC on full data.",
    )
    parser.add_argument(
        "--h3-components",
        type=int,
        default=10,
        help="Fixed number of FA factors for H3 when --h3-no-bic is set.",
    )
    parser.add_argument(
        "--auc-bootstrap",
        type=int,
        default=10000,
        help="Bootstrap resamples for OOF AUC difference p-values.",
    )
    parser.add_argument(
        "--auc-perm-y",
        type=int,
        default=5000,
        help="Label permutations for auxiliary AUC-difference null.",
    )
    parser.add_argument("--no-save", action="store_true", help="Skip writing artifacts")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation")
    args = parser.parse_args()

    if args.synthetic:
        tc, y, icn_domain = synthetic_dataset()
    elif args.npz:
        tc, y, icn_domain = load_npz(args.npz)
    else:
        tc, y, icn_domain = load_fbirn_tc_and_labels(
            args.tc,
            args.labels,
            args.icn_domain,
        )

    if not args.synthetic and not args.npz:
        print(f"Time courses: {args.tc} shape {tuple(tc.shape)}")
        print(
            f"Labels: {args.labels} — HC={int((y == 0).sum())}, SZ={int((y == 1).sum())}"
        )

    # Resolve confound CSV path
    confound_csv: Path | None = None
    if not args.no_confounds:
        if args.confounds_csv is not None:
            confound_csv = args.confounds_csv
        elif not args.synthetic and DEFAULT_CONFOUND_CSV_PATH.is_file():
            confound_csv = DEFAULT_CONFOUND_CSV_PATH
            print(f"Auto-detected confound CSV: {confound_csv}")

    result = run_experiment(
        tc,
        y,
        icn_domain,
        output_dir=args.out,
        confound_csv=confound_csv,
        confound_cols=tuple(args.confound_cols),
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        n_jobs=args.n_jobs,
        full_hyperparameter_search=args.full_grid,
        k_min=args.k_min,
        k_max=args.k_max,
        k_step=args.k_step,
        fa_criterion=args.fa_criterion,
        h2_n_perm=args.h2_perm,
        h3_use_bic=not args.h3_no_bic,
        h3_k_min=args.k_min,
        h3_k_max=args.k_max,
        h3_k_step=args.k_step,
        h3_n_components_fixed=getattr(args, "h3_components", 10),
        auc_bootstrap=args.auc_bootstrap,
        auc_perm_y=args.auc_perm_y,
        save_artifacts=not args.no_save,
        make_figures=not args.no_figures,
    )

    print(f"\nOutputs → {result['output_dir']}")
    print("FNC shape:", result["edges_shape"])
    cm = result.get("confound_meta", {})
    if cm.get("confounds_applied"):
        print(
            f"\nConfounds regressed: {cm['n_regressors']} regressors "
            f"({cm['confound_columns_raw']})"
        )
    else:
        print("\nConfounds: none (raw FNC edges used)")
    print("\nH1 nested CV (ROC-AUC):\n", result["h1_summary"].to_string(index=False))
    print("\nH1 pairwise AUC tests (JSON):\n", json.dumps(result["h1_auc_tests"], indent=2))
    print("\nH2:", result["h2"])
    print("\nH3 selection:", result["h3_selection"])
    print("\nH3 (head):\n", result["h3_summary"].head().to_string(index=False))


if __name__ == "__main__":
    main()
