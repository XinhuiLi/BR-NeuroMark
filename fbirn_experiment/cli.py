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
from fbirn_experiment.confounds import DEFAULT_CONFOUND_COLS
from fbirn_experiment.io_data import load_fbirn_tc_and_labels, load_npz, synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "FBIRN ICN FNC experiment (H1: edges L2 logistic vs ICA; optional FA; H2; H3 ICA loadings)."
        ),
    )
    sub = parser.add_subparsers(dest="command")
    _add_run_parser(sub)
    _add_multiverse_parser(sub)
    _add_regen_h1_latent_figs_parser(sub)
    _add_regen_multiverse_figs_parser(sub)
    _add_study_design_parser(sub)
    _add_mean_fnc_matrices_parser(sub)
    _add_plot_confounds_parser(sub)
    _populate_run_args(parser)

    args = parser.parse_args()
    if args.command == "multiverse":
        _run_multiverse_cmd(args)
    elif args.command == "regen-h1-latent-figs":
        _run_regen_h1_latent_figs_cmd(args)
    elif args.command == "regen-multiverse-figs":
        _run_regen_multiverse_figs_cmd(args)
    elif args.command == "study-design":
        _run_study_design_cmd(args)
    elif args.command == "mean-fnc-matrices":
        _run_mean_fnc_matrices_cmd(args)
    elif args.command == "plot-confounds":
        _run_plot_confounds_cmd(args)
    else:
        _run_main_cmd(args)


def _add_multiverse_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("multiverse", help="Run multiverse analysis.")
    p.add_argument("--tc", type=Path, default=DEFAULT_TC_PATH)
    p.add_argument("--labels", type=Path, default=DEFAULT_LABEL_PATH)
    p.add_argument("--icn-domain", type=Path, default=None)
    p.add_argument("--confounds-csv", type=Path, default=None)
    p.add_argument(
        "--confound-cols", type=str, nargs="+",
        default=["age", "sex", "race", "site", "hm"],
    )
    p.add_argument("--out", type=Path, default=Path("results/multiverse"))
    p.add_argument("--outer-splits", type=int, default=5)
    p.add_argument("--inner-splits", type=int, default=3)
    p.add_argument("--k-min", type=int, default=5)
    p.add_argument("--k-max", type=int, default=50)
    p.add_argument("--k-step", type=int, default=5)
    p.add_argument("--h2-perm", type=int, default=500)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--mini", action="store_true", help="Run mini-multiverse (48 specs) only.")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument(
        "--n-jobs", type=int, default=1,
        help="Parallel workers for multiverse specs (-1 = all cores). Default: 1 (serial).",
    )
    p.add_argument(
        "--connectivity", type=str, nargs="+", default=None,
        help="Override connectivity methods (e.g. pearson_z spearman partial_corr mutual_info).",
    )
    p.add_argument(
        "--confound-strategies", type=str, nargs="+", default=None,
        help="Override confound strategies (e.g. none ols combat).",
    )
    p.add_argument(
        "--reductions", type=str, nargs="+", default=None,
        help="Override reduction methods (e.g. none fa ica pca nmf).",
    )
    p.add_argument(
        "--classifiers", type=str, nargs="+", default=None,
        help="Override classifiers (e.g. elasticnet logistic_l2 svm_linear rf).",
    )
    p.add_argument(
        "--granularities", type=str, nargs="+", default=None,
        help="Override domain granularities (e.g. subdomain_14 domain_7).",
    )


def _run_multiverse_cmd(args: argparse.Namespace) -> None:
    from fbirn_experiment.multiverse import (  # noqa: PLC0415
        enumerate_multiverse,
        run_multiverse,
    )
    from fbirn_experiment.multiverse_figures import (  # noqa: PLC0415
        generate_multiverse_figures,
    )

    if args.synthetic:
        tc, y, icn_domain = synthetic_dataset()
    else:
        tc, y, icn_domain = load_fbirn_tc_and_labels(
            args.tc, args.labels, args.icn_domain,
        )

    confound_csv: Path | None = None
    if args.confounds_csv is not None:
        confound_csv = args.confounds_csv
    elif not args.synthetic and DEFAULT_CONFOUND_CSV_PATH.is_file():
        confound_csv = DEFAULT_CONFOUND_CSV_PATH
        print(f"Auto-detected confound CSV: {confound_csv}")

    if args.mini:
        specs = enumerate_multiverse(
            connectivity=["pearson_z", "spearman"],
            confound=["none", "ols"],
            reduction=["none", "fa", "ica"],
            classifier=["logistic_l2", "svm_linear"],
            domain_granularity=["domain_7", "subdomain_14"],
        )
    else:
        specs = enumerate_multiverse(
            connectivity=args.connectivity,
            confound=args.confound_strategies,
            reduction=args.reductions,
            classifier=args.classifiers,
            domain_granularity=args.granularities,
        )

    print(f"Multiverse: {len(specs)} specifications")

    df = run_multiverse(
        tc, y, icn_domain, specs,
        confound_csv=confound_csv,
        confound_cols=tuple(args.confound_cols),
        output_dir=args.out,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        k_min=args.k_min,
        k_max=args.k_max,
        k_step=args.k_step,
        h2_n_perm=args.h2_perm,
        n_jobs=args.n_jobs,
    )

    if not args.no_figures:
        fig_dir = args.out / "figures"
        generate_multiverse_figures(df, fig_dir)

    print(f"\nMultiverse results → {args.out}")


def _add_regen_h1_latent_figs_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "regen-h1-latent-figs",
        help=(
            "Redraw h1_ica_latent_interpretability.png (and h1_fa_*.png if FA coefs exist in artifacts) "
            "from saved NPZ only (no data / CV re-run)."
        ),
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Experiment output directory (reads run_dir/artifacts, writes run_dir/figures).",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="PNG resolution.",
    )
    p.add_argument(
        "--matrix-col-inches",
        type=float,
        default=None,
        help=(
            "Figure width per matrix column (matplotlib inches). "
            "Default from plot_h1_latent_coefficients_and_loadings (~2.12)."
        ),
    )
    p.add_argument(
        "--matrix-row-inches",
        type=float,
        default=None,
        help=(
            "Figure height per matrix row (matplotlib inches). "
            "Default from plot_h1_latent_coefficients_and_loadings (~1.58)."
        ),
    )
    p.add_argument(
        "--matrix-hspace",
        type=float,
        default=None,
        help="Vertical gap between matrix subplots (matplotlib fraction; default ~0.09).",
    )
    p.add_argument(
        "--matrix-wspace",
        type=float,
        default=None,
        help="Horizontal gap between matrix subplots (matplotlib fraction; default ~0.09).",
    )


def _add_regen_multiverse_figs_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "regen-multiverse-figs",
        help=(
            "Regenerate multiverse PNG/CSV outputs from multiverse_results.csv "
            "(no data load, no spec re-run)."
        ),
    )
    p.add_argument(
        "--multiverse-dir",
        type=Path,
        default=Path("results/multiverse"),
        help=(
            "Directory that contains multiverse_results.csv when --results-csv is omitted "
            "(figures default to multiverse-dir/figures)."
        ),
    )
    p.add_argument(
        "--results-csv",
        type=Path,
        default=None,
        help="Path to multiverse_results.csv (default: multiverse-dir/multiverse_results.csv).",
    )
    p.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Output directory for figures/CSVs (default: parent of CSV / figures).",
    )
    p.add_argument("--dpi", type=int, default=200, help="Figure resolution for PNGs.")


def _run_regen_multiverse_figs_cmd(args: argparse.Namespace) -> None:
    from fbirn_experiment.multiverse_figures import regenerate_multiverse_figures  # noqa: PLC0415

    if args.results_csv is not None:
        csv_path = Path(args.results_csv)
    else:
        csv_path = Path(args.multiverse_dir) / "multiverse_results.csv"
    fig_dir = Path(args.figures_dir) if args.figures_dir is not None else csv_path.parent / "figures"
    regenerate_multiverse_figures(csv_path, figures_dir=fig_dir, dpi=int(args.dpi))


def _add_study_design_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "study-design",
        help="Write a schematic study-design / workflow PNG (no data load).",
    )
    p.add_argument(
        "-o", "--out",
        type=Path,
        default=Path("figures/study_design_workflow.png"),
        help="Output PNG path.",
    )
    p.add_argument("--dpi", type=int, default=300, help="Raster resolution.")


def _run_study_design_cmd(args: argparse.Namespace) -> None:
    from fbirn_experiment.study_design_figure import plot_study_design  # noqa: PLC0415

    plot_study_design(Path(args.out), dpi=int(args.dpi))


def _add_mean_fnc_matrices_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "mean-fnc-matrices",
        help=(
            "Plot mean FNC matrices (controls vs patients) from ICN time courses; "
            "use --all-measures for a 4×3 grid (measures × HC/SZ/diff), or "
            "--all-measures --groups-as-rows for a 3×4 grid (groups × measures)."
        ),
    )
    p.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    p.add_argument("--tc", type=Path, default=DEFAULT_TC_PATH)
    p.add_argument("--labels", type=Path, default=DEFAULT_LABEL_PATH)
    p.add_argument("--icn-domain", type=Path, default=None)
    p.add_argument(
        "--npz",
        type=str,
        default="",
        help="Single .npz with time_courses, y, icn_domain (overrides --tc/--labels)",
    )
    p.add_argument(
        "--all-measures",
        action="store_true",
        help=(
            "Plot all connectivity measures in one figure: rows = pearson_z, spearman, "
            "partial_corr, mutual_info; columns = HC, SZ, SZ−HC. "
            "Mutual information is expensive for large n_subj × n_icns."
        ),
    )
    p.add_argument(
        "--groups-as-rows",
        action="store_true",
        help=(
            "With --all-measures, use a 3×4 layout instead: rows = HC, SZ, group difference; "
            "columns = pearson_z, spearman, partial_corr, mutual_info."
        ),
    )
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (defaults: figures/mean_fnc_pearson_z_hc_sz.png, "
        "figures/mean_fnc_all_measures_hc_sz.pdf with --all-measures, or "
        "figures/mean_fnc_groups_as_rows_hc_sz.pdf with --all-measures --groups-as-rows).",
    )
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory for group-mean connectivity caches: n_hc.npy, n_sz.npy, "
            "mean_hc__<measure>.npy, mean_sz__<measure>.npy, and meta.json. "
            "If these files exist for the requested measures, matrices are loaded "
            "instead of recomputed (use --force-recompute to ignore)."
        ),
    )
    p.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore --cache-dir on-disk matrices and overwrite them after recomputing.",
    )


def _run_mean_fnc_matrices_cmd(args: argparse.Namespace) -> None:
    from fbirn_experiment.figures import (  # noqa: PLC0415
        plot_group_mean_fnc_all_connectivity_measures,
        plot_group_mean_fnc_connectivity,
    )
    from fbirn_experiment.connectivity import group_mean_pearson_fisher_z_matrices  # noqa: PLC0415
    from fbirn_experiment.connectivity_cache import (  # noqa: PLC0415
        save_group_mean_connectivity_cache,
        try_load_group_mean_connectivity_cache,
        validate_cache_matches_data,
    )

    if args.synthetic:
        tc, y, _icn_domain = synthetic_dataset()
    elif args.npz:
        tc, y, _icn_domain = load_npz(args.npz)
    else:
        tc, y, _icn_domain = load_fbirn_tc_and_labels(
            args.tc,
            args.labels,
            args.icn_domain,
        )

    groups_as_rows = bool(args.groups_as_rows)
    if groups_as_rows and not args.all_measures:
        raise SystemExit("--groups-as-rows requires --all-measures.")

    if args.out is not None:
        out = Path(args.out)
    elif args.all_measures and groups_as_rows:
        out = Path("figures/mean_fnc_groups_as_rows_hc_sz.pdf")
    elif args.all_measures:
        out = Path("figures/mean_fnc_all_measures_hc_sz.pdf")
    else:
        out = Path("figures/mean_fnc_pearson_z_hc_sz.pdf")

    cache_path = Path(args.cache_dir) if args.cache_dir is not None else None
    force = bool(args.force_recompute)

    if args.all_measures:
        n_hc, n_sz, n_icns = plot_group_mean_fnc_all_connectivity_measures(
            tc,
            y,
            out,
            icn_domain=_icn_domain,
            dpi=int(args.dpi),
            cache_dir=cache_path,
            force_recompute=force,
            groups_as_rows=groups_as_rows,
        )
        layout = "3×4 (groups × measures)" if groups_as_rows else "4×3 (measures × groups)"
        print(
            f"Wrote {out} (HC n={n_hc}, SZ n={n_sz}, ICNs={n_icns}, all measures, {layout})"
        )
    else:
        loaded = None
        if cache_path is not None and not force:
            loaded = try_load_group_mean_connectivity_cache(cache_path, ["pearson_z"])
        if loaded is not None:
            rows, n_hc, n_sz, _n_icns = loaded
            validate_cache_matches_data(
                tc, y, n_hc=n_hc, n_sz=n_sz, n_icns=_n_icns
            )
            mean_hc, mean_sz = rows[0][1], rows[0][2]
        else:
            mean_hc, mean_sz, n_hc, n_sz = group_mean_pearson_fisher_z_matrices(tc, y)
            if cache_path is not None:
                save_group_mean_connectivity_cache(
                    cache_path,
                    [("pearson_z", mean_hc, mean_sz)],
                    n_hc,
                    n_sz,
                )
        out.parent.mkdir(parents=True, exist_ok=True)
        plot_group_mean_fnc_connectivity(
            mean_hc,
            mean_sz,
            n_hc,
            n_sz,
            out,
            icn_domain=_icn_domain,
            dpi=int(args.dpi),
        )
        print(f"Wrote {out} (HC n={n_hc}, SZ n={n_sz}, ICNs={mean_hc.shape[0]})")


def _add_plot_confounds_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "plot-confounds",
        help="Plot distributions of confound columns (HC vs SZ) from the confounds CSV.",
    )
    p.add_argument(
        "--confounds-csv",
        type=Path,
        default=DEFAULT_CONFOUND_CSV_PATH,
        help=f"Subject-level table (default: {DEFAULT_CONFOUND_CSV_PATH}).",
    )
    p.add_argument("--tc", type=Path, default=DEFAULT_TC_PATH)
    p.add_argument("--labels", type=Path, default=DEFAULT_LABEL_PATH)
    p.add_argument("--icn-domain", type=Path, default=None)
    p.add_argument(
        "--npz",
        type=str,
        default="",
        help="Single .npz with time_courses, y, icn_domain (overrides --tc/--labels).",
    )
    p.add_argument(
        "--confound-cols",
        type=str,
        nargs="+",
        default=list(DEFAULT_CONFOUND_COLS),
        help="Columns to plot (must exist in CSV).",
    )
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("figures/confound_distributions.png"),
        help="Output PNG path.",
    )
    p.add_argument("--dpi", type=int, default=200)


def _run_plot_confounds_cmd(args: argparse.Namespace) -> None:
    from fbirn_experiment.confound_figures import plot_confound_distributions  # noqa: PLC0415
    from fbirn_experiment.confounds import load_confounds  # noqa: PLC0415

    if args.npz:
        _tc, y, _dom = load_npz(args.npz)
    else:
        _tc, y, _dom = load_fbirn_tc_and_labels(
            args.tc,
            args.labels,
            args.icn_domain,
        )

    csv_path = Path(args.confounds_csv)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Confound CSV not found: {csv_path}")

    df = load_confounds(csv_path, tuple(args.confound_cols))
    out = Path(args.out)
    plot_confound_distributions(df, y, out, confound_cols=args.confound_cols, dpi=int(args.dpi))
    print(f"Wrote {out} (n={len(y)}, columns={list(args.confound_cols)})")


def _run_regen_h1_latent_figs_cmd(args: argparse.Namespace) -> None:
    from fbirn_experiment.figures import regenerate_h1_latent_interpretability_figures  # noqa: PLC0415

    run_dir = Path(args.run_dir)
    plot_kw: dict[str, float] = {}
    if args.matrix_col_inches is not None:
        plot_kw["matrix_col_inches"] = float(args.matrix_col_inches)
    if args.matrix_row_inches is not None:
        plot_kw["matrix_row_inches"] = float(args.matrix_row_inches)
    if args.matrix_hspace is not None:
        plot_kw["matrix_hspace"] = float(args.matrix_hspace)
    if args.matrix_wspace is not None:
        plot_kw["matrix_wspace"] = float(args.matrix_wspace)
    regenerate_h1_latent_interpretability_figures(
        run_dir / "artifacts",
        run_dir / "figures",
        dpi=int(args.dpi),
        **plot_kw,
    )


def _add_run_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("run", help="Run the main experiment.")
    _populate_run_args(p)


def _populate_run_args(parser: argparse.ArgumentParser) -> None:
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
    parser.add_argument(
        "--h1-include-fa",
        action="store_true",
        help="Add FA to H1 nested CV and interpretability (default: edges + ICA only).",
    )
    parser.add_argument("--k-min", type=int, default=5, help="Min latent factors (FA BIC / ICA MSE).")
    parser.add_argument("--k-max", type=int, default=50, help="Max latent factors (FA BIC / ICA MSE).")
    parser.add_argument("--k-step", type=int, default=5, help="Step size for component search grid.")
    parser.add_argument(
        "--fa-criterion",
        choices=("bic", "aic"),
        default="bic",
        help="Information criterion for FA component count on each outer-training fold.",
    )
    parser.add_argument("--h2-perm", type=int, default=500)
    parser.add_argument(
        "--h3-no-bic",
        action="store_true",
        help=(
            "Use fixed H3 latent count (--h3-components) instead of automatic k: "
            "BIC/AIC grid for FA, reconstruction-MSE grid for ICA (default H3)."
        ),
    )
    parser.add_argument(
        "--h3-fa",
        action="store_true",
        help="Use factor analysis (BIC k) for H3 instead of default FastICA (reconstruction-MSE k).",
    )
    parser.add_argument(
        "--h3-components",
        type=int,
        default=10,
        help="Fixed number of latent dimensions for H3 when --h3-no-bic is set.",
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


def _run_main_cmd(args: argparse.Namespace) -> None:
    from fbirn_experiment.pipeline import run_experiment  # noqa: PLC0415

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
        h3_decomposition="fa" if getattr(args, "h3_fa", False) else "ica",
        h3_k_min=args.k_min,
        h3_k_max=args.k_max,
        h3_k_step=args.k_step,
        h3_n_components_fixed=getattr(args, "h3_components", 10),
        auc_bootstrap=args.auc_bootstrap,
        auc_perm_y=args.auc_perm_y,
        save_artifacts=not args.no_save,
        make_figures=not args.no_figures,
        h1_include_fa=args.h1_include_fa,
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
