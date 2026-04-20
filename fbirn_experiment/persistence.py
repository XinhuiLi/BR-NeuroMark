"""Save intermediate arrays, tables, and JSON-safe metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (bool, str, int, float)) or obj is None:
        return obj
    return str(obj)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(data), f, indent=2)


def save_h1_outputs(
    out_dir: Path,
    res_edges: Any,
    res_fa: Any,
    res_ica: Any,
    summary: pd.DataFrame,
    auc_tests: dict[str, Any] | None = None,
    *,
    stability_tests: dict[str, Any] | None = None,
    interpretability: dict[str, Any] | None = None,
) -> None:
    out_dir = ensure_dir(out_dir)
    summary.to_csv(out_dir / "h1_nested_cv_summary.csv", index=False)
    save_json(
        out_dir / "h1_best_params_edges.json",
        {"folds": res_edges.best_params_per_fold},
    )
    if res_fa is not None:
        save_json(
            out_dir / "h1_best_params_fa.json",
            {"folds": res_fa.best_params_per_fold},
        )
    save_json(
        out_dir / "h1_best_params_ica.json",
        {"folds": res_ica.best_params_per_fold},
    )
    if res_fa is not None and res_fa.meta_per_fold:
        save_json(
            out_dir / "h1_fa_component_selection_per_fold.json",
            {"folds": res_fa.meta_per_fold},
        )
    if res_ica.meta_per_fold:
        save_json(
            out_dir / "h1_ica_component_selection_per_fold.json",
            {"folds": res_ica.meta_per_fold},
        )
    oof_pack: dict[str, Any] = {
        "y_true": res_edges.y_true_oof,
        "proba_edges": res_edges.proba_oof,
        "proba_ica": res_ica.proba_oof,
        "auc_edges_per_fold": res_edges.outer_aucs,
        "auc_ica_per_fold": res_ica.outer_aucs,
    }
    if res_fa is not None:
        oof_pack["proba_fa"] = res_fa.proba_oof
        oof_pack["auc_fa_per_fold"] = res_fa.outer_aucs
    np.savez_compressed(out_dir / "h1_oof_predictions.npz", **oof_pack)
    if auc_tests is not None:
        save_json(out_dir / "h1_auc_pairwise_tests.json", auc_tests)
    if stability_tests is not None:
        save_json(out_dir / "h1_stability_tests.json", stability_tests)
    if interpretability is not None:
        meta: dict[str, Any] = {
            "hyperparams_edges": interpretability.get("hyperparams_edges"),
            "hyperparams_ica": interpretability.get("hyperparams_ica"),
        }
        if "hyperparams_fa" in interpretability:
            meta["hyperparams_fa"] = interpretability["hyperparams_fa"]
        save_json(out_dir / "h1_interpretability_meta.json", meta)
        coef_pack: dict[str, Any] = {
            "coef_edges": interpretability["coef_edges"],
            "coef_ica_latent": interpretability["coef_ica_latent"],
            "ica_components": interpretability["ica_components"],
        }
        if "coef_fa_latent" in interpretability:
            coef_pack["coef_fa_latent"] = interpretability["coef_fa_latent"]
            coef_pack["fa_components"] = interpretability["fa_components"]
        np.savez_compressed(
            out_dir / "h1_interpretability_coefs.npz",
            **coef_pack,
        )


def save_fnc_bundle(
    out_dir: Path,
    edges: np.ndarray,
    ii: np.ndarray,
    jj: np.ndarray,
    y: np.ndarray,
    icn_domain: np.ndarray,
) -> None:
    out_dir = ensure_dir(out_dir)
    np.savez_compressed(
        out_dir / "fnc_edges_bundle.npz",
        edges=edges,
        triu_i=ii,
        triu_j=jj,
        y=y,
        icn_domain=icn_domain,
    )


def save_h2_outputs(out_dir: Path, h2: dict[str, Any]) -> None:
    out_dir = ensure_dir(out_dir)
    null = h2["null_delta_mean_abs_d"]
    d = h2["edge_cohens_d"]
    meta = {k: v for k, v in h2.items() if k not in ("null_delta_mean_abs_d", "edge_cohens_d", "mask_between", "mask_within")}
    meta["n_null"] = len(null)
    save_json(out_dir / "h2_permutation_summary.json", meta)
    np.savez_compressed(
        out_dir / "h2_arrays.npz",
        null_delta_mean_abs_d=null,
        edge_cohens_d=d,
        mask_between=h2["mask_between"],
        mask_within=h2["mask_within"],
    )


def save_h3_outputs(
    out_dir: Path,
    h3_summary: pd.DataFrame,
    loadings: np.ndarray,
    selection: dict[str, Any] | None = None,
    domain_pair_summary: pd.DataFrame | None = None,
) -> None:
    out_dir = ensure_dir(out_dir)
    h3_summary.to_csv(out_dir / "h3_factor_loading_summary.csv", index=False)
    np.save(out_dir / "h3_factor_loadings.npy", loadings)
    if selection is not None:
        save_json(out_dir / "h3_component_selection.json", selection)
    if domain_pair_summary is not None and not domain_pair_summary.empty:
        domain_pair_summary.to_csv(
            out_dir / "h3_domain_pair_loading_summary.csv", index=False
        )
