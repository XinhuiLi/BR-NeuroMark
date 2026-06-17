"""Multiverse analysis: enumerate specifications and run them."""

from __future__ import annotations

import itertools
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.decomposition import (
    NMF,
    PCA,
    FactorAnalysis,
    FastICA,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from fbirn_experiment.component_selection import (
    select_n_components_fa,
    select_n_components_ica,
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
    edge_domain_mask,
    fnc_edges,
    triu_indices,
)
from fbirn_experiment.connectivity_cache import get_or_compute_connectivity_edges
from fbirn_experiment.domain_labels import (
    DOMAIN_GRANULARITIES,
    DomainGranularity,
    aggregate_domains,
)
from fbirn_experiment.h1_cv import FactorAnalysisTransform, FastICATransform
from fbirn_experiment.h2_test import h2_domain_label_permutation_test

log = logging.getLogger(__name__)

ConfoundStrategy = Literal["none", "ols", "combat"]
ReductionMethod = Literal["none", "fa", "ica", "pca", "nmf"]
ClassifierChoice = Literal["elasticnet", "logistic_l2", "svm_linear", "rf"]

CONFOUND_STRATEGIES: list[ConfoundStrategy] = ["none", "ols", "combat"]
REDUCTION_METHODS: list[ReductionMethod] = ["none", "fa", "ica", "pca", "nmf"]
CLASSIFIER_CHOICES: list[ClassifierChoice] = [
    "elasticnet",
    "logistic_l2",
    "svm_linear",
    "rf",
]


@dataclass
class MultispecConfig:
    connectivity: ConnectivityMethod
    confound: ConfoundStrategy
    reduction: ReductionMethod
    classifier: ClassifierChoice
    domain_granularity: DomainGranularity
    spec_id: int = 0

    def label(self) -> str:
        return (
            f"{self.connectivity}|{self.confound}|{self.reduction}"
            f"|{self.classifier}|{self.domain_granularity}"
        )


def enumerate_multiverse(
    connectivity: Sequence[ConnectivityMethod] | None = None,
    confound: Sequence[ConfoundStrategy] | None = None,
    reduction: Sequence[ReductionMethod] | None = None,
    classifier: Sequence[ClassifierChoice] | None = None,
    domain_granularity: Sequence[DomainGranularity] | None = None,
) -> list[MultispecConfig]:
    """Generate all valid specification tuples."""
    conn = list(connectivity or CONNECTIVITY_METHODS)
    conf = list(confound or CONFOUND_STRATEGIES)
    red = list(reduction or REDUCTION_METHODS)
    clf = list(classifier or CLASSIFIER_CHOICES)
    dom = list(domain_granularity or DOMAIN_GRANULARITIES)

    specs: list[MultispecConfig] = []
    for sid, (c1, c2, r, cl, d) in enumerate(
        itertools.product(conn, conf, red, clf, dom)
    ):
        specs.append(
            MultispecConfig(
                connectivity=c1,
                confound=c2,
                reduction=r,
                classifier=cl,
                domain_granularity=d,
                spec_id=sid,
            )
        )
    return specs


# ── Internal helpers ─────────────────────────────────────────────────────


def _select_k(
    X_scaled: np.ndarray,
    method: ReductionMethod,
    *,
    k_min: int = 5,
    k_max: int = 50,
    k_step: int = 5,
    random_state: int = 0,
) -> int:
    """Pick number of components using each method's default criterion."""
    if method in ("fa", "pca"):
        k, _ = select_n_components_fa(
            X_scaled,
            k_min=k_min,
            k_max=k_max,
            k_step=k_step,
            criterion="bic",
            random_state=random_state,
            max_iter=800,
        )
        return int(k)
    if method in ("ica", "nmf"):
        k, _ = select_n_components_ica(
            X_scaled,
            k_min=k_min,
            k_max=k_max,
            k_step=k_step,
            random_state=random_state,
            max_iter=800,
        )
        return int(k)
    return 0


class _PCATransform:
    def __init__(self, n_components: int = 10):
        self.n_components = n_components

    def fit(self, X: np.ndarray, y: Any = None):
        k = min(self.n_components, X.shape[1], X.shape[0] - 1)
        self.pca_ = PCA(n_components=max(k, 1), random_state=0)
        self.pca_.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca_.transform(X)

    def fit_transform(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    def get_params(self, deep: bool = True) -> dict:
        return {"n_components": self.n_components}

    def set_params(self, **params: Any):
        for k, v in params.items():
            setattr(self, k, v)
        return self


class _NMFTransform:
    def __init__(self, n_components: int = 10, max_iter: int = 500):
        self.n_components = n_components
        self.max_iter = max_iter

    def fit(self, X: np.ndarray, y: Any = None):
        X_nn = X - X.min(axis=0, keepdims=True)
        k = min(self.n_components, X_nn.shape[1], X_nn.shape[0] - 1)
        self.nmf_ = NMF(
            n_components=max(k, 1),
            max_iter=self.max_iter,
            random_state=0,
        )
        self.nmf_.fit(X_nn)
        self._min = X.min(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_nn = X - self._min[np.newaxis, :]
        X_nn = np.clip(X_nn, 0, None)
        return self.nmf_.transform(X_nn)

    def fit_transform(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    def get_params(self, deep: bool = True) -> dict:
        return {"n_components": self.n_components, "max_iter": self.max_iter}

    def set_params(self, **params: Any):
        for k, v in params.items():
            setattr(self, k, v)
        return self


def _make_pipeline(
    reduction: ReductionMethod,
    classifier: ClassifierChoice,
    n_components: int = 20,
) -> Pipeline:
    """Build a scikit-learn Pipeline from reduction + classifier choices."""
    steps: list[tuple[str, Any]] = [("scaler", StandardScaler())]

    if reduction == "fa":
        steps.append(("latent", FactorAnalysisTransform(n_components=n_components)))
    elif reduction == "ica":
        steps.append(
            (
                "latent",
                FastICATransform(
                    n_components=n_components, random_state=0, max_iter=1000
                ),
            )
        )
    elif reduction == "pca":
        steps.append(("latent", _PCATransform(n_components=n_components)))
    elif reduction == "nmf":
        steps.append(("latent", _NMFTransform(n_components=n_components)))

    if classifier == "elasticnet":
        steps.append((
            "clf",
            SGDClassifier(
                loss="log_loss",
                penalty="elasticnet",
                random_state=0,
                max_iter=4000,
                tol=1e-3,
            ),
        ))
    elif classifier == "logistic_l2":
        steps.append((
            "clf",
            LogisticRegression(
                penalty="l2", solver="lbfgs", max_iter=5000, random_state=0
            ),
        ))
    elif classifier == "svm_linear":
        steps.append((
            "clf",
            CalibratedClassifierCV(
                LinearSVC(max_iter=5000, random_state=0, dual="auto"),
                cv=3,
            ),
        ))
    elif classifier == "rf":
        steps.append((
            "clf",
            RandomForestClassifier(
                n_estimators=500, random_state=0, n_jobs=1
            ),
        ))

    return Pipeline(steps)


def _param_grid(classifier: ClassifierChoice) -> dict[str, list]:
    if classifier == "elasticnet":
        return {
            "clf__alpha": list(np.logspace(-4, -1, 5)),
            "clf__l1_ratio": [0.1, 0.5, 0.9],
        }
    if classifier == "logistic_l2":
        return {"clf__C": list(np.logspace(-1, 2, 5))}
    if classifier == "svm_linear":
        return {
            "clf__estimator__C": [0.01, 0.1, 1.0, 10.0],
        }
    if classifier == "rf":
        return {
            "clf__max_depth": [5, 10, None],
            "clf__max_features": ["sqrt", 0.1],
        }
    return {}


# ── Public API ───────────────────────────────────────────────────────────


@dataclass
class SpecResult:
    spec: MultispecConfig
    h1_auc_edges: float
    h1_auc_latent: float
    h1_delta_auc: float
    h2_delta_d: float
    h2_p: float
    h3_wilcoxon_p: float
    h3_n_factors: int
    h3_between_gt_within: int
    duration_s: float
    error: str = ""


def run_single_spec(
    time_courses: np.ndarray,
    y: np.ndarray,
    icn_domain_14: np.ndarray,
    spec: MultispecConfig,
    *,
    confound_csv: Path | str | None = None,
    confound_cols: Sequence[str] = ("age", "sex", "race", "site", "hm"),
    outer_splits: int = 5,
    inner_splits: int = 3,
    k_min: int = 5,
    k_max: int = 50,
    k_step: int = 5,
    h2_n_perm: int = 500,
    random_state: int = 42,
) -> SpecResult:
    """Execute a single multiverse specification and return summary metrics."""
    t0 = time.time()
    try:
        return _run_spec_inner(
            time_courses,
            y,
            icn_domain_14,
            spec,
            confound_csv=confound_csv,
            confound_cols=confound_cols,
            outer_splits=outer_splits,
            inner_splits=inner_splits,
            k_min=k_min,
            k_max=k_max,
            k_step=k_step,
            h2_n_perm=h2_n_perm,
            random_state=random_state,
            t0=t0,
        )
    except Exception as exc:
        return SpecResult(
            spec=spec,
            h1_auc_edges=np.nan,
            h1_auc_latent=np.nan,
            h1_delta_auc=np.nan,
            h2_delta_d=np.nan,
            h2_p=np.nan,
            h3_wilcoxon_p=np.nan,
            h3_n_factors=0,
            h3_between_gt_within=0,
            duration_s=time.time() - t0,
            error=str(exc),
        )


def _run_spec_inner(
    time_courses: np.ndarray,
    y: np.ndarray,
    icn_domain_14: np.ndarray,
    spec: MultispecConfig,
    *,
    confound_csv: Path | str | None,
    confound_cols: Sequence[str],
    outer_splits: int,
    inner_splits: int,
    k_min: int,
    k_max: int,
    k_step: int,
    h2_n_perm: int,
    random_state: int,
    t0: float,
) -> SpecResult:
    n_icns = time_courses.shape[2]
    ii, jj = triu_indices(n_icns)

    # D1: connectivity
    edges, _ = fnc_edges(time_courses, method=spec.connectivity)

    # D2: confound strategy
    confound_matrix: np.ndarray | None = None
    site_labels_inner: np.ndarray | None = None
    remaining_confound_matrix_inner: np.ndarray | None = None

    if spec.confound == "ols" and confound_csv is not None:
        cdf = load_confounds(confound_csv, confound_cols)
        confound_matrix, _ = build_design_matrix(cdf, confound_cols)
        edges_clean = regress_confounds(edges, confound_matrix)
    elif spec.confound == "combat" and confound_csv is not None:
        from neuroCombat import neuroCombat  # noqa: PLC0415

        cdf = load_confounds(confound_csv, confound_cols)
        site_col = "site" if "site" in cdf.columns else cdf.columns[0]
        site_labels_inner = cdf[site_col].values
        covars = pd.DataFrame({"batch": site_labels_inner})
        result = neuroCombat(dat=edges.T, covars=covars, batch_col="batch")
        edges_combat: np.ndarray = result["data"].T
        remaining = [c for c in confound_cols if c != site_col]
        if remaining:
            cdf_rest = cdf[remaining]
            remaining_confound_matrix_inner, _ = build_design_matrix(cdf_rest, remaining)
            edges_clean = regress_confounds(edges_combat, remaining_confound_matrix_inner)
        else:
            edges_clean = edges_combat
    else:
        edges_clean = edges

    # D5: domain granularity
    icn_domain = aggregate_domains(icn_domain_14, spec.domain_granularity)

    # ── H2 ───────────────────────────────────────────────────────────────
    h2 = h2_domain_label_permutation_test(
        edges_clean, y, icn_domain, ii, jj,
        n_perm=h2_n_perm, random_state=random_state,
    )

    # ── H3 (factor loadings) ────────────────────────────────────────────
    h3_wilcoxon_p = np.nan
    h3_n_factors = 0
    h3_between_gt = 0

    if spec.reduction != "none":
        scaler = StandardScaler()
        Xs = scaler.fit_transform(edges_clean)
        k = _select_k(
            Xs, spec.reduction,
            k_min=k_min, k_max=k_max, k_step=k_step, random_state=random_state,
        )
        h3_n_factors = k

        if spec.reduction == "fa":
            decomp = FactorAnalysis(n_components=k, max_iter=2000, random_state=0)
        elif spec.reduction == "ica":
            decomp = FastICA(
                n_components=k, random_state=0, max_iter=1000,
                whiten="unit-variance",
            )
        elif spec.reduction == "pca":
            decomp = PCA(n_components=k, random_state=0)
        elif spec.reduction == "nmf":
            Xs = Xs - Xs.min(axis=0, keepdims=True)
            decomp = NMF(n_components=k, max_iter=500, random_state=0)
        else:
            decomp = None

        if decomp is not None:
            decomp.fit(Xs)
            loadings = (
                decomp.components_
                if hasattr(decomp, "components_")
                else decomp.mixing_.T
                if hasattr(decomp, "mixing_") and decomp.mixing_ is not None
                else np.zeros((k, Xs.shape[1]))
            )
            within, between = edge_domain_mask(icn_domain, ii, jj)
            bw_list, wn_list = [], []
            for comp in range(loadings.shape[0]):
                ell = np.abs(loadings[comp])
                if between.any():
                    bw_list.append(float(np.mean(ell[between])))
                if within.any():
                    wn_list.append(float(np.mean(ell[within])))
            bw = np.array(bw_list)
            wn = np.array(wn_list)
            h3_between_gt = int(np.sum(bw > wn))
            if len(bw) >= 5:
                d = bw - wn
                if not np.allclose(d, 0):
                    try:
                        _, h3_wilcoxon_p = sp_stats.wilcoxon(d)
                    except Exception:
                        pass

    # ── H1 (classification) ─────────────────────────────────────────────
    outer_cv = StratifiedKFold(
        n_splits=outer_splits, shuffle=True, random_state=random_state,
    )
    inner_cv = StratifiedKFold(
        n_splits=inner_splits, shuffle=True, random_state=random_state + 1,
    )
    aucs_edges: list[float] = []
    aucs_latent: list[float] = []

    for fold_idx, (tr, te) in enumerate(outer_cv.split(edges, y)):
        X_tr, X_te = edges[tr], edges[te]
        y_tr, y_te = y[tr], y[te]

        if spec.confound == "ols" and confound_matrix is not None:
            X_tr, X_te = regress_confounds_cv(
                X_tr, X_te, confound_matrix[tr], confound_matrix[te],
            )
        elif spec.confound == "combat" and site_labels_inner is not None:
            X_tr, X_te = _combat_harmonize_cv(
                X_tr, X_te, site_labels_inner[tr], site_labels_inner[te],
            )
            if remaining_confound_matrix_inner is not None:
                X_tr, X_te = regress_confounds_cv(
                    X_tr, X_te,
                    remaining_confound_matrix_inner[tr],
                    remaining_confound_matrix_inner[te],
                )

        # Edges pipeline
        pipe_e = _make_pipeline("none", spec.classifier)
        grid_e = _param_grid(spec.classifier)
        gs_e = GridSearchCV(
            pipe_e, grid_e, cv=inner_cv, scoring="roc_auc",
            n_jobs=1, refit=True, error_score=0.5,
        )
        gs_e.fit(X_tr, y_tr)
        p_e = _safe_predict_proba(gs_e, X_te)
        aucs_edges.append(roc_auc_score(y_te, p_e))

        # Latent pipeline (skip if reduction == none)
        if spec.reduction != "none":
            scaler_sel = StandardScaler()
            X_tr_s = scaler_sel.fit_transform(X_tr)
            k = _select_k(
                X_tr_s, spec.reduction,
                k_min=k_min, k_max=k_max, k_step=k_step,
                random_state=random_state + fold_idx,
            )
            pipe_l = _make_pipeline(spec.reduction, spec.classifier, n_components=k)
            grid_l = _param_grid(spec.classifier)
            gs_l = GridSearchCV(
                pipe_l, grid_l, cv=inner_cv, scoring="roc_auc",
                n_jobs=1, refit=True, error_score=0.5,
            )
            gs_l.fit(X_tr, y_tr)
            p_l = _safe_predict_proba(gs_l, X_te)
            aucs_latent.append(roc_auc_score(y_te, p_l))

    h1_auc_e = float(np.mean(aucs_edges))
    h1_auc_l = float(np.mean(aucs_latent)) if aucs_latent else np.nan
    h1_delta = h1_auc_l - h1_auc_e if aucs_latent else np.nan

    return SpecResult(
        spec=spec,
        h1_auc_edges=h1_auc_e,
        h1_auc_latent=h1_auc_l,
        h1_delta_auc=h1_delta,
        h2_delta_d=float(h2["observed_delta_mean_abs_d"]),
        h2_p=float(h2["p_value_one_sided"]),
        h3_wilcoxon_p=float(h3_wilcoxon_p),
        h3_n_factors=h3_n_factors,
        h3_between_gt_within=h3_between_gt,
        duration_s=time.time() - t0,
    )


def _safe_predict_proba(model: Any, X: np.ndarray) -> np.ndarray:
    """Get probability estimates; fall back to decision_function."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    df = model.decision_function(X)
    from scipy.special import expit  # noqa: PLC0415
    return expit(df)


def _combat_harmonize_cv(
    X_train: np.ndarray,
    X_test: np.ndarray,
    site_train: np.ndarray,
    site_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """CV-safe ComBat: fit on training fold, apply learned parameters to test.

    Prevents data leakage by estimating batch-effect parameters only from
    training data, then applying the same correction to test samples.
    """
    from neuroCombat import neuroCombat  # noqa: PLC0415

    covars_tr = pd.DataFrame({"batch": site_train})
    result = neuroCombat(dat=X_train.T, covars=covars_tr, batch_col="batch")
    X_tr_out = result["data"].T
    est = result["estimates"]

    stand_mean = np.asarray(est["stand.mean"])
    var_pooled = np.asarray(est["var.pooled"])
    gamma_star = np.asarray(est["gamma.star"])
    delta_star = np.asarray(est["delta.star"])

    if stand_mean.ndim == 1:
        stand_mean = stand_mean[:, np.newaxis]
    if var_pooled.ndim == 1:
        var_pooled = var_pooled[:, np.newaxis]

    unique_batches = list(dict.fromkeys(site_train))
    batch_to_idx = {b: i for i, b in enumerate(unique_batches)}

    X_te_T = X_test.T
    s = (X_te_T - stand_mean) / np.sqrt(var_pooled)

    out = np.empty_like(s)
    for j in range(X_test.shape[0]):
        b_idx = batch_to_idx.get(site_test[j])
        if b_idx is not None:
            out[:, j] = (s[:, j] - gamma_star[b_idx]) / np.sqrt(delta_star[b_idx])
        else:
            out[:, j] = s[:, j]

    X_te_out = (out * np.sqrt(var_pooled) + stand_mean).T
    return X_tr_out, X_te_out


def _result_to_row(spec: MultispecConfig, res: SpecResult) -> dict[str, Any]:
    """Convert a SpecResult into a flat dict for CSV / JSON serialization."""
    return {
        "spec_id": spec.spec_id,
        **{k: v for k, v in asdict(spec).items() if k != "spec_id"},
        "h1_auc_edges": res.h1_auc_edges,
        "h1_auc_latent": res.h1_auc_latent,
        "h1_delta_auc": res.h1_delta_auc,
        "h2_delta_d": res.h2_delta_d,
        "h2_p": res.h2_p,
        "h3_wilcoxon_p": res.h3_wilcoxon_p,
        "h3_n_factors": res.h3_n_factors,
        "h3_between_gt_within": res.h3_between_gt_within,
        "duration_s": res.duration_s,
        "error": res.error,
    }


def _save_spec_json(spec_dir: Path, spec_id: int, row: dict[str, Any]) -> None:
    """Persist a single spec result as JSON."""
    with open(spec_dir / f"{spec_id:04d}.json", "w") as f:
        json.dump(
            {k: (v if not isinstance(v, float) or not np.isnan(v) else None)
             for k, v in row.items()},
            f, indent=2,
        )


# ── Orchestrator ─────────────────────────────────────────────────────────


def run_multiverse(
    time_courses: np.ndarray,
    y: np.ndarray,
    icn_domain_14: np.ndarray,
    specs: list[MultispecConfig],
    *,
    confound_csv: Path | str | None = None,
    confound_cols: Sequence[str] = ("age", "sex", "race", "site", "hm"),
    output_dir: Path | str | None = None,
    outer_splits: int = 5,
    inner_splits: int = 3,
    k_min: int = 5,
    k_max: int = 50,
    k_step: int = 5,
    h2_n_perm: int = 500,
    random_state: int = 42,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Run all specifications and return a summary DataFrame.

    Results are incrementally saved to ``output_dir/multiverse_results.csv``
    and individual JSONs to ``output_dir/specs/``.  Already-completed specs
    (detected via existing JSON files with no error) are skipped on resume.
    """
    out = Path(output_dir) if output_dir else Path("results/multiverse")
    out.mkdir(parents=True, exist_ok=True)
    spec_dir = out / "specs"
    spec_dir.mkdir(exist_ok=True)

    # ── Checkpoint resume ────────────────────────────────────────────────
    results: list[dict[str, Any]] = []
    completed_ids: set[int] = set()
    for spec in specs:
        json_path = spec_dir / f"{spec.spec_id:04d}.json"
        if json_path.exists():
            try:
                with open(json_path) as f:
                    saved = json.load(f)
                if not saved.get("error"):
                    completed_ids.add(spec.spec_id)
                    results.append(saved)
            except (json.JSONDecodeError, KeyError):
                pass

    pending = [s for s in specs if s.spec_id not in completed_ids]
    if completed_ids:
        print(
            f"Checkpoint resume: {len(completed_ids)} specs completed, "
            f"{len(pending)} remaining."
        )

    if not pending:
        print("All specifications already completed.")
        df = pd.DataFrame(results)
        df.to_csv(out / "multiverse_results.csv", index=False)
        return df

    # ── Reusable connectivity edge matrices ─────────────────────────────
    edge_cache: dict[str, tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]] = {}
    edge_cache_dir = out / "connectivity_cache"
    conn_methods = sorted({s.connectivity for s in pending})
    for cm in conn_methods:
        edge_cache[cm], cache_hit = get_or_compute_connectivity_edges(
            edge_cache_dir, cm, time_courses
        )
        action = "Loaded cached" if cache_hit else "Computed and cached"
        print(f"{action} FNC edges: {cm} → {edge_cache_dir}")

    # ── Common spec execution kwargs ────────────────────────────────────
    common_kw: dict[str, Any] = dict(
        confound_csv=confound_csv,
        confound_cols=confound_cols,
        outer_splits=outer_splits,
        inner_splits=inner_splits,
        k_min=k_min,
        k_max=k_max,
        k_step=k_step,
        h2_n_perm=h2_n_perm,
        random_state=random_state,
    )

    def _execute_one(spec: MultispecConfig) -> dict[str, Any]:
        t0 = time.time()
        try:
            edges_pre, (ii, jj) = edge_cache[spec.connectivity]
            res = _run_spec_with_precomputed_edges(
                edges_pre, y, icn_domain_14, spec, ii=ii, jj=jj, **common_kw,
            )
            row = _result_to_row(spec, res)
        except Exception as exc:
            row = {
                "spec_id": spec.spec_id,
                **{k: v for k, v in asdict(spec).items() if k != "spec_id"},
                "h1_auc_edges": np.nan,
                "h1_auc_latent": np.nan,
                "h1_delta_auc": np.nan,
                "h2_delta_d": np.nan,
                "h2_p": np.nan,
                "h3_wilcoxon_p": np.nan,
                "h3_n_factors": 0,
                "h3_between_gt_within": 0,
                "duration_s": time.time() - t0,
                "error": str(exc),
            }
            log.warning("spec_id=%d ERROR: %s", spec.spec_id, exc)
        return row

    # ── Execute specs ────────────────────────────────────────────────────
    total = len(specs)
    if n_jobs == 1 or len(pending) == 1:
        for i, spec in enumerate(pending):
            idx = len(completed_ids) + i + 1
            print(f"[{idx}/{total}] spec_id={spec.spec_id}  {spec.label()}")
            row = _execute_one(spec)
            results.append(row)
            _save_spec_json(spec_dir, spec.spec_id, row)
            pd.DataFrame(results).to_csv(out / "multiverse_results.csv", index=False)
            print(f"  done in {row['duration_s']:.1f}s")
    else:
        from joblib import Parallel, delayed  # noqa: PLC0415

        print(f"Running {len(pending)} specs with n_jobs={n_jobs} ...")
        par_rows = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_execute_one)(spec) for spec in pending
        )
        for spec, row in zip(pending, par_rows):
            results.append(row)
            _save_spec_json(spec_dir, spec.spec_id, row)

    df = pd.DataFrame(results)
    df.to_csv(out / "multiverse_results.csv", index=False)
    print(f"\nMultiverse complete: {len(df)} specifications → {out}")
    return df


def _run_spec_with_precomputed_edges(
    edges: np.ndarray,
    y: np.ndarray,
    icn_domain_14: np.ndarray,
    spec: MultispecConfig,
    *,
    ii: np.ndarray,
    jj: np.ndarray,
    confound_csv: Path | str | None,
    confound_cols: Sequence[str],
    outer_splits: int,
    inner_splits: int,
    k_min: int,
    k_max: int,
    k_step: int,
    h2_n_perm: int,
    random_state: int,
) -> SpecResult:
    """Core logic using pre-computed FNC edges."""
    t0 = time.time()

    # D2: confound strategy
    #   - For H2/H3 (observational): full-sample confound removal is appropriate.
    #   - For H1 (predictive): fold-wise confound removal prevents leakage.
    #   We compute edges_clean (full-sample) for H2/H3, and store what's
    #   needed (confound_matrix, site_labels) for fold-wise H1 processing.
    confound_matrix: np.ndarray | None = None
    site_labels: np.ndarray | None = None
    remaining_confound_matrix: np.ndarray | None = None

    if spec.confound != "none" and confound_csv is not None:
        cdf = load_confounds(confound_csv, confound_cols)
        if spec.confound == "combat":
            from neuroCombat import neuroCombat  # noqa: PLC0415

            site_col = "site" if "site" in cdf.columns else list(cdf.columns)[0]
            site_labels = cdf[site_col].values
            covars = pd.DataFrame({"batch": site_labels})
            result = neuroCombat(dat=edges.T, covars=covars, batch_col="batch")
            edges_base: np.ndarray = result["data"].T
            remaining = [c for c in confound_cols if c != site_col]
            if remaining:
                cdf_rest = cdf[remaining]
                remaining_confound_matrix, _ = build_design_matrix(cdf_rest, remaining)
                edges_clean = regress_confounds(edges_base, remaining_confound_matrix)
            else:
                edges_clean = edges_base
        else:
            confound_matrix, _ = build_design_matrix(cdf, confound_cols)
            edges_clean = regress_confounds(edges, confound_matrix)
    else:
        edges_clean = edges

    # D5: domain granularity
    icn_domain = aggregate_domains(icn_domain_14, spec.domain_granularity)

    # ── H2 (full-sample, no CV leakage concern) ─────────────────────────
    h2 = h2_domain_label_permutation_test(
        edges_clean, y, icn_domain, ii, jj,
        n_perm=h2_n_perm, random_state=random_state,
    )

    # ── H3 (full-sample factor loadings) ─────────────────────────────────
    h3_wilcoxon_p = np.nan
    h3_n_factors = 0
    h3_between_gt = 0

    if spec.reduction != "none":
        scaler = StandardScaler()
        Xs = scaler.fit_transform(edges_clean)
        k = _select_k(
            Xs, spec.reduction,
            k_min=k_min, k_max=k_max, k_step=k_step, random_state=random_state,
        )
        h3_n_factors = k

        if spec.reduction == "nmf":
            Xs = Xs - Xs.min(axis=0, keepdims=True)

        if spec.reduction == "fa":
            decomp_obj = FactorAnalysis(n_components=k, max_iter=2000, random_state=0)
        elif spec.reduction == "ica":
            decomp_obj = FastICA(
                n_components=k, random_state=0, max_iter=1000, whiten="unit-variance",
            )
        elif spec.reduction == "pca":
            decomp_obj = PCA(n_components=k, random_state=0)
        else:
            decomp_obj = NMF(n_components=k, max_iter=500, random_state=0)

        decomp_obj.fit(Xs)
        loadings = getattr(decomp_obj, "components_", None)
        if loadings is None and hasattr(decomp_obj, "mixing_") and decomp_obj.mixing_ is not None:
            loadings = decomp_obj.mixing_.T
        if loadings is None:
            loadings = np.zeros((k, Xs.shape[1]))

        within, between = edge_domain_mask(icn_domain, ii, jj)
        bw_arr = (
            np.array([float(np.mean(np.abs(loadings[c][between])))
                       for c in range(loadings.shape[0])])
            if between.any() else np.array([])
        )
        wn_arr = (
            np.array([float(np.mean(np.abs(loadings[c][within])))
                       for c in range(loadings.shape[0])])
            if within.any() else np.array([])
        )

        if len(bw_arr) > 0 and len(wn_arr) > 0:
            h3_between_gt = int(np.sum(bw_arr > wn_arr))
            d = bw_arr - wn_arr
            if len(d) >= 5 and not np.allclose(d, 0):
                try:
                    _, h3_wilcoxon_p = sp_stats.wilcoxon(d)
                except Exception:
                    pass

    # ── H1 (fold-wise, leakage-free) ─────────────────────────────────────
    outer_cv = StratifiedKFold(
        n_splits=outer_splits, shuffle=True, random_state=random_state,
    )
    inner_cv = StratifiedKFold(
        n_splits=inner_splits, shuffle=True, random_state=random_state + 1,
    )
    aucs_edges_list: list[float] = []
    aucs_latent_list: list[float] = []

    for fold_idx, (tr, te) in enumerate(outer_cv.split(edges, y)):
        X_tr, X_te = edges[tr], edges[te]
        y_tr, y_te = y[tr], y[te]

        if spec.confound == "ols" and confound_matrix is not None:
            X_tr, X_te = regress_confounds_cv(
                X_tr, X_te, confound_matrix[tr], confound_matrix[te],
            )
        elif spec.confound == "combat" and site_labels is not None:
            X_tr, X_te = _combat_harmonize_cv(
                X_tr, X_te, site_labels[tr], site_labels[te],
            )
            if remaining_confound_matrix is not None:
                X_tr, X_te = regress_confounds_cv(
                    X_tr, X_te,
                    remaining_confound_matrix[tr],
                    remaining_confound_matrix[te],
                )

        # Edge-only pipeline
        pipe_e = _make_pipeline("none", spec.classifier)
        grid_e = _param_grid(spec.classifier)
        gs_e = GridSearchCV(
            pipe_e, grid_e, cv=inner_cv, scoring="roc_auc",
            n_jobs=1, refit=True, error_score=0.5,
        )
        gs_e.fit(X_tr, y_tr)
        p_e = _safe_predict_proba(gs_e, X_te)
        aucs_edges_list.append(roc_auc_score(y_te, p_e))

        # Latent pipeline
        if spec.reduction != "none":
            scaler_sel = StandardScaler()
            X_tr_s = scaler_sel.fit_transform(X_tr)
            k_fold = _select_k(
                X_tr_s, spec.reduction,
                k_min=k_min, k_max=k_max, k_step=k_step,
                random_state=random_state + fold_idx,
            )
            pipe_l = _make_pipeline(spec.reduction, spec.classifier, n_components=k_fold)
            grid_l = _param_grid(spec.classifier)
            gs_l = GridSearchCV(
                pipe_l, grid_l, cv=inner_cv, scoring="roc_auc",
                n_jobs=1, refit=True, error_score=0.5,
            )
            gs_l.fit(X_tr, y_tr)
            p_l = _safe_predict_proba(gs_l, X_te)
            aucs_latent_list.append(roc_auc_score(y_te, p_l))

    h1_auc_e = float(np.mean(aucs_edges_list))
    h1_auc_l = float(np.mean(aucs_latent_list)) if aucs_latent_list else np.nan
    h1_delta = h1_auc_l - h1_auc_e if aucs_latent_list else np.nan

    return SpecResult(
        spec=spec,
        h1_auc_edges=h1_auc_e,
        h1_auc_latent=h1_auc_l,
        h1_delta_auc=h1_delta,
        h2_delta_d=float(h2["observed_delta_mean_abs_d"]),
        h2_p=float(h2["p_value_one_sided"]),
        h3_wilcoxon_p=float(h3_wilcoxon_p),
        h3_n_factors=h3_n_factors,
        h3_between_gt_within=h3_between_gt,
        duration_s=time.time() - t0,
    )
