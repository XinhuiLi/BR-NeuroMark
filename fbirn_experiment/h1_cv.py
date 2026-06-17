"""H1: nested CV comparing full FNC edges vs optional FA / ICA + classifiers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import FactorAnalysis, FastICA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fbirn_experiment.component_selection import (
    select_n_components_fa,
    select_n_components_ica,
)
from fbirn_experiment.confounds import regress_confounds_cv


class FactorAnalysisTransform(BaseEstimator, TransformerMixin):
    def __init__(self, n_components: int = 10, max_iter: int = 2000):
        self.n_components = n_components
        self.max_iter = max_iter

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        n_features = X.shape[1]
        k = min(self.n_components, n_features - 1, X.shape[0] - 1)
        k = max(k, 1)
        self.fa_ = FactorAnalysis(
            n_components=k, max_iter=self.max_iter, random_state=0
        )
        self.fa_.fit(X)
        self.n_components_fitted_ = k
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.fa_.transform(X)


class FastICATransform(BaseEstimator, TransformerMixin):
    """FastICA sources as features for downstream classifier."""

    def __init__(
        self,
        n_components: int = 10,
        random_state: int = 0,
        max_iter: int = 1000,
    ):
        self.n_components = n_components
        self.random_state = random_state
        self.max_iter = max_iter

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        n_features = X.shape[1]
        k = min(self.n_components, n_features, X.shape[0] - 1)
        k = max(k, 1)
        self.ica_ = FastICA(
            n_components=k,
            random_state=self.random_state,
            max_iter=self.max_iter,
            whiten="unit-variance",
            tol=1e-4,
        )
        self.ica_.fit(X)
        self.n_components_fitted_ = k
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.ica_.transform(X)


def make_edge_pipeline() -> Pipeline:
    """All edge features → standardized logistic regression (L2, same family as latent path)."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=5000,
                    random_state=0,
                ),
            ),
        ]
    )


def make_latent_pipeline(method: Literal["fa", "ica"], n_components: int) -> Pipeline:
    if method == "fa":
        decomp: Any = FactorAnalysisTransform(
            n_components=n_components, max_iter=2000
        )
    else:
        decomp = FastICATransform(
            n_components=n_components, random_state=0, max_iter=1000
        )
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("latent", decomp),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=5000,
                    random_state=0,
                ),
            ),
        ]
    )


@dataclass
class NestedCVResult:
    name: str
    outer_aucs: np.ndarray
    best_params_per_fold: list[dict[str, Any]]
    y_true_oof: np.ndarray = field(default_factory=lambda: np.array([]))
    proba_oof: np.ndarray = field(default_factory=lambda: np.array([]))
    meta_per_fold: list[dict[str, Any]] = field(default_factory=list)


def summarize_h1(
    res_edges: NestedCVResult,
    res_fa: NestedCVResult | None,
    res_ica: NestedCVResult,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "model": res_edges.name,
            "auc_mean": res_edges.outer_aucs.mean(),
            "auc_std": res_edges.outer_aucs.std(),
        },
    ]
    if res_fa is not None:
        rows.append(
            {
                "model": res_fa.name,
                "auc_mean": res_fa.outer_aucs.mean(),
                "auc_std": res_fa.outer_aucs.std(),
            },
        )
    rows.append(
        {
            "model": res_ica.name,
            "auc_mean": res_ica.outer_aucs.mean(),
            "auc_std": res_ica.outer_aucs.std(),
        },
    )
    return pd.DataFrame(rows)


def nested_cv_classifiers(
    X: np.ndarray,
    y: np.ndarray,
    outer_splits: int = 5,
    inner_splits: int = 3,
    random_state: int = 42,
    n_jobs: int = 1,
    full_hyperparameter_search: bool = False,
    *,
    k_min: int = 5,
    k_max: int = 50,
    k_step: int = 5,
    fa_criterion: Literal["bic", "aic"] = "bic",
    fa_bic_max_iter: int = 800,
    ica_select_max_iter: int = 800,
    confound_matrix: np.ndarray | None = None,
    include_fa: bool = False,
) -> tuple[NestedCVResult, NestedCVResult | None, NestedCVResult]:
    """
    Outer CV: on each training fold, select ICA k by reconstruction MSE on scaled
    training features; optionally the same for FA (*include_fa*). Inner CV tunes
    ``clf__C`` for edges (L2 logistic), latent logistic, and (if enabled) FA.
    The latent dimensionality search is performed on the outer-training fold,
    not separately inside each inner-CV split.

    If *confound_matrix* is provided (n_subjects × n_regressors, with intercept),
    confound regression is fit on each outer training fold and applied to both
    train and test splits to avoid information leakage.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (subjects × edges/features); got shape {X.shape}.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X rows ({X.shape[0]}) must match y length ({y.shape[0]}).")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values.")
    classes, counts = np.unique(y, return_counts=True)
    if not np.array_equal(classes, np.array([0, 1])):
        raise ValueError(f"y must be binary with labels {{0, 1}}; got {classes.tolist()}.")
    if confound_matrix is not None:
        confound_matrix = np.asarray(confound_matrix, dtype=np.float64)
        if confound_matrix.ndim != 2 or confound_matrix.shape[0] != X.shape[0]:
            raise ValueError(
                "confound_matrix must be 2D with one row per subject; "
                f"got {confound_matrix.shape} for X shape {X.shape}."
            )
        if not np.all(np.isfinite(confound_matrix)):
            raise ValueError("confound_matrix contains non-finite values.")
    min_class = int(counts.min())
    if outer_splits < 2 or inner_splits < 2:
        raise ValueError("outer_splits and inner_splits must both be at least 2.")
    if outer_splits > min_class or inner_splits > min_class:
        raise ValueError(
            "outer_splits and inner_splits must not exceed the smallest class size "
            f"({min_class})."
        )
    n = len(y)
    outer_cv = StratifiedKFold(
        n_splits=outer_splits, shuffle=True, random_state=random_state
    )
    inner_cv = StratifiedKFold(
        n_splits=inner_splits, shuffle=True, random_state=random_state + 1
    )

    if full_hyperparameter_search:
        Cs = np.logspace(-2, 2, 8)
    else:
        Cs = np.logspace(-1, 2, 5)

    aucs_edges: list[float] = []
    aucs_fa: list[float] = [] if include_fa else []
    aucs_ica: list[float] = []
    params_edges: list[dict[str, Any]] = []
    params_fa: list[dict[str, Any]] = [] if include_fa else []
    params_ica: list[dict[str, Any]] = []
    meta_fa: list[dict[str, Any]] = [] if include_fa else []
    meta_ica: list[dict[str, Any]] = []
    oof_e = np.zeros(n, dtype=np.float64)
    oof_fa = np.zeros(n, dtype=np.float64) if include_fa else np.array([])
    oof_ica = np.zeros(n, dtype=np.float64)
    y_oof = np.zeros(n, dtype=np.int64)

    for fold_idx, (tr, te) in enumerate(outer_cv.split(X, y)):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        y_oof[te] = y_te

        if confound_matrix is not None:
            X_tr, X_te = regress_confounds_cv(
                X_tr, X_te, confound_matrix[tr], confound_matrix[te]
            )

        scaler_sel = StandardScaler()
        X_tr_s = scaler_sel.fit_transform(X_tr)

        if include_fa:
            k_fa, diag_fa = select_n_components_fa(
                X_tr_s,
                k_min=k_min,
                k_max=k_max,
                k_step=k_step,
                criterion=fa_criterion,
                random_state=random_state + fold_idx,
                max_iter=fa_bic_max_iter,
            )
            diag_fa = {**diag_fa, "k_selection_scope": "outer_training_fold"}
        k_ica, diag_ica = select_n_components_ica(
            X_tr_s,
            k_min=k_min,
            k_max=k_max,
            k_step=k_step,
            random_state=random_state + fold_idx,
            max_iter=ica_select_max_iter,
        )
        diag_ica = {**diag_ica, "k_selection_scope": "outer_training_fold"}

        pipe_e = make_edge_pipeline()
        grid_e = {"clf__C": Cs}
        gs_e = GridSearchCV(
            pipe_e,
            grid_e,
            cv=inner_cv,
            scoring="roc_auc",
            n_jobs=n_jobs,
            refit=True,
        )
        gs_e.fit(X_tr, y_tr)
        p_e = gs_e.predict_proba(X_te)[:, 1]
        oof_e[te] = p_e
        aucs_edges.append(roc_auc_score(y_te, p_e))
        params_edges.append(gs_e.best_params_)

        if include_fa:
            pipe_fa = make_latent_pipeline("fa", k_fa)
            grid_fa = {"clf__C": Cs}
            gs_fa = GridSearchCV(
                pipe_fa,
                grid_fa,
                cv=inner_cv,
                scoring="roc_auc",
                n_jobs=n_jobs,
                refit=True,
            )
            gs_fa.fit(X_tr, y_tr)
            p_fa = gs_fa.predict_proba(X_te)[:, 1]
            oof_fa[te] = p_fa
            aucs_fa.append(roc_auc_score(y_te, p_fa))
            bp = gs_fa.best_params_
            params_fa.append({**bp, "fa_n_components_selected": k_fa})
            meta_fa.append({**diag_fa, "k_selected": int(k_fa)})

        pipe_ica = make_latent_pipeline("ica", k_ica)
        grid_ica = {"clf__C": Cs}
        gs_ica = GridSearchCV(
            pipe_ica,
            grid_ica,
            cv=inner_cv,
            scoring="roc_auc",
            n_jobs=n_jobs,
            refit=True,
        )
        gs_ica.fit(X_tr, y_tr)
        p_ica = gs_ica.predict_proba(X_te)[:, 1]
        oof_ica[te] = p_ica
        aucs_ica.append(roc_auc_score(y_te, p_ica))
        params_ica.append({**gs_ica.best_params_, "ica_n_components_selected": k_ica})
        meta_ica.append({**diag_ica, "k_selected": int(k_ica)})

    res_edges = NestedCVResult(
        "edges_logistic",
        np.array(aucs_edges),
        params_edges,
        y_true_oof=y_oof.copy(),
        proba_oof=oof_e,
    )
    res_fa: NestedCVResult | None
    if include_fa:
        res_fa = NestedCVResult(
            "fa_logistic",
            np.array(aucs_fa),
            params_fa,
            y_true_oof=y_oof.copy(),
            proba_oof=oof_fa,
            meta_per_fold=meta_fa,
        )
    else:
        res_fa = None
    res_ica = NestedCVResult(
        "ica_logistic",
        np.array(aucs_ica),
        params_ica,
        y_true_oof=y_oof.copy(),
        proba_oof=oof_ica,
        meta_per_fold=meta_ica,
    )
    return res_edges, res_fa, res_ica


def bootstrap_oof_auc_difference(
    y_true: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    n_boot: int = 10000,
    random_state: int = 0,
) -> tuple[float, float, tuple[float, float]]:
    """Stratified bootstrap subjects on OOF predictions; delta = AUC(a) - AUC(b)."""
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true).astype(int)
    p_a = np.asarray(p_a, dtype=np.float64)
    p_b = np.asarray(p_b, dtype=np.float64)
    if y_true.shape[0] != p_a.shape[0] or y_true.shape[0] != p_b.shape[0]:
        raise ValueError("y_true, p_a, and p_b must have the same length.")
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1.")
    classes = np.unique(y_true)
    if not np.array_equal(classes, np.array([0, 1])):
        raise ValueError(f"y_true must contain binary labels {{0, 1}}; got {classes.tolist()}.")
    idx0 = np.where(y_true == 0)[0]
    idx1 = np.where(y_true == 1)[0]
    if len(idx0) < 1 or len(idx1) < 1:
        raise ValueError("AUC bootstrap requires at least one sample from each class.")
    if not np.all(np.isfinite(p_a)) or not np.all(np.isfinite(p_b)):
        raise ValueError("Predicted scores contain non-finite values.")
    n = len(y_true)

    def delta(idx: np.ndarray) -> float:
        return roc_auc_score(y_true[idx], p_a[idx]) - roc_auc_score(
            y_true[idx], p_b[idx]
        )

    obs = delta(np.arange(n))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate(
            [
                rng.choice(idx0, size=len(idx0), replace=True),
                rng.choice(idx1, size=len(idx1), replace=True),
            ]
        )
        boots[b] = delta(idx)
    p_two = 2 * min(float(np.mean(boots >= obs)), float(np.mean(boots <= obs)))
    p_two = min(p_two, 1.0)
    ci = tuple(np.percentile(boots, [2.5, 97.5]).astype(float))
    return float(obs), float(p_two), ci


def permutation_y_auc_difference(
    y_true: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    n_perm: int = 5000,
    random_state: int = 0,
) -> tuple[float, float]:
    """
    Permute class labels jointly for both scores; null for AUC(a)-AUC(b) under
    label-shuffle (both models tested against random outcomes).
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true).astype(int)
    p_a = np.asarray(p_a, dtype=np.float64)
    p_b = np.asarray(p_b, dtype=np.float64)
    if y_true.shape[0] != p_a.shape[0] or y_true.shape[0] != p_b.shape[0]:
        raise ValueError("y_true, p_a, and p_b must have the same length.")
    if n_perm < 1:
        raise ValueError("n_perm must be at least 1.")
    classes = np.unique(y_true)
    if not np.array_equal(classes, np.array([0, 1])):
        raise ValueError(f"y_true must contain binary labels {{0, 1}}; got {classes.tolist()}.")
    if not np.all(np.isfinite(p_a)) or not np.all(np.isfinite(p_b)):
        raise ValueError("Predicted scores contain non-finite values.")
    obs = roc_auc_score(y_true, p_a) - roc_auc_score(y_true, p_b)
    null = np.empty(n_perm)
    for i in range(n_perm):
        yp = rng.permutation(y_true)
        null[i] = roc_auc_score(yp, p_a) - roc_auc_score(yp, p_b)
    p_two = (1.0 + np.sum(np.abs(null) >= abs(obs))) / (n_perm + 1.0)
    return float(obs), float(p_two)


def wilcoxon_fold_aucs(
    aucs_a: np.ndarray,
    aucs_b: np.ndarray,
) -> tuple[float, float]:
    """Paired Wilcoxon on outer-fold AUCs (same folds)."""
    d = np.asarray(aucs_a, dtype=np.float64) - np.asarray(aucs_b, dtype=np.float64)
    if np.allclose(d, 0):
        return 0.0, 1.0
    try:
        out = stats.wilcoxon(d, alternative="two-sided", method="auto")
    except TypeError:
        out = stats.wilcoxon(d)
    if hasattr(out, "statistic"):
        return float(out.statistic), float(out.pvalue)
    return float(out[0]), float(out[1])


def pairwise_auc_tests(
    res_a: NestedCVResult,
    res_b: NestedCVResult,
    *,
    n_bootstrap: int = 10000,
    n_perm_y: int = 5000,
    random_state: int = 0,
) -> dict[str, Any]:
    """Compare two models: OOF bootstrap, label permutation, fold Wilcoxon."""
    y = res_a.y_true_oof
    obs_boot, p_boot, ci = bootstrap_oof_auc_difference(
        y,
        res_a.proba_oof,
        res_b.proba_oof,
        n_boot=n_bootstrap,
        random_state=random_state,
    )
    obs_perm, p_perm = permutation_y_auc_difference(
        y,
        res_a.proba_oof,
        res_b.proba_oof,
        n_perm=n_perm_y,
        random_state=random_state + 1,
    )
    w_stat, w_p = wilcoxon_fold_aucs(res_a.outer_aucs, res_b.outer_aucs)
    return {
        "comparison": f"{res_a.name}_vs_{res_b.name}",
        "oof_auc_delta_a_minus_b": obs_boot,
        "bootstrap_p_two_sided": p_boot,
        "bootstrap_ci95_delta": list(ci),
        "permutation_y_p_two_sided": p_perm,
        "permutation_y_delta_at_obs_labels": obs_perm,
        "wilcoxon_fold_statistic": w_stat,
        "wilcoxon_fold_p_two_sided": w_p,
        "n_bootstrap": n_bootstrap,
        "n_perm_y": n_perm_y,
        "model_comparison_note": (
            "OOF bootstrap is stratified by class; fold-wise Wilcoxon uses the "
            "small outer-fold sample and should be interpreted descriptively."
        ),
    }


def compute_h1_stability_tests(
    res_edges: NestedCVResult,
    res_fa: NestedCVResult | None,
    res_ica: NestedCVResult,
) -> dict[str, Any]:
    """Compare **stability** of nested-CV AUC across outer folds (lower spread ⇒ more stable).

    Uses Levene and Fligner on per-fold AUC vectors (same folds across models compared).
    When *res_fa* is omitted, only edges vs ICA are tested (two samples).
    """
    ae = np.asarray(res_edges.outer_aucs, dtype=np.float64)
    ai = np.asarray(res_ica.outer_aucs, dtype=np.float64)

    def _row(name: str, a: np.ndarray) -> dict[str, Any]:
        m = float(np.mean(a))
        s = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
        return {
            "model": name,
            "auc_mean": m,
            "auc_std": s,
            "auc_cv": float(s / m) if m > 1e-8 else None,
            "auc_iqr": float(np.subtract(*np.percentile(a, [75, 25]))),
            "auc_range": float(np.max(a) - np.min(a)),
        }

    rows = [_row(res_edges.name, ae)]
    auc_groups: list[np.ndarray] = [ae]
    if res_fa is not None:
        af = np.asarray(res_fa.outer_aucs, dtype=np.float64)
        rows.append(_row(res_fa.name, af))
        auc_groups.append(af)
    rows.append(_row(res_ica.name, ai))
    auc_groups.append(ai)

    le = stats.levene(*auc_groups, center="mean")
    fl = stats.fligner(*auc_groups, center="median")
    return {
        "per_model": rows,
        "levene_statistic": float(le.statistic),
        "levene_pvalue": float(le.pvalue),
        "fligner_statistic": float(fl.statistic),
        "fligner_pvalue": float(fl.pvalue),
        "n_outer_folds": int(len(ae)),
        "note": (
            "Levene/Fligner: H0 equal spread of outer-fold AUCs across models. "
            "Low p suggests different fold-to-fold stability (variance of AUC). "
            "With few outer folds, treat these as descriptive sensitivity checks."
        ),
    }


def _median_hyperparams_edges(params: list[dict[str, Any]]) -> dict[str, float]:
    """Median CV ``clf__C`` for edge logistic; supports legacy elastic-net keys if present."""
    if params and "clf__C" in params[0]:
        return {"clf__C": float(np.median([float(p["clf__C"]) for p in params]))}
    alphas = [float(p["clf__alpha"]) for p in params]
    l1s = [float(p["clf__l1_ratio"]) for p in params]
    return {
        "clf__alpha": float(np.median(alphas)),
        "clf__l1_ratio": float(np.median(l1s)),
    }


def _median_hyperparams_latent(params: list[dict[str, Any]]) -> tuple[int, float]:
    ks: list[int] = []
    cs: list[float] = []
    for p in params:
        k = int(p.get("fa_n_components_selected", p.get("ica_n_components_selected", 0)))
        ks.append(k)
        cs.append(float(p["clf__C"]))
    return int(round(float(np.median(ks)))), float(np.median(cs))


def fit_h1_interpretability_refits(
    X: np.ndarray,
    y: np.ndarray,
    res_edges: NestedCVResult,
    res_fa: NestedCVResult | None,
    res_ica: NestedCVResult,
) -> dict[str, Any]:
    """Refit pipelines on **all** subjects using median CV-selected hyperparameters.

    Exploratory visualization only: optimistic vs nested-CV AUC; use for **relative**
    coefficient structure (edges vs latent factors), not for inference.
    """
    y = np.asarray(y).astype(int)
    X = np.asarray(X, dtype=np.float64)

    hp_e = _median_hyperparams_edges(res_edges.best_params_per_fold)
    pipe_e = make_edge_pipeline()
    pipe_e.set_params(**hp_e)
    pipe_e.fit(X, y)
    clf_e = pipe_e.named_steps["clf"]
    coef_edges = np.asarray(clf_e.coef_, dtype=np.float64).ravel()

    out: dict[str, Any] = {
        "hyperparams_edges": hp_e,
        "coef_edges": coef_edges,
    }

    if res_fa is not None:
        k_fa, c_fa = _median_hyperparams_latent(res_fa.best_params_per_fold)
        pipe_fa = make_latent_pipeline("fa", k_fa)
        pipe_fa.set_params(clf__C=c_fa)
        pipe_fa.fit(X, y)
        coef_fa = np.asarray(
            pipe_fa.named_steps["clf"].coef_, dtype=np.float64
        ).ravel()
        fa_comp = np.asarray(
            pipe_fa.named_steps["latent"].fa_.components_, dtype=np.float64
        )
        out["hyperparams_fa"] = {"n_components": k_fa, "clf__C": c_fa}
        out["coef_fa_latent"] = coef_fa
        out["fa_components"] = fa_comp

    k_ica, c_ica = _median_hyperparams_latent(res_ica.best_params_per_fold)
    pipe_ica = make_latent_pipeline("ica", k_ica)
    pipe_ica.set_params(clf__C=c_ica)
    pipe_ica.fit(X, y)
    coef_ica = np.asarray(
        pipe_ica.named_steps["clf"].coef_, dtype=np.float64
    ).ravel()
    ica_comp = np.asarray(
        pipe_ica.named_steps["latent"].ica_.components_, dtype=np.float64
    )

    out["hyperparams_ica"] = {"n_components": k_ica, "clf__C": c_ica}
    out["coef_ica_latent"] = coef_ica
    out["ica_components"] = ica_comp
    return out


def all_pairwise_auc_tests(
    res_e: NestedCVResult,
    res_fa: NestedCVResult | None,
    res_ica: NestedCVResult,
    *,
    n_bootstrap: int = 10000,
    n_perm_y: int = 5000,
    random_state: int = 0,
) -> dict[str, Any]:
    pairs: list[tuple[NestedCVResult, NestedCVResult, str]] = [
        (res_ica, res_e, "ica_vs_edges"),
    ]
    if res_fa is not None:
        pairs = [
            (res_fa, res_e, "fa_vs_edges"),
            (res_ica, res_e, "ica_vs_edges"),
            (res_fa, res_ica, "fa_vs_ica"),
        ]
    out: dict[str, Any] = {}
    for i, (ra, rb, key) in enumerate(pairs):
        out[key] = pairwise_auc_tests(
            ra,
            rb,
            n_bootstrap=n_bootstrap,
            n_perm_y=n_perm_y,
            random_state=random_state + i * 7919,
        )
    return out
