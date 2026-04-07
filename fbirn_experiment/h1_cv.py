"""H1: nested CV comparing edge-based vs FA vs ICA + logistic classifiers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import FactorAnalysis, FastICA
from sklearn.linear_model import LogisticRegression, SGDClassifier
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
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                SGDClassifier(
                    loss="log_loss",
                    penalty="elasticnet",
                    random_state=0,
                    max_iter=4000,
                    tol=1e-3,
                    fit_intercept=True,
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
    res_fa: NestedCVResult,
    res_ica: NestedCVResult,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": res_edges.name,
                "auc_mean": res_edges.outer_aucs.mean(),
                "auc_std": res_edges.outer_aucs.std(),
            },
            {
                "model": res_fa.name,
                "auc_mean": res_fa.outer_aucs.mean(),
                "auc_std": res_fa.outer_aucs.std(),
            },
            {
                "model": res_ica.name,
                "auc_mean": res_ica.outer_aucs.mean(),
                "auc_std": res_ica.outer_aucs.std(),
            },
        ]
    )


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
) -> tuple[NestedCVResult, NestedCVResult, NestedCVResult]:
    """
    Outer CV: on each training fold, select FA k by BIC/AIC and ICA k by
    reconstruction MSE on scaled training features; inner CV tunes only
    classifier regularization (and edge elastic-net hyperparameters).

    If *confound_matrix* is provided (n_subjects × n_regressors, with intercept),
    confound regression is fit on each outer training fold and applied to both
    train and test splits to avoid information leakage.
    """
    y = np.asarray(y).astype(int)
    n = len(y)
    outer_cv = StratifiedKFold(
        n_splits=outer_splits, shuffle=True, random_state=random_state
    )
    inner_cv = StratifiedKFold(
        n_splits=inner_splits, shuffle=True, random_state=random_state + 1
    )

    if full_hyperparameter_search:
        alphas = np.logspace(-5, 0, 10)
        l1s = (0.15, 0.5, 0.85)
        Cs = np.logspace(-2, 2, 8)
    else:
        alphas = np.logspace(-4, -1, 5)
        l1s = (0.5,)
        Cs = np.logspace(-1, 2, 5)

    aucs_edges: list[float] = []
    aucs_fa: list[float] = []
    aucs_ica: list[float] = []
    params_edges: list[dict[str, Any]] = []
    params_fa: list[dict[str, Any]] = []
    params_ica: list[dict[str, Any]] = []
    meta_fa: list[dict[str, Any]] = []
    meta_ica: list[dict[str, Any]] = []
    oof_e = np.zeros(n, dtype=np.float64)
    oof_fa = np.zeros(n, dtype=np.float64)
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

        k_fa, diag_fa = select_n_components_fa(
            X_tr_s,
            k_min=k_min,
            k_max=k_max,
            k_step=k_step,
            criterion=fa_criterion,
            random_state=random_state + fold_idx,
            max_iter=fa_bic_max_iter,
        )
        k_ica, diag_ica = select_n_components_ica(
            X_tr_s,
            k_min=k_min,
            k_max=k_max,
            k_step=k_step,
            random_state=random_state + fold_idx,
            max_iter=ica_select_max_iter,
        )

        pipe_e = make_edge_pipeline()
        grid_e = {"clf__alpha": alphas, "clf__l1_ratio": l1s}
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

    return (
        NestedCVResult(
            "edges_elasticnet",
            np.array(aucs_edges),
            params_edges,
            y_true_oof=y_oof.copy(),
            proba_oof=oof_e,
        ),
        NestedCVResult(
            "fa_logistic",
            np.array(aucs_fa),
            params_fa,
            y_true_oof=y_oof.copy(),
            proba_oof=oof_fa,
            meta_per_fold=meta_fa,
        ),
        NestedCVResult(
            "ica_logistic",
            np.array(aucs_ica),
            params_ica,
            y_true_oof=y_oof.copy(),
            proba_oof=oof_ica,
            meta_per_fold=meta_ica,
        ),
    )


def bootstrap_oof_auc_difference(
    y_true: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    n_boot: int = 10000,
    random_state: int = 0,
) -> tuple[float, float, tuple[float, float]]:
    """Bootstrap subjects on OOF predictions; delta = AUC(a) - AUC(b)."""
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true).astype(int)
    n = len(y_true)

    def delta(idx: np.ndarray) -> float:
        return roc_auc_score(y_true[idx], p_a[idx]) - roc_auc_score(
            y_true[idx], p_b[idx]
        )

    obs = delta(np.arange(n))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
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
    }


def all_pairwise_auc_tests(
    res_e: NestedCVResult,
    res_fa: NestedCVResult,
    res_ica: NestedCVResult,
    *,
    n_bootstrap: int = 10000,
    n_perm_y: int = 5000,
    random_state: int = 0,
) -> dict[str, Any]:
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
