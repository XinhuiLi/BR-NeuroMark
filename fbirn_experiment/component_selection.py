"""Select latent dimensionality: BIC/AIC for FA, reconstruction MSE for FastICA."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.decomposition import FactorAnalysis, FastICA


def _n_params_fa(n_features: int, n_components: int) -> int:
    """Loadings (k x p) + diagonal uniqueness (p)."""
    return int(n_components * n_features + n_features)


def fa_aic_bic(
    X: np.ndarray,
    n_components: int,
    *,
    random_state: int = 0,
    max_iter: int = 1000,
) -> tuple[float, float, float]:
    """
    Fit FactorAnalysis on X (already scaled). Returns (mean_loglik, AIC, BIC).
    sklearn score(X) is average log-likelihood per sample; total ll ≈ mean * n.
    """
    n, p = X.shape
    k = int(min(max(1, n_components), p - 1, n - 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        fa = FactorAnalysis(
            n_components=k, max_iter=max_iter, random_state=random_state
        )
        fa.fit(X)
    mean_ll = float(fa.score(X))
    ll = mean_ll * n
    npar = _n_params_fa(p, k)
    aic = -2.0 * ll + 2.0 * npar
    bic = -2.0 * ll + np.log(n) * npar
    return mean_ll, float(aic), float(bic)


def select_n_components_fa(
    X: np.ndarray,
    k_min: int = 5,
    k_max: int = 50,
    k_step: int = 5,
    *,
    criterion: str = "bic",
    random_state: int = 0,
    max_iter: int = 1000,
) -> tuple[int, dict[str, float]]:
    """
    Grid search k in range(k_min, k_max+1, k_step) (clamped to sample/feature
    limits).  Returns (best_k, diagnostics with best score and criterion value).
    """
    n, p = X.shape
    k_hi = min(k_max, p - 1, max(1, n - 2))
    k_lo = max(1, min(k_min, k_hi))
    if k_lo > k_hi:
        return 1, {"criterion": criterion, "best_value": float("nan"), "k_hi": float(k_hi)}

    candidates = list(range(k_lo, k_hi + 1, max(1, k_step)))
    if candidates[-1] != k_hi and k_hi not in candidates:
        candidates.append(k_hi)

    best_k = candidates[0]
    best_val = float("inf")
    criterion = criterion.lower()
    if criterion not in {"bic", "aic"}:
        raise ValueError("criterion must be 'bic' or 'aic'")

    for k in candidates:
        try:
            _, aic, bic = fa_aic_bic(
                X, k, random_state=random_state, max_iter=max_iter
            )
        except Exception:
            continue
        val = bic if criterion == "bic" else aic
        if val < best_val:
            best_val = val
            best_k = k

    return best_k, {
        "criterion": criterion,
        "best_value": best_val,
        "k_min_requested": float(k_min),
        "k_max_requested": float(k_max),
        "k_step": k_step,
        "k_candidates": candidates,
        "k_range_used": f"{k_lo}-{k_hi}",
    }


def ica_reconstruction_mse(
    X: np.ndarray,
    n_components: int,
    *,
    random_state: int = 0,
    max_iter: int = 1000,
) -> float:
    """Mean squared error of FastICA reconstruction in the input (scaled) space."""
    n, p = X.shape
    k = int(min(max(1, n_components), p, n))
    ica = FastICA(
        n_components=k,
        random_state=random_state,
        max_iter=max_iter,
        whiten="unit-variance",
        tol=1e-4,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        S = ica.fit_transform(X)
    X_hat = ica.inverse_transform(S)
    return float(np.mean((X - X_hat) ** 2))


def select_n_components_ica(
    X: np.ndarray,
    k_min: int = 5,
    k_max: int = 50,
    k_step: int = 5,
    *,
    random_state: int = 0,
    max_iter: int = 1000,
) -> tuple[int, dict[str, float]]:
    """Choose k minimizing reconstruction MSE (no Gaussian likelihood for ICA)."""
    n, p = X.shape
    k_hi = min(k_max, p, max(1, n - 1))
    k_lo = max(1, min(k_min, k_hi))

    candidates = list(range(k_lo, k_hi + 1, max(1, k_step)))
    if candidates[-1] != k_hi and k_hi not in candidates:
        candidates.append(k_hi)

    best_k = candidates[0]
    best_mse = float("inf")
    for k in candidates:
        try:
            mse = ica_reconstruction_mse(
                X, k, random_state=random_state, max_iter=max_iter
            )
        except Exception:
            continue
        if mse < best_mse:
            best_mse = mse
            best_k = k
    return best_k, {
        "criterion": "ica_reconstruction_mse",
        "best_value": best_mse,
        "k_step": k_step,
        "k_candidates": candidates,
    }
