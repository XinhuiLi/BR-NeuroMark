"""Load time courses, labels, optional domain map; synthetic data."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fbirn_experiment.config import DEFAULT_ICN_DOMAIN_PATH, DEFAULT_NEUROMARK_XLSX_PATH


def load_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    return d["time_courses"], d["y"], d["icn_domain"]


def load_neuromark_labels(
    xlsx_path: str | Path,
    *,
    domain_col: str = "domain_abbrev",
    subdomain_col: str = "subdomain_abbrev",
) -> np.ndarray:
    """Build ``{domain_abbrev}-{subdomain_abbrev}`` labels from the NeuroMark xlsx.

    When domain and subdomain abbreviations are identical (e.g. CB/CB),
    the label is just the domain abbreviation (e.g. ``"CB"``).
    """
    import pandas as pd  # noqa: PLC0415

    df = pd.read_excel(xlsx_path)
    if domain_col not in df.columns or subdomain_col not in df.columns:
        raise ValueError(
            f"xlsx must contain columns '{domain_col}' and '{subdomain_col}'; "
            f"found: {list(df.columns)}"
        )
    labels: list[str] = []
    for _, row in df.iterrows():
        dom = str(row[domain_col]).strip()
        sub = str(row[subdomain_col]).strip()
        labels.append(dom if dom == sub else f"{dom}-{sub}")
    return np.array(labels, dtype=str)


def load_fbirn_tc_and_labels(
    tc_path: str | Path,
    label_path: str | Path,
    icn_domain_path: str | Path | None = None,
    *,
    neuromark_xlsx_path: str | Path | None = None,
    placeholder_n_domains: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tc_p, y_p = Path(tc_path), Path(label_path)
    if not tc_p.is_file():
        raise FileNotFoundError(f"Time course file not found: {tc_p}")
    if not y_p.is_file():
        raise FileNotFoundError(f"Label file not found: {y_p}")

    tc = np.asarray(np.load(tc_p), dtype=np.float64)
    y = np.asarray(np.load(y_p)).astype(int).ravel()

    if tc.ndim != 3:
        raise ValueError(
            f"Expected time courses with shape (n_subj, n_time, n_icns); got {tc.shape}"
        )
    n_subj, _, n_icns = tc.shape
    if y.shape[0] != n_subj:
        raise ValueError(
            f"Label count {y.shape[0]} does not match subjects in TC array ({n_subj})."
        )
    if not np.array_equal(np.sort(np.unique(y)), np.array([0, 1])):
        raise ValueError(
            f"Labels must be binary {{0,1}} (0=HC, 1=SZ); unique values: {np.unique(y)}"
        )

    dom_path = Path(icn_domain_path) if icn_domain_path else DEFAULT_ICN_DOMAIN_PATH
    if dom_path.is_file():
        icn_domain = np.asarray(np.load(dom_path, allow_pickle=True)).ravel()
        if icn_domain.dtype == object:
            icn_domain = icn_domain.astype(str)
        if icn_domain.shape[0] != n_icns:
            raise ValueError(
                f"icn_domain length {icn_domain.shape[0]} != n_icns {n_icns}"
            )
    else:
        xlsx = Path(neuromark_xlsx_path) if neuromark_xlsx_path else DEFAULT_NEUROMARK_XLSX_PATH
        if xlsx.is_file():
            icn_domain = load_neuromark_labels(xlsx)
            if icn_domain.shape[0] != n_icns:
                raise ValueError(
                    f"NeuroMark xlsx has {icn_domain.shape[0]} rows but "
                    f"expected {n_icns} ICNs"
                )
            print(
                f"Built domain labels from NeuroMark xlsx ({xlsx}): "
                f"{len(set(icn_domain))} unique labels"
            )
        else:
            icn_domain = np.arange(n_icns, dtype=np.int64) % placeholder_n_domains
            print(
                f"Warning: ICN domain file not found ({dom_path}). "
                f"Using placeholder domains 0..{placeholder_n_domains - 1}; "
                "H2/H3 are not interpretable until you add a real (105,) domain array."
            )

    return tc, y, icn_domain


def synthetic_dataset(
    n_subj: int = 363,
    n_t: int = 157,
    n_icns: int = 105,
    n_domains: int = 7,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    dom = np.arange(n_icns) % n_domains
    tc = rng.standard_normal((n_subj, n_t, n_icns))
    y = rng.integers(0, 2, size=n_subj)
    signal = np.zeros_like(tc)
    for s in range(n_subj):
        if y[s] == 1:
            for d in range(n_domains):
                idx = np.where(dom == d)[0]
                if len(idx) > 1:
                    u = rng.standard_normal(n_t) * 0.15
                    for k in idx:
                        signal[s, :, k] += u
            between_pairs = [
                (i, j)
                for i in range(n_icns)
                for j in range(i + 1, n_icns)
                if dom[i] != dom[j]
            ]
            for _ in range(80):
                i, j = between_pairs[rng.integers(len(between_pairs))]
                signal[s, :, i] += 0.12 * rng.standard_normal(n_t)
                signal[s, :, j] += 0.12 * rng.standard_normal(n_t)
    tc = tc + signal
    return tc, y.astype(np.int64), dom
