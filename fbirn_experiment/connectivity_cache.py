"""Disk caches for connectivity edge matrices and group-mean FNC matrices."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from fbirn_experiment.connectivity import ConnectivityMethod, fnc_edges

EDGE_CACHE_VERSION = 1
GROUP_MEAN_CACHE_VERSION = 1
META_FILENAME = "meta.json"
N_HC_FILENAME = "n_hc.npy"
N_SZ_FILENAME = "n_sz.npy"


def _safe_method_name(method: str) -> str:
    return method.replace("/", "_").replace("\\", "_")


def _edge_cache_path(cache_dir: Path, method: str) -> Path:
    return cache_dir / f"edges__{_safe_method_name(method)}.npz"


def _meta_path(cache_dir: Path) -> Path:
    return cache_dir / META_FILENAME


def _expected_meta(
    time_courses: np.ndarray,
    method: str,
    edges: np.ndarray,
) -> dict[str, int | str]:
    return {
        "version": EDGE_CACHE_VERSION,
        "method": str(method),
        "n_subjects": int(time_courses.shape[0]),
        "n_timepoints": int(time_courses.shape[1]),
        "n_icns": int(time_courses.shape[2]),
        "n_edges": int(edges.shape[1]),
    }


def _load_cached_edges(
    cache_dir: Path,
    method: str,
    time_courses: np.ndarray,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]] | None:
    path = _edge_cache_path(cache_dir, method)
    if not path.is_file():
        return None

    try:
        with np.load(path) as data:
            edges = np.asarray(data["edges"], dtype=np.float64)
            ii = np.asarray(data["ii"])
            jj = np.asarray(data["jj"])
            meta_raw = str(data["meta"].item())
        meta = json.loads(meta_raw)
    except Exception:
        return None

    expected_n_subjects = int(time_courses.shape[0])
    expected_n_icns = int(time_courses.shape[2])
    expected_n_edges = expected_n_icns * (expected_n_icns - 1) // 2

    if meta.get("version") != EDGE_CACHE_VERSION or meta.get("method") != method:
        return None
    if meta.get("n_subjects") != expected_n_subjects:
        return None
    if meta.get("n_timepoints") != int(time_courses.shape[1]):
        return None
    if meta.get("n_icns") != expected_n_icns:
        return None
    if meta.get("n_edges") != expected_n_edges:
        return None
    if edges.shape != (expected_n_subjects, expected_n_edges):
        return None
    if ii.shape != (expected_n_edges,) or jj.shape != (expected_n_edges,):
        return None
    expected_ii, expected_jj = np.triu_indices(expected_n_icns, k=1)
    if not np.array_equal(ii, expected_ii):
        return None
    if not np.array_equal(jj, expected_jj):
        return None
    if not np.all(np.isfinite(edges)):
        return None

    return edges, (ii, jj)


def save_connectivity_edges_cache(
    cache_dir: Path,
    method: str,
    time_courses: np.ndarray,
    edges_bundle: tuple[np.ndarray, tuple[np.ndarray, np.ndarray]],
) -> None:
    """Persist a connectivity edge matrix for one multiverse connectivity method."""
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    edges, (ii, jj) = edges_bundle
    meta = _expected_meta(time_courses, method, edges)
    np.savez_compressed(
        _edge_cache_path(d, method),
        edges=np.asarray(edges, dtype=np.float64),
        ii=np.asarray(ii),
        jj=np.asarray(jj),
        meta=json.dumps(meta, sort_keys=True),
    )
    meta_path = _meta_path(d)
    all_meta: dict[str, object]
    if meta_path.is_file():
        try:
            all_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            all_meta = {}
    else:
        all_meta = {}
    all_meta["version"] = EDGE_CACHE_VERSION
    all_meta.setdefault("methods", {})
    methods = all_meta["methods"]
    if isinstance(methods, dict):
        methods[str(method)] = meta
    meta_path.write_text(
        json.dumps(all_meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def get_or_compute_connectivity_edges(
    cache_dir: Path,
    method: ConnectivityMethod,
    time_courses: np.ndarray,
    *,
    compute_fn: Callable[
        ...,
        tuple[np.ndarray, tuple[np.ndarray, np.ndarray]],
    ] = fnc_edges,
) -> tuple[tuple[np.ndarray, tuple[np.ndarray, np.ndarray]], bool]:
    """Load a cached edge matrix or compute, save, and return it.

    Returns
    -------
    (edges_bundle, cache_hit)
        ``cache_hit`` is ``True`` only when the bundle came from disk.
    """
    cached = _load_cached_edges(cache_dir, method, time_courses)
    if cached is not None:
        return cached, True

    computed = compute_fn(time_courses, method=method)
    save_connectivity_edges_cache(cache_dir, method, time_courses, computed)
    return computed, False


def _mean_hc_path(cache_dir: Path, measure: str) -> Path:
    return cache_dir / f"mean_hc__{measure}.npy"


def _mean_sz_path(cache_dir: Path, measure: str) -> Path:
    return cache_dir / f"mean_sz__{measure}.npy"


def save_group_mean_connectivity_cache(
    cache_dir: Path,
    rows: Sequence[tuple[str, np.ndarray, np.ndarray]],
    n_hc: int,
    n_sz: int,
) -> None:
    """Write group-mean matrices plus small metadata for plotting reuse."""
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / N_HC_FILENAME, np.int64(n_hc))
    np.save(d / N_SZ_FILENAME, np.int64(n_sz))
    measures: list[str] = []
    for method_key, mean_hc, mean_sz in rows:
        measures.append(str(method_key))
        np.save(_mean_hc_path(d, method_key), np.asarray(mean_hc, dtype=np.float64))
        np.save(_mean_sz_path(d, method_key), np.asarray(mean_sz, dtype=np.float64))
    meta = {"version": GROUP_MEAN_CACHE_VERSION, "measures": measures}
    (d / META_FILENAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def validate_cache_matches_data(
    time_courses: np.ndarray,
    y: np.ndarray,
    *,
    n_hc: int,
    n_sz: int,
    n_icns: int,
) -> None:
    """Ensure cached counts and ICN dimension match the loaded cohort."""
    yv = np.asarray(y).astype(int).ravel()
    if time_courses.shape[0] != yv.shape[0]:
        raise ValueError("time_courses and y length mismatch.")
    got_hc = int((yv == 0).sum())
    got_sz = int((yv == 1).sum())
    if got_hc != n_hc or got_sz != n_sz:
        raise ValueError(
            f"Cache n_hc/n_sz ({n_hc}, {n_sz}) does not match labels ({got_hc}, {got_sz})."
        )
    if int(time_courses.shape[2]) != n_icns:
        raise ValueError(
            f"Cache n_icns {n_icns} does not match time_courses.shape[2] {time_courses.shape[2]}."
        )


def try_load_group_mean_connectivity_cache(
    cache_dir: Path,
    measures: Sequence[str],
) -> tuple[list[tuple[str, np.ndarray, np.ndarray]], int, int, int] | None:
    """Load group-mean matrices if every required cache file exists."""
    d = Path(cache_dir)
    if not d.is_dir():
        return None
    need = [d / N_HC_FILENAME, d / N_SZ_FILENAME]
    for m in measures:
        need.append(_mean_hc_path(d, m))
        need.append(_mean_sz_path(d, m))
    if not all(p.is_file() for p in need):
        return None

    n_hc = int(np.load(d / N_HC_FILENAME))
    n_sz = int(np.load(d / N_SZ_FILENAME))

    rows: list[tuple[str, np.ndarray, np.ndarray]] = []
    n_icns = -1
    for m in measures:
        mean_hc = np.asarray(np.load(_mean_hc_path(d, m)), dtype=np.float64)
        mean_sz = np.asarray(np.load(_mean_sz_path(d, m)), dtype=np.float64)
        bad_shape = (
            mean_hc.shape != mean_sz.shape
            or mean_hc.ndim != 2
            or mean_hc.shape[0] != mean_hc.shape[1]
        )
        if bad_shape:
            return None
        if n_icns < 0:
            n_icns = mean_hc.shape[0]
        elif mean_hc.shape[0] != n_icns:
            return None
        rows.append((m, mean_hc, mean_sz))

    return rows, n_hc, n_sz, n_icns


def read_cache_meta(cache_dir: Path) -> dict[str, Any] | None:
    p = Path(cache_dir) / META_FILENAME
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
