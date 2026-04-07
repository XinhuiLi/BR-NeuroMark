"""H3: exploratory factor loadings — between-/within-domain and hypothesis domain pairs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

from fbirn_experiment.component_selection import select_n_components_fa
from fbirn_experiment.fnc import edge_domain_mask, edge_pair_mask_for_domains

# Literature-derived schizophrenia hypothesis domain pairs.
# Labels follow the NeuroMark 2.2 "{domain_abbrev}-{subdomain_abbrev}" format
# (Iraji et al., 2023 HBM).
#
# Domain–subdomain labels in this dataset:
#   CB          Cerebellar
#   VI-OT       Visual, Occipitotemporal
#   VI-OC       Visual, Occipital
#   PL          Paralimbic
#   SC-EH       Subcortical, Extended Hippocampal
#   SC-ET       Subcortical, Extended Thalamic
#   SC-BG       Subcortical, Basal Ganglia
#   SM          Sensorimotor
#   HC-IT       Higher Cognition, Insular-Temporal
#   HC-TP       Higher Cognition, Temporoparietal
#   HC-FR       Higher Cognition, Frontal
#   TN-CE       Triple Network, Central Executive
#   TN-DM       Triple Network, Default Mode
#   TN-SA       Triple Network, Salience
SZ_HYPOTHESIS_PAIRS: list[tuple[str, str, str]] = [
    (
        "TN-DM", "TN-CE",
        "DMN–Central Executive: aberrant anticorrelation / hypoconnectivity "
        "between DMN and central executive network. "
        "Li et al. 2019 Front Psychiatry 10:482 (doi:10.3389/fpsyt.2019.00482); "
        "Littow et al. 2015 Front Psychiatry 6:26 (doi:10.3389/fpsyt.2015.00026)"
    ),
    (
        "TN-DM", "HC-IT",
        "DMN–Insular/Temporal: DMN alterations underlying auditory verbal "
        "hallucinations; insular-temporal regions link auditory and frontal hubs. "
        "Marino & Spironelli 2022 J Psychiatr Res (PMID:35981441, "
        "doi:10.1016/j.jpsychires.2022.08.006); "
        "Weber et al. 2020 Front Psychiatry 11:227 (doi:10.3389/fpsyt.2020.00227)"
    ),
    (
        "SC-ET", "TN-CE",
        "Thalamic–Central Executive: reduced prefrontal-thalamic functional "
        "connectivity and impaired cognitive control. "
        "Woodward et al. 2012 Am J Psychiatry 169:1092-1099 "
        "(PMID:23032387, doi:10.1176/appi.ajp.2012.12010056); "
        "Giraldo-Chica & Woodward 2017 Schizophr Res 180:58-63 "
        "(doi:10.1016/j.schres.2016.08.005)"
    ),
    (
        "SC-ET", "TN-DM",
        "Thalamic–DMN: thalamo-cortical loop disruption; thalamic hypoconnectivity "
        "with prefrontal/DMN regions and hyperconnectivity with sensorimotor areas. "
        "Woodward et al. 2012 Am J Psychiatry 169:1092-1099 "
        "(PMID:23032387, doi:10.1176/appi.ajp.2012.12010056); "
        "Ferri et al. 2018 Sci Rep 8:3451 (doi:10.1038/s41598-019-39367-z)"
    ),
    (
        "TN-CE", "SM",
        "Central Executive–Sensorimotor: disintegration of sensorimotor networks "
        "from higher-order cognitive/executive control networks. "
        "Kaufmann et al. 2015 Schizophr Bull 41:1326-1335 "
        "(PMID:25943122, doi:10.1093/schbul/sbv060)"
    ),
    (
        "HC-IT", "SM",
        "Insular/Temporal–Sensorimotor: sensory processing disruption; "
        "auditory-sensorimotor hyperconnectivity with subcortical nuclei. "
        "Avram et al. 2018 Neuropsychopharmacology 43:2462-2471 "
        "(doi:10.1038/s41386-018-0059-z); "
        "Kaufmann et al. 2015 Schizophr Bull 41:1326-1335 "
        "(PMID:25943122, doi:10.1093/schbul/sbv060)"
    ),
    (
        "TN-DM", "VI-OC",
        "DMN–Visual (Occipital): posterior DMN hypoconnectivity with occipital "
        "visual network; aberrant dynamic visual-sensory patterns. "
        "Dong et al. 2018 Sci Rep 8:14655 (doi:10.1038/srep14655); "
        "Sendi et al. 2021 Schizophr Res 228:103-111 "
        "(doi:10.1016/j.schres.2020.11.055)"
    ),
    (
        "TN-SA", "TN-DM",
        "Salience–DMN: aberrant salience-to-DMN switching; the triple network "
        "model posits that salience network dysfunction drives DMN/CEN imbalance. "
        "Palaniyappan & Liddle 2012 J Neurol Neurosurg Psychiatry 83:558-567 "
        "(doi:10.1136/jnnp-2011-301452); "
        "Menon 2011 Trends Cogn Sci 15:483-506 "
        "(doi:10.1016/j.tics.2011.08.003)"
    ),
    (
        "TN-SA", "TN-CE",
        "Salience–Central Executive: disrupted salience-driven engagement of "
        "central executive resources; impaired anterior insular mediation. "
        "Menon 2011 Trends Cogn Sci 15:483-506 "
        "(doi:10.1016/j.tics.2011.08.003); "
        "Manoliu et al. 2014 Schizophr Bull 40:428-437 "
        "(doi:10.1093/schbul/sbt037)"
    ),
]


@dataclass
class H3Result:
    summary: pd.DataFrame
    loadings: np.ndarray
    n_components_fitted: int
    selection: dict[str, Any]
    domain_pair_summary: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame()
    )


def _resolve_domain_label(icn_domain: np.ndarray, label: str) -> str | None:
    """
    Match a domain–subdomain label (e.g. 'TN-DM', 'SC-ET', 'SM') against
    the unique labels present in *icn_domain*.  Returns the matching string
    or None if no match is found.
    """
    unique = {str(d) for d in icn_domain}
    if label in unique:
        return label
    label_lc = label.lower()
    hits = [u for u in unique if u.lower() == label_lc]
    if len(hits) == 1:
        return hits[0]
    return None


def h3_factor_loadings_between_within(
    edges: np.ndarray,
    y: np.ndarray,
    icn_domain: np.ndarray,
    ii: np.ndarray,
    jj: np.ndarray,
    *,
    use_bic_selection: bool = True,
    k_min: int = 5,
    k_max: int = 50,
    k_step: int = 5,
    fa_criterion: str = "bic",
    n_components_fixed: int = 10,
    random_state: int = 0,
    bic_max_iter: int = 1000,
    fa_fit_max_iter: int = 2000,
    hypothesis_pairs: Sequence[tuple[str, str, str]] | None = None,
) -> H3Result:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(edges)
    n, p = Xs.shape
    selection: dict[str, Any] = {}

    if use_bic_selection:
        k, diag = select_n_components_fa(
            Xs,
            k_min=k_min,
            k_max=k_max,
            k_step=k_step,
            criterion=fa_criterion,
            random_state=random_state,
            max_iter=bic_max_iter,
        )
        selection = {"mode": "bic", **diag, "k_selected": int(k)}
    else:
        k = int(min(max(1, n_components_fixed), p - 1, n - 1))
        selection = {"mode": "fixed", "k_selected": k}

    fa = FactorAnalysis(
        n_components=k, max_iter=fa_fit_max_iter, random_state=random_state
    )
    fa.fit(Xs)
    loadings = fa.components_

    within, between = edge_domain_mask(icn_domain, ii, jj)

    # ── Per-factor between / within summary ──────────────────────────────
    rows = []
    for comp in range(loadings.shape[0]):
        ell = np.abs(loadings[comp])
        rows.append(
            {
                "factor": comp,
                "mean_abs_loading_between": float(np.mean(ell[between]))
                if between.any()
                else np.nan,
                "mean_abs_loading_within": float(np.mean(ell[within]))
                if within.any()
                else np.nan,
            }
        )

    # ── Hypothesis domain-pair loadings ──────────────────────────────────
    pairs = hypothesis_pairs if hypothesis_pairs is not None else SZ_HYPOTHESIS_PAIRS
    dp_rows: list[dict[str, Any]] = []
    resolved_pairs: list[tuple[str, str, str, str]] = []

    for short_a, short_b, rationale in pairs:
        real_a = _resolve_domain_label(icn_domain, short_a)
        real_b = _resolve_domain_label(icn_domain, short_b)
        if real_a is None or real_b is None:
            continue
        mask = edge_pair_mask_for_domains(icn_domain, ii, jj, real_a, real_b)
        if not mask.any():
            continue
        resolved_pairs.append((short_a, short_b, real_a, real_b))
        for comp in range(loadings.shape[0]):
            ell = np.abs(loadings[comp])
            dp_rows.append(
                {
                    "factor": comp,
                    "domain_a": short_a,
                    "domain_b": short_b,
                    "pair_label": f"{short_a} \u00d7 {short_b}",
                    "rationale": rationale,
                    "n_edges": int(mask.sum()),
                    "mean_abs_loading": float(np.mean(ell[mask])),
                }
            )

    domain_pair_summary = pd.DataFrame(dp_rows)
    selection["hypothesis_pairs_resolved"] = [
        {"short": (a, b), "resolved": (ra, rb)}
        for a, b, ra, rb in resolved_pairs
    ]

    return H3Result(
        summary=pd.DataFrame(rows),
        loadings=loadings,
        n_components_fitted=int(loadings.shape[0]),
        selection=selection,
        domain_pair_summary=domain_pair_summary,
    )
