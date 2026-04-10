"""Domain label aggregation utilities for multiverse analysis."""

from __future__ import annotations

from typing import Literal

import numpy as np

DomainGranularity = Literal["subdomain_14", "domain_7"]

DOMAIN_GRANULARITIES: list[DomainGranularity] = ["subdomain_14", "domain_7"]

_SUBDOMAIN_TO_DOMAIN: dict[str, str] = {
    "CB": "CB",
    "VI-OT": "VI",
    "VI-OC": "VI",
    "PL": "PL",
    "SC-EH": "SC",
    "SC-ET": "SC",
    "SC-BG": "SC",
    "SM": "SM",
    "HC-IT": "HC",
    "HC-TP": "HC",
    "HC-FR": "HC",
    "TN-CE": "TN",
    "TN-DM": "TN",
    "TN-SA": "TN",
}


def aggregate_domains(
    icn_domain: np.ndarray,
    granularity: DomainGranularity = "subdomain_14",
) -> np.ndarray:
    """Return domain labels at the requested granularity.

    ``"subdomain_14"`` returns the labels as-is (14 domain–subdomain labels).
    ``"domain_7"`` maps each subdomain to its parent domain (7 labels).
    """
    if granularity == "subdomain_14":
        return np.array([str(d) for d in icn_domain], dtype=str)

    out: list[str] = []
    for d in icn_domain:
        key = str(d)
        parent = _SUBDOMAIN_TO_DOMAIN.get(key)
        if parent is None:
            parent = key.split("-")[0] if "-" in key else key
        out.append(parent)
    return np.array(out, dtype=str)
