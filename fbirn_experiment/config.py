"""Default paths (project root = parent of this package)."""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_TC_PATH = DEFAULT_DATA_DIR / "FBIRN_ICN_TC_v2.2.npy"
DEFAULT_LABEL_PATH = DEFAULT_DATA_DIR / "FBIRN_label.npy"
DEFAULT_ICN_DOMAIN_PATH = DEFAULT_DATA_DIR / "FBIRN_icn_domain-subdomain_label.npy"
DEFAULT_NEUROMARK_XLSX_PATH = DEFAULT_DATA_DIR / "Neuromark_fMRI_2-2_labels_final.xlsx"
DEFAULT_CONFOUND_CSV_PATH = DEFAULT_DATA_DIR / "FBIRN_data.csv"

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "fbirn_icn_run"
