"""Pipeline path/constant shim -- config.yaml is the single source of truth (§0).

This module no longer defines any configuration of its own. It loads config.yaml
(through src/cfg.py, with the same ${ENV}/${ENV:default} interpolation) and
re-exports the legacy module-level names the standalone data-prep scripts import
(`from config import MIMIC_DIR, ...`). That keeps those scripts unchanged while
there is exactly one place any value is defined: config.yaml.

The scripts run as plain SLURM jobs from this directory, so they only need the
names below; the causal pipeline reads the same config via src.cfg directly.
"""
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src import cfg as _cfg

_C = _cfg.load()


def _path(subkey: str) -> str:
    return str(_C.input(subkey))


# ---- inputs (raw data) ----------------------------------------------------
PROJECT_ROOT = str(_REPO_ROOT)
DATASETS_DIR = _path("datasets_dir")
MIMIC_DIR    = _path("mimic_dir")
ED_DIR       = _path("ed_dir")
CXR_MASTER   = _path("cxr_master")
CXR_IMG_ROOT = _path("cxr_images")

# ---- outputs (derived data) ----------------------------------------------
DATA_DIR    = _path("data_dir")
LINK_DIR    = _path("link_dir")
EVENT_DIR   = _path("event_stream_raw")    # raw unified event-stream (Parquet)
EVENT_CLEAN = _path("event_stream")        # value-cleaned copy (raw is never modified)

# ---- constants ------------------------------------------------------------
# CheXpert label columns in mimic_master_list.csv (encoding: 1=pos, 0=neg,
# 2=not-mentioned, 3=uncertain).
LABELS = list(_C.get("chexpert_labels"))
