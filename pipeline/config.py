"""Central configuration: dataset/output paths and shared constants.

All paths derive from PROJECT_ROOT (the repo root, = parent of this pipeline/
directory) and can be overridden via environment variables, so the pipeline is
portable across machines without editing source.
"""
import os

PROJECT_ROOT = os.environ.get(
    "CTFM_PROJECT_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

# ---- inputs (raw data) ----------------------------------------------------
DATASETS_DIR = os.environ.get("CTFM_DATASETS", os.path.join(PROJECT_ROOT, "Datasets"))
MIMIC_DIR    = os.environ.get("CTFM_MIMIC",    os.path.join(DATASETS_DIR, "physionet.org/files/mimiciv/3.1"))
ED_DIR       = os.environ.get("CTFM_ED",       os.path.join(DATASETS_DIR, "mimic-iv-ed-2.2/ed"))
CXR_MASTER   = os.environ.get("CTFM_CXR_MASTER", os.path.join(DATASETS_DIR, "mimic_master_list.csv"))
CXR_IMG_ROOT = os.environ.get("CTFM_CXR_IMAGES", os.path.join(DATASETS_DIR, "preprocessed"))

# ---- outputs (derived data) ----------------------------------------------
DATA_DIR    = os.environ.get("CTFM_DATA", os.path.join(PROJECT_ROOT, "data"))
LINK_DIR    = os.path.join(DATA_DIR, "link")                 # relational link layer (CSV)
EVENT_DIR   = os.path.join(DATA_DIR, "event_stream")         # raw unified event-stream (Parquet)
EVENT_CLEAN = os.path.join(DATA_DIR, "event_stream_clean")   # value-cleaned copy (raw is never modified)

# ---- constants ------------------------------------------------------------
# CheXpert label columns in mimic_master_list.csv.
# Encoding: 1=positive, 0=negative, 2=not-mentioned, 3=uncertain.
LABELS = [
    "atelectasis", "cardiomegaly", "consolidation", "edema",
    "enlarged_cardiomediastinum", "fracture", "lung_lesion", "lung_opacity",
    "no_finding", "pleural_effusion", "pleural_other", "pneumonia",
    "pneumothorax", "support_devices",
]
