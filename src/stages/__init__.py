"""Stage registry. Each stage is a module exposing `run(cfg, force=False)`.

Order matches the run order in ALIREZA_INSTRUCTIONS.md §9. main_causal.py
dispatches `--stage <name>` here; `--stage all` runs them in this order.
"""
from importlib import import_module

# name -> module under src.stages, in run order (§9)
STAGES = [
    "link",                # §9.3  build/verify MIMIC-IV intersect MIMIC-CXR linkage
    "extract_embeddings",  # §9.4  resize+save images, extract frozen image/note vectors
    "extract_external_cxr",# §3    RAD-DINO embeddings for external CXR sets (quality gate)
    "extract_images_alt",  # §9.11 BiomedCLIP embeddings (robustness encoder swap)
    "emulate",             # §9.5  build emulated-trial cohort(s)
    "probe",               # §9.6  validity probe (gate before estimating)
    "estimate",            # §9.7  AIPW for all conditions; main effects + controls
    "demographics",        # §9.10 subgroup re-estimation by sex and age band
    "robustness",          # §9.11 four swaps (encoder, estimator, window, pooling)
    "external",            # §9.12 structured + plus_notes rerun on eICU
    "consolidate",         # §9.13 merge into the 10 result files (§7)
    "integrity_check",     # §9.14 assert invariants; exactly 10 result files
]


def get(name: str):
    if name not in STAGES:
        raise KeyError(name)
    return import_module(f"src.stages.{name}")
