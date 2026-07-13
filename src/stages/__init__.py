"""Stage registry. The run order is the dependency order and main.py enforces it."""
from importlib import import_module

# THE ORDER MATTERS.
#   emulate must precede extract_embeddings (embedding is scoped to cohort patients)
#   estimate must precede diagnose          (diagnose reads the ESS the estimator paid)
#   synthesize must precede diagnose        (diagnose reads the calibration curve)
#   robustness must precede consolidate     (consolidate reads subgroup rows back)
#   integrity runs LAST and gates everything
STAGES = [
    "link",                # verify the link layer
    "emulate",             # build all four cohorts -> manifests/cohorts.parquet
    "extract_embeddings",  # the GPU stage: images, radtext, histnote (+ images_alt)
    "extract_external",    # external CXR sets, for the informativeness replication
    "estimate",            # the seven adjustment conditions -> effects.csv
    "synthesize",          # semi-synthetic benchmark -> synthetic.csv
    "diagnose",            # probes + incremental confounding -> diagnostics.csv
    "robustness",          # swaps + subgroups -> robustness.csv
    "external",            # eICU structured replication -> effects.csv
    "consolidate",         # all paired contrasts -> contrasts.csv, manifest.csv
    "integrity",           # assert every invariant. FAILS LOUDLY.
]

# stages that take --intervention; the rest ignore it
PER_INTERVENTION = {"emulate", "estimate", "diagnose", "robustness", "external"}


def get(name: str):
    if name not in STAGES:
        raise KeyError(f"unknown stage '{name}'. Known: {', '.join(STAGES)}")
    return import_module(f"src.stages.{name}")
