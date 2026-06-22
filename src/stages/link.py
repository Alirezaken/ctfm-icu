"""§9.3  link -- MIMIC-IV intersect MIMIC-CXR linkage.

The relational link layer is already built by pipeline/build_link_layer.py into
paths.link_dir (patients, admissions, icu_stays, ed_stays, cxr_studies,
cohort_index). This stage verifies it is present and complete, records linkage
counts for the manifest, and is where the project-wide time-zero discipline
assertion will be wired in once cohorts define their time zeros.

It does NOT rebuild the link layer (that is the existing pipeline's job); it
reuses it so the only heavy data pass stays where it already ran.
"""
from __future__ import annotations

import pandas as pd

from src.util import log, die, Checkpoint

_REQUIRED = [
    "patients.csv", "admissions.csv", "icu_stays.csv",
    "ed_stays.csv", "cxr_studies.csv", "cohort_index.csv",
]


def run(cfg, force: bool = False):
    ckpt = Checkpoint(cfg, "link")
    link_dir = cfg.input("link_dir")

    missing = [f for f in _REQUIRED if not (link_dir / f).exists()]
    if missing:
        die(f"link layer incomplete in {link_dir}: missing {missing}. "
            f"Build it first: python pipeline/build_link_layer.py")

    if ckpt.done("verified") and not force:
        log(f"link layer already verified at {link_dir} (use --force to re-check)")
        return

    counts = {}
    for f in _REQUIRED:
        n = sum(1 for _ in open(link_dir / f)) - 1  # minus header
        counts[f] = n
        log(f"  {f:18s} {n:>9,} rows")

    # all-modality cohort spine: who has EHR + CXR (note/image gating happens at
    # cohort build, against the embeddings).
    cohort = pd.read_csv(link_dir / "cohort_index.csv")
    log(f"  cohort_index columns: {list(cohort.columns)}")

    ckpt.mark("verified", counts=counts)
    log("link verified.")
