"""link -- verify the relational link layer is present and complete.

Built by pipeline/build_link_layer.py. This stage does not rebuild it; it checks that the
tables the study actually reads exist, and records their row counts.

Only three of the six link tables are read downstream (patients, icu_stays, cxr_studies).
The other three are built and never used. That is fine and worth knowing, so it is logged
rather than quietly ignored.
"""
from __future__ import annotations

import pandas as pd

from src.util import log, die, Checkpoint

_REQUIRED = ["patients.csv", "icu_stays.csv", "cxr_studies.csv"]
_BUILT_UNUSED = ["admissions.csv", "ed_stays.csv", "cohort_index.csv"]


def run(cfg, force: bool = False, intervention: str = None):
    ckpt = Checkpoint(cfg, "link")
    d = cfg.input("link_dir")

    missing = [f for f in _REQUIRED if not (d / f).exists()]
    if missing:
        die(f"link layer incomplete in {d}: missing {missing}. "
            f"Build it first: python pipeline/build_link_layer.py")

    if ckpt.done("verified") and not force:
        log(f"link layer verified at {d} (use --force to re-check)")
        return

    counts = {}
    for f in _REQUIRED:
        n = sum(1 for _ in open(d / f)) - 1
        counts[f] = n
        log(f"  {f:20s} {n:>10,} rows   [used]")
    for f in _BUILT_UNUSED:
        if (d / f).exists():
            log(f"  {f:20s} {'':>10s}   [built, not read by the analysis]")

    cxr = pd.read_csv(d / "cxr_studies.csv", nrows=1)
    need = {"subject_id", "study_datetime", "views", "edema"}
    absent = need - set(cxr.columns)
    if absent:
        die(f"cxr_studies.csv is missing required columns: {sorted(absent)}")

    ckpt.mark("verified", counts=counts)
    log("link verified.")
