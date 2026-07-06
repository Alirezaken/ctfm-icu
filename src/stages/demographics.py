"""§9.10 / §7.7  demographics -- subgroup re-estimation.

Re-estimate the effect within each sex and each age band, under `structured` and
`full`, with the same cross-fitted AIPW (and the active variant's embedding
reduction). Report the subgroup effect + CI (Kind B), the subgroup bias reduction
vs the crude estimate (Kind B), a support count, and an `undefined` flag when the
subgroup has too few outcome events in an arm to estimate. -> demographics.csv.
"""
from __future__ import annotations

import re
import time
import numpy as np
import pandas as pd

from src.util import log
from src import features, aipw, reduce, results as R
from src.stats import cluster_bootstrap_indices, bootstrap_summary
from src.stages.estimate import _reset

_MIN_ARM_EVENTS = 5   # need >=5 events (and >=5 non-events) in each arm to define a subgroup


def _parse_band(s):
    m = re.match(r"\(([\d.]+),\s*([\d.]+)\]", str(s))
    return (float(m.group(1)), float(m.group(2))) if m else None


def run(cfg, force: bool = False, intervention: str = "fluids_sepsis"):
    cfg.require("demographics.age_bands",
                f"interventions.{intervention}.rct_reference.risk_difference")
    t0 = time.time()
    seed = int(cfg.get("run.seed", 42))
    folds = int(cfg.get("estimator.cross_fitting_folds", 5))
    nboot = int(cfg.get("bootstrap.n_resamples", 10000))
    ref = cfg.get(f"interventions.{intervention}.rct_reference")["risk_difference"]

    cohort = pd.read_parquet(cfg.storage("cohorts", f"{intervention}.parquet"))
    cohort = cohort[cohort["all_modality"] & cohort["arm"].notna()].reset_index(drop=True)
    A = (cohort["arm"] == "active").to_numpy().astype(int)
    Y = cohort["outcome"].to_numpy().astype(int)
    subj = cohort["subject_id"].to_numpy()

    S = features.structured_at_t0(cfg, cohort).to_numpy(dtype=float)
    Ximg = features.pool_embeddings(cfg, cohort, "images")
    Xnote = features.pool_embeddings(cfg, cohort, "notes", "notes_all")
    if cfg.reduction != "none":
        Ximg = reduce.apply(Ximg, cfg.reduction, A, Y, folds, seed, cfg.pca_components)
        Xnote = reduce.apply(Xnote, cfg.reduction, A, Y, folds, seed, cfg.pca_components)
    conds = {"naive": np.ones((len(Y), 1)), "structured": S,
             "full": np.hstack([S, Xnote, Ximg])}

    sexes = cfg.get("demographics.sex_levels", ["F", "M"])
    sex = cohort["sex"].astype(str).to_numpy()
    age = pd.to_numeric(cohort["age_t0"], errors="coerce").to_numpy()
    subgroups = [("sex", s, sex == s) for s in sexes]
    for band in cfg.get("demographics.age_bands", []):
        pb = _parse_band(band)
        if pb:
            lo, hi = pb
            subgroups.append(("age_band", band, (age > lo) & (age <= hi)))

    rows = []
    for stype, sname, mask in subgroups:
        n = int(mask.sum())
        a, y = A[mask], Y[mask]
        ok = (n > 0 and a.min() != a.max()
              and y[a == 1].sum() >= _MIN_ARM_EVENTS and y[a == 0].sum() >= _MIN_ARM_EVENTS
              and (1 - y[a == 1]).sum() >= _MIN_ARM_EVENTS and (1 - y[a == 0]).sum() >= _MIN_ARM_EVENTS)
        if not ok:
            for cond in ("structured", "full"):
                rows.append({"intervention": intervention, "subgroup_type": stype,
                             "subgroup": sname, "condition": cond,
                             "support_count": n, "undefined": True})
            log(f"  {stype}={sname}: n={n} -> undefined (too few arm events)")
            continue

        boot = list(cluster_bootstrap_indices(subj[mask], nboot, seed))
        pv = {}
        for cond, X in conds.items():
            try:
                psi, keep, _ = aipw.crossfit_aipw(X[mask], a, y, folds, seed)
                pt = float(psi[keep].mean() * 100)
                bt = np.array([psi[b][keep[b]].mean() * 100 for b in boot])
                pv[cond] = (pt, bt)
            except Exception as e:
                pv[cond] = None
                log(f"  {stype}={sname}/{cond}: estimation failed ({e})")
        if pv.get("naive") is None:
            continue
        npt, nbt = pv["naive"]
        for cond in ("structured", "full"):
            if pv.get(cond) is None:
                rows.append({"intervention": intervention, "subgroup_type": stype,
                             "subgroup": sname, "condition": cond,
                             "support_count": n, "undefined": True})
                continue
            pt, bt = pv[cond]
            eff = bootstrap_summary(pt, bt)
            br = bootstrap_summary(abs(npt - ref) - abs(pt - ref),
                                   np.abs(nbt - ref) - np.abs(bt - ref))
            rows.append({"intervention": intervention, "subgroup_type": stype,
                         "subgroup": sname, "condition": cond,
                         **eff.as_row("effect_"), **br.as_row("bias_reduction_"),
                         "support_count": n, "undefined": False})
        log(f"  {stype}={sname}: n={n} structured RD={pv['structured'][0]:.1f} "
            f"full RD={pv['full'][0]:.1f}")

    _reset(cfg, "demographics.csv", intervention)
    R.append_rows(cfg, "demographics.csv", rows)
    log(f"demographics[{intervention}] done in {time.time()-t0:,.0f}s -> demographics.csv "
        f"({len(rows)} rows).")
