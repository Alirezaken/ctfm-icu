"""§9.7  estimate -- cross-fitted AIPW for every adjustment condition.

One estimator everywhere (cross-fitted AIPW, LightGBM nuisance); effect = risk
difference at the horizon, in percentage points. Conditions on the shared
all-modality cohort: naive, structured, plus_notes, plus_imaging_only, full
(design_based pending a fuller confounder extraction -- SOFA/comorbidities).

Kind-B reporting (§8): AIPW point + patient-level cluster bootstrap (shared
indices across conditions, so contrasts are paired). Writes main effects + bias
vs the RCT reference to effects.csv, and overlap/ESS diagnostics to cohorts.csv.
negative-control / E-values (controls.csv) are deferred until the negative-control
outcome (icu_acquired_uti) is built.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd

from src.util import log
from src import features, aipw, results as R
from src.stats import cluster_bootstrap_indices, bootstrap_summary


def run(cfg, force: bool = False, intervention: str = "fluids_sepsis"):
    cfg.require("estimator.method", "estimator.cross_fitting_folds",
                "bootstrap.n_resamples", "run.seed",
                f"interventions.{intervention}.rct_reference.risk_difference")
    t0 = time.time()

    cohort = pd.read_parquet(cfg.storage("cohorts", f"{intervention}.parquet"))
    cohort = cohort[cohort["all_modality"] & cohort["arm"].notna()].reset_index(drop=True)
    A = (cohort["arm"] == "active").astype(int).to_numpy()
    Y = cohort["outcome"].astype(int).to_numpy()
    subj = cohort["subject_id"].to_numpy()
    log(f"estimate[{intervention}]: all-modality cohort n={len(cohort):,} "
        f"(active={A.sum():,}, comparator={(A==0).sum():,}, deaths={Y.sum():,})")

    # ---- features (all strictly pre-t0) ----
    S = features.structured_at_t0(cfg, cohort).to_numpy(dtype=float)
    Ximg = features.pool_embeddings(cfg, cohort, "images")
    Xnote = features.pool_embeddings(cfg, cohort, "notes", "notes_clinical")
    conditions = {
        "naive": np.ones((len(Y), 1)),
        "structured": S,
        "plus_notes": np.hstack([S, Xnote]),
        "plus_imaging_only": np.hstack([S, Ximg]),
        "full": np.hstack([S, Xnote, Ximg]),
    }

    # ---- shared bootstrap indices (patient-level, reused across conditions) ----
    seed = int(cfg.get("run.seed", 42))
    nboot = int(cfg.get("bootstrap.n_resamples", 10000))
    folds = int(cfg.get("estimator.cross_fitting_folds", 5))
    boot = list(cluster_bootstrap_indices(subj, nboot, seed))

    ref = cfg.get(f"interventions.{intervention}.rct_reference")
    ref_rd = ref["risk_difference"]; ref_lo, ref_hi = ref["ci"]

    eff_rows, coh_rows = [], []
    for name, X in conditions.items():
        psi, keep, diag = aipw.crossfit_aipw(X, A, Y, folds, seed)
        point = float(psi[keep].mean() * 100)
        bvals = [psi[b][keep[b]].mean() * 100 for b in boot]
        eff = bootstrap_summary(point, bvals)
        bias = bootstrap_summary(point - ref_rd, [v - ref_rd for v in bvals])
        inside = bool(ref_lo <= point <= ref_hi)

        eff_rows.append({
            "intervention": intervention, "condition": name, "cohort": "all_modality",
            "dataset": "mimic", "method": "aipw" if name != "naive" else "unadjusted",
            **eff.as_row("effect_"),
            "ref_rd": ref_rd, "ref_ci_low": ref_lo, "ref_ci_high": ref_hi,
            **bias.as_row("bias_"), "inside_reference_ci": inside,
            "ci_width": round(eff.ci_high - eff.ci_low, 1),
            "effective_sample_size": round(diag["ess"]),
        })
        for metric, val in [("ess", round(diag["ess"])),
                            ("frac_trimmed", round(diag["frac_trimmed"], 4)),
                            ("propensity_min", round(diag["e_min"], 4)),
                            ("propensity_max", round(diag["e_max"], 4)),
                            ("ci_width", round(eff.ci_high - eff.ci_low, 1))]:
            coh_rows.append({"intervention": intervention, "section": "overlap_ess",
                             "metric": f"{metric}__{name}", "stratum": "", "arm": "",
                             "value": val, "support_count": diag["n"]})
        log(f"  {name:18s} RD={point:6.1f} pp  CI[{eff.ci_low:.1f},{eff.ci_high:.1f}]  "
            f"bias={point-ref_rd:+.1f}  inside_ref_CI={inside}  ESS={round(diag['ess'])}")

    R.append_rows(cfg, "effects.csv", eff_rows)
    R.append_rows(cfg, "cohorts.csv", coh_rows)
    log(f"estimate[{intervention}] done in {time.time()-t0:,.0f}s -> effects.csv (+cohorts.csv). "
        f"reference RD={ref_rd} [{ref_lo},{ref_hi}] pp. design_based + controls deferred.")
