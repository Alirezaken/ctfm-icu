"""§9.7  estimate -- cross-fitted AIPW for every adjustment condition.

Plan (§5, §6):
  - One estimator everywhere: cross-fitted AIPW, LightGBM nuisance models for
    propensity and outcome, cross-fitting always on. Effect = risk difference at
    the trial horizon, in percentage points.
  - Conditions (all on the shared cohort): naive, structured, plus_notes,
    plus_imaging_only, full, design_based. Embeddings enter as extra covariates.
  - Administrative censoring at horizon + competing events via IPC weights inside
    the estimator (same covariates as the outcome model).
  - Diagnostics per condition: propensity overlap before/after trimming, ESS
    after weighting, CI width.
  - Outputs this stage feeds: effects.csv (main effects + bias vs RCT reference +
    inside-CI flag + negative-control effect), controls.csv (neg-control + E-values),
    and the overlap/ESS diagnostics in cohorts.csv.
  - Kind-B reporting via src.stats: AIPW point estimate + patient-level cluster
    bootstrap (10000, fixed seed, shared indices across conditions) + IF-CI check.
"""
from src.util import log


def run(cfg, force: bool = False, intervention: str = "fluids_sepsis"):
    cfg.require("estimator.method", "estimator.cross_fitting_folds",
                "bootstrap.n_resamples", "run.seed",
                f"interventions.{intervention}.rct_reference.risk_difference")
    log(f"estimate[{intervention}]: not yet implemented.")
    raise NotImplementedError(
        "estimate pending: needs the emulated cohort, stored embeddings, lightgbm "
        "in the venv, and the RCT reference values from Soroosh."
    )
