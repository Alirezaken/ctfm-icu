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
from src import features, aipw, reduce, results as R
from src.stats import (cluster_bootstrap_indices, bootstrap_summary,
                       e_value, e_value_ci_limit, influence_function_ci,
                       minimum_detectable_effect, standardized_mean_differences)


def _reset(cfg, fname, intervention, section=None):
    """Drop this intervention's prior rows from a result file (idempotent re-runs),
    preserving other interventions and other sections (e.g. emulate's CONSORT)."""
    p = cfg.storage("results", fname)
    if not p.exists():
        return
    df = pd.read_csv(p)
    if "intervention" in df.columns:
        m = df["intervention"] == intervention
        if section and "section" in df.columns:
            secs = [section] if isinstance(section, str) else list(section)
            m = m & (df["section"].isin(secs))
        df[~m].to_csv(p, index=False)


def run(cfg, force: bool = False, intervention: str = "fluids_sepsis"):
    cfg.require("estimator.method", "estimator.cross_fitting_folds",
                "bootstrap.n_resamples", "run.seed",
                f"interventions.{intervention}.rct_reference.risk_difference")
    t0 = time.time()

    full = pd.read_parquet(cfg.storage("cohorts", f"{intervention}.parquet"))
    full = full[full["arm"].notna()].reset_index(drop=True)   # all eligible (§1 larger cohort)
    cohort = full[full["all_modality"]].reset_index(drop=True)  # primary shared cohort
    A = (cohort["arm"] == "active").astype(int).to_numpy()
    Y = cohort["outcome"].astype(int).to_numpy()
    subj = cohort["subject_id"].to_numpy()
    horizon = int(cfg.get(f"interventions.{intervention}.horizon_days"))
    D_obs = features.observed_at_horizon(cfg, cohort, horizon)   # §4 censoring indicator (IPCW)
    log(f"estimate[{intervention}]: all-modality cohort n={len(cohort):,} "
        f"(active={A.sum():,}, comparator={(A==0).sum():,}, deaths={Y.sum():,})")
    nc = features.negative_control_uti(cfg, cohort)     # §6.4 negative-control outcome
    baseline_pct = float(Y[A == 0].mean() * 100)        # comparator mortality, for E-values
    log(f"  negative control (icu_acquired_uti): {int(nc.sum())} positives; "
        f"baseline mortality {baseline_pct:.1f}%")

    seed = int(cfg.get("run.seed", 42))
    folds = int(cfg.get("estimator.cross_fitting_folds", 5))

    # ---- features (all strictly pre-t0) ----
    Sframe = features.structured_at_t0(cfg, cohort)
    S = Sframe.to_numpy(dtype=float)
    Ximg = features.pool_embeddings(cfg, cohort, "images")
    Xnote = features.pool_embeddings(cfg, cohort, "notes", "notes_clinical")   # B1: notes_clinical is primary
    # fix-variant only: compress embeddings so they don't break propensity overlap
    if cfg.reduction != "none":
        Ximg = reduce.apply(Ximg, cfg.reduction, A, Y, folds, seed, cfg.pca_components)
        Xnote = reduce.apply(Xnote, cfg.reduction, A, Y, folds, seed, cfg.pca_components)
        log(f"  embedding reduction '{cfg.reduction}': "
            f"image->{Ximg.shape[1]}d, notes->{Xnote.shape[1]}d")
    conditions = {
        "naive": np.ones((len(Y), 1)),
        "structured": S,
        "plus_notes": np.hstack([S, Xnote]),
        "plus_imaging_only": np.hstack([S, Ximg]),
        "full": np.hstack([S, Xnote, Ximg]),
    }
    # design_based (§5): expert-curated confounder set (no embeddings)
    exp_names = cfg.get(f"interventions.{intervention}.expert_confounders") or []
    db_frame = None
    if exp_names:
        db_frame = features.expert_features(cfg, cohort, exp_names)
        conditions["design_based"] = db_frame.to_numpy(dtype=float)

    # ---- shared bootstrap indices (patient-level, reused across conditions) ----
    nboot = int(cfg.get("bootstrap.n_resamples", 10000))
    boot = list(cluster_bootstrap_indices(subj, nboot, seed))

    ref = cfg.get(f"interventions.{intervention}.rct_reference")
    ref_rd = ref["risk_difference"]; ref_lo, ref_hi = ref["ci"]

    eff_rows, coh_rows, ctrl_rows = [], [], []
    boot_by_cond, point_by_cond = {}, {}
    psi_by_cond, keep_by_cond, e_by_cond = {}, {}, {}
    for name, X in conditions.items():
        psi, keep, diag = aipw.crossfit_aipw(X, A, Y, folds, seed, D=D_obs)
        point = float(psi[keep].mean() * 100)
        bvals = [psi[b][keep[b]].mean() * 100 for b in boot]
        boot_by_cond[name] = np.asarray(bvals, dtype=float)
        point_by_cond[name] = point
        psi_by_cond[name] = psi; keep_by_cond[name] = keep; e_by_cond[name] = diag["e"]
        eff = bootstrap_summary(point, bvals)
        _, if_lo, if_hi = influence_function_ci(psi, keep)      # §8 IF CI, a check
        bias = bootstrap_summary(point - ref_rd, [v - ref_rd for v in bvals])
        inside = bool(ref_lo <= point <= ref_hi)

        # negative control: identical AIPW on the UTI outcome (should be ~null)
        npsi, nkeep, _ = aipw.crossfit_aipw(X, A, nc, folds, seed)
        nc_point = float(npsi[nkeep].mean() * 100)
        nc_eff = bootstrap_summary(nc_point, [npsi[b][nkeep[b]].mean() * 100 for b in boot])
        ctrl_rows.append({
            "intervention": intervention, "condition": name,
            **nc_eff.as_row("negative_control_"),
            "e_value": e_value(point, baseline_pct),
            "e_value_ci_limit": e_value_ci_limit(eff.ci_low, eff.ci_high, baseline_pct),
        })

        eff_rows.append({
            "intervention": intervention, "condition": name, "cohort": "all_modality",
            "dataset": "mimic", "method": "aipw" if name != "naive" else "unadjusted",
            **eff.as_row("effect_"),
            "effect_if_ci_low": if_lo, "effect_if_ci_high": if_hi,
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

    # ---- §2/§6 minimum detectable effect (null interventions) ----
    if cfg.get(f"interventions.{intervention}.rct_reference.target_type") in (None, "null"):
        mde = minimum_detectable_effect(psi_by_cond["structured"], keep_by_cond["structured"])
        if mde is not None:
            coh_rows.append({"intervention": intervention, "section": "mde",
                             "metric": "min_detectable_rd_pp__structured", "stratum": "",
                             "arm": "", "value": mde, "support_count": int(keep_by_cond["structured"].sum())})
            log(f"  minimum detectable RD (structured, 80% power): {mde:.2f} pp")

    # ---- B4: design_based confounder completeness (extracted / named) ----
    if exp_names:
        extracted = db_frame.shape[1] if db_frame is not None else 0
        coh_rows.append({"intervention": intervention, "section": "design_based",
                         "metric": "expert_confounders_extracted_over_named", "stratum": "",
                         "arm": "", "value": f"{extracted}/{len(exp_names)}",
                         "support_count": len(cohort)})
        log(f"  design_based completeness: {extracted}/{len(exp_names)} named confounders extracted")

    # ---- §4 censoring diagnostic (IPCW): fraction with unknown status at horizon ----
    frac_cens = float((D_obs == 0).mean())
    coh_rows.append({"intervention": intervention, "section": "censoring",
                     "metric": f"frac_censored_at_{horizon}d", "stratum": "", "arm": "",
                     "value": round(frac_cens, 4), "support_count": len(D_obs)})
    log(f"  censoring at {horizon}d: {frac_cens*100:.2f}% (IPCW; MIMIC dod gives complete follow-up)")

    # ---- §7.6 covariate balance before/after weighting (structured set) ----
    e_s = np.clip(e_by_cond["structured"], 1e-6, 1 - 1e-6)
    w_ipw = np.where(A == 1, 1.0 / e_s, 1.0 / (1.0 - e_s))
    smd_before = standardized_mean_differences(S, A)
    smd_after = standardized_mean_differences(S, A, weights=w_ipw)
    for j, col in enumerate(Sframe.columns):
        for tag, arr in [("smd_before", smd_before), ("smd_after", smd_after)]:
            v = arr[j]
            coh_rows.append({"intervention": intervention, "section": "balance",
                             "metric": f"{tag}__{col}", "stratum": "", "arm": "",
                             "value": round(float(v), 3) if np.isfinite(v) else "",
                             "support_count": len(A)})

    # ---- §6.6 missingness: imaged (all-modality) vs non-imaged eligible patients ----
    nonimg = full[~full["all_modality"]].reset_index(drop=True)
    for stratum, grp in [("imaged", cohort), ("non_imaged", nonimg)]:
        if not len(grp):
            continue
        for metric, val in [("age_mean", round(float(grp["age_t0"].mean()), 1)),
                            ("sex_male_frac", round(float((grp["sex"] == "M").mean()), 3)),
                            ("mortality_frac", round(float(grp["outcome"].mean()), 3)),
                            ("active_arm_frac", round(float((grp["arm"] == "active").mean()), 3))]:
            coh_rows.append({"intervention": intervention, "section": "missingness",
                             "metric": metric, "stratum": stratum, "arm": "",
                             "value": val, "support_count": int(len(grp))})

    # ---- §7.6 demographic composition by sex and age band, per arm (all-modality) ----
    import re as _re
    def _band(a):
        for b in cfg.get("demographics.age_bands", []):
            m = _re.match(r"\(([\d.]+),\s*([\d.]+)\]", str(b))
            if m and float(m.group(1)) < a <= float(m.group(2)):
                return b
        return None
    coh = cohort.copy()
    coh["_band"] = coh["age_t0"].map(_band)
    for armname in ["active", "comparator"]:
        sub = coh[coh["arm"] == armname]
        for s in cfg.get("demographics.sex_levels", ["F", "M"]):
            coh_rows.append({"intervention": intervention, "section": "demographic_composition",
                             "metric": "sex_count", "stratum": s, "arm": armname,
                             "value": int((sub["sex"] == s).sum()), "support_count": int(len(sub))})
        for b in cfg.get("demographics.age_bands", []):
            coh_rows.append({"intervention": intervention, "section": "demographic_composition",
                             "metric": "age_band_count", "stratum": b, "arm": armname,
                             "value": int((sub["_band"] == b).sum()), "support_count": int(len(sub))})

    # notes_all (radiology-inclusive) conditions for the decomposition only (§6.3):
    # primary is notes_clinical (B1); notes_all is the second decomposition variant.
    # Saved to the boot file only; not primary effect rows.
    Xnote_a = features.pool_embeddings(cfg, cohort, "notes", "notes_all")
    if cfg.reduction != "none":
        Xnote_a = reduce.apply(Xnote_a, cfg.reduction, A, Y, folds, seed, cfg.pca_components)
    for cname, Xc in [("plus_notes_all", np.hstack([S, Xnote_a])),
                      ("full_all", np.hstack([S, Xnote_a, Ximg]))]:
        cpsi, ckeep, _ = aipw.crossfit_aipw(Xc, A, Y, folds, seed, D=D_obs)
        boot_by_cond[cname] = np.asarray([cpsi[b][ckeep[b]].mean() * 100 for b in boot])
        point_by_cond[cname] = float(cpsi[ckeep].mean() * 100)

    # ---- B (§1): additionally report structured/notes on the LARGER cohort ----
    # "You may additionally report the structured and notes conditions on the larger
    # cohort, flagged as such" (§1). Flagged via the `cohort` column; imaging/full
    # conditions require imaging so they stay on the shared all-modality cohort only.
    for scope, smask, names in [
        ("eligible", full["subject_id"].notna().to_numpy(), ["naive", "structured"]),
        ("has_note", full["has_pre_t0_note"].to_numpy(), ["naive", "structured", "plus_notes"]),
    ]:
        sub = full[smask].reset_index(drop=True)
        if len(sub) <= len(cohort):
            continue                                   # not actually larger; skip
        sA = (sub["arm"] == "active").astype(int).to_numpy()
        sY = sub["outcome"].astype(int).to_numpy()
        if sA.min() == sA.max():
            continue
        sS = features.structured_at_t0(cfg, sub).to_numpy(dtype=float)
        scond = {"naive": np.ones((len(sY), 1)), "structured": sS}
        if "plus_notes" in names:
            sN = features.pool_embeddings(cfg, sub, "notes", "notes_clinical")   # B1: notes_clinical primary
            if cfg.reduction != "none":
                sN = reduce.apply(sN, cfg.reduction, sA, sY, folds, seed, cfg.pca_components)
            scond["plus_notes"] = np.hstack([sS, sN])
        sboot = list(cluster_bootstrap_indices(sub["subject_id"].to_numpy(), nboot, seed))
        log(f"  [B §1] scope={scope}: n={len(sub):,} (vs all_modality {len(cohort):,})")
        for name in names:
            psi, keep, diag = aipw.crossfit_aipw(scond[name], sA, sY, folds, seed)
            pt = float(psi[keep].mean() * 100)
            bt = [psi[b][keep[b]].mean() * 100 for b in sboot]
            e = bootstrap_summary(pt, bt)
            _, sif_lo, sif_hi = influence_function_ci(psi, keep)
            bi = bootstrap_summary(pt - ref_rd, [v - ref_rd for v in bt])
            eff_rows.append({
                "intervention": intervention, "condition": name, "cohort": scope,
                "dataset": "mimic", "method": "aipw" if name != "naive" else "unadjusted",
                **e.as_row("effect_"), "effect_if_ci_low": sif_lo, "effect_if_ci_high": sif_hi,
                "ref_rd": ref_rd, "ref_ci_low": ref_lo, "ref_ci_high": ref_hi,
                **bi.as_row("bias_"), "inside_reference_ci": bool(ref_lo <= pt <= ref_hi),
                "ci_width": round(e.ci_high - e.ci_low, 1), "effective_sample_size": round(diag["ess"])})
            coh_rows.append({"intervention": intervention, "section": "overlap_ess",
                             "metric": f"ess__{name}@{scope}", "stratum": "", "arm": "",
                             "value": round(diag["ess"]), "support_count": diag["n"]})
            # §6.6 imaged-subset sensitivity: does restricting to imaged shift `structured`?
            if name == "structured":
                shift = round(pt - point_by_cond["structured"], 1)
                coh_rows.append({"intervention": intervention, "section": "missingness",
                                 "metric": f"structured_shift_imaged_minus_{scope}", "stratum": "",
                                 "arm": "", "value": shift, "support_count": len(sub)})
            log(f"    {scope}/{name}: RD={pt:.1f} ESS={round(diag['ess'])}")

    # persist per-condition bootstrap replicates (paired; needed by consolidate for
    # dissociation/decomposition/comparisons). Not a §7 file -- underscore-prefixed.
    all_conds = list(boot_by_cond)            # 5 primary + notes_clinical sensitivity pair
    np.savez(cfg.storage("results", f"_boot_{intervention}.npz"),
             ref_rd=float(ref_rd), baseline_pct=float(baseline_pct),
             point=np.array([point_by_cond[c] for c in all_conds]),
             conditions=np.array(all_conds), **boot_by_cond)

    _reset(cfg, "effects.csv", intervention)
    _reset(cfg, "controls.csv", intervention)
    _reset(cfg, "cohorts.csv", intervention,
           section=["overlap_ess", "mde", "balance", "missingness", "censoring",
                    "design_based", "demographic_composition"])
    R.append_rows(cfg, "effects.csv", eff_rows)
    R.append_rows(cfg, "controls.csv", ctrl_rows)
    R.append_rows(cfg, "cohorts.csv", coh_rows)
    log(f"estimate[{intervention}] done in {time.time()-t0:,.0f}s -> effects.csv, controls.csv, "
        f"cohorts.csv. reference RD={ref_rd} [{ref_lo},{ref_hi}] pp. design_based deferred.")
