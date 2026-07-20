"""estimate -- the main effects. Seven adjustment conditions per intervention.

  naive            unadjusted
  expert           clinician-curated structured confounders (the design-based competitor)
  structured       all extractable structured covariates at t0   <-- THE BASELINE that every
                                                                     modality condition is
                                                                     contrasted against
  struct_img       structured + RAD-DINO chest X-ray embedding
  struct_radtext   structured + Clinical-Longformer on radiology reports
  struct_histnote  structured + Clinical-Longformer on prior-admission discharge summaries
  multimodal       structured + all three

Reported per condition: the ATE (risk difference, percentage points) with a paired
cluster-bootstrap CI, the influence-function CI as a check, the overlap-weighted ATO, the
bias against the RCT anchor, the divergence Z, the overlap diagnostics that determine
whether any of it means anything, and a negative-control effect.

WHY BOTH ATE AND ATO. Adding high-dimensional covariates makes treatment near-deterministic
for some patients; the propensity mass piles up at the boundaries and the ATE gets fragile
exactly where this study is asking it to be precise. The ATO weights by e(1-e), is bounded,
and targets the clinical-equipoise population. Reporting both means the headline null cannot
be waved away as an artifact of a poorly-supported ATE.

WHY NOT `inside_reference_ci`. STARRT-AKI's published interval is 0.7 pp wide. No
observational estimate will ever land inside it, so a pass/fail on containment carries no
information. `divergence_z` accounts for the uncertainty in BOTH the emulation and the
trial, and is the statistic the emulation-benchmarking literature actually uses.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd

from src.util import log
from src import events as ev
from src import features as F
from src import estimator as EST
from src import results as R
from src.stats import (cluster_bootstrap_indices, bootstrap_summary, influence_function_ci,
                       divergence_z, ci_overlaps, minimum_detectable_effect,
                       standardized_mean_differences, e_value, e_value_ci_limit, round3)


def _guard_impossible(cfg, row: dict) -> dict:
    """A risk difference outside +/-100pp is not wide, it is impossible (Part 10 Rule 3).
    Mirrors robustness.py's subgroup guard: blank the point/CI/bias/derived fields and
    mark `undefined` rather than print a number that cannot exist."""
    max_rd = float(cfg.get("demographics.max_abs_rd_pp", 100))
    pt, lo, hi = row["effect_point"], row["effect_ci_low"], row["effect_ci_high"]
    if not (abs(pt) > max_rd or abs(lo) > max_rd or abs(hi) > max_rd):
        row["undefined"], row["note"] = False, ""
        return row
    for k in ("effect_point", "effect_mean", "effect_std", "effect_ci_low", "effect_ci_high",
              "effect_if_ci_low", "effect_if_ci_high",
              "bias_point", "bias_mean", "bias_std", "bias_ci_low", "bias_ci_high",
              "divergence_z", "e_value", "e_value_ci_limit"):
        row[k] = ""
    row["undefined"] = True
    row["note"] = f"estimate/CI exceeds +/-{max_rd:.0f}pp -- not estimable"
    return row


def build_conditions(cfg, cohort, S):
    """The seven design matrices. Embedding blocks are passed RAW; the reduction happens
    inside the estimator's cross-fitting loop (nested), never here."""
    n = len(cohort)
    blocks = {}
    for mod in ("images", "radtext", "histnote"):
        blocks[mod] = F.modality_block(cfg, cohort, mod)

    return {
        "naive":           (np.ones((n, 1)), []),
        "structured":      (S, []),
        "struct_img":      (S, [blocks["images"]]),
        "struct_radtext":  (S, [blocks["radtext"]]),
        "struct_histnote": (S, [blocks["histnote"]]),
        "multimodal":      (S, [blocks["images"], blocks["radtext"], blocks["histnote"]]),
    }, blocks


def run(cfg, force: bool = False, intervention: str = None):
    names = [intervention] if intervention else list(cfg.get("interventions") or {})
    for name in names:
        _run_one(cfg, name)


def _run_one(cfg, intervention):
    t_start = time.time()
    spec = cfg.get(f"interventions.{intervention}")
    ref = spec["rct_reference"]
    ref_rd = float(ref["risk_difference"])
    ref_lo, ref_hi = [float(v) for v in ref["ci"]]
    ref_src = ref.get("source")
    horizon = int(spec["horizon_days"])

    full = ev.load_cohorts(cfg, intervention)
    full = full[full["arm"].notna()].reset_index(drop=True)
    cohort = full[full["imaged"]].reset_index(drop=True)          # PRIMARY: the imaged cohort

    A = (cohort["arm"] == "active").astype(int).to_numpy()
    Y = cohort["outcome"].astype(int).to_numpy()
    subj = cohort["subject_id"].to_numpy()
    D = cohort["observed_at_horizon"].astype(int).to_numpy()

    log(f"=== estimate[{intervention}] === role={spec.get('role')}")
    log(f"  imaged cohort n={len(cohort):,}  active={A.sum():,}  "
        f"comparator={(A==0).sum():,}  deaths={Y.sum():,}  censored={100*(D==0).mean():.1f}%")

    if A.sum() < 10 or (A == 0).sum() < 10:
        log(f"  *** {intervention}: fewer than 10 patients in an arm -- positivity is "
            f"absent, not merely poor. Estimates will be reported but are NOT interpretable "
            f"as causal effects. This is the pre-specified positivity_failure_case.")

    Sframe = F.structured_at_t0(cfg, cohort)
    S = Sframe.to_numpy(dtype=float)
    nc = F.negative_control_uti(cfg, cohort)
    baseline_pct = float(Y[A == 0].mean() * 100) if (A == 0).any() else np.nan

    conds, blocks = build_conditions(cfg, cohort, S)

    # expert condition, with its extraction completeness recorded
    exp_names = spec.get("expert_confounders") or []
    Xexp, n_extracted, n_requested = F.expert_features(cfg, cohort, exp_names)
    conds["expert"] = (Xexp.to_numpy(dtype=float), [])

    order = cfg.get("conditions")
    boot = cluster_bootstrap_indices(subj, cfg.nboot, cfg.seed)

    eff_rows, coh_rows = [], []
    store = {}                       # condition -> everything consolidate/diagnose needs

    ess_structured = None
    for name in order:
        if name not in conds:
            continue
        X, blks = conds[name]
        psi, keep, diag = EST.crossfit_aipw(
            X, A, Y, cfg.folds, cfg.seed, blocks=blks, reduction=cfg.reduction,
            pca_components=cfg.pca_components, trim=cfg.trim, D=D)

        point = float(psi[keep].mean() * 100)
        bvals = np.array([psi[b][keep[b]].mean() * 100 for b in boot])
        eff = bootstrap_summary(point, bvals)
        _, if_lo, if_hi, se_obs = influence_function_ci(psi, keep)

        ato_pt = diag["ato"]
        ato_b = np.array([EST.ato_from_boot(diag["psi_ato"], diag["h_ato"], b) for b in boot])
        ato = bootstrap_summary(ato_pt, ato_b)

        bias = bootstrap_summary(point - ref_rd, bvals - ref_rd)
        zdiv = divergence_z(point, se_obs if se_obs else 0.0, ref_rd, (ref_lo, ref_hi))
        overlap = ci_overlaps((eff.ci_low, eff.ci_high), (ref_lo, ref_hi))

        if name == "structured":
            ess_structured = diag["ess"]

        # negative control, with the SAME censoring correction as the primary outcome
        npsi, nkeep, _ = EST.crossfit_aipw(
            X, A, nc, cfg.folds, cfg.seed, blocks=blks, reduction=cfg.reduction,
            pca_components=cfg.pca_components, trim=cfg.trim, D=D)
        nc_pt = float(npsi[nkeep].mean() * 100)
        nc_eff = bootstrap_summary(nc_pt, [npsi[b][nkeep[b]].mean() * 100 for b in boot])

        store[name] = {"psi": psi, "keep": keep, "boot": bvals, "point": point,
                       "ess": diag["ess"], "e": diag["e"]}

        eff_rows.append(_guard_impossible(cfg, {
            "intervention": intervention, "condition": name, "cohort": "imaged",
            "dataset": "mimic", "estimator": "aipw" if name != "naive" else "unadjusted",
            "reduction": cfg.reduction if blks else "",
            **eff.as_row("effect_"),
            "effect_if_ci_low": if_lo, "effect_if_ci_high": if_hi,
            **ato.as_row("ato_"),
            "ref_rd": ref_rd, "ref_ci_low": ref_lo, "ref_ci_high": ref_hi,
            "ref_source": str(ref_src),
            **bias.as_row("bias_"),
            "divergence_z": zdiv, "ci_overlaps_rct": overlap,
            "ci_width": round(eff.ci_high - eff.ci_low, 1),
            "effective_sample_size": round(diag["ess"]),
            "ess_ratio_vs_structured": (round(diag["ess"] / ess_structured, 3)
                                        if ess_structured else ""),
            "propensity_min": round3(diag["e_min"]), "propensity_max": round3(diag["e_max"]),
            "frac_trimmed": round3(diag["frac_trimmed"]),
            "frac_censored": round3(diag["frac_censored"]),
            "n_analyzed": diag["n"], "n_active": int(A.sum()),
            "n_comparator": int((A == 0).sum()), "n_events": int(Y.sum()),
            "min_detectable_effect_pp": minimum_detectable_effect(psi, keep),
            "expert_confounders_extracted": n_extracted if name == "expert" else "",
            "expert_confounders_requested": n_requested if name == "expert" else "",
            **nc_eff.as_row("negative_control_"),
            "e_value": e_value(point, baseline_pct),
            "e_value_ci_limit": e_value_ci_limit(eff.ci_low, eff.ci_high, baseline_pct),
        }))

        log(f"  {name:16s} RD={point:7.1f}  CI[{eff.ci_low:6.1f},{eff.ci_high:6.1f}]  "
            f"ATO={ato_pt:6.1f}  bias={point-ref_rd:+6.1f}  Z={zdiv}  ESS={round(diag['ess']):5d}")

    # ---- diagnostics: overlap, balance, missingness, censoring, demographics ----
    coh_rows += _cohort_diagnostics(cfg, intervention, cohort, full, Sframe, A, Y, D, store)

    # ---- larger-scope sensitivity: does the imaging gate itself move `structured`? ----
    eff_rows += _scope_sensitivity(cfg, intervention, full, cohort, ref_rd, ref_lo, ref_hi,
                                   ref_src, store, coh_rows)

    # persist everything consolidate/diagnose needs, paired bootstrap included
    np.savez(cfg.storage("results", f"_boot_{intervention}.npz"),
             ref_rd=ref_rd, baseline_pct=baseline_pct,
             conditions=np.array(list(store)),
             point=np.array([store[c]["point"] for c in store]),
             ess=np.array([store[c]["ess"] for c in store]),
             **{c: store[c]["boot"] for c in store})

    R.reset_rows(cfg, "effects.csv", intervention=intervention, dataset="mimic")
    R.reset_rows(cfg, "cohorts.csv", intervention=intervention,
                 section=["overlap", "balance", "missingness", "censoring",
                          "demographic_composition", "mde"])
    R.append_rows(cfg, "effects.csv", eff_rows)
    R.append_rows(cfg, "cohorts.csv", coh_rows)
    log(f"estimate[{intervention}] done in {time.time()-t_start:,.0f}s")


def _cohort_diagnostics(cfg, iv, cohort, full, Sframe, A, Y, D, store):
    rows = []

    def add(section, metric, value, stratum="", arm="", support=""):
        rows.append({"intervention": iv, "section": section, "metric": metric,
                     "stratum": stratum, "arm": arm, "value": value,
                     "support_count": support})

    for cond, st in store.items():
        add("overlap", f"ess__{cond}", round(st["ess"]), support=len(A))
    add("censoring", "frac_censored_at_horizon", round(float((D == 0).mean()), 4),
        support=len(D))

    # covariate balance before/after IPW on the structured set
    if "structured" in store:
        e = np.clip(store["structured"]["e"], 1e-6, 1 - 1e-6)
        w = np.where(A == 1, 1 / e, 1 / (1 - e))
        S = Sframe.to_numpy(dtype=float)
        for tag, arr in [("smd_before", standardized_mean_differences(S, A)),
                         ("smd_after", standardized_mean_differences(S, A, weights=w))]:
            for j, col in enumerate(Sframe.columns):
                v = arr[j]
                add("balance", f"{tag}__{col}",
                    round(float(v), 3) if np.isfinite(v) else "", support=len(A))

    # demographic composition by arm (required by the reporting contract)
    age = pd.to_numeric(cohort["age_t0"], errors="coerce").to_numpy()
    for armname, mask in [("active", A == 1), ("comparator", A == 0)]:
        if not mask.any():
            continue
        add("demographic_composition", "n", int(mask.sum()), arm=armname)
        add("demographic_composition", "age_mean", round(float(np.nanmean(age[mask])), 1),
            arm=armname)
        add("demographic_composition", "mortality_frac",
            round(float(Y[mask].mean()), 3), arm=armname)
        for s in cfg.get("demographics.sex_levels", ["F", "M"]):
            n = int(((cohort["sex"] == s).to_numpy() & mask).sum())
            add("demographic_composition", "sex_count", n, stratum=s, arm=armname)
        for b in cfg.get("demographics.age_bands", []):
            lo, hi = _band(b)
            n = int(((age > lo) & (age <= hi) & mask).sum())
            add("demographic_composition", "age_band_count", n, stratum=b, arm=armname)

    # missingness: imaged vs non-imaged eligible patients
    nonimg = full[~full["imaged"]].reset_index(drop=True)
    for stratum, grp in [("imaged", cohort), ("not_imaged", nonimg)]:
        if not len(grp):
            continue
        for metric, val in [
            ("n", int(len(grp))),
            ("age_mean", round(float(pd.to_numeric(grp["age_t0"], errors="coerce").mean()), 1)),
            ("sex_male_frac", round(float((grp["sex"] == "M").mean()), 3)),
            ("mortality_frac", round(float(grp["outcome"].mean()), 3)),
            ("active_arm_frac", round(float((grp["arm"] == "active").mean()), 3)),
            ("histnote_coverage", round(float(grp["has_pre_t0_histnote"].mean()), 3)),
        ]:
            add("missingness", metric, val, stratum=stratum, support=len(grp))

    if "structured" in store:
        add("mde", "min_detectable_rd_pp__structured",
            minimum_detectable_effect(store["structured"]["psi"], store["structured"]["keep"]),
            support=int(store["structured"]["keep"].sum()))
    return rows


def _scope_sensitivity(cfg, iv, full, cohort, ref_rd, ref_lo, ref_hi, ref_src, store, coh_rows):
    """Re-estimate `naive` and `structured` on the FULL eligible cohort (no imaging gate).

    This is how we show the imaging gate did not itself select a different causal question.
    If `structured` shifts materially between `eligible` and `imaged`, the gate is a
    confounder of the comparison and must be reported as such.
    """
    rows = []
    sub = full.reset_index(drop=True)
    if len(sub) <= len(cohort):
        return rows

    A = (sub["arm"] == "active").astype(int).to_numpy()
    Y = sub["outcome"].astype(int).to_numpy()
    D = sub["observed_at_horizon"].astype(int).to_numpy()
    if A.min() == A.max():
        return rows

    S = F.structured_at_t0(cfg, sub).to_numpy(dtype=float)
    boot = cluster_bootstrap_indices(sub["subject_id"].to_numpy(), cfg.nboot, cfg.seed)
    log(f"  [scope] eligible n={len(sub):,} (vs imaged {len(cohort):,})")

    for name, X in [("naive", np.ones((len(Y), 1))), ("structured", S)]:
        psi, keep, diag = EST.crossfit_aipw(X, A, Y, cfg.folds, cfg.seed,
                                            trim=cfg.trim, D=D)
        pt = float(psi[keep].mean() * 100)
        bt = np.array([psi[b][keep[b]].mean() * 100 for b in boot])
        e = bootstrap_summary(pt, bt)
        _, if_lo, if_hi, se = influence_function_ci(psi, keep)
        ato_b = np.array([EST.ato_from_boot(diag["psi_ato"], diag["h_ato"], b) for b in boot])
        ato = bootstrap_summary(diag["ato"], ato_b)
        bias = bootstrap_summary(pt - ref_rd, bt - ref_rd)

        rows.append(_guard_impossible(cfg, {
            "intervention": iv, "condition": name, "cohort": "eligible", "dataset": "mimic",
            "estimator": "aipw" if name != "naive" else "unadjusted", "reduction": "",
            **e.as_row("effect_"), "effect_if_ci_low": if_lo, "effect_if_ci_high": if_hi,
            **ato.as_row("ato_"),
            "ref_rd": ref_rd, "ref_ci_low": ref_lo, "ref_ci_high": ref_hi,
            "ref_source": str(ref_src),
            **bias.as_row("bias_"),
            "divergence_z": divergence_z(pt, se or 0.0, ref_rd, (ref_lo, ref_hi)),
            "ci_overlaps_rct": ci_overlaps((e.ci_low, e.ci_high), (ref_lo, ref_hi)),
            "ci_width": round(e.ci_high - e.ci_low, 1),
            "effective_sample_size": round(diag["ess"]),
            "propensity_min": round3(diag["e_min"]), "propensity_max": round3(diag["e_max"]),
            "frac_trimmed": round3(diag["frac_trimmed"]),
            "frac_censored": round3(diag["frac_censored"]),
            "n_analyzed": diag["n"], "n_active": int(A.sum()),
            "n_comparator": int((A == 0).sum()), "n_events": int(Y.sum()),
            "min_detectable_effect_pp": minimum_detectable_effect(psi, keep),
        }))
        if name == "structured" and "structured" in store:
            shift = round(store["structured"]["point"] - pt, 1)
            coh_rows.append({
                "intervention": iv, "section": "missingness",
                "metric": "structured_shift_imaged_minus_eligible", "stratum": "", "arm": "",
                "value": shift, "support_count": len(sub)})
            log(f"  [scope] gate shift on `structured`: {shift:+.1f} pp "
                f"(large => the imaging gate changed the causal question)")
    return rows


def _band(s):
    import re
    m = re.match(r"\(([\d.]+),\s*([\d.]+)\]", str(s))
    return (float(m.group(1)), float(m.group(2))) if m else (-np.inf, np.inf)
