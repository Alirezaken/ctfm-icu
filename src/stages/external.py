"""§9.12 / §6.8  external -- rerun the `structured` condition on eICU-CRD v2.0.

External replication (§8): re-estimate each intervention's effect on a *different*
ICU database, using the SAME estimand (eligibility, two strategies, mortality risk
difference at the trial horizon) and the SAME estimator (cross-fitted AIPW, LightGBM
nuisance, patient-level cluster bootstrap, same seed/indices count). Rows are written
to effects.csv with dataset=eicu so they sit beside the MIMIC rows.

eICU has no linked imaging and (here) no embedded notes, so only the `structured`
condition is run; `plus_notes` stays pending an eICU note-embedding extraction
(config external.conditions lists both). If a modality/arm is too thin to estimate,
the intervention is logged and skipped rather than faked (no fabrication, §0).

OPERATIONALIZATION (documented design choices, analyst -> for Soroosh review; the
estimand is fixed, only its eICU mapping is chosen here):
  - Spine: one ICU unit stay per patient (uniquepid, first stay). Offsets are minutes
    from unit admit. Age ">89"->90; drop age<=0/empty (§1).
  - Outcome: in-hospital mortality (hospitaldischargestatus=Expired) occurring within
    the trial horizon of t0. eICU is in-hospital only, so post-discharge death is not
    observed -- a censoring caveat, noted; horizons here (28/90d) mostly precede
    discharge for decedents.
  - Structured covariates at t0: age, sex, admission weight + APACHE first-day
    physiology (heart rate, MAP, resp rate, temp, WBC, sodium, pH, hematocrit,
    creatinine, albumin, pO2, pCO2, BUN, glucose, bilirubin, FiO2). Treatment-derived
    APACHE fields (intubated/vent/dialysis) are EXCLUDED to avoid arm leakage.
  - Eligibility / arms per intervention: eICU diagnosis strings, treatment strings,
    and labs, mirroring each config definition (see the per-intervention builders).
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd

from src.util import log, die
from src import aipw, results as R
from src.stats import cluster_bootstrap_indices, bootstrap_summary

# APACHE first-day physiology used as the structured set (no treatment-derived
# fields: intubated/vent/dialysis/meds/urine/GCS-components are excluded).
_APS_PHYS = ["heartrate", "meanbp", "respiratoryrate", "temperature", "wbc", "sodium",
             "ph", "hematocrit", "creatinine", "albumin", "pao2", "pco2", "bun",
             "glucose", "bilirubin", "fio2"]


def _eicu(cfg, name, usecols=None):
    return pd.read_csv(cfg.input("eicu_dir") / f"{name}.csv.gz",
                       usecols=usecols, low_memory=False)


def _spine(cfg):
    """One ICU stay per patient with adult age, weight, sex, in-hospital death timing."""
    p = _eicu(cfg, "patient", usecols=[
        "patientunitstayid", "uniquepid", "gender", "age", "admissionweight",
        "hospitaldischargeoffset", "hospitaldischargestatus"])
    age = pd.to_numeric(p["age"].replace(">89", "90"), errors="coerce")
    p = p[age > 0].copy()
    p["age_t0"] = age[age > 0].values
    p["sex_male"] = (p["gender"] == "Male").astype(float)
    p["weight_kg"] = pd.to_numeric(p["admissionweight"], errors="coerce")
    p["died_hosp"] = (p["hospitaldischargestatus"] == "Expired").astype(int)
    p["death_offset"] = pd.to_numeric(p["hospitaldischargeoffset"], errors="coerce")
    # patient-level (§1): keep one unit stay per patient
    p = p.sort_values("patientunitstayid").drop_duplicates("uniquepid", keep="first")
    return p.reset_index(drop=True)


def _aps(cfg):
    """APACHE first-day physiology per stay (eICU uses -1 for missing)."""
    a = _eicu(cfg, "apacheApsVar", usecols=["patientunitstayid"] + _APS_PHYS)
    return a.replace(-1, np.nan).set_index("patientunitstayid")


def _lab(cfg, labnames):
    l = _eicu(cfg, "lab", usecols=["patientunitstayid", "labresultoffset", "labname", "labresult"])
    l = l[l["labname"].isin(labnames)].copy()
    l["labresult"] = pd.to_numeric(l["labresult"], errors="coerce")
    return l.dropna(subset=["labresult"])


def _match(cfg, table, col, offcol, keys):
    """Rows of an eICU table whose lowered `col` contains any of `keys` substrings."""
    df = _eicu(cfg, table, usecols=["patientunitstayid", offcol, col])
    s = df[col].astype(str).str.lower()
    m = pd.Series(False, index=df.index)
    for k in keys:
        m |= s.str.contains(k, na=False, regex=False)
    return df[m].copy()


# ----------------------------------------------------------------- builders
# each returns a cohort frame: patientunitstayid, arm ('active'/'comparator'), t0_offset

def _b_transfusion(cfg, spine):
    """Restrictive (transfuse at Hb<=7) vs liberal (Hb<=9) in septic shock, Hb<9."""
    shock = set(_match(cfg, "diagnosis", "diagnosisstring", "diagnosisoffset",
                       ["septic shock", "sepsis with shock"])["patientunitstayid"])
    hb = _lab(cfg, ["Hgb"])
    low = hb[hb["labresult"] < 9.0]
    t0 = low.groupby("patientunitstayid")["labresultoffset"].min()          # first Hb<9
    elig = t0.index[t0.index.isin(shock)]
    tx = _match(cfg, "treatment", "treatmentstring", "treatmentoffset",
                ["packed red blood cell", "prbc", "transfusion of blood product"])
    first_tx = tx.groupby("patientunitstayid")["treatmentoffset"].min()
    rows = []
    for sid in elig:
        t0off = float(t0[sid])
        if sid in first_tx.index:
            txoff = float(first_tx[sid])
            pre = hb[(hb["patientunitstayid"] == sid) & (hb["labresultoffset"] <= txoff)]["labresult"]
            hb_at_tx = pre.min() if len(pre) else np.nan
            arm = "active" if (np.isnan(hb_at_tx) or hb_at_tx <= 7.0) else "comparator"
        else:
            arm = "active"                                                   # never transfused = restrictive-consistent
        rows.append((sid, arm, t0off))
    return pd.DataFrame(rows, columns=["patientunitstayid", "arm", "t0_offset"])


def _b_rrt(cfg, spine):
    """Accelerated (dialysis <=12h of t0) vs standard/delayed, in AKI stage 2-3."""
    aki = _match(cfg, "diagnosis", "diagnosisstring", "diagnosisoffset",
                 ["acute renal failure", "acute kidney injury", "acute renal insufficiency"])
    t0 = aki.groupby("patientunitstayid")["diagnosisoffset"].min()          # first AKI dx
    # severity gate: creatinine >=2 (proxy for KDIGO 2-3)
    cr = _lab(cfg, ["creatinine"])
    severe = set(cr[cr["labresult"] >= 2.0]["patientunitstayid"])
    elig = t0.index[t0.index.isin(severe)]
    dial = _match(cfg, "treatment", "treatmentstring", "treatmentoffset",
                  ["dialysis", "crrt", "renal replacement"])
    first_d = dial.groupby("patientunitstayid")["treatmentoffset"].min()
    rows = []
    for sid in elig:
        t0off = float(t0[sid])
        if sid in first_d.index and (float(first_d[sid]) - t0off) <= 720:   # 12h
            arm = "active"
        else:
            arm = "comparator"                                              # delayed or never = standard
        rows.append((sid, arm, t0off))
    return pd.DataFrame(rows, columns=["patientunitstayid", "arm", "t0_offset"])


def _b_fluids(cfg, spine):
    """Restrictive (early vasopressor) vs liberal (fluid resuscitation, no early pressor)
    in suspected sepsis with hypotension. t0 = unit admit (eligibility is <=24h of admit).
    Crystalloid mL/kg is not reconstructable cleanly in eICU, so the arm proxy is
    vasopressor-first (restrictive) vs fluid-first (liberal) -- a documented proxy."""
    sepsis = set(_match(cfg, "diagnosis", "diagnosisstring", "diagnosisoffset",
                        ["sepsis", "septic"])["patientunitstayid"])
    vaso = _match(cfg, "treatment", "treatmentstring", "treatmentoffset",
                  ["vasopressor", "norepinephrine", "vasopressin", "phenylephrine",
                   "epinephrine", "dopamine"])
    fluid = _match(cfg, "treatment", "treatmentstring", "treatmentoffset",
                   ["fluid resuscitation", "normal saline", "lactated ringer", "fluid bolus",
                    "isotonic"])
    v_first = vaso.groupby("patientunitstayid")["treatmentoffset"].min()
    f_first = fluid.groupby("patientunitstayid")["treatmentoffset"].min()
    elig = sepsis & (set(v_first.index) | set(f_first.index))
    rows = []
    for sid in elig:
        vo = float(v_first[sid]) if sid in v_first.index else np.inf
        fo = float(f_first[sid]) if sid in f_first.index else np.inf
        if not np.isfinite(vo) and not np.isfinite(fo):
            continue
        arm = "active" if vo <= fo else "comparator"                        # pressor-first = restrictive
        rows.append((sid, arm, 0.0))
    return pd.DataFrame(rows, columns=["patientunitstayid", "arm", "t0_offset"])


def _b_prone(cfg, spine):
    """Prone within 24h vs supine, in severe ARDS (low P/F). Prone is rare in eICU,
    so this cohort is expected to be small (an honest external limit)."""
    ards = _match(cfg, "diagnosis", "diagnosisstring", "diagnosisoffset",
                  ["ards", "acute respiratory distress", "acute lung injury"])
    t0 = ards.groupby("patientunitstayid")["diagnosisoffset"].min()
    aps = _aps(cfg)
    pf = aps["pao2"] / aps["fio2"].where(aps["fio2"] <= 1, aps["fio2"] / 100.0)
    severe = set(pf.index[pf < 150])
    elig = t0.index[t0.index.isin(severe)]
    prone = _match(cfg, "treatment", "treatmentstring", "treatmentoffset", ["prone"])
    first_p = prone.groupby("patientunitstayid")["treatmentoffset"].min()
    rows = []
    for sid in elig:
        t0off = float(t0[sid])
        arm = "active" if (sid in first_p.index and (float(first_p[sid]) - t0off) <= 1440) else "comparator"
        rows.append((sid, arm, t0off))
    return pd.DataFrame(rows, columns=["patientunitstayid", "arm", "t0_offset"])


_BUILDERS = {
    "transfusion_threshold": _b_transfusion,
    "rrt_timing": _b_rrt,
    "fluids_sepsis": _b_fluids,
    "prone_positioning": _b_prone,
}


def _structured_matrix(cohort, spine, aps):
    """age/sex/weight + APACHE physiology, aligned to cohort rows."""
    s = spine.set_index("patientunitstayid").reindex(cohort["patientunitstayid"])
    X = pd.DataFrame(index=cohort.index)
    X["age"] = s["age_t0"].values
    X["sex_male"] = s["sex_male"].values
    X["weight_kg"] = s["weight_kg"].values
    a = aps.reindex(cohort["patientunitstayid"])
    for c in _APS_PHYS:
        X[c] = a[c].values
    return X


def _one(cfg, intervention, spine, aps, seed, folds, nboot):
    iv = cfg.get(f"interventions.{intervention}")
    horizon = int(iv["horizon_days"]) * 1440.0
    ref = iv["rct_reference"]; ref_rd = ref["risk_difference"]; ref_lo, ref_hi = ref["ci"]

    coh = _BUILDERS[intervention](cfg, spine).drop_duplicates("patientunitstayid")
    coh = coh.merge(spine[["patientunitstayid", "uniquepid", "died_hosp", "death_offset"]],
                    on="patientunitstayid", how="inner").reset_index(drop=True)
    if len(coh) < 50:
        log(f"  {intervention}: eICU cohort too small (n={len(coh)}); skip"); return None

    # outcome: in-hospital death within horizon of t0
    within = (coh["death_offset"] - coh["t0_offset"]) <= horizon
    Y = ((coh["died_hosp"] == 1) & within).astype(int).to_numpy()
    A = (coh["arm"] == "active").astype(int).to_numpy()
    if A.min() == A.max() or Y.sum() < 10 or (A == 0).sum() < 20 or (A == 1).sum() < 20:
        log(f"  {intervention}: eICU arms/events too thin "
            f"(n={len(coh)}, active={A.sum()}, deaths={Y.sum()}); skip"); return None

    X = _structured_matrix(coh, spine, aps).to_numpy(dtype=float)
    psi, keep, diag = aipw.crossfit_aipw(X, A, Y, folds, seed)
    point = float(psi[keep].mean() * 100)
    boot = list(cluster_bootstrap_indices(coh["uniquepid"].to_numpy(), nboot, seed))
    bvals = [psi[b][keep[b]].mean() * 100 for b in boot]
    eff = bootstrap_summary(point, bvals)
    bias = bootstrap_summary(point - ref_rd, [v - ref_rd for v in bvals])
    log(f"  {intervention}: eICU n={len(coh)} active={A.sum()} deaths={Y.sum()} "
        f"RD={point:.1f} [{eff.ci_low:.1f},{eff.ci_high:.1f}] ESS={round(diag['ess'])} "
        f"inside_ref={ref_lo <= point <= ref_hi}")
    return {
        "intervention": intervention, "condition": "structured", "cohort": "eicu_all",
        "dataset": "eicu", "method": "aipw",
        **eff.as_row("effect_"), "ref_rd": ref_rd, "ref_ci_low": ref_lo, "ref_ci_high": ref_hi,
        **bias.as_row("bias_"), "inside_reference_ci": bool(ref_lo <= point <= ref_hi),
        "negative_control_point": "", "negative_control_mean": "", "negative_control_std": "",
        "negative_control_ci_low": "", "negative_control_ci_high": "",
        "ci_width": round(eff.ci_high - eff.ci_low, 1),
        "effective_sample_size": round(diag["ess"]),
    }


def run(cfg, force: bool = False):
    eicu = cfg.input("eicu_dir")
    if not eicu.exists():
        die(f"eICU not present at {eicu}. Upload eICU-CRD v2.0 first.")
    t0 = time.time()
    seed = int(cfg.get("run.seed", 42))
    folds = int(cfg.get("estimator.cross_fitting_folds", 5))
    nboot = int(cfg.get("bootstrap.n_resamples", 10000))
    log(f"external[eICU]: structured replication, seed={seed} folds={folds} boot={nboot}")

    spine = _spine(cfg)
    aps = _aps(cfg)
    log(f"  eICU spine: {len(spine):,} patients; APACHE physiology for {len(aps):,} stays")

    rows = []
    for iv in _BUILDERS:
        r = _one(cfg, iv, spine, aps, seed, folds, nboot)
        if r:
            rows.append(r)

    # idempotent: drop prior eICU rows, keep MIMIC rows, append fresh
    p = cfg.storage("results", "effects.csv")
    if p.exists():
        df = pd.read_csv(p)
        if "dataset" in df.columns:
            df[df["dataset"] != "eicu"].to_csv(p, index=False)
    R.append_rows(cfg, "effects.csv", rows)
    log(f"external[eICU] done in {time.time()-t0:,.0f}s -> {len(rows)} structured rows "
        f"(dataset=eicu). plus_notes pending eICU note embeddings.")
