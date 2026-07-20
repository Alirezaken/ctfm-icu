"""external -- eICU-CRD replication. Structured only. Writes to effects.csv.

Same target trials, same estimator, an entirely different population (208 US hospitals,
not one Boston academic centre). If `structured` behaves the same way there, the finding
is not a MIMIC quirk.

STRUCTURED ONLY, AND THAT IS NOT A GAP TO BE FILLED LATER.
The public eICU-CRD release has no linked imaging and no usable free text. There is no
eICU note-embedding stage to write. The previous plan carried `plus_notes` on eICU as
"pending a GPU job"; that job would have produced nothing, because the data does not
exist. Saying so is better than leaving a permanently-pending row in the results.

CENSORING. eICU stops at hospital discharge, so vital status at t0+horizon is known only
if the patient died in hospital within the horizon, or the stay itself reaches it. Someone
discharged alive on day 5 with a 90-day horizon is CENSORED, not a survivor. Handled by
IPCW, same as MIMIC.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd

from src.util import log, die
from src import results as R
from src.stats import cluster_bootstrap_indices, bootstrap_summary, influence_function_ci

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
    p["disp_known"] = p["hospitaldischargestatus"].isin(["Alive", "Expired"]).astype(int)
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


_IO_CACHE = {}


def _intake_output(cfg):
    """intakeOutput rows, read once (229 MB), for crystalloid volume and RBC transfusions."""
    key = str(cfg.input("eicu_dir"))
    if key not in _IO_CACHE:
        _IO_CACHE[key] = _eicu(cfg, "intakeOutput", usecols=[
            "patientunitstayid", "intakeoutputoffset", "cellpath", "cellvaluenumeric"])
    return _IO_CACHE[key]


def _crystalloid_ml_24h(cfg):
    """Cumulative crystalloid volume (mL) in the first 24 h, per stay (real I&O flowsheet)."""
    io = _intake_output(cfg)
    cr = io[io["cellpath"].str.contains("Crystalloids (ml)", na=False, regex=False)
            & (io["intakeoutputoffset"] >= 0) & (io["intakeoutputoffset"] <= 1440)]
    return cr.groupby("patientunitstayid")["cellvaluenumeric"].sum()


def _rbc_first_offset(cfg):
    """Offset (min) of the first red-cell transfusion per stay (I&O blood-products)."""
    io = _intake_output(cfg)
    rbc = io[io["cellpath"].str.contains("Transfuse red blood cells", na=False, regex=False)
             & (io["cellvaluenumeric"] > 0)]
    return rbc.groupby("patientunitstayid")["intakeoutputoffset"].min()


# ----------------------------------------------------------------- builders
# each returns a cohort frame: patientunitstayid, arm ('active'/'comparator'), t0_offset

def _b_transfusion(cfg, spine):
    """Restrictive (transfuse at Hb<=7) vs liberal (Hb<=9) in septic shock, Hb<9.
    Transfusions from the I&O blood-products flowsheet ('Transfuse red blood cells'),
    which captures far more transfusion events than the treatment strings."""
    shock = set(_match(cfg, "diagnosis", "diagnosisstring", "diagnosisoffset",
                       ["septic shock", "sepsis with shock"])["patientunitstayid"])
    hb = _lab(cfg, ["Hgb"])
    low = hb[hb["labresult"] < 9.0]
    t0 = low.groupby("patientunitstayid")["labresultoffset"].min()          # first Hb<9
    elig = t0.index[t0.index.isin(shock)]
    first_tx = _rbc_first_offset(cfg)
    rows = []
    for sid in elig:
        t0off = float(t0[sid])
        if sid in first_tx.index:
            txoff = float(first_tx[sid])
            pre = hb[(hb["patientunitstayid"] == sid) & (hb["labresultoffset"] <= txoff)]
            pre = pre.sort_values("labresultoffset")["labresult"]
            hb_at_tx = pre.iloc[-1] if len(pre) else np.nan     # trigger Hb (last before tx), not nadir
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
    """Restrictive vs liberal fluids in suspected sepsis with hypotension, by REAL
    cumulative crystalloid mL/kg in the first 24 h (I&O flowsheet 'Crystalloids (ml)'
    / admission weight) -- the same estimand as MIMIC. Threshold 30 mL/kg; if that
    splits worse than 70/30 use the cohort median (mirrors MIMIC Decision 1).
    t0 = unit admit (eligibility is within 24 h of admit)."""
    sepsis = set(_match(cfg, "diagnosis", "diagnosisstring", "diagnosisoffset",
                        ["sepsis", "septic"])["patientunitstayid"])
    aps = _aps(cfg)
    hypo = set(aps.index[aps["meanbp"] < 65])
    vaso = set(_match(cfg, "treatment", "treatmentstring", "treatmentoffset",
                      ["vasopressor", "norepinephrine", "vasopressin", "phenylephrine",
                       "epinephrine", "dopamine"])["patientunitstayid"])
    crys = _crystalloid_ml_24h(cfg)
    wt = spine.set_index("patientunitstayid")["weight_kg"]
    elig = sepsis & (hypo | vaso) & set(crys.index)
    mlkg = {}
    for sid in elig:
        w = wt.get(sid, np.nan)
        if w and w > 0 and np.isfinite(w):
            mlkg[sid] = float(crys[sid]) / float(w)
    if len(mlkg) < 50:
        return pd.DataFrame(columns=["patientunitstayid", "arm", "t0_offset"])
    vals = pd.Series(mlkg)
    thr = 30.0
    if not (0.30 <= (vals < thr).mean() <= 0.70):
        thr = float(vals.median())                                          # guarantee overlap
    rows = [(sid, "active" if v < thr else "comparator", 0.0) for sid, v in vals.items()]
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




def _one(cfg, intervention, spine, aps):
    """One eICU intervention: build, estimate `naive` and `structured`, return rows."""
    from src.stats import (cluster_bootstrap_indices, bootstrap_summary,
                           influence_function_ci, divergence_z, ci_overlaps,
                           minimum_detectable_effect, round3)
    from src import estimator as EST
    from src.stages import estimate as EE

    spec = cfg.get(f"interventions.{intervention}")
    horizon_min = int(spec["horizon_days"]) * 1440.0
    ref = spec["rct_reference"]
    ref_rd = float(ref["risk_difference"])
    ref_lo, ref_hi = [float(v) for v in ref["ci"]]

    coh = _BUILDERS[intervention](cfg, spine).drop_duplicates("patientunitstayid")
    coh = coh.merge(
        spine[["patientunitstayid", "uniquepid", "died_hosp", "death_offset", "disp_known"]],
        on="patientunitstayid", how="inner").reset_index(drop=True)
    if len(coh) < 50:
        log(f"  {intervention}: eICU cohort too small (n={len(coh)}); skip")
        return []

    # eICU has NO post-discharge follow-up. Status at the horizon is KNOWN only if the
    # patient died in hospital within it, or the stay itself reaches it. Discharged alive
    # before the horizon => status UNKNOWN => censored, handled by IPCW.
    stay_len = coh["death_offset"] - coh["t0_offset"]
    died_within = (coh["died_hosp"] == 1) & (stay_len <= horizon_min)
    Y = died_within.astype(int).to_numpy()
    A = (coh["arm"] == "active").astype(int).to_numpy()
    D = (died_within | (stay_len >= horizon_min)).astype(int).to_numpy()

    if A.min() == A.max() or Y.sum() < 10 or (A == 0).sum() < 20 or (A == 1).sum() < 20:
        log(f"  {intervention}: eICU arms/events too thin "
            f"(n={len(coh)}, active={A.sum()}, deaths={Y.sum()}); skip")
        return []

    S = np.nan_to_num(_structured_matrix(coh, spine, aps).to_numpy(dtype=float), nan=0.0)
    boot = cluster_bootstrap_indices(coh["uniquepid"].to_numpy(), cfg.nboot, cfg.seed)
    log(f"  {intervention}: n={len(coh):,} active={A.sum():,} deaths={Y.sum():,} "
        f"censored={100*(D==0).mean():.1f}%")

    rows = []
    for name in (cfg.get("external.conditions") or ["naive", "structured"]):
        X = np.ones((len(Y), 1)) if name == "naive" else S
        psi, keep, diag = EST.crossfit_aipw(X, A, Y, cfg.folds, cfg.seed,
                                            trim=cfg.trim, D=D)
        pt = float(psi[keep].mean() * 100)
        bt = np.array([psi[b][keep[b]].mean() * 100 for b in boot])
        e = bootstrap_summary(pt, bt)
        _, if_lo, if_hi, se = influence_function_ci(psi, keep)
        ato_b = np.array([EST.ato_from_boot(diag["psi_ato"], diag["h_ato"], b) for b in boot])
        ato = bootstrap_summary(diag["ato"], ato_b)
        bias = bootstrap_summary(pt - ref_rd, bt - ref_rd)

        rows.append(EE._guard_impossible(cfg, {
            "intervention": intervention, "condition": name, "cohort": "eicu_all",
            "dataset": "eicu",
            "estimator": "aipw" if name != "naive" else "unadjusted", "reduction": "",
            **e.as_row("effect_"),
            "effect_if_ci_low": if_lo, "effect_if_ci_high": if_hi,
            **ato.as_row("ato_"),
            "ref_rd": ref_rd, "ref_ci_low": ref_lo, "ref_ci_high": ref_hi,
            "ref_source": str(ref.get("source")),
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
        log(f"    {name:12s} RD={pt:7.1f}  CI[{e.ci_low:6.1f},{e.ci_high:6.1f}]  "
            f"bias={pt-ref_rd:+6.1f}  ESS={round(diag['ess'])}")
    return rows


def run(cfg, force: bool = False, intervention: str = None):
    from src import results as R

    eicu = cfg.input("eicu_dir")
    if not eicu.exists():
        log(f"eICU not present at {eicu}; skipping external replication")
        return

    t0 = time.time()
    ext = cfg.get("external") or {}
    names = [intervention] if intervention else ext.get("interventions", [])
    names = [n for n in names if n in _BUILDERS]
    log(f"=== external[eICU]: structured replication for {names} ===")

    spine = _spine(cfg)
    aps = _aps(cfg)

    rows = []
    for iv in names:
        try:
            rows += _one(cfg, iv, spine, aps)
        except Exception as e:
            log(f"  {iv}: eICU failed ({e}); skip")

    R.reset_rows(cfg, "effects.csv", dataset="eicu")
    R.append_rows(cfg, "effects.csv", rows)
    log(f"external done in {time.time()-t0:,.0f}s -> effects.csv ({len(rows)} eICU rows)")
