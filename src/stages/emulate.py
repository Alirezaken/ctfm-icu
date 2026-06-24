"""§9.5  emulate -- target-trial emulation; build an intervention's cohort.

New-user active-comparator design at a single time zero (§4). Builds the eligible
cohort, assigns the two arms, fixes the outcome at the horizon, applies the
all-modality gate (>=1 pre-t0 frontal CXR AND >=1 pre-t0 non-radiology note), and
writes CONSORT accounting to cohorts.csv. Time-zero discipline: only data strictly
before t0 is used downstream (the cohort table stores t0; adjustment vars are
pulled pre-t0 in `estimate`).

Membership uses the link layer (cxr_studies timestamps, discharge-note charttimes)
and the cleaned event stream -- it does NOT need the embedding vectors, which are
joined later at `estimate`. So this stage runs before/independently of the GPU job.

Per-intervention clinical logic lives in a builder; fluids_sepsis is implemented
(§9.5 starts here). The cohort table is saved to paths.cohorts/<intervention>.parquet.

DESIGN CHOICES flagged inline (suspected-infection proxy, itemid sets, weight
source) are the analyst's operationalization of the config rules -- listed in
docs/QUESTIONS_FOR_SOROOSH.md for confirmation; not improvised statistics.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd

from src.util import log, Checkpoint
from src import events as ev

# ---- itemid sets (MIMIC-IV d_items / d_labitems), documented design choices ----
MAP_ITEMS = [220052, 220181, 225312]                 # arterial / NBP / ART mean
SBP_ITEMS = [220050, 220179, 225309]                 # arterial / NBP / ART systolic
VASOPRESSOR_ITEMS = [221906, 221289, 229617, 222315, 221749,
                     229630, 229631, 229632, 221662]  # norepi/epi/vaso/phenyl/dopa
CRYSTALLOID_ITEMS = [225158, 220955, 220953, 220956, 226364, 226375]  # NaCl0.9/Ringers/OR/PACU
WEIGHT_ITEMS = [226512, 224639]                      # admission weight kg, daily weight
LACTATE_ITEM = 50813
CREATININE_ITEM = 50912
# suspected-infection proxy: IV antibiotic by drug-name keyword (prescriptions->med)
ANTIBIOTIC_KEYS = ["vancomycin", "piperacillin", "cefepime", "ceftriaxone", "meropenem",
                   "ciprofloxacin", "levofloxacin", "metronidazole", "azithromycin",
                   "ceftazidime", "ampicillin", "gentamicin", "tobramycin", "aztreonam",
                   "linezolid", "daptomycin", "imipenem", "cefazolin", "clindamycin"]


def _age_at(patients, intimes):
    """Approx age at a timestamp: anchor_age + (year(t) - anchor_year)."""
    p = patients.set_index("subject_id")
    yr = pd.to_datetime(intimes["time"]).dt.year.values
    sid = intimes["subject_id"].values
    aa = p["anchor_age"].reindex(sid).values
    ay = p["anchor_year"].reindex(sid).values
    return aa + (yr - ay)


def _build_fluids_sepsis(cfg):
    """Return (cohort_df, consort) for the fluids_sepsis target trial."""
    consort = []  # (step, count)

    icu = ev.link(cfg, "icu_stays")
    icu["intime"] = pd.to_datetime(icu["intime"]); icu["outtime"] = pd.to_datetime(icu["outtime"])
    patients = ev.link(cfg, "patients")
    consort.append(("icu_stays_total", len(icu)))

    # adults (>=18) at ICU intime
    age = patients.set_index("subject_id").reindex(icu["subject_id"].values)
    icu["age_t0"] = (age["anchor_age"].values + (icu["intime"].dt.year.values - age["anchor_year"].values))
    icu = icu[icu["age_t0"] >= 18].copy()
    consort.append(("adults_ge18", len(icu)))

    win_h = 24
    icu["win_end"] = icu["intime"] + pd.Timedelta(hours=win_h)
    stay_ids = set(icu["stay_id"])

    # crystalloid input events within first 24h -> cumulative volume per stay
    cry = ev.read_modality(cfg, "input", CRYSTALLOID_ITEMS,
                           columns=["subject_id", "stay_id", "time", "value_num"])
    cry = cry[cry["stay_id"].isin(stay_ids)].merge(
        icu[["stay_id", "intime", "win_end"]], on="stay_id", how="inner")
    cry = cry[(cry["time"] >= cry["intime"]) & (cry["time"] <= cry["win_end"])]
    cry = cry.sort_values(["stay_id", "time"])
    cry["cum_ml"] = cry.groupby("stay_id")["value_num"].cumsum()
    # time at which cumulative crystalloid first reaches 1000 mL
    reached = cry[cry["cum_ml"] >= 1000].groupby("stay_id")["time"].min().rename("t_1L")
    icu = icu.merge(reached, on="stay_id", how="left")
    icu_1l = icu[icu["t_1L"].notna()].copy()
    consort.append(("received_ge1L_crystalloid_24h", len(icu_1l)))

    # hypotension/vasopressor AFTER t_1L within window -> time zero
    sbp = ev.read_modality(cfg, "chart", SBP_ITEMS, columns=["stay_id", "time", "value_num"])
    sbp = sbp[(sbp["stay_id"].isin(set(icu_1l["stay_id"]))) & (sbp["value_num"] < 100)]
    vaso = ev.read_modality(cfg, "input", VASOPRESSOR_ITEMS, columns=["stay_id", "time", "value_num"])
    vaso = vaso[vaso["stay_id"].isin(set(icu_1l["stay_id"]))]
    hypo = pd.concat([sbp[["stay_id", "time"]], vaso[["stay_id", "time"]]], ignore_index=True)
    hypo = hypo.merge(icu_1l[["stay_id", "t_1L", "win_end"]], on="stay_id", how="inner")
    hypo = hypo[(hypo["time"] > hypo["t_1L"]) & (hypo["time"] <= hypo["win_end"])]
    t0 = hypo.groupby("stay_id")["time"].min().rename("t0")
    icu_t0 = icu_1l.merge(t0, on="stay_id", how="inner")
    consort.append(("sepsis_induced_hypotension_after_fluids", len(icu_t0)))

    # suspected infection: >=1 IV antibiotic (drug-name proxy) in [intime-24h, t0]
    med = ev.read_modality(cfg, "med", itemids=None, columns=["subject_id", "time", "type"])
    key = "|".join(ANTIBIOTIC_KEYS)
    med = med[med["type"].str.contains(key, case=False, na=False)]
    abx = med.merge(icu_t0[["subject_id", "stay_id", "intime", "t0"]], on="subject_id", how="inner")
    abx = abx[(abx["time"] >= abx["intime"] - pd.Timedelta(hours=24)) & (abx["time"] <= abx["t0"])]
    infected = set(abx["stay_id"])
    icu_elig = icu_t0[icu_t0["stay_id"].isin(infected)].copy()
    consort.append(("suspected_infection_antibiotic_proxy", len(icu_elig)))

    # one stay per patient (first qualifying) -- patient-level, no double counting
    icu_elig = icu_elig.sort_values("t0").drop_duplicates("subject_id", keep="first")
    consort.append(("eligible_patients_unique", len(icu_elig)))

    # ---- arm assignment: cumulative crystalloid in [t0, t0+24h] vs 30 ml/kg ----
    wt = ev.read_modality(cfg, "chart", WEIGHT_ITEMS, columns=["subject_id", "time", "value_num"])
    wt = wt[wt["value_num"].between(30, 400)].sort_values("time")
    wt = wt.groupby("subject_id")["value_num"].first().rename("weight_kg")  # earliest plausible wt
    icu_elig = icu_elig.merge(wt, on="subject_id", how="left")

    post = cry.merge(icu_elig[["stay_id", "t0"]], on="stay_id", how="inner")
    post = post[(post["time"] >= post["t0"]) & (post["time"] <= post["t0"] + pd.Timedelta(hours=24))]
    post_vol = post.groupby("stay_id")["value_num"].sum().rename("crystalloid_24h_post_t0")
    icu_elig = icu_elig.merge(post_vol, on="stay_id", how="left")
    icu_elig["crystalloid_24h_post_t0"] = icu_elig["crystalloid_24h_post_t0"].fillna(0.0)
    thresh = 30.0 * icu_elig["weight_kg"]                     # 30 ml/kg
    # restrictive (active) if post-t0 crystalloid below 30 ml/kg threshold; else liberal
    icu_elig["arm"] = np.where(icu_elig["crystalloid_24h_post_t0"] < thresh,
                               "active", "comparator")
    icu_elig.loc[icu_elig["weight_kg"].isna(), "arm"] = pd.NA   # undefined without weight
    consort.append(("arm_active_restrictive", int((icu_elig["arm"] == "active").sum())))
    consort.append(("arm_comparator_liberal", int((icu_elig["arm"] == "comparator").sum())))

    cohort = icu_elig[["subject_id", "hadm_id", "stay_id", "intime", "t0",
                       "age_t0", "weight_kg", "crystalloid_24h_post_t0", "arm"]].copy()
    return cohort, consort


_BUILDERS = {"fluids_sepsis": _build_fluids_sepsis}


def _outcome_and_gate(cfg, intervention, cohort, consort):
    iv = cfg.get(f"interventions.{intervention}")
    horizon = int(iv["horizon_days"])
    patients = ev.link(cfg, "patients")

    # outcome: all-cause mortality within horizon of t0 (MIMIC dod)
    dod = patients.set_index("subject_id")["dod"]
    cohort["dod"] = pd.to_datetime(cohort["subject_id"].map(dod))
    days = (cohort["dod"] - cohort["t0"]).dt.total_seconds() / 86400.0
    cohort["outcome"] = ((days >= 0) & (days <= horizon)).astype(int)

    # sex
    cohort["sex"] = cohort["subject_id"].map(patients.set_index("subject_id")["gender"])

    # all-modality gate: pre-t0 frontal CXR AND pre-t0 non-radiology (discharge) note
    cxr = ev.link(cfg, "cxr_studies")
    cxr["study_datetime"] = pd.to_datetime(cxr["study_datetime"], errors="coerce")
    cxr = cxr[cxr["views"].str.contains("PA|AP", case=False, na=False)]  # frontal only
    pre_cxr = cxr.merge(cohort[["subject_id", "t0"]], on="subject_id", how="inner")
    has_cxr = set(pre_cxr.loc[pre_cxr["study_datetime"] < pre_cxr["t0"], "subject_id"])

    disch = pd.read_csv(cfg.input("notes_dir") / "note" / "discharge.csv.gz",
                        usecols=["subject_id", "charttime"])
    disch["charttime"] = pd.to_datetime(disch["charttime"], errors="coerce")
    pre_note = disch.merge(cohort[["subject_id", "t0"]], on="subject_id", how="inner")
    has_note = set(pre_note.loc[pre_note["charttime"] < pre_note["t0"], "subject_id"])

    cohort["has_pre_t0_cxr"] = cohort["subject_id"].isin(has_cxr)
    cohort["has_pre_t0_note"] = cohort["subject_id"].isin(has_note)
    cohort["all_modality"] = cohort["has_pre_t0_cxr"] & cohort["has_pre_t0_note"]
    consort.append(("with_pre_t0_frontal_cxr", int(cohort["has_pre_t0_cxr"].sum())))
    consort.append(("with_pre_t0_discharge_note", int(cohort["has_pre_t0_note"].sum())))
    consort.append(("all_modality_cohort", int(cohort["all_modality"].sum())))
    consort.append(("all_modality_deaths", int(cohort.loc[cohort["all_modality"], "outcome"].sum())))
    return cohort, consort


def run(cfg, force: bool = False, intervention: str = "fluids_sepsis"):
    iv = cfg.get(f"interventions.{intervention}")
    if iv is None:
        raise KeyError(f"unknown intervention '{intervention}'")
    cfg.require(f"interventions.{intervention}.eligibility",
                f"interventions.{intervention}.time_zero",
                f"interventions.{intervention}.outcome",
                f"interventions.{intervention}.horizon_days",
                "pooling.look_back_window_hours")
    if intervention not in _BUILDERS:
        raise NotImplementedError(
            f"emulate builder for '{intervention}' not implemented yet; "
            f"implemented: {list(_BUILDERS)} (fluids_sepsis first per §9.5).")

    ckpt = Checkpoint(cfg, "emulate")
    t0 = time.time()
    log(f"emulate[{intervention}]: building cohort ...")
    cohort, consort = _BUILDERS[intervention](cfg)
    cohort, consort = _outcome_and_gate(cfg, intervention, cohort, consort)

    # save cohort table
    out = cfg.storage("cohorts")
    out.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(out / f"{intervention}.parquet")

    # CONSORT -> cohorts.csv (long format, §7.6)
    from src import results as R
    rows = [{"intervention": intervention, "section": "consort", "metric": step,
             "stratum": "", "arm": "", "value": cnt, "support_count": cnt}
            for step, cnt in consort]
    R.append_rows(cfg, "cohorts.csv", rows)

    for step, cnt in consort:
        log(f"  {step:42s} {cnt:>8,}")
    ckpt.mark(intervention, n=len(cohort),
              all_modality=int(cohort["all_modality"].sum()))
    log(f"emulate[{intervention}] done in {time.time()-t0:,.0f}s "
        f"-> {out/(intervention+'.parquet')}")
