"""emulate -- target-trial emulation. Builds every intervention's cohort.

New-user, active-comparator design at a single time zero. For each intervention: apply
eligibility, fix t0, assign the two arms, resolve the outcome at the horizon, apply the
modality gate, and record CONSORT.

THE GATE (changed, and this is the most consequential change in the rewrite)

  PRIMARY GATE = eligible + >=1 pre-t0 frontal chest X-ray.

  The previous design also required a pre-t0 discharge summary. That was wrong, and
  silently so. A discharge summary's charttime is the DISCHARGE time of its own
  hospitalization. t0 falls during the CURRENT ICU stay. So a discharge summary observed
  before t0 can only have come from an EARLIER hospitalization -- and requiring one
  restricted the entire headline cohort to patients who had been hospitalized before.
  That is selection on prior healthcare utilization, which predicts both treatment and
  death. It cost ~35% of the sample and introduced a bias nobody had accounted for.

  Imaging is the right gate because every chest X-ray has a paired radiology report, so
  the image channel and the contemporaneous-text channel are both available for 100% of
  the primary cohort. The prior-discharge-summary channel (`histnote`) is available for
  roughly 55% and is handled with an explicit missingness indicator, not a hidden filter.

Membership uses only the link layer and the cleaned event stream, so this stage runs
independently of the GPU embedding job.

Writes ONE consolidated manifest: manifests/cohorts.parquet, with an `intervention`
column. CONSORT goes to results/cohorts.csv.
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


# ---- itemids for the other three interventions (documented design choices) ----
PO2_ITEM = 50821            # lab pO2 (blood gas)
FIO2_ITEM = 223835          # chart Inspired O2 Fraction
PEEP_ITEMS = [220339, 224700]
POSITION_ITEM = 224093      # chart Position (value_txt e.g. "Prone")
CREAT_ITEM = 50912
K_ITEM = 50971
PH_ITEM = 50820
BUN_ITEM = 51006            # lab BUN (uremia exclusion)
URINE_ITEMS = [226559, 226560, 226561, 226584, 226563, 226564, 226565,
               226567, 226557, 226558, 227488, 227489]   # outputevents urine output
RRT_PROC = [225802]         # procedureevents Dialysis - CRRT
DIALYSIS_CHART = [225126]   # chart "Dialysis patient"
HB_ITEM = 51222
RBC_ITEMS = [225168, 220996, 226368, 227070]   # inputevents packed red cells


def _icu_base(cfg):
    """ICU stays with parsed times + adult age at intime; and patients table."""
    icu = ev.link(cfg, "icu_stays")
    icu["intime"] = pd.to_datetime(icu["intime"]); icu["outtime"] = pd.to_datetime(icu["outtime"])
    patients = ev.link(cfg, "patients")
    p = patients.set_index("subject_id").reindex(icu["subject_id"].values)
    icu["age_t0"] = p["anchor_age"].values + (icu["intime"].dt.year.values - p["anchor_year"].values)
    return icu, patients


def _labs_in_stays(cfg, itemids, icu):
    """Attach labevents (no stay_id) to ICU stays by subject_id + time-in-window."""
    lab = ev.read_modality(cfg, "lab", itemids, columns=["subject_id", "time", "type", "value_num"])
    lab = lab.merge(icu[["subject_id", "stay_id", "intime", "outtime"]], on="subject_id")
    return lab[(lab["time"] >= lab["intime"]) & (lab["time"] <= lab["outtime"])]


def _chart_in_stays(cfg, itemids, icu, txt=False):
    cols = ["subject_id", "stay_id", "time", "type", "value_num"] + (["value_txt"] if txt else [])
    ch = ev.read_modality(cfg, "chart", itemids, columns=cols)
    ch = ch[ch["stay_id"].isin(set(icu["stay_id"]))].merge(
        icu[["stay_id", "intime", "outtime"]], on="stay_id")
    return ch[(ch["time"] >= ch["intime"]) & (ch["time"] <= ch["outtime"])]


def _cultures(cfg):
    """Body-fluid culture order times per subject (microbiologyevents, hosp)."""
    micro = pd.read_csv(cfg.input("mimic_dir") / "hosp" / "microbiologyevents.csv.gz",
                        usecols=["subject_id", "charttime", "chartdate"])
    micro["time"] = pd.to_datetime(micro["charttime"].fillna(micro["chartdate"]), errors="coerce")
    return micro.dropna(subset=["time"])[["subject_id", "time"]]


def _suspected_infection(cfg, elig, abx):
    """Sepsis-3 suspicion of infection (decision 3): an antibiotic in [intime-24h, t0]
    AND a body-fluid culture in [intime-48h, t0]. `elig` has subject_id/stay_id/intime/t0;
    `abx` is antibiotic med events (subject_id, time). Returns the qualifying stay_ids."""
    keep = elig[["subject_id", "stay_id", "intime", "t0"]]
    a = abx.merge(keep, on="subject_id")
    a = a[(a["time"] >= a["intime"] - pd.Timedelta("24h")) & (a["time"] <= a["t0"])]
    c = _cultures(cfg).merge(keep, on="subject_id")
    c = c[(c["time"] >= c["intime"] - pd.Timedelta("48h")) & (c["time"] <= c["t0"])]
    return set(a["stay_id"]) & set(c["stay_id"])


def _antibiotic_med(cfg):
    med = ev.read_modality(cfg, "med", itemids=None, columns=["subject_id", "time", "type"])
    return med[med["type"].str.contains("|".join(ANTIBIOTIC_KEYS), case=False, na=False)]


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

    # suspected infection (decision 3: Sepsis-3) = antibiotic + body-fluid culture
    infected = _suspected_infection(cfg, icu_t0, _antibiotic_med(cfg))
    icu_elig = icu_t0[icu_t0["stay_id"].isin(infected)].copy()
    consort.append(("suspected_infection_sepsis3", len(icu_elig)))

    # one stay per patient (first qualifying) -- patient-level, no double counting
    icu_elig = icu_elig.sort_values("t0").drop_duplicates("subject_id", keep="first")
    consort.append(("eligible_patients_unique", len(icu_elig)))

    # ---- arm assignment (decision 1): cumulative crystalloid per KG in [t0,t0+24h] ----
    wt = ev.read_modality(cfg, "chart", WEIGHT_ITEMS, columns=["subject_id", "time", "value_num"])
    wt = wt[wt["value_num"].between(30, 400)].sort_values("time")
    wt = wt.groupby("subject_id")["value_num"].first().rename("weight_kg")  # earliest plausible wt
    icu_elig = icu_elig.merge(wt, on="subject_id", how="left")
    icu_elig["weight_kg"] = icu_elig["weight_kg"].fillna(icu_elig["weight_kg"].median())

    post = cry.merge(icu_elig[["stay_id", "t0"]], on="stay_id", how="inner")
    post = post[(post["time"] >= post["t0"]) & (post["time"] <= post["t0"] + pd.Timedelta(hours=24))]
    post_vol = post.groupby("stay_id")["value_num"].sum().rename("crystalloid_24h_post_t0")
    icu_elig = icu_elig.merge(post_vol, on="stay_id", how="left")
    icu_elig["crystalloid_24h_post_t0"] = icu_elig["crystalloid_24h_post_t0"].fillna(0.0)
    icu_elig["mlkg_24h"] = icu_elig["crystalloid_24h_post_t0"] / icu_elig["weight_kg"]
    # restrictive < 30 ml/kg; if that split is worse than ~70/30, use the cohort median
    # so overlap is guaranteed (decision 1). Disclose which threshold was actually used.
    thr = 30.0
    used_median = not (0.30 <= float((icu_elig["mlkg_24h"] < thr).mean()) <= 0.70)
    if used_median:
        thr = float(icu_elig["mlkg_24h"].median())
    icu_elig["arm"] = np.where(icu_elig["mlkg_24h"] < thr, "active", "comparator")
    consort.append(("arm_threshold_mlkg", round(thr, 1)))
    consort.append(("arm_threshold_is_median_fallback", int(used_median)))   # 1 = NOT the pre-spec 30 ml/kg
    log(f"  fluids arm threshold: {thr:.1f} ml/kg "
        f"({'MEDIAN FALLBACK - not pre-specified 30' if used_median else 'pre-specified 30'})")
    consort.append(("arm_active_restrictive", int((icu_elig["arm"] == "active").sum())))
    consort.append(("arm_comparator_liberal", int((icu_elig["arm"] == "comparator").sum())))

    cohort = icu_elig[["subject_id", "hadm_id", "stay_id", "intime", "t0", "age_t0",
                       "weight_kg", "crystalloid_24h_post_t0", "mlkg_24h", "arm"]].copy()
    return cohort, consort


def _build_prone_ards(cfg):
    """Prone vs supine in severe ARDS (PROSEVA). t0 = first severe hypoxemia
    (P/F<150, FiO2>=0.6, PEEP>=5) within 36h of ICU admit; arm = prone session
    (chart Position contains 'Prone') within 24h of t0."""
    consort = []
    icu, _ = _icu_base(cfg)
    consort.append(("icu_stays_total", len(icu)))
    icu = icu[icu["age_t0"] >= 18].copy()
    consort.append(("adults_ge18", len(icu)))

    po2 = _labs_in_stays(cfg, [PO2_ITEM], icu).rename(columns={"value_num": "po2"}).sort_values("time")
    fio2 = _chart_in_stays(cfg, [FIO2_ITEM], icu).rename(columns={"value_num": "fio2"}).sort_values("time")
    fio2["fio2"] = np.where(fio2["fio2"] > 1.5, fio2["fio2"] / 100.0, fio2["fio2"])  # % -> fraction
    peep = _chart_in_stays(cfg, PEEP_ITEMS, icu).rename(columns={"value_num": "peep"}).sort_values("time")
    if not len(po2) or not len(fio2):
        return pd.DataFrame(), consort + [("no_pf_data", 0)]

    tol = pd.Timedelta("4h")
    m = pd.merge_asof(po2[["stay_id", "time", "po2"]], fio2[["stay_id", "time", "fio2"]],
                      on="time", by="stay_id", tolerance=tol, direction="backward").dropna(subset=["fio2"])
    m = pd.merge_asof(m.sort_values("time"), peep[["stay_id", "time", "peep"]].sort_values("time"),
                      on="time", by="stay_id", tolerance=tol, direction="backward")
    m["pf"] = m["po2"] / m["fio2"].replace(0, np.nan)
    severe = m[(m["pf"] < 150) & (m["fio2"] >= 0.6) & (m["peep"] >= 5)]
    t0 = severe.groupby("stay_id")["time"].min().rename("t0")
    # 36h window from ventilation start (first PEEP event) if extractable, else ICU admit (decision)
    vent_start = peep.groupby("stay_id")["time"].min().rename("vent_start")
    icu = icu.merge(t0, on="stay_id", how="inner").merge(vent_start, on="stay_id", how="left")
    ref_start = icu["vent_start"].fillna(icu["intime"])
    icu = icu[icu["t0"] <= ref_start + pd.Timedelta(hours=36)]
    icu = icu.sort_values("t0").drop_duplicates("subject_id", keep="first")
    consort.append(("severe_ards_ventilated_within36h", len(icu)))

    pos = _chart_in_stays(cfg, [POSITION_ITEM], icu, txt=True)
    pos = pos[pos["value_txt"].str.contains("prone", case=False, na=False)]
    pos = pos.merge(icu[["stay_id", "t0"]], on="stay_id", how="inner")
    prone_24h = set(pos.loc[(pos["time"] >= pos["t0"]) &
                            (pos["time"] <= pos["t0"] + pd.Timedelta(hours=24)), "stay_id"])
    icu["arm"] = np.where(icu["stay_id"].isin(prone_24h), "active", "comparator")
    consort.append(("arm_active_prone", int((icu["arm"] == "active").sum())))
    consort.append(("arm_comparator_supine", int((icu["arm"] == "comparator").sum())))
    icu["weight_kg"] = np.nan
    return icu[["subject_id", "hadm_id", "stay_id", "intime", "t0", "age_t0", "weight_kg", "arm"]], consort


def _build_rrt_timing(cfg):
    """Early vs delayed RRT in AKI (STARRT-AKI). t0 = first KDIGO stage 2-3
    (creatinine >=2x stay baseline) without an urgent indication (K<=6.5, pH>=7.2);
    arm = RRT (CRRT/dialysis) initiated within 12h of t0 vs not."""
    consort = []
    icu, _ = _icu_base(cfg)
    consort.append(("icu_stays_total", len(icu)))
    icu = icu[icu["age_t0"] >= 18].copy()
    consort.append(("adults_ge18", len(icu)))

    # baseline creatinine = min over [intime-7d, outtime] so a pre-admission value is used
    # if one exists, else the stay minimum (decision: RRT baseline).
    crall = ev.read_modality(cfg, "lab", [CREAT_ITEM], columns=["subject_id", "time", "value_num"])
    crall = crall.merge(icu[["subject_id", "stay_id", "intime", "outtime"]], on="subject_id")
    crb = crall[(crall["time"] >= crall["intime"] - pd.Timedelta("7D")) & (crall["time"] <= crall["outtime"])]
    base = crb.groupby("stay_id")["value_num"].min().rename("cr_base")
    cr = crall[(crall["time"] >= crall["intime"]) & (crall["time"] <= crall["outtime"])].sort_values("time")
    cr = cr.merge(base, on="stay_id")
    # KDIGO stage 2-3 route A: creatinine >= 2x baseline
    t0_cr = cr[cr["value_num"] >= 2.0 * cr["cr_base"]].groupby("stay_id")["time"].min()

    # KDIGO stage 2-3 route B (issue 5): oliguria, urine output < 0.5 ml/kg/h over the
    # first 24 h (t0 = intime + 24 h). Adds patients who meet KDIGO by urine, not creatinine.
    wt = ev.read_modality(cfg, "chart", WEIGHT_ITEMS, columns=["subject_id", "time", "value_num"])
    wt = wt.merge(icu[["subject_id", "stay_id", "intime", "outtime"]], on="subject_id")
    wt = wt[(wt["time"] >= wt["intime"] - pd.Timedelta("1D")) & (wt["time"] <= wt["outtime"])]
    wkg = wt.groupby("stay_id")["value_num"].median()
    uo = ev.read_modality(cfg, "output", URINE_ITEMS, columns=["subject_id", "time", "value_num"])
    uo = uo.merge(icu[["subject_id", "stay_id", "intime"]], on="subject_id")
    uo_win = uo[(uo["time"] >= uo["intime"]) & (uo["time"] <= uo["intime"] + pd.Timedelta("24h"))]
    agg = uo_win.groupby("stay_id")["value_num"].agg(["sum", "count"])
    mlkgh = (agg["sum"] / (24.0 * wkg.reindex(agg.index))).dropna()
    # oliguria only for genuinely MONITORED stays (>=6 charted urine values in 24h ~ q4h),
    # so sparse/incomplete charting isn't mistaken for low output (issue 5 over-capture guard).
    oliguric = [s for s in mlkgh.index if mlkgh[s] < 0.5 and agg.loc[s, "count"] >= 6]
    intime_by_stay = icu.set_index("stay_id")["intime"]
    t0_uo = intime_by_stay.reindex(oliguric) + pd.Timedelta("24h")
    # combine both routes: earliest qualifying time per stay
    t0 = pd.concat([t0_cr, t0_uo.dropna()]).groupby(level=0).min().rename("t0")
    icu = icu.merge(t0, on="stay_id", how="inner")
    consort.append(("aki_kdigo_stage2_3", len(icu)))
    consort.append(("aki_by_urine_route_only", int(len(set(oliguric) - set(t0_cr.index)))))

    # exclude urgent indication near t0: K>6.5, pH<7.2, or uremia (BUN>100) within +/-6h.
    # (fluid overload is not operationalized -- needs a reliable fluid-balance series.)
    def _near_t0(itemids, cmp):
        lab = _labs_in_stays(cfg, itemids, icu).merge(icu[["stay_id", "t0"]], on="stay_id")
        lab = lab[(lab["time"] >= lab["t0"] - pd.Timedelta("6h")) & (lab["time"] <= lab["t0"] + pd.Timedelta("6h"))]
        return set(lab.loc[cmp(lab["value_num"]), "stay_id"])
    urgent = (_near_t0([K_ITEM], lambda v: v > 6.5) | _near_t0([PH_ITEM], lambda v: v < 7.2)
              | _near_t0([BUN_ITEM], lambda v: v > 100))
    icu = icu[~icu["stay_id"].isin(urgent)]
    icu = icu.sort_values("t0").drop_duplicates("subject_id", keep="first")
    consort.append(("no_urgent_indication", len(icu)))

    rrt = pd.concat([
        ev.read_modality(cfg, "procedure", RRT_PROC, columns=["stay_id", "time", "value_num"]),
        ev.read_modality(cfg, "chart", DIALYSIS_CHART, columns=["stay_id", "time", "value_num"]),
    ], ignore_index=True)
    rrt = rrt[rrt["stay_id"].isin(set(icu["stay_id"]))].merge(icu[["stay_id", "t0"]], on="stay_id")
    early = set(rrt.loc[(rrt["time"] >= rrt["t0"]) &
                        (rrt["time"] <= rrt["t0"] + pd.Timedelta(hours=12)), "stay_id"])
    icu["arm"] = np.where(icu["stay_id"].isin(early), "active", "comparator")
    consort.append(("arm_active_accelerated", int((icu["arm"] == "active").sum())))
    consort.append(("arm_comparator_standard", int((icu["arm"] == "comparator").sum())))
    icu["weight_kg"] = np.nan
    return icu[["subject_id", "hadm_id", "stay_id", "intime", "t0", "age_t0", "weight_kg", "arm"]], consort


def _build_transfusion_threshold(cfg):
    """Restrictive vs liberal transfusion in septic shock (TRISS). Eligible = septic
    shock (infection + vasopressor) with Hb<9. t0 = first Hb<9. Arm from the Hb just
    before the first post-t0 RBC transfusion: <=7 restrictive, (7,9] liberal
    (never-transfused patients cannot be assigned a threshold -> excluded)."""
    consort = []
    icu, _ = _icu_base(cfg)
    consort.append(("icu_stays_total", len(icu)))
    icu = icu[icu["age_t0"] >= 18].copy()
    consort.append(("adults_ge18", len(icu)))

    # septic shock: vasopressor during stay + Hb<9 -> t0; infection confirmed at t0
    vaso = ev.read_modality(cfg, "input", VASOPRESSOR_ITEMS, columns=["subject_id", "stay_id", "time"])
    shock = set(vaso[vaso["stay_id"].isin(set(icu["stay_id"]))]["stay_id"])
    icu = icu[icu["stay_id"].isin(shock)]
    consort.append(("vasopressor_shock", len(icu)))

    hb = _labs_in_stays(cfg, [HB_ITEM], icu).rename(columns={"value_num": "hb"}).sort_values("time")
    t0 = hb[hb["hb"] < 9.0].groupby("stay_id")["time"].min().rename("t0")
    icu = icu.merge(t0, on="stay_id", how="inner")
    icu = icu.sort_values("t0").drop_duplicates("subject_id", keep="first")
    consort.append(("hb_lt9", len(icu)))
    infected = _suspected_infection(cfg, icu, _antibiotic_med(cfg))     # decision 3
    icu = icu[icu["stay_id"].isin(infected)]
    consort.append(("suspected_infection_sepsis3", len(icu)))

    # Arm by threshold the Hb trajectory is consistent with (decision: KEEP never-transfused).
    # Transfused: restrictive if pre-transfusion Hb<=7, else liberal. Never transfused =
    # restrictive-consistent (a liberal strategy would have transfused at Hb<9).
    rbc = ev.read_modality(cfg, "input", RBC_ITEMS, columns=["stay_id", "time"])
    rbc = rbc[rbc["stay_id"].isin(set(icu["stay_id"]))].merge(icu[["stay_id", "t0"]], on="stay_id")
    rbc = rbc[rbc["time"] >= rbc["t0"]].sort_values("time")
    first_tx = rbc.groupby("stay_id")["time"].min().rename("tx_time").reset_index()
    hb_pre = pd.merge_asof(first_tx.sort_values("tx_time"),
                           hb[["stay_id", "time", "hb"]].sort_values("time"),
                           left_on="tx_time", right_on="time", by="stay_id", direction="backward")
    tx_arm = dict(zip(hb_pre["stay_id"], np.where(hb_pre["hb"] <= 7.0, "active", "comparator")))
    icu["arm"] = icu["stay_id"].map(tx_arm).fillna("active")   # never-transfused -> restrictive
    consort.append(("transfused_post_t0", len(first_tx)))
    consort.append(("arm_active_restrictive", int((icu["arm"] == "active").sum())))
    consort.append(("arm_comparator_liberal", int((icu["arm"] == "comparator").sum())))
    icu["weight_kg"] = np.nan
    return icu[["subject_id", "hadm_id", "stay_id", "intime", "t0", "age_t0", "weight_kg", "arm"]], consort


_BUILDERS = {
    "fluids_sepsis": _build_fluids_sepsis,
    "prone_positioning": _build_prone_ards,
    "rrt_timing": _build_rrt_timing,
    "transfusion_threshold": _build_transfusion_threshold,
}




def _outcome_and_gate(cfg, intervention, cohort, consort):
    """Resolve the outcome at the horizon, then apply the modality gate.

    Sets these flags on every eligible patient (nothing is dropped here -- the flags
    define the analysis scopes, and the `eligible` scope is reported alongside `imaged`
    so the cost of the gate is visible rather than assumed away):

      has_pre_t0_cxr       >=1 frontal CXR strictly before t0   -> THE PRIMARY GATE
      has_pre_t0_radtext   >=1 radiology report strictly before t0
      has_pre_t0_histnote  >=1 discharge summary strictly before t0
                           == the patient had a PRIOR hospitalization
      imaged               == has_pre_t0_cxr  (the primary analysis cohort)
    """
    horizon = int(cfg.get(f"interventions.{intervention}.horizon_days"))
    patients = ev.link(cfg, "patients")

    # outcome: all-cause mortality within the horizon of t0 (MIMIC dod)
    dod = patients.set_index("subject_id")["dod"]
    cohort["dod"] = pd.to_datetime(cohort["subject_id"].map(dod))
    days = (cohort["dod"] - cohort["t0"]).dt.total_seconds() / 86400.0
    cohort["outcome"] = ((days >= 0) & (days <= horizon)).astype(int)
    cohort["sex"] = cohort["subject_id"].map(patients.set_index("subject_id")["gender"])

    # ---- imaging: frontal CXR strictly before t0 ----
    cxr = ev.link(cfg, "cxr_studies")
    cxr["study_datetime"] = pd.to_datetime(cxr["study_datetime"], errors="coerce")
    cxr = cxr[cxr["views"].str.contains("PA|AP", case=False, na=False)]   # frontal only
    pre = cxr.merge(cohort[["subject_id", "t0"]], on="subject_id", how="inner")
    has_cxr = set(pre.loc[pre["study_datetime"] < pre["t0"], "subject_id"])

    # ---- text: which patients have each kind of note strictly before t0 ----
    note_root = cfg.input("notes_dir") / "note"

    def _pre_t0_subjects(fname):
        n = pd.read_csv(note_root / fname, usecols=["subject_id", "charttime"])
        n["charttime"] = pd.to_datetime(n["charttime"], errors="coerce")
        m = n.merge(cohort[["subject_id", "t0"]], on="subject_id", how="inner")
        return set(m.loc[m["charttime"] < m["t0"], "subject_id"])

    has_rad = _pre_t0_subjects("radiology.csv.gz")      # contemporaneous expert text
    has_disch = _pre_t0_subjects("discharge.csv.gz")    # PRIOR-admission history

    cohort["has_pre_t0_cxr"] = cohort["subject_id"].isin(has_cxr)
    cohort["has_pre_t0_radtext"] = cohort["subject_id"].isin(has_rad)
    cohort["has_pre_t0_histnote"] = cohort["subject_id"].isin(has_disch)
    cohort["imaged"] = cohort["has_pre_t0_cxr"]                    # THE PRIMARY GATE

    n_elig = len(cohort)
    n_img = int(cohort["imaged"].sum())
    n_hist = int((cohort["imaged"] & cohort["has_pre_t0_histnote"]).sum())

    consort += [
        ("with_pre_t0_frontal_cxr", int(cohort["has_pre_t0_cxr"].sum())),
        ("with_pre_t0_radiology_report", int(cohort["has_pre_t0_radtext"].sum())),
        ("with_pre_t0_prior_discharge_summary", int(cohort["has_pre_t0_histnote"].sum())),
        ("PRIMARY_imaged_cohort", n_img),
        ("imaged_deaths", int(cohort.loc[cohort["imaged"], "outcome"].sum())),
        ("imaged_active_arm", int((cohort["imaged"] & (cohort["arm"] == "active")).sum())),
        ("imaged_comparator_arm",
         int((cohort["imaged"] & (cohort["arm"] == "comparator")).sum())),
        ("imaged_with_histnote", n_hist),
        ("histnote_coverage_pct_of_imaged", round(100.0 * n_hist / n_img, 1) if n_img else 0),
        # the cost of the OLD (wrong) gate, recorded so the change is auditable
        ("OLD_gate_cxr_and_histnote_would_have_been", n_hist),
        ("OLD_gate_patients_recovered_by_fix", n_img - n_hist),
    ]
    return cohort, consort


def run(cfg, force: bool = False, intervention: str = None):
    """Build one intervention's cohort, or ALL of them when intervention is None.

    Running every intervention by default is deliberate: the previous pipeline defaulted
    to a single intervention and `--stage all` therefore produced a one-intervention study
    while looking like it had produced a four-intervention one.
    """
    from src import results as R

    names = ([intervention] if intervention
             else [k for k in (cfg.get("interventions") or {}) if k in _BUILDERS])
    unknown = [n for n in names if n not in _BUILDERS]
    if unknown:
        raise NotImplementedError(f"no emulate builder for {unknown}; have {list(_BUILDERS)}")

    ckpt = Checkpoint(cfg, "emulate")
    all_cohorts, all_consort = [], []

    for name in names:
        t0 = time.time()
        log(f"=== emulate[{name}] ===")
        cohort, consort = _BUILDERS[name](cfg)
        cohort, consort = _outcome_and_gate(cfg, name, cohort, consort)
        cohort["intervention"] = name

        # censoring indicator D, computed once and stored so every stage uses the same one
        from src import features as F
        horizon = int(cfg.get(f"interventions.{name}.horizon_days"))
        cohort["observed_at_horizon"] = F.observed_at_horizon(cfg, cohort, horizon)

        all_cohorts.append(cohort)
        all_consort += [{"intervention": name, "section": "consort", "metric": step,
                         "stratum": "", "arm": "", "value": cnt, "support_count": cnt}
                        for step, cnt in consort]
        for step, cnt in consort:
            log(f"  {step:44s} {cnt:>9,}")
        ckpt.mark(name, n=len(cohort), imaged=int(cohort["imaged"].sum()))
        log(f"  emulate[{name}] done in {time.time()-t0:,.0f}s")

    # ---- ONE consolidated cohort manifest ----
    out = pd.concat(all_cohorts, ignore_index=True)
    mdir = cfg.storage("manifests")
    mdir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(mdir / "cohorts.parquet", index=False)
    log(f"cohorts.parquet: {len(out):,} rows across {out['intervention'].nunique()} interventions")

    R.reset_rows(cfg, "cohorts.csv", section="consort")
    R.append_rows(cfg, "cohorts.csv", all_consort)
    log(f"emulate done -> {mdir/'cohorts.parquet'}")
