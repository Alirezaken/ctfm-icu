"""Feature assembly: structured covariates at t0, the expert confounder set, the
censoring indicator, the negative-control outcome, and pooled embedding proxies.

TIME-ZERO DISCIPLINE: every value used for adjustment is observed STRICTLY BEFORE t0,
within the look-back window. This is asserted, not assumed -- see stages/integrity.py.

The three embedding modalities are named for what they actually are:
  images    RAD-DINO on pre-t0 frontal chest X-rays               (pixels)
  radtext   Clinical-Longformer on pre-t0 RADIOLOGY REPORTS       (contemporaneous expert text)
  histnote  Clinical-Longformer on pre-t0 DISCHARGE SUMMARIES     (PRIOR-admission history)

The `histnote` naming is load-bearing. A discharge summary's charttime is the discharge
time of its own hospitalization; t0 falls during the CURRENT ICU stay; so any discharge
summary observed before t0 necessarily belongs to an EARLIER hospitalization. It is a
history narrative, not a contemporaneous clinical note. MIMIC-IV-Note contains no nursing
or progress notes, so no contemporaneous free-text clinical narrative exists in this
dataset at all. Calling this channel "clinical notes" would misdescribe the evidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import events as ev
from src.util import log

# ---- structured covariate dictionary: concept -> itemid(s), coalesced --------
VITALS = {
    "map": [220052, 220181, 225312],
    "sbp": [220050, 220179, 225309],
    "hr": [220045],
    "rr": [220210, 224690],
    "spo2": [220277],
    "temp_c": [223762],
    "fio2": [223835],
    "peep": [220339, 224700],
    "plateau_pressure": [224696],
    "gcs": [220739, 223900, 223901],      # eye + verbal + motor (summed below)
}
LABS = {
    "lactate": 50813, "creatinine": 50912, "wbc": 51301, "platelets": 51265,
    "bicarbonate": 50882, "bun": 51006, "potassium": 50971, "sodium": 50983,
    "hemoglobin": 51222, "ph": 50820, "po2": 50821, "bilirubin": 50885, "inr": 51237,
}
_VASO_ITEMS = [221906, 221289, 229617, 222315, 221749, 229630, 229631, 229632, 221662]
_VENT_ITEMS = [223848, 223849, 220339]   # ventilator mode / type -> mechanical ventilation flag
_URINE_ITEMS = [226559, 226560, 226561, 226584, 226563, 226564, 226565,
                226567, 226557, 226558, 227488, 227489]

_COMORB = {
    "heart_failure_history": {(9, "428"), (10, "I50")},
    "ckd_history": {(9, "585"), (10, "N18")},
    "coronary_artery_disease": {(9, "414"), (9, "410"), (10, "I25"), (10, "I21")},
    "immunosuppression": {(9, "279"), (10, "D84"), (10, "Z94")},
}

# config expert-confounder name -> the structured column that carries it
_EXPERT_MAP = {
    "sex": "sex_male",
    "mean_arterial_pressure": "map",
    "vasopressor_dose": "vasopressor_use",
    "vasopressor_use": "vasopressor_use",
    "pao2_fio2": "pao2_fio2",
    "mechanical_ventilation": "mechanical_ventilation",
    "urine_output_6h": "urine_output_6h",
}


def _last_pre_t0(events, cohort, lb, concept_items):
    """Last value of each concept STRICTLY before t0, within the look-back window."""
    out = pd.DataFrame(index=cohort["subject_id"].values)
    if not len(events):
        for c in concept_items:
            out[c] = np.nan
        return out
    e_all = events.merge(cohort[["subject_id", "t0"]], on="subject_id", how="inner")
    e_all = e_all[(e_all["time"] < e_all["t0"]) &
                  (e_all["time"] >= e_all["t0"] - pd.Timedelta(hours=lb))]
    for concept, items in concept_items.items():
        items = items if isinstance(items, list) else [items]
        e = e_all[e_all["type"].isin([str(i) for i in items])].sort_values("time")
        out[concept] = e.groupby("subject_id")["value_num"].last().reindex(out.index)
    return out


def structured_at_t0(cfg, cohort) -> pd.DataFrame:
    """The structured adjustment set S: one row per cohort patient, all pre-t0."""
    lb = int(cfg.get("pooling.look_back_window_hours", 48))
    df = pd.DataFrame(index=cohort["subject_id"].values)
    df["age"] = cohort["age_t0"].values
    df["sex_male"] = (cohort["sex"].values == "M").astype(float)
    df["weight_kg"] = cohort["weight_kg"].values

    chart = ev.read_modality(cfg, "chart",
                             [i for v in VITALS.values() for i in v],
                             columns=["subject_id", "time", "type", "value_num"])
    df = df.join(_last_pre_t0(chart, cohort, lb, VITALS))

    lab = ev.read_modality(cfg, "lab", list(LABS.values()),
                           columns=["subject_id", "time", "type", "value_num"])
    df = df.join(_last_pre_t0(lab, cohort, lb, LABS))

    # derived
    with np.errstate(invalid="ignore", divide="ignore"):
        df["pao2_fio2"] = df["po2"] / df["fio2"].replace(0, np.nan)

    # vasopressor exposure pre-t0
    v = ev.read_modality(cfg, "input", _VASO_ITEMS, columns=["subject_id", "time"])
    v = v.merge(cohort[["subject_id", "t0"]], on="subject_id", how="inner")
    v = v[(v["time"] < v["t0"]) & (v["time"] >= v["t0"] - pd.Timedelta(hours=lb))]
    vaso = cohort["subject_id"].isin(set(v["subject_id"])).astype(float).to_numpy()
    df["vasopressor_use"] = vaso

    # mechanical ventilation pre-t0
    mv = ev.read_modality(cfg, "chart", _VENT_ITEMS, columns=["subject_id", "time"])
    mv = mv.merge(cohort[["subject_id", "t0"]], on="subject_id", how="inner")
    mv = mv[(mv["time"] < mv["t0"]) & (mv["time"] >= mv["t0"] - pd.Timedelta(hours=lb))]
    df["mechanical_ventilation"] = cohort["subject_id"].isin(
        set(mv["subject_id"])).astype(float).to_numpy()

    # urine output in the 6h before t0
    uo = ev.read_modality(cfg, "output", _URINE_ITEMS,
                          columns=["subject_id", "time", "value_num"])
    uo = uo.merge(cohort[["subject_id", "t0"]], on="subject_id", how="inner")
    uo6 = uo[(uo["time"] < uo["t0"]) & (uo["time"] >= uo["t0"] - pd.Timedelta(hours=6))]
    df["urine_output_6h"] = (uo6.groupby("subject_id")["value_num"].sum()
                             .reindex(df.index).to_numpy())

    df["sofa_total"] = _sofa(df, vaso)

    log(f"  structured S: {df.shape[1]} covariates, "
        f"{df.notna().mean().mean()*100:.0f}% populated")
    return df.reset_index(drop=True)


def _sofa(f, vaso_flag):
    """SOFA from the components extractable here (resp, coag, liver, cardio, cns, renal)."""
    def col(c):
        return f[c].to_numpy() if c in f.columns else np.full(len(f), np.nan)
    pf, pl, bi, mp, gc, cr = (col("pao2_fio2"), col("platelets"), col("bilirubin"),
                              col("map"), col("gcs"), col("creatinine"))
    s = np.zeros(len(f))
    s += np.select([pf < 100, pf < 200, pf < 300, pf < 400], [4, 3, 2, 1], 0)
    s += np.select([pl < 20, pl < 50, pl < 100, pl < 150], [4, 3, 2, 1], 0)
    s += np.select([bi >= 12, bi >= 6, bi >= 2, bi >= 1.2], [4, 3, 2, 1], 0)
    s += np.where(vaso_flag > 0, 3, np.where(mp < 70, 1, 0))
    s += np.select([gc < 6, gc < 10, gc < 13, gc < 15], [4, 3, 2, 1], 0)
    s += np.select([cr >= 5, cr >= 3.5, cr >= 2, cr >= 1.2], [4, 3, 2, 1], 0)
    return s


def _comorbidities(cfg, cohort, names):
    dx = pd.read_csv(cfg.input("mimic_dir") / "hosp" / "diagnoses_icd.csv.gz",
                     usecols=["subject_id", "icd_code", "icd_version"])
    dx["code"] = dx["icd_code"].astype(str).str.upper()
    out = pd.DataFrame(index=range(len(cohort)))
    for name in names:
        prefs = _COMORB.get(name)
        if not prefs:
            continue
        m = pd.Series(False, index=dx.index)
        for ver, pre in prefs:
            m |= (dx["icd_version"] == ver) & dx["code"].str.startswith(pre)
        out[name] = cohort["subject_id"].isin(set(dx.loc[m, "subject_id"])).astype(float).values
    return out


def expert_features(cfg, cohort, names):
    """The `expert` condition: the clinician-curated confounder list for this intervention.

    Returns (frame, extracted, requested). The caller MUST record extracted/requested in
    the results, because an expert arm running on a partial confounder set is not the
    competitor the paper claims to beat. integrity.py asserts the ratio clears a floor.
    """
    base = structured_at_t0(cfg, cohort)
    frame = pd.concat([base.reset_index(drop=True),
                       _comorbidities(cfg, cohort, names)], axis=1)
    cols = []
    missing = []
    for n in names:
        c = _EXPERT_MAP.get(n, n)
        if c in frame.columns and c not in cols:
            cols.append(c)
        elif c not in frame.columns:
            missing.append(n)
    log(f"  expert set: {len(cols)}/{len(names)} confounders extracted"
        + (f"  MISSING: {missing}" if missing else ""))
    return frame[cols], len(cols), len(names)


def observed_at_horizon(cfg, cohort, horizon_days: int) -> np.ndarray:
    """Censoring indicator D: 1 if vital status at t0+horizon is KNOWN, 0 if censored.

    MIMIC-IV `dod` comes from a state death registry covering roughly one year past the
    last hospital discharge, so status is known when the patient (a) died by the horizon,
    (b) has a dod after the horizon, or (c) has no dod and the registry window reaches the
    horizon. Discharge is therefore NOT a competing event here: post-discharge death is
    observed. (This is the opposite of eICU -- see stages/external.py.)
    """
    adm = pd.read_csv(cfg.input("mimic_dir") / "hosp" / "admissions.csv.gz",
                      usecols=["subject_id", "dischtime"], parse_dates=["dischtime"])
    last_dis = adm.groupby("subject_id")["dischtime"].max()
    t0 = pd.to_datetime(cohort["t0"])
    dod = pd.to_datetime(cohort["dod"])
    hz = t0 + pd.Timedelta(days=horizon_days)
    ld = pd.to_datetime(last_dis.reindex(cohort["subject_id"].values).values)
    reg_covers = (ld + pd.Timedelta(days=365)) >= hz
    known = dod.notna() | (dod.isna() & pd.Series(reg_covers, index=dod.index))
    return known.fillna(False).astype(int).to_numpy()


def negative_control_uti(cfg, cohort) -> np.ndarray:
    """Negative-control outcome: ICU-acquired UTI (ICD-9 599.0*, ICD-10 N39.0*).

    No treatment studied here plausibly causes a UTI, so a non-null effect on this outcome
    flags residual confounding. Coded on the cohort admission; a documented proxy.
    """
    dx = pd.read_csv(cfg.input("mimic_dir") / "hosp" / "diagnoses_icd.csv.gz",
                     usecols=["hadm_id", "icd_code", "icd_version"])
    code = dx["icd_code"].astype(str).str.upper().str.replace(".", "", regex=False)
    is_uti = (((dx["icd_version"] == 9) & code.str.startswith("5990")) |
              ((dx["icd_version"] == 10) & code.str.startswith("N390")))
    return cohort["hadm_id"].isin(set(dx.loc[is_uti, "hadm_id"])).astype(int).to_numpy()


# --------------------------------------------------------------------------- #
#  Embedding pooling                                                           #
# --------------------------------------------------------------------------- #
def pool_embeddings(cfg, cohort, modality, lookback_h=None, pooling=None):
    """Pool a patient's pre-t0 vectors for one modality into a single proxy.

    Returns (proxy, has_real) where has_real marks patients who actually had a pre-t0
    item of this modality. `has_real` matters: for `histnote` roughly 45% of the imaged
    cohort has no prior-admission discharge summary at all, and those patients get a
    cohort-mean proxy plus a missingness indicator rather than a silent imputation.
    """
    lb = int(lookback_h if lookback_h is not None
             else cfg.get("pooling.look_back_window_hours", 48))
    rule = pooling or cfg.get("pooling.rule", "mean")
    agg = (lambda X: X.max(0)) if rule == "max" else (lambda X: X.mean(0))

    idx, V = ev.load_embeddings(cfg, modality)
    idx = idx.reset_index(drop=True)
    idx["vrow"] = np.arange(len(idx))
    idx["ts"] = pd.to_datetime(idx["ts"], errors="coerce")

    m = idx.merge(cohort[["subject_id", "t0"]], on="subject_id", how="inner")
    m = m[m["ts"] < m["t0"]]                                    # STRICTLY pre-t0
    in_win = m[m["ts"] >= m["t0"] - pd.Timedelta(hours=lb)]

    D = V.shape[1]
    proxy = np.full((len(cohort), D), np.nan, dtype="float32")
    pos = {s: i for i, s in enumerate(cohort["subject_id"].values)}

    for sid, rows in in_win.groupby("subject_id")["vrow"].apply(list).items():
        i = pos.get(sid)
        if i is not None:
            proxy[i] = agg(V[rows])

    # fall back to the most recent pre-t0 item outside the window
    remaining = m[~m["subject_id"].isin(in_win["subject_id"])]
    recent = remaining.sort_values("ts").groupby("subject_id").tail(1)
    for sid, rows in recent.groupby("subject_id")["vrow"].apply(list).items():
        i = pos.get(sid)
        if i is not None and np.isnan(proxy[i, 0]):
            proxy[i] = agg(V[rows])

    has_real = ~np.isnan(proxy[:, 0])
    if has_real.any():
        proxy[~has_real] = proxy[has_real].mean(0)             # cohort-mean for the rest
    else:
        proxy[:] = 0.0

    log(f"  {modality}: {has_real.sum():,}/{len(cohort):,} patients "
        f"({100*has_real.mean():.0f}%) have a real pre-t0 vector")
    return proxy, has_real


def modality_block(cfg, cohort, modality, lookback_h=None, pooling=None):
    """One modality's design block: the pooled proxy, plus a missingness indicator column
    when coverage is partial (histnote). The indicator is a real covariate: having had a
    prior hospitalization is itself prognostic and predicts treatment."""
    proxy, has_real = pool_embeddings(cfg, cohort, modality, lookback_h, pooling)
    if has_real.all():
        return proxy
    return np.hstack([proxy, (~has_real).astype("float32").reshape(-1, 1)])
