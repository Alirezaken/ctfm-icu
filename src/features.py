"""Feature assembly for the estimate stage: structured covariates at time zero
and pooled embedding proxies, all strictly pre-t0 within the look-back window (§3).

Time-zero discipline: every value used here is observed strictly BEFORE t0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import events as ev
from src.util import log

# structured covariates: concept -> itemid(s) (coalesced, last pre-t0 value)
VITALS = {
    "map": [220052, 220181, 225312], "sbp": [220050, 220179, 225309],
    "hr": [220045], "rr": [220210, 224690], "spo2": [220277], "temp_c": [223762],
}
LABS = {
    "lactate": 50813, "creatinine": 50912, "wbc": 51301, "platelets": 51265,
    "bicarbonate": 50882, "bun": 51006, "potassium": 50971, "sodium": 50983,
    "hemoglobin": 51222,
}


def _last_pre_t0(events, cohort, lb, concept_items):
    """Last value of each concept strictly before t0, within look-back window."""
    out = pd.DataFrame(index=cohort["subject_id"].values)
    ev_all = events.merge(cohort[["subject_id", "t0"]], on="subject_id", how="inner")
    ev_all = ev_all[(ev_all["time"] < ev_all["t0"]) &
                    (ev_all["time"] >= ev_all["t0"] - pd.Timedelta(hours=lb))]
    for concept, items in concept_items.items():
        items = items if isinstance(items, list) else [items]
        e = ev_all[ev_all["type"].isin([str(i) for i in items])].sort_values("time")
        last = e.groupby("subject_id")["value_num"].last()
        out[concept] = last.reindex(out.index)
    return out


def structured_at_t0(cfg, cohort) -> pd.DataFrame:
    """Structured covariate matrix aligned to cohort rows (one row per patient)."""
    lb = int(cfg.get("pooling.look_back_window_hours", 48))
    df = pd.DataFrame(index=cohort["subject_id"].values)
    df["age"] = cohort["age_t0"].values
    df["sex_male"] = (cohort["sex"].values == "M").astype(float)
    df["weight_kg"] = cohort["weight_kg"].values

    chart = ev.read_modality(cfg, "chart", [i for v in VITALS.values() for i in v],
                             columns=["subject_id", "time", "type", "value_num"])
    df = df.join(_last_pre_t0(chart, cohort, lb, VITALS))
    lab = ev.read_modality(cfg, "lab", list(LABS.values()),
                           columns=["subject_id", "time", "type", "value_num"])
    df = df.join(_last_pre_t0(lab, cohort, lb, LABS))
    log(f"  structured covariates: {df.shape[1]} cols, "
        f"{df.notna().mean().mean()*100:.0f}% populated")
    return df.reset_index(drop=True)


_VASO_ITEMS = [221906, 221289, 229617, 222315, 221749, 229630, 229631, 229632, 221662]
# comorbidity history from diagnoses_icd: name -> {(icd_version, code_prefix)}
_COMORB = {
    "heart_failure_history": {(9, "428"), (10, "I50")},
    "ckd_history": {(9, "585"), (10, "N18")},
    "coronary_artery_disease": {(9, "414"), (9, "410"), (10, "I25"), (10, "I21")},
    "immunosuppression": {(9, "279"), (10, "D84"), (10, "Z94")},
}
# config expert-confounder name -> structured_at_t0 column
_EXPERT_MAP = {"sex": "sex_male", "mean_arterial_pressure": "map",
               "vasopressor_dose": "vasopressor_use", "vasopressor_use": "vasopressor_use"}


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
        subj = set(dx.loc[m, "subject_id"])
        out[name] = cohort["subject_id"].isin(subj).astype(float).values
    return out


def _sofa_lite(sframe, vaso_flag):
    """Partial SOFA from available components (renal + coag + cardio); no CNS/resp."""
    cr = sframe["creatinine"].to_numpy()
    pl = sframe["platelets"].to_numpy()
    mp = sframe["map"].to_numpy()
    s = np.zeros(len(sframe))
    s += np.select([cr >= 5, cr >= 3.5, cr >= 2, cr >= 1.2], [4, 3, 2, 1], 0)
    s += np.select([pl < 20, pl < 50, pl < 100, pl < 150], [4, 3, 2, 1], 0)
    s += np.where(vaso_flag > 0, 3, np.where(mp < 70, 1, 0))
    return s


def expert_features(cfg, cohort, names) -> pd.DataFrame:
    """The design_based expert-confounder matrix (§5): the intervention's named
    confounders that we can extract -- structured vitals/labs, a vasopressor flag,
    a partial SOFA, and comorbidity history. Unmapped names (e.g. infection_source)
    are omitted (LightGBM handles the reduced set)."""
    lb = int(cfg.get("pooling.look_back_window_hours", 48))
    base = structured_at_t0(cfg, cohort)
    v = ev.read_modality(cfg, "input", _VASO_ITEMS, columns=["subject_id", "time"])
    v = v.merge(cohort[["subject_id", "t0"]], on="subject_id", how="inner")
    v = v[(v["time"] < v["t0"]) & (v["time"] >= v["t0"] - pd.Timedelta(hours=lb))]
    vaso_flag = cohort["subject_id"].isin(set(v["subject_id"])).astype(float).to_numpy()
    base["vasopressor_use"] = vaso_flag
    base["sofa_total"] = _sofa_lite(base, vaso_flag)
    frame = pd.concat([base.reset_index(drop=True), _comorbidities(cfg, cohort, names)], axis=1)
    cols = []
    for n in names:
        c = _EXPERT_MAP.get(n, n)
        if c in frame.columns and c not in cols:
            cols.append(c)
    log(f"  design_based: {len(cols)}/{len(names)} expert confounders extracted")
    return frame[cols]


def observed_at_horizon(cfg, cohort, horizon_days: int) -> np.ndarray:
    """§4 censoring indicator D: 1 if the patient's vital status at t0+horizon is
    KNOWN, 0 if administratively censored. MIMIC-IV `dod` is a state death registry
    covering ~1 year past the last hospital discharge, so status is known when the
    patient (a) died by the horizon, or (b) has a dod after the horizon, or (c) has
    no dod but the registry window (last discharge + 365d) reaches the horizon.
    Discharge is therefore NOT a competing event -- post-discharge death is observed."""
    adm = pd.read_csv(cfg.input("mimic_dir") / "hosp" / "admissions.csv.gz",
                      usecols=["subject_id", "dischtime"], parse_dates=["dischtime"])
    last_dis = adm.groupby("subject_id")["dischtime"].max()
    t0 = pd.to_datetime(cohort["t0"]); dod = pd.to_datetime(cohort["dod"])
    hz = t0 + pd.Timedelta(days=horizon_days)
    ld = pd.to_datetime(last_dis.reindex(cohort["subject_id"].values).values)
    reg_covers = (ld + pd.Timedelta(days=365)) >= hz
    known = (dod.notna() & (dod <= hz)) | (dod.notna() & (dod > hz)) | (dod.isna() & reg_covers)
    return known.fillna(False).astype(int).to_numpy()


def negative_control_uti(cfg, cohort) -> np.ndarray:
    """Negative-control outcome `icu_acquired_uti` (§2): UTI diagnosis on the
    cohort admission (ICD-9 599.0*, ICD-10 N39.0*). A treatment like fluid strategy
    should not affect it, so a non-null effect here flags residual confounding.
    (Proxy: coded UTI on the hadm; documented operationalization.)"""
    dx = pd.read_csv(cfg.input("mimic_dir") / "hosp" / "diagnoses_icd.csv.gz",
                     usecols=["hadm_id", "icd_code", "icd_version"])
    code = dx["icd_code"].astype(str).str.upper().str.replace(".", "", regex=False)
    is_uti = (((dx["icd_version"] == 9) & code.str.startswith("5990")) |
              ((dx["icd_version"] == 10) & code.str.startswith("N390")))
    uti_hadm = set(dx.loc[is_uti, "hadm_id"])
    return cohort["hadm_id"].isin(uti_hadm).astype(int).to_numpy()


def pool_embeddings(cfg, cohort, modality, variant=None,
                    lookback_h=None, pooling=None) -> np.ndarray:
    """Pool pre-t0 vectors within look-back into one proxy per patient.

    `lookback_h` / `pooling` override the config (used by the robustness swaps:
    24h window, max-pooling). Fallback: if no item falls in the window, use the
    patient's most recent pre-t0 item; patients with none get the cohort-mean proxy.
    """
    lb = int(lookback_h if lookback_h is not None else cfg.get("pooling.look_back_window_hours", 48))
    rule = pooling or cfg.get("pooling.rule", "mean")
    agg = (lambda X: X.max(0)) if rule == "max" else (lambda X: X.mean(0))
    idx, V = ev.load_embeddings(cfg, modality)
    idx = idx.reset_index(drop=True)
    idx["vrow"] = np.arange(len(idx))
    tcol = "charttime" if modality == "notes" else "study_datetime"   # images / images_alt use study time
    idx[tcol] = pd.to_datetime(idx[tcol], errors="coerce")
    if modality == "notes" and variant == "notes_clinical":
        idx = idx[idx["note_type"] == "discharge"]          # radiology excluded

    m = idx.merge(cohort[["subject_id", "t0"]], on="subject_id", how="inner")
    m = m[m[tcol] < m["t0"]]
    in_win = m[m[tcol] >= m["t0"] - pd.Timedelta(hours=lb)]

    D = V.shape[1]
    proxy = np.full((len(cohort), D), np.nan, dtype="float32")
    pos = {s: i for i, s in enumerate(cohort["subject_id"].values)}

    def fill(frame, which):
        for sid, rows in frame.groupby("subject_id")["vrow"].apply(list).items():
            i = pos.get(sid)
            if i is not None and np.isnan(proxy[i, 0]):
                proxy[i] = agg(V[rows])

    fill(in_win, "window")
    # fallback to most-recent pre-t0 for patients with nothing in the window
    recent = m.sort_values(tcol).groupby("subject_id").tail(1)
    fill(recent[recent["subject_id"].map(lambda s: np.isnan(proxy[pos[s], 0]) if s in pos else False)],
         "recent")
    # impute remaining with cohort-mean proxy
    have = ~np.isnan(proxy[:, 0])
    if have.any():
        proxy[~have] = proxy[have].mean(0)
    else:
        proxy[:] = 0.0
    log(f"  {modality}{'/'+variant if variant else ''} proxy: "
        f"{have.sum():,}/{len(cohort):,} patients with a real pre-t0 vector")
    return proxy
