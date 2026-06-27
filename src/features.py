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


def pool_embeddings(cfg, cohort, modality, variant=None) -> np.ndarray:
    """Mean-pool pre-t0 vectors within look-back into one proxy per patient.

    Fallback: if no item falls in the look-back window, use the patient's most
    recent pre-t0 item (so all-modality patients always get a proxy). Patients
    with no pre-t0 item at all get the cohort-mean proxy (imputed).
    """
    lb = int(cfg.get("pooling.look_back_window_hours", 48))
    idx, V = ev.load_embeddings(cfg, modality)
    idx = idx.reset_index(drop=True)
    idx["vrow"] = np.arange(len(idx))
    tcol = "study_datetime" if modality == "images" else "charttime"
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
                proxy[i] = V[rows].mean(0)

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
