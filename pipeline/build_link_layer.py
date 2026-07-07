#!/usr/bin/env python3
"""#1  RELATIONAL LINK LAYER  --------------------------------------------------

Instead of one giant flat CSV, produce a small set of single-grain tables joined
by keys (subject_id, hadm_id, stay_id, study_id) -- a star schema, like MIMIC
itself. These are the *index* over the datasets; the bulk time-series lives in the
separate event-stream (build_event_stream.py).

Outputs (data/link/):
  patients.csv      1 row / patient      subject_id -> demographics
  admissions.csv    1 row / hosp adm     hadm_id    -> times, type, outcome, came_via_ed
  icu_stays.csv     1 row / ICU stay     stay_id    -> hadm_id, unit, in/out, los
  ed_stays.csv      1 row / ED visit     stay_id    -> hadm_id, in/out, disposition
  cxr_studies.csv   1 row / CXR study    study_id   -> time, episode link, labels
  cohort_index.csv  1 row / patient      flags + counts; the join "spine"
"""

import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import cfg as _cfg; _C = _cfg.load()
MIMIC  = str(_C.input("mimic_dir"))
ED     = str(_C.input("ed_dir"))
CXR    = str(_C.input("cxr_master"))
OUT    = str(_C.input("link_dir"))
LABELS = list(_C.get("chexpert_labels"))

os.makedirs(OUT, exist_ok=True)


def w(df, name):
    p = os.path.join(OUT, name)
    df.to_csv(p, index=False)
    print(f"  wrote {name:18s} {len(df):>9,} rows x {df.shape[1]} cols", flush=True)


# ----------------------------------------------------------------- patients
print("patients ...", flush=True)
pat = pd.read_csv(os.path.join(MIMIC, "hosp/patients.csv.gz"),
                  usecols=["subject_id", "gender", "anchor_age", "anchor_year",
                           "anchor_year_group", "dod"])
pat["deceased"] = pat["dod"].notna()
w(pat, "patients.csv")

# ----------------------------------------------------------------- admissions
print("admissions ...", flush=True)
adm = pd.read_csv(os.path.join(MIMIC, "hosp/admissions.csv.gz"),
                  usecols=["subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
                           "admission_type", "admission_location", "discharge_location",
                           "insurance", "race", "hospital_expire_flag",
                           "edregtime", "edouttime"],
                  parse_dates=["admittime", "dischtime", "edregtime", "edouttime"])
adm["came_via_ed"] = adm["edregtime"].notna()
adm["ed_los_hours"] = ((adm["edouttime"] - adm["edregtime"]).dt.total_seconds() / 3600).round(2)
adm = adm.drop(columns=["edregtime", "edouttime"])
w(adm, "admissions.csv")

# ----------------------------------------------------------------- icu_stays
print("icu_stays ...", flush=True)
icu = pd.read_csv(os.path.join(MIMIC, "icu/icustays.csv.gz"),
                  usecols=["subject_id", "hadm_id", "stay_id",
                           "first_careunit", "last_careunit", "intime", "outtime", "los"])
w(icu, "icu_stays.csv")

# ----------------------------------------------------------------- ed_stays
print("ed_stays ...", flush=True)
eds = pd.read_csv(os.path.join(ED, "edstays.csv.gz"),
                  usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime",
                           "disposition", "arrival_transport"])
w(eds, "ed_stays.csv")

# ----------------------------------------------------------------- cxr_studies (+ episode link)
print("cxr_studies (temporal episode linkage) ...", flush=True)
cxr = pd.read_csv(CXR, low_memory=False)
cxr["subject_id"] = pd.to_numeric(cxr["subject_id"], errors="coerce")
cxr = cxr.dropna(subset=["subject_id"]).copy()
cxr["subject_id"] = cxr["subject_id"].astype("int64")

# study_datetime from StudyDate (YYYYMMDD) + StudyTime (HHMMSS.fff)
d = pd.to_numeric(cxr["StudyDate"], errors="coerce")
t = pd.to_numeric(cxr["StudyTime"], errors="coerce")
ds = d.astype("Int64").astype("string").str.zfill(8)
ts = t.fillna(0).astype("int64").astype("string").str.zfill(6).str.slice(0, 6)
cxr["study_datetime"] = pd.to_datetime(ds + ts, format="%Y%m%d%H%M%S", errors="coerce")

# collapse images -> studies
agg = {"study_datetime": "first", "split": "first", "subset": "first",
       "view": lambda s: "|".join(sorted(set(s.dropna()))), "jpg_rel_path": "size"}
for lab in LABELS:
    agg[lab] = "first"
studies = cxr.groupby(["subject_id", "study_id"], as_index=False).agg(agg)
studies = studies.rename(columns={"view": "views", "jpg_rel_path": "n_images",
                                  "subset": "shard"})

# build per-subject interval lists in ONE fast pass (numpy arrays + defaultdict;
# pandas groupby over 200k+ subjects is far too slow here)
def intervals_dict(df, tin, tout, vid):
    d = defaultdict(list)
    sids = df["subject_id"].to_numpy()
    tis = pd.to_datetime(df[tin]).to_numpy()
    tos = pd.to_datetime(df[tout]).to_numpy()
    vids = df[vid].to_numpy()
    for s, ti, to, v in zip(sids, tis, tos, vids):
        d[s].append((ti, to, v))
    return d


adm_i = intervals_dict(adm, "admittime", "dischtime", "hadm_id")
icu_i = intervals_dict(icu, "intime", "outtime", "stay_id")
ed_i = intervals_dict(eds, "intime", "outtime", "stay_id")


def match(table, sid, ts):
    # ti==ti is False for NaT; to!=to is True for NaT (open interval)
    for ti, to, vid in table.get(sid, ()):
        if ti == ti and ti <= ts and (to != to or ts <= to):
            return vid, ti
    return None, None


HOUR = np.timedelta64(1, "h")
hadm, adt, icus, eds_id, enc, hfa = [], [], [], [], [], []
for sid, ts in zip(studies["subject_id"].to_numpy(), studies["study_datetime"].to_numpy()):
    if ts != ts:  # NaT
        hadm.append(pd.NA); adt.append(pd.NaT); icus.append(pd.NA); eds_id.append(pd.NA)
        enc.append("UNKNOWN"); hfa.append(pd.NA); continue
    h, a = match(adm_i, sid, ts)
    ic, _ = match(icu_i, sid, ts)
    e, _ = match(ed_i, sid, ts)
    hadm.append(h if h is not None else pd.NA)
    adt.append(a if a is not None else pd.NaT)
    icus.append(ic if ic is not None else pd.NA)
    eds_id.append(e if e is not None else pd.NA)
    enc.append("ICU" if ic is not None else "INPATIENT" if h is not None
               else "ED" if e is not None else "OUTPATIENT")
    hfa.append(round(float((ts - a) / HOUR), 2) if a is not None else pd.NA)

studies["encounter_type"] = enc
studies["hadm_id"] = hadm
studies["icu_stay_id"] = icus
studies["ed_stay_id"] = eds_id
studies["hours_from_admit"] = hfa
studies = studies[["subject_id", "study_id", "study_datetime", "encounter_type",
                   "hadm_id", "icu_stay_id", "ed_stay_id", "hours_from_admit",
                   "n_images", "views", "split", "shard"] + LABELS]
w(studies, "cxr_studies.csv")

# ----------------------------------------------------------------- cohort_index (spine)
print("cohort_index ...", flush=True)
idx = pat[["subject_id"]].copy()
idx = idx.merge(adm.groupby("subject_id").size().rename("n_admissions"), on="subject_id", how="outer")
idx = idx.merge(icu.groupby("subject_id")["stay_id"].nunique().rename("n_icu_stays"), on="subject_id", how="left")
idx = idx.merge(eds.groupby("subject_id")["stay_id"].nunique().rename("n_ed_stays"), on="subject_id", how="left")
idx = idx.merge(studies.groupby("subject_id")["study_id"].nunique().rename("n_cxr_studies"), on="subject_id", how="left")
for c in ["n_admissions", "n_icu_stays", "n_ed_stays", "n_cxr_studies"]:
    idx[c] = idx[c].fillna(0).astype("int64")
idx["has_ehr"] = idx["subject_id"].isin(pat["subject_id"]) | (idx["n_admissions"] > 0)
idx["has_icu"] = idx["n_icu_stays"] > 0
idx["has_ed"] = idx["n_ed_stays"] > 0
idx["has_cxr"] = idx["n_cxr_studies"] > 0
idx["in_multimodal_cohort"] = idx["has_ehr"] & idx["has_cxr"]
idx = idx.sort_values("subject_id").reset_index(drop=True)
w(idx, "cohort_index.csv")

print("\nlink layer complete ->", OUT)
