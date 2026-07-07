#!/usr/bin/env python3
"""#2  EVENT-STREAM  ----------------------------------------------------------

Convert MIMIC-IV source tables into the unified event-stream specified in the
proposal:  (patient_id, time, modality, type, value, is_intervention)
(+ hadm_id / stay_id for grouping, value_num/value_txt/uom for fidelity).

This is the *bulk* time-series data (hundreds of millions of rows), so it is NOT
one CSV. It is written as partitioned Parquet:

    data/event_stream/<modality>/shard=p1x/<source>_cNN_M.parquet

partitioned by `shard` = 'p' + first two digits of subject_id (p10..p19), matching
the CXR sharding. Each source is processed independently (chunked), so sources can
run in parallel as separate SLURM tasks.

Usage:
    build_event_stream.py <source> [--sample-rows N] [--chunksize N]
    build_event_stream.py all       # run every source sequentially
sources: chartevents labevents inputevents outputevents procedureevents prescriptions
"""

import os
import sys
import time
import shutil
import argparse

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import cfg as _cfg; _C = _cfg.load()
MIMIC = str(_C.input("mimic_dir"))
OUT   = str(_C.input("event_stream_raw"))

# unified output columns
COLS = ["subject_id", "hadm_id", "stay_id", "time", "modality",
        "type", "value_num", "value_txt", "uom", "is_intervention", "shard"]

# per-source mapping: file, modality, is_intervention, time col, and a builder
SOURCES = {
    "chartevents": dict(
        path=f"{MIMIC}/icu/chartevents.csv.gz", modality="chart", interv=False,
        usecols=["subject_id", "hadm_id", "stay_id", "charttime", "itemid",
                 "value", "valuenum", "valueuom"],
        time="charttime", typ="itemid", vnum="valuenum", vtxt="value", uom="valueuom"),
    "labevents": dict(
        path=f"{MIMIC}/hosp/labevents.csv.gz", modality="lab", interv=False,
        usecols=["subject_id", "hadm_id", "charttime", "itemid",
                 "value", "valuenum", "valueuom"],
        time="charttime", typ="itemid", vnum="valuenum", vtxt="value", uom="valueuom"),
    "inputevents": dict(
        path=f"{MIMIC}/icu/inputevents.csv.gz", modality="input", interv=True,
        usecols=["subject_id", "hadm_id", "stay_id", "starttime", "itemid",
                 "amount", "amountuom"],
        time="starttime", typ="itemid", vnum="amount", vtxt=None, uom="amountuom"),
    "outputevents": dict(
        path=f"{MIMIC}/icu/outputevents.csv.gz", modality="output", interv=False,
        usecols=["subject_id", "hadm_id", "stay_id", "charttime", "itemid",
                 "value", "valueuom"],
        time="charttime", typ="itemid", vnum="value", vtxt=None, uom="valueuom"),
    "procedureevents": dict(
        path=f"{MIMIC}/icu/procedureevents.csv.gz", modality="procedure", interv=True,
        usecols=["subject_id", "hadm_id", "stay_id", "starttime", "itemid",
                 "value", "valueuom"],
        time="starttime", typ="itemid", vnum="value", vtxt=None, uom="valueuom"),
    "prescriptions": dict(
        path=f"{MIMIC}/hosp/prescriptions.csv.gz", modality="med", interv=True,
        usecols=["subject_id", "hadm_id", "starttime", "drug",
                 "dose_val_rx", "dose_unit_rx"],
        time="starttime", typ="drug", vnum=None, vtxt="dose_val_rx", uom="dose_unit_rx"),
}


def transform(chunk, cfg):
    out = pd.DataFrame()
    out["subject_id"] = chunk["subject_id"].astype("int64")
    out["hadm_id"] = chunk["hadm_id"] if "hadm_id" in chunk else pd.NA
    out["stay_id"] = chunk["stay_id"] if "stay_id" in chunk else pd.NA
    out["time"] = chunk[cfg["time"]]
    out["modality"] = cfg["modality"]
    out["type"] = chunk[cfg["typ"]].astype("string")
    out["value_num"] = pd.to_numeric(chunk[cfg["vnum"]], errors="coerce") if cfg["vnum"] else pd.NA
    out["value_txt"] = chunk[cfg["vtxt"]].astype("string") if cfg["vtxt"] else pd.NA
    out["uom"] = chunk[cfg["uom"]].astype("string") if cfg["uom"] else pd.NA
    out["is_intervention"] = cfg["interv"]
    out["shard"] = "p" + out["subject_id"].astype("string").str.slice(0, 2)
    return out[COLS]


def run_source(name, sample_rows=None, chunksize=5_000_000):
    cfg = SOURCES[name]
    base = os.path.join(OUT, cfg["modality"])
    print(f"\n=== {name} -> {base}  (intervention={cfg['interv']}) ===", flush=True)
    if not sample_rows:                       # clean target for a fresh full run
        shutil.rmtree(base, ignore_errors=True)
    t0 = time.time()
    total = 0
    reader = pd.read_csv(cfg["path"], usecols=cfg["usecols"],
                         chunksize=(sample_rows or chunksize), low_memory=False)
    for i, chunk in enumerate(reader):
        ev = transform(chunk, cfg)
        tbl = pa.Table.from_pandas(ev, preserve_index=False)
        pads.write_dataset(
            tbl, base_dir=base, format="parquet",
            partitioning=["shard"], partitioning_flavor="hive",
            existing_data_behavior="overwrite_or_ignore",
            basename_template=f"{name}_c{i}_{{i}}.parquet")
        total += len(ev)
        print(f"  chunk {i:>3}: +{len(ev):>9,}  (total {total:>11,})  "
              f"{total/(time.time()-t0):,.0f} rows/s", flush=True)
        if sample_rows:
            break
    print(f"=== {name} done: {total:,} rows in {time.time()-t0:,.0f}s ===", flush=True)
    return total


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--sample-rows", type=int, default=None)
    ap.add_argument("--chunksize", type=int, default=5_000_000)
    a = ap.parse_args()
    names = list(SOURCES) if a.source == "all" else [a.source]
    for nm in names:
        run_source(nm, sample_rows=a.sample_rows, chunksize=a.chunksize)
