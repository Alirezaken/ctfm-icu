#!/usr/bin/env python3
"""Value-cleaning pass over the event-stream.

Reads the RAW event-stream (data/event_stream/) and writes a cleaned copy to
data/event_stream_clean/. The raw data is NEVER modified, so this is fully
reversible: to undo, just delete data/event_stream_clean/.

Cleaning rule (applied to `chart` and `lab` only; interventions/outputs are
amounts/durations and are passed through unchanged):
  - For curated vital/lab itemids with a known physiological range: keep a row
    only if its value_num is within [lo, hi]; drop impossible readings and rows
    whose value_num is null for these numeric items.
  - For other chart/lab items (no curated range): drop only absurd magnitudes
    (|value_num| > 1e6) as a safety net for documentation/sentinel errors.

Usage: clean_event_stream.py [modality|all]   (default: all)
"""
import os
import sys
import time
import shutil
import argparse

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import EVENT_DIR, EVENT_CLEAN

# curated physiological ranges by itemid (lo, hi)
RANGES = {
    # ---- vitals (chart / d_items) ----
    "220045": (0, 300),    # Heart Rate
    "220210": (0, 80),     # Respiratory Rate
    "224690": (0, 80),     # Respiratory Rate (Total)
    "220277": (0, 100),    # SpO2
    "220179": (0, 300), "220180": (0, 200), "220181": (0, 250),  # NBP sys/dia/mean
    "220050": (0, 300), "220051": (0, 200), "220052": (0, 250),  # ABP sys/dia/mean
    "223761": (70, 115),   # Temp F
    "223762": (20, 45),    # Temp C
    # ---- labs (lab / d_labitems) ----
    "50912": (0, 50),      # Creatinine
    "50983": (50, 200),    # Sodium
    "50971": (0, 15),      # Potassium
    "50902": (50, 200),    # Chloride
    "50882": (0, 60),      # Bicarbonate
    "50868": (0, 50),      # Anion Gap
    "51006": (0, 300),     # Urea Nitrogen (BUN)
    "50931": (0, 2000),    # Glucose
    "50813": (0, 40),      # Lactate
    "51222": (0, 30),      # Hemoglobin
    "51221": (0, 100),     # Hematocrit
    "51301": (0, 200),     # WBC
    "51265": (0, 2000),    # Platelet
    "50960": (0, 10),      # Magnesium
    "50970": (0, 30),      # Phosphate
    "50893": (0, 30),      # Calcium
    "51237": (0, 30),      # INR(PT)
    "50885": (0, 100),     # Bilirubin
}
LO = {k: v[0] for k, v in RANGES.items()}
HI = {k: v[1] for k, v in RANGES.items()}
GLOBAL_CAP = 1e6
CLEAN_MODS = {"chart", "lab"}


def clean_df(df):
    vn = df["value_num"]
    ranged = df["type"].isin(RANGES)
    lo = df["type"].map(LO)
    hi = df["type"].map(HI)
    drop_ranged = ranged & (vn.isna() | (vn < lo) | (vn > hi))
    drop_global = (~ranged) & vn.notna() & (vn.abs() > GLOBAL_CAP)
    drop = drop_ranged | drop_global
    return df[~drop], int(drop.sum())


def run(modality):
    src = os.path.join(EVENT_DIR, modality)
    dst = os.path.join(EVENT_CLEAN, modality)
    shutil.rmtree(dst, ignore_errors=True)
    passthrough = modality not in CLEAN_MODS
    print(f"\n=== {modality}  ({'pass-through' if passthrough else 'cleaning'}) "
          f"{src} -> {dst} ===", flush=True)
    d = pads.dataset(src, format="parquet", partitioning="hive")
    t0 = time.time()
    total = kept = dropped = 0
    for i, batch in enumerate(d.scanner(batch_size=2_000_000).to_batches()):
        df = batch.to_pandas()
        total += len(df)
        if passthrough:
            out = df
        else:
            out, nd = clean_df(df)
            dropped += nd
        kept += len(out)
        if len(out):
            tbl = pa.Table.from_pandas(out, preserve_index=False)
            pads.write_dataset(
                tbl, base_dir=dst, format="parquet",
                partitioning=["shard"], partitioning_flavor="hive",
                existing_data_behavior="overwrite_or_ignore",
                basename_template=f"{modality}_c{i}_{{i}}.parquet")
    pct = 100.0 * dropped / total if total else 0.0
    print(f"=== {modality} done: kept {kept:,} / {total:,}  "
          f"dropped {dropped:,} ({pct:.3f}%)  in {time.time()-t0:,.0f}s ===", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("modality", nargs="?", default="all")
    a = ap.parse_args()
    mods = (["chart", "lab", "input", "output", "procedure", "med"]
            if a.modality == "all" else [a.modality])
    for m in mods:
        run(m)
    print("\nclean pass complete ->", EVENT_CLEAN, flush=True)
