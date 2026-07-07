#!/usr/bin/env python3
"""Quality-control / verification pass over the unified event-stream.

Streams every modality once (memory-safe) and reports:
  - row counts per modality (+ on-disk size, shard completeness p10..p19)
  - value_num coverage & stats (non-null %, min/max/mean)
  - value_txt / uom coverage
  - time range (min/max)
  - is_intervention consistency (should be constant per modality)
  - plausibility checks on key vital signs & labs (flag out-of-range min/max)

Writes data/qc_report.txt and prints it.
"""
import os
import sys

import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import cfg as _cfg; _C = _cfg.load()
MIMIC_DIR   = str(_C.input("mimic_dir"))
DATA_DIR    = str(_C.input("data_dir"))
EVENT_DIR   = str(_C.input("event_stream_raw"))
EVENT_CLEAN = str(_C.input("event_stream"))

# `qc_event_stream.py clean` -> QC the cleaned copy; default -> QC the raw stream
_CLEAN = len(sys.argv) > 1 and sys.argv[1] == "clean"
EVENT = EVENT_CLEAN if _CLEAN else EVENT_DIR
REPORT = os.path.join(DATA_DIR, "qc_report_clean.txt" if _CLEAN else "qc_report.txt")
EXPECT_SHARDS = {f"p{n}" for n in range(10, 20)}

# key items to spot-check, with plausible physiological ranges (lo, hi)
VITALS = {  # chart (icu d_items)
    "220045": ("Heart Rate", 0, 350),
    "220210": ("Respiratory Rate", 0, 80),
    "220277": ("SpO2", 0, 100),
    "220179": ("NBP systolic", 0, 300),
    "220050": ("ABP systolic", 0, 300),
    "223761": ("Temp (F)", 70, 115),
    "223762": ("Temp (C)", 20, 45),
}
LABS = {  # lab (hosp d_labitems)
    "50912": ("Creatinine", 0, 50),
    "50983": ("Sodium", 50, 200),
    "50971": ("Potassium", 0, 15),
    "50931": ("Glucose", 0, 2000),
    "51222": ("Hemoglobin", 0, 30),
    "50813": ("Lactate", 0, 40),
    "51301": ("WBC", 0, 200),
    "50882": ("Bicarbonate", 0, 60),
}
TARGETS = {"chart": VITALS, "lab": LABS}


def dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / 1e9


def qc_modality(name):
    base = os.path.join(EVENT, name)
    d = ds.dataset(base, format="parquet", partitioning="hive")
    shards = {os.path.basename(os.path.dirname(p)).split("=")[1] for p in d.files}
    targets = TARGETS.get(name, {})
    tgt_ids = set(targets)

    n = vn_nn = vt_nn = uom_nn = 0
    vmin = vmax = vsum = None
    tmin = tmax = None
    interv_vals = set()
    per_type = {}  # itemid -> [count, min, max, sum]

    cols = ["time", "type", "value_num", "value_txt", "uom", "is_intervention"]
    for batch in d.scanner(columns=cols, batch_size=1_000_000).to_batches():
        df = batch.to_pandas()
        n += len(df)
        vn = df["value_num"]
        m = vn.notna()
        vn_nn += int(m.sum())
        if m.any():
            bmin, bmax, bsum = vn.min(), vn.max(), vn.sum()
            vmin = bmin if vmin is None else min(vmin, bmin)
            vmax = bmax if vmax is None else max(vmax, bmax)
            vsum = bsum if vsum is None else vsum + bsum
        vt_nn += int(df["value_txt"].notna().sum())
        uom_nn += int(df["uom"].notna().sum())
        t = df["time"].dropna()
        if len(t):
            bmn, bmx = t.min(), t.max()
            tmin = bmn if tmin is None else min(tmin, bmn)
            tmax = bmx if tmax is None else max(tmax, bmx)
        interv_vals.update(df["is_intervention"].dropna().unique().tolist())
        if tgt_ids:
            sub = df[df["type"].isin(tgt_ids)]
            if len(sub):
                g = sub.groupby("type")["value_num"].agg(["count", "min", "max", "sum"])
                for tid, r in g.iterrows():
                    a = per_type.setdefault(tid, [0, None, None, 0.0])
                    a[0] += int(r["count"])
                    a[1] = r["min"] if a[1] is None else min(a[1], r["min"])
                    a[2] = r["max"] if a[2] is None else max(a[2], r["max"])
                    a[3] += 0.0 if pd.isna(r["sum"]) else float(r["sum"])

    L = []
    L.append(f"\n## {name}   ({n:,} events)")
    miss = EXPECT_SHARDS - shards
    L.append(f"   shards: {len(shards)}/10 present" + (f"  MISSING {sorted(miss)}" if miss else "  (p10-p19 OK)"))
    L.append(f"   is_intervention: {sorted(interv_vals)}" +
             ("" if len(interv_vals) == 1 else "   <-- WARN: not constant!"))
    L.append(f"   time range: {tmin}  ->  {tmax}")
    pct = lambda x: f"{100.0 * x / n:.1f}%" if n else "n/a"
    L.append(f"   value_num: {pct(vn_nn)} non-null"
             + (f"  range [{vmin:.2f}, {vmax:.2f}]  mean {vsum / vn_nn:.2f}" if vn_nn else ""))
    L.append(f"   value_txt: {pct(vt_nn)} non-null    uom: {pct(uom_nn)} non-null")
    if targets:
        L.append(f"   key items (plausibility):")
        for tid, (lbl, lo, hi) in targets.items():
            a = per_type.get(tid)
            if not a or a[0] == 0:
                L.append(f"      {lbl:16s} (item {tid}): NOT FOUND")
                continue
            mean = a[3] / a[0]
            flag = "  <-- OUT OF RANGE" if (a[1] < lo or a[2] > hi) else ""
            L.append(f"      {lbl:16s} n={a[0]:>10,}  min={a[1]:>8.1f} max={a[2]:>8.1f} "
                     f"mean={mean:>7.1f}  (expect {lo}-{hi}){flag}")
    return "\n".join(L), n


def main():
    out = ["EVENT-STREAM QC REPORT", f"generated: {pd.Timestamp.now():%Y-%m-%d %H:%M}",
           f"event_stream dir: {EVENT}",
           f"on-disk size: {dir_size(EVENT):.2f} GB"]
    grand = 0
    for m in ["chart", "lab", "input", "output", "procedure", "med"]:
        if os.path.isdir(os.path.join(EVENT, m)):
            txt, n = qc_modality(m)
            out.append(txt)
            grand += n
    out.append(f"\n## TOTAL events: {grand:,}")
    report = "\n".join(out)
    with open(REPORT, "w") as f:
        f.write(report + "\n")
    print(report)
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
