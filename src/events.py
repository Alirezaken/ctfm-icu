"""Shared readers over the cleaned event-stream and link layer.

The event-stream is partitioned Parquet (data/event_stream_clean/<modality>/shard=pX).
These helpers pull just the rows a stage needs (predicate pushdown on `type`), so
cohort building never loads the full ~629M-row stream.
"""
from __future__ import annotations

import pyarrow.dataset as ds
import pyarrow.compute as pc
import pandas as pd


def read_modality(cfg, modality: str, itemids=None, columns=None) -> pd.DataFrame:
    """Rows of one modality, optionally filtered to a set of `type` codes.

    `type` is stored as a string; itemids may be ints or strs.
    """
    base = cfg.input("event_stream") / modality
    dset = ds.dataset(str(base), format="parquet", partitioning="hive")
    flt = None
    if itemids is not None:
        flt = pc.field("type").isin([str(i) for i in itemids])
    cols = columns or ["subject_id", "hadm_id", "stay_id", "time",
                       "type", "value_num", "value_txt"]
    tbl = dset.to_table(columns=cols, filter=flt)
    df = tbl.to_pandas()
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df


def link(cfg, name: str) -> pd.DataFrame:
    """Read a relational link-layer table by file stem (e.g. 'patients')."""
    return pd.read_csv(cfg.input("link_dir") / f"{name}.csv")
