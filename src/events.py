"""Readers over the cleaned event-stream, the link layer, and the consolidated manifests.

The event-stream is partitioned Parquet (event_stream_clean/<modality>/shard=pX). These
helpers push the itemid filter down into the Parquet scan so cohort building never loads
the full ~629M-row stream.

CONSOLIDATED MANIFESTS. Everything the analysis reads sits in exactly three files under
storage_root/manifests/, so the whole study is auditable without re-running anything:

  cohorts.parquet             one row per (intervention, patient): arm, t0, outcome,
                              censoring flag, modality-availability flags, demographics
  embeddings_index.parquet    one row per embedded item, ALL modalities in one table:
                              subject_id, ts, modality, vrow (pointer into the .npy)
  features_structured.parquet one row per (intervention, patient): the structured
                              covariates S at t0

Vectors stay in .npy blobs (a CSV of 768-d float vectors would be unreadable and
enormous). The index is the auditable part.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.compute as pc


# --------------------------------------------------------------------------- #
#  Event stream                                                                #
# --------------------------------------------------------------------------- #
def read_modality(cfg, modality: str, itemids=None, columns=None) -> pd.DataFrame:
    """Rows of one event-stream modality, optionally filtered to a set of `type` codes."""
    base = cfg.input("event_stream") / modality
    dset = ds.dataset(str(base), format="parquet", partitioning="hive")
    flt = pc.field("type").isin([str(i) for i in itemids]) if itemids is not None else None
    cols = columns or ["subject_id", "hadm_id", "stay_id", "time",
                       "type", "value_num", "value_txt"]
    df = dset.to_table(columns=cols, filter=flt).to_pandas()
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df


def link(cfg, name: str) -> pd.DataFrame:
    """A relational link-layer table by file stem (patients, icu_stays, cxr_studies, ...)."""
    return pd.read_csv(cfg.input("link_dir") / f"{name}.csv")


# --------------------------------------------------------------------------- #
#  Consolidated manifests                                                      #
# --------------------------------------------------------------------------- #
def manifest_path(cfg, name: str):
    return cfg.storage("manifests", name)


def load_cohorts(cfg, intervention: str | None = None) -> pd.DataFrame:
    """The ONE cohort table. Filter to an intervention, or get all of them."""
    df = pd.read_parquet(manifest_path(cfg, "cohorts.parquet"))
    if intervention is not None:
        df = df[df["intervention"] == intervention].reset_index(drop=True)
    return df


def load_structured(cfg, intervention: str) -> pd.DataFrame:
    """The ONE structured-feature table, filtered to an intervention."""
    df = pd.read_parquet(manifest_path(cfg, "features_structured.parquet"))
    return df[df["intervention"] == intervention].reset_index(drop=True)


def load_embeddings(cfg, modality: str):
    """(index_df, V) for one modality. index_df has subject_id, ts, vrow; V[vrow] is the
    float32 vector.

    Row alignment is ASSERTED here, not assumed. A mismatch between the index and the
    vector blob is the most dangerous silent failure available to this pipeline: it would
    attach one patient's chest X-ray to another patient, and every downstream number would
    still look perfectly reasonable.
    """
    idx = pd.read_parquet(manifest_path(cfg, "embeddings_index.parquet"))
    idx = idx[idx["modality"] == modality].reset_index(drop=True)
    if not len(idx):
        raise ValueError(f"no rows in embeddings_index.parquet for modality '{modality}'")

    V = np.load(cfg.storage("embeddings", f"{modality}.npy")).astype("float32")
    if len(idx) != len(V):
        raise ValueError(
            f"{modality}: {len(idx)} index rows vs {len(V)} vectors -- out of sync")
    if int(idx["vrow"].max()) >= len(V):
        raise ValueError(
            f"{modality}: index references row {int(idx['vrow'].max())} but the vector "
            f"blob has only {len(V)} rows -- out of sync")
    return idx, V
