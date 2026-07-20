"""extract_embeddings -- the ONLY GPU stage. Embed once, store, never recompute.

Produces three modalities, plus (optionally) a fourth for the encoder-swap robustness:

  images     RAD-DINO over pre-t0 FRONTAL chest X-rays          (pixels)
  radtext    Clinical-Longformer over RADIOLOGY REPORTS         (contemporaneous expert text)
  histnote   Clinical-Longformer over DISCHARGE SUMMARIES       (PRIOR-admission history)
  images_alt BiomedCLIP over the same X-rays                    (robustness swap only)

Writes:
  embeddings/<modality>.npy              float16 vectors, one row per item
  manifests/embeddings_index.parquet     ONE table, ALL modalities:
                                         subject_id, ts, modality, vrow, source_id

`vrow` is the row index into that modality's .npy. events.load_embeddings() ASSERTS that
the index length and the blob length agree, because a silent misalignment here would
attach one patient's chest X-ray to another and every downstream number would still look
completely plausible.

Scoping: only items belonging to patients who appear in cohorts.parquet are embedded.
Embedding all of MIMIC-CXR would be a large multiple of the compute for vectors the study
never reads. Run `emulate` first.

Checkpointed per (modality, shard) so a preempted job resumes instead of restarting.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd

from src.util import log, Checkpoint
from src import events as ev

_IMG_BATCH = 64
_TXT_BATCH = 8


def _device(torch):
    return "cuda" if torch.cuda.is_available() else "cpu"


def _is_frontal(v) -> bool:
    s = str(v).upper()
    return ("PA" in s or "AP" in s) and "LATERAL" not in s


def _cohort_subjects(cfg) -> set:
    c = ev.load_cohorts(cfg)
    return set(c["subject_id"].unique())


# --------------------------------------------------------------------------- #
#  images                                                                      #
# --------------------------------------------------------------------------- #
def _frontal_image_table(cfg, subjects):
    m = pd.read_csv(cfg.input("cxr_master"), low_memory=False)
    m["subject_id"] = pd.to_numeric(m["subject_id"], errors="coerce")
    m = m.dropna(subset=["subject_id"])
    m["subject_id"] = m["subject_id"].astype("int64")
    m = m[m["subject_id"].isin(subjects)]

    vcol = next((c for c in ("view", "ViewPosition", "view_position") if c in m.columns), None)
    m = m[m[vcol].map(_is_frontal)] if vcol else m

    d = pd.to_numeric(m["StudyDate"], errors="coerce").astype("Int64").astype("string").str.zfill(8)
    t = (pd.to_numeric(m["StudyTime"], errors="coerce").fillna(0).astype("int64")
         .astype("string").str.zfill(6).str.slice(0, 6))
    m["ts"] = pd.to_datetime(d + t, format="%Y%m%d%H%M%S", errors="coerce")
    m = m.dropna(subset=["ts"])

    root = str(cfg.input("cxr_images"))
    rel = next((c for c in ("jpg_rel_path", "path", "img_rel_path") if c in m.columns), None)
    # cxr_images (Datasets/preprocessed) holds the pXX/... tree directly, without the
    # "mimic-cxr-jpg/files/" prefix that mimic_master_list.csv's path column carries.
    relpath = m[rel].astype(str).str.replace(r"^mimic-cxr-jpg/files/", "", regex=True)
    m["abs_path"] = root + "/" + relpath
    m["shard"] = "p" + m["subject_id"].astype(str).str.slice(0, 2)
    m["source_id"] = m["dicom_id"] if "dicom_id" in m.columns else m[rel]
    return m[["subject_id", "source_id", "ts", "abs_path", "shard"]].reset_index(drop=True)


def _embed_images(cfg, subjects, modality, ckpt, force):
    import torch
    from PIL import Image

    tbl = _frontal_image_table(cfg, subjects)
    out_npy = cfg.storage("embeddings", f"{modality}.npy")
    out_idx = cfg.storage("embeddings", f"_{modality}_index.parquet")
    if out_npy.exists() and ckpt.done(modality) and not force:
        log(f"  {modality}: cached ({out_npy})")
        return pd.read_parquet(out_idx)

    dev = _device(torch)
    hf = cfg.get(f"modalities.{modality}.hf_id")
    size = int(cfg.get(f"modalities.{modality}.size", 224))
    log(f"  {modality}: {len(tbl):,} frontal CXRs for cohort patients; {hf}; device={dev}")

    if modality == "images_alt":
        import open_clip
        model, prep = open_clip.create_model_from_pretrained(hf)
        model = model.eval().to(dev)

        def flush(imgs):
            x = torch.stack([prep(i) for i in imgs]).to(dev)
            with torch.no_grad():
                return model.encode_image(x).float().cpu().numpy().astype(np.float16)
    else:
        from transformers import AutoModel, AutoImageProcessor
        proc = AutoImageProcessor.from_pretrained(hf)
        model = AutoModel.from_pretrained(hf).eval().to(dev)

        def flush(imgs):
            inp = proc(images=imgs, return_tensors="pt").to(dev)
            with torch.no_grad():
                o = model(**inp)
            v = getattr(o, "pooler_output", None)
            if v is None:
                v = o.last_hidden_state[:, 0]
            return v.float().cpu().numpy().astype(np.float16)

    vecs, kept, buf = [], [], []
    t0 = time.time()
    for r in tbl.itertuples(index=False):
        try:
            im = Image.open(r.abs_path).convert("RGB").resize((size, size))
        except Exception as e:
            log(f"    WARN unreadable {r.abs_path}: {e}")
            continue
        buf.append(im)
        kept.append((r.subject_id, r.source_id, r.ts))
        if len(buf) == _IMG_BATCH:
            vecs.append(flush(buf))
            buf = []
    if buf:
        vecs.append(flush(buf))
    if not vecs:
        log(f"  {modality}: no readable images")
        return pd.DataFrame()

    V = np.concatenate(vecs, axis=0)
    np.save(out_npy, V)
    idx = pd.DataFrame(kept, columns=["subject_id", "source_id", "ts"])
    idx["modality"] = modality
    idx["vrow"] = np.arange(len(idx))
    assert len(idx) == len(V), "index/vector length mismatch"
    idx.to_parquet(out_idx, index=False)
    ckpt.mark(modality, n=int(len(V)), dim=int(V.shape[1]))
    log(f"  {modality}: {len(V):,} vectors dim={V.shape[1]} in {time.time()-t0:,.0f}s")
    return idx


# --------------------------------------------------------------------------- #
#  text                                                                        #
# --------------------------------------------------------------------------- #
_NOTE_FILE = {"radtext": "radiology.csv.gz", "histnote": "discharge.csv.gz"}


def _embed_text(cfg, subjects, modality, ckpt, force):
    import torch
    from transformers import AutoModel, AutoTokenizer

    out_npy = cfg.storage("embeddings", f"{modality}.npy")
    out_idx = cfg.storage("embeddings", f"_{modality}_index.parquet")
    if out_npy.exists() and ckpt.done(modality) and not force:
        log(f"  {modality}: cached ({out_npy})")
        return pd.read_parquet(out_idx)

    hf = cfg.get(f"modalities.{modality}.hf_id")
    maxtok = int(cfg.get(f"modalities.{modality}.max_tokens", 4096))
    dev = _device(torch)
    path = cfg.input("notes_dir") / "note" / _NOTE_FILE[modality]

    tok = AutoTokenizer.from_pretrained(hf)
    model = AutoModel.from_pretrained(hf).eval().to(dev)
    log(f"  {modality}: {path.name}; {hf}; max_tokens={maxtok}; device={dev}")

    def flush(texts):
        enc = tok(texts, padding=True, truncation=True, max_length=maxtok,
                  return_tensors="pt").to(dev)
        with torch.no_grad():
            o = model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (o.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return pooled.float().cpu().numpy().astype(np.float16)

    vecs, kept, buf = [], [], []
    t0 = time.time()
    for chunk in pd.read_csv(path, usecols=["note_id", "subject_id", "charttime", "text"],
                             chunksize=20000):
        chunk = chunk[chunk["subject_id"].isin(subjects)]
        chunk = chunk.dropna(subset=["text", "charttime"])
        if not len(chunk):
            continue
        for r in chunk.itertuples(index=False):
            buf.append(str(r.text))
            kept.append((r.subject_id, r.note_id, r.charttime))
            if len(buf) == _TXT_BATCH:
                vecs.append(flush(buf))
                buf = []
    if buf:
        vecs.append(flush(buf))
    if not vecs:
        log(f"  {modality}: no notes for cohort patients")
        return pd.DataFrame()

    V = np.concatenate(vecs, axis=0)
    np.save(out_npy, V)
    idx = pd.DataFrame(kept, columns=["subject_id", "source_id", "ts"])
    idx["ts"] = pd.to_datetime(idx["ts"], errors="coerce")
    idx["modality"] = modality
    idx["vrow"] = np.arange(len(idx))
    assert len(idx) == len(V), "index/vector length mismatch"
    idx.to_parquet(out_idx, index=False)
    ckpt.mark(modality, n=int(len(V)), dim=int(V.shape[1]))
    log(f"  {modality}: {len(V):,} vectors dim={V.shape[1]} in {time.time()-t0:,.0f}s")
    return idx


# --------------------------------------------------------------------------- #
def run(cfg, force: bool = False, intervention: str = None):
    cfg.storage("embeddings").mkdir(parents=True, exist_ok=True)
    cfg.storage("manifests").mkdir(parents=True, exist_ok=True)
    ckpt = Checkpoint(cfg, "extract_embeddings")

    subjects = _cohort_subjects(cfg)
    log(f"extract_embeddings: scoping to {len(subjects):,} cohort patients "
        f"(embedding the whole of MIMIC-CXR would be wasted compute)")

    parts = [
        _embed_images(cfg, subjects, "images", ckpt, force),
        _embed_text(cfg, subjects, "radtext", ckpt, force),
        _embed_text(cfg, subjects, "histnote", ckpt, force),
    ]
    # optional robustness encoder; skipped cleanly if open_clip is absent
    try:
        parts.append(_embed_images(cfg, subjects, "images_alt", ckpt, force))
    except ImportError:
        log("  images_alt: open_clip not installed; encoder swap will be skipped")

    idx = pd.concat([p for p in parts if p is not None and len(p)], ignore_index=True)
    idx.to_parquet(ev.manifest_path(cfg, "embeddings_index.parquet"), index=False)
    log(f"embeddings_index.parquet: {len(idx):,} rows across "
        f"{idx['modality'].nunique()} modalities")
    for m, n in idx.groupby("modality").size().items():
        log(f"    {m:12s} {n:>9,} items")
