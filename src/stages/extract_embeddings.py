"""§9.4  extract_embeddings -- the ONLY GPU stage. Run once, store, never recompute.

Per §3:
  - Images: resize frontal CXRs 512->224 once (saved under storage_root/images_224),
    embed with RAD-DINO (frozen), one vector per frontal chest X-ray.
  - Notes: Clinical-Longformer (frozen, 4096-token window), one vector per note.
    Per-note vectors are stored for ALL notes of the imaged subset; the two
    patient proxies (notes_clinical = radiology excluded; notes_all) are formed at
    pooling time by selecting note_type. Notes are scoped to the imaged subset
    (patients with >=1 frontal CXR) -- the only patients that can enter the
    all-modality cohort -- per the compute-discipline rule (§0).
  - Vectors stored as float16 arrays keyed by identifier (dicom_id / note_id);
    analysis tables reference the keys, not the raw vectors.
  - Pooling (mean-pool pre-time-zero items within look_back_window) happens later,
    at cohort build; this stage stores per-item vectors only.
  - Per-shard / per-source checkpointing so the queued job resumes.

Environment: GPU + torch + transformers come from sp_env (see slurm/extract_embeddings.sbatch);
the CPU causal .venv does not carry torch.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.util import log, Checkpoint

# config friendly encoder names -> HuggingFace ids
ENCODER_HF_IDS = {
    "rad-dino": "microsoft/rad-dino",
    "clinical-longformer": "yikuan8/Clinical-Longformer",
    "biomedclip": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
}

_IMG_BATCH = 64
_TXT_BATCH = 8
_NOTE_SOURCES = {"discharge": False, "radiology": True}   # name -> is_radiology
_FRONTAL_EXTRA = {"PA"}                                    # plus anything starting with "AP"


# --------------------------------------------------------------------------- utils
def _device(torch):
    return "cuda" if torch.cuda.is_available() else "cpu"


def _hf_id(cfg, key):
    name = cfg.get(key)
    if name not in ENCODER_HF_IDS:
        raise KeyError(f"unknown encoder '{name}' for {key}; known: {list(ENCODER_HF_IDS)}")
    return ENCODER_HF_IDS[name]


def _is_frontal(view: str) -> bool:
    v = str(view).strip().upper()
    return v in _FRONTAL_EXTRA or v.startswith("AP")


def _study_datetime(df: pd.DataFrame) -> pd.Series:
    d = pd.to_numeric(df["StudyDate"], errors="coerce")
    t = pd.to_numeric(df["StudyTime"], errors="coerce")
    ds = d.astype("Int64").astype("string").str.zfill(8)
    ts = t.fillna(0).astype("int64").astype("string").str.zfill(6).str.slice(0, 6)
    return pd.to_datetime(ds + ts, format="%Y%m%d%H%M%S", errors="coerce")


def _frontal_image_table(cfg) -> pd.DataFrame:
    """Per-frontal-image rows from the CXR master list, with on-disk path + key."""
    master = pd.read_csv(cfg.input("cxr_master"), low_memory=False)
    master = master[master["view"].map(_is_frontal)].copy()
    master["subject_id"] = pd.to_numeric(master["subject_id"], errors="coerce").astype("Int64")
    master = master.dropna(subset=["subject_id"])
    master["study_datetime"] = _study_datetime(master)
    rel = master["jpg_rel_path"].str.split("files/").str[-1]
    master["abs_path"] = str(cfg.input("cxr_images")) + "/" + rel
    master["dicom_id"] = (master["jpg_rel_path"].str.rsplit("/", n=1).str[-1]
                          .str.replace(".jpg", "", regex=False))
    master["shard"] = "p" + master["subject_id"].astype("int64").astype("string").str.slice(0, 2)
    return master[["dicom_id", "subject_id", "study_id", "view", "study_datetime",
                   "shard", "abs_path"]].reset_index(drop=True)


# --------------------------------------------------------------------------- images
def _embed_images(cfg, ckpt, force):
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoImageProcessor

    tbl = _frontal_image_table(cfg)
    out_dir = cfg.storage("embeddings", "images")
    out_dir.mkdir(parents=True, exist_ok=True)
    resized_root = cfg.storage_root / "images_224"
    size = int(cfg.get("images.size", 224))
    dev = _device(torch)
    hf = _hf_id(cfg, "images.encoder")

    log(f"images: {len(tbl):,} frontal CXRs / {tbl['shard'].nunique()} shards; {hf}; device={dev}")
    proc = AutoImageProcessor.from_pretrained(hf)
    model = AutoModel.from_pretrained(hf).eval().to(dev)

    def flush(imgs):
        inp = proc(images=imgs, return_tensors="pt").to(dev)
        with torch.no_grad():
            o = model(**inp)
        v = getattr(o, "pooler_output", None)
        if v is None:
            v = o.last_hidden_state[:, 0]              # CLS token
        return v.float().cpu().numpy().astype(np.float16)

    IDX = ["dicom_id", "subject_id", "study_id", "view", "study_datetime", "shard", "row"]
    for shard, sub in tbl.groupby("shard"):
        unit = f"images_{shard}"
        vec_path = out_dir / f"vectors_{shard}.npy"
        idx_path = out_dir / f"index_{shard}.csv"
        sub = sub.reset_index(drop=True)
        if vec_path.exists() and ckpt.done(unit) and not force:
            n = int(np.load(vec_path, mmap_mode="r").shape[0])
            if not idx_path.exists() and len(sub) == n:   # self-repair index (0-skip run)
                keep = sub.copy(); keep["row"] = np.arange(n)
                keep[IDX].to_csv(idx_path, index=False)
                log(f"  {shard}: cached, index reconstructed ({n:,})")
            else:
                log(f"  {shard}: cached ({n:,})")
            continue
        out_vecs, kept_rows, buf = [], [], []
        t0 = time.time()
        for _, r in sub.iterrows():
            try:
                im = Image.open(r["abs_path"]).convert("RGB").resize((size, size))
            except Exception as e:
                log(f"    WARN unreadable {r['abs_path']}: {e}"); continue
            rp = resized_root / r["abs_path"].split("preprocessed/")[-1]
            rp.parent.mkdir(parents=True, exist_ok=True)
            if not rp.exists():
                im.save(rp, format="JPEG", quality=95)
            buf.append(im); kept_rows.append(r)
            if len(buf) == _IMG_BATCH:
                out_vecs.append(flush(buf)); buf = []
        if buf:
            out_vecs.append(flush(buf))
        if not out_vecs:
            log(f"  {shard}: no readable images"); continue
        vecs = np.concatenate(out_vecs, axis=0)
        np.save(vec_path, vecs)
        keep = pd.DataFrame(kept_rows).reset_index(drop=True)
        keep["row"] = np.arange(len(keep))
        keep[IDX].to_csv(idx_path, index=False)
        ckpt.mark(unit, n=int(len(vecs)), dim=int(vecs.shape[1]))
        log(f"  {shard}: {len(vecs):,} vectors dim={vecs.shape[1]} in {time.time()-t0:,.0f}s")

    parts = sorted(out_dir.glob("index_p*.csv"))
    if parts:
        pd.concat([pd.read_csv(p) for p in parts], ignore_index=True).to_csv(
            out_dir / "index.csv", index=False)
        log(f"images index -> {out_dir/'index.csv'} ({len(parts)} shards)")


# --------------------------------------------------------------------------- notes
def _embed_notes(cfg, ckpt, force):
    import torch
    from transformers import AutoModel, AutoTokenizer

    imaged = set(_frontal_image_table(cfg)["subject_id"].astype("int64").tolist())
    log(f"notes: imaged subset = {len(imaged):,} patients")

    out_dir = cfg.storage("embeddings", "notes")
    out_dir.mkdir(parents=True, exist_ok=True)
    notes_root = cfg.input("notes_dir") / "note"
    max_tok = int(cfg.get("text.max_tokens", 4096))
    dev = _device(torch)
    hf = _hf_id(cfg, "text.encoder")

    tok = AutoTokenizer.from_pretrained(hf)
    model = AutoModel.from_pretrained(hf).eval().to(dev)

    def embed(texts):
        enc = tok(texts, truncation=True, max_length=max_tok, padding=True,
                  return_tensors="pt").to(dev)
        with torch.no_grad():
            o = model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (o.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
        return pooled.float().cpu().numpy().astype(np.float16)

    for source, is_rad in _NOTE_SOURCES.items():
        unit = f"notes_{source}"
        vec_path = out_dir / f"vectors_{source}.npy"
        idx_path = out_dir / f"index_{source}.csv"
        if ckpt.done(unit) and vec_path.exists() and not force:
            log(f"  {source}: cached"); continue
        path = notes_root / f"{source}.csv.gz"
        if not path.exists():
            log(f"  {source}: missing {path}, skip"); continue

        t0 = time.time()
        out_vecs, keys = [], []
        for chunk in pd.read_csv(path, usecols=["note_id", "subject_id", "charttime", "text"],
                                 chunksize=20_000):
            chunk = chunk[chunk["subject_id"].isin(imaged)]
            if chunk.empty:
                continue
            texts = chunk["text"].fillna("").astype(str).tolist()
            for i in range(0, len(texts), _TXT_BATCH):
                out_vecs.append(embed(texts[i:i + _TXT_BATCH]))
            keys.append(chunk[["note_id", "subject_id", "charttime"]])
        if not out_vecs:
            log(f"  {source}: no notes for imaged subset"); continue
        vecs = np.concatenate(out_vecs, axis=0)
        np.save(vec_path, vecs)
        k = pd.concat(keys, ignore_index=True)
        k["note_type"] = source
        k["is_radiology"] = is_rad
        k["row"] = np.arange(len(k))
        k.to_csv(idx_path, index=False)
        ckpt.mark(unit, n=int(len(vecs)), dim=int(vecs.shape[1]))
        log(f"  {source}: {len(vecs):,} note vectors dim={vecs.shape[1]} in {time.time()-t0:,.0f}s")

    parts = [p for p in sorted(out_dir.glob("index_*.csv")) if p.name != "index.csv"]
    if parts:
        pd.concat([pd.read_csv(p) for p in parts], ignore_index=True).to_csv(
            out_dir / "index.csv", index=False)
        log(f"notes index -> {out_dir/'index.csv'}")


# --------------------------------------------------------------------------- §3 quality gate
def _ka(values_pct):
    b = np.asarray(values_pct, float); b = b[np.isfinite(b)]
    r = lambda x: round(float(x), 1)
    return r(np.mean(b)), r(np.std(b, ddof=1)), r(np.percentile(b, 2.5)), r(np.percentile(b, 97.5))


def _embed_paths(proc, model, paths, dev, size, batch=64):
    """Embed a list of image paths with the loaded encoder; returns (V, ok_mask)."""
    import torch
    from PIL import Image
    vecs, ok, buf = [], [], []

    def flush(imgs):
        inp = proc(images=imgs, return_tensors="pt").to(dev)
        with torch.no_grad():
            o = model(**inp)
        v = getattr(o, "pooler_output", None)
        if v is None:
            v = o.last_hidden_state[:, 0]
        return v.float().cpu().numpy().astype(np.float16)

    for p in paths:
        try:
            im = Image.open(p).convert("RGB").resize((size, size))
        except Exception:
            ok.append(False); continue
        ok.append(True); buf.append(im)
        if len(buf) == batch:
            vecs.append(flush(buf)); buf = []
    if buf:
        vecs.append(flush(buf))
    V = np.concatenate(vecs, axis=0).astype("float32") if vecs else np.zeros((0, 1))
    return V, np.array(ok)


def _quality_gate(cfg, ckpt, force):
    """Embed external CXR sets with RAD-DINO and check the embeddings separate
    known findings (linear probe, AUROC/AUPRC). A check, not a result -> probe.csv
    with modality 'image@<dataset>'. These embeddings never enter the causal models."""
    import torch
    from transformers import AutoModel, AutoImageProcessor
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, average_precision_score
    from src.stats import cluster_bootstrap_indices

    qg = cfg.get("quality_gate")
    if not qg:
        log("quality gate: no config; skip"); return
    labels = qg.get("probe_labels", [])
    sample = int(qg.get("sample_per_dataset", 30000))
    seed = int(cfg.get("run.seed", 42))
    size = int(cfg.get("images.size", 224))
    dev = _device(torch)
    hf = _hf_id(cfg, "images.encoder")
    proc = AutoImageProcessor.from_pretrained(hf)
    model = AutoModel.from_pretrained(hf).eval().to(dev)

    rows = []
    for ds, spec in qg["datasets"].items():
        if ckpt.done(f"gate_{ds}") and not force:
            log(f"  gate {ds}: cached"); continue
        master = spec.get("master")
        if not master or not os.path.exists(master):
            log(f"  gate {ds}: master missing ({master}); skip"); continue
        df = pd.read_csv(master, low_memory=False)
        vc = spec.get("view_col")
        if vc and vc in df:
            df = df[df[vc].map(_is_frontal)]
        df = df.reset_index(drop=True)
        if "path_cols" in spec:
            a, b = spec["path_cols"]
            df["_p"] = spec["images"] + "/" + df[a].astype(str) + "/" + df[b].astype(str)
        else:
            df["_p"] = spec["images"] + "/" + df[spec["path_col"]].astype(str)
        if len(df) > sample:
            df = df.sample(sample, random_state=seed).reset_index(drop=True)

        t0 = time.time()
        V, ok = _embed_paths(proc, model, df["_p"].tolist(), dev, size)
        df = df[ok].reset_index(drop=True)
        log(f"  gate {ds}: embedded {len(df):,} frontal images in {time.time()-t0:,.0f}s")

        pidcol = next((c for c in ("PatientID", "patient_id") if c in df), None)
        rng = np.random.default_rng(seed)
        if pidcol:
            pats = df[pidcol].unique()
            test_pats = set(rng.choice(pats, size=max(1, int(len(pats) * 0.3)), replace=False))
            te = df[pidcol].isin(test_pats).to_numpy()
        else:
            te = rng.random(len(df)) < 0.3
        tr = ~te
        sc = StandardScaler().fit(V[tr])
        Vt, Ve = sc.transform(V[tr]), sc.transform(V[te])
        subj = (df.loc[te, pidcol].to_numpy() if pidcol else np.arange(int(te.sum())))

        for lab in labels:
            col = spec.get("label_map", {}).get(lab)
            if not col or col not in df:
                continue
            y = (pd.to_numeric(df[col], errors="coerce") == 1).astype(int).to_numpy()
            if y[tr].sum() < 20 or y[te].sum() < 10:
                log(f"    {ds}/{lab}: too few positives (tr={y[tr].sum()}, te={y[te].sum()}); skip")
                continue
            clf = LogisticRegression(max_iter=1000).fit(Vt, y[tr])
            p = clf.predict_proba(Ve)[:, 1]; yte = y[te]
            auroc = roc_auc_score(yte, p) * 100; auprc = average_precision_score(yte, p) * 100
            br, bp = [], []
            for bidx in cluster_bootstrap_indices(subj, 1000, seed):
                yb, pb = yte[bidx], p[bidx]
                if yb.min() == yb.max():
                    continue
                br.append(roc_auc_score(yb, pb) * 100); bp.append(average_precision_score(yb, pb) * 100)
            rm, rs, rl, rh = _ka(br); pm, ps, pl, ph = _ka(bp)
            rows.append({"modality": f"image@{ds}", "target_confounder": lab,
                         "auroc_mean": rm, "auroc_std": rs, "auroc_ci_low": rl, "auroc_ci_high": rh,
                         "auprc_mean": pm, "auprc_std": ps, "auprc_ci_low": pl, "auprc_ci_high": ph})
            log(f"    {ds}/{lab}: AUROC={auroc:.1f} AUPRC={auprc:.1f}")
        ckpt.mark(f"gate_{ds}", n=int(len(df)))

    if rows:
        R.append_rows(cfg, "probe.csv", rows)
        log(f"quality gate -> probe.csv ({len(rows)} external rows)")


# --------------------------------------------------------------------------- entry
def run(cfg, force: bool = False):
    cfg.require("images.encoder", "text.encoder", "pooling.look_back_window_hours")
    ckpt = Checkpoint(cfg, "extract_embeddings")
    t0 = time.time()
    _embed_images(cfg, ckpt, force)
    _embed_notes(cfg, ckpt, force)
    _quality_gate(cfg, ckpt, force)   # §3: external CXR embedding-quality check
    log(f"extract_embeddings done in {time.time()-t0:,.0f}s")
