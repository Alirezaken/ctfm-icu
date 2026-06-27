"""§3 quality gate -- RAD-DINO embeddings for the external CXR datasets.

Embeds PadChest / ChestX-ray14 / CheXpert with the SAME frozen image encoder
(RAD-DINO, 224) used for MIMIC-CXR, so we can later check the embeddings separate
known labels (a quality gate, not a result; §3). These sets have no linked ICU
trajectories and never enter the causal models.

Per dataset (paths.external_cxr.<name>, a dict or null):
  csv, images_root, id_col, view_col, path (template over {images_root} + columns).
Frontal only (PA / AP*). Vectors stored float16, keyed by id_col, under
storage_root/embeddings/external/<name>/. Per-dataset checkpoint, so a dataset
added later (e.g. CheXpert once uploaded) is embedded on re-run while the cached
ones are skipped.

Runs on the GPU env (sp_env); see slurm/extract_external_cxr.sbatch.
"""
from __future__ import annotations

import re
import time

import numpy as np
import pandas as pd

from src.util import log, Checkpoint
from src.stages.extract_embeddings import _is_frontal, _hf_id, _device, _IMG_BATCH


def _abs_paths(df: pd.DataFrame, template: str, images_root: str) -> pd.Series:
    """Build per-row absolute image paths from a `{images_root}/{col}/...` template."""
    toks = [t for t in re.findall(r"{(\w+)}", template) if t != "images_root"]
    sub = df.copy()
    for t in toks:
        sub[t] = sub[t].astype(str)
    return sub.apply(
        lambda r: template.format(images_root=images_root, **{t: r[t] for t in toks}),
        axis=1,
    )


def _embed_dataset(cfg, name: str, spec: dict, ckpt, force: bool):
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoImageProcessor

    out_dir = cfg.storage("embeddings", "external", name)
    out_dir.mkdir(parents=True, exist_ok=True)
    vec_path, idx_path = out_dir / "vectors.npy", out_dir / "index.csv"
    if ckpt.done(name) and vec_path.exists() and not force:
        log(f"  {name}: cached"); return

    df = pd.read_csv(spec["csv"], low_memory=False)
    df = df[df[spec["view_col"]].map(_is_frontal)].reset_index(drop=True)
    df["abs"] = _abs_paths(df, spec["path"], spec["images_root"])

    size = int(cfg.get("images.size", 224))
    dev = _device(torch)
    hf = _hf_id(cfg, "images.encoder")
    log(f"  {name}: {len(df):,} frontal images; {hf}; device={dev}")
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
    for _, r in df.iterrows():
        try:
            im = Image.open(r["abs"]).convert("RGB").resize((size, size))
        except Exception as e:
            log(f"    WARN unreadable {r['abs']}: {e}"); continue
        buf.append(im); kept.append(r[spec["id_col"]])
        if len(buf) == _IMG_BATCH:
            vecs.append(flush(buf)); buf = []
    if buf:
        vecs.append(flush(buf))
    if not vecs:
        log(f"  {name}: no readable images"); return

    arr = np.concatenate(vecs, axis=0)
    np.save(vec_path, arr)
    pd.DataFrame({"image_id": kept, "row": np.arange(len(kept))}).to_csv(idx_path, index=False)
    ckpt.mark(name, n=int(len(arr)), dim=int(arr.shape[1]))
    log(f"  {name}: {len(arr):,} vectors dim={arr.shape[1]} in {time.time()-t0:,.0f}s")


def run(cfg, force: bool = False):
    cfg.require("images.encoder")
    ext = cfg.get("paths.external_cxr") or {}
    present = {k: v for k, v in ext.items() if isinstance(v, dict)}
    missing = [k for k, v in ext.items() if not isinstance(v, dict)]
    if missing:
        log(f"external CXR not yet on disk (skipped, add path + re-run): {missing}")
    if not present:
        log("no external CXR datasets configured; nothing to do."); return

    ckpt = Checkpoint(cfg, "extract_external_cxr")
    t0 = time.time()
    log(f"external CXR embedding: {list(present)}")
    for name, spec in present.items():
        _embed_dataset(cfg, name, spec, ckpt, force)
    log(f"external CXR embedding done in {time.time()-t0:,.0f}s")
