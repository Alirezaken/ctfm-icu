"""extract_external -- RAD-DINO embeddings for the EXTERNAL chest X-ray datasets.

PadChest / ChestX-ray14 / CheXpert, embedded with the SAME frozen encoder used on MIMIC.
These never enter any causal model. Their only job is to answer the question a reviewer
will certainly ask about the headline claim:

    "You say the embedding is informative but useless. Maybe it's just not informative,
     and your MIMIC probe is overfitting to MIMIC."

If RAD-DINO separates pulmonary edema on three independent CXR corpora it has never seen,
that objection is closed, and the paper's tension -- INFORMATIVE yet NOT INCREMENTALLY
CONFOUNDING -- stands on solid ground. Without this, the whole argument is one probe on
one dataset.

Writes embeddings/external/<name>.npy and <name>_index.csv. diagnose.py reads them.
"""
from __future__ import annotations

import re
import time

import numpy as np
import pandas as pd

from src.util import log, Checkpoint
from src.stages.extract_embeddings import _is_frontal, _device, _IMG_BATCH


def _abs_paths(df, template, images_root):
    toks = [t for t in re.findall(r"{(\w+)}", template) if t != "images_root"]
    sub = df.copy()
    for t in toks:
        sub[t] = sub[t].astype(str)
    return sub.apply(
        lambda r: template.format(images_root=images_root, **{t: r[t] for t in toks}),
        axis=1)


def _embed(cfg, name, spec, ckpt, force):
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoImageProcessor

    out = cfg.storage("embeddings", "external")
    out.mkdir(parents=True, exist_ok=True)
    vpath, ipath = out / f"{name}.npy", out / f"{name}_index.csv"
    if vpath.exists() and ckpt.done(name) and not force:
        log(f"  {name}: cached")
        return

    df = pd.read_csv(spec["csv"], low_memory=False)
    df = df[df[spec["view_col"]].map(_is_frontal)].reset_index(drop=True)
    cap = int(cfg.get("quality_gate.sample_per_dataset", 30000))
    if len(df) > cap:
        df = df.sample(cap, random_state=cfg.seed).reset_index(drop=True)
    df["abs"] = _abs_paths(df, spec["path"], spec["images_root"])

    hf = cfg.get("modalities.images.hf_id")
    size = int(cfg.get("modalities.images.size", 224))
    dev = _device(torch)
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
    for r in df.itertuples(index=False):
        try:
            im = Image.open(getattr(r, "abs")).convert("RGB").resize((size, size))
        except Exception:
            continue
        buf.append(im)
        kept.append(getattr(r, spec["id_col"]))
        if len(buf) == _IMG_BATCH:
            vecs.append(flush(buf))
            buf = []
    if buf:
        vecs.append(flush(buf))
    if not vecs:
        log(f"  {name}: no readable images")
        return

    V = np.concatenate(vecs, axis=0)
    np.save(vpath, V)
    pd.DataFrame({"image_id": kept, "vrow": np.arange(len(kept))}).to_csv(ipath, index=False)
    ckpt.mark(name, n=int(len(V)), dim=int(V.shape[1]))
    log(f"  {name}: {len(V):,} vectors dim={V.shape[1]} in {time.time()-t0:,.0f}s")


def run(cfg, force: bool = False, intervention: str = None):
    ext = cfg.get("paths.external_cxr") or {}
    present = {k: v for k, v in ext.items() if isinstance(v, dict)}
    if not present:
        log("no external CXR datasets configured; nothing to do")
        return
    ckpt = Checkpoint(cfg, "extract_external")
    log(f"external CXR embedding (quality gate): {list(present)}")
    for name, spec in present.items():
        _embed(cfg, name, spec, ckpt, force)
