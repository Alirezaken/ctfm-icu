"""extract_images_alt -- second frozen CXR image encoder for the robustness swap.

Embeds the same frontal MIMIC-CXRs with BiomedCLIP (robustness_swaps.encoder_alt)
into embeddings/images_alt/, mirroring the RAD-DINO layout so robustness's encoder
swap can pool from it. GPU stage; BiomedCLIP loads via open_clip (not transformers),
so it needs open_clip in the env (sp_env). Per-shard checkpoint/resume; CSV indices.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd

from src.util import log, Checkpoint
from src.stages.extract_embeddings import _frontal_image_table, _device

_BIOMEDCLIP = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
_IMG_BATCH = 64
_IDX = ["dicom_id", "subject_id", "study_id", "view", "study_datetime", "shard", "row"]


def run(cfg, force: bool = False):
    import torch
    import open_clip
    from PIL import Image

    tbl = _frontal_image_table(cfg)
    out = cfg.storage("embeddings", "images_alt")
    out.mkdir(parents=True, exist_ok=True)
    dev = _device(torch)
    log(f"images_alt: {len(tbl):,} frontal CXRs; BiomedCLIP; device={dev}")
    model, prep = open_clip.create_model_from_pretrained(_BIOMEDCLIP)
    model = model.eval().to(dev)
    ckpt = Checkpoint(cfg, "extract_images_alt")

    def flush(imgs):
        x = torch.stack([prep(i) for i in imgs]).to(dev)
        with torch.no_grad():
            f = model.encode_image(x)
        return f.float().cpu().numpy().astype(np.float16)

    for shard, sub in tbl.groupby("shard"):
        unit = f"alt_{shard}"
        vec_path = out / f"vectors_{shard}.npy"
        idx_path = out / f"index_{shard}.csv"
        sub = sub.reset_index(drop=True)
        if vec_path.exists() and ckpt.done(unit) and not force:
            log(f"  {shard}: cached"); continue
        vecs, kept, buf = [], [], []
        t0 = time.time()
        for _, r in sub.iterrows():
            try:
                buf.append(Image.open(r["abs_path"]).convert("RGB")); kept.append(r)
            except Exception as e:
                log(f"    WARN unreadable {r['abs_path']}: {e}")
            if len(buf) == _IMG_BATCH:
                vecs.append(flush(buf)); buf = []
        if buf:
            vecs.append(flush(buf))
        if not vecs:
            log(f"  {shard}: no readable images"); continue
        V = np.concatenate(vecs, axis=0)
        np.save(vec_path, V)
        keep = pd.DataFrame(kept).reset_index(drop=True)
        keep["row"] = np.arange(len(keep))
        keep[_IDX].to_csv(idx_path, index=False)
        ckpt.mark(unit, n=int(len(V)), dim=int(V.shape[1]))
        log(f"  {shard}: {len(V):,} vectors dim={V.shape[1]} in {time.time()-t0:,.0f}s")

    parts = sorted(out.glob("index_p*.csv"))
    if parts:
        pd.concat([pd.read_csv(p) for p in parts], ignore_index=True).to_csv(out / "index.csv", index=False)
        log(f"images_alt index -> {out/'index.csv'} ({len(parts)} shards)")
