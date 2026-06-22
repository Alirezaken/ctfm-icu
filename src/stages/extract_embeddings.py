"""§9.4  extract_embeddings -- the ONLY GPU stage. Run once, store, never recompute.

Plan (§3):
  - Resize frontal CXRs 512 -> 224 ONCE, save the resized files to storage, and
    extract from the saved files (never resize on the fly). Frontal only
    (PA/AP/AP-variants); drop lateral, per encoders.image.view_filter.
  - Images: RAD-DINO, frozen, off the shelf, 224. One vector per frontal CXR.
  - Notes: Clinical-Longformer, frozen, 4096-token window. One vector per note.
    Two patient-level note proxies: notes_clinical (radiology excluded, default)
    and notes_all (radiology included; decomposition only).
  - Store vectors keyed by image/note identifier under paths.embeddings; analysis
    tables reference the keys, not the raw vectors.
  - Pooling happens at cohort build (per patient + time zero, mean-pool pre-time-
    zero items within lookback.hours); this stage stores per-item vectors.
  - Quality gate: sanity-check image embeddings separate known labels on the
    external CXR sets (PadChest available now; CheXpert/ChestX-ray14 pending).

Requires: GPU; torch + transformers (not yet in the venv -- env-pin step).
"""
from src.util import log

# config names friendly encoders; map to the HuggingFace ids used to load them.
ENCODER_HF_IDS = {
    "rad-dino": "microsoft/rad-dino",
    "clinical-longformer": "yikuan8/Clinical-Longformer",
    "biomedclip": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
}


def run(cfg, force: bool = False):
    cfg.require("images.encoder", "text.encoder",
                "pooling.look_back_window_hours")  # window recorded with the vectors
    log("extract_embeddings: not yet implemented (the one GPU stage).")
    raise NotImplementedError(
        "extract_embeddings pending: needs torch+transformers in the venv, "
        "RAD-DINO + Clinical-Longformer, MIMIC-IV-Note (download pending), and a GPU job."
    )
