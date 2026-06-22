"""§9.6 / §6.5  probe -- validity probe (a GATE, run early on the first intervention).

Held-out prediction of each proxy's target confounder:
  - image proxy  -> labeled pulmonary edema (MIMIC-CXR-JPG Edema label).
  - note proxy   -> its target confounder (probe.notes_target_confounder).
Report AUROC and AUPRC (Kind A: bootstrap mean/std/95% CI, percent, 1 dp) ->
probe.csv.

If a modality cannot predict its confounder, STOP and tell Soroosh -- do not
proceed to estimate (§9.6). The image side is unblocked now (MIMIC-CXR Edema
labels are in mimic_master_list.csv); the note side waits on note embeddings.
"""
from src.util import log


def run(cfg, force: bool = False, intervention: str = "fluids_sepsis"):
    cfg.require("probe.image_target_confounder")  # note target pending Soroosh
    log("probe: not yet implemented (validity gate).")
    raise NotImplementedError(
        "probe pending: needs stored image/note embeddings. Image side can run "
        "first (MIMIC-CXR Edema label); note side needs the note target confounder."
    )
