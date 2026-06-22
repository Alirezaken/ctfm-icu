"""§9.11 / §6.9  robustness -- recompute the structured->full bias reduction per
intervention under four swaps:
  - encoder:   a second frozen CXR image encoder (encoders.image_robustness).
  - estimator: TMLE in place of AIPW.
  - window:    a different look-back window (lookback robustness_swaps.lookback_hours_alt).
  - pooling:   max instead of mean (robustness_swaps.pooling_alt).
-> robustness.csv (value = structured-to-full bias reduction, Kind B).
"""
from src.util import log


def run(cfg, force: bool = False):
    cfg.require("robustness_swaps.encoder_alt",
                "robustness_swaps.look_back_window_alt_hours")
    log("robustness: not yet implemented.")
    raise NotImplementedError(
        "robustness pending: needs the estimate stage plus a 2nd CXR encoder and "
        "the alternate look-back window."
    )
