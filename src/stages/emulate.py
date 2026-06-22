"""§9.5  emulate -- target-trial emulation; build the cohort for an intervention.

Plan (§4):
  - New-user active-comparator design at a single time zero; the two arms are the
    two strategies (never a placebo proxy).
  - Eligibility matching the trial as closely as the data allows.
  - Time-zero discipline: only data STRICTLY BEFORE time zero may be used as
    adjustment variables, including note/image embeddings. Assert in code.
  - Shared all-modality cohort: structured data + >=1 pre-time-zero non-radiology
    note + >=1 pre-time-zero frontal CXR. Every condition for an intervention is
    estimated on the identical patient set. (Larger structured/notes cohorts may
    be reported too, flagged.)
  - CONSORT accounting: eligible, exclusions w/ reasons, arm sizes.
  - Mean-pool pre-time-zero images and notes within lookback.hours into the image
    proxy and the two note proxies.

Run order starts with fluids_sepsis only (§9.5), confirm end-to-end, then expand.

Usage: --stage emulate --intervention fluids_sepsis
"""
from src.util import log


def run(cfg, force: bool = False, intervention: str = "fluids_sepsis"):
    iv = cfg.get(f"interventions.{intervention}")
    if iv is None:
        raise KeyError(f"unknown intervention '{intervention}'")
    cfg.require(f"interventions.{intervention}.eligibility",
                f"interventions.{intervention}.time_zero",
                f"interventions.{intervention}.primary_outcome",
                f"interventions.{intervention}.horizon_days",
                "lookback.hours")
    log(f"emulate[{intervention}]: not yet implemented.")
    raise NotImplementedError(
        "emulate pending implementation: trial definitions are operationalized in "
        "config; this stage still needs the cohort-build code and the stored "
        "embeddings for the all-modality cohort gate."
    )
