"""§9.10 / §6.7  demographics -- subgroup re-estimation.

Re-estimate the effect within each sex and each age band, for every intervention,
under `structured` and `full`, with the same AIPW. Report the subgroup effect and
CI (Kind B), the subgroup-specific bias reduction, and a support count. Mark a
subgroup undefined (with its support count) when it has too few outcome events.
-> demographics.csv.
"""
from src.util import log


def run(cfg, force: bool = False):
    cfg.require("age.bands")  # banding must be fixed (and confirmed by Soroosh)
    log("demographics: not yet implemented.")
    raise NotImplementedError("demographics pending: needs the estimate stage and confirmed age bands.")
