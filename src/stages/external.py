"""§9.12 / §6.8  external -- rerun on eICU-CRD (no imaging there).

Rerun only the `structured` and `plus_notes` conditions on eICU and append rows
to effects.csv with a dataset key (dataset=eicu). Same AIPW estimator, same
reporting. This is the only stage blocked on the eICU upload; everything upstream
runs on internal MIMIC without it.
"""
from src.util import log
from src.util import die


def run(cfg, force: bool = False):
    eicu = cfg.input("eicu_dir")
    if not eicu.exists():
        die(f"eICU not present at {eicu}. Upload eICU-CRD v2.0 first; the rest of "
            f"the pipeline does not depend on it.")
    log("external: not yet implemented.")
    raise NotImplementedError("external pending: needs the estimate stage and the eICU upload.")
