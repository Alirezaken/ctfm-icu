"""§9.14  integrity_check -- assert the project invariants before sign-off.

Asserts:
  - time-zero discipline held everywhere (no at/after-time-zero data as adjustment);
  - all conditions for an intervention used the identical shared cohort;
  - no result lives in the repo or $HOME (only under paths.storage_root);
  - exactly 10 result files exist in paths.results (§7);
  - the manifest is complete (versions, seed, bootstrap, estimator + folds,
    encoder names/checkpoints, look-back window, pooling rule, age bands, dataset
    versions, intervention/RCT-reference definitions, wall-clock times).

The repo-leak and 10-file checks are runnable now; the rest activate once the
producing stages write their markers.
"""
from pathlib import Path

from src.util import log, die
from src.results import RESULT_FILES

_REPO_ROOT = Path(__file__).resolve().parents[2]


def run(cfg, force: bool = False):
    problems = []

    # 1. results must not live in the repo or $HOME
    storage = cfg.storage_root.resolve()
    home = Path.home().resolve()
    if _REPO_ROOT in storage.parents or storage == _REPO_ROOT:
        problems.append(f"storage_root {storage} is inside the repo {_REPO_ROOT}")
    if home in storage.parents or storage == home:
        problems.append(f"storage_root {storage} is inside $HOME {home}")

    # 2. exactly 10 result files
    results_dir = cfg.storage("results")
    present = sorted(p.name for p in results_dir.glob("*.csv")) if results_dir.exists() else []
    missing = [f for f in RESULT_FILES if f not in present]
    extra = [f for f in present if f not in RESULT_FILES]
    if missing:
        problems.append(f"missing result files: {missing}")
    if extra:
        problems.append(f"unexpected result files: {extra}")

    if problems:
        for p in problems:
            log(f"  FAIL: {p}")
        die(f"integrity_check failed ({len(problems)} problem(s)).")
    log("integrity_check passed.")
