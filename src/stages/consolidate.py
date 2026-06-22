"""§9.13 / §7  consolidate -- the 10 result files on the storage path.

Exactly 10 files (paths.results), long format, reproducible from these alone:
  effects, dissociation, decomposition, controls, probe, cohorts, demographics,
  robustness, comparisons, manifest. Schemas + the single writer live in
  src/results.py; §8 column kinds (A perf / B causal / C statistical) are never
  mixed in a column or file.

What runs now:
  - write the exact-schema templates for all 10 files (header-only if not yet
    populated by upstream stages);
  - build manifest.csv from values known today (config + package versions);
    fabricated numbers are never written -- unknown cells stay empty (§0).

Still TODO (needs the per-stage estimates to exist): merge the per-intervention
estimate/probe/cohort outputs into effects/controls/probe/cohorts, and compute the
cross-stage tables -- the specificity grid (dissociation), the modality
decomposition over both note variants (decomposition), and the key-contrast
p-values with BH-FDR (comparisons).
"""
from src import results
from src.util import log


def run(cfg, force: bool = False):
    written = results.write_templates(cfg, force=force)
    n = results.build_manifest(cfg)
    log(f"manifest.csv written ({n} keys).")
    log(f"result files ready under {cfg.storage('results')}: "
        f"{len(results.RESULT_FILES)} files "
        f"({len(written)} freshly templated).")
    log("note: effect/probe/cohort cells fill in once the estimate stages run.")
