"""ONE definition of every statistic in the project (§8).

Three result kinds, kept strictly separate, computed by this one shared set of
helpers so there is a single definition of mean, std, CI, p, and bootstrap
everywhere. Never mix the kinds in a column or a file.

  Kind A  performance metrics  (AUROC/AUPRC): bootstrap mean, std, 95% CI, percent, 1 dp.
  Kind B  causal effects       (risk diffs, bias, bias reductions, subgroup effects):
                                natural scale = percentage points, 1 dp; patient-level
                                (cluster) bootstrap mean/std/95% CI + the point estimate.
  Kind C  statistical measures (p-values, correlations, kappa, E-values): natural scale,
                                never percent, never with mean/std/CI.

The fully-implemented helpers here are estimator-agnostic and need no data files,
so they are written now; estimator-specific pieces (AIPW influence-function CIs,
E-values) are stubbed until the estimate stage lands.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


# --------------------------------------------------------------------------- #
#  Shared bootstrap machinery                                                  #
# --------------------------------------------------------------------------- #
def cluster_bootstrap_indices(clusters: np.ndarray, n_resamples: int, seed: int):
    """Patient-level (cluster) bootstrap index sets, generated ONCE and reused
    across conditions so paired contrasts are valid (§8 Kind B).

    Yields, for each of n_resamples draws, the row indices belonging to a sample
    of clusters drawn with replacement. `clusters` is the per-row cluster id
    (e.g. subject_id) aligned to the analysis table.
    """
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(clusters, return_inverse=True)
    rows_by_cluster = [np.where(inv == i)[0] for i in range(len(uniq))]
    n = len(uniq)
    for _ in range(n_resamples):
        drawn = rng.integers(0, n, size=n)
        yield np.concatenate([rows_by_cluster[c] for c in drawn])


@dataclass
class Estimate:
    """A Kind-B causal quantity: point estimate + bootstrap summary, in pp."""
    point: float          # AIPW point estimate, reported alongside (§8)
    mean: float           # bootstrap mean
    std: float            # bootstrap std
    ci_low: float         # 2.5th percentile
    ci_high: float        # 97.5th percentile

    def as_row(self, prefix: str = "") -> dict:
        return {f"{prefix}{k}": _round1(v) for k, v in asdict(self).items()}


def bootstrap_summary(point: float, boot_values, ci: float = 95.0) -> Estimate:
    """Summarise a vector of bootstrap replicates into a Kind-B Estimate (1 dp)."""
    b = np.asarray(boot_values, dtype=float)
    b = b[np.isfinite(b)]
    lo, hi = (100 - ci) / 2, 100 - (100 - ci) / 2
    return Estimate(
        point=float(point),
        mean=float(np.mean(b)),
        std=float(np.std(b, ddof=1)),
        ci_low=float(np.percentile(b, lo)),
        ci_high=float(np.percentile(b, hi)),
    )


# --------------------------------------------------------------------------- #
#  Kind C: multiple-testing                                                    #
# --------------------------------------------------------------------------- #
def benjamini_hochberg(pvals) -> np.ndarray:
    """BH-FDR-adjusted p-values across a family, full precision (§8 Kind C).

    Only the key contrasts get p-values; this adjusts that family. Returned at
    full precision (no rounding) -- p-values never carry a mean, std or CI.
    """
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    # enforce monotonicity from the largest rank downward
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


# --------------------------------------------------------------------------- #
#  Formatting                                                                  #
# --------------------------------------------------------------------------- #
def _round1(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 1)


# --------------------------------------------------------------------------- #
#  AIPW influence-function CI (§8 Kind B check) + E-values (§6.4 Kind C)       #
# --------------------------------------------------------------------------- #
def influence_function_ci(psi, keep=None, ci: float = 95.0):
    """Influence-function 95% CI, reported as a check alongside the bootstrap CI.
    psi = per-patient AIPW influence values; effect = mean(psi)*100 pp,
    SE = std(psi)/sqrt(n). Returns (point, lo, hi) in percentage points."""
    p = np.asarray(psi, float)
    if keep is not None:
        p = p[np.asarray(keep, bool)]
    p = p[np.isfinite(p)]
    n = p.size
    point = float(p.mean() * 100)
    se = float(p.std(ddof=1) / np.sqrt(n) * 100)
    z = 1.959963984540054 if ci == 95.0 else float(
        __import__("scipy.stats", fromlist=["norm"]).norm.ppf(1 - (1 - ci / 100) / 2))
    return _round1(point), _round1(point - z * se), _round1(point + z * se)


def _rd_to_rr(rd_pp: float, baseline_pct: float):
    """Approximate risk ratio from a risk difference (pp) given the comparator
    baseline risk (percent), oriented >=1 (VanderWeele-Ding convention)."""
    p0 = baseline_pct / 100.0
    p1 = p0 + rd_pp / 100.0
    if not (0 < p0 < 1) or not (0 < p1 < 1):
        return None                      # effect implies an impossible risk -> E-value undefined
    rr = p1 / p0
    return rr if rr >= 1 else 1.0 / rr


def e_value(rd_pp: float, baseline_pct: float):
    """E-value for a risk-difference effect (Kind C, §6.4): minimum strength of
    unmeasured confounding (risk-ratio scale) that could explain the effect away."""
    rr = _rd_to_rr(rd_pp, baseline_pct)
    if rr is None:
        return None
    return round(rr + np.sqrt(rr * (rr - 1)), 2)


def e_value_ci_limit(ci_low_pp: float, ci_high_pp: float, baseline_pct: float):
    """E-value for the CI limit nearest the null (RD=0); 1.0 if the CI crosses 0."""
    if ci_low_pp <= 0 <= ci_high_pp:
        return 1.0
    bound = ci_low_pp if abs(ci_low_pp) < abs(ci_high_pp) else ci_high_pp
    return e_value(bound, baseline_pct)
