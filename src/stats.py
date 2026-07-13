"""ONE definition of every statistic in the project.

Three result kinds, kept strictly separate. Never mix them in a column or a file.

  Kind A  performance metrics (AUROC/AUPRC/dAUC): bootstrap mean, std, 95% CI. Percent, 1dp.
  Kind B  causal quantities (risk differences, bias, bias reduction, subgroup effects):
          natural scale = percentage points, 1dp; patient-level cluster bootstrap
          mean/std/95% CI, reported alongside the point estimate.
  Kind C  statistical measures (p-values, divergence Z, E-values, MDBR): natural scale,
          never percent, never carrying a mean/std/CI.

Key metric changes vs. the original design:
  * `inside_reference_ci` is GONE. STARRT-AKI's RCT interval is 0.7pp wide; no
    observational estimate lands inside it, so the metric was guaranteed to fail and
    carried no information.
  * `divergence_z` replaces it: (RD_obs - RD_rct) / sqrt(SE_obs^2 + SE_rct^2). This is
    the standard emulation-vs-trial agreement statistic and it accounts for BOTH
    uncertainties, which an inside/outside check does not.
  * `minimum_detectable_bias_reduction` makes a null DECISIVE rather than merely
    underpowered: it states the smallest bias reduction the design could have found.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


# --------------------------------------------------------------------------- #
#  Shared bootstrap machinery                                                  #
# --------------------------------------------------------------------------- #
def cluster_bootstrap_indices(clusters: np.ndarray, n_resamples: int, seed: int) -> list:
    """Patient-level (cluster) bootstrap index sets, generated ONCE per cohort and
    reused across every condition so that contrasts between conditions are PAIRED.

    Pairing is what makes the bias-reduction CIs tight enough for the null to be
    decisive: the marginal CI on each condition's effect is wide, but the CI on the
    DIFFERENCE between two conditions on the same patients is much narrower.
    """
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(clusters, return_inverse=True)
    rows_by_cluster = [np.where(inv == i)[0] for i in range(len(uniq))]
    n = len(uniq)
    out = []
    for _ in range(n_resamples):
        drawn = rng.integers(0, n, size=n)
        out.append(np.concatenate([rows_by_cluster[c] for c in drawn]))
    return out


@dataclass
class Estimate:
    """A Kind-B causal quantity: point estimate + bootstrap summary, in percentage points."""
    point: float
    mean: float
    std: float
    ci_low: float
    ci_high: float

    def as_row(self, prefix: str = "") -> dict:
        return {f"{prefix}{k}": _round1(v) for k, v in asdict(self).items()}


def bootstrap_summary(point: float, boot_values, ci: float = 95.0) -> Estimate:
    """Summarize bootstrap replicates into a Kind-B Estimate."""
    b = np.asarray(boot_values, dtype=float)
    b = b[np.isfinite(b)]
    if b.size == 0:
        return Estimate(float(point), np.nan, np.nan, np.nan, np.nan)
    lo, hi = (100 - ci) / 2, 100 - (100 - ci) / 2
    return Estimate(
        point=float(point),
        mean=float(np.mean(b)),
        std=float(np.std(b, ddof=1)) if b.size > 1 else 0.0,
        ci_low=float(np.percentile(b, lo)),
        ci_high=float(np.percentile(b, hi)),
    )


def ka_summary(values_pct, ci: float = 95.0) -> tuple:
    """Kind-A: (mean, std, ci_low, ci_high) from bootstrap replicates, percent, 1dp."""
    b = np.asarray(values_pct, dtype=float)
    b = b[np.isfinite(b)]
    if b.size == 0:
        return (None, None, None, None)
    lo, hi = (100 - ci) / 2, 100 - (100 - ci) / 2
    return (_round1(np.mean(b)),
            _round1(np.std(b, ddof=1) if b.size > 1 else 0.0),
            _round1(np.percentile(b, lo)),
            _round1(np.percentile(b, hi)))


# --------------------------------------------------------------------------- #
#  Kind C: emulation-vs-trial agreement                                        #
# --------------------------------------------------------------------------- #
def divergence_z(rd_obs_pp: float, se_obs_pp: float,
                 rd_rct_pp: float, rct_ci_pp: tuple) -> float:
    """Standardized divergence between the emulation and the trial (Kind C).

        Z = (RD_obs - RD_rct) / sqrt(SE_obs^2 + SE_rct^2)

    |Z| < 1.96 == the emulation is statistically compatible with the trial, accounting
    for the uncertainty in BOTH. This replaces the old `inside_reference_ci` flag, which
    ignored the emulation's own uncertainty entirely and was therefore uninformative for
    trials with very tight intervals (e.g. STARRT-AKI, CI width 0.7pp).
    """
    lo, hi = rct_ci_pp
    se_rct = (hi - lo) / (2 * 1.959963984540054)
    denom = float(np.sqrt(se_obs_pp ** 2 + se_rct ** 2))
    if denom <= 0 or not np.isfinite(denom):
        return None
    return round(float((rd_obs_pp - rd_rct_pp) / denom), 3)


def ci_overlaps(ci_a: tuple, ci_b: tuple) -> bool:
    """Do two intervals overlap? An honest, weak compatibility check reported alongside Z."""
    (a_lo, a_hi), (b_lo, b_hi) = ci_a, ci_b
    if any(v is None or not np.isfinite(v) for v in (a_lo, a_hi, b_lo, b_hi)):
        return None
    return bool(a_lo <= b_hi and b_lo <= a_hi)


# --------------------------------------------------------------------------- #
#  Kind C: decisiveness of a null                                              #
# --------------------------------------------------------------------------- #
def minimum_detectable_effect(psi, keep=None, power: float = 0.80, alpha: float = 0.05):
    """Smallest true risk difference (pp) this cohort could detect, from the estimator's
    realized standard error. Uses the actual influence values, so it reflects the true
    precision after weighting and trimming, not a nominal N."""
    from scipy.stats import norm
    p = np.asarray(psi, float)
    if keep is not None:
        p = p[np.asarray(keep, bool)]
    p = p[np.isfinite(p)]
    if p.size < 2:
        return None
    se = float(p.std(ddof=1) / np.sqrt(p.size) * 100)
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return round(z * se, 2)


def minimum_detectable_bias_reduction(boot_diff, power: float = 0.80, alpha: float = 0.05):
    """THE metric that makes the paper's null decisive (Kind C).

    `boot_diff` are the PAIRED bootstrap replicates of a bias-reduction contrast
    (|bias_structured| - |bias_condition|). Their spread is the realized SE of that
    contrast. MDBR = (z_{1-a/2} + z_power) * SE.

    Reading: "this design could have detected a bias reduction of MDBR pp or larger with
    80% power; we observed X pp [CI]." Without this, a null is just an absence of
    evidence. With it, the null is a bounded, quantitative claim.
    """
    from scipy.stats import norm
    b = np.asarray(boot_diff, float)
    b = b[np.isfinite(b)]
    if b.size < 2:
        return None
    se = float(b.std(ddof=1))
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return round(z * se, 2)


def bootstrap_p_two_sided(boot_values) -> float:
    """Two-sided bootstrap p-value for H0: quantity = 0 (Kind C)."""
    b = np.asarray(boot_values, float)
    b = b[np.isfinite(b)]
    if b.size == 0:
        return 1.0
    n = b.size
    p = 2.0 * min((b >= 0).mean(), (b <= 0).mean())
    return float(min(max(p, 1.0 / n), 1.0))


def benjamini_hochberg(pvals) -> np.ndarray:
    """BH-FDR across a family, full precision. p-values never carry a mean/std/CI."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


# --------------------------------------------------------------------------- #
#  Kind B check: influence-function CI                                         #
# --------------------------------------------------------------------------- #
def influence_function_ci(psi, keep=None, ci: float = 95.0):
    """Influence-function 95% CI, reported as a check alongside the bootstrap CI.
    Returns (point, lo, hi, se) in percentage points."""
    p = np.asarray(psi, float)
    if keep is not None:
        p = p[np.asarray(keep, bool)]
    p = p[np.isfinite(p)]
    if p.size < 2:
        return (None, None, None, None)
    point = float(p.mean() * 100)
    se = float(p.std(ddof=1) / np.sqrt(p.size) * 100)
    z = 1.959963984540054
    return (_round1(point), _round1(point - z * se), _round1(point + z * se), se)


# --------------------------------------------------------------------------- #
#  Kind C: E-values                                                            #
# --------------------------------------------------------------------------- #
def _rd_to_rr(rd_pp: float, baseline_pct: float):
    p0 = baseline_pct / 100.0
    p1 = p0 + rd_pp / 100.0
    if not (0 < p0 < 1) or not (0 < p1 < 1):
        return None
    rr = p1 / p0
    return rr if rr >= 1 else 1.0 / rr


def e_value(rd_pp: float, baseline_pct: float):
    """Minimum strength of unmeasured confounding (risk-ratio scale) that could explain
    the effect away (VanderWeele & Ding)."""
    rr = _rd_to_rr(rd_pp, baseline_pct)
    if rr is None:
        return None
    return round(rr + np.sqrt(rr * (rr - 1)), 2)


def e_value_ci_limit(ci_low_pp: float, ci_high_pp: float, baseline_pct: float):
    """E-value for the CI limit nearest the null; 1.0 if the CI crosses 0."""
    if ci_low_pp is None or ci_high_pp is None:
        return None
    if ci_low_pp <= 0 <= ci_high_pp:
        return 1.0
    bound = ci_low_pp if abs(ci_low_pp) < abs(ci_high_pp) else ci_high_pp
    return e_value(bound, baseline_pct)


# --------------------------------------------------------------------------- #
#  Covariate balance                                                           #
# --------------------------------------------------------------------------- #
def standardized_mean_differences(X, A, weights=None):
    """Per-column SMD between arms, before weighting (weights=None) or after.
    s_pool comes from the UNWEIGHTED arm variances so before/after share a scale."""
    X = np.asarray(X, float)
    A = np.asarray(A, int)
    t, c = A == 1, A == 0
    with np.errstate(invalid="ignore"):
        sp = np.sqrt((np.nanvar(X[t], axis=0, ddof=1) + np.nanvar(X[c], axis=0, ddof=1)) / 2)
    sp = np.where(sp > 0, sp, np.nan)
    if weights is None:
        m1 = np.nanmean(X[t], axis=0)
        m0 = np.nanmean(X[c], axis=0)
    else:
        w = np.asarray(weights, float)

        def wm(mask):
            xm, wm_ = X[mask], w[mask]
            ok = ~np.isnan(xm)
            num = np.nansum(np.where(ok, xm * wm_[:, None], 0.0), axis=0)
            den = np.sum(np.where(ok, wm_[:, None], 0.0), axis=0)
            return num / np.where(den > 0, den, np.nan)

        m1, m0 = wm(t), wm(c)
    return (m1 - m0) / sp


# --------------------------------------------------------------------------- #
#  Formatting                                                                  #
# --------------------------------------------------------------------------- #
def _round1(x):
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else round(x, 1)


def round3(x):
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else round(x, 3)
