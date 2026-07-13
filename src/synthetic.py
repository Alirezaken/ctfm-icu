"""SEMI-SYNTHETIC BENCHMARK -- the proof that the null is a fact about the data,
not a failure of the method.

The critique that would kill this paper if we had no answer to it:
    "You found no bias reduction from imaging. How do we know your estimator isn't just
     broken, or your embedding pipeline isn't just wrong?"

The answer has to be a setting where the TRUE causal effect is known and where imaging
genuinely IS a confounder, and where we then show the same code recovers the truth. That
is what this module builds.

CONSTRUCTION (semi-synthetic: everything real except A and Y)
  real:      the cohort, the structured covariates S, the RAD-DINO image embeddings
  real:      U = cross-fitted P(edema | image embedding)
             This is not a made-up confounder. It is the actual, validated signal the
             image channel carries -- the same signal that scores AUROC 89.8 in the probe.
  derived:   U_perp = U - proj(U | S)
             The component of the image signal that is NOT already in the structured
             record. THIS is the only part that can possibly do any confounding work,
             and isolating it is the whole point.
  simulated: A ~ Bernoulli(sigmoid(a0 + a1'S~ + gamma * U_perp~))
             Y ~ Bernoulli(sigmoid(b0 + b1'S~ + delta * U_perp~ + tau * A))
             with tau calibrated so the true risk difference equals a known target.

  gamma  = how strongly the image-borne confounder drives TREATMENT
  delta  = how strongly it drives the OUTCOME
  Confounding bias in a structured-only analysis requires BOTH gamma>0 and delta>0.

WHAT THE SWEEP PROVES
  1. POSITIVE CONTROL. At gamma,delta > 0 the `structured` condition is biased by a known
     amount and `struct_img` recovers tau. The estimator, the embeddings, and the
     reduction all work. A null on real data therefore means something.
  2. CALIBRATION CURVE. Bias reduction is a smooth, monotone function of the measured ICI
     (the diagnostic in src/diagnostic.py). The diagnostic PREDICTS what the estimator
     will deliver, before you run it.
  3. THE MECHANISM. The redundancy sweep holds total confounding fixed (gamma, delta
     constant) and varies only how much of U is already predictable from S. Bias reduction
     falls to zero as redundancy -> 1, even though the confounder is just as strong. This
     is the exact regime the four real interventions sit in.

THE PAYOFF FIGURE
  Plot observed bias reduction against measured ICI for all 25 synthetic cells. Then
  overlay the four real interventions at their measured ICI. They land at ICI ~ 0, where
  the curve says bias reduction should be ~0 -- and where it is. The theory predicts the
  null.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

from src.estimator import _LGBM
from src.util import log


def image_confounder(V_img, edema_label, folds, seed):
    """U = cross-fitted P(edema | image embedding).

    Real, validated image signal -- not a synthetic construct. Cross-fitted so that no
    patient's own label contributes to their own U.
    """
    V = np.asarray(V_img, float)
    y = np.asarray(edema_label, int)
    u = np.zeros(len(y))
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(V):
        if len(np.unique(y[tr])) < 2:
            u[te] = float(y[tr].mean())
            continue
        u[te] = LGBMClassifier(**_LGBM).fit(V[tr], y[tr]).predict_proba(V[te])[:, 1]
    return u


def residualize(u, S):
    """U_perp = U - proj(U | S): the part of the image signal not already in structured.

    If the structured record already predicts U perfectly, U_perp is zero and the image
    can carry no incremental confounding no matter how informative it is about edema.
    That is the paper's whole argument, made arithmetic.
    """
    S = np.nan_to_num(np.asarray(S, float), nan=0.0)
    u = np.asarray(u, float)
    S_std = StandardScaler().fit_transform(S)
    fit = LinearRegression().fit(S_std, u)
    resid = u - fit.predict(S_std)
    r2 = float(fit.score(S_std, u))
    return resid, r2


def _std(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else np.zeros_like(x)


def _calibrate_tau(p_base, target_rd):
    """Find the logit-scale tau giving the requested true risk difference.

    The true RD under the simulated DGP is E[Y(1)] - E[Y(0)] = mean(expit(l + tau)) -
    mean(expit(l)), which is monotone in tau, so a bisection is exact and cheap.
    """
    lo, hi = -6.0, 6.0
    for _ in range(80):
        mid = (lo + hi) / 2
        rd = float(np.mean(expit(p_base + mid)) - np.mean(expit(p_base)))
        if rd < target_rd:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def simulate(S, U_perp, gamma, delta, target_rd_pp, rng, s_coef_treat=None,
             s_coef_out=None):
    """Simulate (A, Y) with a known true risk difference.

    S        real structured covariates (drive both A and Y, as in real life)
    U_perp   the incremental image signal (the confounder we control)
    gamma    U_perp -> A strength
    delta    U_perp -> Y strength
    Returns (A, Y, true_rd_pp).
    """
    S = np.nan_to_num(np.asarray(S, float), nan=0.0)
    Ss = StandardScaler().fit_transform(S)
    n, d = Ss.shape
    u = _std(U_perp)

    # fixed structured coefficients across the sweep so that only gamma/delta vary
    if s_coef_treat is None:
        s_coef_treat = rng.normal(0, 0.3, d)
    if s_coef_out is None:
        s_coef_out = rng.normal(0, 0.3, d)

    lin_a = Ss @ s_coef_treat + gamma * u
    lin_a = lin_a - lin_a.mean()                 # centre -> ~50% treated, good overlap
    A = rng.binomial(1, expit(lin_a))

    lin_y = Ss @ s_coef_out + delta * u
    lin_y = lin_y - lin_y.mean() - 0.8           # base risk ~31%
    tau = _calibrate_tau(lin_y, target_rd_pp / 100.0)
    true_rd = float(np.mean(expit(lin_y + tau)) - np.mean(expit(lin_y))) * 100

    Y = rng.binomial(1, expit(lin_y + tau * A))
    return A.astype(int), Y.astype(int), true_rd


def make_redundant(U_perp, S, rho, rng):
    """Build a confounder that is `rho` fraction predictable from the structured set.

    rho = 0  -> the confounder is entirely outside S (the image is the only way to see it)
    rho = 1  -> the confounder is entirely inside S (the image is fully redundant)

    Total confounding strength is held constant by re-standardizing, so any change in bias
    reduction across the sweep is attributable to REDUNDANCY and nothing else. This is the
    experiment that isolates the mechanism.
    """
    S = np.nan_to_num(np.asarray(S, float), nan=0.0)
    Ss = StandardScaler().fit_transform(S)
    # a fixed random direction in structured space = the "already measured" component
    w = rng.normal(0, 1, Ss.shape[1])
    inside = _std(Ss @ w)
    outside = _std(U_perp)
    mixed = np.sqrt(rho) * inside + np.sqrt(max(0.0, 1.0 - rho)) * outside
    return _std(mixed)
