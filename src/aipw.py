"""Cross-fitted AIPW (§5): the one estimator everywhere.

Risk difference E[Y^1]-E[Y^0] via augmented IPW with LightGBM nuisance models for
both the propensity e(X)=P(A=1|X) and the outcome m_a(X)=E[Y|A=a,X], cross-fitted
(out-of-fold predictions) because the embedding covariates are high-dimensional.

Returns the per-patient AIPW influence value psi (so CIs come from resampling psi
rather than refitting 10k times), an overlap-trim keep mask, and diagnostics. The
point estimate is mean(psi[keep]); a condition's design matrix X selects the
adjustment set (naive uses a constant column).
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier

_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=15,
               min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
               reg_lambda=1.0, verbosity=-1, n_jobs=4)


def crossfit_aipw(X, A, Y, folds, seed, trim=0.01, D=None):
    """Cross-fitted AIPW risk difference, with optional inverse-probability-of-
    censoring weighting (§4). D = observed indicator (1 = outcome status known at the
    horizon, 0 = administratively censored). When D is None or all-ones this is plain
    AIPW; otherwise a censoring model Kc(X)=P(D=1|X) is cross-fitted (same covariates
    as the outcome model), outcome models are fit on the observed only, and the
    augmentation term is weighted by D/Kc so the risk difference stays well defined."""
    X = np.asarray(X, dtype=float); A = np.asarray(A, int); Y = np.asarray(Y, int)
    n = len(Y)
    D = np.ones(n, int) if D is None else np.asarray(D, int)
    e = np.zeros(n); m1 = np.zeros(n); m0 = np.zeros(n); Kc = np.ones(n)
    censored = D.min() == 0
    kf = KFold(folds, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        e[te] = LGBMClassifier(**_PARAMS).fit(X[tr], A[tr]).predict_proba(X[te])[:, 1]
        if censored:
            Kc[te] = LGBMClassifier(**_PARAMS).fit(X[tr], D[tr]).predict_proba(X[te])[:, 1]
        for a, m in ((1, m1), (0, m0)):
            sel = tr[(A[tr] == a) & (D[tr] == 1)]        # outcome model on the observed
            m[te] = LGBMClassifier(**_PARAMS).fit(X[sel], Y[sel]).predict_proba(X[te])[:, 1]
    e = np.clip(e, 1e-6, 1 - 1e-6)
    Kc = np.clip(Kc, trim, 1.0)
    keep = (e > trim) & (e < 1 - trim)
    cw = D / Kc                                          # censoring weight (1 when uncensored)
    psi = (m1 - m0) + cw * (A * (Y - m1) / e - (1 - A) * (Y - m0) / (1 - e))
    w = np.where(A == 1, 1.0 / e, 1.0 / (1 - e))
    ess = (w[keep].sum() ** 2) / np.sum(w[keep] ** 2)
    diag = {"e_min": float(e.min()), "e_max": float(e.max()),
            "frac_trimmed": float((~keep).mean()), "ess": float(ess),
            "n": int(keep.sum()), "e": e,                # propensity, for §7.6 balance/weighting
            "frac_censored": float((D == 0).mean())}
    return psi, keep, diag


def crossfit_tmle(X, A, Y, folds, seed, trim=0.01):
    """Cross-fitted TMLE for the risk difference (robustness swap for AIPW).

    Same interface/return as crossfit_aipw: a per-patient efficient-influence
    pseudo-outcome psi (mean = the targeted ATE), an overlap-trim mask, diagnostics.
    """
    from scipy.special import logit, expit
    X = np.asarray(X, dtype=float); A = np.asarray(A, int); Y = np.asarray(Y, int)
    n = len(Y)
    g = np.zeros(n); Q0 = np.zeros(n); Q1 = np.zeros(n)
    kf = KFold(folds, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        g[te] = LGBMClassifier(**_PARAMS).fit(X[tr], A[tr]).predict_proba(X[te])[:, 1]
        om = LGBMClassifier(**_PARAMS).fit(np.column_stack([X[tr], A[tr]]), Y[tr])
        Q1[te] = om.predict_proba(np.column_stack([X[te], np.ones(len(te))]))[:, 1]
        Q0[te] = om.predict_proba(np.column_stack([X[te], np.zeros(len(te))]))[:, 1]
    eps_c = 1e-6
    g = np.clip(g, trim, 1 - trim)
    Q0 = np.clip(Q0, eps_c, 1 - eps_c); Q1 = np.clip(Q1, eps_c, 1 - eps_c)
    QA = np.where(A == 1, Q1, Q0)
    H = A / g - (1 - A) / (1 - g)                       # clever covariate
    # targeting: 1-D Newton for the fluctuation eps (logistic, offset logit(QA))
    off = logit(QA); eps = 0.0
    for _ in range(100):
        p = expit(off + eps * H)
        score = np.sum(H * (Y - p)); info = np.sum(H * H * p * (1 - p))
        if info < 1e-12:
            break
        step = score / info; eps += step
        if abs(step) < 1e-8:
            break
    Q1s = expit(logit(Q1) + eps * (1.0 / g))
    Q0s = expit(logit(Q0) - eps * (1.0 / (1 - g)))
    QAs = expit(off + eps * H)
    keep = (g > trim) & (g < 1 - trim)
    psi = (Q1s - Q0s) + H * (Y - QAs)                  # efficient influence pseudo-outcome
    w = np.where(A == 1, 1.0 / g, 1.0 / (1 - g))
    ess = (w[keep].sum() ** 2) / np.sum(w[keep] ** 2)
    diag = {"e_min": float(g.min()), "e_max": float(g.max()),
            "frac_trimmed": float((~keep).mean()), "ess": float(ess), "n": int(keep.sum())}
    return psi, keep, diag
