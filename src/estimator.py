"""Cross-fitted doubly-robust estimation. The one estimator used everywhere.

Three things here that the previous version got wrong and that matter:

1. NESTED CROSS-FITTING OF THE EMBEDDING REDUCTION.
   Embeddings are 768-d per modality. Feeding them raw to the nuisance models destroys
   propensity overlap, so each modality is first reduced to a low-dimensional score.
   That reduction is SUPERVISED (it uses A and Y), so it must be fit inside the
   cross-fitting loop, not before it.

   The failure mode, if you reduce once on the full cohort and then cross-fit AIPW on
   the result: a patient in AIPW's TEST fold contributed to the reduction model that
   produced the scores of patients in AIPW's TRAINING fold. Their outcome leaks into the
   training signal. Changing the reduction's random seed does NOT fix this -- it changes
   which rows share a fold, not the fact that every row's score saw every other row.
   The leak makes confidence intervals too narrow.

   Correct construction (implemented below):
     for each outer AIPW fold k:
         fit the reduction on outer-TRAIN rows only
         score outer-TRAIN rows by an INNER cross-fit (honest, out-of-fold)
         score outer-TEST rows with the model fit on all of outer-TRAIN (never saw them)
         fit the nuisance models on the outer-TRAIN design matrix
         predict the outer-TEST rows

2. IPCW EVERYWHERE, INCLUDING TMLE AND THE NEGATIVE CONTROL.
   D = 1 if vital status at t0+horizon is known, 0 if administratively censored. A
   censoring model Kc(X) = P(D=1|X) is cross-fitted on the same covariates; outcome
   models are fit on the observed only; the augmentation term is weighted by D/Kc.

3. OVERLAP WEIGHTS (ATO) REPORTED ALONGSIDE THE ATE.
   Adding high-dimensional covariates makes treatment near-deterministic for some
   patients, so the ATE becomes fragile exactly where this paper is looking. The ATO
   (Li, Morgan & Zaslavsky) weights by e(1-e), is bounded, and targets the
   clinical-equipoise population. Reporting both means a reviewer cannot dismiss the
   headline null as an artifact of a poorly-supported ATE.

All estimators return (psi, keep, diag) with the same contract:
  psi   per-patient influence value; the point estimate is mean(psi[keep]) * 100
  keep  overlap-trim mask
  diag  propensity range, trimmed fraction, ESS, ATO, n
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier

_LGBM = dict(n_estimators=300, learning_rate=0.05, num_leaves=15,
             min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
             reg_lambda=1.0, verbosity=-1, n_jobs=4)


# --------------------------------------------------------------------------- #
#  Nested reduction: fit inside each outer fold                                #
# --------------------------------------------------------------------------- #
def _design_for_fold(S, blocks, tr, te, A, Y, method, folds, seed, k):
    """Build the outer-fold design matrices (X_tr, X_te) with a properly nested reduction.

    S       structured covariates (used raw; low-dimensional, no reduction needed)
    blocks  list of raw embedding matrices, one per modality
    tr, te  outer-fold row indices
    """
    if not blocks:
        return S[tr], S[te]

    cols_tr, cols_te = [S[tr]], [S[te]]
    for bi, E in enumerate(blocks):
        E_tr = E[tr]
        if method == "none" or E.shape[1] <= 1:
            cols_tr.append(E[tr])
            cols_te.append(E[te])
            continue

        if method == "pca":
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler().fit(E_tr)
            pca = PCA(n_components=min(k, E_tr.shape[1]),
                      random_state=seed + bi).fit(sc.transform(E_tr))
            cols_tr.append(pca.transform(sc.transform(E_tr)))
            cols_te.append(pca.transform(sc.transform(E[te])))
            continue

        if method == "score":
            inner = KFold(folds, shuffle=True, random_state=seed + 1000 + bi)
            tr_scores, te_scores = [], []
            for target_all in (A, Y):
                t_tr = np.asarray(target_all, int)[tr]
                if len(np.unique(t_tr)) < 2:
                    const = float(t_tr.mean()) if len(t_tr) else 0.0
                    tr_scores.append(np.full(len(tr), const))
                    te_scores.append(np.full(len(te), const))
                    continue
                # honest OOF scores for the outer-training rows
                s_tr = np.zeros(len(tr))
                for i_tr, i_te in inner.split(E_tr):
                    if len(np.unique(t_tr[i_tr])) < 2:
                        s_tr[i_te] = float(t_tr[i_tr].mean())
                        continue
                    m = LGBMClassifier(**_LGBM).fit(E_tr[i_tr], t_tr[i_tr])
                    s_tr[i_te] = m.predict_proba(E_tr[i_te])[:, 1]
                # outer-test rows scored by a model fit on ALL outer-training rows
                m_full = LGBMClassifier(**_LGBM).fit(E_tr, t_tr)
                s_te = m_full.predict_proba(E[te])[:, 1]
                tr_scores.append(s_tr)
                te_scores.append(s_te)
            cols_tr.append(np.column_stack(tr_scores))
            cols_te.append(np.column_stack(te_scores))
            continue

        raise ValueError(f"unknown reduction method '{method}'")

    return np.hstack(cols_tr), np.hstack(cols_te)


# --------------------------------------------------------------------------- #
#  AIPW                                                                        #
# --------------------------------------------------------------------------- #
def crossfit_aipw(S, A, Y, folds, seed, blocks=None, reduction="none",
                  pca_components=30, trim=0.01, D=None):
    """Cross-fitted AIPW risk difference, with IPCW and a properly nested reduction.

    Returns (psi, keep, diag). Point estimate = mean(psi[keep]) * 100 (percentage points).
    diag carries the ATO, the propensity range, the trimmed fraction and the ESS.
    """
    S = np.asarray(S, dtype=float)
    A = np.asarray(A, int)
    Y = np.asarray(Y, int)
    n = len(Y)
    blocks = [np.asarray(b, dtype=float) for b in (blocks or [])]
    D = np.ones(n, int) if D is None else np.asarray(D, int)
    censored = D.min() == 0

    e = np.zeros(n); m1 = np.zeros(n); m0 = np.zeros(n); Kc = np.ones(n)
    kf = KFold(folds, shuffle=True, random_state=seed)

    for tr, te in kf.split(S):
        X_tr, X_te = _design_for_fold(S, blocks, tr, te, A, Y,
                                      reduction, folds, seed, pca_components)

        e[te] = LGBMClassifier(**_LGBM).fit(X_tr, A[tr]).predict_proba(X_te)[:, 1]

        if censored:
            if len(np.unique(D[tr])) > 1:
                Kc[te] = LGBMClassifier(**_LGBM).fit(X_tr, D[tr]).predict_proba(X_te)[:, 1]
            else:
                Kc[te] = float(D[tr].mean())

        for a, m in ((1, m1), (0, m0)):
            sel = np.where((A[tr] == a) & (D[tr] == 1))[0]     # outcome model on the observed
            if len(sel) == 0 or len(np.unique(Y[tr][sel])) < 2:
                m[te] = float(Y[tr][sel].mean()) if len(sel) else 0.0
                continue
            m[te] = LGBMClassifier(**_LGBM).fit(X_tr[sel], Y[tr][sel]).predict_proba(X_te)[:, 1]

    e = np.clip(e, 1e-6, 1 - 1e-6)
    Kc = np.clip(Kc, trim, 1.0)
    keep = (e > trim) & (e < 1 - trim)
    cw = D / Kc                                                # 1 when uncensored

    # ---- ATE (AIPW influence values) ----
    psi = (m1 - m0) + cw * (A * (Y - m1) / e - (1 - A) * (Y - m0) / (1 - e))

    # ---- ATO (overlap weights; Li, Morgan & Zaslavsky) ----
    # h = e(1-e) is bounded and maximal at e=0.5, so the ATO targets the
    # clinical-equipoise population and is stable exactly where the ATE is not.
    h = e * (1 - e)
    psi_ato = (h * (m1 - m0)
               + cw * (A * (1 - e) * (Y - m1) - (1 - A) * e * (Y - m0)))
    ato = float(psi_ato.sum() / h.sum() * 100) if h.sum() > 0 else np.nan

    w = np.where(A == 1, 1.0 / e, 1.0 / (1 - e))
    ess = (w[keep].sum() ** 2) / np.sum(w[keep] ** 2) if keep.any() else 0.0

    diag = {
        "e": e, "e_min": float(e.min()), "e_max": float(e.max()),
        "frac_trimmed": float((~keep).mean()),
        "ess": float(ess), "n": int(keep.sum()),
        "frac_censored": float((D == 0).mean()),
        "ato": ato, "psi_ato": psi_ato, "h_ato": h,
    }
    return psi, keep, diag


# --------------------------------------------------------------------------- #
#  TMLE (robustness swap)                                                      #
# --------------------------------------------------------------------------- #
def crossfit_tmle(S, A, Y, folds, seed, blocks=None, reduction="none",
                  pca_components=30, trim=0.01, D=None):
    """Cross-fitted TMLE for the risk difference. Same interface and IPCW handling as
    crossfit_aipw, so the estimator swap in the robustness stage is like-for-like."""
    from scipy.special import logit, expit

    S = np.asarray(S, dtype=float)
    A = np.asarray(A, int)
    Y = np.asarray(Y, int)
    n = len(Y)
    blocks = [np.asarray(b, dtype=float) for b in (blocks or [])]
    D = np.ones(n, int) if D is None else np.asarray(D, int)
    censored = D.min() == 0

    g = np.zeros(n); Q0 = np.zeros(n); Q1 = np.zeros(n); Kc = np.ones(n)
    kf = KFold(folds, shuffle=True, random_state=seed)

    for tr, te in kf.split(S):
        X_tr, X_te = _design_for_fold(S, blocks, tr, te, A, Y,
                                      reduction, folds, seed, pca_components)
        g[te] = LGBMClassifier(**_LGBM).fit(X_tr, A[tr]).predict_proba(X_te)[:, 1]

        if censored and len(np.unique(D[tr])) > 1:
            Kc[te] = LGBMClassifier(**_LGBM).fit(X_tr, D[tr]).predict_proba(X_te)[:, 1]

        obs = np.where(D[tr] == 1)[0]
        if len(obs) == 0 or len(np.unique(Y[tr][obs])) < 2:
            Q1[te] = Q0[te] = float(Y[tr][obs].mean()) if len(obs) else 0.0
            continue
        om = LGBMClassifier(**_LGBM).fit(
            np.column_stack([X_tr[obs], A[tr][obs]]), Y[tr][obs])
        Q1[te] = om.predict_proba(np.column_stack([X_te, np.ones(len(te))]))[:, 1]
        Q0[te] = om.predict_proba(np.column_stack([X_te, np.zeros(len(te))]))[:, 1]

    eps_c = 1e-6
    g = np.clip(g, trim, 1 - trim)
    Kc = np.clip(Kc, trim, 1.0)
    Q0 = np.clip(Q0, eps_c, 1 - eps_c)
    Q1 = np.clip(Q1, eps_c, 1 - eps_c)
    QA = np.where(A == 1, Q1, Q0)

    cw = D / Kc
    H = cw * (A / g - (1 - A) / (1 - g))               # clever covariate, IPCW-weighted

    # targeting: 1-D Newton step for the fluctuation eps (logistic, offset logit(QA))
    off = logit(QA)
    eps = 0.0
    for _ in range(100):
        p = expit(off + eps * H)
        score = np.sum(D * H * (Y - p))
        info = np.sum(D * H * H * p * (1 - p))
        if info < 1e-12:
            break
        step = score / info
        eps += step
        if abs(step) < 1e-8:
            break

    Q1s = expit(logit(Q1) + eps * (1.0 / g))
    Q0s = expit(logit(Q0) - eps * (1.0 / (1 - g)))
    QAs = expit(off + eps * H)

    keep = (g > trim) & (g < 1 - trim)
    psi = (Q1s - Q0s) + H * (Y - QAs)

    w = np.where(A == 1, 1.0 / g, 1.0 / (1 - g))
    ess = (w[keep].sum() ** 2) / np.sum(w[keep] ** 2) if keep.any() else 0.0

    h = g * (1 - g)
    psi_ato = h * (Q1s - Q0s) + cw * (A * (1 - g) * (Y - Q1s) - (1 - A) * g * (Y - Q0s))
    ato = float(psi_ato.sum() / h.sum() * 100) if h.sum() > 0 else np.nan

    diag = {
        "e": g, "e_min": float(g.min()), "e_max": float(g.max()),
        "frac_trimmed": float((~keep).mean()),
        "ess": float(ess), "n": int(keep.sum()),
        "frac_censored": float((D == 0).mean()),
        "ato": ato, "psi_ato": psi_ato, "h_ato": h,
    }
    return psi, keep, diag


def get_estimator(name: str):
    return crossfit_tmle if name == "tmle" else crossfit_aipw


def ato_from_boot(psi_ato, h_ato, idx):
    """Bootstrap replicate of the ATO on a resampled index set."""
    hs = h_ato[idx].sum()
    return float(psi_ato[idx].sum() / hs * 100) if hs > 0 else np.nan
