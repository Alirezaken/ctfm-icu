"""Embedding-covariate reduction for the fix variants (overlap/positivity).

Raw embeddings (768-dim/modality) as AIPW covariates destroy propensity overlap on
the small causal cohort. These reductions compress each modality before it enters
the estimator, restoring overlap while keeping the causal signal. Applied only in
fix-variant runs; the canonical run (config.yaml, §5) uses raw embeddings (kept as a
documented failure case).
"""
from __future__ import annotations

import numpy as np

_LGBM = dict(n_estimators=300, learning_rate=0.05, num_leaves=15,
             min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
             reg_lambda=1.0, verbosity=-1, n_jobs=4)


def _crossfit_proba(X, target, folds, seed):
    """Out-of-fold P(target=1 | X) via LightGBM (cross-fitted, no leakage)."""
    from sklearn.model_selection import KFold
    from lightgbm import LGBMClassifier
    target = np.asarray(target, int)
    s = np.zeros(len(target))
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(X):
        s[te] = LGBMClassifier(**_LGBM).fit(X[tr], target[tr]).predict_proba(X[te])[:, 1]
    return s


def apply(X, method: str, A=None, Y=None, folds: int = 5, seed: int = 42, k: int = 30):
    """Reduce a per-patient embedding matrix X (n x d) for one modality.

    - 'score' : Soroosh's approved fix -- TWO cross-fitted low-dim scores per
      modality: a propensity score P(A=1|X) and a separate prognostic score
      P(Y=1|X). Adjusting for both restores overlap and avoids the circularity of
      reusing a treatment-only score in the outcome model.
    - 'pscore': single cross-fitted propensity score P(A=1|X) (earlier variant).
    - 'pca'   : unsupervised PCA to k components (reported as a sensitivity).
    - 'none'  : unchanged (raw; documented failure case).
    """
    X = np.asarray(X, dtype=float)
    # C4: decouple the reduction cross-fit partition from the downstream AIPW one
    # (which uses `seed`), so the two layers do not share fold assignments (which would
    # let an AIPW test-fold patient influence a reduction score used as a train feature).
    red_seed = seed + 10007
    if method == "none" or X.shape[1] <= 1:
        return X
    if method == "pca":
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        Xs = StandardScaler().fit_transform(X)
        return PCA(n_components=min(k, Xs.shape[1]), random_state=red_seed).fit_transform(Xs)
    if method == "pscore":
        if A is None:
            raise ValueError("pscore reduction needs the treatment vector A")
        return _crossfit_proba(X, A, folds, red_seed).reshape(-1, 1)
    if method == "score":
        if A is None or Y is None:
            raise ValueError("score reduction needs both A (treatment) and Y (outcome)")
        ps = _crossfit_proba(X, A, folds, red_seed)     # propensity score
        pg = _crossfit_proba(X, Y, folds, red_seed)     # prognostic score
        return np.column_stack([ps, pg])
    raise ValueError(f"unknown reduction method '{method}'")
