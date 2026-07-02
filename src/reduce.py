"""Embedding-covariate reduction for the fix variants (overlap/positivity).

Raw embeddings (768-dim/modality) as AIPW covariates destroy propensity overlap on
the small causal cohort. These reductions compress each modality before it enters
the estimator, restoring overlap while keeping the causal signal. Applied only in
fix-variant runs; the canonical run (config.yaml, §5) uses raw embeddings.
"""
from __future__ import annotations

import numpy as np


def apply(X, method: str, A=None, folds: int = 5, seed: int = 42, k: int = 30):
    """Reduce a per-patient embedding matrix X (n x d) for one modality.

    - 'pscore': cross-fitted P(A=1 | X) -> one column (the sufficient balancing
      summary of the embedding for confounding). Best overlap by construction.
    - 'pca'   : unsupervised PCA to k components.
    - 'none'  : unchanged (raw).
    """
    X = np.asarray(X, dtype=float)
    if method == "none" or X.shape[1] <= 1:
        return X
    if method == "pca":
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        Xs = StandardScaler().fit_transform(X)
        return PCA(n_components=min(k, Xs.shape[1]), random_state=seed).fit_transform(Xs)
    if method == "pscore":
        from sklearn.model_selection import KFold
        from lightgbm import LGBMClassifier
        if A is None:
            raise ValueError("pscore reduction needs the treatment vector A")
        A = np.asarray(A, int)
        n = len(A)
        s = np.zeros(n)
        params = dict(n_estimators=300, learning_rate=0.05, num_leaves=15,
                      min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                      reg_lambda=1.0, verbosity=-1, n_jobs=4)
        for tr, te in KFold(folds, shuffle=True, random_state=seed).split(X):
            s[te] = LGBMClassifier(**params).fit(X[tr], A[tr]).predict_proba(X[te])[:, 1]
        return s.reshape(-1, 1)
    raise ValueError(f"unknown reduction method '{method}'")
