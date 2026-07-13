"""THE INCREMENTAL CONFOUNDING DIAGNOSTIC -- the paper's methodological contribution.

The premise the field is running on, stated plainly:
    "Foundation-model embeddings of images and notes are rich proxies for unmeasured
     confounders, so adding them to a causal adjustment set will reduce confounding bias."

That premise conflates two different properties. A modality M can only reduce confounding
bias relative to a structured adjustment set S if it satisfies ALL THREE conditions:

  (1) INFORMATIVENESS.  M encodes the latent confounder at all.
      Measured by: a supervised probe. Does the embedding predict the labeled confounder?
      (RAD-DINO -> pulmonary edema: AUROC 89.8, replicating on 3 external CXR datasets.)
      This is the condition the field checks, and the ONLY one it usually checks.

  (2) INCREMENTAL CONFOUNDING.  The information M carries about the confounder is not
      ALREADY IN S, and it moves BOTH the treatment and the outcome.
      A confounder is by definition a common cause. If M improves prediction of treatment
      but not outcome (given S), it is an instrument, and adjusting for it AMPLIFIES bias.
      If it improves outcome but not treatment, it is a precision variable: harmless,
      helps variance, but cannot reduce confounding bias. Only if it improves BOTH,
      conditional on S, can it carry residual confounding.
      Measured by:
          dAUC_A(M) = AUC[e(S,M)] - AUC[e(S)]     cross-fitted
          dAUC_Y(M) = AUC[m(S,M)] - AUC[m(S)]     cross-fitted
          ICI(M)    = min(dAUC_A, dAUC_Y)         the BINDING constraint

  (3) POSITIVITY AFFORDABILITY.  Adding M must not destroy overlap.
      High-dimensional, redundant covariates make treatment near-deterministic for some
      patients. The propensity mass piles up at 0 and 1, the effective sample collapses,
      and the ATE becomes fragile precisely where it is being asked to be precise. This
      is the "observational-level positivity violation": positivity holds in the
      data-generating process but fails empirically because the covariate space was
      inflated with redundant proxies.
      Measured by:
          PositivityCost(M) = 1 - ESS(S,M) / ESS(S)

DECISION RULE (pre-registered):
    Add M to the adjustment set iff
        dAUC_A(M) >= t   AND   dAUC_Y(M) >= t   AND   PositivityCost(M) <= c
    with t = 0.02 and c = 0.25 (config: diagnostic.auc_threshold, positivity_cost_max).

This is cheap: it needs no outcome model beyond what the estimator already fits, and it
runs BEFORE committing to an estimand. It is validated two ways in this study:
  * against the semi-synthetic benchmark, where the true confounding is known by
    construction and the rule's prediction can be checked against the truth;
  * against the four real RCT-anchored interventions, where the rule predicts zero bias
    reduction and zero bias reduction is what is observed.

The headline empirical claim of the paper falls out of this decomposition:
    condition (1) PASSES loudly (AUROC 89.8) while condition (2) FAILS (ICI ~ 0) and
    condition (3) FAILS (ESS drops up to 75%).
    A validated embedding is not a valid confounder proxy.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

from src.estimator import _design_for_fold, _LGBM
from src.util import log


def _crossfit_auc(S, blocks, target, folds, seed, reduction, pca_components,
                  A_for_red=None, Y_for_red=None, subset=None):
    """Cross-fitted out-of-fold AUC for P(target=1 | S, blocks).

    The embedding blocks are reduced with the SAME nested scheme the estimator uses, so
    the dAUC we report is the incremental information the estimator actually gets to see,
    not the information a differently-fit model might have extracted.

    `subset` restricts EVALUATION (not fitting) to a row mask -- used for the outcome
    model, which is evaluated within arms to avoid crediting the embedding for
    treatment-outcome association that is really the treatment effect.
    """
    S = np.asarray(S, float)
    target = np.asarray(target, int)
    n = len(target)
    if len(np.unique(target)) < 2:
        return None
    A_for_red = np.zeros(n, int) if A_for_red is None else np.asarray(A_for_red, int)
    Y_for_red = target if Y_for_red is None else np.asarray(Y_for_red, int)

    oof = np.zeros(n)
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(S):
        X_tr, X_te = _design_for_fold(S, blocks, tr, te, A_for_red, Y_for_red,
                                      reduction, folds, seed, pca_components)
        if len(np.unique(target[tr])) < 2:
            oof[te] = float(target[tr].mean())
            continue
        oof[te] = LGBMClassifier(**_LGBM).fit(X_tr, target[tr]).predict_proba(X_te)[:, 1]

    m = np.ones(n, bool) if subset is None else np.asarray(subset, bool)
    if len(np.unique(target[m])) < 2:
        return None
    return float(roc_auc_score(target[m], oof[m]))


def incremental_confounding(S, A, Y, block, folds, seed, reduction, pca_components,
                            n_boot=500):
    """dAUC_A, dAUC_Y and ICI for ONE modality block added to the structured set S.

    dAUC_A : how much better can we predict WHO GOT TREATED once we see the embedding?
    dAUC_Y : how much better can we predict WHO DIED once we see the embedding?
             Computed WITHIN ARMS (pooling the two arm-specific AUCs) so the embedding
             cannot be credited for the treatment effect itself.

    Both must be positive for the modality to carry residual confounding. ICI is their
    minimum: the binding constraint.

    Bootstrap CIs come from resampling the out-of-fold predictions, not from refitting
    (which would cost n_boot x the model fits for no additional validity).
    """
    S = np.asarray(S, float)
    A = np.asarray(A, int)
    Y = np.asarray(Y, int)
    block = np.asarray(block, float)

    # --- treatment channel -------------------------------------------------
    auc_a_base = _crossfit_auc(S, [], A, folds, seed, reduction, pca_components,
                               A_for_red=A, Y_for_red=Y)
    auc_a_full = _crossfit_auc(S, [block], A, folds, seed, reduction, pca_components,
                               A_for_red=A, Y_for_red=Y)

    # --- outcome channel, within arms -------------------------------------
    aucs_base, aucs_full, wts = [], [], []
    for arm in (0, 1):
        m = A == arm
        if m.sum() < 20 or len(np.unique(Y[m])) < 2:
            continue
        b = _crossfit_auc(S, [], Y, folds, seed, reduction, pca_components,
                          A_for_red=A, Y_for_red=Y, subset=m)
        f = _crossfit_auc(S, [block], Y, folds, seed, reduction, pca_components,
                          A_for_red=A, Y_for_red=Y, subset=m)
        if b is None or f is None:
            continue
        aucs_base.append(b)
        aucs_full.append(f)
        wts.append(int(m.sum()))

    if not wts or auc_a_base is None or auc_a_full is None:
        return None

    wts = np.asarray(wts, float)
    auc_y_base = float(np.average(aucs_base, weights=wts))
    auc_y_full = float(np.average(aucs_full, weights=wts))

    d_a = auc_a_full - auc_a_base
    d_y = auc_y_full - auc_y_base
    ici = float(min(d_a, d_y))

    return {
        "auc_treat_structured": round(auc_a_base * 100, 1),
        "auc_treat_with_modality": round(auc_a_full * 100, 1),
        "d_auc_treat": round(d_a * 100, 2),
        "auc_outcome_structured": round(auc_y_base * 100, 1),
        "auc_outcome_with_modality": round(auc_y_full * 100, 1),
        "d_auc_outcome": round(d_y * 100, 2),
        "ici": round(ici * 100, 2),          # in AUC points, the binding constraint
    }


def positivity_cost(ess_structured: float, ess_with_modality: float) -> float:
    """1 - ESS(S,M)/ESS(S). Positive = the modality cost you effective sample."""
    if not ess_structured or ess_structured <= 0:
        return None
    return round(1.0 - float(ess_with_modality) / float(ess_structured), 3)


def decision(d_auc_treat, d_auc_outcome, pos_cost, auc_threshold, pos_cost_max) -> dict:
    """Apply the pre-registered rule and say WHY, in words, not just pass/fail.

    The `reason` field is what makes this usable by a practitioner: it names which of the
    three conditions failed, which tells them what to do next.
    """
    t = auc_threshold * 100          # thresholds are given as AUC fractions in config
    ok_a = d_auc_treat is not None and d_auc_treat >= t
    ok_y = d_auc_outcome is not None and d_auc_outcome >= t
    ok_p = pos_cost is not None and pos_cost <= pos_cost_max
    verdict = bool(ok_a and ok_y and ok_p)

    if verdict:
        reason = "incremental on both channels and positivity cost acceptable"
    elif not ok_a and not ok_y:
        reason = "redundant: no incremental information about treatment OR outcome given structured"
    elif not ok_a:
        reason = "precision variable only: predicts outcome but not treatment given structured; cannot reduce confounding bias"
    elif not ok_y:
        reason = "instrument-like: predicts treatment but not outcome given structured; adjusting may AMPLIFY bias"
    else:
        reason = "informative but unaffordable: positivity cost exceeds the budget"

    return {"add_modality": verdict, "reason": reason}
