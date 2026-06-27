"""§9.6 / §6.5  probe -- validity probe (a GATE, run before estimating).

Held-out linear-probe prediction of each proxy's target confounder:
  - image proxy -> labeled pulmonary edema (MIMIC-CXR-JPG Edema label).
  - note proxy  -> its target confounder (only if a labeled target exists).
A logistic-regression probe is trained on the official train split and evaluated
on the test split; AUROC and AUPRC reported as Kind A (bootstrap mean/std/95% CI,
percent, 1 dp) -> probe.csv. If a modality cannot predict its confounder, STOP and
tell Soroosh (§9.6); we log a loud GATE-FAIL but still record the numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.util import log
from src import events as ev
from src import results as R

_GATE_AUROC = 0.70   # below this on the test split -> gate fails, tell Soroosh


def _ka_summary(values_pct):
    """Kind-A summary (percent, 1 dp): mean/std/95% CI of a bootstrap vector."""
    b = np.asarray(values_pct, float); b = b[np.isfinite(b)]
    r = lambda x: round(float(x), 1)
    return r(np.mean(b)), r(np.std(b, ddof=1)), r(np.percentile(b, 2.5)), r(np.percentile(b, 97.5))


def _image_probe(cfg):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, average_precision_score
    from src.stats import cluster_bootstrap_indices

    label = cfg.get("interventions.fluids_sepsis.imaging_confounder_label", "edema")
    idx, V = ev.load_embeddings(cfg, "images")
    cxr = ev.link(cfg, "cxr_studies")[["study_id", label, "split"]]
    df = idx.merge(cxr, on="study_id", how="left").reset_index(drop=True)

    # CheXpert encoding here: 1=positive, 0=negative, 2=not-mentioned, 3=uncertain.
    # Edema has essentially no explicit 0s, so use the standard convention:
    # positive = 1; negative = explicit-negative OR not-mentioned (0 or 2);
    # drop uncertain (3) and missing.
    y = pd.to_numeric(df[label], errors="coerce")
    keep = y.isin([0, 1, 2]).to_numpy()
    df, X = df[keep].reset_index(drop=True), V[keep]
    y = (y[keep] == 1).astype(int).to_numpy()
    tr = (df["split"] == "train").to_numpy()
    te = (df["split"] == "test").to_numpy()
    log(f"  image probe '{label}': train={tr.sum():,} test={te.sum():,} "
        f"(pos rate train={y[tr].mean():.3f} test={y[te].mean():.3f})")

    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(X[tr]), y[tr])
    p = clf.predict_proba(sc.transform(X[te]))[:, 1]
    yte = y[te]
    auroc = roc_auc_score(yte, p) * 100
    auprc = average_precision_score(yte, p) * 100

    # Kind-A bootstrap: cluster by patient on the test set
    subj = df.loc[te, "subject_id"].to_numpy()
    seed = cfg.get("run.seed", 42)
    n_boot = min(1000, cfg.get("bootstrap.n_resamples", 1000))
    bo_roc, bo_prc = [], []
    for bidx in cluster_bootstrap_indices(subj, n_boot, seed):
        yb, pb = yte[bidx], p[bidx]
        if yb.min() == yb.max():
            continue
        bo_roc.append(roc_auc_score(yb, pb) * 100)
        bo_prc.append(average_precision_score(yb, pb) * 100)
    rm, rs, rl, rh = _ka_summary(bo_roc)
    pm, ps, pl, ph = _ka_summary(bo_prc)
    gate = "PASS" if auroc / 100 >= _GATE_AUROC else "FAIL"
    log(f"  image probe '{label}': AUROC={auroc:.1f}  AUPRC={auprc:.1f}  GATE={gate}")
    if gate == "FAIL":
        log(f"  *** GATE-FAIL: image proxy cannot predict {label} (AUROC<{_GATE_AUROC}); "
            f"tell Soroosh before estimating (§9.6).")
    return {"modality": "image", "target_confounder": label,
            "auroc_mean": rm, "auroc_std": rs, "auroc_ci_low": rl, "auroc_ci_high": rh,
            "auprc_mean": pm, "auprc_std": ps, "auprc_ci_low": pl, "auprc_ci_high": ph}


def run(cfg, force: bool = False, intervention: str = "fluids_sepsis"):
    cfg.require(f"interventions.{intervention}.imaging_confounder_label")
    log(f"probe[{intervention}]: image validity gate ...")
    rows = [_image_probe(cfg)]

    note_target = cfg.get(f"interventions.{intervention}.notes_confounder")
    if note_target:
        log(f"  note probe target '{note_target}': no labeled ground truth available "
            f"-> deferred (needs a label definition from Soroosh).")
    R.append_rows(cfg, "probe.csv", rows)
    log("probe done -> probe.csv")
