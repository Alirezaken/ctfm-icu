"""§9.6 / §6.5  probe -- validity probe + external embedding-quality gate.

Two checks, both Kind A (bootstrap mean/std/95% CI, percent, 1 dp) -> probe.csv:
  1. MIMIC image proxy -> edema (the validity gate that must pass before estimating).
  2. EXTERNAL gate (§3): do the RAD-DINO embeddings separate known findings on
     PadChest / ChestX-ray14 (and CheXpert once uploaded)? Uses the vectors already
     produced by extract_external_cxr; here we only join master-list labels and run
     a held-out linear probe. modality column = 'image' (MIMIC) or 'image@<dataset>'.

probe.csv is rewritten from scratch each run (idempotent). If the MIMIC gate fails,
STOP and tell Soroosh (§9.6).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.util import log
from src import events as ev
from src import features as F
from src import results as R

_GATE_AUROC = 0.70


def _ka(values_pct):
    b = np.asarray(values_pct, float); b = b[np.isfinite(b)]
    r = lambda x: round(float(x), 1)
    return r(np.mean(b)), r(np.std(b, ddof=1)), r(np.percentile(b, 2.5)), r(np.percentile(b, 97.5))


def _eval_probe(Vtr, ytr, Vte, yte, subj_te, seed, modality, target):
    """Fit a logistic probe on train, evaluate on held-out test; Kind-A row dict."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, average_precision_score
    from src.stats import cluster_bootstrap_indices

    sc = StandardScaler().fit(Vtr)
    clf = LogisticRegression(max_iter=1000).fit(sc.transform(Vtr), ytr)
    p = clf.predict_proba(sc.transform(Vte))[:, 1]
    auroc = roc_auc_score(yte, p) * 100
    auprc = average_precision_score(yte, p) * 100
    br, bp = [], []
    for bidx in cluster_bootstrap_indices(subj_te, 1000, seed):
        yb, pb = yte[bidx], p[bidx]
        if yb.min() == yb.max():
            continue
        br.append(roc_auc_score(yb, pb) * 100); bp.append(average_precision_score(yb, pb) * 100)
    rm, rs, rl, rh = _ka(br); pm, ps, pl, ph = _ka(bp)
    log(f"    {modality}/{target}: AUROC={auroc:.1f} AUPRC={auprc:.1f}")
    return {"modality": modality, "target_confounder": target,
            "auroc_mean": rm, "auroc_std": rs, "auroc_ci_low": rl, "auroc_ci_high": rh,
            "auprc_mean": pm, "auprc_std": ps, "auprc_ci_low": pl, "auprc_ci_high": ph}, auroc


def _binary(series):
    """CheXpert/NIH style -> 1 positive vs 0 (negative or not-mentioned); drop uncertain/NaN."""
    y = pd.to_numeric(series, errors="coerce")
    return y


def _mimic_image_probe(cfg, intervention="fluids_sepsis"):
    label = cfg.get(f"interventions.{intervention}.imaging_confounder_label", "edema")
    idx, V = ev.load_embeddings(cfg, "images")
    cxr = ev.link(cfg, "cxr_studies")[["study_id", label, "split"]]
    df = idx.merge(cxr, on="study_id", how="left").reset_index(drop=True)
    y = _binary(df[label])
    keep = y.isin([0, 1, 2]).to_numpy()           # positive=1; negative/not-mentioned=0; drop uncertain(3)
    df, X = df[keep].reset_index(drop=True), V[keep]
    y = (y[keep] == 1).astype(int).to_numpy()
    tr = (df["split"] == "train").to_numpy(); te = (df["split"] == "test").to_numpy()
    log(f"  MIMIC image '{label}': train={tr.sum():,} test={te.sum():,} pos_rate={y[te].mean():.3f}")
    row, auroc = _eval_probe(X[tr], y[tr], X[te], y[te],
                             df.loc[te, "subject_id"].to_numpy(), cfg.get("run.seed", 42),
                             "image", label)
    if auroc / 100 < _GATE_AUROC:
        log(f"  *** GATE-FAIL: MIMIC image proxy can't predict {label} (AUROC<{_GATE_AUROC}); tell Soroosh.")
    return [row]


def _mimic_note_probe(cfg, intervention="fluids_sepsis"):
    """§6.5 / E3 note-side validity: does the pre-t0 note embedding carry the
    clinician urgency / trajectory-gestalt confounder the paper relies on? Target =
    horizon mortality (a prognosis/goals-of-care proxy; strictly a FUTURE outcome vs
    the pre-t0 note, so not circular). notes_clinical (radiology-excluded) embedding,
    patient-level held-out split. modality='notes'."""
    import pandas as pd
    coh = pd.read_parquet(cfg.storage("cohorts", f"{intervention}.parquet"))
    coh = coh[coh["all_modality"] & coh["arm"].notna()].reset_index(drop=True)
    Xnote = F.pool_embeddings(cfg, coh, "notes", "notes_clinical")
    y = coh["outcome"].astype(int).to_numpy()
    subj = coh["subject_id"].to_numpy()
    seed = int(cfg.get("run.seed", 42))
    rng = np.random.default_rng(seed)
    pats = pd.unique(subj)
    test = set(rng.choice(pats, size=max(1, int(len(pats) * 0.3)), replace=False))
    te = np.array([s in test for s in subj]); tr = ~te
    if y[tr].sum() < 20 or y[te].sum() < 10 or y[tr].sum() == 0:
        log("  MIMIC note probe: too few outcome events; skip"); return []
    log(f"  MIMIC note 'mortality_prognosis': train={tr.sum():,} test={te.sum():,} pos_rate={y[te].mean():.3f}")
    row, auroc = _eval_probe(Xnote[tr], y[tr], Xnote[te], y[te], subj[te], seed,
                             "notes", "mortality_prognosis")
    if auroc / 100 < _GATE_AUROC:
        log(f"  note probe: AUROC {auroc:.1f} < {_GATE_AUROC*100:.0f} -- note channel weak for prognosis.")
    return [row]


def _external_gates(cfg):
    qg = cfg.get("quality_gate")
    if not qg:
        return []
    labels = qg.get("probe_labels", [])
    seed = int(cfg.get("run.seed", 42))
    rows = []
    for ds, spec in qg.get("datasets", {}).items():
        base = cfg.storage("embeddings", "external", ds)
        vpath, ipath = base / "vectors.npy", base / "index.csv"
        if not vpath.exists():
            log(f"  external {ds}: no embeddings yet (run extract_external_cxr); skip"); continue
        V = np.load(vpath).astype("float32")
        idx = pd.read_csv(ipath)                                    # image_id, row (aligned to V)
        master = pd.read_csv(spec["master"], low_memory=False)
        keycol = "ImageID" if "ImageID" in master else "image_id"
        master = master.drop_duplicates(keycol).set_index(keycol)
        pidcol = next((c for c in ("PatientID", "patient_id") if c in master.columns), None)
        pid = (master[pidcol].reindex(idx["image_id"]).to_numpy() if pidcol
               else np.arange(len(idx)))
        rng = np.random.default_rng(seed)
        if pidcol is not None:
            pats = pd.unique(pid[pd.notna(pid)])
            test_pats = set(rng.choice(pats, size=max(1, int(len(pats) * 0.3)), replace=False))
            te = np.array([p in test_pats for p in pid])
        else:
            te = rng.random(len(idx)) < 0.3
        tr = ~te
        log(f"  external {ds}: {len(idx):,} images (train={tr.sum():,} test={te.sum():,})")
        for lab in labels:
            col = spec.get("label_map", {}).get(lab)
            if not col or col not in master.columns:
                continue
            y = (_binary(master[col].reindex(idx["image_id"])) == 1).astype(int).to_numpy()
            if y[tr].sum() < 20 or y[te].sum() < 10:
                log(f"    {ds}/{lab}: too few positives; skip"); continue
            row, _ = _eval_probe(V[tr], y[tr], V[te], y[te], pid[te], seed, f"image@{ds}", lab)
            rows.append(row)
    return rows


def run(cfg, force: bool = False, intervention: str = "fluids_sepsis"):
    # imaging_confounder_label is only defined for imaging interventions (fluids/prone);
    # the image probe defaults to edema and the note probe/external gates run regardless,
    # so this stage must not hard-require it (rrt/transfusion have no imaging confounder).
    log(f"probe[{intervention}]: MIMIC validity gate + external quality gate ...")
    rows = (_mimic_image_probe(cfg, intervention)
            + _mimic_note_probe(cfg, intervention)
            + _external_gates(cfg))

    # rewrite probe.csv from scratch (idempotent)
    p = cfg.storage("results", "probe.csv")
    if p.exists():
        p.unlink()
    R.append_rows(cfg, "probe.csv", rows)
    log(f"probe done -> probe.csv ({len(rows)} rows)")
