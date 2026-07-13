"""diagnose -- the paper's contribution, as a stage. Writes diagnostics.csv.

Runs the three-condition check for every (intervention x modality) cell:

  (1) INFORMATIVENESS   probe: does the embedding encode the target confounder at all?
                        image  -> labeled pulmonary edema (MIMIC-CXR CheXpert label)
                        radtext-> the same edema label (the radiologist's own words)
                        histnote-> mortality prognosis (no label exists for a history note)
                        PLUS an external replication of the image probe on PadChest,
                        ChestX-ray14 and CheXpert, which is what makes the informativeness
                        claim credible rather than a MIMIC artifact.

  (2) INCREMENTAL CONFOUNDING   dAUC_treat and dAUC_outcome, cross-fitted, conditional on
                        the structured set. Both must be positive for the modality to
                        carry residual confounding. ICI = min of the two.

  (3) POSITIVITY COST   1 - ESS(S,M)/ESS(S), read from what the estimator actually paid.

Then the pre-registered decision rule, and -- the point of the whole exercise -- the
PREDICTED bias reduction from the semi-synthetic calibration curve, placed next to the
OBSERVED bias reduction from the real data. If those two agree, the diagnostic works.

Order matters: run AFTER estimate (needs the ESS and the observed bias reductions) and
AFTER synthesize (needs the calibration curve). main.py enforces this.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd

from src.util import log
from src import events as ev
from src import features as F
from src import results as R
from src.diagnostic import incremental_confounding, positivity_cost, decision
from src.stats import ka_summary, cluster_bootstrap_indices

_MOD_FOR_COND = {"struct_img": "images", "struct_radtext": "radtext",
                 "struct_histnote": "histnote"}


# --------------------------------------------------------------------------- #
#  (1) informativeness probes                                                  #
# --------------------------------------------------------------------------- #
def _probe(V_tr, y_tr, V_te, y_te, subj_te, seed, n_boot=500):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, average_precision_score

    sc = StandardScaler().fit(V_tr)
    clf = LogisticRegression(max_iter=1000).fit(sc.transform(V_tr), y_tr)
    p = clf.predict_proba(sc.transform(V_te))[:, 1]

    aur, aup = [], []
    for b in cluster_bootstrap_indices(subj_te, n_boot, seed):
        yb, pb = y_te[b], p[b]
        if yb.min() == yb.max():
            continue
        aur.append(roc_auc_score(yb, pb) * 100)
        aup.append(average_precision_score(yb, pb) * 100)
    return ka_summary(aur), ka_summary(aup), float(roc_auc_score(y_te, p) * 100)


def _mimic_probe(cfg, cohort, modality, label):
    """Probe a modality's pooled patient-level proxy against a labeled confounder."""
    seed = cfg.seed
    X = F.modality_block(cfg, cohort, modality)

    if label == "mortality_prognosis":
        y = cohort["outcome"].astype(int).to_numpy()
    else:
        cxr = ev.link(cfg, "cxr_studies")[["subject_id", label]]
        cxr = cxr.dropna(subset=[label])
        # patient is positive if ANY pre-t0 study carried the label
        pos = set(cxr.loc[pd.to_numeric(cxr[label], errors="coerce") == 1, "subject_id"])
        seen = set(cxr["subject_id"])
        keep = cohort["subject_id"].isin(seen).to_numpy()
        if keep.sum() < 100:
            return None
        y = cohort["subject_id"].isin(pos).astype(int).to_numpy()
        X, y = X[keep], y[keep]
        cohort = cohort[keep].reset_index(drop=True)

    if len(np.unique(y)) < 2 or y.sum() < 20:
        log(f"    probe {modality}/{label}: too few positives; skip")
        return None

    rng = np.random.default_rng(seed)
    pats = cohort["subject_id"].unique()
    te_pats = set(rng.choice(pats, size=max(1, int(len(pats) * 0.3)), replace=False))
    te = cohort["subject_id"].isin(te_pats).to_numpy()
    tr = ~te
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        return None

    (rm, rs, rl, rh), (pm, ps, pl, ph), auroc = _probe(
        X[tr], y[tr], X[te], y[te], cohort.loc[te, "subject_id"].to_numpy(), seed,
        int(cfg.get("diagnostic.bootstrap_reps", 500)))
    log(f"    probe {modality}/{label}: AUROC={auroc:.1f}  (n_test={int(te.sum()):,})")
    return {"probe_auroc_mean": rm, "probe_auroc_std": rs,
            "probe_auroc_ci_low": rl, "probe_auroc_ci_high": rh,
            "probe_auprc_mean": pm, "probe_auprc_std": ps,
            "probe_auprc_ci_low": pl, "probe_auprc_ci_high": ph,
            "probe_target": label, "probe_n_test": int(te.sum())}


def _external_probes(cfg):
    """Replicate the IMAGE probe on external CXR datasets. This is what turns
    'the embedding is informative' from a MIMIC artifact into a real property of the
    encoder -- and it is the claim the whole paper's tension rests on, so it has to hold
    outside MIMIC or the headline ('informative yet useless') collapses."""
    rows = []
    qg = cfg.get("quality_gate") or {}
    labels = qg.get("probe_labels", [])
    ext = cfg.get("paths.external_cxr") or {}
    seed = cfg.seed

    for ds, spec in ext.items():
        if not isinstance(spec, dict):
            continue
        vpath = cfg.storage("embeddings", "external", f"{ds}.npy")
        ipath = cfg.storage("embeddings", "external", f"{ds}_index.csv")
        if not vpath.exists():
            log(f"    external {ds}: no embeddings (run extract_external); skip")
            continue
        V = np.load(vpath).astype("float32")
        idx = pd.read_csv(ipath)
        master = pd.read_csv(spec["csv"], low_memory=False)
        key = spec["id_col"]
        master = master.drop_duplicates(key).set_index(key)
        pidcol = next((c for c in ("PatientID", "patient_id", "subject_id")
                       if c in master.columns), None)
        pid = (master[pidcol].reindex(idx["image_id"]).to_numpy()
               if pidcol else np.arange(len(idx)))

        rng = np.random.default_rng(seed)
        if pidcol:
            pats = pd.unique(pid[pd.notna(pid)])
            te_p = set(rng.choice(pats, size=max(1, int(len(pats) * 0.3)), replace=False))
            te = np.array([p in te_p for p in pid])
        else:
            te = rng.random(len(idx)) < 0.3
        tr = ~te

        for lab in labels:
            col = spec.get("label_map", {}).get(lab)
            if not col or col not in master.columns:
                continue
            y = (pd.to_numeric(master[col].reindex(idx["image_id"]), errors="coerce") == 1)
            y = y.astype(int).to_numpy()
            if y[tr].sum() < 20 or y[te].sum() < 10:
                continue
            (rm, rs, rl, rh), (pm, ps, pl, ph), auroc = _probe(
                V[tr], y[tr], V[te], y[te], pid[te], seed,
                int(cfg.get("diagnostic.bootstrap_reps", 500)))
            log(f"    external {ds}/{lab}: AUROC={auroc:.1f}")
            rows.append({
                "intervention": "", "modality": f"images@{ds}",
                "check": "informativeness_external",
                "probe_auroc_mean": rm, "probe_auroc_std": rs,
                "probe_auroc_ci_low": rl, "probe_auroc_ci_high": rh,
                "probe_auprc_mean": pm, "probe_auprc_std": ps,
                "probe_auprc_ci_low": pl, "probe_auprc_ci_high": ph,
                "probe_target": lab, "probe_n_test": int(te.sum()),
            })
    return rows


# --------------------------------------------------------------------------- #
#  calibration curve from the synthetic sweep                                  #
# --------------------------------------------------------------------------- #
def _calibration(cfg):
    """Fit bias_reduction ~ f(ICI) on the synthetic sweep so we can PREDICT what a real
    modality will deliver from its measured ICI alone. An isotonic fit, because the
    relationship is monotone by theory but not linear."""
    p = cfg.storage("results", "synthetic.csv")
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df = df[(df["sweep"] == "gamma_delta") & df["ici"].notna()
            & df["bias_reduction"].notna()]
    if len(df) < 8:
        return None
    from sklearn.isotonic import IsotonicRegression
    g = df.groupby("ici", as_index=False)["bias_reduction"].mean().sort_values("ici")
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(
        g["ici"].to_numpy(), g["bias_reduction"].to_numpy())
    log(f"  calibration curve fitted on {len(g)} synthetic ICI levels")
    return iso


# --------------------------------------------------------------------------- #
def run(cfg, force: bool = False, intervention: str = None):
    t0 = time.time()
    names = [intervention] if intervention else list(cfg.get("interventions") or {})
    iso = _calibration(cfg)
    rows = []

    for iv in names:
        spec = cfg.get(f"interventions.{iv}")
        log(f"=== diagnose[{iv}] ===")
        cohort = ev.load_cohorts(cfg, iv)
        cohort = cohort[cohort["arm"].notna() & cohort["imaged"]].reset_index(drop=True)
        A = (cohort["arm"] == "active").astype(int).to_numpy()
        Y = cohort["outcome"].astype(int).to_numpy()
        S = F.structured_at_t0(cfg, cohort).to_numpy(dtype=float)

        # what the estimator actually paid, and actually delivered
        bp = cfg.storage("results", f"_boot_{iv}.npz")
        ess, observed = {}, {}
        if bp.exists():
            b = np.load(bp, allow_pickle=True)
            conds = [str(c) for c in b["conditions"]]
            ess = dict(zip(conds, [float(x) for x in b["ess"]]))
            ref = float(b["ref_rd"])
            pts = dict(zip(conds, [float(x) for x in b["point"]]))
            if "structured" in pts:
                b0 = abs(pts["structured"] - ref)
                observed = {c: round(b0 - abs(pts[c] - ref), 2)
                            for c in conds if c in _MOD_FOR_COND}

        img_label = spec.get("imaging_confounder_label", "edema")
        probe_target = {"images": img_label, "radtext": img_label,
                        "histnote": "mortality_prognosis"}

        for cond, mod in _MOD_FOR_COND.items():
            block = F.modality_block(cfg, cohort, mod)

            pr = _mimic_probe(cfg, cohort, mod, probe_target[mod]) or {}
            ic = incremental_confounding(S, A, Y, block, cfg.folds, cfg.seed,
                                         cfg.reduction, cfg.pca_components) or {}

            pc = positivity_cost(ess.get("structured"), ess.get(cond))
            dec = decision(ic.get("d_auc_treat"), ic.get("d_auc_outcome"), pc,
                           float(cfg.get("diagnostic.auc_threshold", 0.02)),
                           float(cfg.get("diagnostic.positivity_cost_max", 0.25)))

            pred = ""
            if iso is not None and ic.get("ici") is not None:
                pred = round(float(iso.predict([ic["ici"]])[0]), 2)

            rows.append({
                "intervention": iv, "modality": mod, "check": "full",
                **pr, **ic,
                "ess_structured": round(ess.get("structured", 0)) or "",
                "ess_with_modality": round(ess.get(cond, 0)) or "",
                "positivity_cost": pc,
                "add_modality": dec["add_modality"], "reason": dec["reason"],
                "observed_bias_reduction_pp": observed.get(cond, ""),
                "predicted_bias_reduction_pp": pred,
            })
            log(f"  {mod:9s} dAUC_A={ic.get('d_auc_treat')}  "
                f"dAUC_Y={ic.get('d_auc_outcome')}  ICI={ic.get('ici')}  "
                f"posCost={pc}  -> add={dec['add_modality']} ({dec['reason']})")

    rows += _external_probes(cfg)

    R.reset_rows(cfg, "diagnostics.csv", intervention=names + [""])
    R.append_rows(cfg, "diagnostics.csv", rows)
    log(f"diagnose done in {time.time()-t0:,.0f}s -> diagnostics.csv ({len(rows)} rows)")
