"""synthesize -- the semi-synthetic benchmark. Writes synthetic.csv.

This is the stage that makes the paper's null defensible. Everything is real except the
treatment and the outcome, and those are simulated with a KNOWN causal effect.

  real:      the fluids_sepsis imaged cohort, its structured covariates, its RAD-DINO
             image embeddings
  real:      U = cross-fitted P(edema | image embedding). The actual validated signal the
             image carries -- the same one that scores AUROC ~90 in the probe.
  derived:   U_perp = U - proj(U | S). The part of the image signal that is NOT already in
             the structured record. Only this part can do any confounding work.
  simulated: A and Y, with tau calibrated to a known risk difference.

TWO SWEEPS

  1. gamma x delta (5 x 5 x 20 replicates)
     gamma = how hard the image-borne confounder pushes TREATMENT
     delta = how hard it pushes the OUTCOME
     Expected and checked: `structured` is biased whenever gamma>0 AND delta>0, and
     `struct_img` recovers tau. This is the POSITIVE CONTROL. If it fails, the estimator
     or the embeddings are broken and no null on real data means anything.
     The (ICI, bias_reduction) pairs from these 25 cells are the CALIBRATION CURVE.

  2. redundancy (5 levels)
     Hold gamma and delta FIXED at a level where confounding is strong, and vary only how
     much of the confounder is already predictable from S. Bias reduction must fall to
     zero as redundancy -> 1, even though the confounder is exactly as strong throughout.
     This is the mechanism experiment. It shows the failure is REDUNDANCY, not weakness --
     which is precisely the regime the four real interventions turn out to live in.

The payoff: overlay the real interventions' measured ICI on the calibration curve. They
sit at ICI ~ 0, where the curve says bias reduction should be ~0, and where the real data
says it is. The theory predicts the null.

PARALLELISM AND CHECKPOINTING. Each of the 30 cells (25 gamma/delta + 5 redundancy) needs
20 replicates, each an independent cross-fitted estimation -- expensive (several nested
cross-fitted model fits apiece) and embarrassingly parallel across replicates. The 20
replicates of a cell run concurrently via joblib, sized to the machine's CPU count so as
not to oversubscribe past what each LightGBM fit already uses internally (estimator._LGBM
n_jobs). Each cell's rows are written to synthetic.csv and checkpointed as soon as that
cell finishes, so a preempted or timed-out job resumes instead of restarting: this stage
used to hold everything in memory and write once at the very end, which meant a SLURM
timeout lost the entire sweep.
"""
from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.util import log, Checkpoint
from src import events as ev
from src import features as F
from src import estimator as EST
from src import results as R
from src.diagnostic import incremental_confounding
from src.synthetic import image_confounder, residualize, simulate, make_redundant


def _edema_label(cfg, cohort):
    """Patient-level pre-t0 edema label from the MIMIC-CXR CheXpert annotations."""
    cxr = ev.link(cfg, "cxr_studies")[["subject_id", "edema"]]
    cxr = cxr.dropna(subset=["edema"])
    pos = set(cxr.loc[pd.to_numeric(cxr["edema"], errors="coerce") == 1, "subject_id"])
    return cohort["subject_id"].isin(pos).astype(int).to_numpy()


def _one_cell(S, Vimg, U_perp, gamma, delta, target_rd, rng, cfg, sct, sco):
    """Simulate one (gamma, delta) cell and estimate under `structured` and `struct_img`."""
    A, Y, true_rd = simulate(S, U_perp, gamma, delta, target_rd, rng,
                             s_coef_treat=sct, s_coef_out=sco)
    if len(np.unique(A)) < 2 or len(np.unique(Y)) < 2:
        return None

    psi_s, keep_s, dg_s = EST.crossfit_aipw(
        S, A, Y, cfg.folds, cfg.seed, trim=cfg.trim)
    psi_i, keep_i, dg_i = EST.crossfit_aipw(
        S, A, Y, cfg.folds, cfg.seed, blocks=[Vimg], reduction=cfg.reduction,
        pca_components=cfg.pca_components, trim=cfg.trim)

    rd_s = float(psi_s[keep_s].mean() * 100)
    rd_i = float(psi_i[keep_i].mean() * 100)
    bias_s = rd_s - true_rd
    bias_i = rd_i - true_rd

    ic = incremental_confounding(S, A, Y, Vimg, cfg.folds, cfg.seed,
                                 cfg.reduction, cfg.pca_components) or {}
    pc = (1.0 - dg_i["ess"] / dg_s["ess"]) if dg_s["ess"] > 0 else None

    return {
        "true_rd_pp": round(true_rd, 2), "n": int(len(A)),
        "rd_structured": round(rd_s, 2), "rd_struct_img": round(rd_i, 2),
        "bias_structured": round(bias_s, 2), "bias_struct_img": round(bias_i, 2),
        # positive = adding the image moved the estimate CLOSER to the known truth
        "bias_reduction": round(abs(bias_s) - abs(bias_i), 2),
        "d_auc_treat": ic.get("d_auc_treat"), "d_auc_outcome": ic.get("d_auc_outcome"),
        "ici": ic.get("ici"),
        "ess_structured": round(dg_s["ess"]), "ess_struct_img": round(dg_i["ess"]),
        "positivity_cost": round(pc, 3) if pc is not None else None,
    }


def _n_workers() -> int:
    """How many replicates to run concurrently. Each LightGBM fit inside a replicate
    already uses estimator._LGBM['n_jobs'] threads; dividing the allocated CPU count by
    that avoids oversubscribing past what SLURM actually gave this job."""
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK")
              or os.environ.get("SLURM_JOB_CPUS_PER_NODE")
              or os.cpu_count() or 4)
    per_fit = int(EST._LGBM.get("n_jobs", 4))
    return max(1, cpus // per_fit)


def run(cfg, force: bool = False, intervention: str = None):
    t0 = time.time()
    scfg = cfg.get("synthetic")
    base = scfg["base_intervention"]
    target_rd = float(scfg["target_rd_pp"])
    reps = int(scfg["replicates"])
    n_workers = _n_workers()

    log(f"=== synthesize (semi-synthetic benchmark on the real {base} cohort) ===")
    log(f"  running {reps} replicates/cell across {n_workers} parallel workers "
        f"({EST._LGBM.get('n_jobs', 4)} threads/fit)")

    cohort = ev.load_cohorts(cfg, base)
    cohort = cohort[cohort["arm"].notna() & cohort["imaged"]].reset_index(drop=True)
    S = F.structured_at_t0(cfg, cohort).to_numpy(dtype=float)
    S = np.nan_to_num(S, nan=0.0)
    Vimg = F.modality_block(cfg, cohort, "images")

    # the REAL image-borne confounder
    edema = _edema_label(cfg, cohort)
    U = image_confounder(Vimg, edema, cfg.folds, cfg.seed)
    U_perp, r2 = residualize(U, S)
    log(f"  U = P(edema | image), n={len(U):,}, edema prevalence={edema.mean():.3f}")
    log(f"  R^2 of U on the structured set = {r2:.3f}  "
        f"-> {100*(1-r2):.0f}% of the image signal is INCREMENTAL")

    rng = np.random.default_rng(cfg.seed)
    # fixed structured coefficients across the whole sweep, so only gamma/delta vary
    sct = rng.normal(0, 0.3, S.shape[1])
    sco = rng.normal(0, 0.3, S.shape[1])

    ckpt = Checkpoint(cfg, "synthesize")
    resuming = any(ckpt.dir.glob("*.json")) and not force
    if force:
        ckpt.clear()
    if not resuming:
        R.reset_rows(cfg, "synthetic.csv", sweep=["gamma_delta", "redundancy"])
    log(f"  {'resuming from checkpointed cells' if resuming else 'starting fresh'}")

    def _run_cell(key, cell_meta, cell_tasks):
        """Run one cell's replicates in parallel; append + checkpoint immediately."""
        if ckpt.done(key) and not force:
            log(f"  {key}: cached, skipping")
            return None
        results = Parallel(n_jobs=n_workers, prefer="processes")(cell_tasks)
        cell_rows = []
        for rep, r in enumerate(results):
            if r is None:
                continue
            cell_rows.append({**cell_meta, "replicate": rep,
                              "confounder_r2_on_structured": round(r2, 3), **r})
        if cell_rows:
            R.append_rows(cfg, "synthetic.csv", cell_rows)
        ckpt.mark(key, n=len(cell_rows))
        return cell_rows

    # ---------------- sweep 1: gamma x delta (the calibration curve) ----------
    for gamma in scfg["gamma_grid"]:
        for delta in scfg["delta_grid"]:
            key = f"gamma_delta_g{gamma}_d{delta}"
            tasks = (delayed(_one_cell)(S, Vimg, U_perp, float(gamma), float(delta),
                                        target_rd, np.random.default_rng(cfg.seed + 7919 * rep),
                                        cfg, sct, sco)
                    for rep in range(reps))
            done = _run_cell(key, {"sweep": "gamma_delta", "gamma": gamma,
                                   "delta": delta, "redundancy": ""}, tasks)
            if done:
                log(f"  gamma={gamma} delta={delta}: "
                    f"bias(struct)={np.mean([d['bias_structured'] for d in done]):+6.2f}  "
                    f"bias(+img)={np.mean([d['bias_struct_img'] for d in done]):+6.2f}  "
                    f"reduction={np.mean([d['bias_reduction'] for d in done]):+6.2f}  "
                    f"ICI={np.mean([d['ici'] for d in done if d['ici'] is not None]):.2f}")

    # ---------------- sweep 2: redundancy (the mechanism) ---------------------
    g = float(scfg["redundancy_gamma"])
    d = float(scfg["redundancy_delta"])
    log(f"  --- redundancy sweep at gamma={g}, delta={d} (confounding strength FIXED) ---")
    for rho in scfg["redundancy_grid"]:
        key = f"redundancy_r{rho}"

        def _task(rep, rho=rho):
            rr = np.random.default_rng(cfg.seed + 104729 * rep)
            U_mix = make_redundant(U_perp, S, float(rho), rr)
            return _one_cell(S, Vimg, U_mix, g, d, target_rd, rr, cfg, sct, sco)

        tasks = (delayed(_task)(rep) for rep in range(reps))
        done = _run_cell(key, {"sweep": "redundancy", "gamma": g, "delta": d,
                               "redundancy": rho}, tasks)
        if done:
            log(f"  redundancy={rho}: "
                f"reduction={np.mean([x['bias_reduction'] for x in done]):+6.2f}  "
                f"ICI={np.mean([x['ici'] for x in done if x['ici'] is not None]):.2f}")

    # ---- the positive control, checked here and asserted in integrity.py ----
    # Re-read from disk rather than relying on in-memory rows, so this is correct
    # whether this run computed everything fresh or resumed from checkpoints.
    all_rows = pd.read_csv(cfg.storage("results", "synthetic.csv"))
    pc = cfg.get("positive_controls.synthetic_image_recovers_tau") or {}
    ga, de = float(pc.get("at_gamma", 1.5)), float(pc.get("at_delta", 1.5))
    cell = all_rows[(all_rows["sweep"] == "gamma_delta") & (all_rows["gamma"] == ga)
                    & (all_rows["delta"] == de)]
    if len(cell):
        got = float(cell["bias_reduction"].mean())
        need = float(pc.get("min_bias_reduction_pp", 2.0))
        verdict = "PASS" if got >= need else "*** FAIL ***"
        log(f"  POSITIVE CONTROL (gamma={ga}, delta={de}): bias reduction "
            f"{got:+.2f} pp (need >= {need}) -> {verdict}")
        if got < need:
            log("  *** The image channel FAILED to recover a confounder that is in it BY "
                "CONSTRUCTION. Do not interpret any real-data null until this passes: it "
                "means the estimator, the reduction, or the embeddings are broken.")

    log(f"synthesize done in {time.time()-t0:,.0f}s -> synthetic.csv ({len(all_rows)} rows)")
