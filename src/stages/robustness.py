"""§9.11 / §7.8  robustness -- recompute the structured->full bias reduction per
intervention under four swaps, to check the modality conclusion is stable:
  - window:    alternate look-back (robustness_swaps.look_back_window_alt_hours).
  - pooling:   max instead of mean (robustness_swaps.pooling_alt).
  - estimator: TMLE in place of AIPW.
  - encoder:   a second frozen CXR image encoder (robustness_swaps.encoder_alt) --
               needs its embeddings under embeddings/images_alt/; skipped with a
               note until that GPU run exists.
-> robustness.csv (Kind B: structured-to-full bias reduction).

Runs under the active variant (embedding reduction + results dir), covering every
intervention that has a cohort. Bias reduction = |bias_structured| - |bias_full|.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd

from src.util import log
from src import features, aipw, reduce, results as R
from src.stats import cluster_bootstrap_indices, bootstrap_summary


def _bias_reduction(cfg, cohort, ref, seed, folds, nboot, pool_kwargs, estimator,
                    image_source="images"):
    A = (cohort["arm"] == "active").to_numpy().astype(int)
    Y = cohort["outcome"].to_numpy().astype(int)
    S = features.structured_at_t0(cfg, cohort).to_numpy(dtype=float)
    Ximg = features.pool_embeddings(cfg, cohort, image_source, **pool_kwargs)
    Xnote = features.pool_embeddings(cfg, cohort, "notes", "notes_all", **pool_kwargs)
    if cfg.reduction != "none":
        Ximg = reduce.apply(Ximg, cfg.reduction, A, Y, folds, seed, cfg.pca_components)
        Xnote = reduce.apply(Xnote, cfg.reduction, A, Y, folds, seed, cfg.pca_components)
    est = aipw.crossfit_tmle if estimator == "tmle" else aipw.crossfit_aipw
    boot = list(cluster_bootstrap_indices(cohort["subject_id"].to_numpy(), nboot, seed))
    out = {}
    for name, Xc in [("structured", S), ("full", np.hstack([S, Xnote, Ximg]))]:
        psi, keep, _ = est(Xc, A, Y, folds, seed)
        pt = float(psi[keep].mean() * 100)
        bt = np.array([psi[b][keep[b]].mean() * 100 for b in boot])
        out[name] = (pt, bt)
    (sp, sb), (fp, fb) = out["structured"], out["full"]
    return bootstrap_summary(abs(sp - ref) - abs(fp - ref),
                             np.abs(sb - ref) - np.abs(fb - ref))


def run(cfg, force: bool = False):
    seed = int(cfg.get("run.seed", 42))
    folds = int(cfg.get("estimator.cross_fitting_folds", 5))
    nboot = int(cfg.get("bootstrap.n_resamples", 10000))
    alt_win = int(cfg.get("robustness_swaps.look_back_window_alt_hours", 24))
    alt_pool = cfg.get("robustness_swaps.pooling_alt", "max")
    have_alt_encoder = (cfg.storage("embeddings", "images_alt") / "index.csv").exists()

    swaps = [
        ("window", {"lookback_h": alt_win}, "aipw", "images"),
        ("pooling", {"pooling": alt_pool}, "aipw", "images"),
        ("estimator", {}, "tmle", "images"),
    ]
    if have_alt_encoder:
        swaps.append(("encoder", {}, "aipw", "images_alt"))   # BiomedCLIP

    cohorts_dir = cfg.storage("cohorts")
    rows = []
    for pq in sorted(cohorts_dir.glob("*.parquet")):
        iv = pq.stem
        ref = cfg.get(f"interventions.{iv}.rct_reference")
        if not ref or ref.get("risk_difference") is None:
            continue
        ref_rd = ref["risk_difference"]
        cohort = pd.read_parquet(pq)
        cohort = cohort[cohort["all_modality"] & cohort["arm"].notna()].reset_index(drop=True)
        if cohort["arm"].nunique() < 2:
            continue
        t0 = time.time()
        for swap, pool_kwargs, estimator, image_source in swaps:
            est = _bias_reduction(cfg, cohort, ref_rd, seed, folds, nboot,
                                  pool_kwargs, estimator, image_source)
            rows.append({"intervention": iv, "swap": swap,
                         **est.as_row("structured_to_full_bias_reduction_")})
        if not have_alt_encoder:
            log(f"  {iv}: encoder swap skipped (no images_alt embeddings yet; run BiomedCLIP)")
        log(f"  {iv}: {len(swaps)} swaps done in {time.time()-t0:,.0f}s")

    p = cfg.storage("results", "robustness.csv")
    if p.exists():
        p.unlink()
    R.append_rows(cfg, "robustness.csv", rows)
    log(f"robustness done -> robustness.csv ({len(rows)} rows; encoder swap "
        f"{'included' if have_alt_encoder else 'pending BiomedCLIP'}).")
