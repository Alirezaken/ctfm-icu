"""robustness -- does the conclusion survive the choices we had to make? Writes robustness.csv.

TWO FAMILIES.

  family = "swap"      Re-run the structured -> multimodal bias reduction under a different
                       encoder, estimator, look-back window, pooling rule, embedding
                       reduction, or trim threshold. The paper's claim is that the modality
                       adds nothing; that claim would be worthless if it depended on any one
                       of these knobs. Every swap carries the SAME IPCW censoring correction
                       as the primary analysis, so each comparison is like-for-like.

  family = "subgroup"  Re-estimate `structured` and `multimodal` within each sex and each
                       age band. A subgroup with too few events in an arm, or one that
                       produces a risk difference outside +/-100 pp (physically impossible),
                       is written as `undefined` rather than printed. Reporting an impossible
                       interval as if it were a finding is worse than reporting nothing.
"""
from __future__ import annotations

import re
import time
import numpy as np
import pandas as pd

from src.util import log
from src import events as ev
from src import features as F
from src import estimator as EST
from src import results as R
from src.stats import cluster_bootstrap_indices, bootstrap_summary


def _band(s):
    m = re.match(r"\(([\d.]+),\s*([\d.]+)\]", str(s))
    return (float(m.group(1)), float(m.group(2))) if m else None


def _bias_reduction(cfg, cohort, ref, est_name, blocks_fn, reduction, trim, boot):
    """|bias(structured)| - |bias(multimodal)| under one configuration."""
    A = (cohort["arm"] == "active").astype(int).to_numpy()
    Y = cohort["outcome"].astype(int).to_numpy()
    D = cohort["observed_at_horizon"].astype(int).to_numpy()
    S = F.structured_at_t0(cfg, cohort).to_numpy(dtype=float)
    blocks = blocks_fn()
    est = EST.get_estimator(est_name)

    out = {}
    for name, blks in [("structured", []), ("multimodal", blocks)]:
        psi, keep, _ = est(S, A, Y, cfg.folds, cfg.seed, blocks=blks,
                           reduction=reduction, pca_components=cfg.pca_components,
                           trim=trim, D=D)
        pt = float(psi[keep].mean() * 100)
        bt = np.array([psi[b][keep[b]].mean() * 100 for b in boot])
        out[name] = (pt, bt)

    (sp, sb), (fp, fb) = out["structured"], out["multimodal"]
    return bootstrap_summary(abs(sp - ref) - abs(fp - ref),
                             np.abs(sb - ref) - np.abs(fb - ref))


def run(cfg, force: bool = False, intervention: str = None):
    t0 = time.time()
    names = [intervention] if intervention else list(cfg.get("interventions") or {})
    sw = cfg.get("robustness_swaps") or {}
    alt_win = int(sw.get("look_back_window_alt_hours", 24))
    alt_pool = sw.get("pooling_alt", "max")
    alt_red = sw.get("reduction_alt", "pca")
    alt_trim = float(sw.get("trim_alt", 0.05))
    have_alt_enc = cfg.storage("embeddings", "images_alt.npy").exists()

    rows = []
    for iv in names:
        spec = cfg.get(f"interventions.{iv}")
        ref = float(spec["rct_reference"]["risk_difference"])
        cohort = ev.load_cohorts(cfg, iv)
        cohort = cohort[cohort["arm"].notna() & cohort["imaged"]].reset_index(drop=True)
        if cohort["arm"].nunique() < 2:
            continue
        boot = cluster_bootstrap_indices(cohort["subject_id"].to_numpy(), cfg.nboot, cfg.seed)
        log(f"=== robustness[{iv}] ===")

        def base_blocks(win=None, pool=None, img="images"):
            return lambda: [F.modality_block(cfg, cohort, img, win, pool),
                            F.modality_block(cfg, cohort, "radtext", win, pool),
                            F.modality_block(cfg, cohort, "histnote", win, pool)]

        swaps = [
            ("primary",   base_blocks(),                        "aipw", cfg.reduction, cfg.trim, ""),
            ("window",    base_blocks(win=alt_win),             "aipw", cfg.reduction, cfg.trim, f"{alt_win}h"),
            ("pooling",   base_blocks(pool=alt_pool),           "aipw", cfg.reduction, cfg.trim, alt_pool),
            ("estimator", base_blocks(),                        "tmle", cfg.reduction, cfg.trim, "tmle"),
            ("reduction", base_blocks(),                        "aipw", alt_red,       cfg.trim, alt_red),
            ("trim",      base_blocks(),                        "aipw", cfg.reduction, alt_trim, str(alt_trim)),
        ]
        if have_alt_enc:
            swaps.append(("encoder", base_blocks(img="images_alt"), "aipw",
                          cfg.reduction, cfg.trim, "biomedclip"))

        for swap, blocks_fn, est, red, trim, note in swaps:
            e = _bias_reduction(cfg, cohort, ref, est, blocks_fn, red, trim, boot)
            rows.append({"intervention": iv, "family": "swap", "swap": swap,
                         "condition": "structured_to_multimodal", "subgroup": "",
                         **e.as_row("value_"),
                         "support_count": len(cohort), "undefined": False, "note": note})
            log(f"  swap={swap:10s} bias reduction {e.point:+6.2f} "
                f"[{e.ci_low:+6.2f},{e.ci_high:+6.2f}]")
        if not have_alt_enc:
            log("  encoder swap skipped (no images_alt embeddings)")

        rows += _subgroups(cfg, iv, cohort, ref, boot)

    R.reset_rows(cfg, "robustness.csv", intervention=names)
    R.append_rows(cfg, "robustness.csv", rows)
    log(f"robustness done in {time.time()-t0:,.0f}s -> robustness.csv ({len(rows)} rows)")


def _subgroups(cfg, iv, cohort, ref, boot_all):
    rows = []
    min_ev = int(cfg.get("demographics.min_arm_events", 5))
    max_rd = float(cfg.get("demographics.max_abs_rd_pp", 100))

    A = (cohort["arm"] == "active").astype(int).to_numpy()
    Y = cohort["outcome"].astype(int).to_numpy()
    D = cohort["observed_at_horizon"].astype(int).to_numpy()
    S = F.structured_at_t0(cfg, cohort).to_numpy(dtype=float)
    blocks = [F.modality_block(cfg, cohort, m) for m in ("images", "radtext", "histnote")]

    sex = cohort["sex"].astype(str).to_numpy()
    age = pd.to_numeric(cohort["age_t0"], errors="coerce").to_numpy()
    groups = [("sex", s, sex == s) for s in cfg.get("demographics.sex_levels", ["F", "M"])]
    for b in cfg.get("demographics.age_bands", []):
        pb = _band(b)
        if pb:
            groups.append(("age_band", b, (age > pb[0]) & (age <= pb[1])))

    for gtype, gname, mask in groups:
        n = int(mask.sum())
        a, y = A[mask], Y[mask]
        ok = (n > 0 and len(np.unique(a)) == 2
              and y[a == 1].sum() >= min_ev and y[a == 0].sum() >= min_ev
              and (1 - y[a == 1]).sum() >= min_ev and (1 - y[a == 0]).sum() >= min_ev)
        if not ok:
            for cond in ("structured", "multimodal"):
                rows.append({"intervention": iv, "family": "subgroup", "swap": "",
                             "condition": cond, "subgroup": f"{gtype}:{gname}",
                             "support_count": n, "undefined": True,
                             "note": "too few events in an arm"})
            log(f"  subgroup {gtype}={gname}: n={n} -> undefined")
            continue

        boot = cluster_bootstrap_indices(cohort.loc[mask, "subject_id"].to_numpy(),
                                         cfg.nboot, cfg.seed)
        for cond, blks in [("structured", []),
                           ("multimodal", [b[mask] for b in blocks])]:
            try:
                psi, keep, _ = EST.crossfit_aipw(
                    S[mask], a, y, cfg.folds, cfg.seed, blocks=blks,
                    reduction=cfg.reduction, pca_components=cfg.pca_components,
                    trim=cfg.trim, D=D[mask])
                pt = float(psi[keep].mean() * 100)
                bt = np.array([psi[b][keep[b]].mean() * 100 for b in boot])
                e = bootstrap_summary(pt, bt)
            except Exception as exc:
                rows.append({"intervention": iv, "family": "subgroup", "swap": "",
                             "condition": cond, "subgroup": f"{gtype}:{gname}",
                             "support_count": n, "undefined": True,
                             "note": f"estimation failed: {exc}"})
                continue

            # a risk difference outside +/-100pp is not wide, it is impossible.
            # Report `undefined` rather than print a number that cannot exist.
            if (abs(e.point) > max_rd or abs(e.ci_low) > max_rd or abs(e.ci_high) > max_rd):
                rows.append({"intervention": iv, "family": "subgroup", "swap": "",
                             "condition": cond, "subgroup": f"{gtype}:{gname}",
                             "support_count": n, "undefined": True,
                             "note": f"estimate/CI exceeds +/-{max_rd:.0f}pp -- not estimable"})
                log(f"  subgroup {gtype}={gname}/{cond}: impossible CI -> undefined")
                continue

            rows.append({"intervention": iv, "family": "subgroup", "swap": "",
                         "condition": cond, "subgroup": f"{gtype}:{gname}",
                         **e.as_row("value_"),
                         "support_count": n, "undefined": False, "note": ""})
    return rows
