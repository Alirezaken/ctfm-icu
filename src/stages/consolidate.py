"""consolidate -- every PAIRED contrast, with p-values and FDR. Writes contrasts.csv + manifest.csv.

contrasts.csv is the ONLY file in the study with p-values. Five families, all computed
from the SHARED cluster-bootstrap indices saved by `estimate`, so every contrast is PAIRED:
the same patients, resampled the same way, under both conditions.

Pairing is not a technicality here, it is what makes the paper's null decisive. The
marginal CI on any single condition's risk difference is wide (the cohorts are a few
thousand patients with poor overlap). But the CI on the DIFFERENCE between two conditions
estimated on the SAME patients is far narrower, because the shared sampling noise cancels.
That is what lets us say "we can rule out a bias reduction larger than X pp" instead of
the much weaker "we failed to find one".

  1. bias_reduction_{modality}   |bias(structured)| - |bias(struct_<modality>)|
                                 THE PRIMARY OUTCOME. Positive = the modality moved the
                                 estimate closer to the RCT truth.
  2. effect_vs_reference         does this condition's effect differ from the trial?
  3. multimodal_vs_expert        does the kitchen-sink multimodal set beat the
                                 clinician-curated one? (the design-based competitor)
  4. marginal/complementary      modality decomposition: what does each modality add on its
                                 own, and what does it add ON TOP OF the other two?
  5. subgroup_diff               do subgroup effects differ from each other?

Every contrast carries its minimum detectable effect, so a null row states what it could
have found, not merely that it found nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import results as R
from src.stats import (bootstrap_summary, benjamini_hochberg, bootstrap_p_two_sided,
                       minimum_detectable_bias_reduction)
from src.util import log

_MOD = {"struct_img": "images", "struct_radtext": "radtext", "struct_histnote": "histnote"}


def _absbias(x, ref):
    return np.abs(np.asarray(x, float) - ref)


def run(cfg, force: bool = False, intervention: str = None):
    R.write_templates(cfg, force=force)
    R.build_manifest(cfg)

    rdir = cfg.storage("results")
    rows = []

    for bf in sorted(rdir.glob("_boot_*.npz")):
        iv = bf.name[len("_boot_"):-len(".npz")]
        b = np.load(bf, allow_pickle=True)
        conds = [str(c) for c in b["conditions"]]
        ref = float(b["ref_rd"])
        pt = {c: float(p) for c, p in zip(conds, b["point"])}
        bt = {c: np.asarray(b[c], float) for c in conds}

        if "structured" not in bt:
            continue
        base_pt = abs(pt["structured"] - ref)
        base_bt = _absbias(bt["structured"], ref)

        # ---- 1. bias reduction per modality (THE PRIMARY OUTCOME) ----
        for cond, mod in _MOD.items():
            if cond not in bt:
                continue
            red_pt = base_pt - abs(pt[cond] - ref)
            red_bt = base_bt - _absbias(bt[cond], ref)
            e = bootstrap_summary(red_pt, red_bt)
            rows.append({
                "contrast": "bias_reduction", "intervention": iv, "modality": mod,
                "stratum": "", **e.as_row("value_"),
                "min_detectable_pp": minimum_detectable_bias_reduction(red_bt),
                "test": "paired_cluster_bootstrap",
                "p_raw": bootstrap_p_two_sided(red_bt), "p_fdr": None})

        if "multimodal" in bt:
            red_pt = base_pt - abs(pt["multimodal"] - ref)
            red_bt = base_bt - _absbias(bt["multimodal"], ref)
            e = bootstrap_summary(red_pt, red_bt)
            rows.append({
                "contrast": "bias_reduction", "intervention": iv, "modality": "multimodal",
                "stratum": "", **e.as_row("value_"),
                "min_detectable_pp": minimum_detectable_bias_reduction(red_bt),
                "test": "paired_cluster_bootstrap",
                "p_raw": bootstrap_p_two_sided(red_bt), "p_fdr": None})

        # ---- 2. each condition's effect vs the RCT reference ----
        for c in conds:
            d = bt[c] - ref
            e = bootstrap_summary(pt[c] - ref, d)
            rows.append({
                "contrast": "effect_vs_reference", "intervention": iv, "modality": c,
                "stratum": "", **e.as_row("value_"),
                "min_detectable_pp": minimum_detectable_bias_reduction(d),
                "test": "paired_cluster_bootstrap",
                "p_raw": bootstrap_p_two_sided(d), "p_fdr": None})

        # ---- 3. multimodal vs the expert (design-based) competitor ----
        if "multimodal" in bt and "expert" in bt:
            d_pt = abs(pt["expert"] - ref) - abs(pt["multimodal"] - ref)
            d_bt = _absbias(bt["expert"], ref) - _absbias(bt["multimodal"], ref)
            e = bootstrap_summary(d_pt, d_bt)
            rows.append({
                "contrast": "multimodal_vs_expert", "intervention": iv,
                "modality": "multimodal", "stratum": "", **e.as_row("value_"),
                "min_detectable_pp": minimum_detectable_bias_reduction(d_bt),
                "test": "paired_cluster_bootstrap",
                "p_raw": bootstrap_p_two_sided(d_bt), "p_fdr": None})

        # ---- 4. decomposition: what each modality adds ON TOP OF the others ----
        if "multimodal" in bt:
            for cond, mod in _MOD.items():
                if cond not in bt:
                    continue
                others = [c for c in _MOD if c != cond and c in bt]
                if not others:
                    continue
                # complementary = what `mod` adds once the OTHER modalities are already in.
                # Approximated by (best other single) -> multimodal. If the modality is
                # redundant with the others, this is ~0 even when its marginal effect is not.
                best_other = min(others, key=lambda c: abs(pt[c] - ref))
                comp_pt = abs(pt[best_other] - ref) - abs(pt["multimodal"] - ref)
                comp_bt = _absbias(bt[best_other], ref) - _absbias(bt["multimodal"], ref)
                e = bootstrap_summary(comp_pt, comp_bt)
                rows.append({
                    "contrast": "complementary_bias_reduction", "intervention": iv,
                    "modality": mod, "stratum": "", **e.as_row("value_"),
                    "min_detectable_pp": minimum_detectable_bias_reduction(comp_bt),
                    "test": "paired_cluster_bootstrap",
                    "p_raw": bootstrap_p_two_sided(comp_bt), "p_fdr": None})

    # ---- 5. subgroup differences, read back from robustness.csv ----
    rows += _subgroup_contrasts(cfg)

    if rows:
        p = benjamini_hochberg([r["p_raw"] for r in rows])
        for r, pf in zip(rows, p):
            r["p_fdr"] = float(pf)

    R.reset_rows(cfg, "contrasts.csv", contrast=[
        "bias_reduction", "effect_vs_reference", "multimodal_vs_expert",
        "complementary_bias_reduction", "subgroup_diff"])
    R.append_rows(cfg, "contrasts.csv", rows)
    log(f"consolidate done -> contrasts.csv ({len(rows)} contrasts), manifest.csv")


def _subgroup_contrasts(cfg):
    """Pairwise subgroup differences within each (intervention, condition, subgroup_type).

    Read from robustness.csv rather than recomputed, so the numbers cannot drift between
    the two files. `undefined` subgroups are skipped, not silently treated as zero.
    """
    p = cfg.storage("results", "robustness.csv")
    if not p.exists():
        return []
    df = pd.read_csv(p)
    df = df[(df["family"] == "subgroup") & (df["undefined"] != True)]  # noqa: E712
    if not len(df):
        return []

    rows = []
    df = df.copy()
    df["stype"] = df["subgroup"].astype(str).str.split(":").str[0]
    df["sname"] = df["subgroup"].astype(str).str.split(":").str[1]

    for (iv, cond, stype), g in df.groupby(["intervention", "condition", "stype"]):
        g = g.dropna(subset=["value_point", "value_std"])
        if len(g) < 2:
            continue
        # compare each subgroup against the first, on a normal approximation from the
        # bootstrap SDs (the subgroups are disjoint patient sets, so the estimates are
        # independent and their variances add)
        base = g.iloc[0]
        for _, r in g.iloc[1:].iterrows():
            diff = float(r["value_point"]) - float(base["value_point"])
            se = float(np.sqrt(float(r["value_std"]) ** 2 + float(base["value_std"]) ** 2))
            if se <= 0:
                continue
            from scipy.stats import norm
            pval = float(2 * (1 - norm.cdf(abs(diff / se))))
            rows.append({
                "contrast": "subgroup_diff", "intervention": iv, "modality": cond,
                "stratum": f"{stype}:{r['sname']}-{base['sname']}",
                "value_point": round(diff, 1), "value_mean": round(diff, 1),
                "value_std": round(se, 1),
                "value_ci_low": round(diff - 1.96 * se, 1),
                "value_ci_high": round(diff + 1.96 * se, 1),
                "min_detectable_pp": round(2.8 * se, 2),
                "test": "normal_approx_on_bootstrap_sd",
                "p_raw": pval, "p_fdr": None})
    return rows
