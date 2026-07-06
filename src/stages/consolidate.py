"""§9.13 / §7  consolidate -- assemble the cross-stage result files.

Writes the 10 exact-schema templates + manifest, then computes the tables derived
from the per-condition estimates + their paired bootstrap replicates (saved by
`estimate` as results/_boot_<intervention>.npz):

  - dissociation.csv  (§7.2) specificity grid: bias reduction vs `structured` from
    each added modality.
  - decomposition.csv (§7.3) modality decomposition: marginal + complementary
    (non-overlapping) bias reduction per modality (notes_variant=clinical; notes_all
    needs a second estimate run).
  - comparisons.csv   (§7.9) the ONLY p-values file: key contrasts (effect vs RCT
    reference, and each modality's bias reduction) via the cluster bootstrap, with
    Benjamini-Hochberg FDR across the family.

Reads/writes the active variant's results dir, so re-running under --variant <fix>
reproduces every table under that method. Bias = effect - RCT reference (pp);
"bias reduction" = |bias_structured| - |bias_condition| (positive = closer to ref).
"""
from __future__ import annotations

import numpy as np

from src import results as R
from src.stats import bootstrap_summary, benjamini_hochberg
from src.util import log

_MOD = {"plus_notes": "notes", "plus_imaging_only": "imaging", "full": "full"}


def _p_two_sided(boot_diff):
    """Bootstrap two-sided p-value for H0: quantity = 0."""
    b = np.asarray(boot_diff, float)
    b = b[np.isfinite(b)]
    n = max(len(b), 1)
    p = 2.0 * min((b >= 0).mean(), (b <= 0).mean())
    return float(min(max(p, 1.0 / n), 1.0))


def _load_boot(path):
    b = np.load(path, allow_pickle=True)
    conds = [str(c) for c in b["conditions"]]
    return {"ref": float(b["ref_rd"]),
            "pt": {c: float(p) for c, p in zip(conds, b["point"])},
            "bt": {c: b[c] for c in conds}, "conds": conds}


def _absbias(x, ref):
    return np.abs(np.asarray(x) - ref)


def run(cfg, force: bool = False):
    R.write_templates(cfg, force=force)
    R.build_manifest(cfg)

    results_dir = cfg.storage("results")
    boot_files = sorted(results_dir.glob("_boot_*.npz"))
    diss, decomp, comp = [], [], []

    for bf in boot_files:
        iv = bf.name[len("_boot_"):-len(".npz")]
        d = _load_boot(bf)
        ref, pt, bt = d["ref"], d["pt"], d["bt"]
        if "structured" not in bt:
            continue
        bias0_pt = abs(pt["structured"] - ref)
        bias0_bt = _absbias(bt["structured"], ref)

        # --- dissociation: bias reduction vs structured, per added modality ---
        for cond, mod in _MOD.items():
            if cond not in bt:
                continue
            red_pt = bias0_pt - abs(pt[cond] - ref)
            red_bt = bias0_bt - _absbias(bt[cond], ref)
            diss.append({"intervention": iv, "modality": mod,
                         **bootstrap_summary(red_pt, red_bt).as_row("bias_reduction_vs_structured_")})
            comp.append({"comparison": f"bias_reduction_{mod}_vs_structured",
                         "intervention": iv, "stratum": "",
                         "test": "cluster_bootstrap", "statistic": round(red_pt, 3),
                         "p_raw": _p_two_sided(red_bt), "p_fdr": None})

        # --- decomposition: marginal + complementary, notes & imaging, BOTH note
        #     variants (§6.3): 'all' (radiology-inclusive, primary) and 'clinical'.
        for variant, pn, fu in [("all", "plus_notes", "full"),
                                ("clinical", "plus_notes_clinical", "full_clinical")]:
            if not ({pn, "plus_imaging_only", fu} <= set(bt)):
                continue
            for mod, own, other in [("notes", pn, "plus_imaging_only"),
                                    ("imaging", "plus_imaging_only", pn)]:
                marg_pt = bias0_pt - abs(pt[own] - ref)
                marg_bt = bias0_bt - _absbias(bt[own], ref)
                comp_pt = abs(pt[other] - ref) - abs(pt[fu] - ref)   # added on top of the other
                comp_bt = _absbias(bt[other], ref) - _absbias(bt[fu], ref)
                decomp.append({"intervention": iv, "modality": mod, "notes_variant": variant,
                               **bootstrap_summary(marg_pt, marg_bt).as_row("marginal_bias_reduction_"),
                               **bootstrap_summary(comp_pt, comp_bt).as_row("complementary_bias_reduction_")})

        # --- comparisons: each condition's effect vs the RCT reference ---
        for cond in d["conds"]:
            diff_bt = np.asarray(bt[cond]) - ref
            comp.append({"comparison": f"effect_vs_reference[{cond}]",
                         "intervention": iv, "stratum": "",
                         "test": "cluster_bootstrap", "statistic": round(pt[cond] - ref, 3),
                         "p_raw": _p_two_sided(diff_bt), "p_fdr": None})

    # BH-FDR across the whole comparisons family (§8 Kind C)
    if comp:
        p = benjamini_hochberg([r["p_raw"] for r in comp])
        for r, pf in zip(comp, p):
            r["p_fdr"] = float(pf)

    for fname, rows in [("dissociation.csv", diss), ("decomposition.csv", decomp),
                        ("comparisons.csv", comp)]:
        path = cfg.storage("results", fname)
        if path.exists():
            path.unlink()
        R.append_rows(cfg, fname, rows)
        log(f"  {fname}: {len(rows)} rows")

    # a fix-variant dir needs the variant-INDEPENDENT outputs too (image quality gate;
    # CONSORT counts) so it is a complete, comparable 10-file set. Copy them from the
    # canonical results/ (they don't depend on the embedding reduction).
    if cfg._results_override:
        import shutil
        canon = cfg.storage_root / cfg.get("paths.results", "results")
        if (canon / "probe.csv").exists():
            shutil.copy(canon / "probe.csv", results_dir / "probe.csv")
        # merge canonical CONSORT rows into the variant cohorts.csv (keep its overlap_ess)
        cvar, ccanon = results_dir / "cohorts.csv", canon / "cohorts.csv"
        if ccanon.exists() and cvar.exists():
            import pandas as pd
            a = pd.read_csv(cvar); b = pd.read_csv(ccanon)
            consort = b[b["section"] != "overlap_ess"]
            pd.concat([consort, a], ignore_index=True).to_csv(cvar, index=False)
        log("  copied variant-independent outputs (probe, CONSORT) from canonical results/")

    log(f"consolidate done -> {results_dir} ({len(boot_files)} intervention(s)).")
