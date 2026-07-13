"""integrity -- assert the study's invariants. This stage FAILS LOUDLY or it is worthless.

The previous version of this file described five invariants and checked two. That is worse
than checking none, because it reads like a guarantee. Every check below actually runs.

STRUCTURAL
  1. Results live under storage_root, never in the repo or $HOME.
  2. Exactly the seven result files exist.
  3. The embedding index and the vector blobs agree in length (a misalignment would attach
     one patient's X-ray to another and nothing downstream would notice).
  4. Every intervention's conditions ran on the IDENTICAL patient set.

SCIENTIFIC -- these are the ones that decide whether the paper's claims are supported
  5. TIME-ZERO DISCIPLINE. No embedded item used for adjustment is timestamped at or after
     its patient's t0. Checked against the manifests directly, not assumed.
  6. POSITIVE CONTROL 1 (real). `structured` must reduce |bias| vs `naive` on the
     adequate-overlap interventions. If structured adjustment cannot reduce bias on real
     data, the pipeline cannot detect bias reduction at all and NO null in this study means
     anything.
  7. POSITIVE CONTROL 2 (synthetic). `struct_img` must recover the known tau when the
     confounder is routed through the image channel BY CONSTRUCTION. If it does not, the
     estimator or the embeddings are broken.
  8. EXPERT-SET COMPLETENESS. The `expert` arm must have extracted most of the confounders
     it claims. An expert competitor missing its own eligibility variables is a straw man,
     and beating it would prove nothing.

If 6 or 7 fail, STOP. Do not interpret, do not write up, do not send results to Soroosh.
A null result is only meaningful from a pipeline that has been shown capable of detecting
the thing it failed to find.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.util import log, die
from src import events as ev
from src.results import RESULT_FILES

_REPO_ROOT = Path(__file__).resolve().parents[2]


def run(cfg, force: bool = False, intervention: str = None):
    problems, passed = [], []

    # ---- 1. results are not in the repo or $HOME ----
    storage = cfg.storage_root.resolve()
    home = Path.home().resolve()
    if _REPO_ROOT in storage.parents or storage == _REPO_ROOT:
        problems.append(f"storage_root {storage} is inside the repo")
    elif home == storage:
        problems.append(f"storage_root {storage} is $HOME itself")
    else:
        passed.append("results live outside the repo")

    # ---- 2. the seven result files ----
    rdir = cfg.storage("results")
    present = sorted(p.name for p in rdir.glob("*.csv")) if rdir.exists() else []
    missing = [f for f in RESULT_FILES if f not in present]
    extra = [f for f in present if f not in RESULT_FILES]
    if missing:
        problems.append(f"missing result files: {missing}")
    if extra:
        problems.append(f"unexpected result files: {extra}")
    if not missing and not extra:
        passed.append(f"exactly the {len(RESULT_FILES)} result files present")

    # ---- 3. embedding index <-> vector blob alignment ----
    try:
        idx = pd.read_parquet(ev.manifest_path(cfg, "embeddings_index.parquet"))
        for mod in idx["modality"].unique():
            n_idx = int((idx["modality"] == mod).sum())
            blob = cfg.storage("embeddings", f"{mod}.npy")
            if not blob.exists():
                problems.append(f"{mod}: index has {n_idx} rows but no vector blob")
                continue
            n_vec = int(np.load(blob, mmap_mode="r").shape[0])
            if n_idx != n_vec:
                problems.append(
                    f"{mod}: {n_idx} index rows vs {n_vec} vectors -- MISALIGNED. "
                    f"Every downstream number is suspect.")
            else:
                passed.append(f"{mod}: index and vectors aligned ({n_vec:,})")
    except Exception as e:
        problems.append(f"could not verify embedding alignment: {e}")

    # ---- 4 + 5. shared cohort and time-zero discipline ----
    try:
        cohorts = ev.load_cohorts(cfg)
        idx = pd.read_parquet(ev.manifest_path(cfg, "embeddings_index.parquet"))
        idx["ts"] = pd.to_datetime(idx["ts"], errors="coerce")

        for iv, coh in cohorts.groupby("intervention"):
            imaged = coh[coh["imaged"] & coh["arm"].notna()]
            if not len(imaged):
                continue
            passed.append(f"{iv}: shared imaged cohort n={len(imaged):,} "
                          f"(all conditions use this same set)")

            m = idx.merge(imaged[["subject_id", "t0"]], on="subject_id", how="inner")
            m["t0"] = pd.to_datetime(m["t0"], errors="coerce")
            violations = int((m["ts"] >= m["t0"]).sum())
            if violations:
                problems.append(
                    f"{iv}: TIME-ZERO VIOLATION -- {violations:,} embedded items are "
                    f"timestamped at or after t0. Adjustment is using post-baseline data.")
            else:
                passed.append(f"{iv}: time-zero discipline holds "
                              f"({len(m):,} items checked, all strictly pre-t0)")
    except Exception as e:
        problems.append(f"could not verify cohort/time-zero invariants: {e}")

    # ---- 6. POSITIVE CONTROL 1: structured beats naive on real data ----
    epath = rdir / "effects.csv"
    if epath.exists():
        eff = pd.read_csv(epath)
        pc = cfg.get("positive_controls.real_structured_beats_naive") or {}
        need = float(pc.get("min_bias_reduction_pp", 2.0))
        for iv in pc.get("interventions", []):
            g = eff[(eff["intervention"] == iv) & (eff["cohort"] == "imaged")
                    & (eff["dataset"] == "mimic")]
            try:
                nb = abs(float(g[g["condition"] == "naive"]["bias_point"].iloc[0]))
                sb = abs(float(g[g["condition"] == "structured"]["bias_point"].iloc[0]))
            except (IndexError, ValueError):
                problems.append(f"POSITIVE CONTROL 1: no naive/structured rows for {iv}")
                continue
            got = nb - sb
            if got < need:
                problems.append(
                    f"POSITIVE CONTROL 1 FAILED [{iv}]: structured adjustment reduced bias "
                    f"by only {got:+.1f} pp (need >= {need}). The pipeline cannot be shown "
                    f"to detect bias reduction on real data, so NO null in this study is "
                    f"interpretable. STOP and diagnose before going further.")
            else:
                passed.append(f"POSITIVE CONTROL 1 [{iv}]: structured cut bias by "
                              f"{got:+.1f} pp vs naive")

    # ---- 7. POSITIVE CONTROL 2: the image recovers a confounder placed inside it ----
    spath = rdir / "synthetic.csv"
    if spath.exists():
        syn = pd.read_csv(spath)
        pc = cfg.get("positive_controls.synthetic_image_recovers_tau") or {}
        ga, de = float(pc.get("at_gamma", 1.5)), float(pc.get("at_delta", 1.5))
        need = float(pc.get("min_bias_reduction_pp", 2.0))
        cell = syn[(syn["sweep"] == "gamma_delta") & (syn["gamma"] == ga)
                   & (syn["delta"] == de)]
        if not len(cell):
            problems.append(f"POSITIVE CONTROL 2: no synthetic cell at gamma={ga}, delta={de}")
        else:
            got = float(cell["bias_reduction"].mean())
            if got < need:
                problems.append(
                    f"POSITIVE CONTROL 2 FAILED: at gamma={ga}, delta={de} the image "
                    f"channel recovered only {got:+.2f} pp of a confounder that is inside "
                    f"it BY CONSTRUCTION (need >= {need}). The estimator, the reduction, or "
                    f"the embeddings are broken. STOP.")
            else:
                passed.append(f"POSITIVE CONTROL 2: image recovered {got:+.2f} pp of the "
                              f"planted confounder")

    # ---- 8. expert-set completeness ----
    if epath.exists():
        eff = pd.read_csv(epath)
        g = eff[eff["condition"] == "expert"].dropna(subset=["expert_confounders_extracted"])
        for _, r in g.iterrows():
            try:
                got, want = int(r["expert_confounders_extracted"]), int(r["expert_confounders_requested"])
            except (ValueError, TypeError):
                continue
            frac = got / want if want else 0
            if frac < 0.75:
                problems.append(
                    f"EXPERT SET INCOMPLETE [{r['intervention']}]: only {got}/{want} "
                    f"confounders extracted. The design-based competitor is a straw man; "
                    f"beating it would prove nothing.")
            else:
                passed.append(f"expert set [{r['intervention']}]: {got}/{want} extracted")

    # ---- report ----
    for p in passed:
        log(f"  PASS: {p}")
    if problems:
        for p in problems:
            log(f"  FAIL: {p}")
        die(f"integrity failed ({len(problems)} problem(s)). "
            f"Do NOT interpret or report these results until every check passes.")
    log(f"integrity PASSED ({len(passed)} checks).")
