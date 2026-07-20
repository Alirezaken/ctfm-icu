"""The SEVEN result files. One exact schema, one writer, defined in one place.

Consolidated from the previous ten so that every number a reviewer needs to check a claim
sits next to the claim, and so that Alireza (and Soroosh) can audit the study by reading
seven files instead of ten cross-referenced ones.

  effects.csv      every (intervention x condition x cohort x dataset): the risk
                   difference, its CI, the bias against the RCT anchor, the divergence Z,
                   the ATO, and the overlap diagnostics that determine whether the number
                   means anything.
  diagnostics.csv  the paper's contribution. Per (intervention x modality): the probe
                   AUROC (is it informative?), dAUC_treat and dAUC_outcome (is it
                   INCREMENTALLY confounding?), the positivity cost (can we afford it?),
                   and the decision-rule verdict with a plain-language reason.
  contrasts.csv    every PAIRED contrast with a p-value and BH-FDR. Bias reductions,
                   modality decomposition, subgroup differences, multimodal-vs-expert.
                   The ONLY file with p-values.
  cohorts.csv      CONSORT, demographics by arm, covariate balance, missingness,
                   censoring, minimum detectable effects.
  synthetic.csv    the semi-synthetic calibration sweep. The proof the estimator works
                   and the null is a fact about the data.
  robustness.csv   encoder / estimator / window / pooling / reduction / trim swaps, and
                   the subgroup re-estimations.
  manifest.csv     full reproducibility record.

Column KINDS are baked into the schema (see stats.py):
  Kind B (causal)      -> 5 columns: _point _mean _std _ci_low _ci_high   [percentage points]
  Kind A (performance) -> 4 columns: _mean _std _ci_low _ci_high          [percent]
  Kind C (statistical) -> 1 column, natural scale, never with a mean/CI
`append_rows` refuses any column not in the schema, so structure cannot drift silently.
"""
from __future__ import annotations

import csv

from src.util import log


def _kb(name: str) -> list[str]:
    """Kind-B causal quantity -> 5 columns."""
    return [f"{name}_point", f"{name}_mean", f"{name}_std",
            f"{name}_ci_low", f"{name}_ci_high"]


def _ka(name: str) -> list[str]:
    """Kind-A performance metric -> 4 columns."""
    return [f"{name}_mean", f"{name}_std", f"{name}_ci_low", f"{name}_ci_high"]


SCHEMAS: dict[str, list[str]] = {

    # ---------------------------------------------------------------- 1
    "effects.csv": (
        ["intervention", "condition", "cohort", "dataset", "estimator", "reduction"]
        + _kb("effect")                       # the risk difference (ATE)
        + ["effect_if_ci_low", "effect_if_ci_high"]        # influence-function CI, a check
        + _kb("ato")                          # overlap-weighted estimand
        + ["ref_rd", "ref_ci_low", "ref_ci_high", "ref_source"]
        + _kb("bias")                         # effect - RCT reference
        + ["divergence_z",                    # Kind C: replaces inside_reference_ci
           "ci_overlaps_rct",                 # honest weak compatibility check
           "ci_width", "effective_sample_size", "ess_ratio_vs_structured",
           "propensity_min", "propensity_max", "frac_trimmed", "frac_censored",
           "n_analyzed", "n_active", "n_comparator", "n_events",
           "min_detectable_effect_pp",
           "expert_confounders_extracted", "expert_confounders_requested"]
        + _kb("negative_control")
        + ["e_value", "e_value_ci_limit", "undefined", "note"]
    ),

    # ---------------------------------------------------------------- 2
    # THE CONTRIBUTION. One row per (intervention, modality).
    "diagnostics.csv": (
        ["intervention", "modality", "check"]
        # (1) informativeness: does the embedding encode the confounder at all?
        + _ka("probe_auroc") + _ka("probe_auprc") + ["probe_target", "probe_n_test"]
        # (2) incremental confounding: is that information NEW, and does it move BOTH channels?
        + ["auc_treat_structured", "auc_treat_with_modality", "d_auc_treat",
           "auc_outcome_structured", "auc_outcome_with_modality", "d_auc_outcome",
           "ici"]
        # (3) positivity: can we afford it?
        + ["ess_structured", "ess_with_modality", "positivity_cost"]
        # the rule
        + ["add_modality", "reason",
           "observed_bias_reduction_pp", "predicted_bias_reduction_pp"]
    ),

    # ---------------------------------------------------------------- 3
    "contrasts.csv": (
        ["contrast", "intervention", "modality", "stratum"]
        + _kb("value")
        + ["min_detectable_pp",               # Kind C: makes a null decisive
           "test", "p_raw", "p_fdr"]
    ),

    # ---------------------------------------------------------------- 4
    "cohorts.csv": [
        "intervention", "section", "metric", "stratum", "arm", "value", "support_count",
    ],

    # ---------------------------------------------------------------- 5
    "synthetic.csv": (
        ["sweep", "gamma", "delta", "redundancy", "replicate",
         "true_rd_pp", "n", "confounder_r2_on_structured"]
        + ["rd_structured", "rd_struct_img",
           "bias_structured", "bias_struct_img", "bias_reduction",
           "d_auc_treat", "d_auc_outcome", "ici",
           "ess_structured", "ess_struct_img", "positivity_cost"]
    ),

    # ---------------------------------------------------------------- 6
    "robustness.csv": (
        ["intervention", "family", "swap", "condition", "subgroup"]
        + _kb("value")
        + ["support_count", "undefined", "note"]
    ),

    # ---------------------------------------------------------------- 7
    "manifest.csv": ["key", "value"],
}

RESULT_FILES = list(SCHEMAS)


def write_templates(cfg, force: bool = False) -> list[str]:
    """Header-only CSVs for all seven files. Existing non-empty files are left alone
    unless force=True."""
    out = cfg.storage("results")
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for fname, cols in SCHEMAS.items():
        p = out / fname
        if p.exists() and p.stat().st_size > 0 and not force:
            continue
        with open(p, "w", newline="") as fh:
            csv.writer(fh).writerow(cols)
        written.append(fname)
    log(f"result templates ready in {out} "
        f"({len(written)} written, {len(SCHEMAS) - len(written)} already present)")
    return written


def append_rows(cfg, fname: str, rows: list):
    """Append rows, enforcing the schema. An unknown column is an error, not a warning:
    a silently-dropped column is a silently-lost result."""
    if fname not in SCHEMAS:
        raise KeyError(f"{fname} is not one of the seven result files")
    cols = SCHEMAS[fname]
    p = cfg.storage("results", fname)
    if not p.exists():
        write_templates(cfg)
    with open(p, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="raise")
        for r in rows:
            unknown = set(r) - set(cols)
            if unknown:
                raise ValueError(f"{fname}: unknown column(s) {sorted(unknown)}")
            w.writerow({c: r.get(c, "") for c in cols})


def reset_rows(cfg, fname: str, **match):
    """Drop rows matching a set of column==value filters, so a stage can re-run
    idempotently without wiping other stages' rows from the same file."""
    import pandas as pd
    p = cfg.storage("results", fname)
    if not p.exists():
        return
    df = pd.read_csv(p)
    if not len(df):
        return
    m = pd.Series(True, index=df.index)
    for col, val in match.items():
        if col not in df.columns:
            return
        vals = val if isinstance(val, (list, tuple, set)) else [val]
        m &= df[col].isin(list(vals))
    df[~m].to_csv(p, index=False)


def build_manifest(cfg) -> int:
    """The reproducibility record. Everything needed to re-run this study exactly.

    Values that are genuinely unknown at write time are written EMPTY, never invented.
    """
    import importlib.metadata as md
    rows = []

    def add(k, v):
        rows.append((k, "" if v is None else str(v)))

    add("study", "RCT-anchored audit of multimodal causal adjustment in the ICU")
    add("seed", cfg.get("run.seed"))
    add("bootstrap.n_resamples", cfg.get("bootstrap.n_resamples"))
    add("bootstrap.unit", cfg.get("bootstrap.unit"))
    add("bootstrap.reuse_indices_across_conditions",
        cfg.get("bootstrap.reuse_indices_across_conditions"))

    # estimator
    for k in ["method", "nuisance_model", "cross_fitting_folds", "trim",
              "embedding_reduction", "reduction_nested", "pca_components", "report_ato"]:
        add(f"estimator.{k}", cfg.get(f"estimator.{k}"))
    add("effect_measure", cfg.get("run.effect_measure"))
    add("effect_scale", cfg.get("run.effect_scale"))

    # modalities: what each channel ACTUALLY is
    for name in ["images", "radtext", "histnote", "images_alt"]:
        add(f"modality.{name}.encoder", cfg.get(f"modalities.{name}.encoder"))
        add(f"modality.{name}.hf_id", cfg.get(f"modalities.{name}.hf_id"))
        add(f"modality.{name}.source", cfg.get(f"modalities.{name}.source"))
    add("modality.images.views", cfg.get("modalities.images.views"))
    add("modality.histnote.caveat", cfg.get("modalities.histnote.note"))

    # pooling / gate
    add("pooling.rule", cfg.get("pooling.rule"))
    add("pooling.look_back_window_hours", cfg.get("pooling.look_back_window_hours"))
    add("cohort.primary_gate", cfg.get("cohort.primary_gate"))

    # diagnostic thresholds
    for k in ["auc_threshold", "positivity_cost_max", "bootstrap_reps"]:
        add(f"diagnostic.{k}", cfg.get(f"diagnostic.{k}"))

    # synthetic benchmark
    for k in ["base_intervention", "gamma_grid", "delta_grid", "replicates",
              "target_rd_pp", "redundancy_grid"]:
        add(f"synthetic.{k}", cfg.get(f"synthetic.{k}"))

    # robustness axes (ALL of them, not just the encoder)
    for k in ["encoder_alt", "estimator_alt", "look_back_window_alt_hours",
              "pooling_alt", "reduction_alt", "trim_alt"]:
        add(f"robustness.{k}", cfg.get(f"robustness_swaps.{k}"))

    # demographics
    add("demographics.age_bands", cfg.get("demographics.age_bands"))
    add("demographics.min_arm_events", cfg.get("demographics.min_arm_events"))

    # interventions: definition + RCT anchor + how the anchor was derived
    for iv, spec in (cfg.get("interventions") or {}).items():
        add(f"intervention.{iv}.role", spec.get("role"))
        add(f"intervention.{iv}.arms", spec.get("arms"))
        add(f"intervention.{iv}.outcome", spec.get("outcome"))
        add(f"intervention.{iv}.horizon_days", spec.get("horizon_days"))
        r = spec.get("rct_reference") or {}
        add(f"intervention.{iv}.rct",
            f"{r.get('source')}: RD={r.get('risk_difference')} CI={r.get('ci')} "
            f"@{r.get('horizon_days')}d")
        add(f"intervention.{iv}.rct_derivation", r.get("derivation"))

    # dataset versions
    for k, v in (cfg.get("dataset_versions") or {}).items():
        add(f"dataset_version.{k}", v)

    # environment
    for pkg in ["numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "lightgbm", "pyyaml"]:
        try:
            add(f"pkg.{pkg}", md.version(pkg))
        except md.PackageNotFoundError:
            add(f"pkg.{pkg}", None)

    # wall-clock, filled by the stages that know it
    add("walltime.embeddings_sec", None)
    add("walltime.estimate_sec", None)
    add("walltime.synthetic_sec", None)

    p = cfg.storage("results", "manifest.csv")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(SCHEMAS["manifest.csv"])
        w.writerows(rows)
    log(f"manifest.csv: {len(rows)} keys")
    return len(rows)
