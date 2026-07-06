"""The 10 result files (§7) -- one exact schema, one writer, in one place.

Defines the column layout of every result CSV per §7, with the column "kinds"
of §8 baked in (Kind A performance / Kind B causal-effect / Kind C statistical),
so no stage invents its own format. `write_templates` lays down header-only files;
stages append rows through `append_rows`, which refuses any column not in the
schema (the mirror of the no-fabrication rule -- structure can't drift silently).

Kind-B quantities expand to five columns: <name>_point (AIPW point estimate),
_mean, _std, _ci_low, _ci_high (patient-level cluster bootstrap, §8).
Kind-A metrics expand to four: _mean, _std, _ci_low, _ci_high (percent).
"""
from __future__ import annotations

import csv

from src.util import log


def _kb(name: str) -> list[str]:
    """Kind-B causal-effect quantity -> 5 columns (point + bootstrap summary)."""
    return [f"{name}_point", f"{name}_mean", f"{name}_std",
            f"{name}_ci_low", f"{name}_ci_high"]


def _ka(name: str) -> list[str]:
    """Kind-A performance metric -> 4 columns (bootstrap summary, percent)."""
    return [f"{name}_mean", f"{name}_std", f"{name}_ci_low", f"{name}_ci_high"]


# filename -> ordered column list. Keys first, then values.
SCHEMAS: dict[str, list[str]] = {
    # 1. main effects (§7.1)
    "effects.csv": (
        ["intervention", "condition", "cohort", "dataset", "method"]
        + _kb("effect")
        + ["effect_if_ci_low", "effect_if_ci_high"]   # §8 influence-function CI, a check
        + ["ref_rd", "ref_ci_low", "ref_ci_high"]
        + _kb("bias")
        + ["inside_reference_ci"]
        + _kb("negative_control")
        + ["ci_width", "effective_sample_size"]
    ),
    # 2. specificity grid (§7.2)
    "dissociation.csv": (
        ["intervention", "modality"] + _kb("bias_reduction_vs_structured")
    ),
    # 3. modality decomposition (§7.3)
    "decomposition.csv": (
        ["intervention", "modality", "notes_variant"]
        + _kb("marginal_bias_reduction") + _kb("complementary_bias_reduction")
    ),
    # 4. negative controls + sensitivity (§7.4)
    "controls.csv": (
        ["intervention", "condition"]
        + _kb("negative_control")
        + ["e_value", "e_value_ci_limit"]          # Kind C, natural scale
    ),
    # 5. validity probe (§7.5)
    "probe.csv": (
        ["modality", "target_confounder"] + _ka("auroc") + _ka("auprc")
    ),
    # 6. cohorts / CONSORT / overlap / ESS / MDE / missingness / demographics (§7.6)
    #    heterogeneous -> long format (one quantity per row)
    "cohorts.csv": [
        "intervention", "section", "metric", "stratum", "arm",
        "value", "support_count",
    ],
    # 7. demographics / subgroups (§7.7)
    "demographics.csv": (
        ["intervention", "subgroup_type", "subgroup", "condition"]
        + _kb("effect") + _kb("bias_reduction")
        + ["support_count", "undefined"]
    ),
    # 8. robustness swaps (§7.8)
    "robustness.csv": (
        ["intervention", "swap"] + _kb("structured_to_full_bias_reduction")
    ),
    # 9. comparisons -- the ONLY file with p-values (§7.9); no percent, no mean/CI
    "comparisons.csv": [
        "comparison", "intervention", "stratum",
        "test", "statistic", "p_raw", "p_fdr",
    ],
    # 10. manifest -- reproducibility key/value (§7.10)
    "manifest.csv": ["key", "value"],
}

RESULT_FILES = list(SCHEMAS)


def write_templates(cfg, force: bool = False) -> list[str]:
    """Create header-only CSVs for all 10 result files under paths.results.
    Existing non-empty files are left untouched unless force=True."""
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
    log(f"result templates ready in {out} ({len(written)} written, "
        f"{len(SCHEMAS) - len(written)} already present)")
    return written


def build_manifest(cfg) -> int:
    """Write manifest.csv from values that are genuinely known now: config
    settings and installed package versions. Unknown-yet values (wall-clock
    times, encoder checkpoints once pinned, look-back/age bands once chosen)
    are written as empty -- never fabricated (§0). Returns row count."""
    import importlib.metadata as md

    rows: list[tuple[str, str]] = []

    def add(k, v):
        rows.append((k, "" if v is None else str(v)))

    # reproducibility
    add("seed", cfg.get("run.seed"))
    add("bootstrap.n_resamples", cfg.get("bootstrap.n_resamples"))
    add("bootstrap.unit", cfg.get("bootstrap.unit"))
    # estimator
    add("estimator.method", cfg.get("estimator.method"))
    add("effect_measure", cfg.get("run.effect_measure"))
    add("effect_scale", cfg.get("run.effect_scale"))
    add("estimator.cross_fitting_folds", cfg.get("estimator.cross_fitting_folds"))
    add("estimator.nuisance_model", cfg.get("estimator.nuisance_model"))
    # encoders
    add("encoder.image", cfg.get("images.encoder"))
    add("encoder.image.size", cfg.get("images.size"))
    add("encoder.text", cfg.get("text.encoder"))
    add("encoder.text.max_tokens", cfg.get("text.max_tokens"))
    add("encoder.image_alt", cfg.get("robustness_swaps.encoder_alt"))
    # pooling / window / bands
    add("pooling.rule", cfg.get("pooling.rule"))
    add("look_back_window_hours", cfg.get("pooling.look_back_window_hours"))
    add("age_bands", cfg.get("demographics.age_bands"))
    # dataset versions (§1, pinned in config)
    for k, v in (cfg.get("dataset_versions") or {}).items():
        add(f"dataset_version.{k}", v)
    # package versions of the run environment
    for pkg in ["numpy", "pandas", "pyarrow", "scipy", "scikit-learn",
                "lightgbm", "pyyaml"]:
        try:
            add(f"pkg.{pkg}", md.version(pkg))
        except md.PackageNotFoundError:
            add(f"pkg.{pkg}", None)
    # wall-clock (filled by the extraction/estimation stages)
    add("walltime.extract_embeddings_sec", None)
    add("walltime.estimate_sec", None)

    p = cfg.storage("results", "manifest.csv")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(SCHEMAS["manifest.csv"])
        w.writerows(rows)
    return len(rows)


def append_rows(cfg, fname: str, rows: list[dict]):
    """Append rows to a result file, enforcing the schema (no unknown columns)."""
    if fname not in SCHEMAS:
        raise KeyError(f"{fname} is not one of the 10 result files")
    cols = SCHEMAS[fname]
    p = cfg.storage("results", fname)
    if not p.exists():
        write_templates(cfg)
    with open(p, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="raise")
        for r in rows:
            unknown = set(r) - set(cols)
            if unknown:
                raise ValueError(f"{fname}: unknown column(s) {unknown}")
            w.writerow({c: r.get(c, "") for c in cols})
