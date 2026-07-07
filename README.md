# Multimodal Adjustment for ICU Treatment-Effect Estimation

Causal inference pipeline that estimates treatment effects for four ICU
interventions using observational MIMIC-IV data, and validates the estimates
against published randomized trial results. The core question: do frozen
chest X-ray and clinical-note embeddings, used as adjustment covariates,
reduce confounding bias beyond what structured data alone can achieve?

## Overview

The pipeline emulates each intervention as a target trial inside MIMIC-IV,
estimates the effect under six covariate adjustment conditions (none →
structured → notes → imaging → all three → expert-curated), and compares
each estimate to the corresponding RCT risk difference. The effect measure is
the risk difference at the trial horizon, in percentage points.

**Interventions and reference trials**

| Intervention | Design role | Reference trial(s) |
|---|---|---|
| Prone positioning vs supine (severe ARDS) | Positive control | PROSEVA |
| Conservative vs liberal fluids (sepsis) | Null calibration | CLOVERS, CLASSIC |
| Early vs delayed renal replacement therapy (AKI) | Null calibration | STARRT-AKI |
| Restrictive vs liberal transfusion threshold | Null calibration | TRICC, TRISS |

**Adjustment conditions** (all on the shared all-modality cohort)

| Condition | Covariates |
|---|---|
| `naive` | None |
| `structured` | Vitals, labs, demographics at time zero |
| `plus_notes` | Structured + Clinical-Longformer note embeddings |
| `plus_imaging_only` | Structured + RAD-DINO image embeddings |
| `full` | Structured + notes + images |
| `design_based` | Expert-curated structured confounder set |

**Estimator**: cross-fitted AIPW with LightGBM nuisance models throughout.
Embeddings are frozen (no fine-tuning) and extracted once; every downstream
stage runs on CPU.

## Data

MIMIC-IV is credentialed patient data — access requires a PhysioNet data use
agreement. Raw data and all derived tables are excluded from this repository
via `.gitignore`.

| Dataset | Version | Role |
|---|---|---|
| MIMIC-IV | 3.1 | Structured EHR (backbone) |
| MIMIC-IV-ED | 2.2 | Emergency department covariates |
| MIMIC-IV-Note | 2.2 | Free-text clinical notes |
| MIMIC-CXR-JPG | 2.1.0 | Chest X-ray images + CheXpert labels |
| eICU-CRD | 2.0 | External validation (structured only) |

## Repository structure

```
.
├── config.yaml              # Single source of truth: all paths, hyperparams,
│                            #   intervention definitions, RCT reference values
├── main_causal.py           # Sole entry point: --stage <name>|all [--intervention k]
├── requirements.txt
│
├── src/
│   ├── cfg.py               # config.yaml loader with ${ENV:default} interpolation
│   ├── aipw.py              # Cross-fitted AIPW estimator
│   ├── features.py          # Covariate assembly (structured + embedding pooling)
│   ├── reduce.py            # Embedding dimensionality reduction (PCA / p-score)
│   ├── stats.py             # Cluster bootstrap, BH-FDR, E-values, IF-CIs
│   ├── results.py           # Output file schemas and writers
│   ├── util.py              # Logging and checkpoint markers
│   └── stages/
│       ├── link.py          # §9.3  Verify MIMIC-IV × MIMIC-CXR linkage
│       ├── extract_embeddings.py  # §9.4  RAD-DINO images + Clinical-Longformer notes
│       ├── emulate.py       # §9.5  Target-trial cohort construction
│       ├── probe.py         # §9.6  Validity gate: proxy → target confounder AUROC
│       ├── estimate.py      # §9.7  AIPW effects for all conditions
│       ├── demographics.py  # §9.10 Subgroup re-estimation (sex, age band)
│       ├── robustness.py    # §9.11 Four sensitivity swaps
│       ├── external.py      # §9.12 eICU replication (structured + notes)
│       ├── consolidate.py   # §9.13 Merge into 10 result files
│       └── integrity_check.py  # §9.14 Invariant assertions before sign-off
│
└── pipeline/                # One-time data preparation (SLURM batch jobs)
    ├── build_link_layer.py  # Join MIMIC-IV patients × CXR studies → link/
    ├── build_event_stream.py  # Unify time-series events → Parquet
    ├── clean_event_stream.py  # Value cleaning and range filtering
    └── qc_event_stream.py   # QC checks on the event stream
```

## Setup

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For the embedding extraction step only (GPU node), additionally install:

```bash
pip install torch transformers pillow
```

### 2. Configure paths

Edit `config.yaml` — all paths support `${ENV_VAR:default}` interpolation,
so you can override them with environment variables without touching the file:

```bash
export CTFM_STORAGE=/path/to/storage   # where embeddings and results are written
export CTFM_DATASETS=/path/to/datasets # where MIMIC-IV, MIMIC-CXR, etc. live
```

### 3. Run a stage

```bash
# Single stage
python main_causal.py --stage link
python main_causal.py --stage extract_embeddings
python main_causal.py --stage emulate --intervention fluids_sepsis

# Full pipeline (runs stages in order)
python main_causal.py --stage all

# Resume after interruption — each stage checkpoints internally
python main_causal.py --stage estimate --intervention fluids_sepsis
```

All results are written to `paths.storage_root` (set in `config.yaml`),
never inside the repository.

## Design principles

- **One config file.** `config.yaml` holds every path, hyperparameter,
  intervention definition, and RCT reference value. Nothing is defined
  anywhere else.
- **No fabrication.** Every number in every result file is computed from
  real data. Missing values are left explicitly empty, never filled with
  estimates or placeholders.
- **Time-zero discipline.** Only data strictly before time zero may enter
  the adjustment set. Anything at or after leaks the outcome; this is
  asserted in code, not just assumed.
- **Compute discipline.** The only GPU-intensive step is the one-time
  embedding extraction. Every subsequent stage runs on CPU.
- **Checkpoint and resume.** Every stage writes progress markers so
  interrupted cluster jobs resume rather than restart.


