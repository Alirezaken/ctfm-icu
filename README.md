# Foundation-model embeddings predict the confounder but do not reduce confounding bias in critical care

## Overview

Foundation-model embeddings of chest X-rays and clinical notes are increasingly proposed as
confounder proxies for causal adjustment in observational EHR studies — the intuition being
that a richer, validated representation of a patient's clinical state should absorb more
confounding than the structured record alone. This repository is the code for an audit of
that intuition. It emulates four ICU target trials, each anchored to a completed randomized
controlled trial (PROSEVA, CLOVERS/CLASSIC, STARRT-AKI, TRISS), so that adjustment quality can
be checked against real trial evidence rather than against internal consistency alone.

Each target trial is estimated under seven adjustment conditions — a naive comparison,
adjustment on the structured EHR record alone, and adjustment on the structured record plus
each of three candidate embeddings (a chest-X-ray encoder and two clinical-text encoders) —
using cross-fitted, doubly robust estimation. A pre-estimation diagnostic and a
semi-synthetic benchmark with a known, planted effect are built into the pipeline so that a
null finding can be told apart from a pipeline that simply isn't working. External
replication on an independent ICU database, and an image-embedding quality gate on three
public chest-X-ray datasets, complete the validation. The findings of this audit are reported
in the accompanying manuscript; this repository documents the method and reproduces the
pipeline, not the findings.

## Key features

- **Target-trial emulation anchored to RCTs**: four ICU interventions, each mapped onto a
  completed randomized trial's population, exposure, and outcome definition, so an estimated
  effect can be compared against that trial's own effect size and confidence interval.
- **Cross-fitted doubly robust estimation**: augmented inverse-probability weighting (AIPW)
  with LightGBM nuisance models, targeted maximum-likelihood estimation (TMLE) as a
  robustness check, inverse-probability-of-censoring weights (IPCW) for informative dropout,
  and overlap (ATO) weights where positivity is limited.
- **Seven adjustment conditions per intervention**: naive, structured-only, and structured
  plus each of three candidate embeddings, isolating what each modality contributes beyond
  the structured record alone.
- **A pre-estimation diagnostic** (`diagnose`): an incremental-confounding index and a
  treatment/outcome discrimination comparison that characterize a candidate modality's
  relationship to confounding before the estimation stage runs, independent of the eventual
  effect estimate.
- **A semi-synthetic benchmark** (`synthesize`): the identical pipeline re-run on real
  covariates with a simulated treatment, outcome, and known true effect, so "no signal in
  this data" can be told apart from "the pipeline failed to detect a signal that is there."
- **External and cross-dataset validation**: structured and structured-plus-notes replication
  on an independent ICU database (eICU-CRD), and an image-embedding informativeness check on
  three public chest-X-ray datasets (PadChest, ChestX-ray14, CheXpert) that never enters the
  causal estimates.
- **Statistical discipline**: patient-level cluster bootstrap confidence intervals,
  Benjamini-Hochberg correction for multiple comparisons, and a strict separation between
  performance statistics, causal-effect statistics, and p-values, so they are never collapsed
  into one number (`src/stats.py`).

## Target trials

The audit covers four ICU interventions, each mapped to a completed RCT used as the
ground-truth reference for that trial's own effect size:

| Intervention | RCT anchor | RCT effect (RD, 95% CI) | Horizon | Role |
|---|---|---|---|---|
| `fluids_sepsis` | CLOVERS / CLASSIC (meta) | −0.6 pp [−3.4, 2.3] | 90d | Primary null calibration — best sample size, best overlap |
| `transfusion_threshold` | TRISS (TRICC as sensitivity) | −1.9 pp [−8.1, 4.2] | 90d | Null calibration — key confounder (hemoglobin) is already in the structured record |
| `rrt_timing` | STARRT-AKI | +0.2 pp [−0.3, 0.4] | 90d | Null calibration — tests whether "urgency" information lives in text |
| `prone_positioning` | PROSEVA | −16.8 pp [−24.5, −9.1] | 28d | Positive control; pre-specified positivity-limited case |

All four share a negative-control outcome (`icu_acquired_uti`), asserted to show no effect
under any adjustment condition. Effects are reported as risk difference, active arm minus
comparator, in percentage points throughout.

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/Alirezaken/ctfm-icu.git
cd ctfm-icu
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The core pipeline (data preparation, estimation, diagnostics) needs only pandas, pyarrow,
numpy, pyyaml, lightgbm, scikit-learn, and scipy. The one GPU stage (`extract_embeddings`)
additionally needs `torch`, `transformers`, and `pillow` (commented out in
`requirements.txt` — install on the GPU node only).

### 2. Configuration

`config.yaml` is the single source of truth; `main.py` reads nothing else. Machine-specific
paths live under `paths:` and support `${ENV_VAR:default}` interpolation:

```yaml
paths:
  storage_root: ${CTFM_STORAGE:/path/to/ctfm_storage}   # results, embeddings, checkpoints
  datasets_dir: ${CTFM_DATASETS:/path/to/Datasets}       # raw MIMIC-IV / MIMIC-CXR / eICU
  data_dir:     ${CTFM_DATA:/path/to/data}               # link layer + event stream
```

Everything below the `paths:` block is the study design — interventions, arms, RCT
references, adjustment conditions, estimator settings — and is shared across machines.

### 3. Data access

MIMIC-IV, MIMIC-IV-ED, MIMIC-IV-Note, and MIMIC-CXR are credentialed PhysioNet datasets and
are not redistributed here; eICU-CRD (external validation) is likewise credentialed. Obtain
access through PhysioNet, then point `datasets_dir` at your local copy. None of this data is
committed to the repository (see `.gitignore`).

## Pipeline

Stages run through `main.py`, the only entry point, in the dependency order below, and
checkpoint and resume automatically (they run under SLURM and get preempted). Every
per-intervention stage runs all four interventions unless `--intervention` narrows it:

```bash
python main.py --list           # show every stage
python main.py --stage all      # run the full pipeline, every intervention
```

### Data preparation

```bash
python main.py --stage link
python main.py --stage emulate
```

`link` verifies the patient/admission/ICU/chest-X-ray link layer built once by `pipeline/`.
`emulate` builds the four target-trial cohorts — eligibility, exposure arms, time-zero,
censoring — into `manifests/cohorts.parquet`.

### Embeddings

```bash
python main.py --stage extract_embeddings
python main.py --stage extract_external
```

The only GPU stage. `extract_embeddings` computes the pre-time-zero image and text
embeddings for the cohort; `extract_external` computes embeddings for the external
informativeness gate. Both checkpoint per shard so a preempted SLURM job resumes rather than
restarts.

### Estimation

```bash
python main.py --stage estimate --intervention fluids_sepsis
python main.py --stage estimate
```

Cross-fitted AIPW/TMLE across all seven adjustment conditions, written to `effects.csv`.

### Validation

```bash
python main.py --stage synthesize
python main.py --stage diagnose
python main.py --stage robustness
python main.py --stage external
```

`synthesize` runs the semi-synthetic benchmark; `diagnose` computes the pre-estimation
diagnostic; `robustness` runs encoder and estimator swaps plus demographic subgroup checks;
`external` runs the eICU-CRD replication.

### Consolidation

```bash
python main.py --stage consolidate
python main.py --stage integrity
```

`consolidate` assembles every paired contrast into `contrasts.csv` and `manifest.csv`.
`integrity` asserts every validation gate below and fails loudly on violation.

## Data channels

| Channel | What it actually is |
|---|---|
| `images` | pre-time-zero frontal chest X-ray (RAD-DINO embedding) |
| `radtext` | pre-time-zero **radiology reports** — contemporaneous expert text |
| `histnote` | pre-time-zero **prior-admission discharge summaries** — MIMIC-IV-Note has no nursing or progress notes, so any pre-time-zero discharge summary is necessarily from an earlier hospitalization, not the current admission |

## Validation gates

The `integrity` stage asserts two pre-specified positive controls and fails the run loudly if
either breaks — a null result is only informative if the pipeline is shown capable of
detecting the thing it failed to find:

1. **Real:** structured-only adjustment must reduce bias relative to the naive comparison
   wherever overlap is adequate.
2. **Synthetic:** the structured-plus-image condition must recover the known planted effect
   when the confounder is built into the image by construction.

## File overview

- `config.yaml` — study design and paths; the single source of truth, read by nothing else.
- `main.py` — the only entry point; dispatches to a stage by name.
- `pipeline/build_link_layer.py`, `pipeline/build_event_stream.py` — one-time data
  preparation: patient/admission/ICU/chest-X-ray linkage and the unified clinical event
  stream.
- `src/cfg.py` — config loader with environment-variable interpolation.
- `src/events.py` — event-stream, link-layer, and manifest readers.
- `src/features.py` — structured covariates, the expert confounder set, censoring, and
  embedding pooling.
- `src/estimator.py` — cross-fitted AIPW/TMLE, IPCW, and overlap (ATO) weighting.
- `src/diagnostic.py` — the pre-estimation diagnostic (incremental-confounding index,
  treatment/outcome discrimination).
- `src/synthetic.py` — the semi-synthetic benchmark: real covariates, simulated
  treatment/outcome, known effect.
- `src/stats.py` — bootstrap, paired bootstrap, permutation tests, and FDR correction; the
  one definition of each statistic used everywhere else.
- `src/results.py` — the seven output table schemas.
- `src/stages/` — the eleven pipeline stages, in dependency order.

## Outputs

Every stage writes under `paths.storage_root/results`, never beside the code: `effects.csv`,
`diagnostics.csv`, `contrasts.csv`, `cohorts.csv`, `synthetic.csv`, `robustness.csv`,
`manifest.csv`.

## Acknowledgments

Study design and methodology by Soroosh Tayebi Arasteh.

## License

MIT — see [LICENSE](LICENSE).
