# Foundation-model embeddings predict the confounder but do not reduce confounding bias in critical care

## Overview

Foundation-model embeddings of chest X-rays and clinical text are increasingly proposed as
confounder proxies for causal inference on EHR data — the intuition being that a richer,
validated representation of a patient's state should adjust away more bias than the
structured record alone. This repository audits that intuition directly, using target-trial
emulation anchored to four completed RCTs (PROSEVA, CLOVERS/CLASSIC, STARRT-AKI, TRISS) as
ground truth.

**Finding:** a validated embedding is not a valid confounder proxy. The image and text
embeddings (RAD-DINO, Clinical-Longformer) are highly informative about the confounders they
are meant to proxy — externally replicated at AUROC 76–94 — yet adding them to the adjustment
set reduces confounding bias by zero (incremental confounding index −0.23 to +1.30, all near
zero) beyond what the structured EHR record already provides, because the confounding
information is redundant with, not additional to, the structured record. For the
positivity-constrained intervention, adding the image also destroys effective sample size. A
semi-synthetic benchmark with a known planted effect confirms the mechanism is redundancy,
not weak signal: the same pipeline **does** recover a known effect when the confounder is
planted in the image by construction. **Overlap, not proxy richness, is the binding
constraint on EHR causal inference.**

## Method

- **Target-trial emulation**, four ICU interventions, each anchored to a published RCT effect
  size for validation against ground truth rather than internal consistency alone.
- **Cross-fitted doubly-robust estimation** (AIPW, with TMLE as a robustness check),
  LightGBM nuisance models, IPCW for informative censoring, and overlap (ATO) weights.
- **Seven adjustment conditions** per intervention, spanning naive, structured-only, and
  structured-plus-each-embedding, isolating exactly what each modality adds on top of the
  structured record.
- **A pre-estimation diagnostic** (`diagnose`) — the incremental confounding index (ICI) and
  ΔAUC on treatment/outcome — that predicts whether a candidate modality can possibly help
  *before* paying the estimation cost, rather than only reading it off the final effect.
- **A semi-synthetic benchmark** (`synthesize`) with simulated treatment/outcome and a known
  true effect, run through the identical pipeline, to distinguish "no signal" from "no
  effect" as an explanation for a null.
- **External replication** on eICU-CRD (structured, and structured+notes) as an independent
  cohort check, and on PadChest / ChestX-ray14 / CheXpert as an image-embedding
  informativeness quality gate (never enters the causal models).

## Target trials

| Intervention | RCT anchor | RCT effect (RD, 95% CI) | Horizon | Role |
|---|---|---|---|---|
| `fluids_sepsis` | CLOVERS / CLASSIC (meta) | −0.6 pp [−3.4, 2.3] | 90d | Primary null calibration — best n, best overlap |
| `transfusion_threshold` | TRISS (TRICC as sensitivity) | −1.9 pp [−8.1, 4.2] | 90d | Null calibration — key confounder (Hb) is structured, so imaging must not help |
| `rrt_timing` | STARRT-AKI | +0.2 pp [−0.3, 0.4] | 90d | Null calibration — tests whether "urgency" information lives in text |
| `prone_positioning` | PROSEVA | −16.8 pp [−24.5, −9.1] | 28d | Positive control; pre-specified positivity-failure case |

All four share a negative-control outcome (`icu_acquired_uti`) asserted to show no effect
under any adjustment condition, and risk difference is reported as active-arm-minus-comparator
in percentage points throughout (never mixed with p-values or performance metrics — see
`src/stats.py`).

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/Alirezaken/ctfm-icu.git
cd ctfm-icu
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The core pipeline (data prep, estimation, diagnostics) needs only pandas, pyarrow, numpy,
pyyaml, lightgbm, scikit-learn, and scipy. The one GPU stage (`extract_embeddings`) additionally
needs `torch`, `transformers`, and `pillow` (commented out in `requirements.txt` — install on
the GPU node only).

### 2. Configuration

`config.yaml` is the single source of truth; `main.py` reads nothing else. Machine-specific
paths live under `paths:` and support `${ENV_VAR:default}` interpolation:

```yaml
paths:
  storage_root: ${CTFM_STORAGE:/path/to/ctfm_storage}   # results, embeddings, checkpoints
  datasets_dir: ${CTFM_DATASETS:/path/to/Datasets}       # raw MIMIC-IV / MIMIC-CXR / eICU
  data_dir:     ${CTFM_DATA:/path/to/data}               # link layer + event stream
```

Everything below the `paths:` block is the study design (interventions, arms, RCT references,
adjustment conditions, estimator settings) and is shared across machines.

### 3. Data access

MIMIC-IV, MIMIC-IV-ED, MIMIC-IV-Note, and MIMIC-CXR are credentialed PhysioNet datasets and
are not redistributed here; eICU-CRD (external validation) is likewise credentialed. Obtain
access through PhysioNet, then point `datasets_dir` at your local copy. None of this data is
committed to the repository (see `.gitignore`).

## Pipeline

Stages run in dependency order through `main.py`, the only entry point, and checkpoint/resume
automatically (they run under SLURM and get preempted):

```bash
python main.py --list                                  # show all stages
python main.py --stage <name>                           # run one stage, all interventions
python main.py --stage estimate --intervention fluids_sepsis   # narrow to one intervention
python main.py --stage all                              # the full pipeline, every intervention
```

| Order | Stage | Produces |
|---|---|---|
| 1 | `link` | verifies the patient/admission/ICU/CXR link layer |
| 2 | `emulate` | the four target-trial cohorts → `manifests/cohorts.parquet` |
| 3 | `extract_embeddings` | GPU stage: image, radiology-text, and history-note embeddings |
| 4 | `extract_external` | embeddings for the external CXR informativeness gate |
| 5 | `estimate` | cross-fitted AIPW/TMLE across all seven adjustment conditions → `effects.csv` |
| 6 | `synthesize` | the semi-synthetic benchmark with known effect → `synthetic.csv` |
| 7 | `diagnose` | incremental confounding index and ΔAUC → `diagnostics.csv` |
| 8 | `robustness` | encoder/estimator swaps and demographic subgroups → `robustness.csv` |
| 9 | `external` | eICU-CRD structured replication → `effects.csv` |
| 10 | `consolidate` | every paired contrast → `contrasts.csv`, `manifest.csv` |
| 11 | `integrity` | asserts every validation gate; fails loudly on violation |

## Data channels

| Channel | What it actually is |
|---|---|
| `images` | pre-t0 frontal chest X-ray (RAD-DINO embedding) |
| `radtext` | pre-t0 **radiology reports** — contemporaneous expert text |
| `histnote` | pre-t0 **prior-admission discharge summaries** — MIMIC-IV-Note has no nursing/progress notes, so any pre-t0 discharge summary is necessarily from an earlier hospitalization, not the current admission |

## Outputs

All results are written under `paths.storage_root/results`, never beside the code:
`effects.csv`, `diagnostics.csv`, `contrasts.csv`, `cohorts.csv`, `synthetic.csv`,
`robustness.csv`, `manifest.csv`.

## Validation gates

The `integrity` stage asserts two pre-specified positive controls and fails the run loudly if
either breaks — a null is only informative if the pipeline is shown capable of detecting the
thing it failed to find:

1. **Real:** `structured` adjustment must reduce bias relative to `naive` wherever overlap is
   adequate.
2. **Synthetic:** `struct_img` must recover the known planted effect when the confounder is
   built into the image by construction.

## Repository layout

```
config.yaml          study design and paths. Single source of truth.
main.py              the only entry point
pipeline/            data prep: link layer + event stream (run once)
src/
  cfg.py             config loader
  events.py          event-stream, link-layer, and manifest readers
  features.py        structured covariates, expert confounder set, censoring, embedding pooling
  estimator.py       cross-fitted AIPW/TMLE, NESTED reduction, IPCW, overlap (ATO) weights
  diagnostic.py      incremental confounding index (ICI), ΔAUC on treatment and outcome
  synthetic.py       semi-synthetic benchmark: real data, simulated treatment/outcome, known effect
  stats.py           one definition per statistic; performance, causal-effect, and p-value
                      kinds are kept separate and never mixed
  results.py         the seven output schemas
  stages/            the eleven pipeline stages, in dependency order
```

## Acknowledgments

Study design and methodology by Soroosh Tayebi Arasteh.

## License

MIT — see [LICENSE](LICENSE).
