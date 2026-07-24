# Foundation model embeddings predict the confounder but struggle in reducing confounding bias in critical care

## Overview

This is the official repository of the paper **Foundation model embeddings predict the confounder but struggle in reducing confounding bias in critical care**.

Preprint version: [link to be added].

Foundation-model embeddings of chest X-rays and clinical notes are increasingly proposed as confounder proxies for causal adjustment in observational ICU studies, on the intuition that a richer, validated representation of a patient's clinical state should absorb more confounding than the structured record alone. This repository is the code for an audit of that intuition. It emulates four ICU target trials, each anchored to a completed randomized controlled trial (PROSEVA, CLOVERS/CLASSIC, STARRT-AKI, TRISS), so that adjustment quality can be checked against real trial evidence rather than against internal consistency alone. Each target trial is estimated under seven adjustment conditions — a naive comparison, a clinician-curated expert confounder set, adjustment on the full structured EHR record, adjustment on that record plus each of three candidate embeddings (a chest-X-ray encoder and two clinical-text encoders), and all three embeddings together — using cross-fitted, doubly robust estimation. A pre-estimation diagnostic and a semi-synthetic benchmark with a known, planted effect are built into the pipeline so that a null finding can be told apart from a pipeline that simply is not working. External replication on an independent multi-hospital ICU database, and an image-embedding quality gate on public chest-X-ray datasets, complete the validation. The findings are reported in the accompanying manuscript; this repository documents the method and reproduces the pipeline, not the findings.

## Encoder panel

Every encoder is frozen, open-weight, ungated, and applied locally only to extract embeddings; none is fine-tuned, and no closed or API model is used. Each embedding is pooled over items observed strictly before time zero within a fixed look-back window, so nothing measured after the treatment decision can enter the adjustment set. The three channels are named for what they actually are: the image channel reads the pixels of a frontal chest radiograph; the report channel is contemporaneous expert text; the history channel is a prior-admission discharge summary, because the note release contains no contemporaneous nursing or progress notes.

| Encoder | Identifier | Channel | Role |
|---|---|---|---|
| RAD-DINO | `microsoft/rad-dino` | Frontal chest X-ray (`images`) | Core image encoder |
| Clinical-Longformer | `yikuan8/Clinical-Longformer` | Radiology reports (`radtext`) | Core text encoder |
| Clinical-Longformer | `yikuan8/Clinical-Longformer` | Prior-admission discharge summaries (`histnote`) | Core text encoder |
| BiomedCLIP | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | Frontal chest X-ray (`images_alt`) | Encoder swap, robustness only |

The image-embedding quality gate additionally applies the same frozen RAD-DINO encoder to public chest-X-ray datasets (PadChest, ChestX-ray14, CheXpert). These never enter any causal model; their only job is to characterize what the image channel encodes.

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/Alirezaken/ctfm-icu.git
cd ctfm-icu
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The core pipeline (data preparation, estimation, diagnostics) uses pandas, PyArrow, NumPy, PyYAML, LightGBM, scikit-learn, and SciPy. The single GPU stage that extracts embeddings additionally needs PyTorch, Hugging Face Transformers, Pillow, and open_clip_torch (for the BiomedCLIP encoder swap); these are commented out in `requirements.txt` and installed on the GPU node only.

### 2. Configuration

`config.yaml` is the single source of truth; `main.py` reads nothing else. Machine-specific paths live under `paths:` and support `${ENV_VAR:default}` interpolation, so the same file is portable across machines without editing source:

```yaml
paths:
  storage_root: ${CTFM_STORAGE:/path/to/ctfm_storage}   # results, embeddings, manifests, checkpoints
  datasets_dir: ${CTFM_DATASETS:/path/to/Datasets}      # raw MIMIC-IV / MIMIC-CXR / eICU-CRD, read-only
  data_dir:     ${CTFM_DATA:/path/to/data}              # link layer + event stream
```

Everything below the `paths:` block is the study design — interventions, arms, RCT references, adjustment conditions, estimator settings, and the diagnostic thresholds — and is shared across machines.

> **Data access and third-party compliance (your responsibility).** MIMIC-IV, MIMIC-IV-ED, MIMIC-IV-Note, and MIMIC-CXR are credentialed PhysioNet datasets; eICU-CRD (external replication) is likewise credentialed; the public chest-X-ray datasets (PadChest, ChestX-ray14, CheXpert) are obtained separately under their own terms. This repository redistributes no dataset and no model weights; it points to public sources and loads checkpoints and data you obtain yourself. Before using any dataset, model, or service referenced here, YOU are responsible for reviewing and complying with its license, terms of use, data-use agreement, and any applicable privacy, ethics, and regulatory requirements for your jurisdiction and intended use. Point `datasets_dir` at your local copy once access is granted; none of this data is committed to the repository (see `.gitignore`).

## Pipeline

The study runs as a sequence of stages exposed through `main.py`, the single entry point, in a fixed dependency order. Each stage reads from the shared config and writes intermediate results that the next stage consumes, so a run can be interrupted and resumed. List every stage, then run one stage or the whole pipeline:

```bash
python main.py --list           # show every stage, in dependency order
python main.py --stage all      # run the full pipeline, every intervention
```

Every per-intervention stage runs all four target trials by default; pass `--intervention` to narrow it to one of `fluids_sepsis`, `transfusion_threshold`, `rrt_timing`, or `prone_positioning`. Pass `--force` to ignore existing checkpoints.

**Resume caveat.** Stages checkpoint per unit of work and skip units already finished on restart, because these jobs sit in a SLURM queue and get preempted. Resume cannot detect code changes: if you edit a stage, re-run it with `--force`, or delete its checkpoint markers, so it recomputes rather than skipping finished units and keeping stale numbers.

### Data preparation (CPU)

The relational link layer and the unified event stream are built once from the raw PhysioNet tables by the scripts in `pipeline/`; the `link` stage then verifies they are present before the study proper begins.

```bash
python main.py --stage link       # verify the link layer and event stream are present and complete
python main.py --stage emulate    # build the four target-trial cohorts: eligibility, arms, time zero, censoring
```

### Extract embeddings (GPU)

```bash
python main.py --stage extract_embeddings   # image and text embeddings for the cohort
python main.py --stage extract_external     # RAD-DINO embeddings for the image-quality gate
```

The only GPU stages. Both checkpoint per shard, so a preempted job resumes rather than restarts.

### Estimation and validation (CPU)

```bash
python main.py --stage estimate      # the seven adjustment conditions per target trial
python main.py --stage synthesize    # the semi-synthetic benchmark with a known, planted effect
python main.py --stage diagnose      # the pre-estimation incremental-confounding diagnostic
python main.py --stage robustness    # encoder / estimator / window / pooling / reduction / trim swaps and subgroups
python main.py --stage external      # structured-only replication on the external ICU database
```

### Consolidation and integrity (CPU)

```bash
python main.py --stage consolidate   # assemble every paired contrast, with p-values and BH-FDR
python main.py --stage integrity     # assert every invariant and positive control; fails loudly on violation
```

## File overview

- `config.yaml` — the single source of truth: machine paths plus the entire study design (interventions, adjustment ladder, estimator, diagnostic thresholds, semi-synthetic sweep, robustness swaps). Read only through `src/cfg.py`.
- `main.py` — the single entry point; dispatches to a stage by name and enforces the dependency order.
- `pipeline/build_link_layer.py`, `pipeline/build_event_stream.py`, `pipeline/clean_event_stream.py`, `pipeline/qc_event_stream.py` — one-time raw-data preparation: the relational link layer, the unified event stream, its value-cleaning pass, and the quality-control report.
- `src/cfg.py` — config loader with `${ENV}` interpolation and fail-loud checks for missing values.
- `src/events.py` — readers over the cleaned event stream, the link layer, and the consolidated manifests.
- `src/features.py` — structured covariates at time zero, the expert confounder set, the censoring indicator, the negative-control outcome, and pooled embedding proxies.
- `src/estimator.py` — cross-fitted, doubly robust estimation with nested embedding reduction, censoring weights, and overlap weighting.
- `src/diagnostic.py` — the pre-estimation incremental-confounding diagnostic (the paper's methodological contribution).
- `src/synthetic.py` — the semi-synthetic benchmark: real covariates and embeddings, simulated treatment and outcome, and a known effect.
- `src/stats.py` — bootstrap, paired cluster bootstrap, permutation tests, and Benjamini-Hochberg correction; the one definition of every statistic used elsewhere.
- `src/results.py` — the fixed result-table schemas and the single writer that enforces them, so structure cannot drift silently.
- `src/util.py` — logging and filesystem checkpoint/resume markers.
- `src/stages/` — the eleven pipeline stages, one module each, in dependency order.

## Citation

If you use this repository, please cite our paper:

```bibtex
@misc{sharafiyan2026foundation,
  title  = {Foundation model embeddings predict the confounder but struggle in reducing confounding bias in critical care},
  author = {Sharafiyan, Alireza and Kuhl, Christiane and Alhaskir, Mohamed and Bienzeisler, Jonas and Maier, Andreas and Kather, Jakob Nikolas and Nebelung, Sven and Truhn, Daniel and Tayebi Arasteh, Soroosh},
  year   = {2026},
  note   = {Preprint}
}
```

## License

MIT License. See `LICENSE` for details.
