# RCT-Anchored Audit of Multimodal Causal Adjustment in the ICU

**Thesis:** a validated embedding is not a valid confounder proxy. Foundation-model image
and text embeddings are highly informative about the confounders they are meant to proxy,
yet they reduce confounding bias by zero, because the information is already in the
structured record. Meanwhile they destroy positivity. **Overlap, not proxy richness, is the
binding constraint on EHR causal inference.**

## What this is
Four ICU target trials anchored to real RCTs (PROSEVA, CLOVERS/CLASSIC, STARRT-AKI, TRISS),
seven adjustment conditions, cross-fitted doubly-robust estimation, and a pre-estimation
diagnostic that predicts whether any candidate modality will help — validated against both
RCT ground truth and a semi-synthetic benchmark with known effects.

## Layout
```
config.yaml          the study design. SINGLE SOURCE OF TRUTH.
main.py              the only entry point
pipeline/            data prep: link layer + event stream (run once)
src/
  cfg.py             config loader
  events.py          event-stream, link-layer and manifest readers
  features.py        structured covariates, expert set, censoring, embedding pooling
  estimator.py       cross-fitted AIPW/TMLE, NESTED reduction, IPCW, overlap weights (ATO)
  diagnostic.py      THE CONTRIBUTION: incremental confounding (dAUC_A, dAUC_Y, ICI)
  synthetic.py       semi-synthetic benchmark: real data, simulated A/Y, known tau
  stats.py           one definition of every statistic; Kind A/B/C kept separate
  results.py         the 7 output schemas
  stages/            the 11 pipeline stages, in dependency order
```

## Run
```bash
python main.py --list
python main.py --stage all          # every stage, all four interventions
```
Every per-intervention stage runs **all four** interventions by default.

## The three modalities, named for what they are
| Channel | Reality |
|---|---|
| `images` | pre-t0 frontal CXR (RAD-DINO) |
| `radtext` | pre-t0 **radiology reports** — contemporaneous expert text |
| `histnote` | pre-t0 **prior-admission discharge summaries** — MIMIC-IV-Note has no nursing/progress notes, so a discharge summary before t0 is necessarily from an earlier hospitalization |

## Outputs
`effects.csv`, `diagnostics.csv`, `contrasts.csv`, `cohorts.csv`, `synthetic.csv`,
`robustness.csv`, `manifest.csv` — under `paths.storage_root/results`.

## Positive controls (both asserted in `integrity`; the run fails loudly if either breaks)
1. **Real:** `structured` must reduce bias vs `naive` where overlap is adequate.
2. **Synthetic:** `struct_img` must recover a known tau when the confounder is planted in
   the image by construction.

A null from a pipeline not shown capable of detecting the thing it failed to find is worthless.
