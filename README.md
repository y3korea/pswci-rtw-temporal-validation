# Temporal validation, shift diagnosis and recalibration of return-to-work prediction models (PSWCI, 2013–2024)

Analysis code for the manuscript *"Temporal Validation, Shift Diagnosis and Recalibration of Return-to-Work
Prediction Models in Three National Workers' Compensation Cohorts"* (Wansuk Choi, Miran Goo, Seoa Park, Yusung Jang, Myeong-Chul Park (corresponding author); prepared for submission
to the Journal of the Korea Society of Computer and Information).

## Aim
Models that predict employment one year ahead are developed on injured workers from PSWCI cohorts 1–2 (wave 1 in
2013 and 2018) and validated, without re-tuning, on cohort 3 (wave 1 in 2023). Two co-primary tasks: job retention
(A1, workers employed at wave 1) and re-entry (A2, workers not employed at wave 1); the pooled task is secondary and
benchmarked against a status-only model. Performance is reported in the analysed sample and, with the wave-2
longitudinal weights, in the insured population. The code reproduces:

- development and tuning: LR, ridge LR with one-hot ordinal predictors (ridge-OH), random forest, XGBoost, LightGBM,
  CatBoost, explainable boosting machine (EBM), plus null, status-only and parsimonious baselines; primary model by
  repeated cross-validated log-loss;
- temporal-validation metrics with a design-based bootstrap (AUROC, AUPRC, Brier, CITL, slope, ICI/E50/E90,
  calibration curves with bands, decision curves against the best default strategy), within-stratum performance of
  the pooled task, Riley sample-size criteria, attrition;
- shift diagnosis by source cohort: adversarial validation, conditional cohort effects, an outcome-in-domain-classifier
  test, a DISDE-type decomposition (X-shift vs Y|X shift on a shared support), case-mix expected AUROC and
  importance-weighting truncation sensitivity;
- updating: label-free importance-weighted recalibration (the method of Park et al., AISTATS 2020, with Platt-type
  scaling; evaluated, not proposed), lagged labelled updating (C1 model, C2 update samples,
  C3 evaluation) with closed testing, and a within-C3 design;
- exploratory fairness audit (TPR/FPR/PPV/NPV, calibration-implied TPR, BH q-values), attribution stability,
  secondary designs and sensitivity analyses (IPCW, restricted predictors, commuting injuries, work ability, weighted
  training, 20 seeds).

The analysis plan, with dated amendments, is in `docs/analysis_plan.md`. It was written before outcome modelling but
was not time-stamped or registered; a debug run computed cohort-3 results before the first full run, and all v2
analyses are post hoc (see the plan's "Status of pre-specification").

## Data access
The PSWCI microdata (산재보험패널조사) are public-use data released free of charge by the Labor Welfare Research
Institute of the Korea Workers' Compensation & Welfare Service (근로복지공단 근로복지연구원). Anyone can download them
from <https://www.pswci.or.kr/Data/Download.aspx> (button "산재보험패널 데이터 신청하기": online application with
identity verification, then immediate download) after accepting the conditions of use: research purposes only (no
commercial use), no lending or transfer to third parties, citation of the source ("Panel Study of Workers'
Compensation Insurance, Korea Workers' Compensation & Welfare Service Labor Welfare Research Institute"), and
notification of resulting theses, papers or presentations to the institute (pswci@comwel.or.kr). Because of the
no-transfer condition the raw files are not included here; this repository contains no data and no person-level
outputs. Place the three downloaded cohort folders (as distributed) under one directory, `DATA_ROOT`:

```
DATA_ROOT/
  1차 코호트 1~5차년도 산재보험패널조사 데이터 및 코드북/   (contains *LONG_DATA*.sav)
  2차 코호트 1~5차년도 산재보험패널조사 데이터 및 코드북/   (contains pswci2_05_long*.sav)
  3차 코호트 1~2차년도 산재보험패널조사 데이터 및 코드북/   (contains *(4. 통합 데이터_long)*.sav)
```

No person-level data or row-level predictions are stored in this repository. `.gitignore` excludes raw and derived
data files with bracket patterns (e.g. `*.[sS][aA][vV]`), so the match does not depend on `core.ignorecase`.

## Reproduce
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt                  # includes catboost 1.2.10 and interpret-core 0.7.8
export DATA_ROOT="/path/to/industrial_pannel"   # folder with the three cohort folders
export RTW_OUT_DIR="$PWD/rtw_outputs"            # aggregated tables (CSV) and figures (PNG/PDF)
export RTW_CACHE_DIR="$PWD/rtw_cache"            # fitted models / row-level predictions: never share
export JOBLIB_TEMP_FOLDER="$PWD/joblib_tmp"      # avoids loss of joblib memmaps in system temp folders
python src/run_all.py                            # about 1 hour on 4 CPU cores (Apple silicon), n_jobs = 4
python src/export_models.py "$RTW_CACHE_DIR" "$RTW_OUT_DIR/models"   # model files: no person-level data, no per-leaf counts
```
`RTW_STAGES` runs a subset of stages (describe, evaluate, secondary, shift, iwrecal, lagged, withinc3, fairness,
extras, sensitivity, report); development models are cached. `src/pswci_harmonize.py` rebuilds the harmonised long
file from the raw SPSS files (set `RTW_USE_PARQUET=1` to read a previously built
`DATA_ROOT/00_derived_data/pswci_harmonized_long.parquet` instead). A Google Colab notebook that embeds the same code
is provided with the manuscript materials.

## Code layout
| File | Content |
|---|---|
| `src/pswci_harmonize.py` | harmonisation of cohorts 1–3 (shared project script) |
| `src/rtw_config.py` | predictors, tasks, thresholds, constants, environment paths |
| `src/rtw_data.py` | analytic data set, task definitions (A1, A2, pooled), design matrices |
| `src/rtw_models.py` | model pipelines, search spaces, tuning, OOF predictions, attributions |
| `src/rtw_develop.py` | development, repeated-CV model selection, prediction in the test cohort |
| `src/rtw_metrics.py` | metrics, ICI, design-based bootstrap, decision curves, calibration curves |
| `src/rtw_shift.py` | adversarial validation, cohort effects, outcome-in-domain test, DISDE-type decomposition, IW sensitivity |
| `src/rtw_update.py` | IW recalibration, lagged and within-C3 updating, closed testing |
| `src/rtw_fair.py` | exploratory fairness audit |
| `src/rtw_analyses.py` | descriptives, sample size, evaluation, attrition, within-strata, sensitivity analyses |
| `src/rtw_figures.py`, `src/rtw_tables.py` | figures and formatted tables from the aggregated CSVs |
| `src/run_all.py` | end-to-end pipeline |
| `src/export_models.py` | export of fitted model parameters (TRIPOD+AI item 22) |

Small cells (n < 5) are suppressed in subgroup tables.

## Licence, citation and AI use
Code: MIT licence (see `LICENSE`). Please cite the article (see `CITATION.cff`).
Generative AI (Anthropic Claude: Opus 5.5, claude-opus-5-5, and Fable 5.1, claude-fable-5-1) proposed the analyses and wrote and ran this code; the authors are
responsible for reviewing and verifying it. [Authors: update this statement after your own code review.]
