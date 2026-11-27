# Temporal validation, shift diagnosis and recalibration of return-to-work prediction models (PSWCI, 2013–2024)

Analysis code for the manuscript *"Temporal Validation, Shift Diagnosis and Recalibration of Return-to-Work
Prediction Models in Three National Workers' Compensation Cohorts"* (Wansuk Choi, Miran Goo, Seoa Park, Yusung Jang, Myeong-Chul Park (corresponding author); accepted for publication
in the Journal of The Korea Society of Computer and Information, Vol. 31, No. 12, December 2026).

## Aim
Models that predict employment one year ahead are developed on injured workers from PSWCI cohorts 1–2 (wave 1 in
2013 and 2018) and validated, without re-tuning, on cohort 3 (wave 1 in 2023). Two co-primary tasks: job retention
(A1, workers employed at wave 1) and re-entry (A2, workers not employed at wave 1); the pooled task is secondary and
benchmarked against a status-only model. Performance is reported in the analysed sample and, with the wave-2
survey weights (longitudinal in cohorts 2–3; the single wave-2 weight wt02 in cohort 1), in the population from
which each cohort was sampled; temporal-validation estimates use the cohort-3 weights. The code reproduces:

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
  secondary designs and sensitivity analyses (inverse probability of response weighting, restricted predictors,
  commuting injuries, work ability, weighted
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
outputs. Data version used: cohort 1, 5th-wave integrated long file V4 (`PSWCI_5TH_03_LONG_DATA_V4.sav`; V4 release
of 2018-12-05, which corrected work ability for two respondents); cohort 2, `pswci2_05_long` (5th-wave integrated
long file, 2022 survey; waves 1–5); cohort 3, 2024 integrated long file of waves 1–2
(`(4. 통합 데이터_long) 2024 PSWCI_3차 코호트 1-2차년도`);
downloaded from <https://www.pswci.or.kr/Data/Download.aspx> on 17 August 2026.
Place the three downloaded cohort folders (as distributed) under one directory, `DATA_ROOT`:

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
python src/export_models.py "$RTW_CACHE_DIR" models   # model files: no person-level data, no per-leaf counts
```
The exported files are committed in `models/` (file-name codes: RidgeOH = ridge-OH, LR-small = parsimonious LR,
Status = status-only LR).
`RTW_STAGES` runs a subset of stages (describe, evaluate, secondary, shift, iwrecal, lagged, withinc3, fairness,
attribution, extras, sensitivity, report); development models are cached. `src/pswci_harmonize.py` rebuilds the harmonised long
file from the raw SPSS files (set `RTW_USE_PARQUET=1` to read a previously built
`DATA_ROOT/00_derived_data/pswci_harmonized_long.parquet` instead). `notebooks/rtw_temporal_validation_colab.ipynb`
embeds the same code for Google Colab (outputs cleared).

Revision 1 (journal review, 2026-11-13): `python src/rtw_revision1.py` (same environment variables; set `RTW_N_JOBS=1`
on machines without multiprocessing) refits CatBoost and the EBM with the saved hyperparameters of
`tables/tuning_primary.csv` (no re-tuning), checks that the refitted C3 AUROC and CITL equal the values of the main analysis
(`refit_check.csv`) and writes the paired C3 comparison (`paired_catboost_ebm.csv`, Supplementary Table S28).
Supplementary Fig. S9 (survey-weighted calibration curves) is drawn from `tables/calcurve_*.csv` by
`python src/make_fig_s9.py "$RTW_OUT_DIR/tables" figS9.png` (no refitting).
Main-text Fig. 1 re-plots, at column width, the model-free prevalence rows of `tables/shift_disde.csv` (the rows of
Supplementary Table S15 and Fig. S3b): `python src/make_main_fig1.py "$RTW_OUT_DIR/tables" fig1.png`.

### Which output produces which table or figure
| Article item | Aggregate output (in `RTW_OUT_DIR/tables`) |
|---|---|
| Table 2, Supplementary Tables S6–S9 | `perf_C3_<task>_unweighted.csv`, `perf_C3_<task>_weighted.csv`, `perf_C3_diffs_*` |
| Fig. 1, Table S15 | `shift_disde.csv`, `shift_disde_restricted.csv` |
| Table S3 | `tuning_primary.csv` |
| Tables S10–S11, Figs. S1–S2 | `calibration_change_*`, `dca_*`, `calcurve_*` |
| Tables S13–S18 | `shift_*.csv`, `update_iw_recalibration.csv` |
| Tables S19–S20, Figs. S4–S5 | `update_lagged_*.csv`, `update_withinC3_*.csv` |
| Tables S21–S24 | `secondary_designs.csv`, `sensitivity_*.csv`, `seed_variability_*.csv`, `attribution_stability_*.csv` |
| Table S28, Fig. S9 | `revision1/tables/paired_catboost_ebm.csv` and `revision1/tables/refit_check.csv` (under `RTW_OUT_DIR`), `calcurve_*.csv` |

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
| `src/rtw_revision1.py` | revision 1: paired C3 comparison of CatBoost and the EBM (Table S28) |
| `src/make_fig_s9.py` | revision 1: survey-weighted calibration curves (Fig. S9) from the saved curve tables |
| `src/make_main_fig1.py` | main-text Fig. 1 from the model-free prevalence rows of `shift_disde.csv` |
| `models/` | exported model files written by `src/export_models.py` (no person-level data, no per-leaf counts) |
| `notebooks/` | Colab notebook (code identical to `src/`, no outputs) |
| `docs/analysis_plan.md` | analysis plan with dated amendments |

No output table contains a cell with 1–4 workers: prediction-histogram counts below 5 are suppressed
(`predhist_*.csv`), and the smallest subgroup in the fairness audit has n = 85.

## Licence, citation and AI use
Code: MIT licence (see `LICENSE`). Please cite the article (see `CITATION.cff`).
Generative AI (Anthropic Claude: Opus 5.5, claude-opus-5-5, and Fable 5.1, claude-fable-5-1; for the final check of
revision 1, the corrections that followed and the language editing of the revised manuscript also Fable 5.2,
claude-fable-5-2) proposed the research
questions and analyses, wrote and ran this code (which produced all numerical results), searched and screened the
literature, conducted an internal pre-submission review with AI-simulated reviewers, carried out internal consistency
audits of data, code, text and references, whose findings led to corrections, and drafted and condensed the
manuscript text; the authors verified the analyses, numbers and references, revised the text and take full
responsibility for the content.
