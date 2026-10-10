# Temporal external validation of return-to-work prediction models (PSWCI, 2013–2024)

Analysis code for the manuscript *"Temporal External Validation of Machine Learning Models for Predicting Return to
Work After Occupational Injury: Dataset Shift, Calibration Drift, and Fairness Across Three National Cohorts
(2013–2024)"* ([Author 1], [Author 2]; submitted to the Journal of the Korea Society of Computer and Information).

## Aim
Models that predict employment one year ahead are trained on injured workers from PSWCI cohorts 1–2
(wave 1 in 2013 and 2018) and validated, without any re-tuning, on cohort 3 (wave 1 in 2023). The code
reproduces: development and tuning (logistic regression, elastic net, random forest, XGBoost, LightGBM, a
persistence baseline and a 5-variable logistic baseline); temporal-validation metrics with bootstrap CIs
(AUROC, AUPRC, Brier, calibration-in-the-large, calibration slope, ECE, calibration curves, decision curves);
dataset-shift diagnostics (PSI/SMD, adversarial validation, transition probabilities, importance-weighting
attribution); model-updating learning curves; subgroup fairness; SHAP-importance stability; secondary
designs (chronological chain, leave-one-cohort-out) and sensitivity analyses (survey weights, IPCW for
attrition, comparability-restricted predictors, commuting injuries, work-ability reference period).
The analysis plan is in `docs/analysis_plan.md` (written before outcome modelling but not time-stamped or registered; see its dated Amendments section).

## Data access (not included)
The Panel Study of Workers' Compensation Insurance (PSWCI, 산재보험패널조사) microdata are owned by the
Korea Workers' Compensation & Welfare Service Research Institute (근로복지공단 근로복지연구원) and **may not be
redistributed**. Request them from the official PSWCI site: <https://www.pswci.or.kr/Data/Download.aspx>
("데이터 신청하기"; research use only, no transfer to third parties; users are asked to notify the institute
of resulting publications). Place the three cohort folders (as distributed) under one directory, `DATA_ROOT`:

```
DATA_ROOT/
  1차 코호트 1~5차년도 산재보험패널조사 데이터 및 코드북/   (contains *LONG_DATA*.sav)
  2차 코호트 1~5차년도 산재보험패널조사 데이터 및 코드북/   (contains pswci2_05_long*.sav)
  3차 코호트 1~2차년도 산재보험패널조사 데이터 및 코드북/   (contains *(4. 통합 데이터_long)*.sav)
```

No person-level data, fitted models or row-level predictions are stored in this repository.

## Reproduce
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export DATA_ROOT="/path/to/industrial_pannel"   # folder with the three cohort folders
export RTW_OUT_DIR="$PWD/rtw_outputs"            # aggregated tables (CSV) and figures (PNG/PDF)
export RTW_CACHE_DIR="$PWD/rtw_cache"            # fitted models / row-level predictions: never share
python src/run_all.py                            # ~30-45 min on 4 CPU cores
```
`src/pswci_harmonize.py` rebuilds the harmonized long file from the raw SPSS files (set `RTW_USE_PARQUET=1`
to read a previously built `DATA_ROOT/00_derived_data/pswci_harmonized_long.parquet` instead).
A Google Colab notebook that embeds the same code is provided with the manuscript materials.

## Code layout
| File | Content |
|---|---|
| `src/pswci_harmonize.py` | harmonization of cohorts 1–3 (shared project script, v2) |
| `src/rtw_config.py` | predictors, constants, environment paths |
| `src/rtw_data.py` | analytic data set, task definitions, design matrices |
| `src/rtw_models.py` | model pipelines, tuning, OOF predictions, TreeSHAP |
| `src/rtw_develop.py` | development in C1+C2 and prediction in the test cohort |
| `src/rtw_metrics.py` | metrics, bootstrap CIs, decision curves, calibration curves |
| `src/rtw_analyses.py` | all pre-specified analyses (writes aggregated CSV tables) |
| `src/rtw_figures.py` | figures from the aggregated tables |
| `src/run_all.py` | end-to-end pipeline |

Small cells (n < 5) are suppressed in subgroup tables.

## Licence and citation
Code: MIT licence (see `LICENSE`). Please cite the article (see `CITATION.cff`).
Generative AI (Anthropic Claude) was used to draft the analysis code, to search and screen the literature, and to
draft and edit the manuscript text. The authors reviewed and verified all analyses, code, references and text and
take full responsibility for the content.
