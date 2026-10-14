# Pre-specified analysis plan — KSCI_P1 Temporal validation of RTW prediction models

Version 1.1 — v1.0 written 2026-10-10 before any outcome model was fitted; v1.1 (same day) incorporates
the shared-data correction v2 (see "Amendment v1.1" at the end) **before any validation-cohort (C3) prediction
was computed**.
Status at the time of writing: the harmonized file was inspected for sample sizes,
missingness and wave-1 predictor distributions by cohort, and the persistence accuracies /
re-entry counts given in the brief were reproduced (C1 0.809, C2 0.797, C3 0.832;
re-entry 261/535, 402/1,130, 357/1,232). No model involving the wave-2 outcome had been fitted.
Any later deviation is listed in the dated Amendments section with its reason and reported in the manuscript.

## 1. Data and unit of analysis
* Source: `00_derived_data/pswci_harmonized_long.parquet` (re-built from raw SPSS files in the Colab notebook by `pswci_harmonize.py`).
* Unit: wave-1 participants who also participated at wave 2 (C1 n = 1,803; C2 n = 2,965; C3 n = 3,341).
  Wave 1 = 2013 (C1), 2018 (C2), 2023 (C3); wave 2 = one year later.
* Development set = C1 + C2 pooled (n = 4,768). Temporal (external) validation set = C3 (n = 3,341). C3 is not used for any modelling decision.

## 2. Outcomes (tasks)
* **Task A (primary): employment** — employed at wave 2 (`emp3 == 1`), all eligible participants.
* **Task B (co-primary): re-entry** — employed at wave 2 among those not employed at wave 1
  (dev: 535 + 1,130 = 1,665, 663 events; C3: 1,232, 357 events).
* All probabilities are modelled as P(employed at wave 2). For decision-curve and fairness analyses the
  *intervention target* is the complementary event (Task A: not employed at wave 2; Task B: not re-entered),
  i.e., a case manager would flag a worker for intensive return-to-work support when the predicted risk
  of non-employment, 1 − p̂, exceeds a threshold p_t.

## 3. Predictors (wave-1, harmonized, comparable across cohorts)
female, age_grp, edu, married, hh_size, disease, commuting, care_dur, tenure, disab_cat, svc_any, svc_job,
firm_size, union, job_status, industry, employed, emp6, ex_days, regular_ex, work_ability,
work_ability_postcare, chronic, phys_limitation, sensory_impair, adl_help, adl_mental, adl_indoor,
adl_outdoor, adl_work, pain_freq, pain_interf, smoker, drinker (34 variables; v1.0 listed `limitation`, which
the v2 data correction showed to be sensory impairment — replaced by the correctly labelled pair).
Excluded by design: poor_health, fully_recovered, preinj_poor_health (scales not comparable across cohorts).

Encoding (fitted on development data only):
* Nominal → one-hot with first level as reference: age_grp (ref ≤39), job_status (ref regular), industry (ref construction — alphabetical first), disab_cat (ref grade 1–3, first level).
* `employed` and `emp6` are collinear (employed ≡ emp6 ∈ {1–4}). They are encoded as **employed_w1 plus three nested indicators**: re-employed with a new employer (emp6 = 2), self-employed/unpaid family worker (emp6 ∈ {3,4}) [reference among employed: original job]; and job-seeking unemployed (emp6 = 5) [reference among non-employed: not in labour force].
  Task B (all non-employed at wave 1): employed_w1 and the two employed-only indicators are constant and dropped; the job-seeking indicator is kept.
* All other variables numeric (ordinal scales kept as scores). hh_size truncated at 6.
* Missing values (only firm_size, union; < 2%): median imputation + missing-indicator columns (indicators are created for variables with missing values in the development data).
* Standardisation (z-score) for the two logistic models.

## 4. Models
Candidate models (all use the full predictor set):
1. LR — unpenalised maximum-likelihood logistic regression.
2. EN — elastic-net logistic regression (saga), grid: C ∈ 10^{−3…1} (9 values) × l1_ratio ∈ {0.1, 0.5, 0.9, 1.0}.
3. RF — random forest, 500 trees; random search (n_iter = 20): max_features ∈ {sqrt, 0.2, 0.33, 0.5}, min_samples_leaf ∈ {1, 5, 10, 20, 40}, max_depth ∈ {None, 6, 10, 14}.
4. XGB — XGBoost; random search (n_iter = 40): n_estimators 100–1000, learning_rate log-uniform 0.01–0.2, max_depth 2–6, min_child_weight 1–20, subsample 0.6–1.0, colsample_bytree 0.5–1.0, reg_lambda log-uniform 0.1–20.
5. LGBM — LightGBM; random search (n_iter = 40): n_estimators 100–1000, learning_rate 0.01–0.2, num_leaves 4–48, min_child_samples 10–120, subsample 0.6–1.0 (bagging_freq 1), colsample_bytree 0.5–1.0, reg_lambda 0–20.

Tuning: 5-fold cross-validation inside the development set only, folds stratified by cohort × outcome, seed 2026; selection criterion = **CV log-loss** (a proper scoring rule, chosen to favour calibrated models). The tuned configuration is refitted on the whole development set. Out-of-fold (OOF) predictions with the selected configuration on the same folds give internal-validation performance.

**Primary model** for downstream analyses (fairness, updating): the candidate with the lowest development CV log-loss (chosen separately for Task A and Task B). **SHAP model**: the tree model (RF/XGB/LGBM) with the lowest CV log-loss.

Mandatory baselines:
* (a) **Persistence rule** (Task A): predicted wave-2 status = wave-1 status (deterministic). Reported as accuracy, sensitivity/specificity, and as a strategy in decision curves. **Persistence model**: logistic regression on employed_w1 only (probabilistic version; AUROC identical to the rule) — gives Brier and calibration. For Task B the persistence analogue predicts "no re-entry" for everyone (null model; intercept-only logistic fitted on development data).
* (b) **5-variable logistic model**: age_grp, disab_cat, employed_w1, work_ability, pain_interf (Task B: the same without employed_w1 → 4 variables).

## 5. Performance in the temporal validation cohort (C3)
Metrics for every model: AUROC; AUPRC for the minority class of each task (Task A: non-employment; Task B: re-entry) with the prevalence as no-skill reference; Brier score; scaled Brier = 1 − Brier / [ȳ(1 − ȳ)] (ȳ = observed C3 prevalence); calibration-in-the-large (CITL = intercept α of logit P(y) = α + logit p̂, offset); calibration slope (β of logit P(y) = a + β·logit p̂); ECE (10 equal-frequency bins); mean predicted vs observed; accuracy at p̂ = 0.5 (to compare with persistence).
Uncertainty: 1,000 non-parametric bootstrap resamples of C3 (models fixed), percentile 95% CIs. Paired bootstrap CIs for differences vs the persistence model and vs the 5-variable model (ΔAUROC, ΔBrier).
Calibration plots: LOWESS smoother (frac = 2/3) with decile points and histogram of predictions.
Decision-curve analysis: net benefit for flagging workers at risk of non-employment, thresholds p_t = 0.05–0.80, for each model, treat-all, treat-none and the persistence rule (Task A); bootstrap CIs for ΔNB (primary model − persistence rule) at p_t = 0.2, 0.3, 0.4, 0.5.
Internal (development OOF) performance reported alongside for comparison.

## 6. Dataset-shift diagnostics (Task A population; Task B prevalence also reported)
* Feature shift: standardized mean difference (SMD, dev vs C3; categorical: per level, maximum |SMD| reported) and population stability index (PSI; categories as bins; ε = 1e−4) per predictor. Conventional PSI reading: < 0.10 small, 0.10–0.25 moderate, > 0.25 large.
* Adversarial validation: LightGBM domain classifier (fixed parameters: 300 trees, learning_rate 0.05, num_leaves 15, min_child_samples 40) distinguishing dev vs C3 from the predictors; 5-fold stratified OOF AUROC with bootstrap CI; drivers = mean |SHAP| of the domain classifier. Reference contrast: C1 vs C2.
* Label / concept shift: outcome prevalence by cohort; transition probabilities P(employed w2 | employed w1) and P(employed w2 | not employed w1) by cohort with Wilson 95% CIs.
* Covariate-shift attribution via importance weighting: a logistic domain classifier (same preprocessing; cross-fitted, 5 folds) estimates p(C3 | x); density-ratio weights for development rows w = [p/(1 − p)] × (n_dev / n_C3), truncated at the 99th percentile and normalised to mean 1; Kish effective sample size reported. The development OOF predictions of each model are evaluated (i) unweighted and (ii) importance-weighted; observed C3 performance is (iii). Covariate-shift component = (ii) − (i); total change = (iii) − (i); residual (label/concept shift + other) = (iii) − (ii). Bootstrap CIs (1,000) by resampling dev and C3 rows with weights held fixed. Also: covariate-shift-predicted C3 prevalence (weighted dev outcome mean) vs observed C3 prevalence.

## 7. Model updating in C3 (learning curves)
Repeated R = 50 times: split C3 at random 50/50 (stratified by outcome) into an *update pool* and a *held-out evaluation half*; from the pool draw random subsamples of n = 100, 250, 500, 1000 (Task B: n = 100, 250, 500 because its pool is ~616); apply to the primary model:
(0) no update; (1) intercept recalibration (logit offset); (2) logistic recalibration (intercept + slope); (3) refit on the C3 subsample only (same model class and hyperparameters); (4) pooled refit on development + C3 subsample (same hyperparameters).
Evaluate on the held-out half: AUROC, Brier, CITL, calibration slope, ECE; summarise mean and 2.5th–97.5th percentiles across repetitions.

## 8. Fairness (Task A, C3, primary model and full LR; Task B primary model in supplement)
Subgroups: sex (ref male); age group (ref ≤39); disability (none = no grade; mild = grades 8–14 [ref, largest]; severe = grades 1–7); accident type (occupational disease vs injury [accident or commuting], ref injury); job status at injury (regular [ref] vs non-regular = temporary, daily, self-employed).
Metrics per subgroup: n, prevalence, AUROC, CITL, observed/expected ratio, TPR and FPR of the flag "risk of non-employment ≥ p_t". **Pre-specified p_t = 0.30** (≈ development prevalence of non-employment, 27%, rounded up: flags workers whose risk exceeds the base rate and implies accepting ≈2.3 unnecessary referrals per correctly identified worker, a plausible trade-off for low-cost RTW counselling). TPR = proportion of truly non-employed workers who are flagged (equal opportunity), FPR = proportion of employed workers who are flagged.
Gaps (group − reference) with 1,000 bootstrap 95% CIs (resampling within C3). Development OOF subgroup metrics reported for comparison (were gaps already present before deployment?). Sensitivity: p_t = 0.50.

## 9. Explainability and its stability
TreeSHAP for the SHAP model. Global importance = mean |SHAP| per original variable (one-hot / nested columns summed per row before taking absolute values). Compare (a) the development model on development data vs (b) the same specification refitted on all of C3 (SHAP on C3): Spearman ρ of importance across variables with a bootstrap CI (100 bootstrap refits of the C3 model), top-10 overlap. Also cohort-specific refits (C1, C2, C3 separately; pairwise ρ) and the development model's SHAP evaluated on C3 data (attribution shift with a fixed model).

## 10. Secondary validation designs
* Chronological chain: train C1 → test C2; train C1+C2 → test C3 (= primary).
* Leave-one-cohort-out: train C2+C3 → test C1; train C1+C3 → test C2; train C1+C2 → test C3.
For each training set all candidate models are re-tuned with the same CV procedure (reduced search: n_iter 20 for XGB/LGBM, 10 for RF) and evaluated with AUROC, Brier, CITL, slope (1,000 bootstrap CIs). Both tasks.

## 11. Sensitivity analyses
S1. Survey-weighted evaluation in C3 using the wave-2 longitudinal weight (`wt_long`): weighted AUROC, Brier, CITL, slope, ECE for all models.
S2. Weighted training (development rows weighted by wt_long normalised to mean 1 within cohort) + weighted evaluation — primary model and full LR.
S3. Attrition (IPCW): per cohort, logistic model of wave-2 participation on wave-1 predictors; stabilised inverse-probability weights truncated at the 99th percentile; applied to training (development) and evaluation (C3) — primary model and full LR.
S4. Comparability-restricted predictor set: drop svc_any (definition differs: no medical-rehabilitation flag in C2), drinker (unlabelled code in C1), adl_mental/adl_indoor/adl_outdoor/adl_work (C1 response distribution differs markedly from C2–C3, identified from descriptive distributions before outcome modelling), sensory_impair and phys_limitation (flagged "check comparability / use with caution" in the v2 harmonization notes; C3 levels markedly higher) — primary model and full LR refitted (primary hyperparameters) and validated in C3.
S5. Fairness at p_t = 0.50.
S6. Excluding commuting injuries from C3 evaluation (coverage absent in C1, began 2018).
S7. Work-ability reference period (v2 correction: C1 wave-1 work ability refers to ~90 days after care closure, not the interview date; C2/C3 = current). (a) Development on C2 only (all candidate models re-tuned with the primary search), validated in C3; (b) work_ability removed from the predictor set (primary model and full LR refitted with primary hyperparameters; the 5-variable baseline becomes 4 variables), validated in C3.

## Amendments (dated; the plan body above is otherwise unchanged)

### Amendment v1.1 — 2026-10-10, about 18:55 KST, BEFORE any cohort-3 prediction was computed
* Shared data correction v2: `limitation` was item 1 (sensory impairment). Predictor set now uses `phys_limitation` (walking/stairs/lifting) and `sensory_impair` (34 predictors).
* Added S7 (work-ability reference period in C1) and extended S4 with sensory_impair and phys_limitation.
* Disclosure: before the correction, a timing test of the tuning code was run on the development set only (CV log-loss, no C3 predictions, old predictor set). It informed no analytic choice; the search spaces and selection rule are unchanged.
* Sections 3 and 11 of the body were edited in place at this amendment. Original v1.0 wording: the predictor list contained `limitation` (33 variables); S4 dropped only svc_any, drinker and the four ADL difficulty items; S7 did not exist.

### Amendment v1.2 — 2026-10-10, about 19:50 KST, AFTER the results were seen: record of deviations (no analysis was changed in response to results)
1. **Task B fairness threshold.** Section 8 did not state a threshold for Task B. It was set by the same base-rate rule as for Task A (development prevalence of the target event, non-re-entry: 1 − 663/1,665 = 0.60) before the fairness results were inspected.
2. **SHAP stability benchmark (post hoc addition).** In addition to the pre-specified bootstrap refits (B = 100 for Task A; 20 for Task B to limit run time), a split-half benchmark was added: 50 repetitions comparing importance rankings between disjoint, equal-sized halves of C3, of the development set, and between them. Reason: to separate temporal instability from sampling noise when interpreting ρ.
3. **Updating summaries.** Section 7 specified the mean and 2.5th–97.5th percentiles. One configuration (Task B, refit on 100 C3 records) produced degenerate predictions with an extreme calibration slope in one repetition, which made the mean meaningless; learning curves and the manuscript therefore report the **median** (with the same percentiles) and, additionally, the share of repetitions in which an update improved on no update in the same held-out half. Means remain in `updating_summary_task*.csv`.
4. **Calibration slope of the Task B null model** (constant predictions) is undefined and not reported.
5. **Additional derived quantities** (not pre-specified, descriptive only): thresholds at which the decision curves of XGBoost and the persistence rule differ by < 0.01 (`key_numbers.csv`), and paired bootstrap differences versus the full LR (in addition to persistence and the 5-variable model).
6. **Secondary-design tuning sizes** were as planned (n_iter 20 XGB/LGBM, 10 RF; EN full grid); S7a used the full primary search.
No change was made to outcomes, predictors (after amendment v1.1), models, tuning rules, primary-model selection or test metrics after C3 results were seen.

### Amendment v2.0 — 2026-10-10, about 21:00 KST, AFTER an internal pre-submission review using AI-simulated reviewers (all analyses below are post hoc)
Written before the revised analyses were run; it records the revised design, not results. (Terminology note added about 22:00 KST the same day: the heading originally read "AFTER peer review ... reviewer-driven"; the review was an internal pre-submission review with AI-simulated reviewers, not journal peer review. No substantive change.) Every v2.0 analysis is
labelled post hoc in the manuscript. The v1 results are archived in `05_results/archive_v1/`.
* **Task structure.** (A1) job retention: employed at wave 2 among workers employed at wave 1; (A2) re-entry:
  employed at wave 2 among workers not employed at wave 1 (= former Task B). A1 and A2 are co-primary. The pooled
  task (former Task A) is secondary and is benchmarked against a status-only logistic model (wave-1 employment plus
  the nested emp6 indicators). A1 and A2 null benchmark: development prevalence.
* **Estimands.** Performance in the analysed sample (unweighted) and in the insured target population (weighted by the
  wave-2 longitudinal weight) are co-primary. Bootstrap CIs in C3 resample within the C3 sampling strata (design-based,
  1,000 resamples).
* **Models.** LR (linear scores, as v1), ridge LR with all ordinal predictors one-hot encoded (ridge-OH, C tuned),
  random forest, XGBoost, LightGBM (widened searches incl. depth-1 trees, design-specific random seeds), CatBoost and an
  explainable boosting machine (EBM). Elastic net is dropped (superseded by ridge-OH). TabPFN is not run (its model
  weights require a licensed download that the authors have not approved). Primary model per task = lowest mean
  log-loss over 2 x 5-fold repeated CV in the development set; ridge-OH is always reported as transparent comparator.
* **Metrics.** As v1 plus ICI/E50/E90 (spline-based logistic calibration curve), no-skill AUPRC, calibration curves with
  bootstrap bands; ECE retained in the supplement. Calibration slope is undefined (reported as such) when predictions
  are (near-)constant or the fit does not converge.
* **Decision curves.** Event = undesired outcome (A1 job loss, A2 no re-entry, pooled non-employment). Net benefit vs the
  best default strategy (flag all / flag none / status rule) at p_t = 0.10, 0.20, 0.30, 0.40 (A2 also 0.50, 0.60),
  unweighted and weighted.
* **Shift.** Cohort-specific designs C1->C3 and C2->C3 in addition to the pooled design; adversarial validation per
  cohort pair (full and comparability-restricted predictor sets); conditional cohort effects from a pooled logistic
  model with cohort indicators; test of whether the outcome improves a domain classifier given x; a DISDE-type
  decomposition (Cai, Namkoong & Yadlowsky) of the performance change into X-shift (source side), Y|X shift on the
  shared support and X-shift (target side), with the domain classifier re-estimated inside 200 bootstrap resamples;
  model-based (case-mix) expected AUROC; importance-weighting truncation sensitivity (95/99/99.9/none), ESS and balance.
  Terminology: "outcome-prevalence change" (not "label shift") unless P(x|y) invariance is examined.
* **Updating.** (i) Label-free importance-weighted logistic recalibration (Platt-type, fitted on source OOF
  predictions weighted by density ratios to the unlabelled target covariates) in pooled, C1->C3, C2->C3 and C1->C2
  designs; (ii) lagged labelled updating: model developed in C1, updated with random C2 samples (n = 100, 250, 500, 1,000,
  all), evaluated in C3 (no update, intercept, logistic, closed-testing choice among these, pooled refit, ridge refit on
  the update sample alone with CV-tuned penalty), 50 repetitions; (iii) the v1 within-C3 split design in the supplement
  with degenerate fits counted and a CV-tuned ridge refit added.
* **Fairness (exploratory).** A1 and A2 (and pooled in supplement), primary model and ridge-OH; thresholds = development
  prevalence of the undesired outcome (A1 0.10, A2 0.60, pooled 0.30); TPR, FPR, FNR, PPV, NPV, AUROC, CITL, O/E;
  calibration-implied TPR; Benjamini-Hochberg q-values; survey-weighted version.
* **Sample size.** Riley et al. (2020) development criteria and validation precision for A1, A2, pooled.
* **Sensitivity.** IPCW, comparability-restricted sets (v1 S4 and S4b = S4 + svc_job + tenure), excluding commuting,
  work ability removed, weighted training, XGBoost seed variability (20 seeds).
* **Note on Amendment v1.2.** Its statement that no analysis was changed in response to results cannot be verified externally, because the v1 debug run had already computed cohort-3 results (see Status).
* **Measurement.** Dependent interviewing of economic activity (D**001000) and of work ability is described; A2 outcome is
  collected with the independent "last week" questions, A1 outcome with the job-continuation question.

### Amendment v2.1 — 2026-10-11 (written AFTER the v2 results had been seen; descriptive additions and record of deviations)
* **Added after v2 results were seen (post hoc, descriptive).** (i) Performance of the pooled-task models within wave-1
  employment strata (`perf_C3_P_within_strata.csv`), so that the pooled AUROC is never reported without the
  within-stratum values; (ii) a wave-1 to wave-2 attrition table by cohort and wave-1 employment status, with counts of
  missing longitudinal weights and outcomes among wave-2 respondents (`attrition.csv`; both zero). Both were added
  after an independent check of the v2 draft, not because of their values.
* **Deviations of the v2 run from the v2.0 text.** Decision-curve thresholds were implemented per task (A1 0.05, 0.10,
  0.20, 0.30; A2 0.40, 0.50, 0.60, 0.70; pooled 0.10, 0.20, 0.30, 0.40; `rtw_config.DCA_THRESHOLDS`), centred on the
  development prevalence of the undesired event. Seed variability was run for the selected primary model (CatBoost in
  all tasks), not for XGBoost. Fairness for the pooled task is reported in the supplement only.
* **Run history.** The within-C3 updating stage of the final v2 run failed once (joblib BrokenProcessPool after its
  temporary memmap folder was removed) and was re-run with a dedicated JOBLIB_TEMP_FOLDER. The run was then paused by
  the user during the seed-variability stage (2026-10-10, about 22:12 KST); on 2026-10-11 the stages `extras`,
  `sensitivity` and `report` were re-run from the cached development models. Between the two parts only the new
  `extras` stage (within-strata, attrition), table formatting and the number-formatting precision were changed; no
  code used by the earlier stages was modified.
* **Terminology.** The plan's original title ("Pre-specified analysis plan") is kept as written on 2026-10-10. Because the plan was not time-stamped or registered and a debug run preceded the first full run, the manuscript describes it as "written before outcome modelling, not registered", not as pre-specified.
* **Primary-model choice.** Unchanged rule (lowest repeated-CV log-loss in the development data). CatBoost was selected
  in all three tasks with small margins (e.g. A2 0.5819 vs 0.5820 for the EBM); XGBoost gave nearly identical C3
  results. The manuscript reports this explicitly.

### Amendment v2.2 — 2026-10-11, about 07:15 KST (AFTER a second internal pre-submission review using AI-simulated reviewers; all results had been seen)
* **Status of timings (self-reported).** All v2 runs (2026-10-10 21:19, 21:42, 22:04; 2026-10-11 05:12) used code that
  was committed to the local git repository only afterwards (commit 7ea059d, 2026-10-11 05:35). The times given in
  Amendments v2.0 and v2.1 (for example, v2.0 written "before the revised analyses were run", and v2.1 "no code used by
  the earlier stages was modified") are therefore self-reported and cannot be checked against version control. An
  independent re-run of the committed code from the raw files reproduced every table (max |diff| < 1e-14).
* **Correction to Amendment v2.1.** v2.1 said that XGBoost gave "nearly identical C3 results" and that "the manuscript
  reports this explicitly". This holds for AUROC only, and the manuscript did not report it at that time. For re-entry,
  XGBoost was better calibrated in C3 than CatBoost (slope 1.00 vs 1.31; CITL −0.13 vs −0.23). The manuscript now
  reports this.
* **Descriptive additions (post hoc).** (i) Components of the repeated-CV log-loss (repeat 1 reuses the tuning folds;
  repeat 2 uses independent folds) and the model each rule would select (v1 single-CV rule: EBM for A1 and A2;
  independent repeat: CatBoost in every task) — `tuning_primary.csv`; (ii) survey-weight diagnostics (Kish ESS, weight
  ratio, single-unit strata) — `weights_design.csv`; (iii) prevalence DISDE decomposition with the domain classifier
  restricted to predictor set S4b — `shift_disde_restricted.csv`.
* **Relabelling and export (no change to any estimate).** Prevalence rows of `shift_disde.csv` are labelled
  "model-free" (they use observed outcomes only). Exported CatBoost models no longer contain per-leaf training counts
  (`leaf_weights`; some pooled-task leaves held < 5 records), the training-loss history or identifiers; linear models
  are exported as intercept and original-scale coefficients only. Scoring is unchanged (checked).
* **Interpretation (no new analysis).** Label-free importance-weighted recalibration is the method of Park et al.
  (AISTATS 2020) with Platt-type scaling; it is evaluated, not proposed. Algorithm 1 step 2(c) is conditional. Closed
  testing at n = 100–250 often declined needed updates; this is reported.

### Status of pre-specification
This plan was written (v1.0, about 18:46 KST) before any outcome model was fitted. A debug run of the full pipeline with reduced resampling (19:01-19:23 KST) computed cohort-3 results before the final v1 run; the code changes made afterwards are listed in Amendment v1.2 and run_log.txt. The plan was not time-stamped by an external service, committed to version control, or registered before the analysis; it was first committed to the local git repository after the analysis had been run.

## Software / reproducibility
Python 3.13.6, scikit-learn 1.6.1, xgboost 3.2.0, lightgbm 4.6.0, catboost 1.2.10, interpret-core 0.7.8, shap 0.48.0, statsmodels 0.14.6; seed 2026; n_jobs ≤ 4. Reporting follows TRIPOD+AI (2024).

### Amendment v2.3 — 2026-10-15 KST (consistency audit; all results had been seen; no model was refitted and no estimate quoted in the manuscript changed)
*Appended at the end of the file; Amendments v1.1–v2.2 above are unchanged.*
* **Correction of improvement shares (Tables S18–S19).** `rtw_update.summarise_updating` now counts an update as
  better only if it improves the metric by more than 1e-10. Earlier values counted floating-point ties as improvements
  (closed testing that keeps the model returns expit(logit(p)), which differs from p only by rounding noise, and
  intercept recalibration cannot change the calibration slope). The repetitions themselves are unchanged; the
  summaries were recomputed from the saved repetition files (`update_lagged_reps.csv`, `update_withinC3_reps.csv`,
  read with round-trip float parsing; the previous function reproduced the previous summaries byte for byte), so the
  result is identical to re-running the `lagged` and `withinc3` stages with the cached models. Only `share_better_*`
  columns changed (18 lagged rows, 14 within-C3 rows); medians, percentiles, closed-testing choices and the shares
  quoted in the manuscript are unchanged.
* **Small-cell suppression.** Prediction-histogram counts of 1–4 workers (`predhist_*.csv`, not used by any table,
  figure or text) are now suppressed in the code and were suppressed in the saved files (flag `suppressed_lt5`).
* **Outcome measurement (record; no analysis changed).** The "last week" economic-activity questions refer to the week
  before the interview in C1 and C2 (wave-2 outcome waves) and in C3 wave 1, but the C3 wave-2 (2024) questionnaire
  defines "last week" as the week containing 1 August (C3 wave-1 fieldwork: 21 August to 26 November 2023). The
  job-continuation question refers to the interview date. For C3 re-entry the outcome week can therefore fall less than
  a year after the wave-1 interview; this measurement change coincides with the validation cohort and is reported as a
  limitation. The earlier manuscript statement that employment was assessed for the week containing 1 August in all
  cohorts was wrong.
* **Missing values (record).** All missing predictor values are questionnaire skips for workers self-employed at injury
  (workplace union in C2 and C3; firm size for solo self-employed workers in C3); C1 asked both items of all workers.
  Development-median imputation therefore assigns these C3 workers firm size category 3 (10–29 workers) without an
  indicator (firm size was never missing in development), and the union missingness indicator marks self-employment in
  C2 and C3 only.
* **Commuting injuries (correction of the S6 rationale).** C1 records only work accident versus occupational disease,
  so commuting injuries cannot be identified in C1 and are coded 0 (a structural zero), rather than "coverage absent in
  C1"; C2 commuting injuries occurred under the pre-2018 rule (employer-controlled commuting only). Table S3 shows
  "n/a" for C1.
* **Labels and display (no change to any value).** The S3 attrition weights are inverse probability of response weights
  (IPRW; labelled "IPCW" in the code and CSV files). Formatted tables print model, design, component and cohort names
  as in the manuscript, U+2212 minus signs and thousands separators. Figures: Fig. 1 follows Algorithm 1; Fig. 4a shows
  CatBoost for the three C3 designs only (the C1 → C2 rows plotted XGBoost without saying so) on a log axis; other
  figure changes are layout and labels only (see run_log.txt, 2026-10-15).

### Addendum to Amendment v2.3 — 2026-10-15 KST (consistency audit, fix round 2; display only; no analysis, CSV file or value changed)

* **Formatted-table headers.** `rtw_tables.py` now prints readable headers for the remaining code-style columns of the
  tuning table (Table S2: log-loss of each repetition; model chosen by the single tuning-CV rule of v1, the repeated-CV
  rule of v2.0 and the independent repetition alone), the attrition table (Table S3), the Riley et al. sample-size table
  (Table S4: development and C3 n and minority-class events, candidate parameters, required n under each criterion,
  expected shrinkage, LR out-of-fold calibration slope, O/E 95% CI factors), the subgroup audit (Table S11:
  calibration-implied TPR, TPR/FPR/AUROC gaps with BH q values) and the importance-weighting table (Table S16: number of
  predictors with |SMD| > 0.1 before and after weighting).
* **Domain classifier.** The domain-classifier column of the DISDE information table and the importance-weighting table
  (Tables S14 and S16) is headed "Domain classifier" and shows "Logistic regression" and "LightGBM" instead of the codes
  LR and LGBM (the CSV files keep the codes).
* `tables_formatted.md` was regenerated with `rtw_tables.make_all`; only 7 header lines and 33 domain-classifier cells
  changed; `key_numbers.csv` is byte-identical. The manuscript numbers (numbers_trace) are unchanged.
