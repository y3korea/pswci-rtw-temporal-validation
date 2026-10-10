"""Configuration: predictors, tasks, constants and paths (revision v2)."""
from __future__ import annotations

import os

SEED = 2026
N_JOBS = int(os.environ.get("RTW_N_JOBS", "4"))     # cap parallelism (shared machine)
B_BOOT = int(os.environ.get("RTW_B_BOOT", "1000"))  # bootstrap resamples for C3 metrics
B_SHIFT = int(os.environ.get("RTW_B_SHIFT", "200"))  # bootstrap resamples with re-estimated domain classifiers
R_UPDATE = int(os.environ.get("RTW_R_UPDATE", "50"))  # repetitions for updating experiments
B_SHAP = int(os.environ.get("RTW_B_SHAP", "50"))
FAST = os.environ.get("RTW_FAST", "0") == "1"       # debug mode: tiny searches (never used for reported results)

NUM_VARS = [
    "female", "edu", "married", "hh_size", "disease", "commuting", "care_dur", "tenure",
    "svc_any", "svc_job", "firm_size", "union", "ex_days", "regular_ex", "work_ability",
    "work_ability_postcare", "chronic", "phys_limitation", "sensory_impair", "adl_help", "adl_mental", "adl_indoor",
    "adl_outdoor", "adl_work", "pain_freq", "pain_interf", "smoker", "drinker",
]
# ordinal scores that ridge-OH encodes as categories (levels fixed a priori)
ORDINAL_LEVELS = {
    "edu": [1, 2, 3, 4, 5], "hh_size": [1, 2, 3, 4, 5, 6], "care_dur": [1, 2, 3, 4, 5, 6],
    "tenure": list(range(1, 15)), "firm_size": [1, 2, 3, 4, 5, 6], "ex_days": list(range(0, 8)),
    "work_ability": list(range(0, 11)), "work_ability_postcare": list(range(0, 11)), "adl_help": [1, 2, 3, 4],
    "adl_mental": [1, 2, 3, 4, 5], "adl_indoor": [1, 2, 3, 4, 5], "adl_outdoor": [1, 2, 3, 4, 5],
    "adl_work": [1, 2, 3, 4, 5], "pain_freq": [1, 2, 3, 4, 5], "pain_interf": [1, 2, 3, 4],
}
NOM_LEVELS = {
    "age_grp": [1, 2, 3, 4],
    "job_status": [1, 2, 3, 4],
    "industry": ["construction", "manufacturing", "other", "transport"],
    "disab_cat": [1, 2, 3, 4, 5, 6],
}
EMP_VARS = ["employed", "emp6"]
PREDICTORS = NUM_VARS + list(NOM_LEVELS) + EMP_VARS          # 34 original variables
RESTRICTED_DROP = ["svc_any", "drinker", "adl_mental", "adl_indoor", "adl_outdoor", "adl_work",
                   "sensory_impair", "phys_limitation"]
RESTRICTED_DROP_B = RESTRICTED_DROP + ["svc_job", "tenure"]   # v2: + definition change (svc_job) and C1 tenure anomaly
SMALL_VARS = ["age_grp", "disab_cat", "employed", "work_ability", "pain_interf"]   # parsimonious LR

VAR_LABELS = {
    "female": "Female", "age_grp": "Age group", "edu": "Education", "married": "Married",
    "hh_size": "Household size", "disease": "Occupational disease", "commuting": "Commuting injury",
    "care_dur": "Duration of medical care", "tenure": "Job tenure at injury",
    "disab_cat": "Disability grade category", "svc_any": "Any rehabilitation service",
    "svc_job": "Vocational/employment-support service", "firm_size": "Firm size", "union": "Union at workplace",
    "job_status": "Job status at injury", "industry": "Industry", "employed": "Employed (wave 1)",
    "emp6": "Employment type (wave 1)", "ex_days": "Exercise days/week", "regular_ex": "Regular exercise",
    "work_ability": "Work ability (0-10)", "work_ability_postcare": "Work ability after care (recall)",
    "chronic": "Chronic disease", "phys_limitation": "Physical limitation",
    "sensory_impair": "Sensory impairment", "adl_help": "Need for help in ADL",
    "adl_mental": "Difficulty: mental activities", "adl_indoor": "Difficulty: indoor activities",
    "adl_outdoor": "Difficulty: outdoor activities", "adl_work": "Difficulty: work activities",
    "pain_freq": "Pain frequency", "pain_interf": "Pain interference", "smoker": "Current smoker",
    "drinker": "Current drinker",
}

# tasks: A1 retention (employed w1), A2 re-entry (not employed w1), P pooled (secondary)
TASKS = {"A1": "Job retention (employed at wave 1)", "A2": "Re-entry (not employed at wave 1)",
         "P": "Pooled employment (secondary)"}
# undesired event per task (target of an RTW-support flag): y = employed at wave 2, event = 1 - y
EVENT_LABEL = {"A1": "job loss", "A2": "no re-entry", "P": "non-employment"}
FAIR_THRESHOLD = {"A1": 0.10, "A2": 0.60, "P": 0.30}     # development prevalence of the undesired event (rounded)
DCA_THRESHOLDS = {"A1": [0.05, 0.10, 0.20, 0.30], "A2": [0.40, 0.50, 0.60, 0.70], "P": [0.10, 0.20, 0.30, 0.40]}

CANDIDATES = ["LR", "RidgeOH", "RF", "XGB", "LGBM", "CatBoost", "EBM"]
BASELINES = {"A1": ["Null", "LR-small"], "A2": ["Null", "LR-small"], "P": ["Null", "Status", "LR-small"]}
MODEL_LABEL = {"Null": "Null (development prevalence)", "Status": "Status-only LR", "LR-small": "Parsimonious LR",
               "LR": "LR (linear scores)", "RidgeOH": "Ridge LR (one-hot)", "RF": "Random forest", "XGB": "XGBoost",
               "LGBM": "LightGBM", "CatBoost": "CatBoost", "EBM": "EBM"}


def paths():
    data_root = os.environ.get(
        "DATA_ROOT", "/content/drive/MyDrive/2026_fall semester/Dataset/industrial_pannel")
    out_dir = os.environ.get("RTW_OUT_DIR", os.path.join(os.getcwd(), "rtw_outputs"))
    cache_dir = os.environ.get("RTW_CACHE_DIR", os.path.join(os.getcwd(), "rtw_cache"))
    for d in (out_dir, cache_dir, os.path.join(out_dir, "tables"), os.path.join(out_dir, "figures")):
        os.makedirs(d, exist_ok=True)
    return data_root, out_dir, cache_dir
