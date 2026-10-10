"""Configuration: predictors, constants and paths for the temporal-validation study."""
from __future__ import annotations

import os

SEED = 2026
N_JOBS = int(os.environ.get("RTW_N_JOBS", "4"))     # cap parallelism (shared machine)
B_BOOT = int(os.environ.get("RTW_B_BOOT", "1000"))  # bootstrap resamples for test metrics
R_UPDATE = int(os.environ.get("RTW_R_UPDATE", "50"))  # repetitions for updating learning curves
B_SHAP = int(os.environ.get("RTW_B_SHAP", "100"))   # bootstrap refits for SHAP-rank stability

# ---- predictors (wave 1) -------------------------------------------------------------
# numeric / binary / ordinal scores
NUM_VARS = [
    "female", "edu", "married", "hh_size", "disease", "commuting", "care_dur", "tenure",
    "svc_any", "svc_job", "firm_size", "union", "ex_days", "regular_ex", "work_ability",
    "work_ability_postcare", "chronic", "phys_limitation", "sensory_impair", "adl_help", "adl_mental", "adl_indoor",
    "adl_outdoor", "adl_work", "pain_freq", "pain_interf", "smoker", "drinker",
]
# nominal -> one-hot with first level as reference (levels fixed a priori, not learned from data)
NOM_LEVELS = {
    "age_grp": [1, 2, 3, 4],
    "job_status": [1, 2, 3, 4],
    "industry": ["construction", "manufacturing", "other", "transport"],
    "disab_cat": [1, 2, 3, 4, 5, 6],
}
# employment status at wave 1: employed + nested emp6 indicators (see analysis plan, section 3)
EMP_VARS = ["employed", "emp6"]
PREDICTORS = NUM_VARS + list(NOM_LEVELS) + EMP_VARS          # 34 original variables
EXCLUDED_NONCOMPARABLE = ["poor_health", "fully_recovered", "preinj_poor_health"]
RESTRICTED_DROP = ["svc_any", "drinker", "adl_mental", "adl_indoor", "adl_outdoor", "adl_work",
                   "sensory_impair", "phys_limitation"]
LR5_VARS = ["age_grp", "disab_cat", "employed", "work_ability", "pain_interf"]

VAR_LABELS = {
    "female": "Female", "age_grp": "Age group", "edu": "Education", "married": "Married",
    "hh_size": "Household size", "disease": "Occupational disease", "commuting": "Commuting injury",
    "care_dur": "Duration of medical care", "tenure": "Job tenure at injury",
    "disab_cat": "Disability grade category", "svc_any": "Any rehabilitation service",
    "svc_job": "Vocational rehabilitation service", "firm_size": "Firm size", "union": "Union at workplace",
    "job_status": "Job status at injury", "industry": "Industry", "employed": "Employed (wave 1)",
    "emp6": "Employment type (wave 1)", "ex_days": "Exercise days/week", "regular_ex": "Regular exercise",
    "work_ability": "Work ability (0-10)", "work_ability_postcare": "Work ability after care (recall)",
    "chronic": "Chronic disease", "phys_limitation": "Physical limitation (walk/stairs/lift)",
    "sensory_impair": "Sensory impairment", "adl_help": "Need for help in ADL",
    "adl_mental": "Difficulty: mental activities", "adl_indoor": "Difficulty: indoor activities",
    "adl_outdoor": "Difficulty: outdoor activities", "adl_work": "Difficulty: work activities",
    "pain_freq": "Pain frequency", "pain_interf": "Pain interference", "smoker": "Current smoker",
    "drinker": "Current drinker",
}

MODEL_ORDER = ["Persistence", "LR-5", "LR", "EN", "RF", "XGB", "LGBM"]
ML_MODELS = ["LR", "EN", "RF", "XGB", "LGBM"]
TREE_MODELS = ["RF", "XGB", "LGBM"]
TASKS = {"A": "Employment at wave 2", "B": "Re-entry among non-employed at wave 1"}


def paths():
    """Resolve DATA_ROOT / OUT_DIR / CACHE_DIR from environment variables."""
    data_root = os.environ.get(
        "DATA_ROOT", "/content/drive/MyDrive/2026_fall semester/Dataset/industrial_pannel")
    out_dir = os.environ.get("RTW_OUT_DIR", os.path.join(os.getcwd(), "rtw_outputs"))
    cache_dir = os.environ.get("RTW_CACHE_DIR", os.path.join(os.getcwd(), "rtw_cache"))
    for d in (out_dir, cache_dir, os.path.join(out_dir, "tables"), os.path.join(out_dir, "figures")):
        os.makedirs(d, exist_ok=True)
    return data_root, out_dir, cache_dir
