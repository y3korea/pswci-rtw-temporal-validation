"""Analytic data set and design matrices (wave-1 predictors -> wave-2 employment)."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from rtw_config import NOM_LEVELS, NUM_VARS, PREDICTORS


def load_long(data_root: str, use_parquet: bool | None = None) -> pd.DataFrame:
    """Load the harmonized long file; build it from raw SPSS files if no parquet is used."""
    pq = os.path.join(data_root, "00_derived_data", "pswci_harmonized_long.parquet")
    if use_parquet is None:
        use_parquet = os.environ.get("RTW_USE_PARQUET", "0") == "1"
    if use_parquet and os.path.exists(pq):
        return pd.read_parquet(pq)
    from pswci_harmonize import build_harmonized
    return build_harmonized(data_root)


def build_analytic(long: pd.DataFrame) -> pd.DataFrame:
    """One row per wave-1 participant: wave-1 predictors, wave-2 participation/outcome, weights."""
    w1 = long[(long["wave"] == 1) & (long["participated"] == 1)].set_index("uid")
    w2 = long[long["wave"] == 2].set_index("uid")
    a = w1[["cohort", "pid", "wt_long", "wt_cross", "accident_type", "disab_grade"] + PREDICTORS].copy()
    a = a.rename(columns={"wt_long": "wt_long_w1", "wt_cross": "wt_cross_w1"})
    a["part2"] = w2["participated"].reindex(a.index).fillna(0).astype(int)
    a["y"] = w2["employed"].reindex(a.index)               # employed at wave 2 (emp3 == 1)
    a["wt_w2"] = w2["wt_long"].reindex(a.index)            # longitudinal weight at wave 2
    a["hh_size"] = a["hh_size"].clip(upper=6)
    # fairness / descriptive groupings
    a["sex_grp"] = np.where(a["female"] == 1, "Female", "Male")
    a["age_lab"] = a["age_grp"].map({1: "<=39", 2: "40-49", 3: "50-59", 4: ">=60"})
    a["disab_grp"] = np.select([a["disab_cat"] == 6, a["disab_cat"].isin([3, 4, 5]), a["disab_cat"].isin([1, 2])],
                               ["No grade", "Mild (8-14)", "Severe (1-7)"], default="NA")
    a["acc_grp"] = np.where(a["disease"] == 1, "Disease", "Injury")
    a["job_grp"] = np.where(a["job_status"] == 1, "Regular", "Non-regular")
    return a


def task_frame(a: pd.DataFrame, task: str) -> pd.DataFrame:
    """Eligible rows for a task (participated at wave 2; task B: not employed at wave 1)."""
    d = a[(a["part2"] == 1) & a["y"].notna()].copy()
    if task == "B":
        d = d[d["employed"] == 0].copy()
    d["y"] = d["y"].astype(int)
    return d


def design_matrix(d: pd.DataFrame, task: str, variables: list[str] | None = None):
    """Return (X, groups) where groups maps each design column to its original variable.

    One-hot levels are fixed a priori (no data-driven learning), so this step cannot leak
    information between development and validation sets. Imputation/scaling happen inside
    the sklearn pipelines and are fitted on development data only.
    """
    variables = list(variables or PREDICTORS)
    cols, groups = {}, {}
    for v in NUM_VARS:
        if v in variables:
            cols[v] = d[v].astype(float)
            groups[v] = v
    for v, levels in NOM_LEVELS.items():
        if v in variables:
            for lev in levels[1:]:
                name = f"{v}={lev}"
                cols[name] = (d[v] == lev).astype(float).where(d[v].notna())
                groups[name] = v
    if "employed" in variables or "emp6" in variables:
        e6 = d["emp6"]
        if task == "A":
            if "employed" in variables:
                cols["employed"] = d["employed"].astype(float)
                groups["employed"] = "employed"
            if "emp6" in variables:
                cols["emp6=new_employer"] = (e6 == 2).astype(float)
                cols["emp6=self_employed"] = e6.isin([3, 4]).astype(float)
                groups["emp6=new_employer"] = groups["emp6=self_employed"] = "emp6"
        if "emp6" in variables:
            cols["emp6=job_seeking"] = (e6 == 5).astype(float)
            groups["emp6=job_seeking"] = "emp6"
    X = pd.DataFrame(cols, index=d.index)
    return X, groups


def lr5_vars(task: str) -> list[str]:
    from rtw_config import LR5_VARS
    return [v for v in LR5_VARS if not (task == "B" and v == "employed")]
