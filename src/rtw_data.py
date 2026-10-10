"""Analytic data set and design matrices (wave-1 predictors -> wave-2 employment), revision v2."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from rtw_config import NOM_LEVELS, NUM_VARS, ORDINAL_LEVELS, PREDICTORS, SMALL_VARS


def load_long(data_root: str, use_parquet: bool | None = None) -> pd.DataFrame:
    pq = os.path.join(data_root, "00_derived_data", "pswci_harmonized_long.parquet")
    if use_parquet is None:
        use_parquet = os.environ.get("RTW_USE_PARQUET", "0") == "1"
    if use_parquet and os.path.exists(pq):
        return pd.read_parquet(pq)
    from pswci_harmonize import build_harmonized
    return build_harmonized(data_root)


def build_analytic(long: pd.DataFrame) -> pd.DataFrame:
    """One row per wave-1 participant: wave-1 predictors, wave-2 participation/outcome, weights, strata."""
    w1 = long[(long["wave"] == 1) & (long["participated"] == 1)].set_index("uid")
    w2 = long[long["wave"] == 2].set_index("uid")
    a = w1[["cohort", "pid", "wt_long", "accident_type", "disab_grade", "strata"] + PREDICTORS].copy()
    a = a.rename(columns={"wt_long": "wt_long_w1"})
    a["part2"] = w2["participated"].reindex(a.index).fillna(0).astype(int)
    a["y"] = w2["employed"].reindex(a.index)               # employed at wave 2 (emp3 == 1)
    a["wt_w2"] = w2["wt_long"].reindex(a.index)            # longitudinal weight at wave 2
    a["hh_size"] = a["hh_size"].clip(upper=6)
    a["sex_grp"] = np.where(a["female"] == 1, "Female", "Male")
    a["age_lab"] = a["age_grp"].map({1: "<=39", 2: "40-49", 3: "50-59", 4: ">=60"})
    a["disab_grp"] = np.select([a["disab_cat"] == 6, a["disab_cat"].isin([3, 4, 5]), a["disab_cat"].isin([1, 2])],
                               ["No grade", "Mild (8-14)", "Severe (1-7)"], default="NA")
    a["acc_grp"] = np.where(a["disease"] == 1, "Disease", "Injury")
    a["job_grp"] = np.where(a["job_status"] == 1, "Regular", "Non-regular")
    return a


def task_frame(a: pd.DataFrame, task: str) -> pd.DataFrame:
    """Eligible rows: wave-2 respondents; A1 employed at wave 1; A2 not employed at wave 1; P all."""
    d = a[(a["part2"] == 1) & a["y"].notna()].copy()
    if task in ("A1", "B1"):
        d = d[d["employed"] == 1].copy()
    elif task in ("A2", "B"):
        d = d[d["employed"] == 0].copy()
    d["y"] = d["y"].astype(int)
    return d


def design_matrix(d: pd.DataFrame, task: str, variables: list[str] | None = None, ordinal_onehot: bool = False):
    """Return (X, groups): design columns and the original variable of each column.

    Levels of all one-hot encodings are fixed a priori (no data-driven learning; no leakage).
    ordinal_onehot=True encodes the ordinal scores of ORDINAL_LEVELS as categories (ridge-OH model).
    Employment status at wave 1 is encoded as employed + nested emp6 indicators; constant columns of a task dropped.
    """
    variables = list(variables or PREDICTORS)
    cols, groups = {}, {}
    for v in NUM_VARS:
        if v not in variables:
            continue
        if ordinal_onehot and v in ORDINAL_LEVELS:
            levels = ORDINAL_LEVELS[v]
            for lev in levels[1:]:
                name = f"{v}={lev}"
                cols[name] = (d[v] == lev).astype(float).where(d[v].notna())
                groups[name] = v
        else:
            cols[v] = d[v].astype(float)
            groups[v] = v
    for v, levels in NOM_LEVELS.items():
        if v in variables:
            for lev in levels[1:]:
                name = f"{v}={lev}"
                cols[name] = (d[v] == lev).astype(float).where(d[v].notna())
                groups[name] = v
    e6 = d["emp6"]
    if task in ("P",):
        if "employed" in variables:
            cols["employed"] = d["employed"].astype(float)
            groups["employed"] = "employed"
    if "emp6" in variables:
        if task in ("P", "A1"):
            cols["emp6=new_employer"] = (e6 == 2).astype(float)
            cols["emp6=self_employed"] = e6.isin([3, 4]).astype(float)
            groups["emp6=new_employer"] = groups["emp6=self_employed"] = "emp6"
        if task in ("P", "A2"):
            cols["emp6=job_seeking"] = (e6 == 5).astype(float)
            groups["emp6=job_seeking"] = "emp6"
    X = pd.DataFrame(cols, index=d.index)
    return X, groups


def small_vars(task: str) -> list[str]:
    return [v for v in SMALL_VARS if not (task != "P" and v == "employed")]


def status_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c == "employed" or c.startswith("emp6=")]
