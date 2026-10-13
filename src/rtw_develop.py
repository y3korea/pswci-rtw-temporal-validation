"""Development (tuning in the training cohorts) and prediction in the test cohort, revision v2."""
from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np
import pandas as pd

from rtw_config import BASELINES, CANDIDATES, N_JOBS, SEED
from rtw_data import design_matrix, small_vars, status_columns, task_frame
from rtw_metrics import logloss
from rtw_models import ONEHOT_MODELS, design_seed, fit, make_folds, oof_predict, tune


def designs_for(d, task, variables=None):
    Xn, gn = design_matrix(d, task, variables, ordinal_onehot=False)
    Xo, go = design_matrix(d, task, variables, ordinal_onehot=True)
    return {"num": (Xn, gn), "oh": (Xo, go)}


def develop(a, task, train_cohorts=(1, 2), test_cohort=3, reduced=False, tag="primary", cache_dir=None,
            log=print, models=None, variables=None, select_repeats=2):
    """Tune each candidate (CV log-loss), compute OOF and test predictions, select the primary model by
    repeated-CV log-loss. Returns a dict (cached)."""
    models = list(models or CANDIDATES)
    cache = None if cache_dir is None else os.path.join(cache_dir, f"dev_{tag}_{task}.joblib")
    if cache and os.path.exists(cache):
        log(f"[develop] cache {os.path.basename(cache)}")
        return joblib.load(cache)
    t0 = time.time()
    d = task_frame(a, task)
    D = designs_for(d, task, variables)
    tr = d[d["cohort"].isin(train_cohorts)]
    te = d[d["cohort"] == test_cohort] if test_cohort is not None else d.iloc[:0]
    yd, yt = tr["y"].values, te["y"].values
    strata = tr["cohort"].values * 10 + yd
    folds = make_folds(yd, strata)
    folds_rep = [make_folds(yd, strata, seed=SEED + 101 * r) for r in range(1, select_repeats)]
    seed = design_seed(tag, task)
    res = {"task": task, "tag": tag, "train_cohorts": list(train_cohorts), "test_cohort": test_cohort,
           "dev_index": tr.index.values, "test_index": te.index.values, "y_dev": yd, "y_test": yt,
           "dev_cohort": tr["cohort"].values, "params": {}, "cv_logloss": {}, "rep_logloss": {},
           "tune_seconds": {}, "n_candidates": {}, "oof": {}, "test": {}, "estimators": {},
           "groups": {k: v[1] for k, v in D.items()}, "columns": {k: list(v[0].columns) for k, v in D.items()}}
    for m in models:
        X = D["oh" if m in ONEHOT_MODELS else "num"][0]
        Xd, Xt = X.loc[tr.index], X.loc[te.index]
        params, ll, sec, nc = tune(m, Xd, yd, folds, n_jobs=N_JOBS, reduced=reduced, seed=seed)
        res["params"][m], res["cv_logloss"][m], res["tune_seconds"][m], res["n_candidates"][m] = params, ll, sec, nc
        res["oof"][m] = oof_predict(m, params, Xd, yd, folds, n_jobs=N_JOBS)
        rep = [logloss(yd, res["oof"][m])]
        for fr in folds_rep:
            rep.append(logloss(yd, oof_predict(m, params, Xd, yd, fr, n_jobs=N_JOBS)))
        res["rep_logloss"][m] = float(np.mean(rep))
        est = fit(m, params, Xd, yd, n_jobs=N_JOBS, seed=seed)
        res["estimators"][m] = est
        if len(te):
            res["test"][m] = est.predict_proba(Xt)[:, 1]
        log(f"[develop {tag} {task}] {m}: CV ll {ll:.4f}; repeated-CV ll {res['rep_logloss'][m]:.4f} "
            f"({sec:.0f}s, {nc} cand.) {params}")
    # baselines -------------------------------------------------------------------------------
    Xn = D["num"][0]
    for b in BASELINES[task]:
        if b == "Null":
            cols = Xn.columns[:1].tolist()
        elif b == "Status":
            cols = status_columns(Xn)
        else:
            cols = [c for c in Xn.columns if D["num"][1][c] in small_vars(task)]
        name = "Null" if b == "Null" else ("Status" if b == "Status" else "LR-small")
        res["oof"][b] = oof_predict(name, None, Xn.loc[tr.index, cols], yd, folds)
        est = fit(name, None, Xn.loc[tr.index, cols], yd)
        res["estimators"][b] = est
        res[f"cols_{b}"] = cols
        if len(te):
            res["test"][b] = est.predict_proba(Xn.loc[te.index, cols])[:, 1]
    if task == "P" and len(te):
        res["status_flag_test"] = (te["employed"].values == 0).astype(int)   # status rule: flag not employed at w1
    res["primary"] = min(res["rep_logloss"], key=res["rep_logloss"].get)
    trees = {m: v for m, v in res["rep_logloss"].items() if m in ("RF", "XGB", "LGBM", "CatBoost", "EBM")}
    res["attr_model"] = min(trees, key=trees.get) if trees else None
    res["seconds"] = time.time() - t0
    log(f"[develop {tag} {task}] primary={res['primary']} (repeated-CV log-loss) ({res['seconds']:.0f}s)")
    if cache:
        joblib.dump(res, cache)
    return res


def params_table(res, repeats=2):
    """Tuning summary. Round 2 adds the two components of the repeated-CV log-loss: repeat 1 reuses the tuning folds
    (OOF log-loss), repeat 2 uses independent folds (= 2 x mean - repeat 1 when select_repeats = 2), and flags the
    model each rule would select (single tuning-CV rule of v1; repeated-CV rule of v2.0; independent repeat only)."""
    rows = []
    for m in res["params"]:
        rep1 = logloss(res["y_dev"], res["oof"][m])
        rows.append({"task": res["task"], "design": res["tag"], "model": m, "cv_logloss": res["cv_logloss"][m],
                     "repeated_cv_logloss": res["rep_logloss"][m], "logloss_repeat1_tuning_folds": rep1,
                     "logloss_repeat2_independent_folds": (2 * res["rep_logloss"][m] - rep1) if repeats == 2 else np.nan,
                     "n_candidates": res["n_candidates"][m], "tune_seconds": round(res["tune_seconds"][m], 1),
                     "params": json.dumps(res["params"][m], default=str)})
    df = pd.DataFrame(rows)
    for col, flag in (("cv_logloss", "selected_v1_single_cv"), ("repeated_cv_logloss", "selected_v2_repeated_cv"),
                      ("logloss_repeat2_independent_folds", "selected_independent_repeat")):
        df[flag] = (df[col] == df[col].min()) if df[col].notna().any() else False
    return df
