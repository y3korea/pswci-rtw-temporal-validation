"""Development (tuning in C1+C2) and temporal validation (C3) of all models for one task."""
from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np
import pandas as pd

from rtw_config import ML_MODELS, N_JOBS, SEED
from rtw_data import design_matrix, lr5_vars, task_frame
from rtw_models import fit, make_folds, oof_predict, tune


def split(a, task, train_cohorts, test_cohort, variables=None):
    d = task_frame(a, task)
    X, groups = design_matrix(d, task, variables)
    tr = d[d["cohort"].isin(train_cohorts)]
    te = d[d["cohort"] == test_cohort]
    return d, X, groups, tr, te


def develop(a, task, train_cohorts=(1, 2), test_cohort=3, reduced=False, tag="primary",
            cache_dir=None, log=print, variables=None, fixed_params=None):
    """Tune (or use fixed_params), compute development OOF predictions, refit and predict test cohort."""
    cache = None if cache_dir is None else os.path.join(cache_dir, f"dev_{tag}_task{task}.joblib")
    if cache and os.path.exists(cache):
        log(f"[develop] loaded cache {os.path.basename(cache)}")
        return joblib.load(cache)
    t0 = time.time()
    d, X, groups, tr, te = split(a, task, train_cohorts, test_cohort, variables)
    Xd, yd = X.loc[tr.index], tr["y"].values
    Xt, yt = X.loc[te.index], te["y"].values
    strata = (tr["cohort"].values * 10 + yd)
    folds = make_folds(yd, strata)
    res = {"task": task, "tag": tag, "train_cohorts": list(train_cohorts), "test_cohort": test_cohort,
           "n_dev": int(len(yd)), "n_test": int(len(yt)), "events_dev": int(yd.sum()),
           "events_test": int(yt.sum()), "columns": list(X.columns), "groups": groups,
           "params": {}, "cv_logloss": {}, "tune_seconds": {}, "oof": {}, "test": {},
           "dev_index": tr.index.values, "test_index": te.index.values, "y_dev": yd, "y_test": yt,
           "dev_cohort": tr["cohort"].values}
    for m in ML_MODELS:
        if fixed_params is not None:
            params, ll, sec = fixed_params[m], np.nan, 0.0
        else:
            params, ll, sec = tune(m, Xd, yd, folds, n_jobs=N_JOBS, reduced=reduced)
        res["params"][m], res["cv_logloss"][m], res["tune_seconds"][m] = params, ll, sec
        res["oof"][m] = oof_predict(m, params, Xd, yd, folds, n_jobs=N_JOBS)
        est = fit(m, params, Xd, yd, n_jobs=N_JOBS)
        res["test"][m] = est.predict_proba(Xt)[:, 1]
        res.setdefault("estimators", {})[m] = est
        log(f"[develop {tag} task {task}] {m}: CV log-loss {ll:.4f} ({sec:.0f}s) params={params}")
    # baselines -------------------------------------------------------------------------
    lr5_cols = [c for c in X.columns if groups[c] in lr5_vars(task)]
    res["oof"]["LR-5"] = oof_predict("LR-5", None, Xd[lr5_cols], yd, folds)
    est5 = fit("LR-5", None, Xd[lr5_cols], yd)
    res["test"]["LR-5"] = est5.predict_proba(Xt[lr5_cols])[:, 1]
    res["estimators"]["LR-5"] = est5
    res["lr5_cols"] = lr5_cols
    if task == "A":
        res["oof"]["Persistence"] = oof_predict("Persistence", None, Xd[["employed"]], yd, folds)
        estp = fit("Persistence", None, Xd[["employed"]], yd)
        res["test"]["Persistence"] = estp.predict_proba(Xt[["employed"]])[:, 1]
        res["persistence_flag_test"] = (Xt["employed"].values == 0).astype(int)  # flag = not employed at w1
        res["persistence_flag_dev"] = (Xd["employed"].values == 0).astype(int)
        res["estimators"]["Persistence"] = estp
    else:  # null model: development re-entry prevalence for everyone
        oof = np.full(len(yd), np.nan)
        for trf, tef in folds:
            oof[tef] = yd[trf].mean()
        res["oof"]["Persistence"] = oof
        res["test"]["Persistence"] = np.full(len(yt), yd.mean())
    ll = {m: v for m, v in res["cv_logloss"].items() if np.isfinite(v)}
    res["primary"] = min(ll, key=ll.get) if ll else None
    tree_ll = {m: ll[m] for m in ll if m in ("RF", "XGB", "LGBM")}
    res["shap_model"] = min(tree_ll, key=tree_ll.get) if tree_ll else None
    res["seconds"] = time.time() - t0
    log(f"[develop {tag} task {task}] primary={res['primary']} shap_model={res['shap_model']} "
        f"({res['seconds']:.0f}s)")
    if cache:
        joblib.dump(res, cache)
    return res


def params_table(res):
    rows = []
    for m in ML_MODELS:
        rows.append({"task": res["task"], "design": res["tag"], "model": m,
                     "cv_logloss": res["cv_logloss"][m], "tune_seconds": round(res["tune_seconds"][m], 1),
                     "params": json.dumps(res["params"][m], default=str)})
    return pd.DataFrame(rows)
