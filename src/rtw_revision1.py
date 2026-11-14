"""Revision 1 (2026-11-13): paired C3 comparison of the final-rule primary model (CatBoost) with the model the
planned single tuning-CV rule would have chosen (EBM), requested by reviewers 3 and 4.

No re-tuning: both learners are refitted on C1+C2 with the hyperparameters saved in tables/tuning_primary.csv and
the design seed of the primary run. The script first checks that the refitted C3 AUROC/CITL equal the published
values (tables/perf_C3_<task>_unweighted.csv), then computes paired differences (CatBoost minus EBM) with the same
1,000 stratified bootstrap resamples as the main analysis, unweighted and with wave-2 survey weights.
Only aggregate results are written (revision1/tables/paired_catboost_ebm.csv). Row-level data are never written.

Usage: DATA_ROOT=... RTW_USE_PARQUET=1 RTW_OUT_DIR=<05_results> python rtw_revision1.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rtw_config import B_BOOT, N_JOBS, SEED  # noqa: E402
from rtw_data import build_analytic, load_long, task_frame  # noqa: E402
from rtw_develop import designs_for  # noqa: E402
from rtw_metrics import all_metrics, boot_indices  # noqa: E402
from rtw_models import ONEHOT_MODELS, design_seed, fit  # noqa: E402

METRICS = ("AUROC", "CITL", "Slope", "ICI", "Brier", "LogLoss")


def main():
    data_root = os.environ["DATA_ROOT"]
    out = os.environ.get("RTW_OUT_DIR", os.getcwd())
    tdir = os.environ.get("RTW_REV_DIR", os.path.join(out, "revision1", "tables"))
    os.makedirs(tdir, exist_ok=True)
    tun = pd.read_csv(os.path.join(out, "tables", "tuning_primary.csv"))
    a = build_analytic(load_long(data_root))
    rows, checks = [], []
    for task in ("A1", "A2", "P"):
        d = task_frame(a, task)
        D = designs_for(d, task)
        tr, te = d[d["cohort"].isin((1, 2))], d[d["cohort"] == 3]
        yd, yt = tr["y"].values, te["y"].values.astype(int)
        seed = design_seed("primary", task)
        preds = {}
        for m in ("CatBoost", "EBM"):
            params = json.loads(tun[(tun.task == task) & (tun.model == m)]["params"].iloc[0])
            X = D["oh" if m in ONEHOT_MODELS else "num"][0]
            est = fit(m, params, X.loc[tr.index], yd, n_jobs=N_JOBS, seed=seed)
            preds[m] = est.predict_proba(X.loc[te.index])[:, 1]
        pub = pd.read_csv(os.path.join(out, "tables", f"perf_C3_{task}_unweighted.csv"))
        for m, p in preds.items():
            got = all_metrics(yt, p, task, which=("AUROC", "CITL"))
            for k in ("AUROC", "CITL"):
                ref = float(pub[(pub.model == m) & (pub.metric == k)]["estimate"].iloc[0])
                checks.append({"task": task, "model": m, "metric": k, "published": ref, "refit": got[k],
                               "abs_diff": abs(got[k] - ref)})
        idxs = boot_indices(len(yt), B_BOOT, SEED, te["strata"].values)
        for lab, w in (("unweighted", None), ("weighted", te["wt_w2"].values)):
            pt = {m: all_metrics(yt, p, task, w, METRICS) for m, p in preds.items()}
            samp = {k: [] for k in METRICS}
            for idx in idxs:
                wb = None if w is None else np.asarray(w)[idx]
                r = {m: all_metrics(yt[idx], p[idx], task, wb, METRICS) for m, p in preds.items()}
                for k in METRICS:
                    samp[k].append(r["CatBoost"][k] - r["EBM"][k])
            for k in METRICS:
                s = np.asarray(samp[k], float)
                s = s[np.isfinite(s)]
                rows.append({"task": task, "weighting": lab, "metric": k, "CatBoost": pt["CatBoost"][k],
                             "EBM": pt["EBM"][k], "diff_CatBoost_minus_EBM": pt["CatBoost"][k] - pt["EBM"][k],
                             "lo": np.percentile(s, 2.5), "hi": np.percentile(s, 97.5), "n_boot": len(s)})
        print(f"[revision1] {task} done", flush=True)
    pd.DataFrame(checks).to_csv(os.path.join(tdir, "refit_check.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(tdir, "paired_catboost_ebm.csv"), index=False)


if __name__ == "__main__":
    main()
