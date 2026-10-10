"""Export the fitted development models (TRIPOD+AI item 22) without any person-level data.

For each task (A1, A2, P) and the primary development design (C1+C2):
  * CatBoost (primary model): native JSON model file (tree splits, borders, leaf values) plus the imputation medians;
  * ridge-OH, LR-small and (pooled task) the status-only model: intercept and coefficients on the standardised and on
    the original scale, with the imputation medians and scaling constants.
Only fitted parameters and aggregate constants are written. Usage:
    python export_models.py CACHE_DIR OUT_DIR
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd


def feature_names(pipe, n_in):
    imp = pipe.named_steps["imp"]
    names_in = list(getattr(imp, "feature_names_in_", [f"x{i}" for i in range(n_in)]))
    try:
        out = list(imp.get_feature_names_out(names_in))
    except Exception:  # pragma: no cover
        out = names_in + [f"missingindicator_{names_in[i]}" for i in imp.indicator_.features_]
    return names_in, out


def export_linear(pipe, model, task, out):
    imp, sc, clf = pipe.named_steps["imp"], pipe.named_steps["sc"], pipe.named_steps["clf"]
    names_in, names_out = feature_names(pipe, len(imp.statistics_))
    beta = clf.coef_.ravel()
    mu, sd = sc.mean_, np.where(sc.scale_ > 0, sc.scale_, 1.0)
    b_orig = beta / sd
    a_orig = float(clf.intercept_[0] - np.sum(beta * mu / sd))
    med = dict(zip(names_in, imp.statistics_))
    rows = [{"feature": "(intercept)", "imputation_median": np.nan, "scaler_mean": np.nan, "scaler_sd": np.nan,
             "coef_standardised": float(clf.intercept_[0]), "coef_original_scale": a_orig}]
    for f, b, m, s, bo in zip(names_out, beta, mu, sd, b_orig):
        rows.append({"feature": f, "imputation_median": med.get(f, np.nan), "scaler_mean": m, "scaler_sd": s,
                     "coef_standardised": b, "coef_original_scale": bo})
    pd.DataFrame(rows).to_csv(os.path.join(out, f"model_{task}_{model}_coefficients.csv"), index=False)


def export_catboost(pipe, task, out):
    imp, clf = pipe.named_steps["imp"], pipe.named_steps["clf"]
    names_in, names_out = feature_names(pipe, len(imp.statistics_))
    clf.save_model(os.path.join(out, f"model_{task}_CatBoost.json"), format="json")
    pd.DataFrame({"feature": names_in, "imputation_median": imp.statistics_}).to_csv(
        os.path.join(out, f"model_{task}_CatBoost_imputation.csv"), index=False)
    with open(os.path.join(out, f"model_{task}_CatBoost_inputs.json"), "w") as f:
        json.dump({"input_order_after_imputer": names_out,
                   "note": "Predicted probability = employed at wave 2. Inputs: median-imputed predictors followed by "
                           "missing-value indicators, in this order (design matrix from rtw_data.design_matrix)."}, f,
                  indent=1)


def main(cache, out):
    os.makedirs(out, exist_ok=True)
    for task in ("A1", "A2", "P"):
        r = joblib.load(os.path.join(cache, f"dev_primary_{task}.joblib"))
        export_catboost(r["estimators"]["CatBoost"], task, out)
        for m in ("RidgeOH", "LR-small", "Status"):
            if m in r["estimators"]:
                export_linear(r["estimators"][m], m, task, out)
    print("models exported to", out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
