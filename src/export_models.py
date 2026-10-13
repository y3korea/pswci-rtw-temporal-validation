"""Export the fitted development models (TRIPOD+AI item 22) without any person-level data.

For each task (A1, A2, P) and the primary development design (C1+C2):
  * CatBoost (primary model): native JSON model file (tree splits, borders, leaf values) plus the imputation medians.
    Training counts per leaf (`leaf_weights`), the training-loss history, file names, GUID and timestamp are removed
    (round 2: small-cell protection; leaves of the pooled-task depth-3 trees can hold < 5 training records). These
    fields are not needed for scoring; predictions are checked to be unchanged after stripping.
  * ridge-OH, LR-small and (pooled task) the status-only model: intercept and coefficients on the original scale with
    the imputation medians (scaling constants are not exported, so category counts cannot be derived).
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
    rows = [{"feature": "(intercept)", "imputation_median": np.nan, "coefficient": a_orig}]
    for f, bo in zip(names_out, b_orig):
        rows.append({"feature": f, "imputation_median": med.get(f, np.nan), "coefficient": bo})
    pd.DataFrame(rows).to_csv(os.path.join(out, f"model_{task}_{model}_coefficients.csv"), index=False)


def check_catboost(clf, path, n_features, n=2000, seed=1):
    """Predictions of the stripped JSON model must equal those of the fitted model (random inputs)."""
    from catboost import CatBoostClassifier
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 3, size=(n, n_features))
    m2 = CatBoostClassifier()
    m2.load_model(path, format="json")
    d = np.max(np.abs(clf.predict_proba(X)[:, 1] - m2.predict_proba(X)[:, 1]))
    if d > 1e-9:
        raise RuntimeError(f"stripped model differs ({d})")


def export_catboost(pipe, task, out):
    imp, clf = pipe.named_steps["imp"], pipe.named_steps["clf"]
    names_in, names_out = feature_names(pipe, len(imp.statistics_))
    path = os.path.join(out, f"model_{task}_CatBoost.json")
    clf.save_model(path, format="json")
    with open(path) as fh:
        js = json.load(fh)
    for tree in js["oblivious_trees"]:
        tree.pop("leaf_weights", None)
    for k in ("training", "train_finish_time", "model_guid", "output_options"):
        js["model_info"].pop(k, None)
    with open(path, "w") as fh:
        json.dump(js, fh)
    check_catboost(clf, path, len(names_out))
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
