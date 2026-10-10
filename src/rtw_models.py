"""Model definitions, hyper-parameter tuning (development data only), OOF predictions, TreeSHAP."""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from rtw_config import SEED

warnings.filterwarnings("ignore", category=UserWarning)


def make_estimator(name: str, params: dict | None = None, n_jobs: int = 1, seed: int = SEED) -> Pipeline:
    imp = SimpleImputer(strategy="median", add_indicator=True)
    if name in ("LR", "LR-5", "Persistence"):
        clf = LogisticRegression(penalty=None, max_iter=10000)
        steps = [("imp", imp), ("sc", StandardScaler()), ("clf", clf)]
    elif name == "EN":
        clf = LogisticRegression(penalty="elasticnet", solver="saga", max_iter=20000, tol=1e-4,
                                 random_state=seed)
        steps = [("imp", imp), ("sc", StandardScaler()), ("clf", clf)]
    elif name == "RF":
        clf = RandomForestClassifier(n_estimators=500, n_jobs=n_jobs, random_state=seed)
        steps = [("imp", imp), ("clf", clf)]
    elif name == "XGB":
        clf = XGBClassifier(tree_method="hist", n_jobs=n_jobs, random_state=seed, eval_metric="logloss",
                            verbosity=0)
        steps = [("imp", imp), ("clf", clf)]
    elif name == "LGBM":
        clf = LGBMClassifier(n_jobs=n_jobs, random_state=seed, verbose=-1, subsample_freq=1)
        steps = [("imp", imp), ("clf", clf)]
    else:
        raise ValueError(name)
    pipe = Pipeline(steps)
    if params:
        pipe.set_params(**params)
    return pipe


def search_space(name: str, reduced: bool = False):
    """Return (space, n_iter, kind). kind = 'grid' | 'random' | None."""
    if name == "EN":
        return ({"clf__C": list(np.logspace(-3, 1, 9)), "clf__l1_ratio": [0.1, 0.5, 0.9, 1.0]}, None, "grid")
    if name == "RF":
        return ({"clf__max_features": ["sqrt", 0.2, 0.33, 0.5],
                 "clf__min_samples_leaf": [1, 5, 10, 20, 40],
                 "clf__max_depth": [None, 6, 10, 14]}, 10 if reduced else 20, "random")
    if name == "XGB":
        return ({"clf__n_estimators": randint(100, 1001), "clf__learning_rate": loguniform(0.01, 0.2),
                 "clf__max_depth": randint(2, 7), "clf__min_child_weight": randint(1, 21),
                 "clf__subsample": uniform(0.6, 0.4), "clf__colsample_bytree": uniform(0.5, 0.5),
                 "clf__reg_lambda": loguniform(0.1, 20)}, 20 if reduced else 40, "random")
    if name == "LGBM":
        return ({"clf__n_estimators": randint(100, 1001), "clf__learning_rate": loguniform(0.01, 0.2),
                 "clf__num_leaves": randint(4, 49), "clf__min_child_samples": randint(10, 121),
                 "clf__subsample": uniform(0.6, 0.4), "clf__colsample_bytree": uniform(0.5, 0.5),
                 "clf__reg_lambda": uniform(0, 20)}, 20 if reduced else 40, "random")
    return (None, None, None)


def make_folds(y, strata, k=5, seed=SEED):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), strata))


def tune(name, X, y, folds, n_jobs=4, reduced=False, seed=SEED):
    """Tune by CV log-loss. Returns (best_params, best_cv_logloss, seconds)."""
    t0 = time.time()
    space, n_iter, kind = search_space(name, reduced)
    if kind is None:  # unpenalised LR: no tuning; still report CV log-loss
        est = make_estimator(name)
        from sklearn.model_selection import cross_val_score
        s = cross_val_score(est, X, y, cv=folds, scoring="neg_log_loss", n_jobs=n_jobs)
        return {}, float(-s.mean()), time.time() - t0
    inner_jobs = n_jobs if name == "RF" else 1
    outer_jobs = 1 if name == "RF" else n_jobs
    est = make_estimator(name, n_jobs=inner_jobs, seed=seed)
    if kind == "grid":
        gs = GridSearchCV(est, space, scoring="neg_log_loss", cv=folds, n_jobs=outer_jobs, refit=False)
    else:
        gs = RandomizedSearchCV(est, space, n_iter=n_iter, scoring="neg_log_loss", cv=folds,
                                n_jobs=outer_jobs, refit=False, random_state=seed)
    gs.fit(X, y)
    best = gs.best_params_
    best = {k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()}
    return best, float(-gs.best_score_), time.time() - t0


def oof_predict(name, params, X, y, folds, n_jobs=4, sample_weight=None):
    oof = np.full(len(y), np.nan)
    for tr, te in folds:
        est = make_estimator(name, params, n_jobs=n_jobs)
        fit_kw = {} if sample_weight is None else {"clf__sample_weight": np.asarray(sample_weight)[tr]}
        est.fit(X.iloc[tr], np.asarray(y)[tr], **fit_kw)
        oof[te] = est.predict_proba(X.iloc[te])[:, 1]
    return oof


def fit(name, params, X, y, n_jobs=4, sample_weight=None):
    est = make_estimator(name, params, n_jobs=n_jobs)
    fit_kw = {} if sample_weight is None else {"clf__sample_weight": np.asarray(sample_weight)}
    est.fit(X, np.asarray(y), **fit_kw)
    return est


# ---- TreeSHAP (log-odds scale) ------------------------------------------------------------

def tree_shap(pipe: Pipeline, X: pd.DataFrame, max_rows: int | None = None, seed=SEED):
    """SHAP values for the fitted tree pipeline on X (after imputation). Returns DataFrame."""
    Xt_arr = pipe.named_steps["imp"].transform(X)
    names = list(pipe.named_steps["imp"].get_feature_names_out(X.columns))
    Xt = pd.DataFrame(Xt_arr, columns=names, index=X.index)
    if max_rows is not None and len(Xt) > max_rows:
        Xt = Xt.sample(max_rows, random_state=seed)
    clf = pipe.named_steps["clf"]
    if isinstance(clf, LGBMClassifier):
        contrib = clf.predict(Xt, pred_contrib=True)
        sv = contrib[:, :-1]
    elif isinstance(clf, XGBClassifier):
        import xgboost as xgb
        booster = clf.get_booster()
        sv = booster.predict(xgb.DMatrix(Xt.values, feature_names=None), pred_contribs=True)[:, :-1]
    else:  # random forest
        import shap
        expl = shap.TreeExplainer(clf)
        v = expl.shap_values(Xt.values, check_additivity=False)
        sv = v[1] if isinstance(v, list) else (v[:, :, 1] if np.ndim(v) == 3 else v)
    return pd.DataFrame(sv, columns=names, index=Xt.index)


def aggregate_shap(sv: pd.DataFrame, groups: dict) -> pd.Series:
    """Mean |SHAP| per original variable (columns of the same variable summed per row first).

    Missing-indicator columns ('missingindicator_x') are attributed to variable x.
    """
    g = {}
    for c in sv.columns:
        base = c.replace("missingindicator_", "")
        g[c] = groups.get(base, base)
    agg = sv.T.groupby(pd.Series(g)).sum().T
    return agg.abs().mean().sort_values(ascending=False)
