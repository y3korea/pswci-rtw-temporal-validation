"""Model definitions, tuning (development data only), OOF predictions and attributions (revision v2)."""
from __future__ import annotations

import time
import warnings
import zlib

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from rtw_config import FAST, SEED

warnings.filterwarnings("ignore")

ONEHOT_MODELS = {"RidgeOH"}          # models that use the ordinal-one-hot design matrix


def design_seed(tag: str, task: str) -> int:
    """Design-specific seed so that random searches do not reuse one candidate list across designs."""
    return SEED + zlib.crc32(f"{tag}|{task}".encode()) % 10000


def make_estimator(name: str, params: dict | None = None, n_jobs: int = 1, seed: int = SEED) -> Pipeline:
    imp = SimpleImputer(strategy="median", add_indicator=True)
    if name in ("LR", "LR-small", "Status"):
        steps = [("imp", imp), ("sc", StandardScaler()), ("clf", LogisticRegression(penalty=None, max_iter=10000))]
    elif name == "Null":
        steps = [("imp", imp), ("clf", DummyClassifier(strategy="prior"))]
    elif name == "RidgeOH":
        steps = [("imp", imp), ("sc", StandardScaler()),
                 ("clf", LogisticRegression(penalty="l2", C=0.1, max_iter=10000))]
    elif name == "RF":
        steps = [("imp", imp), ("clf", RandomForestClassifier(n_estimators=500, n_jobs=n_jobs, random_state=seed))]
    elif name == "XGB":
        steps = [("imp", imp), ("clf", XGBClassifier(tree_method="hist", n_jobs=n_jobs, random_state=seed,
                                                     eval_metric="logloss", verbosity=0))]
    elif name == "LGBM":
        steps = [("imp", imp), ("clf", LGBMClassifier(n_jobs=n_jobs, random_state=seed, verbose=-1, subsample_freq=1))]
    elif name == "CatBoost":
        from catboost import CatBoostClassifier
        steps = [("imp", imp), ("clf", CatBoostClassifier(verbose=0, thread_count=n_jobs, random_seed=seed,
                                                          allow_writing_files=False))]
    elif name == "EBM":
        from interpret.glassbox import ExplainableBoostingClassifier
        steps = [("imp", imp), ("clf", ExplainableBoostingClassifier(n_jobs=n_jobs, random_state=seed, outer_bags=8))]
    else:
        raise ValueError(name)
    pipe = Pipeline(steps)
    if params:
        pipe.set_params(**params)
    return pipe


def search_space(name: str, reduced: bool = False):
    """(space, n_iter, kind). Widened ranges (v2): depth-1 trees, small/large leaves, wider learning rates."""
    if FAST:
        reduced = True
    if name == "RidgeOH":
        return ({"clf__C": list(np.logspace(-4, 1, 11))}, None, "grid")
    if name == "RF":
        return ({"clf__max_features": ["sqrt", 0.1, 0.2, 0.33, 0.5],
                 "clf__min_samples_leaf": [1, 5, 10, 20, 40, 80],
                 "clf__max_depth": [3, 4, 6, 8, 10, 14, None]}, 4 if FAST else (8 if reduced else 20), "random")
    if name == "XGB":
        return ({"clf__n_estimators": randint(50, 1501), "clf__learning_rate": loguniform(0.005, 0.3),
                 "clf__max_depth": randint(1, 7), "clf__min_child_weight": loguniform(1, 50),
                 "clf__subsample": uniform(0.5, 0.5), "clf__colsample_bytree": uniform(0.3, 0.7),
                 "clf__reg_lambda": loguniform(0.01, 50)}, 4 if FAST else (20 if reduced else 50), "random")
    if name == "LGBM":
        return ({"clf__n_estimators": randint(50, 1501), "clf__learning_rate": loguniform(0.005, 0.3),
                 "clf__num_leaves": randint(2, 65), "clf__min_child_samples": randint(5, 201),
                 "clf__subsample": uniform(0.5, 0.5), "clf__colsample_bytree": uniform(0.3, 0.7),
                 "clf__reg_lambda": uniform(0, 50)}, 4 if FAST else (20 if reduced else 50), "random")
    if name == "CatBoost":
        return ({"clf__iterations": randint(100, 1001), "clf__learning_rate": loguniform(0.01, 0.3),
                 "clf__depth": randint(1, 7), "clf__l2_leaf_reg": loguniform(1, 30)},
                3 if FAST else (8 if reduced else 20), "random")
    if name == "EBM":
        return ({"clf__interactions": [0, 10], "clf__learning_rate": [0.01, 0.04]}, None, "grid")
    return (None, None, None)


def make_folds(y, strata, k=5, seed=SEED):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), strata))


def tune(name, X, y, folds, n_jobs=4, reduced=False, seed=SEED):
    """Tune by CV log-loss. Returns (best_params, best_cv_logloss, seconds, n_candidates)."""
    t0 = time.time()
    space, n_iter, kind = search_space(name, reduced)
    if kind is None:
        est = make_estimator(name)
        s = cross_val_score(est, X, y, cv=folds, scoring="neg_log_loss", n_jobs=n_jobs)
        return {}, float(-s.mean()), time.time() - t0, 1
    inner_jobs = n_jobs if name in ("RF", "CatBoost", "EBM") else 1
    outer_jobs = 1 if name in ("RF", "CatBoost", "EBM") else n_jobs
    est = make_estimator(name, n_jobs=inner_jobs, seed=seed)
    if kind == "grid":
        gs = GridSearchCV(est, space, scoring="neg_log_loss", cv=folds, n_jobs=outer_jobs, refit=False)
    else:
        gs = RandomizedSearchCV(est, space, n_iter=n_iter, scoring="neg_log_loss", cv=folds,
                                n_jobs=outer_jobs, refit=False, random_state=seed)
    gs.fit(X, y)
    best = {k: (v.item() if hasattr(v, "item") else v) for k, v in gs.best_params_.items()}
    ncand = len(gs.cv_results_["params"])
    return best, float(-gs.best_score_), time.time() - t0, ncand


def oof_predict(name, params, X, y, folds, n_jobs=4, sample_weight=None):
    oof = np.full(len(y), np.nan)
    for tr, te in folds:
        est = make_estimator(name, params, n_jobs=n_jobs)
        kw = {} if sample_weight is None else {"clf__sample_weight": np.asarray(sample_weight)[tr]}
        est.fit(X.iloc[tr], np.asarray(y)[tr], **kw)
        oof[te] = est.predict_proba(X.iloc[te])[:, 1]
    return oof


def fit(name, params, X, y, n_jobs=4, sample_weight=None, seed=SEED):
    est = make_estimator(name, params, n_jobs=n_jobs, seed=seed)
    kw = {} if sample_weight is None else {"clf__sample_weight": np.asarray(sample_weight)}
    est.fit(X, np.asarray(y), **kw)
    return est


def ridge_cv(X, y, n_jobs=4, seed=SEED):
    """Ridge LR with C chosen by 5-fold CV log-loss on the given data (used for small-sample refits)."""
    folds = make_folds(y, y, seed=seed)
    best, ll, _, _ = tune("RidgeOH", X, y, folds, n_jobs=n_jobs)
    return fit("RidgeOH", best, X, y, n_jobs=n_jobs)


# ---- attributions (log-odds scale) -----------------------------------------------------------

def attributions(pipe: Pipeline, X: pd.DataFrame, max_rows: int | None = None, seed=SEED):
    """Exact additive attributions: TreeSHAP for tree ensembles, term contributions for EBM, coef*x for LR."""
    Xt_arr = pipe.named_steps["imp"].transform(X)
    names = list(pipe.named_steps["imp"].get_feature_names_out(X.columns))
    Xt = pd.DataFrame(Xt_arr, columns=names, index=X.index)
    if max_rows is not None and len(Xt) > max_rows:
        Xt = Xt.sample(max_rows, random_state=seed)
    clf = pipe.named_steps["clf"]
    cname = type(clf).__name__
    if cname == "LGBMClassifier":
        sv = clf.predict(Xt, pred_contrib=True)[:, :-1]
    elif cname == "XGBClassifier":
        import xgboost as xgb
        sv = clf.get_booster().predict(xgb.DMatrix(Xt.values), pred_contribs=True)[:, :-1]
    elif cname == "CatBoostClassifier":
        from catboost import Pool
        sv = clf.get_feature_importance(Pool(Xt.values), type="ShapValues")[:, :-1]
    elif cname == "ExplainableBoostingClassifier":
        terms = clf.eval_terms(Xt.values)          # per-term contributions (mains + interactions)
        sv = np.zeros((len(Xt), len(names)))
        for j, feats in enumerate(clf.term_features_):
            share = terms[:, j] / len(feats)
            for fidx in feats:
                sv[:, fidx] += share
    elif cname == "RandomForestClassifier":
        import shap
        v = shap.TreeExplainer(clf).shap_values(Xt.values, check_additivity=False)
        sv = v[1] if isinstance(v, list) else (v[:, :, 1] if np.ndim(v) == 3 else v)
    else:  # linear pipelines: coef * standardised x (centered)
        Z = pipe.named_steps["sc"].transform(Xt_arr) if "sc" in pipe.named_steps else Xt_arr
        Z = pd.DataFrame(Z, index=X.index).loc[Xt.index].values
        sv = Z * clf.coef_[0]
    return pd.DataFrame(sv, columns=names, index=Xt.index)


def aggregate_attr(sv: pd.DataFrame, groups: dict) -> pd.Series:
    g = {c: groups.get(c.replace("missingindicator_", ""), c.replace("missingindicator_", "")) for c in sv.columns}
    agg = sv.T.groupby(pd.Series(g)).sum().T
    return agg.abs().mean().sort_values(ascending=False)
