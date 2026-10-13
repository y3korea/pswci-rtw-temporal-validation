"""Dataset-shift diagnostics (revision v2).

* outcome-prevalence change and transition probabilities (unweighted and survey-weighted)
* conditional cohort effects (pooled logistic model with cohort indicators)
* test of whether the outcome improves a domain classifier given x (conditional/concept shift)
* adversarial validation per cohort pair (full and comparability-restricted predictor sets)
* DISDE-type decomposition (Cai, Namkoong & Yadlowsky): performance change from source to target split into
  X-shift on the source side, Y|X shift on the shared support and X-shift on the target side; the shared
  distribution has density proportional to p_S(x) p_T(x) / (p_S(x) + p_T(x)), so the weights are bounded in (0, 1)
  and need no truncation. The domain classifier is re-estimated inside every bootstrap resample.
* model-based (case-mix) expected AUROC; full importance-weighting truncation sensitivity, ESS and balance
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.model_selection import StratifiedKFold

from rtw_config import B_SHIFT, N_JOBS, PREDICTORS, RESTRICTED_DROP_B, SEED, VAR_LABELS
from rtw_data import design_matrix, task_frame
from rtw_metrics import (EPS, auroc, brier, cal_slope, citl, logloss, newton_logistic, wilson)
from rtw_models import aggregate_attr, make_estimator


# ---- prevalence ------------------------------------------------------------------------------

def prevalence_table(a):
    rows = []
    for task in ("P", "A1", "A2"):
        d = task_frame(a, task)
        for c in (1, 2, 3):
            s = d[d["cohort"] == c]
            p, lo, hi = wilson(int(s["y"].sum()), len(s))
            pw = np.average(s["y"], weights=s["wt_w2"])
            rows.append({"task": task, "cohort": c, "n": len(s), "employed_w2": int(s["y"].sum()), "prev": p,
                         "lo": lo, "hi": hi, "prev_weighted": pw})
    return pd.DataFrame(rows)


# ---- conditional cohort effects --------------------------------------------------------------

def cohort_effects(a, task):
    import statsmodels.api as sm
    d = task_frame(a, task)
    X, _ = design_matrix(d, task)
    X = X.fillna(X.median())
    X = X.loc[:, X.std() > 0]
    X = (X - X.mean()) / X.std()
    X["C2"] = (d["cohort"] == 2).astype(float)
    X["C3"] = (d["cohort"] == 3).astype(float)
    mod = sm.Logit(d["y"].values, sm.add_constant(X)).fit(disp=0, maxiter=300)
    ci = mod.conf_int()
    cov = mod.cov_params()
    rows = []
    for k in ("C2", "C3"):
        rows.append({"task": task, "contrast": f"{k} vs C1", "log_odds": mod.params[k], "lo": ci.loc[k, 0],
                     "hi": ci.loc[k, 1], "p": mod.pvalues[k]})
    diff = mod.params["C3"] - mod.params["C2"]
    se = np.sqrt(cov.loc["C3", "C3"] + cov.loc["C2", "C2"] - 2 * cov.loc["C2", "C3"])
    rows.append({"task": task, "contrast": "C3 vs C2", "log_odds": diff, "lo": diff - 1.96 * se,
                 "hi": diff + 1.96 * se, "p": 2 * (1 - __import__("scipy").stats.norm.cdf(abs(diff / se)))})
    # unadjusted (crude) log-odds differences for comparison
    for (c1, c0) in ((2, 1), (3, 1), (3, 2)):
        y1, y0 = d.loc[d.cohort == c1, "y"].mean(), d.loc[d.cohort == c0, "y"].mean()
        rows.append({"task": task, "contrast": f"crude C{c1} vs C{c0}",
                     "log_odds": np.log(y1 / (1 - y1)) - np.log(y0 / (1 - y0)), "lo": np.nan, "hi": np.nan, "p": np.nan})
    return pd.DataFrame(rows)


def outcome_in_domain_test(a, task, src, tgt=(3,)):
    """LRT: does y improve a logistic domain classifier (target vs source) given x? (conditional shift test)."""
    d = task_frame(a, task)
    d = d[d["cohort"].isin(list(src) + list(tgt))]
    X, _ = design_matrix(d, task)
    X = X.fillna(X.median())
    X = X.loc[:, X.std() > 0]
    Z = ((X - X.mean()) / X.std()).values
    z = d["cohort"].isin(tgt).astype(int).values
    X0 = np.column_stack([np.ones(len(z)), Z])
    X1 = np.column_stack([X0, d["y"].values])
    def ll(Xm):
        b, ok = newton_logistic(Xm, z)
        mu = np.clip(1 / (1 + np.exp(-(Xm @ b))), EPS, 1 - EPS)
        return np.sum(z * np.log(mu) + (1 - z) * np.log(1 - mu)), b
    l0, _ = ll(X0)
    l1, b1 = ll(X1)
    stat = 2 * (l1 - l0)
    return {"task": task, "source": "+".join(f"C{c}" for c in src), "target": "C3", "coef_y": b1[-1],
            "lrt": stat, "p": float(chi2.sf(max(stat, 0), 1))}


# ---- domain classifiers ----------------------------------------------------------------------

def _lgbm_domain():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=40,
                          random_state=SEED, n_jobs=N_JOBS, verbose=-1)


def domain_probs(Xs, Xt, seed=SEED, kind="LR"):
    """Cross-fitted P(target | x) for the stacked source and target rows (5 folds)."""
    Xa = pd.concat([Xs, Xt])
    z = np.r_[np.zeros(len(Xs)), np.ones(len(Xt))].astype(int)
    pz = np.zeros(len(z))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xa, z):
        est = make_estimator("LR") if kind == "LR" else _lgbm_domain()
        est.fit(Xa.iloc[tr], z[tr])
        pz[te] = est.predict_proba(Xa.iloc[te])[:, 1]
    pz = np.clip(pz, EPS, 1 - EPS)
    return pz[: len(Xs)], pz[len(Xs):], auroc(z, pz)


def density_ratio(pi, ns, nt):
    """r(x) = p_T(x) / p_S(x) from P(target | x) with the sample-size prior corrected."""
    return pi / (1 - pi) * (ns / nt)


def adversarial(a, pairs, B=1000, restricted=False):
    rows, drivers = [], []
    variables = [v for v in PREDICTORS if v not in RESTRICTED_DROP_B] if restricted else None
    d = task_frame(a, "P")
    X, groups = design_matrix(d, "P", variables)
    for src, tgt in pairs:
        m = d["cohort"].isin(list(src) + list(tgt))
        Xm = X[m]
        z = d.loc[m, "cohort"].isin(tgt).astype(int).values
        oof = np.zeros(len(z))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xm, z):
            oof[te] = _lgbm_domain().fit(Xm.iloc[tr], z[tr]).predict_proba(Xm.iloc[te])[:, 1]
        rng = np.random.default_rng(SEED)
        bs = [auroc(z[i], oof[i]) for i in (rng.integers(0, len(z), len(z)) for _ in range(B))]
        lab = f"{'+'.join(f'C{c}' for c in src)} vs {'+'.join(f'C{c}' for c in tgt)}"
        rows.append({"contrast": lab, "predictors": "restricted" if restricted else "all", "n": len(z),
                     "AUROC": auroc(z, oof), "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5)})
        clf = _lgbm_domain().fit(Xm, z)
        sv = pd.DataFrame(clf.predict(Xm, pred_contrib=True)[:, :-1], columns=Xm.columns)
        imp = aggregate_attr(sv, groups)
        drivers.append(pd.DataFrame({"contrast": lab, "predictors": rows[-1]["predictors"], "variable": imp.index,
                                     "label": [VAR_LABELS.get(v, v) for v in imp.index],
                                     "mean_abs_shap": imp.values}))
    return pd.DataFrame(rows), pd.concat(drivers)


# ---- DISDE-type decomposition ----------------------------------------------------------------

def _metric(name, y, p, w=None):
    p = np.clip(p, EPS, 1 - EPS)
    if name == "Prevalence":
        return float(np.average(y, weights=w))
    if name == "Brier":
        return brier(y, p, w)
    if name == "LogLoss":
        return logloss(y, p, w)
    if name == "AUROC":
        return auroc(y, p, w)
    if name == "CITL":
        return citl(y, p, w)
    if name == "Slope":
        return cal_slope(y, p, w)
    raise ValueError(name)


DISDE_METRICS = ("Prevalence", "Brier", "LogLoss", "AUROC", "CITL", "Slope")


def _disde_once(ys, ps_d, yt, pt_d, pis, pit, ns, nt):
    rs, rt = density_ratio(pis, ns, nt), density_ratio(pit, ns, nt)
    ws = rs / (1 + rs)          # shared / source density ratio (up to a constant)
    wt = 1 / (1 + rt)           # shared / target density ratio (up to a constant)
    out = {}
    for model in ps_d:
        ps, pt = ps_d[model], pt_d[model]
        for m in DISDE_METRICS:
            if m == "Prevalence" and model != list(ps_d)[0]:
                continue
            mP, mSP = _metric(m, ys, ps), _metric(m, ys, ps, ws)
            mSQ, mQ = _metric(m, yt, pt, wt), _metric(m, yt, pt)
            out[(model, m)] = (mP, mSP, mSQ, mQ)
    ess_s = ws.sum() ** 2 / np.sum(ws ** 2)
    ess_t = wt.sum() ** 2 / np.sum(wt ** 2)
    return out, ess_s, ess_t


def disde(Xs, ys, ps_d, Xt, yt, pt_d, B=B_SHIFT, seed=SEED, kind="LR"):
    """Decompose metric(target) - metric(source) into X-shift (source side), Y|X shift (shared support) and
    X-shift (target side) for every model in ps_d/pt_d (dicts of source OOF / target predictions).
    The domain classifier is re-estimated inside each bootstrap resample (source and target resampled)."""
    ys, yt = np.asarray(ys), np.asarray(yt)
    ps_d = {k: np.asarray(v) for k, v in ps_d.items()}
    pt_d = {k: np.asarray(v) for k, v in pt_d.items()}
    pis, pit, dauc = domain_probs(Xs, Xt, seed, kind)
    point, ess_s, ess_t = _disde_once(ys, ps_d, yt, pt_d, pis, pit, len(ys), len(yt))
    rng = np.random.default_rng(seed)
    boots = {k: [] for k in point}
    for b in range(B):
        i_s = rng.integers(0, len(ys), len(ys))
        i_t = rng.integers(0, len(yt), len(yt))
        bps, bpt, _ = domain_probs(Xs.iloc[i_s], Xt.iloc[i_t], seed + b + 1, kind)
        o, _, _ = _disde_once(ys[i_s], {k: v[i_s] for k, v in ps_d.items()}, yt[i_t],
                              {k: v[i_t] for k, v in pt_d.items()}, bps, bpt, len(i_s), len(i_t))
        for k in point:
            boots[k].append(o[k])
    rows = []
    for (model, m), (mP, mSP, mSQ, mQ) in point.items():
        comps = {"total": mQ - mP, "X_shift_source_side": mSP - mP, "YgivenX_shift": mSQ - mSP,
                 "X_shift_target_side": mQ - mSQ}
        bb = np.array(boots[(model, m)], float)
        bcomp = {"total": bb[:, 3] - bb[:, 0], "X_shift_source_side": bb[:, 1] - bb[:, 0],
                 "YgivenX_shift": bb[:, 2] - bb[:, 1], "X_shift_target_side": bb[:, 3] - bb[:, 2]}
        for k, v in comps.items():
            s = bcomp[k][np.isfinite(bcomp[k])]
            # the prevalence decomposition uses observed outcomes only (no model): label it model-free (round 2)
            rows.append({"model": "model-free" if m == "Prevalence" else model, "metric": m, "component": k,
                         "estimate": v,
                         "lo": np.percentile(s, 2.5) if len(s) else np.nan,
                         "hi": np.percentile(s, 97.5) if len(s) else np.nan,
                         "source_value": mP, "shared_source_value": mSP, "shared_target_value": mSQ,
                         "target_value": mQ})
    info = {"domain_auroc": dauc, "ess_source_shared": ess_s, "ess_target_shared": ess_t,
            "n_source": len(ys), "n_target": len(yt), "B": B, "classifier": kind}
    return pd.DataFrame(rows), info


# ---- case-mix benchmark ----------------------------------------------------------------------

def casemix_auroc(p, B=200, seed=SEED):
    """Model-based expected AUROC if the model were calibrated in this population (Vergouwe et al.)."""
    rng = np.random.default_rng(seed)
    p = np.asarray(p, float)
    return float(np.mean([auroc(rng.binomial(1, p), p) for _ in range(B)]))


# ---- full importance weighting: truncation sensitivity and balance ---------------------------

def iw_sensitivity(Xs, ys, ps, Xt, yt, pt, caps=(95, 99, 99.9, None)):
    rows = []
    for kind in ("LR", "LGBM"):
        pis, pit, dauc = domain_probs(Xs, Xt, SEED, kind)
        r = density_ratio(pis, len(ys), len(yt))
        Zs = Xs.fillna(Xs.median())
        Zt = Xt.fillna(Xs.median())
        sd = np.sqrt((Zs.var() + Zt.var()) / 2).replace(0, np.nan)
        smd_before = ((Zt.mean() - Zs.mean()) / sd).abs()
        for cap in caps:
            w = np.minimum(r, np.percentile(r, cap)) if cap is not None else r.copy()
            w = w / w.mean()
            mw = (Zs.mul(w, axis=0)).sum() / w.sum()
            smd_after = ((Zt.mean() - mw) / sd).abs()
            rows.append({"classifier": kind, "domain_auroc": dauc, "cap_pct": "none" if cap is None else cap,
                         "ESS": w.sum() ** 2 / np.sum(w ** 2), "max_w": float(w.max()),
                         "prev_dev": float(np.mean(ys)), "prev_weighted": float(np.average(ys, weights=w)),
                         "prev_target": float(np.mean(yt)),
                         "max_abs_smd_before": float(smd_before.max()), "n_smd_gt_0.1_before": int((smd_before > 0.1).sum()),
                         "max_abs_smd_after": float(smd_after.max()), "n_smd_gt_0.1_after": int((smd_after > 0.1).sum()),
                         "weighted_CITL": citl(ys, np.clip(ps, EPS, 1 - EPS), w),
                         "weighted_Slope": cal_slope(ys, np.clip(ps, EPS, 1 - EPS), w),
                         "weighted_AUROC": auroc(ys, ps, w),
                         "target_CITL": citl(yt, np.clip(pt, EPS, 1 - EPS)), "target_Slope": cal_slope(yt, pt),
                         "target_AUROC": auroc(yt, pt)})
    return pd.DataFrame(rows)
