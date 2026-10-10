"""Model updating (revision v2).

(i) Label-free importance-weighted recalibration (IW-recal): a logistic recalibration a + b*logit(p) fitted on the
    source out-of-fold predictions, weighted by density ratios to the UNLABELLED target covariates; no target outcomes.
(ii) Lagged labelled updating: model developed in C1, updated with random C2 samples, evaluated in C3.
(iii) Within-C3 split design (v1 design, corrected): update half / evaluation half of C3.
Closed testing (Vergouwe et al. 2017) chooses among no update, intercept and logistic recalibration by LR tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import chi2

from rtw_config import B_SHIFT, N_JOBS, SEED
from rtw_metrics import (EPS, auroc, brier, cal_slope, citl, expit, ici, logistic_recal, logit, newton_logistic)
from rtw_models import fit, ridge_cv
from rtw_shift import density_ratio, domain_probs

UPD_METRICS = ("AUROC", "Brier", "CITL", "Slope", "ICI")


def _m(y, p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    i, _, _ = ici(y, p)
    return {"AUROC": auroc(y, p), "Brier": brier(y, p), "CITL": citl(y, p), "Slope": cal_slope(y, p), "ICI": i}


def recal_params(y, p, w=None, kind="logistic"):
    lp = logit(p)
    if kind == "intercept":
        b, ok = logistic_recal(y, lp, w, slope=False)
        return (float(b[0]), 1.0) if ok else (0.0, 1.0)
    b, ok = logistic_recal(y, lp, w, slope=True)
    return (float(b[0]), float(b[1])) if ok else (0.0, 1.0)


def apply_recal(p, ab):
    return expit(ab[0] + ab[1] * logit(p))


def closed_test(y, p, alpha=0.05):
    """Vergouwe et al. (2017) closed testing among no update (M0), intercept (M1), logistic recalibration (M2)."""
    y = np.asarray(y, float)
    lp = logit(p)
    def ll(mu):
        mu = np.clip(mu, EPS, 1 - EPS)
        return np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu))
    l0 = ll(expit(lp))
    b1, _ = logistic_recal(y, lp, None, slope=False)
    l1 = ll(expit(b1[0] + lp))
    b2, _ = logistic_recal(y, lp, None, slope=True)
    l2 = ll(expit(b2[0] + b2[1] * lp))
    if chi2.sf(2 * (l2 - l0), 2) >= alpha:
        return "none", (0.0, 1.0)
    if chi2.sf(2 * (l2 - l1), 1) < alpha:
        return "logistic", (float(b2[0]), float(b2[1]))
    return "intercept", (float(b1[0]), 1.0)


# ---- (i) label-free importance-weighted recalibration -----------------------------------------

def iw_recal_eval(Xs, ys, ps_d, Xt, yt, pt_d, B=B_SHIFT, seed=SEED, cap=99):
    """For every model: no update, unweighted OOF recalibration, IW intercept and IW logistic recalibration
    (no target labels used). Bootstrap resamples source and target; domain weights and recalibrations are
    re-estimated in every resample."""
    ys, yt = np.asarray(ys), np.asarray(yt)
    ps_d = {k: np.asarray(v) for k, v in ps_d.items()}
    pt_d = {k: np.asarray(v) for k, v in pt_d.items()}

    def once(i_s, i_t, sd):
        pis, pit, _ = domain_probs(Xs.iloc[i_s], Xt.iloc[i_t], sd)
        r = density_ratio(pis, len(i_s), len(i_t))
        w = np.minimum(r, np.percentile(r, cap))
        w = w / w.mean()
        y_s, y_t = ys[i_s], yt[i_t]
        res, pars = {}, {}
        for model in ps_d:
            p_s, p_t = ps_d[model][i_s], pt_d[model][i_t]
            ab = recal_params(y_s, p_s, w)
            preds = {"No update": p_t,
                     "OOF recalibration (unweighted)": apply_recal(p_t, recal_params(y_s, p_s)),
                     "IW intercept recalibration": apply_recal(p_t, recal_params(y_s, p_s, w, "intercept")),
                     "IW logistic recalibration": apply_recal(p_t, ab)}
            res[model] = {k: _m(y_t, v) for k, v in preds.items()}
            pars[model] = ab
        return res, pars
    point, pars = once(np.arange(len(ys)), np.arange(len(yt)), seed)
    rng = np.random.default_rng(seed + 7)
    boots = [once(rng.integers(0, len(ys), len(ys)), rng.integers(0, len(yt), len(yt)), seed + b + 11)[0]
             for b in range(B)]
    rows = []
    for model in point:
        for meth in point[model]:
            for k in UPD_METRICS:
                s = np.array([bb[model][meth][k] for bb in boots], float)
                s0 = np.array([bb[model]["No update"][k] for bb in boots], float)
                d = s - s0
                if k in ("CITL",):
                    d = np.abs(s) - np.abs(s0)
                if k == "Slope":
                    d = np.abs(s - 1) - np.abs(s0 - 1)
                okd = np.isfinite(d)
                est_d = point[model][meth][k] - point[model]["No update"][k]
                if k == "CITL":
                    est_d = abs(point[model][meth][k]) - abs(point[model]["No update"][k])
                if k == "Slope":
                    est_d = abs(point[model][meth][k] - 1) - abs(point[model]["No update"][k] - 1)
                rows.append({"model": model, "method": meth, "metric": k, "estimate": point[model][meth][k],
                             "lo": np.nanpercentile(s, 2.5), "hi": np.nanpercentile(s, 97.5),
                             "improvement_vs_none": est_d,
                             "imp_lo": np.percentile(d[okd], 2.5) if okd.sum() else np.nan,
                             "imp_hi": np.percentile(d[okd], 97.5) if okd.sum() else np.nan,
                             "iw_a": pars[model][0], "iw_b": pars[model][1]})
    return pd.DataFrame(rows)


# ---- (ii) lagged labelled updating ------------------------------------------------------------

def _lag_rep(r, model, params, X1, y1, X2, y2, p2, X3, y3, p3, Xoh2, Xoh3, sizes):
    rng = np.random.default_rng(SEED + 3000 + r)
    rows = []

    def rec(n, meth, p, extra=None):
        d = {"rep": r, "n": n, "method": meth, **_m(y3, p)}
        if extra:
            d.update(extra)
        rows.append(d)
    for n in sizes:
        if n == "all":
            if r > 0:
                continue
            idx = np.arange(len(y2))
        else:
            idx = rng.choice(len(y2), n, replace=False)
        nn = len(idx)
        rec(nn, "Intercept recalibration", apply_recal(p3, recal_params(y2[idx], p2[idx], kind="intercept")))
        rec(nn, "Logistic recalibration", apply_recal(p3, recal_params(y2[idx], p2[idx])))
        choice, ab = closed_test(y2[idx], p2[idx])
        rec(nn, "Closed testing", apply_recal(p3, ab), {"choice": choice})
        est = fit(model, params, pd.concat([X1, X2.iloc[idx]]), np.r_[y1, y2[idx]], n_jobs=1)
        rec(nn, "Pooled refit (C1 + C2 sample)", est.predict_proba(X3)[:, 1])
        try:
            rr = ridge_cv(Xoh2.iloc[idx], y2[idx], n_jobs=1, seed=SEED + r)
            rec(nn, "Ridge refit on C2 sample only", rr.predict_proba(Xoh3)[:, 1])
        except Exception:  # noqa: BLE001 - e.g. a single class in a tiny sample
            pass
    return rows


def lagged_updating(model, params, X1, y1, X2, y2, X3, y3, Xoh2, Xoh3, base_est, R=50,
                    sizes=(100, 250, 500, 1000, "all")):
    p2 = base_est.predict_proba(X2)[:, 1]
    p3 = base_est.predict_proba(X3)[:, 1]
    sizes = [s for s in sizes if s == "all" or s <= len(y2)]
    reps = Parallel(n_jobs=N_JOBS)(delayed(_lag_rep)(r, model, params, X1, y1, X2, y2, p2, X3, y3, p3, Xoh2, Xoh3,
                                                     sizes) for r in range(R))
    long = pd.DataFrame([row for rep in reps for row in rep])
    base = _m(y3, p3)
    long = pd.concat([pd.DataFrame([{"rep": 0, "n": 0, "method": "No update (C1 model)", **base}]), long],
                     ignore_index=True)
    return long


# ---- (iii) within-C3 split design --------------------------------------------------------------

def _c3_rep(r, model, params, Xd, yd, Xt, yt, p_base, Xoht, sizes):
    rng = np.random.default_rng(SEED + 1000 + r)
    pos, neg = np.where(yt == 1)[0], np.where(yt == 0)[0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    pool = np.r_[pos[: len(pos) // 2], neg[: len(neg) // 2]]
    hold = np.r_[pos[len(pos) // 2:], neg[len(neg) // 2:]]
    yh = yt[hold]
    rows = [{"rep": r, "n": 0, "method": "No update", **_m(yh, p_base[hold])}]
    for n in sizes:
        if n > len(pool):
            continue
        sub = rng.choice(pool, n, replace=False)
        for kind, lab in (("intercept", "Intercept recalibration"), ("logistic", "Logistic recalibration")):
            rows.append({"rep": r, "n": n, "method": lab,
                         **_m(yh, apply_recal(p_base[hold], recal_params(yt[sub], p_base[sub], kind=kind)))})
        ch, ab = closed_test(yt[sub], p_base[sub])
        rows.append({"rep": r, "n": n, "method": "Closed testing", "choice": ch, **_m(yh, apply_recal(p_base[hold], ab))})
        est = fit(model, params, Xt.iloc[sub], yt[sub], n_jobs=1)
        rows.append({"rep": r, "n": n, "method": "Refit on C3 sample (development hyperparameters)",
                     **_m(yh, est.predict_proba(Xt.iloc[hold])[:, 1])})
        try:
            rr = ridge_cv(Xoht.iloc[sub], yt[sub], n_jobs=1, seed=SEED + r)
            rows.append({"rep": r, "n": n, "method": "Ridge refit on C3 sample (CV-tuned)",
                         **_m(yh, rr.predict_proba(Xoht.iloc[hold])[:, 1])})
        except Exception:  # noqa: BLE001
            pass
        est = fit(model, params, pd.concat([Xd, Xt.iloc[sub]]), np.r_[yd, yt[sub]], n_jobs=1)
        rows.append({"rep": r, "n": n, "method": "Pooled refit (dev + C3 sample)",
                     **_m(yh, est.predict_proba(Xt.iloc[hold])[:, 1])})
    return rows


def within_c3_updating(model, params, Xd, yd, Xt, yt, p_base, Xoht, R=50, sizes=(100, 250, 500, 1000)):
    reps = Parallel(n_jobs=N_JOBS)(delayed(_c3_rep)(r, model, params, Xd, yd, Xt, yt, p_base, Xoht, sizes)
                                   for r in range(R))
    return pd.DataFrame([row for rep in reps for row in rep])


def summarise_updating(long, baseline_label):
    """Median and 2.5-97.5 percentiles; paired comparisons with the baseline of the same repetition."""
    base = long[long["method"] == baseline_label].set_index("rep")
    rows = []
    for (meth, n), d in long.groupby(["method", "n"], sort=False):
        d = d.set_index("rep")
        if len(base) == 1:
            b = pd.DataFrame([base.iloc[0]] * len(d), index=d.index)
        else:
            b = base.reindex(d.index)
        row = {"method": meth, "n": int(n), "reps": len(d)}
        for k in UPD_METRICS:
            v = d[k].astype(float)
            row[f"{k}_median"] = v.median()
            row[f"{k}_p2_5"] = v.quantile(0.025)
            row[f"{k}_p97_5"] = v.quantile(0.975)
            row[f"{k}_n_undefined"] = int(v.isna().sum())
            row[f"{k}_median_diff_vs_base"] = (v - b[k].astype(float)).median()
        row["share_better_absCITL"] = float(np.mean(d["CITL"].abs() < b["CITL"].abs()))
        row["share_better_slope"] = float(np.mean((d["Slope"] - 1).abs() < (b["Slope"] - 1).abs()))
        row["share_better_Brier"] = float(np.mean(d["Brier"] < b["Brier"]))
        row["share_better_ICI"] = float(np.mean(d["ICI"] < b["ICI"]))
        row["mean_absCITL"] = float(d["CITL"].abs().mean())
        if "choice" in d:
            ch = d["choice"].dropna()
            for c in ("none", "intercept", "logistic"):
                row[f"closed_choice_{c}"] = float(np.mean(ch == c)) if len(ch) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
