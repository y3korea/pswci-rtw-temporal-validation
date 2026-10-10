"""Discrimination, calibration, overall and clinical-utility metrics with bootstrap CIs.

All predictions p are P(employed at wave 2). Optional sample weights w are supported
(survey weights, IPCW, importance weights).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expit(z):
    return 1.0 / (1.0 + np.exp(-z))


def _w(y, w):
    return np.ones(len(y)) if w is None else np.asarray(w, float)


def logistic_recal(y, lp, w=None, slope=True, n_iter=50):
    """Newton-Raphson fit of logit P(y) = a + b*lp (slope=True) or a + lp (offset, slope=False)."""
    y = np.asarray(y, float)
    lp = np.asarray(lp, float)
    w = _w(y, w)
    if slope:
        X = np.column_stack([np.ones_like(lp), lp])
        beta = np.array([0.0, 1.0])
        off = 0.0
    else:
        X = np.ones((len(lp), 1))
        beta = np.array([0.0])
        off = lp
    for _ in range(n_iter):
        eta = X @ beta + off
        mu = expit(eta)
        g = X.T @ (w * (y - mu))
        H = (X * (w * mu * (1 - mu))[:, None]).T @ X
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def citl(y, p, w=None):
    return float(logistic_recal(y, logit(p), w, slope=False)[0])


def cal_slope(y, p, w=None):
    lp = logit(p)
    if np.std(lp) < 1e-8:          # (near-)constant predictions: slope undefined
        return np.nan
    return float(logistic_recal(y, lp, w, slope=True)[1])


def brier(y, p, w=None):
    w = _w(y, w)
    return float(np.sum(w * (np.asarray(p) - np.asarray(y)) ** 2) / np.sum(w))


def scaled_brier(y, p, w=None):
    w = _w(y, w)
    ybar = np.sum(w * y) / np.sum(w)
    return 1.0 - brier(y, p, w) / (ybar * (1 - ybar))


def ece(y, p, w=None, bins=10):
    """Expected calibration error with equal-frequency bins of the predictions."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    w = _w(y, w)
    order = np.argsort(p, kind="mergesort")
    tot = np.sum(w)
    e = 0.0
    for idx in np.array_split(order, bins):
        if len(idx) == 0:
            continue
        wb = w[idx]
        sw = wb.sum()
        e += sw / tot * abs(np.sum(wb * y[idx]) / sw - np.sum(wb * p[idx]) / sw)
    return float(e)


def auroc(y, p, w=None):
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p, sample_weight=w))


def auprc_minority(y, p, task, w=None):
    """AUPRC for the minority class: task A = non-employment, task B = re-entry."""
    if task == "A":
        return float(average_precision_score(1 - np.asarray(y), 1 - np.asarray(p), sample_weight=w))
    return float(average_precision_score(y, p, sample_weight=w))


def accuracy(y, p, w=None, thr=0.5):
    w = _w(y, w)
    return float(np.sum(w * ((np.asarray(p) >= thr).astype(int) == np.asarray(y))) / np.sum(w))


def all_metrics(y, p, task, w=None) -> dict:
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    wv = _w(y, w)
    return {
        "AUROC": auroc(y, p, w),
        "AUPRC": auprc_minority(y, p, task, w),
        "Brier": brier(y, p, w),
        "sBrier": scaled_brier(y, p, w),
        "CITL": citl(y, p, w),
        "Slope": cal_slope(y, p, w),
        "ECE": ece(y, p, w),
        "Accuracy": accuracy(y, p, w),
        "MeanPred": float(np.sum(wv * p) / np.sum(wv)),
        "Observed": float(np.sum(wv * y) / np.sum(wv)),
    }


CORE = ["AUROC", "AUPRC", "Brier", "sBrier", "CITL", "Slope", "ECE", "Accuracy", "MeanPred", "Observed"]


def bootstrap(y, preds: dict, task, B=1000, seed=2026, w=None, diffs_vs=("Persistence", "LR-5"),
              metrics=("AUROC", "AUPRC", "Brier", "sBrier", "CITL", "Slope", "ECE", "Accuracy")):
    """Bootstrap the test set (models fixed). Returns summary table with point, lo, hi.

    Paired differences (model - reference) for AUROC, Brier and Accuracy vs each reference in diffs_vs.
    """
    y = np.asarray(y, int)
    n = len(y)
    rng = np.random.default_rng(seed)
    point = {m: all_metrics(y, p, task, w) for m, p in preds.items()}
    samples = {m: {k: [] for k in metrics} for m in preds}
    dsamp = {}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        wb = None if w is None else np.asarray(w)[idx]
        res = {}
        for m, p in preds.items():
            pb = np.asarray(p)[idx]
            r = {}
            for k in metrics:
                if k == "AUROC":
                    r[k] = auroc(yb, pb, wb)
                elif k == "AUPRC":
                    r[k] = auprc_minority(yb, pb, task, wb)
                elif k == "Brier":
                    r[k] = brier(yb, pb, wb)
                elif k == "sBrier":
                    r[k] = scaled_brier(yb, pb, wb)
                elif k == "CITL":
                    r[k] = citl(yb, np.clip(pb, EPS, 1 - EPS), wb)
                elif k == "Slope":
                    r[k] = cal_slope(yb, np.clip(pb, EPS, 1 - EPS), wb)
                elif k == "ECE":
                    r[k] = ece(yb, pb, wb)
                elif k == "Accuracy":
                    r[k] = accuracy(yb, pb, wb)
                samples[m][k].append(r[k])
            res[m] = r
        for ref in diffs_vs:
            if ref not in res:
                continue
            for m in preds:
                if m == ref:
                    continue
                for k in ("AUROC", "Brier", "Accuracy"):
                    if k in metrics:
                        dsamp.setdefault((m, ref, k), []).append(res[m][k] - res[ref][k])
    rows = []
    for m in preds:
        for k in metrics:
            s = np.asarray(samples[m][k], float)
            rows.append({"model": m, "metric": k, "estimate": point[m][k],
                         "lo": np.nanpercentile(s, 2.5), "hi": np.nanpercentile(s, 97.5)})
        for k in ("MeanPred", "Observed"):
            rows.append({"model": m, "metric": k, "estimate": point[m][k], "lo": np.nan, "hi": np.nan})
    drows = []
    for (m, ref, k), s in dsamp.items():
        s = np.asarray(s, float)
        drows.append({"model": m, "reference": ref, "metric": k,
                      "estimate": point[m][k] - point[ref][k],
                      "lo": np.nanpercentile(s, 2.5), "hi": np.nanpercentile(s, 97.5),
                      "p_boot_le0": float(np.mean(s <= 0))})
    return pd.DataFrame(rows), pd.DataFrame(drows)


def fmt_ci(est, lo, hi, d=3):
    return f"{est:.{d}f} ({lo:.{d}f}–{hi:.{d}f})"


# ---- decision-curve analysis --------------------------------------------------------------

def net_benefit(event, flag, pt, w=None):
    """Net benefit of flagging (flag=1) for an intervention targeting `event`."""
    event = np.asarray(event, int)
    flag = np.asarray(flag, int)
    w = _w(event, w)
    n = np.sum(w)
    tp = np.sum(w * (flag == 1) * (event == 1)) / n
    fp = np.sum(w * (flag == 1) * (event == 0)) / n
    return tp - fp * pt / (1 - pt)


def dca_table(y, preds: dict, thresholds, persistence_flag=None, w=None):
    """Net benefit for flagging workers at risk of NOT achieving the outcome (risk = 1 - p)."""
    event = 1 - np.asarray(y, int)
    rows = []
    for pt in thresholds:
        rows.append({"threshold": pt, "strategy": "Treat all",
                     "net_benefit": net_benefit(event, np.ones_like(event), pt, w)})
        rows.append({"threshold": pt, "strategy": "Treat none", "net_benefit": 0.0})
        if persistence_flag is not None:
            rows.append({"threshold": pt, "strategy": "Persistence rule",
                         "net_benefit": net_benefit(event, persistence_flag, pt, w)})
        for m, p in preds.items():
            flag = ((1 - np.asarray(p)) >= pt).astype(int)
            rows.append({"threshold": pt, "strategy": m, "net_benefit": net_benefit(event, flag, pt, w)})
    return pd.DataFrame(rows)


def dca_diff_boot(y, p, ref_flag, thresholds, B=1000, seed=2026):
    """Bootstrap CI for NB(model) - NB(reference rule) at selected thresholds."""
    y = np.asarray(y, int)
    event = 1 - y
    n = len(y)
    rng = np.random.default_rng(seed)
    p = np.asarray(p)
    ref_flag = np.asarray(ref_flag)
    out = []
    est = {pt: net_benefit(event, ((1 - p) >= pt).astype(int), pt) - net_benefit(event, ref_flag, pt)
           for pt in thresholds}
    samp = {pt: [] for pt in thresholds}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        for pt in thresholds:
            samp[pt].append(net_benefit(event[idx], ((1 - p[idx]) >= pt).astype(int), pt)
                            - net_benefit(event[idx], ref_flag[idx], pt))
    for pt in thresholds:
        s = np.asarray(samp[pt])
        out.append({"threshold": pt, "delta_nb": est[pt], "lo": np.percentile(s, 2.5),
                    "hi": np.percentile(s, 97.5)})
    return pd.DataFrame(out)


def lowess_curve(y, p, frac=2 / 3, grid=None):
    from statsmodels.nonparametric.smoothers_lowess import lowess
    grid = np.linspace(np.quantile(p, 0.01), np.quantile(p, 0.99), 50) if grid is None else grid
    fit = lowess(np.asarray(y, float), np.asarray(p, float), frac=frac, it=0, xvals=grid)
    return grid, fit


def decile_points(y, p, bins=10):
    order = np.argsort(p)
    out = []
    for idx in np.array_split(order, bins):
        out.append((float(np.mean(np.asarray(p)[idx])), float(np.mean(np.asarray(y)[idx])), len(idx)))
    return pd.DataFrame(out, columns=["pred", "obs", "n"])


def wilson(k, n, alpha=0.05):
    from statsmodels.stats.proportion import proportion_confint
    lo, hi = proportion_confint(k, n, alpha=alpha, method="wilson")
    return k / n, lo, hi
