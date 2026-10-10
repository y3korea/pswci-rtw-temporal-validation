"""Discrimination, calibration, overall and utility metrics with (design-based) bootstrap CIs (revision v2).

Predictions p are P(employed at wave 2). Sample weights w (survey weights, IPCW, importance weights) are optional.
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
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


def _w(y, w):
    return np.ones(len(y)) if w is None else np.asarray(w, float)


def newton_logistic(X, y, w=None, offset=None, n_iter=60, tol=1e-9):
    """Weighted logistic regression by Newton-Raphson. Returns (beta, converged)."""
    y = np.asarray(y, float)
    w = _w(y, w)
    off = 0.0 if offset is None else np.asarray(offset, float)
    beta = np.zeros(X.shape[1])
    for _ in range(n_iter):
        mu = expit(X @ beta + off)
        g = X.T @ (w * (y - mu))
        H = (X * (w * mu * (1 - mu))[:, None]).T @ X
        try:
            step = np.linalg.solve(H + 1e-10 * np.eye(len(beta)), g)
        except np.linalg.LinAlgError:
            return beta, False
        beta = beta + step
        if not np.all(np.isfinite(beta)):
            return beta, False
        if np.max(np.abs(step)) < tol:
            return beta, True
    return beta, False


def logistic_recal(y, lp, w=None, slope=True):
    lp = np.asarray(lp, float)
    if slope:
        X = np.column_stack([np.ones_like(lp), lp])
        beta, ok = newton_logistic(X, y, w)
        return beta, ok
    beta, ok = newton_logistic(np.ones((len(lp), 1)), y, w, offset=lp)
    return beta, ok


def citl(y, p, w=None):
    b, ok = logistic_recal(y, logit(p), w, slope=False)
    return float(b[0]) if ok else np.nan


def cal_slope(y, p, w=None):
    """Calibration slope; undefined (NaN) for (near-)constant predictions, non-convergence or |slope| > 50."""
    lp = logit(p)
    if np.std(lp) < 1e-3:
        return np.nan
    b, ok = logistic_recal(y, lp, w, slope=True)
    if not ok or abs(b[1]) > 50:
        return np.nan
    return float(b[1])


def brier(y, p, w=None):
    w = _w(y, w)
    return float(np.sum(w * (np.asarray(p) - np.asarray(y)) ** 2) / np.sum(w))


def scaled_brier(y, p, w=None):
    w = _w(y, w)
    ybar = np.sum(w * y) / np.sum(w)
    return 1.0 - brier(y, p, w) / (ybar * (1 - ybar))


def logloss(y, p, w=None):
    w = _w(y, w)
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))) / np.sum(w))


def ece(y, p, w=None, bins=10):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    w = _w(y, w)
    order = np.argsort(p, kind="mergesort")
    tot = np.sum(w)
    e = 0.0
    for idx in np.array_split(order, bins):
        if len(idx):
            wb = w[idx]
            e += wb.sum() / tot * abs(np.sum(wb * y[idx]) / wb.sum() - np.sum(wb * p[idx]) / wb.sum())
    return float(e)


def _rcs_basis(x, knots):
    """Restricted cubic spline basis (Harrell): x plus len(knots)-2 non-linear terms."""
    k = np.asarray(knots, float)
    t = len(k)
    cols = [x]
    norm = (k[-1] - k[0]) ** 2
    for j in range(t - 2):
        term = (np.maximum(x - k[j], 0) ** 3
                - np.maximum(x - k[t - 2], 0) ** 3 * (k[t - 1] - k[j]) / (k[t - 1] - k[t - 2])
                + np.maximum(x - k[t - 1], 0) ** 3 * (k[t - 2] - k[j]) / (k[t - 1] - k[t - 2]))
        cols.append(term / norm)
    return np.column_stack(cols)


def calibration_curve_fit(y, p, w=None):
    """Flexible calibration: logistic regression of y on an RCS of logit(p) (4 knots). Returns a function."""
    lp = logit(p)
    if np.std(lp) < 1e-3:
        return None
    knots = np.quantile(lp, [0.05, 0.35, 0.65, 0.95])
    if len(np.unique(knots)) < 4:
        knots = np.quantile(lp, [0.05, 0.95])
        B = lambda z: np.column_stack([np.ones_like(z), z])  # noqa: E731
    else:
        B = lambda z: np.column_stack([np.ones_like(z), _rcs_basis(z, knots)])  # noqa: E731
    beta, ok = newton_logistic(B(lp), y, w)
    if not ok:
        return None
    return lambda q: expit(B(logit(q)) @ beta)


def ici(y, p, w=None):
    """Integrated calibration index and E50/E90 (Austin & Steyerberg) from the spline calibration curve."""
    f = calibration_curve_fit(y, p, w)
    if f is None:
        return np.nan, np.nan, np.nan
    d = np.abs(f(p) - np.asarray(p))
    wv = _w(y, w)
    order = np.argsort(d)
    cw = np.cumsum(wv[order]) / wv.sum()
    e50 = d[order][np.searchsorted(cw, 0.5)]
    e90 = d[order][min(np.searchsorted(cw, 0.9), len(d) - 1)]
    return float(np.sum(wv * d) / wv.sum()), float(e50), float(e90)


def auroc(y, p, w=None):
    if len(np.unique(y)) < 2 or np.ptp(np.asarray(p, float)) < 1e-12:
        return np.nan if len(np.unique(y)) < 2 else 0.5
    return float(roc_auc_score(y, p, sample_weight=w))


def minority_event(task):
    """AUPRC is computed for the minority class: A1 job loss, A2 re-entry, P non-employment."""
    return "re-entry" if task in ("A2", "B") else "not employed"


def auprc(y, p, task, w=None):
    y = np.asarray(y)
    p = np.asarray(p)
    if task in ("A2", "B"):
        return float(average_precision_score(y, p, sample_weight=w))
    return float(average_precision_score(1 - y, 1 - p, sample_weight=w))


def auprc_noskill(y, task, w=None):
    wv = _w(y, w)
    prev = np.sum(wv * y) / wv.sum()
    return float(prev if task in ("A2", "B") else 1 - prev)


METRICS = ("AUROC", "AUPRC", "Brier", "sBrier", "LogLoss", "CITL", "Slope", "ICI", "E50", "E90", "ECE")


def all_metrics(y, p, task, w=None, which=METRICS) -> dict:
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    wv = _w(y, w)
    out = {}
    if any(k in which for k in ("ICI", "E50", "E90")):
        out["ICI"], out["E50"], out["E90"] = ici(y, p, w)
    for k in which:
        if k == "AUROC":
            out[k] = auroc(y, p, w)
        elif k == "AUPRC":
            out[k] = auprc(y, p, task, w)
        elif k == "Brier":
            out[k] = brier(y, p, w)
        elif k == "sBrier":
            out[k] = scaled_brier(y, p, w)
        elif k == "LogLoss":
            out[k] = logloss(y, p, w)
        elif k == "CITL":
            out[k] = citl(y, p, w)
        elif k == "Slope":
            out[k] = cal_slope(y, p, w)
        elif k == "ECE":
            out[k] = ece(y, p, w)
    out["MeanPred"] = float(np.sum(wv * p) / wv.sum())
    out["Observed"] = float(np.sum(wv * y) / wv.sum())
    out["NoSkillAUPRC"] = auprc_noskill(y, task, w)
    return {k: out[k] for k in list(which) + ["MeanPred", "Observed", "NoSkillAUPRC"]}


def boot_indices(n, B, seed, strata=None):
    """Bootstrap index sets; resampling within strata when strata is given (design-based)."""
    rng = np.random.default_rng(seed)
    if strata is None:
        return [rng.integers(0, n, n) for _ in range(B)]
    strata = np.asarray(strata)
    groups = [np.where(strata == s)[0] for s in pd.unique(strata)]
    out = []
    for _ in range(B):
        out.append(np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups]))
    return out


def evaluate(y, preds: dict, task, B=1000, seed=2026, w=None, strata=None, refs=(), which=METRICS,
             label="unweighted"):
    """Point estimates and percentile CIs for every model; paired differences vs each model in refs."""
    y = np.asarray(y, int)
    idxs = boot_indices(len(y), B, seed, strata)
    point = {m: all_metrics(y, p, task, w, which) for m, p in preds.items()}
    samples = {m: {k: [] for k in which} for m in preds}
    dsamp = {}
    for idx in idxs:
        yb = y[idx]
        wb = None if w is None else np.asarray(w)[idx]
        res = {}
        for m, p in preds.items():
            res[m] = all_metrics(yb, np.asarray(p)[idx], task, wb, which)
            for k in which:
                samples[m][k].append(res[m][k])
        for ref in refs:
            if ref not in res:
                continue
            for m in preds:
                if m != ref:
                    for k in ("AUROC", "Brier", "LogLoss", "ICI"):
                        if k in which:
                            dsamp.setdefault((m, ref, k), []).append(res[m][k] - res[ref][k])
    rows = []
    for m in preds:
        for k in which:
            s = np.asarray(samples[m][k], float)
            ok = np.isfinite(s)
            rows.append({"model": m, "metric": k, "estimate": point[m][k],
                         "lo": np.percentile(s[ok], 2.5) if ok.sum() else np.nan,
                         "hi": np.percentile(s[ok], 97.5) if ok.sum() else np.nan,
                         "n_undefined": int((~ok).sum()), "weighting": label})
        for k in ("MeanPred", "Observed", "NoSkillAUPRC"):
            rows.append({"model": m, "metric": k, "estimate": point[m][k], "lo": np.nan, "hi": np.nan,
                         "n_undefined": 0, "weighting": label})
    drows = []
    for (m, ref, k), s in dsamp.items():
        s = np.asarray(s, float)
        drows.append({"model": m, "reference": ref, "metric": k, "estimate": point[m][k] - point[ref][k],
                      "lo": np.nanpercentile(s, 2.5), "hi": np.nanpercentile(s, 97.5),
                      "p_two_sided": float(min(1.0, 2 * min(np.mean(s <= 0), np.mean(s >= 0)))),
                      "weighting": label})
    return pd.DataFrame(rows), pd.DataFrame(drows)


# ---- decision curves -------------------------------------------------------------------------

def net_benefit(event, flag, pt, w=None):
    event = np.asarray(event, int)
    flag = np.asarray(flag, int)
    w = _w(event, w)
    n = np.sum(w)
    tp = np.sum(w * (flag == 1) * (event == 1)) / n
    fp = np.sum(w * (flag == 1) * (event == 0)) / n
    return tp - fp * pt / (1 - pt)


def dca_curves(y, preds: dict, thresholds, rule_flags: dict | None = None, w=None):
    """Net benefit of flagging for the undesired event (1 - y) with risk = 1 - p."""
    event = 1 - np.asarray(y, int)
    rows = []
    for pt in thresholds:
        rows.append({"threshold": pt, "strategy": "Flag all", "net_benefit": net_benefit(event, np.ones_like(event), pt, w)})
        rows.append({"threshold": pt, "strategy": "Flag none", "net_benefit": 0.0})
        for r, fl in (rule_flags or {}).items():
            rows.append({"threshold": pt, "strategy": r, "net_benefit": net_benefit(event, fl, pt, w)})
        for m, p in preds.items():
            rows.append({"threshold": pt, "strategy": m,
                         "net_benefit": net_benefit(event, ((1 - np.asarray(p)) >= pt).astype(int), pt, w)})
    return pd.DataFrame(rows)


def dca_vs_best_default(y, p, thresholds, rule_flags: dict | None = None, w=None, strata=None, B=1000, seed=2026):
    """NB(model) - max(NB(flag all), 0, NB(rules)) with bootstrap CIs (design-based if strata)."""
    y = np.asarray(y, int)
    event = 1 - y
    p = np.asarray(p)
    rule_flags = {k: np.asarray(v) for k, v in (rule_flags or {}).items()}

    def delta(idx, pt):
        ev = event[idx]
        wb = None if w is None else np.asarray(w)[idx]
        nb_m = net_benefit(ev, ((1 - p[idx]) >= pt).astype(int), pt, wb)
        defaults = [0.0, net_benefit(ev, np.ones_like(ev), pt, wb)] + [net_benefit(ev, f[idx], pt, wb)
                                                                       for f in rule_flags.values()]
        return nb_m - max(defaults), nb_m
    full = np.arange(len(y))
    idxs = boot_indices(len(y), B, seed, strata)
    rows = []
    for pt in thresholds:
        est, nbm = delta(full, pt)
        s = np.array([delta(i, pt)[0] for i in idxs])
        rows.append({"threshold": pt, "nb_model": nbm, "delta_vs_best_default": est,
                     "lo": np.percentile(s, 2.5), "hi": np.percentile(s, 97.5)})
    return pd.DataFrame(rows)


def calibration_band(y, p, w=None, B=200, seed=2026, strata=None, grid=None):
    """Spline calibration curve on a grid with pointwise percentile bands."""
    p = np.asarray(p, float)
    grid = np.linspace(np.quantile(p, 0.02), np.quantile(p, 0.98), 40) if grid is None else grid
    f = calibration_curve_fit(y, p, w)
    est = f(grid) if f is not None else np.full(len(grid), np.nan)
    curves = []
    for idx in boot_indices(len(y), B, seed, strata):
        fb = calibration_curve_fit(np.asarray(y)[idx], p[idx], None if w is None else np.asarray(w)[idx])
        if fb is not None:
            curves.append(fb(grid))
    curves = np.array(curves)
    return pd.DataFrame({"pred": grid, "obs": est, "lo": np.nanpercentile(curves, 2.5, axis=0),
                         "hi": np.nanpercentile(curves, 97.5, axis=0)})


def wilson(k, n, alpha=0.05):
    from statsmodels.stats.proportion import proportion_confint
    lo, hi = proportion_confint(k, n, alpha=alpha, method="wilson")
    return k / n, lo, hi


def bh_qvalues(p):
    p = np.asarray(p, float)
    n = np.sum(np.isfinite(p))
    q = np.full(len(p), np.nan)
    idx = np.where(np.isfinite(p))[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order] * n / (np.arange(1, n + 1))
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.minimum(ranked, 1)
    return q
