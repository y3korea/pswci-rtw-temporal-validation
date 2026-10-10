"""Exploratory fairness audit (revision v2).

Flag = predicted risk of the undesired event (1 - p) >= threshold (development prevalence of the event).
Per group: AUROC, CITL, O/E, TPR (flagged | event), FPR (flagged | no event), FNR, PPV, NPV, and the TPR implied by
the model's own probabilities if it were calibrated in the group (sufficiency check, cf. Kleinberg et al.).
Gaps vs a reference group with bootstrap CIs and two-sided bootstrap p-values -> Benjamini-Hochberg q-values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rtw_metrics import EPS, auroc, bh_qvalues, boot_indices, citl

FAIR_GROUPS = {"Sex": ("sex_grp", "Male"), "Age group": ("age_lab", "<=39"),
               "Disability": ("disab_grp", "Mild (8-14)"), "Accident type": ("acc_grp", "Injury"),
               "Job status": ("job_grp", "Regular")}
GAP_METRICS = ("AUROC", "CITL", "TPR", "FPR", "PPV", "NPV")


def _gm(y, p, w, pt):
    event = (y == 0)
    r = 1 - p
    flag = r >= pt
    wv = np.ones(len(y)) if w is None else w
    def wm(mask, val):
        return np.sum(wv[mask] * val[mask]) / np.sum(wv[mask]) if np.sum(wv[mask]) > 0 else np.nan
    out = {"n": int(len(y)), "observed": float(np.average(y, weights=wv)), "mean_pred": float(np.average(p, weights=wv))}
    out["OE"] = out["observed"] / out["mean_pred"]
    out["AUROC"] = auroc(y, p, w) if 0 < y.mean() < 1 else np.nan
    out["CITL"] = citl(y, np.clip(p, EPS, 1 - EPS), w) if 0 < y.mean() < 1 else np.nan
    out["TPR"] = wm(event, flag.astype(float))
    out["FPR"] = wm(~event, flag.astype(float))
    out["FNR"] = 1 - out["TPR"] if np.isfinite(out["TPR"]) else np.nan
    out["PPV"] = wm(flag, event.astype(float))
    out["NPV"] = wm(~flag, (~event).astype(float))
    out["TPR_implied"] = float(np.sum(wv * r * flag) / np.sum(wv * r))
    out["flag_rate"] = float(np.average(flag, weights=wv))
    return out


def fairness(te, y, p, task, model, pt, B=1000, seed=2026, w=None, strata=None, label="unweighted"):
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    idxs = boot_indices(len(y), B, seed, strata)
    rows = []
    for gname, (col, ref) in FAIR_GROUPS.items():
        g = te[col].values
        levels = [lv for lv in pd.unique(g)]
        def all_levels(idx):
            res = {}
            for lv in levels:
                m = idx[g[idx] == lv]
                if len(m) >= 5:
                    res[lv] = _gm(y[m], p[m], None if w is None else np.asarray(w)[m], pt)
            return res
        point = all_levels(np.arange(len(y)))
        bs = [all_levels(i) for i in idxs]
        for lv, mv in point.items():
            r = {"task": task, "model": model, "weighting": label, "threshold": pt, "group": gname, "level": lv,
                 "reference": ref, **mv}
            for k in GAP_METRICS + ("TPR_implied",):
                s = np.array([b[lv][k] for b in bs if lv in b], float)
                r[f"{k}_lo"], r[f"{k}_hi"] = np.nanpercentile(s, 2.5), np.nanpercentile(s, 97.5)
            if lv != ref and ref in point:
                for k in GAP_METRICS:
                    gap = np.array([b[lv][k] - b[ref][k] for b in bs if lv in b and ref in b], float)
                    gap = gap[np.isfinite(gap)]
                    r[f"gap_{k}"] = mv[k] - point[ref][k]
                    r[f"gap_{k}_lo"] = np.percentile(gap, 2.5) if len(gap) else np.nan
                    r[f"gap_{k}_hi"] = np.percentile(gap, 97.5) if len(gap) else np.nan
                    r[f"gap_{k}_p"] = float(min(1, 2 * min(np.mean(gap <= 0), np.mean(gap >= 0)))) if len(gap) else np.nan
            rows.append(r)
    t = pd.DataFrame(rows)
    pcols = [c for c in t.columns if c.endswith("_p") and c.startswith("gap_")]
    allp = t[pcols].values.ravel()
    q = bh_qvalues(allp).reshape(t[pcols].shape)
    for j, c in enumerate(pcols):
        t[c.replace("_p", "_q")] = q[:, j]
    small = t["n"] < 5
    t.loc[small, [c for c in t.columns if c not in ("task", "model", "weighting", "threshold", "group", "level",
                                                     "reference", "n")]] = np.nan
    return t
