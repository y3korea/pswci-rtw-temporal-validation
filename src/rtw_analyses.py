"""Analyses for revision v2 (writes aggregated CSV tables only)."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr, weightedtau

from rtw_config import (B_BOOT, B_SHAP, DCA_THRESHOLDS, FAIR_THRESHOLD, N_JOBS, NOM_LEVELS, PREDICTORS,
                        RESTRICTED_DROP, RESTRICTED_DROP_B, SEED, VAR_LABELS)
from rtw_data import design_matrix, small_vars, task_frame
from rtw_develop import designs_for
from rtw_metrics import (EPS, all_metrics, boot_indices, cal_slope, calibration_band, citl, dca_curves,
                         dca_vs_best_default, evaluate, logloss)
from rtw_models import ONEHOT_MODELS, aggregate_attr, attributions, fit, make_folds, oof_predict

QUICK = ("AUROC", "Brier", "LogLoss", "CITL", "Slope", "ICI")


def T(out, name):
    return os.path.join(out, "tables", name)


def _suppress(df, k=5):
    if "n" in df:
        small = df["n"] < k
        keep = {"task", "model", "group", "level", "cohort", "n", "weighting", "design", "analysis"}
        df.loc[small, [c for c in df.columns if c not in keep]] = np.nan
    return df


# =============================================================================================
# Descriptives and sample size
# =============================================================================================

def descriptives(a, out):
    s = a[a["part2"] == 1]
    rows = []

    def add(label, fn, kind="pct"):
        r = {"characteristic": label, "kind": kind}
        for c in (1, 2, 3):
            d = s[s["cohort"] == c]
            v = np.asarray(fn(d), float)
            w = d["wt_w2"].values
            ok = ~np.isnan(v)
            r[f"C{c}_unw"] = (100 if kind == "pct" else 1) * np.nanmean(v)
            r[f"C{c}_w"] = (100 if kind == "pct" else 1) * np.sum(w[ok] * v[ok]) / np.sum(w[ok])
        rows.append(r)
    for c in (1, 2, 3):
        pass
    n1 = {"characteristic": "Wave-1 participants, n", "kind": "n"}
    n2 = {"characteristic": "Analysed (responded at wave 2), n", "kind": "n"}
    rr = {"characteristic": "Wave-2 response, %", "kind": "pct_only"}
    for c in (1, 2, 3):
        n1[f"C{c}_unw"] = int((a["cohort"] == c).sum())
        n2[f"C{c}_unw"] = int(((a["cohort"] == c) & (a["part2"] == 1)).sum())
        rr[f"C{c}_unw"] = 100 * n2[f"C{c}_unw"] / n1[f"C{c}_unw"]
    rows += [n1, n2, rr]
    add("Female", lambda d: d["female"])
    add("Age >=60", lambda d: d["age_grp"] == 4)
    add("College or higher education", lambda d: d["edu"] == 5)
    add("Occupational disease", lambda d: d["disease"])
    add("Commuting injury", lambda d: d["commuting"])
    add("Disability grade 1-7", lambda d: d["disab_cat"].isin([1, 2]))
    add("No disability grade", lambda d: d["disab_cat"] == 6)
    add("Vocational/employment-support service", lambda d: d["svc_job"])
    add("Regular job at injury", lambda d: d["job_status"] == 1)
    add("Work ability, mean (0-10)", lambda d: d["work_ability"], kind="mean")
    add("High pain interference", lambda d: (d["pain_interf"] >= 3).astype(float).where(d["pain_interf"].notna()))
    add("Physical limitation", lambda d: d["phys_limitation"])
    add("Chronic disease", lambda d: d["chronic"])
    add("Employed at wave 1", lambda d: d["employed"])
    add("Employed at wave 2", lambda d: d["y"])
    add("A1: retained among employed at wave 1", lambda d: d["y"].where(d["employed"] == 1))
    add("A2: re-entered among not employed at wave 1", lambda d: d["y"].where(d["employed"] == 0))
    t = pd.DataFrame(rows)
    t.to_csv(T(out, "table1_characteristics.csv"), index=False)
    # task counts
    cnt = []
    for task in ("A1", "A2", "P"):
        d = task_frame(a, task)
        for c in (1, 2, 3):
            x = d[d.cohort == c]
            cnt.append({"task": task, "cohort": c, "n": len(x), "employed_w2": int(x["y"].sum()),
                        "undesired_events": int((1 - x["y"]).sum())})
    pd.DataFrame(cnt).to_csv(T(out, "task_counts.csv"), index=False)
    return t


def riley(res_by_task, a, out):
    """Riley et al. (2020) development sample-size criteria with R2_CS anticipated from the OOF log-likelihood
    of the unpenalised LR; expected shrinkage at the actual n; validation precision (O/E)."""
    rows = []
    for task, res in res_by_task.items():
        yd = res["y_dev"]
        n = len(yd)
        phi = yd.mean()
        P = len(res["columns"]["num"])
        def ll(y, p):
            p = np.clip(p, 1e-9, 1 - 1e-9)
            return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        l0 = ll(yd, np.full(n, phi))
        l1 = ll(yd, res["oof"]["LR"])
        r2cs = max(1 - np.exp(-2 * (l1 - l0) / n), 1e-6)
        maxr2 = 1 - np.exp(2 * l0 / n)
        n1 = P / ((0.9 - 1) * np.log(1 - r2cs / 0.9))
        s2 = r2cs / (r2cs + 0.05 * maxr2)
        n2 = P / ((s2 - 1) * np.log(1 - r2cs / s2))
        n3 = (1.96 / 0.05) ** 2 * phi * (1 - phi)
        shrink = 1 + P / (n * np.log(1 - r2cs))
        yt = res["y_test"]
        ph = yt.mean()
        se = np.sqrt((1 - ph) / (len(yt) * ph))
        rows.append({"task": task, "n_dev": n, "events_min_class": int(min(yd.sum(), n - yd.sum())),
                     "parameters": P, "EPP": min(yd.sum(), n - yd.sum()) / P, "R2cs_anticipated": r2cs,
                     "max_R2cs": maxr2, "n_crit1_shrinkage0.9": n1, "n_crit2_optimism": n2, "n_crit3_intercept": n3,
                     "n_required": max(n1, n2, n3), "expected_shrinkage_at_n": shrink,
                     "lr_oof_slope": cal_slope(yd, res["oof"]["LR"]), "n_test": len(yt),
                     "test_events_min": int(min(yt.sum(), len(yt) - yt.sum())),
                     "OE_95CI_lower_factor": np.exp(-1.96 * se), "OE_95CI_upper_factor": np.exp(1.96 * se)})
    t = pd.DataFrame(rows)
    t.to_csv(T(out, "sample_size_riley.csv"), index=False)
    return t


def predictor_table(a, out):
    s = a[a["part2"] == 1]
    rows = []
    for v in PREDICTORS:
        enc = ("one-hot" if v in NOM_LEVELS else "employed + nested emp6 indicators" if v in ("employed", "emp6")
               else "score (LR, trees); one-hot in ridge-OH")
        r = {"variable": v, "label": VAR_LABELS.get(v, v), "encoding": enc,
             "S4": "dropped" if v in RESTRICTED_DROP else "kept",
             "S4b": "dropped" if v in RESTRICTED_DROP_B else "kept"}
        for c in (1, 2, 3):
            r[f"missing_C{c}_%"] = round(100 * s.loc[s["cohort"] == c, v].isna().mean(), 1)
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(T(out, "tableS1_predictors.csv"), index=False)
    return t


# =============================================================================================
# Temporal validation (primary design)
# =============================================================================================

def _ordered(res):
    order = ["Null", "Status", "LR-small", "LR", "RidgeOH", "RF", "XGB", "LGBM", "CatBoost", "EBM"]
    return {m: res["test"][m] for m in order if m in res["test"]}


def evaluate_task(res, a, out, B=B_BOOT, log=print):
    task = res["task"]
    d = task_frame(a, task)
    te = d.loc[res["test_index"]]
    yt = res["y_test"]
    preds = _ordered(res)
    refs = ("RidgeOH", "Status" if task == "P" else "LR-small")
    tabs = []
    for lab, w in (("unweighted", None), ("weighted", te["wt_w2"].values)):
        s, dd = evaluate(yt, preds, task, B=B, seed=SEED, w=w, strata=te["strata"].values, refs=refs, label=lab)
        s["task"] = task
        dd["task"] = task
        s.to_csv(T(out, f"perf_C3_{task}_{lab}.csv"), index=False)
        dd.to_csv(T(out, f"perf_C3_diffs_{task}_{lab}.csv"), index=False)
        tabs.append(s)
    dev = []
    for m, p in res["oof"].items():
        r = all_metrics(res["y_dev"], p, task)
        r.update({"model": m, "task": task, "cv_logloss": res["cv_logloss"].get(m, np.nan),
                  "repeated_cv_logloss": res["rep_logloss"].get(m, np.nan)})
        dev.append(r)
    pd.DataFrame(dev).to_csv(T(out, f"perf_devOOF_{task}.csv"), index=False)
    # slope / CITL change from development OOF to C3 (independent bootstraps of both samples)
    rng = np.random.default_rng(SEED)
    rows = []
    yd = res["y_dev"]
    bd = [rng.integers(0, len(yd), len(yd)) for _ in range(B)]
    bt = boot_indices(len(yt), B, SEED + 1, te["strata"].values)
    for m in preds:
        if m == "Null":
            continue
        po, pt = np.clip(res["oof"][m], EPS, 1 - EPS), np.clip(preds[m], EPS, 1 - EPS)
        for k, f in (("Slope", cal_slope), ("CITL", citl)):
            dv, tv = f(yd, po), f(yt, pt)
            s = np.array([f(yt[j], pt[j]) - f(yd[i], po[i]) for i, j in zip(bd, bt)], float)
            s = s[np.isfinite(s)]
            rows.append({"task": task, "model": m, "metric": k, "dev_oof": dv, "C3": tv, "change": tv - dv,
                         "lo": np.percentile(s, 2.5), "hi": np.percentile(s, 97.5)})
    pd.DataFrame(rows).to_csv(T(out, f"calibration_change_{task}.csv"), index=False)
    # decision curves
    thr = np.round(np.arange(0.02, 0.901, 0.01), 2)
    rules = {"Status rule": res["status_flag_test"]} if task == "P" else None
    keep = {m: preds[m] for m in (res["primary"], "RidgeOH", "LR-small", "Status") if m in preds}
    for lab, w in (("unweighted", None), ("weighted", te["wt_w2"].values)):
        dca_curves(yt, keep, thr, rules, w).assign(task=task, weighting=lab).to_csv(
            T(out, f"dca_{task}_{lab}.csv"), index=False)
        rows = []
        for m in (res["primary"], "RidgeOH"):
            dd = dca_vs_best_default(yt, preds[m], DCA_THRESHOLDS[task], rules, w, te["strata"].values, B=B, seed=SEED)
            dd["model"], dd["task"], dd["weighting"] = m, task, lab
            rows.append(dd)
        pd.concat(rows).to_csv(T(out, f"dca_delta_{task}_{lab}.csv"), index=False)
    # calibration curves with bands
    cal = []
    for m in dict.fromkeys([res["primary"], "RidgeOH", "LR-small"]):
        for lab, w in (("unweighted", None), ("weighted", te["wt_w2"].values)):
            cb = calibration_band(yt, preds[m], w, B=200, seed=SEED, strata=te["strata"].values)
            cb["model"], cb["weighting"], cb["task"] = m, lab, task
            cal.append(cb)
    pd.concat(cal).to_csv(T(out, f"calcurve_{task}.csv"), index=False)
    bins = np.linspace(0, 1, 41)
    hist = []
    for m in (res["primary"], "RidgeOH"):
        h, _ = np.histogram(preds[m], bins=bins)
        hist.append(pd.DataFrame({"model": m, "bin_lo": bins[:-1], "bin_hi": bins[1:], "count": h}))
    pd.concat(hist).assign(task=task).to_csv(T(out, f"predhist_{task}.csv"), index=False)
    from sklearn.metrics import roc_curve
    roc = []
    grid = np.linspace(0, 1, 201)
    for m, p in preds.items():
        fpr, tpr, _ = roc_curve(yt, p)
        roc.append(pd.DataFrame({"model": m, "fpr": grid, "tpr": np.interp(grid, fpr, tpr)}))
    pd.concat(roc).assign(task=task).to_csv(T(out, f"roc_{task}.csv"), index=False)
    log(f"[evaluate {task}] done; primary {res['primary']}")


# =============================================================================================
# Secondary designs
# =============================================================================================

def quick_eval(y, preds, task, B, label, strata=None, w=None, extra=None):
    s, _ = evaluate(y, preds, task, B=B, seed=SEED, w=w, strata=strata, refs=(), which=QUICK, label="unweighted")
    s["analysis"], s["task"], s["n"], s["events"] = label, task, len(y), int(np.sum(y))
    for k, v in (extra or {}).items():
        s[k] = v
    return s


# =============================================================================================
# Attribution stability (supplement)
# =============================================================================================

def _imp(est, X, groups, max_rows=None):
    return aggregate_attr(attributions(est, X, max_rows=max_rows), groups)


def _split(r, model, params, Xd, yd, Xt, yt, groups, vars_):
    rng = np.random.default_rng(SEED + 9000 + r)
    h = len(yt) // 2
    it, idv = rng.permutation(len(yt)), rng.permutation(len(yd))
    parts = {"C3a": (Xt, yt, it[:h]), "C3b": (Xt, yt, it[h:2 * h]), "Da": (Xd, yd, idv[:h]), "Db": (Xd, yd, idv[h:2 * h])}
    imp = {}
    for k, (X, y, idx) in parts.items():
        est = fit(model, params, X.iloc[idx], y[idx], n_jobs=1)
        imp[k] = _imp(est, X.iloc[idx], groups).reindex(vars_).fillna(0).values
    rho = lambda u, v: spearmanr(imp[u], imp[v]).statistic  # noqa: E731
    wt = lambda u, v: weightedtau(imp[u], imp[v]).statistic  # noqa: E731
    return {"rep": r, "rho_within_C3": rho("C3a", "C3b"), "rho_within_dev": rho("Da", "Db"),
            "rho_between": np.mean([rho("Da", "C3a"), rho("Db", "C3b")]),
            "wtau_within_C3": wt("C3a", "C3b"), "wtau_within_dev": wt("Da", "Db"),
            "wtau_between": np.mean([wt("Da", "C3a"), wt("Db", "C3b")])}


def attribution_stability(res, a, out, R=30, log=print):
    task = res["task"]
    model = res["attr_model"]
    params = res["params"][model]
    d = task_frame(a, task)
    D = designs_for(d, task)
    X, groups = D["oh" if model in ONEHOT_MODELS else "num"]
    Xd, Xt = X.loc[res["dev_index"]], X.loc[res["test_index"]]
    yd, yt = res["y_dev"], res["y_test"]
    est = res["estimators"][model]
    svd = attributions(est, Xd)
    svt = attributions(est, Xt)
    vars_ = sorted(set(groups.values()))
    imp_d = aggregate_attr(svd, groups).reindex(vars_).fillna(0)
    imp_t = aggregate_attr(svt, groups).reindex(vars_).fillna(0)
    # per-feature difference (deployed model on C3 minus on dev), row bootstrap
    gmap = {c: groups.get(c.replace("missingindicator_", ""), c) for c in svd.columns}
    Ad = svd.T.groupby(pd.Series(gmap)).sum().T.abs().reindex(columns=vars_).fillna(0)
    At = svt.T.groupby(pd.Series(gmap)).sum().T.abs().reindex(columns=vars_).fillna(0)
    rng = np.random.default_rng(SEED)
    diffs = np.array([At.values[rng.integers(0, len(At), len(At))].mean(0) - Ad.values[rng.integers(0, len(Ad), len(Ad))].mean(0)
                      for _ in range(200)])
    est_c3 = fit(model, params, Xt, yt, n_jobs=N_JOBS)
    imp_r = _imp(est_c3, Xt, groups).reindex(vars_).fillna(0)
    tab = pd.DataFrame({"variable": vars_, "label": [VAR_LABELS.get(v, v) for v in vars_],
                        "deployed_on_dev": imp_d.values, "deployed_on_C3": imp_t.values,
                        "diff_C3_minus_dev": imp_t.values - imp_d.values,
                        "diff_lo": np.percentile(diffs, 2.5, axis=0), "diff_hi": np.percentile(diffs, 97.5, axis=0),
                        "refit_on_C3": imp_r.values}).sort_values("deployed_on_dev", ascending=False)
    tab["task"], tab["model"] = task, model
    tab.to_csv(T(out, f"attribution_{task}.csv"), index=False)
    sh = pd.DataFrame(Parallel(n_jobs=N_JOBS)(delayed(_split)(r, model, params, Xd, yd, Xt, yt, groups, vars_)
                                              for r in range(R)))
    top = lambda s: set(s.sort_values(ascending=False).index[:10])  # noqa: E731
    summ = {"task": task, "model": model,
            "rho_deployed_dev_vs_C3": spearmanr(imp_d, imp_t).statistic,
            "wtau_deployed_dev_vs_C3": weightedtau(imp_d.values, imp_t.values).statistic,
            "rho_dev_vs_C3_refit": spearmanr(imp_d, imp_r).statistic,
            "wtau_dev_vs_C3_refit": weightedtau(imp_d.values, imp_r.values).statistic,
            "top10_overlap_refit": len(top(imp_d) & top(imp_r)), "split_R": R}
    for c in sh.columns:
        if c != "rep":
            summ[f"{c}_median"] = sh[c].median()
            summ[f"{c}_p2_5"] = sh[c].quantile(0.025)
            summ[f"{c}_p97_5"] = sh[c].quantile(0.975)
    pd.DataFrame([summ]).to_csv(T(out, f"attribution_stability_{task}.csv"), index=False)
    log(f"[attribution {task}] {model}: rho deployed {summ['rho_deployed_dev_vs_C3']:.3f}, refit {summ['rho_dev_vs_C3_refit']:.3f}")
    return tab, summ


# =============================================================================================
# Sensitivity analyses
# =============================================================================================

def ipcw_weights(a):
    X, _ = design_matrix(a, "P")
    from rtw_models import make_estimator
    w = pd.Series(np.nan, index=a.index)
    for c in (1, 2, 3):
        m = (a["cohort"] == c).values
        z = a.loc[m, "part2"].values
        p = make_estimator("LR").fit(X[m], z).predict_proba(X[m])[:, 1]
        sw = z.mean() / p
        w[m] = np.minimum(sw, np.percentile(sw, 99))
    return w


def sensitivity(res, a, out, B=B_BOOT, log=print):
    task = res["task"]
    prim = res["primary"]
    models = list(dict.fromkeys([prim, "RidgeOH"]))
    d = task_frame(a, task)
    D = designs_for(d, task)
    dev, te = d.loc[res["dev_index"]], d.loc[res["test_index"]]
    yd, yt = res["y_dev"], res["y_test"]
    st = te["strata"].values
    rows = [quick_eval(yt, {m: res["test"][m] for m in models}, task, B, "Primary analysis", st)]

    def refit(m, Xsrc, w=None):
        X = Xsrc
        return fit(m, res["params"][m], X.loc[dev.index], yd, n_jobs=N_JOBS, sample_weight=w).predict_proba(
            X.loc[te.index])[:, 1]
    Xof = lambda m, D_=D: D_["oh" if m in ONEHOT_MODELS else "num"][0]  # noqa: E731
    wdev = (dev["wt_w2"] / dev.groupby("cohort")["wt_w2"].transform("mean")).values
    p = {m: refit(m, Xof(m), wdev) for m in models}
    rows.append(quick_eval(yt, p, task, B, "Survey-weighted training, weighted evaluation", st, te["wt_w2"].values))
    ip = ipcw_weights(a)
    p = {m: refit(m, Xof(m), ip.reindex(dev.index).values) for m in models}
    rows.append(quick_eval(yt, p, task, B, "IPCW training and evaluation", st, ip.reindex(te.index).values))
    for lab, drop in (("Restricted predictors (S4)", RESTRICTED_DROP), ("Restricted predictors (S4b)", RESTRICTED_DROP_B),
                      ("Work ability removed", ["work_ability"])):
        keepv = [v for v in PREDICTORS if v not in drop]
        D2 = designs_for(d, task, keepv)
        p = {m: refit(m, Xof(m, D2)) for m in models}
        rows.append(quick_eval(yt, p, task, B, lab, st))
    m6 = (te["commuting"] != 1).values
    rows.append(quick_eval(yt[m6], {m: res["test"][m][m6] for m in models}, task, B,
                           "C3 excluding commuting injuries", st[m6]))
    t = pd.concat(rows, ignore_index=True)
    t.to_csv(T(out, f"sensitivity_{task}.csv"), index=False)
    log(f"[sensitivity {task}] done")
    return t


def seed_variability(res, a, out, n_seeds=20):
    task = res["task"]
    m = res["primary"]
    if m in ("LR", "RidgeOH"):
        return None
    d = task_frame(a, task)
    X = designs_for(d, task)["oh" if m in ONEHOT_MODELS else "num"][0]
    Xd, Xt = X.loc[res["dev_index"]], X.loc[res["test_index"]]
    yt = res["y_test"]
    rows = []
    for s in range(n_seeds):
        p = fit(m, res["params"][m], Xd, res["y_dev"], n_jobs=N_JOBS, seed=SEED + 500 + s).predict_proba(Xt)[:, 1]
        r = all_metrics(yt, p, task, which=("AUROC", "Brier", "CITL", "Slope"))
        r["seed"] = s
        rows.append(r)
    t = pd.DataFrame(rows)
    t["task"], t["model"] = task, m
    t.to_csv(T(out, f"seed_variability_{task}.csv"), index=False)
    return t


# =============================================================================================
# Extras: within-stratum performance of the pooled models; attrition
# =============================================================================================

def within_strata(res_p, a, out, B=B_BOOT):
    """Pooled-task models evaluated within wave-1 employment strata (employed / not employed)."""
    d = task_frame(a, "P")
    te = d.loc[res_p["test_index"]]
    yt = res_p["y_test"]
    rows = []
    for e, lab in ((1, "employed at wave 1"), (0, "not employed at wave 1")):
        m = (te["employed"].values == e)
        preds = {k: v[m] for k, v in _ordered(res_p).items() if k not in ("Null",)}
        for wl, w in (("unweighted", None), ("weighted", te["wt_w2"].values[m])):
            s, _ = evaluate(yt[m], preds, "P", B=B, seed=SEED, w=w, strata=te["strata"].values[m], refs=(),
                            which=("AUROC", "CITL", "Slope", "ICI"), label=wl)
            s["stratum"], s["n"], s["employed_w2"] = lab, int(m.sum()), int(yt[m].sum())
            rows.append(s)
    t = pd.concat(rows, ignore_index=True)
    t.to_csv(T(out, "perf_C3_P_within_strata.csv"), index=False)
    return t


def attrition(a, out):
    """Wave-1 to wave-2 response by cohort and wave-1 employment; missing wave-2 weights among respondents."""
    rows = []
    for c in (1, 2, 3):
        d = a[a["cohort"] == c]
        r = {"cohort": c, "wave1_n": len(d), "wave2_n": int(d["part2"].sum()),
             "response_pct": 100 * d["part2"].mean(),
             "response_pct_employed_w1": 100 * d.loc[d["employed"] == 1, "part2"].mean(),
             "response_pct_not_employed_w1": 100 * d.loc[d["employed"] == 0, "part2"].mean(),
             "missing_wt_w2_among_respondents": int(d.loc[d["part2"] == 1, "wt_w2"].isna().sum()),
             "missing_outcome_among_respondents": int(d.loc[d["part2"] == 1, "y"].isna().sum())}
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(T(out, "attrition.csv"), index=False)
    return t


def weights_design(a, out):
    """Round 2: survey-weight diagnostics per cohort and task (analysed wave-2 respondents): Kish effective sample size
    of the wave-2 longitudinal weights, max/min weight ratio, number of sampling strata and of strata with a single
    analysed unit (these make the within-stratum bootstrap mildly anti-conservative)."""
    rows = []
    for task in ("A1", "A2", "P"):
        d = task_frame(a, task)
        for c in (1, 2, 3):
            g = d[d["cohort"] == c]
            w = g["wt_w2"].values.astype(float)
            has_strata = g["strata"].notna().any()      # sampling strata are available for the evaluation cohort
            sizes = g.groupby("strata").size()
            rows.append({"task": task, "cohort": c, "n": len(g), "kish_ess": w.sum() ** 2 / np.sum(w ** 2),
                         "ess_pct": 100 * (w.sum() ** 2 / np.sum(w ** 2)) / len(g), "max_min_weight_ratio": w.max() / w.min(),
                         "n_strata": int(len(sizes)) if has_strata else np.nan,
                         "single_unit_strata": int((sizes == 1).sum()) if has_strata else np.nan})
    t = pd.DataFrame(rows)
    t.to_csv(T(out, "weights_design.csv"), index=False)
    return t


def disde_restricted(res, sec, a, out, log=print):
    """Round 2 robustness: prevalence DISDE decomposition (model-free) with the domain classifier restricted to the
    comparability-restricted predictor set S4b, for the pooled and cohort-specific designs."""
    import rtw_shift as SH
    from rtw_develop import designs_for
    variables = [v for v in PREDICTORS if v not in RESTRICTED_DROP_B]
    rows = []
    for task in ("A1", "A2", "P"):
        d = task_frame(a, task)
        Xn = designs_for(d, task, variables=variables)["num"][0]
        for lab, r in (("C1+C2 -> C3", res[task]), ("C1 -> C3", sec[("c1to3", task)]),
                       ("C2 -> C3", sec[("c2to3", task)])):
            Xs, Xt = Xn.loc[r["dev_index"]], Xn.loc[r["test_index"]]
            tab, _ = SH.disde(Xs, r["y_dev"], {"RidgeOH": r["oof"]["RidgeOH"]}, Xt, r["y_test"],
                              {"RidgeOH": r["test"]["RidgeOH"]})
            tab = tab[tab.metric == "Prevalence"].copy()
            tab["task"], tab["design"], tab["predictors"] = task, lab, "S4b (restricted)"
            rows.append(tab)
        log(f"[disde restricted {task}] done")
    t = pd.concat(rows, ignore_index=True)
    t.to_csv(T(out, "shift_disde_restricted.csv"), index=False)
    return t
