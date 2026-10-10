"""All pre-specified analyses (see 05_results/analysis_plan.md). Writes aggregated CSV tables only."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from rtw_config import (MODEL_ORDER, N_JOBS, NOM_LEVELS, NUM_VARS, PREDICTORS, RESTRICTED_DROP, SEED,
                        VAR_LABELS)
from rtw_data import design_matrix, lr5_vars, task_frame
from rtw_metrics import (EPS, all_metrics, auroc, bootstrap, brier, cal_slope, citl, dca_diff_boot,
                         dca_table, decile_points, ece, expit, logistic_recal, logit, lowess_curve, wilson)
from rtw_models import aggregate_shap, fit, make_estimator, tree_shap


def _t(out, name):
    return os.path.join(out, "tables", name)


def _order(preds):
    return {m: preds[m] for m in MODEL_ORDER if m in preds}


def _suppress_small(df, ncol="n", k=5):
    """Blank out estimates from cells with n < k (privacy rule)."""
    if ncol in df:
        small = df[ncol] < k
        for c in df.columns:
            if c not in ("n", "group", "level", "variable", "cohort", "model", "task"):
                df.loc[small, c] = np.nan
    return df


# =============================================================================================
# 1. Descriptives
# =============================================================================================

def descriptives(a, out):
    rows = []
    s = a[a["part2"] == 1]
    def add(label, fn, kind="pct"):
        r = {"characteristic": label}
        for c in (1, 2, 3):
            d = s[s["cohort"] == c]
            w = d["wt_w2"].values
            v = fn(d).astype(float)
            ok = ~np.isnan(v)
            if kind == "pct":
                r[f"C{c}_unw"] = 100 * np.nanmean(v)
                r[f"C{c}_w"] = 100 * np.sum(w[ok] * v[ok]) / np.sum(w[ok])
            else:
                r[f"C{c}_unw"] = np.nanmean(v)
                r[f"C{c}_sd"] = np.nanstd(v, ddof=1)
                r[f"C{c}_w"] = np.sum(w[ok] * v[ok]) / np.sum(w[ok])
        r["kind"] = kind
        rows.append(r)
    n_rows = {"characteristic": "Wave-1 participants, n"}
    n2 = {"characteristic": "Analysed (participated at wave 2), n (%)"}
    for c in (1, 2, 3):
        n_rows[f"C{c}_unw"] = int((a["cohort"] == c).sum())
        n2[f"C{c}_unw"] = int(((a["cohort"] == c) & (a["part2"] == 1)).sum())
        n2[f"C{c}_unw"] = f"{n2[f'C{c}_unw']} ({100 * n2[f'C{c}_unw'] / n_rows[f'C{c}_unw']:.1f})"
    rows += [dict(n_rows, kind="n"), dict(n2, kind="n")]
    add("Female", lambda d: d["female"])
    for k, lab in [(1, "Age <=39"), (2, "Age 40-49"), (3, "Age 50-59"), (4, "Age >=60")]:
        add(lab, lambda d, k=k: (d["age_grp"] == k))
    add("College or higher education", lambda d: (d["edu"] == 5))
    add("Occupational disease", lambda d: d["disease"])
    add("Commuting injury", lambda d: d["commuting"])
    add("Disability grade 1-7 (severe)", lambda d: d["disab_cat"].isin([1, 2]))
    add("Disability grade 8-14 (mild)", lambda d: d["disab_cat"].isin([3, 4, 5]))
    add("No disability grade", lambda d: (d["disab_cat"] == 6))
    add("Vocational rehabilitation service", lambda d: d["svc_job"])
    add("Any rehabilitation service", lambda d: d["svc_any"])
    add("Regular job at injury", lambda d: (d["job_status"] == 1))
    add("Employed at wave 1", lambda d: d["employed"])
    add("Job-seeking unemployed at wave 1", lambda d: (d["emp6"] == 5))
    add("Work ability, mean (0-10)", lambda d: d["work_ability"], kind="mean")
    add("High pain interference", lambda d: (d["pain_interf"] >= 3).astype(float).where(d["pain_interf"].notna()))
    add("Physical limitation (walk/stairs/lift)", lambda d: d["phys_limitation"])
    add("Sensory impairment", lambda d: d["sensory_impair"])
    add("Chronic disease", lambda d: d["chronic"])
    add("Employed at wave 2 (Task A outcome)", lambda d: d["y"])
    re = {"characteristic": "Re-entry among non-employed at wave 1 (Task B), events/n (%)", "kind": "n"}
    for c in (1, 2, 3):
        d = s[(s["cohort"] == c) & (s["employed"] == 0)]
        re[f"C{c}_unw"] = f"{int(d['y'].sum())}/{len(d)} ({100 * d['y'].mean():.1f})"
        re[f"C{c}_w"] = 100 * np.sum(d["wt_w2"] * d["y"]) / np.sum(d["wt_w2"])
    rows.append(re)
    pers = {"characteristic": "Persistence accuracy (wave-2 status = wave-1 status)", "kind": "acc"}
    for c in (1, 2, 3):
        d = s[s["cohort"] == c]
        pers[f"C{c}_unw"] = float((d["employed"] == d["y"]).mean())
    rows.append(pers)
    t = pd.DataFrame(rows)
    t.to_csv(_t(out, "table1_characteristics.csv"), index=False)
    return t


def predictor_table(a, out):
    s = a[a["part2"] == 1]
    rows = []
    for v in PREDICTORS:
        enc = ("one-hot (ref = first level)" if v in NOM_LEVELS else
               "employed_w1 + nested emp6 indicators" if v in ("employed", "emp6") else "numeric score")
        r = {"variable": v, "label": VAR_LABELS.get(v, v), "encoding": enc,
             "restricted_set": "dropped" if v in RESTRICTED_DROP else "kept"}
        for c in (1, 2, 3):
            r[f"missing_C{c}_%"] = round(100 * s.loc[s["cohort"] == c, v].isna().mean(), 1)
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(_t(out, "tableS1_predictors.csv"), index=False)
    return t


# =============================================================================================
# 2. Temporal validation performance
# =============================================================================================

def evaluate(res, out, B, log=print):
    task = res["task"]
    yt, yd = res["y_test"], res["y_dev"]
    preds = _order(res["test"])
    summ, diffs = bootstrap(yt, preds, task, B=B, seed=SEED, diffs_vs=("Persistence", "LR-5", "LR"))
    summ["task"] = task
    diffs["task"] = task
    dev_rows = []
    for m, p in _order(res["oof"]).items():
        r = all_metrics(yd, p, task)
        r.update({"model": m, "task": task, "cv_logloss": res["cv_logloss"].get(m, np.nan)})
        dev_rows.append(r)
    dev = pd.DataFrame(dev_rows)
    summ.to_csv(_t(out, f"perf_C3_task{task}.csv"), index=False)
    diffs.to_csv(_t(out, f"perf_C3_diffs_task{task}.csv"), index=False)
    dev.to_csv(_t(out, f"perf_devOOF_task{task}.csv"), index=False)
    # persistence rule details (task A)
    if task == "A":
        rows = []
        for lab, y, flag in (("development", yd, res["persistence_flag_dev"]),
                             ("C3", yt, res["persistence_flag_test"])):
            pred_emp = 1 - flag
            rows.append({"set": lab, "n": len(y), "accuracy": np.mean(pred_emp == y),
                         "sensitivity_employment": np.mean(pred_emp[y == 1] == 1),
                         "specificity_employment": np.mean(pred_emp[y == 0] == 0)})
        pd.DataFrame(rows).to_csv(_t(out, "persistence_rule_taskA.csv"), index=False)
    # decision curves
    thr = np.round(np.arange(0.05, 0.801, 0.01), 2)
    pflag = res.get("persistence_flag_test")
    dca = dca_table(yt, preds, thr, persistence_flag=pflag)
    dca["task"] = task
    dca.to_csv(_t(out, f"dca_task{task}.csv"), index=False)
    prim = res["primary"]
    ref_flag = pflag if task == "A" else np.ones(len(yt), int)   # task B: vs treat-all
    dd = dca_diff_boot(yt, preds[prim], ref_flag, [0.2, 0.3, 0.4, 0.5], B=B, seed=SEED)
    dd["model"], dd["reference"], dd["task"] = prim, ("Persistence rule" if task == "A" else "Treat all"), task
    dd.to_csv(_t(out, f"dca_delta_task{task}.csv"), index=False)
    # calibration curves (aggregated)
    cal = []
    dec = []
    for m, p in preds.items():
        if m == "Persistence":
            continue
        g, f = lowess_curve(yt, p)
        cal.append(pd.DataFrame({"model": m, "pred": g, "obs_lowess": f}))
        dp = decile_points(yt, p)
        dp["model"] = m
        dec.append(dp)
    pd.concat(cal).assign(task=task).to_csv(_t(out, f"calcurve_task{task}.csv"), index=False)
    pd.concat(dec).assign(task=task).to_csv(_t(out, f"caldeciles_task{task}.csv"), index=False)
    # prediction histograms (aggregated counts)
    bins = np.linspace(0, 1, 41)
    hist = []
    for m, p in preds.items():
        h, _ = np.histogram(p, bins=bins)
        hist.append(pd.DataFrame({"model": m, "bin_lo": bins[:-1], "bin_hi": bins[1:], "count": h}))
    pd.concat(hist).assign(task=task).to_csv(_t(out, f"predhist_task{task}.csv"), index=False)
    # ROC curves (aggregated: thresholds on a fixed grid of FPR)
    from sklearn.metrics import roc_curve
    roc = []
    grid = np.linspace(0, 1, 201)
    for m, p in preds.items():
        fpr, tpr, _ = roc_curve(yt, p)
        roc.append(pd.DataFrame({"model": m, "fpr": grid, "tpr": np.interp(grid, fpr, tpr)}))
    pd.concat(roc).assign(task=task).to_csv(_t(out, f"roc_task{task}.csv"), index=False)
    log(f"[evaluate task {task}] primary {prim}: " + ", ".join(
        f"{k}={summ[(summ.model == prim) & (summ.metric == k)].estimate.iloc[0]:.3f}"
        for k in ("AUROC", "Brier", "CITL", "Slope")))
    return summ, diffs, dev


# =============================================================================================
# 3. Dataset shift
# =============================================================================================

def _psi(p_dev, p_new, eps=1e-4):
    p_dev = np.clip(p_dev, eps, None)
    p_new = np.clip(p_new, eps, None)
    return float(np.sum((p_new - p_dev) * np.log(p_new / p_dev)))


def feature_shift(a, out):
    d = task_frame(a, "A")
    dev, c3 = d[d["cohort"].isin([1, 2])], d[d["cohort"] == 3]
    c1, c2 = d[d["cohort"] == 1], d[d["cohort"] == 2]
    rows = []
    for v in PREDICTORS:
        def smd(x0, x1):
            x0, x1 = x0.dropna().astype(float), x1.dropna().astype(float)
            sd = np.sqrt((x0.var(ddof=1) + x1.var(ddof=1)) / 2)
            return (x1.mean() - x0.mean()) / sd if sd > 0 else 0.0
        if v in NOM_LEVELS or v == "emp6":
            levels = NOM_LEVELS.get(v, [1, 2, 3, 4, 5, 6])
            smds = [smd((dev[v] == L).astype(float), (c3[v] == L).astype(float)) for L in levels]
            s_val = smds[int(np.argmax(np.abs(smds)))]
            s12 = [smd((c1[v] == L).astype(float), (c2[v] == L).astype(float)) for L in levels]
            s12_val = s12[int(np.argmax(np.abs(s12)))]
        else:
            s_val = smd(dev[v], c3[v])
            s12_val = smd(c1[v], c2[v])
        cats = sorted(set(dev[v].dropna().unique()) | set(c3[v].dropna().unique()), key=str)
        def dist(x):
            vc = x.value_counts(normalize=False, dropna=False)
            arr = np.array([vc.get(cv, 0) for cv in cats] + [x.isna().sum()], float)
            return arr / arr.sum()
        psi = _psi(dist(dev[v]), dist(c3[v]))
        psi12 = _psi(dist(c1[v]), dist(c2[v]))
        rows.append({"variable": v, "label": VAR_LABELS.get(v, v), "SMD_dev_vs_C3": s_val,
                     "PSI_dev_vs_C3": psi, "SMD_C1_vs_C2": s12_val, "PSI_C1_vs_C2": psi12})
    t = pd.DataFrame(rows).sort_values("PSI_dev_vs_C3", ascending=False)
    t.to_csv(_t(out, "shift_features.csv"), index=False)
    return t


def _lgbm_domain():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=40,
                          random_state=SEED, n_jobs=N_JOBS, verbose=-1)


def adversarial(a, out, B):
    d = task_frame(a, "A")
    X, groups = design_matrix(d, "A")
    rows, drivers = [], []
    for lab, c0, c1 in (("Development (C1+C2) vs C3", (1, 2), (3,)), ("C1 vs C2", (1,), (2,))):
        m = d["cohort"].isin(c0 + c1)
        Xm = X[m]
        z = d.loc[m, "cohort"].isin(c1).astype(int).values
        oof = np.zeros(len(z))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xm, z):
            clf = _lgbm_domain().fit(Xm.iloc[tr], z[tr])
            oof[te] = clf.predict_proba(Xm.iloc[te])[:, 1]
        rng = np.random.default_rng(SEED)
        bs = [auroc(z[i], oof[i]) for i in (rng.integers(0, len(z), len(z)) for _ in range(B))]
        rows.append({"contrast": lab, "n": len(z), "AUROC": auroc(z, oof),
                     "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5)})
        clf = _lgbm_domain().fit(Xm, z)
        contrib = clf.predict(Xm, pred_contrib=True)[:, :-1]
        sv = pd.DataFrame(contrib, columns=Xm.columns)
        imp = aggregate_shap(sv, groups)
        drivers.append(pd.DataFrame({"contrast": lab, "variable": imp.index, "mean_abs_shap": imp.values}))
    t = pd.DataFrame(rows)
    t.to_csv(_t(out, "shift_adversarial.csv"), index=False)
    dr = pd.concat(drivers)
    dr["label"] = dr["variable"].map(VAR_LABELS)
    dr.to_csv(_t(out, "shift_adversarial_drivers.csv"), index=False)
    return t, dr


def transitions(a, out):
    d = task_frame(a, "A")
    rows = []
    for c in (1, 2, 3):
        for e in (1, 0):
            s = d[(d["cohort"] == c) & (d["employed"] == e)]
            k, n = int(s["y"].sum()), len(s)
            p, lo, hi = wilson(k, n)
            rows.append({"cohort": c, "employed_w1": e, "k": k, "n": n, "p": p, "lo": lo, "hi": hi})
        s = d[d["cohort"] == c]
        p, lo, hi = wilson(int(s["y"].sum()), len(s))
        rows.append({"cohort": c, "employed_w1": "all", "k": int(s["y"].sum()), "n": len(s), "p": p,
                     "lo": lo, "hi": hi})
    t = pd.DataFrame(rows)
    t.to_csv(_t(out, "shift_transitions.csv"), index=False)
    return t


def importance_weighting(res, a, out, B, log=print):
    """Covariate-shift attribution: development OOF performance re-weighted to C3 covariates."""
    task = res["task"]
    d = task_frame(a, task)
    X, _ = design_matrix(d, task)
    dev_idx, te_idx = res["dev_index"], res["test_index"]
    Xall = pd.concat([X.loc[dev_idx], X.loc[te_idx]])
    z = np.r_[np.zeros(len(dev_idx)), np.ones(len(te_idx))].astype(int)
    pz = np.zeros(len(z))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xall, z):
        est = make_estimator("LR").fit(Xall.iloc[tr], z[tr])
        pz[te] = est.predict_proba(Xall.iloc[te])[:, 1]
    p_dev = np.clip(pz[: len(dev_idx)], EPS, 1 - EPS)
    w = p_dev / (1 - p_dev) * (len(dev_idx) / len(te_idx))
    cap = np.percentile(w, 99)
    w = np.minimum(w, cap)
    w = w / w.mean()
    ess = w.sum() ** 2 / np.sum(w ** 2)
    yd, yt = res["y_dev"], res["y_test"]
    metrics = {"AUROC": auroc, "Brier": brier, "CITL": lambda y, p, w=None: citl(y, np.clip(p, EPS, 1 - EPS), w),
               "Slope": lambda y, p, w=None: cal_slope(y, np.clip(p, EPS, 1 - EPS), w)}
    rng = np.random.default_rng(SEED)
    boots = [(rng.integers(0, len(yd), len(yd)), rng.integers(0, len(yt), len(yt))) for _ in range(B)]
    rows = []
    for m in [mm for mm in MODEL_ORDER if mm in res["oof"]]:
        po, pt_ = res["oof"][m], res["test"][m]
        for k, f in metrics.items():
            i_ = f(yd, po)
            ii = f(yd, po, w)
            iii = f(yt, pt_)
            bs = []
            for bd, bt in boots:
                b_i, b_ii, b_iii = f(yd[bd], po[bd]), f(yd[bd], po[bd], w[bd]), f(yt[bt], pt_[bt])
                bs.append((b_iii - b_i, b_ii - b_i, b_iii - b_ii))
            bs = np.array(bs)
            rows.append({"task": task, "model": m, "metric": k, "dev_oof": i_, "dev_oof_weighted": ii,
                         "C3": iii, "total_change": iii - i_, "total_lo": np.percentile(bs[:, 0], 2.5),
                         "total_hi": np.percentile(bs[:, 0], 97.5), "covariate_part": ii - i_,
                         "cov_lo": np.percentile(bs[:, 1], 2.5), "cov_hi": np.percentile(bs[:, 1], 97.5),
                         "residual_part": iii - ii, "res_lo": np.percentile(bs[:, 2], 2.5),
                         "res_hi": np.percentile(bs[:, 2], 97.5)})
    t = pd.DataFrame(rows)
    t.to_csv(_t(out, f"shift_importance_weighting_task{task}.csv"), index=False)
    info = pd.DataFrame([{"task": task, "domain_auroc_LR_crossfit": auroc(z, pz), "weight_cap_99pct": cap,
                          "ESS": ess, "n_dev": len(yd), "dev_prevalence": yd.mean(),
                          "dev_prevalence_weighted": np.sum(w * yd) / np.sum(w), "C3_prevalence": yt.mean()}])
    info.to_csv(_t(out, f"shift_iw_info_task{task}.csv"), index=False)
    log(f"[shift task {task}] domain AUROC(LR)={auroc(z, pz):.3f} ESS={ess:.0f} "
        f"prev dev={yd.mean():.3f} weighted={np.sum(w * yd) / np.sum(w):.3f} C3={yt.mean():.3f}")
    return t, info


# =============================================================================================
# 4. Model updating (learning curves)
# =============================================================================================

def _update_rep(r, model, params, Xd, yd, Xt, yt, p_base, sizes):
    rng = np.random.default_rng(SEED + 1000 + r)
    pos, neg = np.where(yt == 1)[0], np.where(yt == 0)[0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    pool = np.r_[pos[: len(pos) // 2], neg[: len(neg) // 2]]
    hold = np.r_[pos[len(pos) // 2:], neg[len(neg) // 2:]]
    yh = yt[hold]
    rows = []

    def rec(n, method, p):
        p = np.clip(p, EPS, 1 - EPS)
        rows.append({"rep": r, "n": n, "method": method, "AUROC": auroc(yh, p), "Brier": brier(yh, p),
                     "CITL": citl(yh, p), "Slope": cal_slope(yh, p), "ECE": ece(yh, p)})

    rec(0, "No update", p_base[hold])
    for n in sizes:
        if n > len(pool):
            continue
        sub = rng.choice(pool, n, replace=False)
        lp_sub, lp_h = logit(p_base[sub]), logit(p_base[hold])
        a0 = logistic_recal(yt[sub], lp_sub, slope=False)[0]
        rec(n, "Intercept recalibration", expit(a0 + lp_h))
        ab = logistic_recal(yt[sub], lp_sub, slope=True)
        rec(n, "Logistic recalibration", expit(ab[0] + ab[1] * lp_h))
        try:
            est = fit(model, params, Xt.iloc[sub], yt[sub], n_jobs=1)
            rec(n, "Refit on C3 sample", est.predict_proba(Xt.iloc[hold])[:, 1])
        except Exception:  # noqa: BLE001 - e.g. non-convergence; recorded as missing
            pass
        Xp = pd.concat([Xd, Xt.iloc[sub]])
        yp = np.r_[yd, yt[sub]]
        est = fit(model, params, Xp, yp, n_jobs=1)
        rec(n, "Pooled refit (dev + C3 sample)", est.predict_proba(Xt.iloc[hold])[:, 1])
    return rows


def updating(res, a, out, R, log=print):
    task = res["task"]
    model = res["primary"]
    params = res["params"][model]
    d = task_frame(a, task)
    X, _ = design_matrix(d, task)
    Xd, Xt = X.loc[res["dev_index"]], X.loc[res["test_index"]]
    sizes = [100, 250, 500, 1000] if task == "A" else [100, 250, 500]
    reps = Parallel(n_jobs=N_JOBS)(delayed(_update_rep)(r, model, params, Xd, res["y_dev"], Xt,
                                                         res["y_test"], res["test"][model], sizes)
                                   for r in range(R))
    long = pd.DataFrame([row for rep in reps for row in rep])
    long["task"], long["model"] = task, model
    long.to_csv(_t(out, f"updating_reps_task{task}.csv"), index=False)
    summ = (long.melt(id_vars=["rep", "n", "method", "task", "model"], var_name="metric")
            .groupby(["task", "model", "method", "n", "metric"])["value"]
            .agg(mean="mean", p2_5=lambda x: np.nanpercentile(x, 2.5), p97_5=lambda x: np.nanpercentile(x, 97.5),
                 n_ok="count").reset_index())
    summ.to_csv(_t(out, f"updating_summary_task{task}.csv"), index=False)
    log(f"[updating task {task}] model {model}: {R} reps done")
    return summ


# =============================================================================================
# 5. Fairness
# =============================================================================================

FAIR_GROUPS = {"Sex": ("sex_grp", "Male"), "Age group": ("age_lab", "<=39"),
               "Disability": ("disab_grp", "Mild (8-14)"), "Accident type": ("acc_grp", "Injury"),
               "Job status": ("job_grp", "Regular")}


def _group_metrics(y, p, g, pt):
    out = {}
    risk_flag = (1 - p) >= pt          # flag for RTW support (risk of non-employment >= pt)
    event = y == 0                      # target event: not employed / not re-entered
    for lev in pd.unique(g):
        m = g == lev
        ym, pm, fm, em = y[m], p[m], risk_flag[m], event[m]
        out[lev] = {"n": int(m.sum()), "prevalence": ym.mean(), "mean_pred": pm.mean(),
                    "OE": ym.mean() / pm.mean(), "AUROC": auroc(ym, pm),
                    "CITL": citl(ym, np.clip(pm, EPS, 1 - EPS)) if 0 < ym.mean() < 1 else np.nan,
                    "TPR": fm[em].mean() if em.sum() else np.nan,
                    "FPR": fm[~em].mean() if (~em).sum() else np.nan}
    return out


def fairness(res, a, out, models, pt, B, tag=""):
    task = res["task"]
    d = task_frame(a, task)
    te = d.loc[res["test_index"]]
    dv = d.loc[res["dev_index"]]
    yt, yd = res["y_test"], res["y_dev"]
    rng = np.random.default_rng(SEED)
    boots = [rng.integers(0, len(yt), len(yt)) for _ in range(B)]
    rows = []
    for model in models:
        p, po = res["test"][model], res["oof"][model]
        for gname, (col, ref) in FAIR_GROUPS.items():
            g = te[col].values
            point = _group_metrics(yt, p, g, pt)
            devp = _group_metrics(yd, po, dv[col].values, pt)
            bs = [_group_metrics(yt[b], p[b], g[b], pt) for b in boots]
            for lev, mv in point.items():
                r = {"task": task, "model": model, "threshold": pt, "group": gname, "level": lev,
                     "reference": ref, **mv}
                for k in ("AUROC", "CITL", "TPR", "FPR", "OE"):
                    s = np.array([b[lev][k] for b in bs if lev in b], float)
                    r[f"{k}_lo"], r[f"{k}_hi"] = np.nanpercentile(s, 2.5), np.nanpercentile(s, 97.5)
                    r[f"dev_{k}"] = devp.get(lev, {}).get(k, np.nan)
                    if lev != ref:
                        gap = np.array([b[lev][k] - b[ref][k] for b in bs if lev in b and ref in b], float)
                        r[f"gap_{k}"] = mv[k] - point[ref][k]
                        r[f"gap_{k}_lo"], r[f"gap_{k}_hi"] = np.nanpercentile(gap, 2.5), np.nanpercentile(gap, 97.5)
                        r[f"dev_gap_{k}"] = devp.get(lev, {}).get(k, np.nan) - devp.get(ref, {}).get(k, np.nan)
                rows.append(r)
    t = _suppress_small(pd.DataFrame(rows))
    t.to_csv(_t(out, f"fairness_task{task}_pt{int(round(pt * 100))}{tag}.csv"), index=False)
    return t


# =============================================================================================
# 6. SHAP stability
# =============================================================================================

def _shap_imp(pipe, X, groups, max_rows):
    sv = tree_shap(pipe, X, max_rows=max_rows)
    return aggregate_shap(sv, groups)


def _boot_shap(b, model, params, Xt, yt, groups, max_rows, imp_dev, vars_):
    rng = np.random.default_rng(SEED + 5000 + b)
    idx = rng.integers(0, len(yt), len(yt))
    est = fit(model, params, Xt.iloc[idx], yt[idx], n_jobs=1)
    imp = _shap_imp(est, Xt.iloc[idx], groups, max_rows).reindex(vars_).fillna(0)
    rho = spearmanr(imp_dev.values, imp.values).statistic
    top_dev = set(imp_dev.sort_values(ascending=False).index[:10])
    top_b = set(imp.sort_values(ascending=False).index[:10])
    return rho, len(top_dev & top_b)


def _split_half(r, model, params, Xd, yd, Xt, yt, groups, max_rows, vars_):
    """Post hoc benchmark: importance-rank agreement within C3, within development, and between them,
    each from disjoint random halves of equal size (n = |C3|/2)."""
    rng = np.random.default_rng(SEED + 9000 + r)
    h = len(yt) // 2
    pt_ = rng.permutation(len(yt))
    pd_ = rng.permutation(len(yd))
    parts = {"C3a": (Xt, yt, pt_[:h]), "C3b": (Xt, yt, pt_[h:2 * h]),
             "Da": (Xd, yd, pd_[:h]), "Db": (Xd, yd, pd_[h:2 * h])}
    imp = {}
    for k, (X_, y_, idx) in parts.items():
        est = fit(model, params, X_.iloc[idx], y_[idx], n_jobs=1)
        imp[k] = _shap_imp(est, X_.iloc[idx], groups, max_rows).reindex(vars_).fillna(0).values
    rho = lambda u, v: spearmanr(imp[u], imp[v]).statistic  # noqa: E731
    return {"rep": r, "within_C3": rho("C3a", "C3b"), "within_dev": rho("Da", "Db"),
            "between_dev_C3": np.mean([rho("Da", "C3a"), rho("Db", "C3b")])}


def shap_stability(res, a, out, B, log=print, R_split=50):
    task = res["task"]
    model = res["shap_model"]
    params = res["params"][model]
    d = task_frame(a, task)
    X, groups = design_matrix(d, task)
    Xd, Xt = X.loc[res["dev_index"]], X.loc[res["test_index"]]
    yt = res["y_test"]
    max_rows = 1000 if model == "RF" else None
    vars_ = sorted(set(groups.values()))
    est_dev = res["estimators"][model]
    imp = {"dev_model_on_dev": _shap_imp(est_dev, Xd, groups, max_rows),
           "dev_model_on_C3": _shap_imp(est_dev, Xt, groups, max_rows)}
    est_c3 = fit(model, params, Xt, yt, n_jobs=N_JOBS)
    imp["C3_refit_on_C3"] = _shap_imp(est_c3, Xt, groups, max_rows)
    for c in (1, 2, 3):
        m = d["cohort"] == c
        est_c = fit(model, params, X[m], d.loc[m, "y"].values, n_jobs=N_JOBS)
        imp[f"C{c}_only"] = _shap_imp(est_c, X[m], groups, max_rows)
    T = pd.DataFrame(imp).reindex(vars_).fillna(0)
    T["label"] = T.index.map(VAR_LABELS)
    T = T.sort_values("dev_model_on_dev", ascending=False)
    T.to_csv(_t(out, f"shap_importance_task{task}.csv"))
    cols = [c for c in T.columns if c != "label"]
    rho = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for i in cols:
        for j in cols:
            rho.loc[i, j] = spearmanr(T[i], T[j]).statistic
    rho.to_csv(_t(out, f"shap_rho_matrix_task{task}.csv"))
    bs = Parallel(n_jobs=N_JOBS)(delayed(_boot_shap)(b, model, params, Xt, yt, groups, max_rows,
                                                     T["dev_model_on_dev"], T.index) for b in range(B))
    bs = np.array(bs, float)
    top_dev = set(T["dev_model_on_dev"].sort_values(ascending=False).index[:10])
    top_c3 = set(T["C3_refit_on_C3"].sort_values(ascending=False).index[:10])
    summ = pd.DataFrame([{"task": task, "model": model,
                          "rho_dev_vs_C3refit": rho.loc["dev_model_on_dev", "C3_refit_on_C3"],
                          "rho_boot_lo": np.percentile(bs[:, 0], 2.5), "rho_boot_hi": np.percentile(bs[:, 0], 97.5),
                          "top10_overlap": len(top_dev & top_c3),
                          "top10_overlap_boot_median": np.median(bs[:, 1]),
                          "rho_dev_on_dev_vs_dev_on_C3": rho.loc["dev_model_on_dev", "dev_model_on_C3"],
                          "rho_C1_vs_C2": rho.loc["C1_only", "C2_only"], "rho_C1_vs_C3": rho.loc["C1_only", "C3_only"],
                          "rho_C2_vs_C3": rho.loc["C2_only", "C3_only"], "B": B,
                          "shap_rows_subsample": max_rows}])
    sh = Parallel(n_jobs=N_JOBS)(delayed(_split_half)(r, model, params, Xd, res["y_dev"], Xt, yt, groups,
                                                      max_rows, T.index) for r in range(R_split))
    sh = pd.DataFrame(sh)
    sh.to_csv(_t(out, f"shap_splithalf_reps_task{task}.csv"), index=False)
    for k in ("within_C3", "within_dev", "between_dev_C3"):
        summ[f"splithalf_{k}_median"] = sh[k].median()
        summ[f"splithalf_{k}_lo"] = sh[k].quantile(0.025)
        summ[f"splithalf_{k}_hi"] = sh[k].quantile(0.975)
    summ["splithalf_R"] = R_split
    summ.to_csv(_t(out, f"shap_stability_task{task}.csv"), index=False)
    log(f"[shap task {task}] {model}: rho dev vs C3 refit = {summ.rho_dev_vs_C3refit.iloc[0]:.3f}")
    return T, summ


# =============================================================================================
# 7. Secondary designs and sensitivity analyses
# =============================================================================================

QUICK = ("AUROC", "Brier", "CITL", "Slope", "ECE")


def quick_eval(y, preds, task, B, w=None, label="", extra=None):
    summ, _ = bootstrap(y, _order(preds) if set(preds) <= set(MODEL_ORDER) else preds, task, B=B,
                        seed=SEED, w=w, diffs_vs=(), metrics=QUICK)
    summ["analysis"] = label
    summ["task"] = task
    summ["n"] = len(y)
    summ["events"] = int(np.sum(y))
    if extra:
        for k, v in extra.items():
            summ[k] = v
    return summ


def sensitivity(res, a, out, B, log=print):
    task = res["task"]
    prim = res["primary"]
    d = task_frame(a, task)
    X, groups = design_matrix(d, task)
    dev, te = d.loc[res["dev_index"]], d.loc[res["test_index"]]
    Xd, Xt = X.loc[dev.index], X.loc[te.index]
    yd, yt = res["y_dev"], res["y_test"]
    models = [prim] + (["LR"] if prim != "LR" else [])
    out_rows = []
    # S0: reference (unweighted, as primary)
    out_rows.append(quick_eval(yt, {m: res["test"][m] for m in models}, task, B, label="Primary analysis"))
    # S1: survey-weighted evaluation (all models)
    out_rows.append(quick_eval(yt, res["test"], task, B, w=te["wt_w2"].values,
                               label="S1 Survey-weighted evaluation"))
    # S2: weighted training + weighted evaluation
    wdev = dev["wt_w2"] / dev.groupby("cohort")["wt_w2"].transform("mean")
    p2 = {m: fit(m, res["params"][m], Xd, yd, n_jobs=N_JOBS, sample_weight=wdev.values).predict_proba(Xt)[:, 1]
          for m in models}
    out_rows.append(quick_eval(yt, p2, task, B, w=te["wt_w2"].values,
                               label="S2 Survey-weighted training and evaluation"))
    # S3: IPCW for wave-2 attrition
    ipcw = ipcw_weights(a)
    wd, wt = ipcw.reindex(dev.index).values, ipcw.reindex(te.index).values
    p3 = {m: fit(m, res["params"][m], Xd, yd, n_jobs=N_JOBS, sample_weight=wd).predict_proba(Xt)[:, 1]
          for m in models}
    out_rows.append(quick_eval(yt, p3, task, B, w=wt, label="S3 IPCW (attrition) training and evaluation"))
    # S4: comparability-restricted predictor set
    keep = [v for v in PREDICTORS if v not in RESTRICTED_DROP]
    X4, _ = design_matrix(d, task, keep)
    p4 = {m: fit(m, res["params"][m], X4.loc[dev.index], yd, n_jobs=N_JOBS).predict_proba(X4.loc[te.index])[:, 1]
          for m in models}
    out_rows.append(quick_eval(yt, p4, task, B, label="S4 Comparability-restricted predictors"))
    # S6: excluding commuting injuries from C3
    m6 = (te["commuting"] != 1).values
    out_rows.append(quick_eval(yt[m6], {m: res["test"][m][m6] for m in res["test"]}, task, B,
                               label="S6 C3 excluding commuting injuries"))
    # S7b: work ability removed
    keep7 = [v for v in PREDICTORS if v != "work_ability"]
    X7, g7 = design_matrix(d, task, keep7)
    p7 = {m: fit(m, res["params"][m], X7.loc[dev.index], yd, n_jobs=N_JOBS).predict_proba(X7.loc[te.index])[:, 1]
          for m in models}
    c4 = [c for c in X7.columns if g7[c] in lr5_vars(task) and g7[c] != "work_ability"]
    p7["LR-4 (no work ability)"] = fit("LR-5", None, X7.loc[dev.index, c4], yd).predict_proba(X7.loc[te.index, c4])[:, 1]
    out_rows.append(quick_eval(yt, p7, task, B, label="S7b Work ability removed"))
    t = pd.concat(out_rows, ignore_index=True)
    t.to_csv(_t(out, f"sensitivity_task{task}.csv"), index=False)
    log(f"[sensitivity task {task}] done")
    return t


def ipcw_weights(a):
    """Stabilised inverse probability of wave-2 participation (per cohort), truncated at 99th pct."""
    X, _ = design_matrix(a, "A")
    w = pd.Series(np.nan, index=a.index)
    for c in (1, 2, 3):
        m = (a["cohort"] == c).values
        z = a.loc[m, "part2"].values
        est = make_estimator("LR").fit(X[m], z)
        p = est.predict_proba(X[m])[:, 1]
        sw = z.mean() / p
        sw = np.minimum(sw, np.percentile(sw, 99))
        w[m] = sw
    return w


def secondary_designs(a, out, B, cache_dir, log=print):
    from rtw_develop import develop
    designs = [("C1 -> C2 (chain)", (1,), 2, True, "chain12"),
               ("C2+C3 -> C1 (LOCO)", (2, 3), 1, True, "loco1"),
               ("C1+C3 -> C2 (LOCO)", (1, 3), 2, True, "loco2"),
               ("C2 -> C3 (S7a: C2-only development)", (2,), 3, False, "c2only")]
    rows = []
    for task in ("A", "B"):
        for lab, trc, tec, reduced, tag in designs:
            r = develop(a, task, trc, tec, reduced=reduced, tag=tag, cache_dir=cache_dir, log=log)
            s = quick_eval(r["y_test"], r["test"], task, B, label=lab,
                           extra={"primary_in_design": r["primary"]})
            rows.append(s)
            ll = pd.DataFrame({"model": list(r["cv_logloss"]), "cv_logloss": list(r["cv_logloss"].values())})
            ll["design"], ll["task"] = lab, task
            ll.to_csv(_t(out, f"secondary_cvlogloss_{tag}_task{task}.csv"), index=False)
    t = pd.concat(rows, ignore_index=True)
    t.to_csv(_t(out, "secondary_designs.csv"), index=False)
    return t
