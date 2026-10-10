"""Format aggregated CSV outputs as Markdown tables (revision v2) and derive key numbers quoted in the text."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from rtw_config import VAR_LABELS

SHORT = {"Null": "Null", "Status": "Status-only LR", "LR-small": "Parsimonious LR", "LR": "LR (scores)",
         "RidgeOH": "Ridge-OH", "RF": "Random forest", "XGB": "XGBoost", "LGBM": "LightGBM", "CatBoost": "CatBoost",
         "EBM": "EBM"}
ORDER = ["Null", "Status", "LR-small", "LR", "RidgeOH", "RF", "XGB", "LGBM", "CatBoost", "EBM"]
TASKS = ("A1", "A2", "P")


def T(out, name):
    return pd.read_csv(os.path.join(out, "tables", name))


def f(x, d=3):
    if pd.isna(x):
        return "–"
    s = f"{x:.{d}f}"
    while float(s) == 0 and x != 0 and d < 6:
        d += 1
        s = f"{x:.{d}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def ci(e, lo, hi, d=3):
    if pd.isna(e):
        return "–"
    if pd.isna(lo):
        return f(e, d)
    return f"{f(e, d)} ({f(lo, d)}, {f(hi, d)})"


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]).replace("|", "/") for c in cols) + " |")
    return "\n".join(out)


def primary(out, task):
    t = T(out, "tuning_primary.csv")
    t = t[t.task == task]
    return t.loc[t.repeated_cv_logloss.idxmin(), "model"]


def table1(out):
    t = T(out, "table1_characteristics.csv")
    rows = []
    for _, r in t.iterrows():
        row = {"Characteristic": r.characteristic}
        for c in (1, 2, 3):
            u, w = r[f"C{c}_unw"], r.get(f"C{c}_w", np.nan)
            if r.kind == "n":
                row[f"C{c}"] = f"{int(u):,}"
            elif r.kind == "pct_only":
                row[f"C{c}"] = f"{u:.1f}"
            elif r.kind == "mean":
                row[f"C{c}"] = f"{u:.2f} ({w:.2f})"
            else:
                row[f"C{c}"] = f"{u:.1f} ({w:.1f})"
        rows.append(row)
    return md(pd.DataFrame(rows).rename(columns={"C1": "Cohort 1", "C2": "Cohort 2", "C3": "Cohort 3"}))


def table2(out, models_per_task=None):
    rows = []
    for task in TASKS:
        u = T(out, f"perf_C3_{task}_unweighted.csv")
        w = T(out, f"perf_C3_{task}_weighted.csv")
        prim = primary(out, task)
        base = ["Status", "LR-small"] if task == "P" else ["Null", "LR-small"]
        models = list(dict.fromkeys(base + ["LR", "RidgeOH", prim]))
        for m in models:
            gu = u[u.model == m].set_index("metric")
            gw = w[w.model == m].set_index("metric")
            if len(gu) == 0:
                continue
            nul = m == "Null"
            rows.append({"Task": task, "Model": SHORT[m] + (" *" if m == prim else ""),
                         "AUROC (sample)": "–" if nul else ci(*gu.loc["AUROC", ["estimate", "lo", "hi"]]),
                         "AUROC (population)": "–" if nul else ci(*gw.loc["AUROC", ["estimate", "lo", "hi"]]),
                         "CITL (sample)": ci(*gu.loc["CITL", ["estimate", "lo", "hi"]], d=2),
                         "Slope (sample)": "–" if nul else ci(*gu.loc["Slope", ["estimate", "lo", "hi"]], d=2),
                         "CITL (pop.)": ci(*gw.loc["CITL", ["estimate", "lo", "hi"]], d=2),
                         "Slope (pop.)": "–" if nul else ci(*gw.loc["Slope", ["estimate", "lo", "hi"]], d=2),
                         "ICI (sample)": "–" if nul else f(gu.loc["ICI", "estimate"])})
    return md(pd.DataFrame(rows))


def table3(out):
    dis = T(out, "shift_disde.csv")
    iw = T(out, "update_iw_recalibration.csv")
    rows = []
    for task in ("A1", "A2", "P"):
        prim = primary(out, task)
        for des in ("C1 -> C3", "C2 -> C3", "C1+C2 -> C3"):
            pv = dis[(dis.task == task) & (dis.design == des) & (dis.metric == "Prevalence")].set_index("component")
            m = prim if ((dis.task == task) & (dis.design == des) & (dis.model == prim)).any() else "XGB"
            cl = dis[(dis.task == task) & (dis.design == des) & (dis.metric == "CITL") & (dis.model == m)].set_index("component")
            ir = iw[(iw.task == task) & (iw.design == des) & (iw.model == m)]
            def g(meth, k):
                r = ir[(ir.method == meth) & (ir.metric == k)]
                return r.iloc[0].estimate if len(r) else np.nan
            if len(pv) == 0:
                continue
            rows.append({"Task": task, "Design": des.replace(" -> ", " → "), "Model": SHORT[m],
                         "Δ prevalence": ci(*pv.loc["total", ["estimate", "lo", "hi"]]),
                         "yx_prev": ci(*pv.loc["YgivenX_shift", ["estimate", "lo", "hi"]]),
                         "CITL in C3": f(cl.loc["total", "target_value"], 2),
                         "yx_citl": ci(*cl.loc["YgivenX_shift", ["estimate", "lo", "hi"]], d=2),
                         "Slope: none → IW recal.": f"{f(g('No update', 'Slope'), 2)} → {f(g('IW logistic recalibration', 'Slope'), 2)}",
                         "CITL: none → IW recal.": f"{f(g('No update', 'CITL'), 2)} → {f(g('IW logistic recalibration', 'CITL'), 2)}"})
    df = pd.DataFrame(rows)
    df.columns = ["Task", "Design", "Model", "Δ employment prevalence", "Y|X component (prev.)", "CITL in C3",
                  "Y|X component (CITL)", "Slope: none → IW", "CITL: none → IW"]
    return md(df)


def perf_full(out, task, lab):
    u = T(out, f"perf_C3_{task}_{lab}.csv")
    dev = T(out, f"perf_devOOF_{task}.csv").set_index("model")
    rows = []
    for m in ORDER:
        g = u[u.model == m].set_index("metric")
        if len(g) == 0:
            continue
        rows.append({"Model": SHORT[m], "Dev AUROC (OOF)": f(dev.loc[m, "AUROC"]) if m in dev.index else "–",
                     **{k: ci(*g.loc[k, ["estimate", "lo", "hi"]], d=4 if k in ("Brier", "LogLoss") else 3)
                        for k in ("AUROC", "AUPRC", "Brier", "LogLoss", "CITL", "Slope", "ICI", "E50", "E90", "ECE")},
                     "No-skill AUPRC": f(g.loc["NoSkillAUPRC", "estimate"]),
                     "Slope undefined (boot)": int(g.loc["Slope", "n_undefined"])})
    return md(pd.DataFrame(rows))


def diffs(out, task, lab):
    d = T(out, f"perf_C3_diffs_{task}_{lab}.csv")
    d = d[d.metric.isin(["AUROC", "Brier", "ICI"])]
    rows = [{"Model": SHORT[r.model], "Reference": SHORT[r.reference], "Metric": r.metric,
             "Difference": ci(r.estimate, r.lo, r.hi, 4)} for _, r in d.iterrows()]
    return md(pd.DataFrame(rows))


def generic(out, name, cols=None, d=3):
    t = T(out, name)
    if cols:
        t = t[cols]
    for c in t.columns:
        if pd.api.types.is_float_dtype(t[c]):
            t[c] = t[c].map(lambda x: f(x, d))
    return md(t)


def key_numbers(out):
    rows = []
    t1 = T(out, "table1_characteristics.csv").set_index("characteristic")
    for c in (1, 2, 3):
        rows.append((f"response_pct_C{c}", t1.loc["Wave-2 response, %", f"C{c}_unw"], "table1_characteristics.csv"))
    for task in TASKS:
        prim = primary(out, task)
        rows.append((f"primary_is_{task}_{prim}", 1, "tuning_primary.csv"))
        for lab in ("unweighted", "weighted"):
            dd = T(out, f"dca_delta_{task}_{lab}.csv")
            for _, r in dd.iterrows():
                rows.append((f"dca_{task}_{lab}_{r.model}_pt{r.threshold:.2f}", r.delta_vs_best_default, f"dca_delta_{task}_{lab}.csv"))
    k = pd.DataFrame(rows, columns=["name", "value", "source"])
    k.to_csv(os.path.join(out, "tables", "key_numbers.csv"), index=False)
    return k


def make_all(out):
    key_numbers(out)
    parts = {"Main Table 1": table1(out), "Main Table 2": table2(out), "Main Table 3": table3(out)}
    for task in TASKS:
        for lab in ("unweighted", "weighted"):
            parts[f"Performance {task} {lab}"] = perf_full(out, task, lab)
            parts[f"Differences {task} {lab}"] = diffs(out, task, lab)
    parts["Tuning primary"] = generic(out, "tuning_primary.csv", d=4)
    parts["Tuning secondary"] = generic(out, "tuning_secondary.csv", d=4)
    parts["Sample size"] = generic(out, "sample_size_riley.csv")
    parts["Calibration change"] = "\n\n".join(generic(out, f"calibration_change_{t}.csv") for t in TASKS)
    parts["DCA delta"] = "\n\n".join(generic(out, f"dca_delta_{t}_{lab}.csv", d=4) for t in TASKS
                                     for lab in ("unweighted", "weighted"))
    parts["Prevalence"] = generic(out, "shift_prevalence.csv")
    parts["Cohort effects"] = generic(out, "shift_cohort_effects.csv")
    parts["Outcome in domain"] = generic(out, "shift_outcome_in_domain_test.csv")
    parts["Adversarial"] = generic(out, "shift_adversarial.csv")
    parts["DISDE"] = generic(out, "shift_disde.csv", ["task", "design", "model", "metric", "component", "estimate", "lo", "hi"], d=4)
    parts["DISDE info"] = generic(out, "shift_disde_info.csv")
    parts["Case mix"] = generic(out, "shift_casemix.csv")
    parts["IW sensitivity"] = generic(out, "shift_iw_sensitivity.csv")
    parts["IW recalibration"] = generic(out, "update_iw_recalibration.csv",
                                        ["task", "design", "model", "method", "metric", "estimate", "lo", "hi",
                                         "improvement_vs_none", "imp_lo", "imp_hi"], d=4)
    lg = T(out, "update_lagged_summary.csv")
    parts["Lagged updating"] = generic(out, "update_lagged_summary.csv",
                                       ["task", "model", "method", "n", "reps", "AUROC_median", "Brier_median", "CITL_median",
                                        "Slope_median", "ICI_median", "ICI_p2_5", "ICI_p97_5", "share_better_ICI",
                                        "share_better_Brier", "Slope_n_undefined"] +
                                       [c for c in lg.columns if c.startswith("closed_choice")], d=4)
    wc = T(out, "update_withinC3_summary.csv")
    parts["Within-C3 updating"] = generic(out, "update_withinC3_summary.csv",
                                          ["task", "model", "method", "n", "reps", "AUROC_median", "Brier_median",
                                           "CITL_median", "Slope_median", "ICI_median", "mean_absCITL",
                                           "share_better_absCITL", "share_better_slope", "share_better_Brier",
                                           "Slope_n_undefined"] + [c for c in wc.columns if c.startswith("closed_choice")], d=4)
    fa = T(out, "fairness.csv")
    keep = ["task", "model", "weighting", "threshold", "group", "level", "n", "observed", "OE", "AUROC", "CITL", "TPR",
            "TPR_implied", "FPR", "PPV", "NPV", "gap_TPR", "gap_TPR_lo", "gap_TPR_hi", "gap_TPR_q", "gap_FPR",
            "gap_FPR_lo", "gap_FPR_hi", "gap_FPR_q", "gap_AUROC", "gap_AUROC_q"]
    parts["Fairness"] = generic(out, "fairness.csv", [c for c in keep if c in fa.columns])
    parts["Attribution"] = "\n\n".join(generic(out, f"attribution_{t}.csv") for t in TASKS)
    parts["Attribution stability"] = "\n\n".join(generic(out, f"attribution_stability_{t}.csv") for t in TASKS)
    parts["Sensitivity"] = "\n\n".join(generic(out, f"sensitivity_{t}.csv", ["analysis", "model", "metric", "estimate", "lo", "hi", "n"])
                                       for t in TASKS)
    sd = [n for n in os.listdir(os.path.join(out, "tables")) if n.startswith("seed_variability_")]
    parts["Seed variability"] = "\n\n".join(generic(out, n) for n in sorted(sd))
    parts["Secondary designs"] = generic(out, "secondary_designs.csv",
                                         ["task", "analysis", "model", "metric", "estimate", "lo", "hi", "n", "events"])
    parts["Predictors"] = generic(out, "tableS1_predictors.csv")
    parts["Within strata"] = generic(out, "perf_C3_P_within_strata.csv",
                                     ["stratum", "weighting", "model", "metric", "estimate", "lo", "hi", "n", "employed_w2"])
    parts["Attrition"] = generic(out, "attrition.csv", d=1)
    with open(os.path.join(out, "tables_formatted.md"), "w", encoding="utf-8") as fh:
        for k, v in parts.items():
            fh.write(f"## {k}\n\n{v}\n\n")
    return parts


if __name__ == "__main__":
    make_all(sys.argv[1])
