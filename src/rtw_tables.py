"""Format aggregated CSV outputs as Markdown tables (revision v2) and derive key numbers quoted in the text.

Display-layer consistency corrections 2026-10-15 (the CSV files and every value are unchanged): minus signs are U+2212
as in the text; model, design, decomposition-component and cohort codes are printed with the names used in the
manuscript; count columns get thousands separators; code-style column names are replaced by readable headers;
ranges and inequalities in labels use en dashes and typographic signs; "|" in cells is escaped instead of being
replaced by "/"; the C1 commuting-injury cell of Table S3 is "n/a" (a structural zero: C1 records only work accident
versus occupational disease); the inverse-probability-of-response-weighting sensitivity row is labelled IPRW.

Second display pass 2026-10-15 (consistency audit, fix round 2; the CSV files and every value are unchanged): readable
headers also for the remaining code-style columns of the tuning (Table S2), attrition (Table S3), sample-size (Table
S4), subgroup-audit (Table S11) and importance-weighting (Table S16) tables, and the domain-classifier column of the
DISDE information and importance-weighting tables (Tables S14 and S16) is headed "Domain classifier" with the values
"Logistic regression" and "LightGBM" instead of the codes LR and LGBM.
"""
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
DESIGN = {"primary": "C1+C2 \u2192 C3 (primary)", "c1to3": "C1 \u2192 C3", "c2to3": "C2 \u2192 C3",
          "chain12": "C1 \u2192 C2 (chronological)", "loco1": "C2+C3 \u2192 C1 (leave one cohort out)",
          "loco2": "C1+C3 \u2192 C2 (leave one cohort out)"}
COMP = {"X_shift_source_side": "X-shift (source)", "YgivenX_shift": "Y|X shift", "X_shift_target_side": "X-shift (target)",
        "total": "Total change"}
LABELS = {"Age >=60": "Age \u226560", "Disability grade 1-7": "Disability grade 1\u20137",
          "Work ability, mean (0-10)": "Work ability, mean (0\u201310)", "<=39": "\u226439", ">=60": "\u226560",
          "40-49": "40\u201349", "50-59": "50\u201359", "Mild (8-14)": "Mild (8\u201314)",
          "Severe (1-7)": "Severe (1\u20137)", "Work ability (0-10)": "Work ability (0\u201310)"}
COUNT_COLS = ["n", "n_dev", "n_test", "events", "events_min_class", "test_events_min", "wave1_n", "wave2_n",
              "employed_w2", "n_strata", "single_unit_strata", "n_candidates", "reps", "Slope_n_undefined", "n_required",
              "n_crit1_shrinkage0.9", "n_crit2_optimism", "n_crit3_intercept", "n_source", "n_target", "parameters"]
ONE_DECIMAL = ["tune_seconds", "missing_C1_%", "missing_C2_%", "missing_C3_%"]
COLS = {"kish_ess": "Kish ESS", "ess_pct": "ESS (%)", "max_min_weight_ratio": "Max/min weight ratio", "n_strata": "Strata",
        "single_unit_strata": "Single-unit strata", "R2cs_anticipated": "Anticipated R\u00b2CS",
        "max_R2cs": "Max R\u00b2CS", "OE": "O/E", "coef_y": "Outcome coefficient", "lrt": "LRT statistic",
        "max_abs_smd_before": "Max \\|SMD\\| before", "max_abs_smd_after": "Max \\|SMD\\| after",  # headers: escaped
        "cv_logloss": "CV log-loss (tuning folds)", "repeated_cv_logloss": "Repeated-CV log-loss",
        "tune_seconds": "Tuning time (s)", "n_candidates": "Candidates"}
COLS.update({  # second display pass (2026-10-15, fix round 2): remaining code-style headers
    # Table S2 (tuning)
    "logloss_repeat1_tuning_folds": "Log-loss, repeat 1 (tuning folds)",
    "logloss_repeat2_independent_folds": "Log-loss, repeat 2 (independent folds)",
    "selected_v1_single_cv": "Chosen by single tuning CV (v1)",
    "selected_v2_repeated_cv": "Chosen by repeated CV (v2)",
    "selected_independent_repeat": "Chosen by independent repeat",
    # Table S3 (attrition)
    "wave1_n": "Wave-1 n", "wave2_n": "Wave-2 respondents", "response_pct": "Response (%)",
    "response_pct_employed_w1": "Response, employed at wave 1 (%)",
    "response_pct_not_employed_w1": "Response, not employed at wave 1 (%)",
    "missing_wt_w2_among_respondents": "Respondents without wave-2 weight",
    "missing_outcome_among_respondents": "Respondents without outcome",
    # Table S4 (sample size; Riley et al. criteria)
    "n_dev": "Development n", "events_min_class": "Development events (minority class)", "parameters": "Candidate parameters",
    "n_crit1_shrinkage0.9": "Required n, criterion 1 (shrinkage 0.9)",
    "n_crit2_optimism": "Required n, criterion 2 (optimism)",
    "n_crit3_intercept": "Required n, criterion 3 (intercept)", "n_required": "Required n (largest)",
    "expected_shrinkage_at_n": "Expected shrinkage at development n", "lr_oof_slope": "LR (scores) OOF calibration slope",
    "n_test": "C3 n", "test_events_min": "C3 events (minority class)",
    "OE_95CI_lower_factor": "O/E 95% CI lower factor", "OE_95CI_upper_factor": "O/E 95% CI upper factor",
    # Table S11 (subgroup audit)
    "TPR_implied": "Calibration-implied TPR",
    "gap_TPR": "TPR gap", "gap_TPR_lo": "TPR gap lo", "gap_TPR_hi": "TPR gap hi", "gap_TPR_q": "TPR gap q (BH)",
    "gap_FPR": "FPR gap", "gap_FPR_lo": "FPR gap lo", "gap_FPR_hi": "FPR gap hi", "gap_FPR_q": "FPR gap q (BH)",
    "gap_AUROC": "AUROC gap", "gap_AUROC_q": "AUROC gap q (BH)",
    # Tables S14 and S16 (domain classifier, balance)
    "classifier": "Domain classifier",
    "n_smd_gt_0.1_before": "Predictors with \\|SMD\\| > 0.1 before",   # headers: escaped pipes
    "n_smd_gt_0.1_after": "Predictors with \\|SMD\\| > 0.1 after"})
DOMAIN_CLF = {"LR": "Logistic regression", "LGBM": "LightGBM"}   # domain-classifier codes (rtw_shift)
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
    if s.startswith("-") and float(s) == 0:
        return s[1:]
    return "\u2212" + s[1:] if s.startswith("-") else s   # U+2212 minus, as in the text


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
        out.append("| " + " | ".join(str(r[c]).replace("|", "\\|") for c in cols) + " |")
    return "\n".join(out)


def primary(out, task):
    t = T(out, "tuning_primary.csv")
    t = t[t.task == task]
    return t.loc[t.repeated_cv_logloss.idxmin(), "model"]


def table1(out):
    t = T(out, "table1_characteristics.csv")
    rows = []
    for _, r in t.iterrows():
        row = {"Characteristic": LABELS.get(r.characteristic, r.characteristic)}
        for c in (1, 2, 3):
            u, w = r[f"C{c}_unw"], r.get(f"C{c}_w", np.nan)
            if c == 1 and r.characteristic == "Commuting injury":
                row["C1"] = "n/a"   # structural zero: C1 records only work accident vs occupational disease
            elif r.kind == "n":
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
        base = ["Status"] if task == "P" else ["Null"]          # round 2: trimmed (full tables in S5-S7)
        models = list(dict.fromkeys(base + ["RidgeOH", prim]))
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
                         "CITL (population)": ci(*gw.loc["CITL", ["estimate", "lo", "hi"]], d=2),
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
            rows.append({"Task": task, "Design": des.replace(" -> ", " → "),
                         "Δ prevalence": ci(*pv.loc["total", ["estimate", "lo", "hi"]]),
                         "yx_prev": ci(*pv.loc["YgivenX_shift", ["estimate", "lo", "hi"]]),
                         "CITL in C3": f(cl.loc["total", "target_value"], 2),
                         "yx_citl": ci(*cl.loc["YgivenX_shift", ["estimate", "lo", "hi"]], d=2)})
    df = pd.DataFrame(rows)   # round 2: IW columns moved to Table S17 / Fig. 4a; prevalence rows are model-free
    df.columns = ["Task", "Design", "Δ employment prevalence (model-free)", "Y\\|X component (prev.)",
                  f"CITL in C3 ({SHORT[primary(out, 'A2')]})", "Y\\|X component (CITL)"]   # escaped pipe for Markdown
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


def _count(x):
    return "\u2013" if pd.isna(x) else f"{int(round(float(x))):,}"


def generic(out, name, cols=None, d=3):
    t = T(out, name)
    if cols:
        t = t[cols]
    t = t.copy()
    # display names (2026-10-15); the CSV values used by the manuscript placeholders are unchanged
    for c in ("model", "reference"):
        if c in t.columns:
            t[c] = t[c].map(lambda x: SHORT.get(x, x))
    if "classifier" in t.columns:   # domain classifier (not a prediction model): spelled out (2026-10-15)
        t["classifier"] = t["classifier"].map(lambda x: DOMAIN_CLF.get(x, x))
    if "design" in t.columns:
        t["design"] = t["design"].map(lambda x: DESIGN.get(x, x))
    for c in ("design", "analysis"):
        if c in t.columns:
            t[c] = t[c].astype(str).str.replace(" -> ", " \u2192 ", regex=False)
    if "component" in t.columns:
        t["component"] = t["component"].map(lambda x: COMP.get(x, x))
    if "cohort" in t.columns and pd.api.types.is_integer_dtype(t["cohort"]):
        t["cohort"] = t["cohort"].map(lambda x: f"C{x}")
    for c in ("level", "reference", "label", "characteristic"):
        if c in t.columns:
            t[c] = t[c].map(lambda x: LABELS.get(x, x))
    for c in COUNT_COLS:
        if c in t.columns and pd.api.types.is_numeric_dtype(t[c]):
            t[c] = t[c].map(_count)
    for c in ONE_DECIMAL:
        if c in t.columns and pd.api.types.is_numeric_dtype(t[c]):
            t[c] = t[c].map(lambda x: f(x, 1))
    for c in t.columns:
        if pd.api.types.is_float_dtype(t[c]):
            t[c] = t[c].map(lambda x: f(x, d))
    ren = {c: COLS[c] for c in t.columns if c in COLS}
    ren.update({c: "\u03c1 " + c[4:] for c in t.columns if c.startswith("rho_")})
    ren.update({c: "\u03c4w " + c[5:] for c in t.columns if c.startswith("wtau_")})
    return md(t.rename(columns=ren))


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
                                       for t in TASKS).replace("| IPCW training and evaluation |",
                                                               "| IPRW, training and evaluation |")
    sd = [n for n in os.listdir(os.path.join(out, "tables")) if n.startswith("seed_variability_")]
    parts["Seed variability"] = "\n\n".join(generic(out, n) for n in sorted(sd))
    parts["Secondary designs"] = generic(out, "secondary_designs.csv",
                                         ["task", "analysis", "model", "metric", "estimate", "lo", "hi", "n", "events"])
    parts["Predictors"] = generic(out, "tableS1_predictors.csv")
    parts["Within strata"] = generic(out, "perf_C3_P_within_strata.csv",
                                     ["stratum", "weighting", "model", "metric", "estimate", "lo", "hi", "n", "employed_w2"])
    parts["Attrition"] = generic(out, "attrition.csv", d=1)
    parts["Weights design"] = generic(out, "weights_design.csv", d=1)
    parts["DISDE restricted"] = generic(out, "shift_disde_restricted.csv",
                                        ["task", "design", "predictors", "model", "metric", "component", "estimate",
                                         "lo", "hi"], d=4)
    with open(os.path.join(out, "tables_formatted.md"), "w", encoding="utf-8") as fh:
        for k, v in parts.items():
            fh.write(f"## {k}\n\n{v}\n\n")
    return parts


if __name__ == "__main__":
    make_all(sys.argv[1])
