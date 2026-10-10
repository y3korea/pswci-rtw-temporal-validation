"""Format the aggregated CSV outputs into Markdown tables for the manuscript and supplement."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from rtw_config import VAR_LABELS

NAMES = {"Persistence": "Persistence model", "LR-5": "5-variable LR", "LR": "Logistic regression",
         "EN": "Elastic net", "RF": "Random forest", "XGB": "XGBoost", "LGBM": "LightGBM"}
NAMES_B = dict(NAMES, **{"Persistence": "Null model (dev. prevalence)", "LR-5": "4-variable LR"})
ORDER = ["Persistence", "LR-5", "LR", "EN", "RF", "XGB", "LGBM"]


def _f(x, d):
    s = f"{x:.{d}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def ci(e, lo, hi, d=3):
    if pd.isna(e):
        return "–"
    if pd.isna(lo):
        return _f(e, d)
    return f"{_f(e, d)} ({_f(lo, d)}, {_f(hi, d)})"


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def T(out, name):
    return pd.read_csv(os.path.join(out, "tables", name))


def table1(out):
    t = T(out, "table1_characteristics.csv")
    rows = []
    for _, r in t.iterrows():
        row = {"Characteristic": r["characteristic"]}
        for c in (1, 2, 3):
            u, w = r[f"C{c}_unw"], r.get(f"C{c}_w", np.nan)
            if r["kind"] == "pct":
                row[f"C{c}"] = f"{float(u):.1f} ({float(w):.1f})"
            elif r["kind"] == "mean":
                row[f"C{c}"] = f"{float(u):.2f} ± {float(r[f'C{c}_sd']):.2f} ({float(w):.2f})"
            elif r["kind"] == "acc":
                row[f"C{c}"] = f"{float(u):.3f}"
            elif "Re-entry" in r["characteristic"]:
                row[f"C{c}"] = f"{u} [{float(w):.1f}]"
            else:
                row[f"C{c}"] = str(u)
        rows.append(row)
    return md(pd.DataFrame(rows).rename(columns={"C1": "Cohort 1 (2013-14)", "C2": "Cohort 2 (2018-19)",
                                                 "C3": "Cohort 3 (2023-24)"}))


def perf_table(out, task, metrics=("AUROC", "AUPRC", "Brier", "sBrier", "CITL", "Slope", "ECE", "Accuracy")):
    t = T(out, f"perf_C3_task{task}.csv")
    dev = T(out, f"perf_devOOF_task{task}.csv").set_index("model")
    names = NAMES if task == "A" else NAMES_B
    rows = []
    for m in ORDER:
        d = t[t.model == m].set_index("metric")
        if len(d) == 0:
            continue
        r = {"Model": names[m], "Dev. AUROC (OOF)": f"{dev.loc[m, 'AUROC']:.3f}"}
        for k in metrics:
            dd = 3
            r[k] = ci(d.loc[k, "estimate"], d.loc[k, "lo"], d.loc[k, "hi"], dd) if not (
                task == "B" and m == "Persistence" and k in ("Slope", "AUROC", "AUPRC")) else "–"
        rows.append(r)
    return md(pd.DataFrame(rows))


def perf_compact(out):
    """Main-text Table 2: both tasks, key metrics."""
    rows = []
    for task in ("A", "B"):
        t = T(out, f"perf_C3_task{task}.csv")
        dev = T(out, f"perf_devOOF_task{task}.csv").set_index("model")
        names = NAMES if task == "A" else NAMES_B
        for m in ORDER:
            d = t[t.model == m].set_index("metric")
            null = task == "B" and m == "Persistence"
            rows.append({"Task": task, "Model": names[m],
                         "AUROC dev (OOF)": "–" if null else f"{dev.loc[m, 'AUROC']:.3f}",
                         "AUROC C3": "–" if null else ci(d.loc["AUROC", "estimate"], d.loc["AUROC", "lo"], d.loc["AUROC", "hi"]),
                         "AUPRC": "–" if null else f"{d.loc['AUPRC', 'estimate']:.3f}",
                         "Brier": f"{d.loc['Brier', 'estimate']:.3f}",
                         "CITL": ci(d.loc["CITL", "estimate"], d.loc["CITL", "lo"], d.loc["CITL", "hi"], 2),
                         "Slope": "–" if null else ci(d.loc["Slope", "estimate"], d.loc["Slope", "lo"], d.loc["Slope", "hi"], 2),
                         "ECE": f"{d.loc['ECE', 'estimate']:.3f}",
                         "Accuracy": f"{d.loc['Accuracy', 'estimate']:.3f}"})
    return md(pd.DataFrame(rows))


def table1_compact(out):
    """Main-text Table 1 (selected rows)."""
    t = T(out, "table1_characteristics.csv").set_index("characteristic")
    keep = ["Wave-1 participants, n", "Analysed (participated at wave 2), n (%)", "Female", "Age >=60",
            "College or higher education", "Occupational disease", "Commuting injury", "Disability grade 1-7 (severe)",
            "No disability grade", "Vocational rehabilitation service", "Regular job at injury", "Employed at wave 1",
            "Work ability, mean (0-10)", "High pain interference", "Physical limitation (walk/stairs/lift)",
            "Chronic disease", "Employed at wave 2 (Task A outcome)",
            "Re-entry among non-employed at wave 1 (Task B), events/n (%)",
            "Persistence accuracy (wave-2 status = wave-1 status)"]
    rows = []
    for k in keep:
        r = t.loc[k]
        row = {"Characteristic": k.replace(" (Task A outcome)", " (Task A)")}
        for c in (1, 2, 3):
            u, w = r[f"C{c}_unw"], r.get(f"C{c}_w", np.nan)
            if r["kind"] == "pct":
                row[f"C{c}"] = f"{float(u):.1f} ({float(w):.1f})"
            elif r["kind"] == "mean":
                row[f"C{c}"] = f"{float(u):.2f} ({float(w):.2f})"
            elif r["kind"] == "acc":
                row[f"C{c}"] = f"{float(u):.3f}"
            elif "Re-entry" in k:
                row[f"C{c}"] = f"{u} [{float(w):.1f}]"
            else:
                row[f"C{c}"] = str(u)
        rows.append(row)
    return md(pd.DataFrame(rows).rename(columns={"C1": "Cohort 1", "C2": "Cohort 2", "C3": "Cohort 3"}))


def shift_main(out):
    """Main-text Table 3: covariate-shift attribution for the primary model and LR (Task A and B)."""
    rows = []
    for task in ("A", "B"):
        t = T(out, f"shift_importance_weighting_task{task}.csv")
        for m in ("XGB", "LR"):
            for k in ("AUROC", "CITL"):
                r = t[(t.model == m) & (t.metric == k)].iloc[0]
                rows.append({"Task": task, "Model": NAMES[m], "Metric": k, "Dev. OOF": f"{r.dev_oof:.3f}",
                             "Dev. re-weighted": f"{r.dev_oof_weighted:.3f}", "C3": f"{r.C3:.3f}",
                             "Total change": ci(r.total_change, r.total_lo, r.total_hi),
                             "Covariate shift": ci(r.covariate_part, r.cov_lo, r.cov_hi),
                             "Residual": ci(r.residual_part, r.res_lo, r.res_hi)})
    return md(pd.DataFrame(rows))


def diffs_table(out, task):
    t = T(out, f"perf_C3_diffs_task{task}.csv")
    names = NAMES if task == "A" else NAMES_B
    rows = []
    for ref in ("Persistence", "LR-5", "LR"):
        for m in ORDER:
            d = t[(t.model == m) & (t.reference == ref)].set_index("metric")
            if len(d) == 0:
                continue
            rows.append({"Model": names[m], "Reference": names[ref],
                         "ΔAUROC": ci(d.loc["AUROC", "estimate"], d.loc["AUROC", "lo"], d.loc["AUROC", "hi"]),
                         "ΔBrier": ci(d.loc["Brier", "estimate"], d.loc["Brier", "lo"], d.loc["Brier", "hi"], 4),
                         "ΔAccuracy": ci(d.loc["Accuracy", "estimate"], d.loc["Accuracy", "lo"], d.loc["Accuracy", "hi"])})
    return md(pd.DataFrame(rows))


def shift_table(out):
    adv = T(out, "shift_adversarial.csv")
    tr = T(out, "shift_transitions.csv")
    rows = []
    for _, r in adv.iterrows():
        rows.append({"Quantity": f"Adversarial AUROC, {r.contrast}", "Value": ci(r.AUROC, r.lo, r.hi)})
    for c in (1, 2, 3):
        for e, lab in (("1", "employed"), ("0", "not employed")):
            r = tr[(tr.cohort == c) & (tr.employed_w1.astype(str) == e)].iloc[0]
            rows.append({"Quantity": f"C{c}: P(employed w2 | {lab} w1)",
                         "Value": f"{r.p:.3f} ({r.lo:.3f}, {r.hi:.3f}); {int(r.k)}/{int(r.n)}"})
    for task in ("A", "B"):
        i = T(out, f"shift_iw_info_task{task}.csv").iloc[0]
        rows.append({"Quantity": f"Task {task}: outcome prevalence dev / dev re-weighted to C3 covariates / C3",
                     "Value": f"{i.dev_prevalence:.3f} / {i.dev_prevalence_weighted:.3f} / {i.C3_prevalence:.3f}"})
        rows.append({"Quantity": f"Task {task}: domain-classifier AUROC (LR, cross-fitted); ESS of weights",
                     "Value": f"{i.domain_auroc_LR_crossfit:.3f}; {i.ESS:.0f} of {int(i.n_dev)}"})
    return md(pd.DataFrame(rows))


def iw_table(out, task, models=("Persistence", "LR-5", "LR", "XGB")):
    t = T(out, f"shift_importance_weighting_task{task}.csv")
    rows = []
    for m in models:
        for k in ("AUROC", "Brier", "CITL", "Slope"):
            r = t[(t.model == m) & (t.metric == k)]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            dd = 4 if k == "Brier" else 3
            rows.append({"Model": (NAMES if task == "A" else NAMES_B)[m], "Metric": k,
                         "Dev. OOF": f"{r.dev_oof:.{dd}f}", "Dev. re-weighted": f"{r.dev_oof_weighted:.{dd}f}",
                         "C3 observed": f"{r.C3:.{dd}f}",
                         "Total change": ci(r.total_change, r.total_lo, r.total_hi, dd),
                         "Covariate-shift part": ci(r.covariate_part, r.cov_lo, r.cov_hi, dd),
                         "Residual part": ci(r.residual_part, r.res_lo, r.res_hi, dd)})
    return md(pd.DataFrame(rows))


def feature_shift_table(out, n=None):
    t = T(out, "shift_features.csv")
    if n:
        t = t.head(n)
    t = t.assign(Variable=t.variable.map(VAR_LABELS))
    return md(pd.DataFrame({"Variable": t.Variable, "SMD dev vs C3": t.SMD_dev_vs_C3.round(3),
                            "PSI dev vs C3": t.PSI_dev_vs_C3.round(3), "SMD C1 vs C2": t.SMD_C1_vs_C2.round(3),
                            "PSI C1 vs C2": t.PSI_C1_vs_C2.round(3)}))


def updating_table(out, task):
    s = updating_robust(out, task)
    rows = []
    for _, d in s.iterrows():
        rows.append({"Method": d.method, "n": int(d.n),
                     **{k: ci(d[f"{k}_median"], d[f"{k}_p2_5"], d[f"{k}_p97_5"], 4 if k == "Brier" else 3)
                        for k in ("AUROC", "Brier", "CITL", "Slope")},
                     "Mean abs. CITL": f"{d.mean_abs_CITL:.3f}", "Median abs. (slope - 1)": f"{d.mean_abs_slope_minus1:.3f}",
                     "Reps with smaller abs. CITL than no update": f"{100 * d.share_better_abs_CITL:.0f}%",
                     "Reps with lower Brier than no update": f"{100 * d.share_better_Brier:.0f}%"})
    order = ["No update", "Intercept recalibration", "Logistic recalibration", "Refit on C3 sample",
             "Pooled refit (dev + C3 sample)"]
    df = pd.DataFrame(rows)
    df["o"] = df.Method.map({m: i for i, m in enumerate(order)})
    return md(df.sort_values(["o", "n"]).drop(columns="o"))


def fairness_table(out, task, pt, model=None):
    t = T(out, f"fairness_task{task}_pt{pt}.csv")
    if model:
        t = t[t.model == model]
    rows = []
    for _, r in t.iterrows():
        rows.append({"Model": r.model, "Group": r.group, "Level": r.level, "n": int(r.n),
                     "Observed": f"{r.prevalence:.3f}", "O/E": f"{r.OE:.3f}",
                     "AUROC": ci(r.AUROC, r.AUROC_lo, r.AUROC_hi), "CITL": ci(r.CITL, r.CITL_lo, r.CITL_hi),
                     "TPR": ci(r.TPR, r.TPR_lo, r.TPR_hi), "FPR": ci(r.FPR, r.FPR_lo, r.FPR_hi),
                     "ΔTPR vs ref": "ref" if r.level == r.reference else ci(r.gap_TPR, r.gap_TPR_lo, r.gap_TPR_hi),
                     "ΔFPR vs ref": "ref" if r.level == r.reference else ci(r.gap_FPR, r.gap_FPR_lo, r.gap_FPR_hi),
                     "Dev ΔTPR": "" if r.level == r.reference else f"{r.dev_gap_TPR:.3f}"})
    return md(pd.DataFrame(rows))


def sens_table(out, task):
    t = T(out, f"sensitivity_task{task}.csv")
    rows = []
    for (a, m), d in t.groupby(["analysis", "model"], sort=False):
        d = d.set_index("metric")
        rows.append({"Analysis": a, "Model": NAMES.get(m, m), "n": int(d["n"].iloc[0]),
                     **{k: ci(d.loc[k, "estimate"], d.loc[k, "lo"], d.loc[k, "hi"], 4 if k == "Brier" else 3)
                        for k in ("AUROC", "Brier", "CITL", "Slope")}})
    return md(pd.DataFrame(rows))


def secondary_table(out, task):
    t = T(out, "secondary_designs.csv")
    t = t[t.task == task]
    rows = []
    for (a, m), d in t.groupby(["analysis", "model"], sort=False):
        d = d.set_index("metric")
        rows.append({"Design": a, "Model": (NAMES if task == "A" else NAMES_B).get(m, m),
                     "n test (events)": f"{int(d['n'].iloc[0])} ({int(d['events'].iloc[0])})",
                     **{k: ci(d.loc[k, "estimate"], d.loc[k, "lo"], d.loc[k, "hi"], 4 if k == "Brier" else 3)
                        for k in ("AUROC", "Brier", "CITL", "Slope")},
                     "Lowest CV log-loss": d["primary_in_design"].iloc[0]})
    return md(pd.DataFrame(rows))


def tuning_table(out):
    t = T(out, "tuning_primary.csv")
    return md(t.assign(cv_logloss=t.cv_logloss.round(4)))


def shap_table(out, task):
    t = pd.read_csv(os.path.join(out, "tables", f"shap_importance_task{task}.csv"), index_col=0)
    cols = ["dev_model_on_dev", "dev_model_on_C3", "C3_refit_on_C3", "C1_only", "C2_only", "C3_only"]
    t = t[["label"] + cols].round(3)
    return md(t.reset_index(drop=True))


def dca_delta_table(out):
    rows = []
    for task in ("A", "B"):
        t = T(out, f"dca_delta_task{task}.csv")
        for _, r in t.iterrows():
            rows.append({"Task": task, "Model": r.model, "Reference": r.reference, "Threshold": r.threshold,
                         "ΔNet benefit": ci(r.delta_nb, r.lo, r.hi, 4)})
    return md(pd.DataFrame(rows))


CODING = {
    "female": "1 = female", "edu": "1 none ... 5 college or higher", "married": "1 = currently married",
    "hh_size": "persons (truncated at 6)", "disease": "1 = occupational disease (administrative)",
    "commuting": "1 = commuting injury (covered from 2018)", "care_dur": "1 <=3 m ... 6 >2 y",
    "tenure": "14 classes, 1 <1 m ... 14 >=20 y", "svc_any": "1 = any rehabilitation service (admin.)",
    "svc_job": "1 = vocational rehabilitation service (admin.)", "firm_size": "1 <5 ... 6 >=300 employees",
    "union": "1 = union at injured workplace", "ex_days": "days/week 0-7", "regular_ex": ">=3 days and >=30 min/day",
    "work_ability": "0 (lost) - 10 (full); C1 wave 1 refers to ~90 days after care",
    "work_ability_postcare": "0-10, recalled for end of care", "chronic": "1 = current chronic disease",
    "phys_limitation": "1 = substantial limitation walking/stairs/lifting", "sensory_impair": "1 = vision/hearing/speech impairment",
    "adl_help": "1 none ... 4 much help needed", "adl_mental": "1 none ... 5 very difficult",
    "adl_indoor": "1 none ... 5 very difficult", "adl_outdoor": "1 none ... 5 very difficult",
    "adl_work": "1 none ... 5 very difficult", "pain_freq": "1 never ... 5 always", "pain_interf": "1 not at all ... 4 always",
    "smoker": "1 = current smoker", "drinker": "1 = current drinker", "age_grp": "<=39, 40-49, 50-59, >=60",
    "job_status": "regular, temporary, daily, self-employed", "industry": "construction, manufacturing, transport, other",
    "disab_cat": "grade 1-3, 4-7, 8-9, 10-12, 13-14, none", "employed": "1 = employed at wave 1 (emp3 = 1)",
    "emp6": "new employer / self-employed or unpaid family (if employed); job-seeking (if not employed)",
}


def predictor_md(out):
    t = T(out, "tableS1_predictors.csv")
    t.insert(2, "coding", t.variable.map(CODING))
    return md(t)


def key_numbers(out):
    """Derived scalar quantities quoted in the text (written to tables/key_numbers.csv)."""
    rows = []
    t1 = T(out, "table1_characteristics.csv").set_index("characteristic")
    r = t1.loc["Analysed (participated at wave 2), n (%)"]
    for c in (1, 2, 3):
        rows.append(("response_pct_C%d" % c, float(str(r[f"C{c}_unw"]).split("(")[1].rstrip(")")),
                     "table1_characteristics.csv: Analysed row"))
    for task, ref in (("A", "Persistence rule"), ("B", "Treat all")):
        d = T(out, f"dca_task{task}.csv").pivot(index="threshold", columns="strategy", values="net_benefit")
        diff = d["XGB"] - d[ref]
        close = diff[diff.abs() < 0.01].index.values
        rows.append((f"dca{task}_xgb_minus_ref_min", float(diff.min()), f"dca_task{task}.csv"))
        rows.append((f"dca{task}_xgb_minus_ref_max", float(diff.max()), f"dca_task{task}.csv"))
        if len(close):
            rows.append((f"dca{task}_close_lo", float(close.min()), f"dca_task{task}.csv: |NB(XGB)-NB({ref})|<0.01"))
            rows.append((f"dca{task}_close_hi", float(close.max()), f"dca_task{task}.csv: |NB(XGB)-NB({ref})|<0.01"))
            rows.append((f"dca{task}_close_n", float(len(close)), f"dca_task{task}.csv"))
        above = diff[diff >= 0.01].index.values
        # smallest threshold from which XGB exceeds the reference by >= 0.01 at every higher threshold
        thr = sorted(d.index.values)
        first = None
        for i_, tv in enumerate(thr):
            if all(diff.loc[x] >= 0.01 for x in thr[i_:]):
                first = tv
                break
        if first is not None:
            rows.append((f"dca{task}_xgb_better_from", float(first), f"dca_task{task}.csv: NB(XGB)-NB({ref})>=0.01 thereafter"))
        mta = (d["XGB"] - d["Treat all"]).min()
        rows.append((f"dca{task}_xgb_minus_treatall_min", float(mta), f"dca_task{task}.csv"))
    k = pd.DataFrame(rows, columns=["name", "value", "source"])
    k.to_csv(os.path.join(out, "tables", "key_numbers.csv"), index=False)
    return k


def updating_robust(out, task):
    """Median/percentile summary and absolute calibration error per repetition (paired with no update)."""
    r = T(out, f"updating_reps_task{task}.csv")
    base = r[r.method == "No update"].set_index("rep")
    rows = []
    for (meth, n), d in r.groupby(["method", "n"], sort=False):
        d = d.set_index("rep")
        b = base.reindex(d.index)
        row = {"task": task, "method": meth, "n": int(n), "reps": len(d)}
        for k in ("AUROC", "Brier", "CITL", "Slope", "ECE"):
            row[f"{k}_median"] = d[k].median()
            row[f"{k}_p2_5"] = d[k].quantile(0.025)
            row[f"{k}_p97_5"] = d[k].quantile(0.975)
        row["mean_abs_CITL"] = d["CITL"].abs().mean()
        row["mean_abs_slope_minus1"] = (d["Slope"] - 1).abs().median()  # median: robust to degenerate refits
        row["share_better_abs_CITL"] = float(np.mean(d["CITL"].abs() < b["CITL"].abs()))
        row["share_better_abs_slope"] = float(np.mean((d["Slope"] - 1).abs() < (b["Slope"] - 1).abs()))
        row["share_better_Brier"] = float(np.mean(d["Brier"] < b["Brier"]))
        rows.append(row)
    u = pd.DataFrame(rows)
    u.to_csv(os.path.join(out, "tables", f"updating_robust_task{task}.csv"), index=False)
    return u


def make_all(out):
    for task in ("A", "B"):
        updating_robust(out, task)
    key_numbers(out)
    parts = {
        "Main Table 1": table1_compact(out), "Main Table 2": perf_compact(out), "Main Table 3": shift_main(out),
        "Predictors": predictor_md(out),
        "Table 1": table1(out), "Table 2A": perf_table(out, "A"), "Table 2B": perf_table(out, "B"),
        "Paired differences A": diffs_table(out, "A"), "Paired differences B": diffs_table(out, "B"),
        "Shift summary": shift_table(out), "Importance weighting A": iw_table(out, "A"),
        "Importance weighting B": iw_table(out, "B"), "Feature shift": feature_shift_table(out),
        "Updating A": updating_table(out, "A"), "Updating B": updating_table(out, "B"),
        "Fairness A pt30": fairness_table(out, "A", 30), "Fairness A pt50": fairness_table(out, "A", 50),
        "Fairness B pt60": fairness_table(out, "B", 60), "Sensitivity A": sens_table(out, "A"),
        "Sensitivity B": sens_table(out, "B"), "Secondary A": secondary_table(out, "A"),
        "Secondary B": secondary_table(out, "B"), "Tuning": tuning_table(out), "SHAP A": shap_table(out, "A"),
        "SHAP B": shap_table(out, "B"), "DCA delta": dca_delta_table(out),
    }
    with open(os.path.join(out, "tables_formatted.md"), "w", encoding="utf-8") as f:
        for k, v in parts.items():
            f.write(f"## {k}\n\n{v}\n\n")
    return parts


if __name__ == "__main__":
    make_all(sys.argv[1])
