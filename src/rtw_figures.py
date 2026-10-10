"""Publication figures, drawn only from the aggregated CSV tables (no row-level data needed).

Identity is carried by line style + marker + legend (readable in grayscale); colour is secondary.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rtw_config import VAR_LABELS  # noqa: E402

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300, "figure.dpi": 100,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.4,
})

STY = {  # model -> (colour, linestyle, marker)
    "Persistence": ("#7a7a7a", (0, (1, 1.2)), None),
    "LR-5": ("#111111", (0, (7, 2.5)), "x"),
    "LR": ("#2a78d6", "-", "o"),
    "EN": ("#eb6834", (0, (4, 1.5)), "s"),
    "RF": ("#1baf7a", (0, (1.5, 1)), "^"),
    "XGB": ("#eda100", (0, (5, 1.5, 1, 1.5)), "D"),
    "LGBM": ("#e87ba4", (0, (6, 1.5, 1, 1.5, 1, 1.5)), "v"),
    "Treat all": ("#555555", (0, (2, 2)), None),
    "Treat none": ("#999999", "-", None),
    "Persistence rule": ("#7a7a7a", (0, (1, 1.2)), "o"),
}
LAB = {"Persistence": "Persistence", "LR-5": "5-variable LR", "LR": "LR", "EN": "Elastic net", "RF": "Random forest",
       "XGB": "XGBoost", "LGBM": "LightGBM", "Treat all": "Flag all", "Treat none": "Flag none",
       "Persistence rule": "Persistence rule"}
METHOD_STY = {
    "No update": ("#111111", (0, (1, 1.2)), None),
    "Intercept recalibration": ("#2a78d6", "-", "o"),
    "Logistic recalibration": ("#eb6834", (0, (4, 1.5)), "s"),
    "Refit on C3 sample": ("#1baf7a", (0, (1.5, 1)), "^"),
    "Pooled refit (dev + C3 sample)": ("#7a4fb0", (0, (5, 1.5, 1, 1.5)), "D"),
}


def _save(fig, out, name):
    d = os.path.join(out, "figures")
    fig.savefig(os.path.join(d, name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(d, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def _tab(out, name):
    return pd.read_csv(os.path.join(out, "tables", name))


def _line(ax, x, y, key, sty=STY, lab=LAB, lw=1.1, markevery=None, **kw):
    c, ls, mk = sty[key]
    ax.plot(x, y, color=c, ls=ls, marker=mk, ms=3.2, lw=lw, markevery=markevery, label=lab.get(key, key),
            mfc="white" if mk not in (None, "x") else None, mew=0.8, **kw)


def _primary(out, task):
    t = _tab(out, "tuning_primary.csv")
    t = t[t.task == task]
    return t.loc[t.cv_logloss.idxmin(), "model"]


# ---------------------------------------------------------------------------------------------
def fig1_design(out):
    t1 = _tab(out, "table1_characteristics.csv").set_index("characteristic")
    n_all = t1.loc["Wave-1 participants, n"]
    n_an = t1.loc["Analysed (participated at wave 2), n (%)"]
    fig, ax = plt.subplots(figsize=(6.9, 2.3))
    ax.grid(False)
    ax.axvspan(2019.95, 2022.95, color="#eeeeee", zorder=0)
    ax.text(2021.45, 3.62, "COVID-19 period\n(2020-2022)", ha="center", va="bottom", fontsize=6.3, color="#444")
    ax.axvline(2018, color="#888", lw=0.6, ls=(0, (2, 2)))
    ax.text(2018.08, 0.42, "Commuting injuries\ncovered from 2018", fontsize=6.0, color="#444", va="bottom")
    rows = [(1, 2012, 3.0, "Cohort 1", "Development"), (2, 2017, 2.0, "Cohort 2", "Development"),
            (3, 2022, 1.0, "Cohort 3", "Temporal validation")]
    for c, care, yv, lab, role in rows:
        fc = "#d9d9d9" if role == "Development" else "#ffffff"
        ax.add_patch(plt.Rectangle((care - 0.45, yv - 0.32), 2.9, 0.64, fc=fc, ec="#333", lw=0.7, zorder=1,
                                   hatch=None if role == "Development" else "////"))
        ax.plot([care], [yv], marker="|", ms=9, color="#333", zorder=3)
        ax.annotate("", xy=(care + 2, yv), xytext=(care + 1, yv),
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#111"), zorder=3)
        ax.plot([care + 1], [yv], "o", ms=5, mfc="#111", mec="#111", zorder=4)
        ax.plot([care + 2], [yv], "s", ms=5, mfc="white", mec="#111", zorder=4)
        ax.text(care - 0.55, yv, f"{lab}\n{role}", ha="right", va="center", fontsize=6.6)
        ax.text(care + 2.6, yv, f"wave 1 n = {int(float(n_all[f'C{c}_unw'])):,}\n"
                f"analysed n = {int(str(n_an[f'C{c}_unw']).split()[0]):,}", ha="left", va="center", fontsize=6.2)
    ax.set_xlim(2008.6, 2026.4)
    ax.set_ylim(0.3, 4.2)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks(range(2012, 2026, 1))
    ax.tick_params(axis="x", labelsize=6)
    h = [plt.Line2D([], [], marker="|", ls="", color="#333", ms=8, label="End of compensated care"),
         plt.Line2D([], [], marker="o", ls="", mfc="#111", mec="#111", ms=5, label="Wave 1: predictors"),
         plt.Line2D([], [], marker="s", ls="", mfc="white", mec="#111", ms=5, label="Wave 2: outcome (employed)")]
    ax.legend(handles=h, loc="upper left", ncol=3, frameon=False, bbox_to_anchor=(0.0, 1.13))
    _save(fig, out, "fig1_study_design")


def fig2_roc_cal(out):
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 6.2))
    for r, task in enumerate(("A", "B")):
        roc = _tab(out, f"roc_task{task}.csv")
        perf = _tab(out, f"perf_C3_task{task}.csv")
        auc = perf[perf.metric == "AUROC"].set_index("model")["estimate"]
        ax = axes[r, 0]
        for m in ["Persistence", "LR-5", "LR", "EN", "RF", "XGB", "LGBM"]:
            d = roc[roc.model == m]
            if len(d) == 0:
                continue
            c, ls, mk = STY[m]
            name = "Null (prevalence)" if (task == "B" and m == "Persistence") else LAB[m]
            if task == "B" and m == "LR-5":
                name = "4-variable LR"
            ax.plot(d.fpr, d.tpr, color=c, ls=ls, marker=mk, ms=3, markevery=20, lw=1.0,
                    mfc="white" if mk not in (None, "x") else None, mew=0.7, label=f"{name} ({auc[m]:.3f})")
        ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=0.6)
        ax.set_xlabel("1 - specificity")
        ax.set_ylabel("Sensitivity")
        ax.set_title(f"({'a' if task == 'A' else 'c'}) ROC, Task {task}: "
                     f"{'employment' if task == 'A' else 're-entry'} (cohort 3)", loc="left")
        ax.legend(loc="lower right", frameon=False, title="AUROC", title_fontsize=6.5)
        ax.set_aspect("equal")
        cal = _tab(out, f"calcurve_task{task}.csv")
        dec = _tab(out, f"caldeciles_task{task}.csv")
        ax = axes[r, 1]
        prim = _primary(out, task)
        for m in ["LR-5", "LR", "EN", "RF", "XGB", "LGBM"]:
            d = cal[cal.model == m]
            name = "4-variable LR" if (task == "B" and m == "LR-5") else LAB[m]
            c, ls, mk = STY[m]
            ax.plot(d.pred, d.obs_lowess, color=c, ls=ls, lw=1.0, label=name)
        dp = dec[dec.model == prim]
        ax.plot(dp.pred, dp.obs, ls="", marker=STY[prim][2], ms=3.5, mfc="white", mec="#111", mew=0.8,
                label=f"{LAB[prim]} deciles")
        ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=0.6)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted probability of employment at wave 2")
        ax.set_ylabel("Observed proportion")
        ax.set_title(f"({'b' if task == 'A' else 'd'}) Calibration, Task {task} (cohort 3)", loc="left")
        ax.legend(loc="upper left", frameon=False)
        ax.set_aspect("equal")
    fig.tight_layout()
    _save(fig, out, "fig2_roc_calibration")


def fig3_dca(out):
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9))
    for i, task in enumerate(("A", "B")):
        d = _tab(out, f"dca_task{task}.csv")
        prim = _primary(out, task)
        ax = axes[i]
        strategies = ["Treat all", "Treat none"] + (["Persistence rule"] if task == "A" else []) + ["LR-5", "LR"]
        if prim not in strategies:
            strategies.append(prim)
        ymax = 0
        for s in strategies:
            dd = d[d.strategy == s]
            name = LAB.get(s, s)
            if task == "B" and s == "LR-5":
                name = "4-variable LR"
            if task == "B" and s == "Treat all":
                name = "Flag all (= persistence)"
            c, ls, mk = STY[s]
            ax.plot(dd.threshold, dd.net_benefit, color=c, ls=ls, marker=mk, ms=2.8, markevery=6, lw=1.0,
                    mfc="white" if mk not in (None, "x") else None, mew=0.7, label=name)
            ymax = max(ymax, dd.net_benefit.max())
        ax.set_ylim(-0.03, ymax * 1.1)
        ax.set_xlim(0.05, 0.8)
        ax.set_xlabel("Threshold probability of non-employment" if task == "A"
                      else "Threshold probability of no re-entry")
        ax.set_ylabel("Net benefit")
        ax.set_title(f"({'a' if task == 'A' else 'b'}) Task {task}: flagging for RTW support (cohort 3)", loc="left")
        ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    _save(fig, out, "fig3_decision_curves")


def fig4_shift(out):
    fs = _tab(out, "shift_features.csv")
    adv = _tab(out, "shift_adversarial.csv").set_index("contrast")
    drv = _tab(out, "shift_adversarial_drivers.csv")
    tr = _tab(out, "shift_transitions.csv")
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.6))
    ax = axes[0, 0]
    top = fs.sort_values("PSI_dev_vs_C3", ascending=False).head(12).iloc[::-1]
    yy = np.arange(len(top))
    ax.barh(yy, top.PSI_dev_vs_C3, color="#4d4d4d", height=0.6, label="Development (C1+C2) vs C3")
    ax.plot(top.PSI_C1_vs_C2, yy, "o", mfc="white", mec="#111", ms=3.5, label="C1 vs C2 (reference)")
    ax.set_yticks(yy)
    ax.set_yticklabels([VAR_LABELS.get(v, v) for v in top.variable], fontsize=6)
    ax.axvline(0.1, color="#888", lw=0.6, ls=(0, (2, 2)))
    ax.set_xlabel("Population stability index (dashed line: 0.10)")
    ax.set_title("(a) Feature shift, 12 largest PSI", loc="left")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2, fontsize=5.8)
    ax = axes[0, 1]
    lab = "Development (C1+C2) vs C3"
    d = drv[drv.contrast == lab].sort_values("mean_abs_shap", ascending=False).head(10).iloc[::-1]
    yy = np.arange(len(d))
    ax.barh(yy, d.mean_abs_shap, color="#4d4d4d", height=0.6)
    ax.set_yticks(yy)
    ax.set_yticklabels([VAR_LABELS.get(v, v) for v in d.variable], fontsize=6)
    a1, a2 = adv.loc[lab], adv.loc["C1 vs C2"]
    ax.set_xlabel("Mean |SHAP| of domain classifier (log-odds)")
    ax.set_title(f"(b) Adversarial validation: AUROC {a1.AUROC:.3f} (dev vs C3)\n"
                 f"     [reference C1 vs C2: {a2.AUROC:.3f}]", loc="left", fontsize=7)
    ax = axes[1, 0]
    for e, mk, lab2, off in ((1, "o", "Employed at wave 1", -0.06), (0, "s", "Not employed at wave 1", 0.06)):
        d = tr[tr.employed_w1.astype(str) == str(e)]
        x = d.cohort.astype(int).values + off
        ax.errorbar(x, d.p, yerr=[d.p - d.lo, d.hi - d.p], fmt=mk, ms=4, mfc="white" if e == 0 else "#111",
                    mec="#111", ecolor="#111", elinewidth=0.8, capsize=2, label=lab2, ls="-", lw=0.6, color="#555")
        for xi, pi in zip(x, d.p):
            ax.text(xi + 0.08, pi, f"{pi:.2f}", fontsize=5.8, va="center")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["C1 (2013-14)", "C2 (2018-19)", "C3 (2023-24)"])
    ax.set_xlim(0.7, 3.4)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion employed at wave 2")
    ax.set_title("(c) Transition probabilities (95% CI)", loc="left")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.42, 0.66), fontsize=6)
    ax = axes[1, 1]
    labs, vals = [], []
    for task in ("A", "B"):
        i = _tab(out, f"shift_iw_info_task{task}.csv").iloc[0]
        vals.append([i.dev_prevalence, i.dev_prevalence_weighted, i.C3_prevalence])
    vals = np.array(vals)
    x = np.arange(2)
    hatches = [None, "////", None]
    colors = ["#bdbdbd", "#ffffff", "#4d4d4d"]
    names = ["Development (observed)", "Development re-weighted to C3 covariates", "C3 (observed)"]
    for j in range(3):
        b = ax.bar(x + (j - 1) * 0.26, vals[:, j], width=0.24, color=colors[j], edgecolor="#111", lw=0.6,
                   hatch=hatches[j], label=names[j])
        for xi, v in zip(x + (j - 1) * 0.26, vals[:, j]):
            ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", fontsize=5.6)
    ax.set_xticks(x)
    ax.set_xticklabels(["Task A: employed", "Task B: re-entry"])
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_ylabel("Outcome prevalence")
    ax.set_title("(d) Label shift explained by covariate shift", loc="left")
    ax.legend(frameon=True, framealpha=1, edgecolor="none", loc="upper right", fontsize=5.8)
    fig.tight_layout()
    _save(fig, out, "fig4_shift_diagnostics")


def fig_updating(out, task, name):
    r = _tab(out, f"updating_reps_task{task}.csv")
    s = (r.melt(id_vars=["rep", "n", "method", "task", "model"], var_name="metric")
         .groupby(["method", "n", "metric"])["value"]
         .agg(mean="median", p2_5=lambda x: x.quantile(0.025), p97_5=lambda x: x.quantile(0.975)).reset_index())
    # NOTE: column 'mean' holds the MEDIAN across repetitions (robust to degenerate small-sample refits)
    metrics = [("AUROC", "AUROC"), ("Slope", "Calibration slope"), ("CITL", "Calibration-in-the-large"),
               ("Brier", "Brier score")]
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 2.2))
    for i, (k, lab) in enumerate(metrics):
        ax = axes[i]
        base = s[(s.method == "No update") & (s.metric == k)].iloc[0]
        ax.axhspan(base.p2_5, base.p97_5, color="#eeeeee", zorder=0)
        ax.axhline(base["mean"], color="#111", lw=0.9, ls=(0, (1, 1.2)), label="No update")
        if k == "Slope":
            ax.axhline(1, color="#bbbbbb", lw=0.6)
        if k == "CITL":
            ax.axhline(0, color="#bbbbbb", lw=0.6)
        for j, meth in enumerate([m for m in METHOD_STY if m != "No update"]):
            d = s[(s.method == meth) & (s.metric == k)].sort_values("n")
            if len(d) == 0:
                continue
            c, ls, mk = METHOD_STY[meth]
            off = (j - 1.5) * 0.025
            x = np.log10(d.n) + off
            ax.plot(x, d["mean"], color=c, ls=ls, marker=mk, ms=3, lw=1.0, mfc="white", mew=0.7, label=meth)
            ax.vlines(x, d.p2_5, d.p97_5, color=c, lw=0.6)
        ns = sorted(s[s.n > 0].n.unique())
        ax.set_xticks(np.log10(ns))
        ax.set_xticklabels([str(int(n)) for n in ns])
        ax.set_xlabel("C3 updating sample size (n)")
        ax.set_title(lab, loc="left")
        if k == "Slope":
            lo, hi = ax.get_ylim()
            ax.set_ylim(max(lo, 0), min(hi, 2.5))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.09))
    fig.suptitle("")
    fig.tight_layout()
    _save(fig, out, name)


def fig6_fairness(out):
    t = _tab(out, "fairness_taskA_pt30.csv")
    prim = _primary(out, "A")
    metrics = [("AUROC", "(a) Δ AUROC"), ("CITL", "(b) Δ CITL"), ("TPR", "(c) Δ TPR"), ("FPR", "(d) Δ FPR")]
    rows = t[(t.level != t.reference)]
    order = []
    for g in ["Sex", "Age group", "Disability", "Accident type", "Job status"]:
        for lv in rows[(rows.group == g) & (rows.model == prim)].level:
            order.append((g, lv))
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 2.9), sharey=True)
    yy = np.arange(len(order))[::-1]
    for i, (k, lab) in enumerate(metrics):
        ax = axes[i]
        for model, off, mk, fc in ((prim, 0.12, "o", "#111"), ("LR", -0.12, "s", "white")):
            if model == prim and model == "LR":
                off = 0
            d = rows[rows.model == model].set_index(["group", "level"])
            if len(d) == 0 or (model == "LR" and prim == "LR"):
                continue
            est = [d.loc[o, f"gap_{k}"] for o in order]
            lo = [d.loc[o, f"gap_{k}_lo"] for o in order]
            hi = [d.loc[o, f"gap_{k}_hi"] for o in order]
            ax.errorbar(est, yy + off, xerr=[np.subtract(est, lo), np.subtract(hi, est)], fmt=mk, ms=3.2,
                        mfc=fc, mec="#111", ecolor="#555", elinewidth=0.7, capsize=1.5, label=LAB[model])
        ax.axvline(0, color="#888", lw=0.6)
        ax.set_title(lab, loc="left", fontsize=7)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(yy)
    ref = rows.drop_duplicates("group").set_index("group")["reference"]
    axes[0].set_yticklabels([f"{g}: {lv} vs {ref[g]}" for g, lv in order], fontsize=6)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.6, 1.05))
    fig.tight_layout()
    _save(fig, out, "fig6_fairness_forest")


def fig7_shap(out):
    T = pd.read_csv(os.path.join(out, "tables", "shap_importance_taskA.csv"), index_col=0)
    st = _tab(out, "shap_stability_taskA.csv").iloc[0]
    rho = pd.read_csv(os.path.join(out, "tables", "shap_rho_matrix_taskA.csv"), index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.2), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    top = T.sort_values("dev_model_on_dev", ascending=False).head(15).iloc[::-1]
    yy = np.arange(len(top))
    for y_, a_, b_ in zip(yy, top.dev_model_on_dev, top.C3_refit_on_C3):
        ax.plot([a_, b_], [y_, y_], color="#999", lw=0.8)
    ax.plot(top.dev_model_on_dev, yy, "o", color="#111", ms=4, label="Development model (C1+C2)")
    ax.plot(top.C3_refit_on_C3, yy, "s", mfc="white", mec="#111", ms=4, label="Refitted on C3")
    ax.set_yticks(yy)
    ax.set_yticklabels([VAR_LABELS.get(v, v) for v in top.index], fontsize=6)
    ax.set_xlabel("Mean |SHAP| (log-odds)")
    ax.set_title(f"(a) Global importance, {LAB[st.model]} (Task A)", loc="left")
    ax.legend(frameon=False, loc="lower right")
    ax.text(0.98, 0.22, f"Spearman ρ (dev vs C3 refit) = {st.rho_dev_vs_C3refit:.2f}\n"
            f"top-10 overlap = {int(st.top10_overlap)}/10\n"
            f"split-half ρ, median: within C3 {st.splithalf_within_C3_median:.2f};\n"
            f"within dev {st.splithalf_within_dev_median:.2f}; between {st.splithalf_between_dev_C3_median:.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6, bbox=dict(fc="white", ec="#ccc", lw=0.5))
    ax = axes[1]
    keys = ["C1_only", "C2_only", "C3_only", "dev_model_on_dev", "dev_model_on_C3"]
    names = ["C1 refit", "C2 refit", "C3 refit", "Dev model\non dev", "Dev model\non C3"]
    M = rho.loc[keys, keys].values.astype(float)
    ax.imshow(M, cmap="Greys", vmin=0, vmax=1)
    for i in range(len(keys)):
        for j in range(len(keys)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if M[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(keys)))
    ax.set_yticks(range(len(keys)))
    ax.set_xticklabels(names, fontsize=5.8, rotation=45, ha="right")
    ax.set_yticklabels(names, fontsize=5.8)
    ax.grid(False)
    ax.set_title("(b) Spearman ρ between importance rankings", loc="left")
    fig.tight_layout()
    _save(fig, out, "fig7_shap_stability")


def figS_secondary(out):
    t = _tab(out, "secondary_designs.csv")
    prim = pd.concat([_tab(out, "perf_C3_taskA.csv").assign(task="A"), _tab(out, "perf_C3_taskB.csv").assign(task="B")])
    prim["analysis"] = "C1+C2 -> C3 (primary)"
    t = pd.concat([t, prim[prim.metric.isin(["AUROC", "CITL"])]], ignore_index=True)
    designs = ["C1 -> C2 (chain)", "C1+C2 -> C3 (primary)", "C2 -> C3 (S7a: C2-only development)",
               "C2+C3 -> C1 (LOCO)", "C1+C3 -> C2 (LOCO)"]
    models = ["Persistence", "LR-5", "LR", "EN", "RF", "XGB", "LGBM"]
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 4.6), sharey=True)
    for r, task in enumerate(("A", "B")):
        for c, k in enumerate(("AUROC", "CITL")):
            ax = axes[r, c]
            for j, m in enumerate(models):
                if task == "B" and m == "Persistence" and k == "AUROC":
                    continue  # null model: AUROC 0.5 by construction
                d = t[(t.task == task) & (t.metric == k) & (t.model == m)].set_index("analysis").reindex(designs)
                yy = np.arange(len(designs))[::-1] + (j - 3) * 0.1
                col, ls, mk = STY[m]
                ax.errorbar(d.estimate, yy, xerr=[d.estimate - d.lo, d.hi - d.estimate], fmt=mk or "o", ms=2.8,
                            color=col, mfc="white", mec=col, elinewidth=0.6, capsize=0, label=LAB[m])
            if k == "CITL":
                ax.axvline(0, color="#888", lw=0.6)
            ax.set_yticks(np.arange(len(designs))[::-1])
            ax.set_yticklabels(designs, fontsize=6)
            ax.set_title(f"Task {task}: {k}", loc="left")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.55, 1.03))
    fig.tight_layout()
    _save(fig, out, "figS2_secondary_designs")


def make_all(out):
    fig1_design(out)
    fig2_roc_cal(out)
    fig3_dca(out)
    fig4_shift(out)
    fig_updating(out, "A", "fig5_updating_learning_curves")
    fig_updating(out, "B", "figS1_updating_taskB")
    fig6_fairness(out)
    fig7_shap(out)
    figS_secondary(out)
