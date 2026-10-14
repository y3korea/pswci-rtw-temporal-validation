"""Figures for revision v2, drawn only from aggregated CSV tables (greyscale-readable).

Layout pass 2026-10-13 (no change to what is plotted): legends that sat on marks were moved outside the axes
(_legend_outside) or into a corner verified to be empty, the "all (n)" x tick label of Fig. 4b-c is staggered so
it cannot collide with "1000", the smallest text was raised to 6 pt, and the within-C3 series got distinct markers.
Second pass (same day): the Fig. S1 legend is now a figure legend built from the Pooled panel, which carries all six
strategies (the in-panel legend of A1 lacked "Status rule"), "Flag none" got a dash-dot linestyle so that it is not
colour-only versus "Parsimonious LR", and the Fig. 2 calibration legends were tightened so they clear the CI bands.
Consistency corrections 2026-10-15 (no analysis re-run; drawn from the same CSVs): Fig. 1 now follows the steps of
Algorithm 1 (label-free step before outcomes arrive; no re-development loop); estimands are labelled "sample" and
"population" as in the text; model names follow the tables (Random forest, Status-only LR, LR (scores)); Fig. 3a axis
is a log odds ratio and Fig. 3b is the DISDE-type decomposition (C1+C2 tick); Fig. 4a shows CatBoost only (the C1 -> C2
rows, silently XGBoost, were dropped) on a log axis with an outside legend, and the Fig. 4b-c whiskers are dodged by
method; the subgroup levels are ordered consistently with typographic signs; Fig. S4 uses the Fig. 4 styles and has an
x label and a "No update" legend entry; a zero attribution in Fig. S5 is drawn at the axis floor instead of dropped.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from rtw_config import MODEL_LABEL, VAR_LABELS  # noqa: E402

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 7.8, "axes.labelsize": 7.3,
    "legend.fontsize": 6.3, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.4,
})
GREYS = ["#111111", "#555555", "#8a8a8a", "#b5b5b5"]
TASK_TITLE = {"A1": "A1 retention", "A2": "A2 re-entry", "P": "Pooled"}
DESIGNS = ["C1 -> C3", "C2 -> C3", "C1+C2 -> C3"]
SHORT = {"Null": "Null", "Status": "Status-only LR", "LR-small": "Parsimonious LR", "LR": "LR (scores)",
         "RidgeOH": "Ridge-OH", "RF": "Random forest", "XGB": "XGBoost", "LGBM": "LightGBM", "CatBoost": "CatBoost",
         "EBM": "EBM"}
ESTIMAND = {"unweighted": "sample", "weighted": "population"}
ESTIMAND_LONG = {"unweighted": "sample (unweighted)", "weighted": "population (weighted)"}
# display labels only (the CSV values stay as written by the analysis code)
LABELS = {"Age >=60": "Age \u226560", "Disability grade 1-7": "Disability grade 1\u20137",
          "Work ability, mean (0-10)": "Work ability, mean (0\u201310)", "<=39": "\u226439", ">=60": "\u226560",
          "40-49": "40\u201349", "50-59": "50\u201359", "Mild (8-14)": "Mild (8\u201314)",
          "Severe (1-7)": "Severe (1\u20137)", "Work ability (0-10)": "Work ability (0\u201310)"}


def _lab(x):
    return LABELS.get(x, x)


def _save(fig, out, name):
    d = os.path.join(out, "figures")
    fig.savefig(os.path.join(d, name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(d, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def _legend_outside(fig, axes_span, handles, labels, ncol, side="below", fontsize=6.0, gap=0.006):
    """Figure legend centred on the given axes and placed entirely outside the figure area (below or above), so it
    can never cover marks, tick labels or axis labels; bbox_inches="tight" in _save extends the saved page to
    include it. Call after fig.tight_layout() so that the axes positions are final."""
    x0 = min(a.get_position().x0 for a in axes_span)
    x1 = max(a.get_position().x1 for a in axes_span)
    if side == "below":
        anchor, loc = ((x0 + x1) / 2, -gap), "upper center"
    else:
        anchor, loc = ((x0 + x1) / 2, 1 + gap), "lower center"
    return fig.legend(handles, labels, loc=loc, bbox_to_anchor=anchor, ncol=ncol, frameon=False, fontsize=fontsize,
                      handlelength=1.8, columnspacing=1.2, handletextpad=0.5)


def _t(out, name):
    return pd.read_csv(os.path.join(out, "tables", name))


def _primary(out, task):
    t = _t(out, "tuning_primary.csv")
    t = t[(t.task == task)]
    return t.loc[t.repeated_cv_logloss.idxmin(), "model"]


def fig1_framework(out):
    """Protocol figure; the boxes follow the steps of Algorithm 1 in the manuscript (2026-10-15)."""
    fig, ax = plt.subplots(figsize=(6.9, 2.2))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    steps = [("Harmonise", "C1\u2013C3\n(wave 1 in 2013,\n2018 and 2023)"),
             ("Step 1: Develop", "C1+C2; seven\nlearners; selection\nby repeated cross-\nvalidation; keep\nout-of-fold\npredictions"),
             ("Step 2: New covariates\n(no outcomes yet)", "adversarial AUROC;\ndensity ratio and effective\n"
              "sample size; importance-\nweighted recalibration only if\nthe shift is mainly in X and the\n"
              "weighted out-of-fold calibration\nimproves; otherwise keep the\nmodel and monitor"),
             ("Step 3a\u2013b:\nOutcomes arrive", "evaluate in the\nsample and the\npopulation; decompose\nthe change by\nsource cohort"),
             ("Step 3c\u2013d:\nUpdate and audit", "recalibrate with\nrecent records when\nmiscalibration is\nclear; audit\nsubgroup errors")]
    widths = [0.13, 0.175, 0.245, 0.18, 0.18]
    gap, y0, h = 0.019, 0.25, 0.73
    xs = []
    x = 0.008
    for i, ((title, body), w) in enumerate(zip(steps, widths)):
        xs.append((x, w))
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.005,rounding_size=0.02",
                                    fc="#f0f0f0" if i % 2 == 0 else "white", ec="#222222", lw=0.8))
        ax.text(x + w / 2, y0 + h - 0.05, title, ha="center", va="top", fontsize=6.3, fontweight="bold",
                linespacing=1.15)
        ax.text(x + w / 2, y0 + h - 0.25, body, ha="center", va="top", fontsize=5.6, linespacing=1.25)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + w + gap, y0 + h / 2), xytext=(x + w, y0 + h / 2),
                        arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#222222", mutation_scale=7,
                                        shrinkA=0, shrinkB=0))
        x += w + gap
    x_from = xs[4][0] + xs[4][1] / 2
    x_to = xs[2][0] + xs[2][1] / 2
    yl = 0.12
    ax.plot([x_from, x_from, x_to], [y0 - 0.01, yl, yl], color="#777777", lw=0.7, ls="--")
    ax.annotate("", xy=(x_to, y0 - 0.01), xytext=(x_to, yl),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#777777", ls="--", shrinkA=0, shrinkB=0))
    ax.text((x_from + x_to) / 2, yl - 0.02, "next cohort: return to step 2", ha="center", va="top", fontsize=6.0,
            color="#555555")
    _save(fig, out, "fig1_framework")


def fig2_performance(out):
    """(a,b) calibration curves with bands, unweighted and weighted, for A1 and A2; (c) AUROC by model."""
    fig = plt.figure(figsize=(6.9, 4.9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.15])
    for j, task in enumerate(("A1", "A2")):
        cal = _t(out, f"calcurve_{task}.csv")
        prim = _primary(out, task)
        for i, lab in enumerate(("unweighted", "weighted")):
            ax = fig.add_subplot(gs[i, j])
            for m, col, ls in ((prim, "#111111", "-"), ("RidgeOH", "#777777", (0, (4, 2)))):
                d = cal[(cal.model == m) & (cal.weighting == lab)]
                if len(d) == 0:
                    continue
                ax.fill_between(d.pred, d.lo, d.hi, color=col, alpha=0.15, lw=0)
                ax.plot(d.pred, d.obs, color=col, ls=ls, lw=1.1, label=SHORT.get(m, m))
            lo = 0.4 if task == "A1" else 0
            ax.plot([lo, 1], [lo, 1], color="#bbbbbb", lw=0.6)
            ax.set_xlim(lo, 1)
            ax.set_ylim(lo, 1)
            ax.set_title(f"({'abcd'[2 * j + i]}) {TASK_TITLE[task]}, {ESTIMAND[lab]}", loc="left")
            ax.set_xlabel("Predicted P(employed)")
            ax.set_ylabel("Observed")
            # compact legend: with the default pad/handle length the "Ridge-OH" label touched the CI band in (b)
            ax.legend(frameon=False, loc="upper left", borderaxespad=0.2, handlelength=1.5)
    ax = fig.add_subplot(gs[:, 2])
    rows = []
    for task in ("A1", "A2", "P"):
        for lab in ("unweighted", "weighted"):
            t = _t(out, f"perf_C3_{task}_{lab}.csv")
            t = t[(t.metric == "AUROC") & (t.model != "Null")]
            for _, r in t.iterrows():
                rows.append((task, lab, r.model, r.estimate, r.lo, r.hi))
    df = pd.DataFrame(rows, columns=["task", "w", "model", "est", "lo", "hi"])
    order = ["Status", "LR-small", "LR", "RidgeOH", "RF", "XGB", "LGBM", "CatBoost", "EBM"]
    ylab, y = [], 0
    for task in ("A1", "A2", "P"):
        for m in order:
            dd = df[(df.task == task) & (df.model == m)]
            if len(dd) == 0:
                continue
            for lab, mk, fc, off in (("unweighted", "o", "#111111", 0.15), ("weighted", "s", "white", -0.15)):
                r = dd[dd.w == lab]
                if len(r):
                    r = r.iloc[0]
                    ax.errorbar(r.est, -y + off, xerr=np.clip(np.array([[r.est - r.lo], [r.hi - r.est]], dtype=float), 0, None), fmt=mk, ms=2.8, mfc=fc,
                                mec="#111111", ecolor="#666666", elinewidth=0.6, capsize=0,
                                label=ESTIMAND_LONG[lab] if (task == "A1" and m == "LR-small") else None)
            ylab.append((-y, f"{task}: {SHORT[m]}"))
            y += 1
        y += 0.6
    ax.set_yticks([p for p, _ in ylab])
    ax.set_yticklabels([l for _, l in ylab], fontsize=6.0)
    ax.set_xlabel("AUROC in C3 (95% CI)")
    ax.set_title("(e) Discrimination", loc="left")
    h_e, l_e = ax.get_legend_handles_labels()
    fig.tight_layout()
    # legend of (e) below the panel (2026-10-15): inside the axes the longer estimand labels touched the P rows
    _legend_outside(fig, [ax], h_e, l_e, ncol=1, side="below")
    _save(fig, out, "fig2_performance")


def fig3_shift(out):
    ce = _t(out, "shift_cohort_effects.csv")
    dis = _t(out, "shift_disde.csv")
    adv = _t(out, "shift_adversarial.csv")
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.6), gridspec_kw={"width_ratios": [1, 1.5, 0.9]})
    ax = axes[0]
    y = 0
    labels = []
    for task in ("A1", "A2", "P"):
        for c in ("C2 vs C1", "C3 vs C1", "C3 vs C2"):
            r = ce[(ce.task == task) & (ce.contrast == c)].iloc[0]
            ax.errorbar(r.log_odds, -y, xerr=np.clip(np.array([[r.log_odds - r.lo], [r.hi - r.log_odds]], dtype=float), 0, None), fmt="o", ms=3, color="#111111",
                        elinewidth=0.7)
            labels.append((-y, f"{TASK_TITLE[task]}: {c}"))
            y += 1
        y += 0.5
    ax.axvline(0, color="#888888", lw=0.6)
    ax.set_yticks([p for p, _ in labels])
    ax.set_yticklabels([l for _, l in labels], fontsize=6.0)
    ax.set_xlabel("Adjusted log odds ratio of employment")
    ax.set_title("(a) Conditional cohort effects", loc="left")
    ax = axes[1]
    comps = [("X_shift_source_side", "X-shift (source)", "#d9d9d9", None), ("YgivenX_shift", "Y|X shift", "#444444", None),
             ("X_shift_target_side", "X-shift (target)", "white", "////")]
    xs = []
    k = 0
    for task in ("A1", "A2"):
        for des in DESIGNS:
            sub = dis[(dis.task == task) & (dis.design == des) & (dis.metric == "Prevalence")]
            if len(sub) == 0:
                continue
            bottom_pos, bottom_neg = 0, 0
            for comp, lab, fc, hatch in comps:
                r = sub[sub.component == comp].iloc[0]
                v = r.estimate
                b = bottom_pos if v >= 0 else bottom_neg
                ax.bar(k, v, bottom=b, color=fc, edgecolor="#111111", lw=0.5, hatch=hatch, width=0.7,
                       label=lab if k == 0 else None)
                if v >= 0:
                    bottom_pos += v
                else:
                    bottom_neg += v
            tot = sub[sub.component == "total"].iloc[0]
            ax.plot(k, tot.estimate, "D", ms=3.5, mfc="white", mec="#111111", label="Total change" if k == 0 else None)
            xs.append((k, {"C1 -> C3": "C1", "C2 -> C3": "C2", "C1+C2 -> C3": "C1+C2"}[des] + f"\n{task}"))
            k += 1
        k += 0.6
    ax.axhline(0, color="#888888", lw=0.6)
    ax.set_xticks([p for p, _ in xs])
    ax.set_xticklabels([l for _, l in xs], fontsize=6.0)
    ax.set_ylabel("Change in employment prevalence")
    ax.set_xlabel("Source cohort(s) → C3")
    ax.set_title("(b) DISDE-type decomposition", loc="left")
    h_b, l_b = ax.get_legend_handles_labels()   # legend drawn below the panel (an inside legend covered the C1/A2 bar)
    ax = axes[2]
    a = adv[adv.predictors == "all"].reset_index(drop=True)
    b = adv[adv.predictors == "restricted"].reset_index(drop=True)
    for i, r in a.iterrows():
        ax.errorbar(r.AUROC, -i + 0.12, xerr=np.clip(np.array([[r.AUROC - r.lo], [r.hi - r.AUROC]], dtype=float), 0, None), fmt="o", ms=3, color="#111111",
                    label="all predictors" if i == 0 else None)
    for i, r in b.iterrows():
        ax.errorbar(r.AUROC, -i - 0.12, xerr=np.clip(np.array([[r.AUROC - r.lo], [r.hi - r.AUROC]], dtype=float), 0, None), fmt="s", ms=3, mfc="white",
                    mec="#111111", ecolor="#111111", label="restricted (S4b)" if i == 0 else None)
    ax.set_yticks([-i for i in range(len(a))])
    ax.set_yticklabels(a.contrast.str.replace("+", "+", regex=False), fontsize=6.0)
    ax.set_xlabel("Domain-classifier AUROC")
    ax.set_title("(c) Adversarial validation", loc="left")
    h_c, l_c = ax.get_legend_handles_labels()   # legend drawn below the panel (an inside legend covered C1 vs C2)
    fig.tight_layout()
    _legend_outside(fig, [axes[1]], h_b, l_b, ncol=2, side="below")
    _legend_outside(fig, [axes[2]], h_c, l_c, ncol=1, side="below")
    _save(fig, out, "fig3_shift")


def fig4_updating(out):
    iw = _t(out, "update_iw_recalibration.csv")
    lg = _t(out, "update_lagged_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.5))
    ax = axes[0]
    meths = [("No update", "o", "#111111"), ("OOF recalibration (unweighted)", "^", "#888888"),
             ("IW logistic recalibration", "s", "white")]
    IW_LABEL = {"No update": "No update", "OOF recalibration (unweighted)": "Unweighted recal. (out-of-fold)",
                "IW logistic recalibration": "Importance-weighted recal."}
    ylab = []
    y = 0
    for task in ("A1", "A2"):
        prim = _primary(out, task)
        for des in DESIGNS:   # 2026-10-15: C3 designs only, primary model only (no silent XGBoost fallback)
            sub = iw[(iw.task == task) & (iw.design == des) & (iw.model == prim) & (iw.metric == "Slope")]
            if len(sub) == 0:
                raise ValueError(f"no {prim} rows for {task} {des} in update_iw_recalibration.csv")
            for j, (meth, mk, fc) in enumerate(meths):
                r = sub[sub.method == meth]
                if len(r):
                    r = r.iloc[0]
                    ax.errorbar(r.estimate, -y + (1 - j) * 0.18, xerr=np.clip(np.array([[r.estimate - r.lo], [r.hi - r.estimate]], dtype=float), 0, None),
                                fmt=mk, ms=2.8, mfc=fc, mec="#111111", ecolor="#777777", elinewidth=0.6,
                                label=IW_LABEL[meth] if y == 0 else None)
            ylab.append((-y, f"{task}: {des.replace(' -> ', '→')}"))
            y += 1
        y += 0.5
    ax.axvline(1, color="#888888", lw=0.6)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 4, 8])
    ax.set_xticklabels(["0.5", "1", "2", "4", "8"])
    ax.minorticks_off()
    ax.set_yticks([p for p, _ in ylab])
    ax.set_yticklabels([l for _, l in ylab], fontsize=6.0)
    ax.set_xlabel("Calibration slope in C3 (log scale)")
    ax.set_title("(a) Label-free recalibration", loc="left")
    h_a, l_a = ax.get_legend_handles_labels()   # drawn below the panel: on a log axis an inside legend covered rows
    sty = {"Intercept recalibration": ("#111111", "-", "o"), "Logistic recalibration": ("#555555", (0, (4, 2)), "s"),
           "Closed testing": ("#888888", (0, (1, 1)), "^"), "Pooled refit (C1 + C2 sample)": ("#111111", (0, (5, 1, 1, 1)), "D"),
           "Ridge refit on C2 sample only": ("#aaaaaa", "-", "v")}
    for k, task in enumerate(("A1", "A2")):
        ax = axes[k + 1]
        s = lg[lg.task == task]
        base = s[s.method == "No update (C1 model)"].iloc[0]
        ax.axhline(base["ICI_median"], color="#bbbbbb", lw=1.2, label="No update (C1 model)")
        for mi, (meth, (c, ls, mk)) in enumerate(sty.items()):
            d = s[(s.method == meth)].sort_values("n")
            if len(d) == 0:
                continue
            x = np.log10(d.n) + (mi - 2) * 0.012   # dodge so that each method's whisker is visible
            ax.plot(x, d["ICI_median"], color=c, ls=ls, marker=mk, ms=2.8, lw=0.9, mfc="white", label=meth)
            ax.vlines(x, d["ICI_p2_5"], d["ICI_p97_5"], color=c, lw=0.5)
        ns = sorted(s[s.n > 0].n.unique())
        ticks = [n for n in ns if n in (100, 250, 500, 1000)] + [max(ns)]
        ax.set_xticks(np.log10(ticks))
        # "all (n)" is written one line lower than the numeric ticks: log10(1130) sits next to log10(1000)
        ax.set_xticklabels([str(int(n)) for n in ticks[:-1]] + [f"\nall\n({int(ticks[-1])})"], fontsize=6.0)
        ax.set_xlabel("C2 records used for updating")
        ax.set_ylabel("ICI in C3")
        ax.set_title(f"({'bc'[k]}) Lagged update, {TASK_TITLE[task]}", loc="left")
    fig.tight_layout()
    # shared legend of (b) and (c) under those panels; an inside legend covered the no-update and refit lines in (c)
    _legend_outside(fig, axes[1:], *axes[2].get_legend_handles_labels(), ncol=3, side="below")
    _legend_outside(fig, [axes[0]], h_a, l_a, ncol=1, side="below")
    _save(fig, out, "fig4_updating")


def figS_fairness(out):
    f = _t(out, "fairness.csv")
    for task in ("A1", "A2"):
        sub = f[(f.task == task) & (f.weighting == "unweighted")]
        prim = _primary(out, task)
        rows = sub[(sub.level != sub.reference) & sub.gap_TPR.notna()]
        order = rows[rows.model == prim][["group", "level"]].drop_duplicates().values.tolist()
        # 2026-10-15: keep the group order but sort the levels within a group (age was non-monotonic in Fig. S2)
        rank = {lv: k for k, lv in enumerate(["Female", "40-49", "50-59", ">=60", "Severe (1-7)", "No grade",
                                               "Disease", "Non-regular"])}
        order = sorted(order, key=lambda o: (order.index(next(p for p in order if p[0] == o[0])), rank.get(o[1], 99)))
        mets = [("AUROC", "Δ AUROC"), ("CITL", "Δ CITL"), ("TPR", "Δ TPR"), ("FPR", "Δ FPR")]
        fig, axes = plt.subplots(1, 4, figsize=(6.9, 2.6), sharey=True)
        yy = np.arange(len(order))[::-1]
        for i, (k, lab) in enumerate(mets):
            ax = axes[i]
            for model, off, mk, fc in ((prim, 0.12, "o", "#111111"), ("RidgeOH", -0.12, "s", "white")):
                d = rows[rows.model == model].set_index(["group", "level"])
                if len(d) == 0 or (model == "RidgeOH" and prim == "RidgeOH"):
                    continue
                e = np.array([d.loc[tuple(o), f"gap_{k}"] for o in order])
                lo = np.array([d.loc[tuple(o), f"gap_{k}_lo"] for o in order])
                hi = np.array([d.loc[tuple(o), f"gap_{k}_hi"] for o in order])
                ax.errorbar(e, yy + off, xerr=np.clip(np.array([e - lo, hi - e], dtype=float), 0, None), fmt=mk, ms=2.8, mfc=fc, mec="#111111",
                            ecolor="#666666", elinewidth=0.6, label=SHORT.get(model, model))
            ax.axvline(0, color="#888888", lw=0.6)
            ax.set_title(lab, loc="left")
            ax.grid(axis="y", visible=False)
        ref = rows.drop_duplicates("group").set_index("group")["reference"]
        axes[0].set_yticks(yy)
        axes[0].set_yticklabels([f"{g}: {_lab(lv)} vs {_lab(ref[g])}" for g, lv in order], fontsize=6.0)
        h, l = axes[0].get_legend_handles_labels()
        fig.tight_layout()
        _legend_outside(fig, axes, h, l, ncol=2, side="above", fontsize=6.3)
        _save(fig, out, f"figS_fairness_{task}")


def figS_dca(out):
    fig, axes = plt.subplots(2, 3, figsize=(6.9, 4.2))
    for j, task in enumerate(("A1", "A2", "P")):
        for i, lab in enumerate(("unweighted", "weighted")):
            ax = axes[i, j]
            d = _t(out, f"dca_{task}_{lab}.csv")
            # "Flag none" is dash-dot so that it differs from "Parsimonious LR" by more than its grey level
            styles = {"Flag all": ("#777777", (0, (2, 2))), "Flag none": ("#aaaaaa", (0, (3, 1, 1, 1))),
                      "Status rule": ("#555555", (0, (1, 1))), "RidgeOH": ("#777777", (0, (4, 2))),
                      "LR-small": ("#bbbbbb", "-")}
            prim = _primary(out, task)
            styles[prim] = ("#111111", "-")
            ymax = 0
            for s, (c, ls) in styles.items():
                dd = d[d.strategy == s]
                if len(dd):
                    ax.plot(dd.threshold, dd.net_benefit, color=c, ls=ls, lw=1.0, label=SHORT.get(s, s))
                    ymax = max(ymax, dd.net_benefit.max())
            ax.set_ylim(-0.02, ymax * 1.1 + 1e-3)
            ax.set_xlim(0.02, 0.9)
            ax.set_title(f"{TASK_TITLE[task]}, {ESTIMAND[lab]}", loc="left")
            ax.set_xlabel("Threshold risk of undesired outcome")
            ax.set_ylabel("Net benefit")
    fig.tight_layout()
    # one legend above all panels, taken from the Pooled-unweighted axes: only the pooled tables contain the
    # "Status rule" strategy, so a legend built from the A1 panel would miss that curve
    _legend_outside(fig, list(axes.flat), *axes[0, 2].get_legend_handles_labels(), ncol=6, side="above")
    _save(fig, out, "figS_decision_curves")


def figS_withinc3(out):
    s = _t(out, "update_withinC3_summary.csv")
    fig, axes = plt.subplots(2, 3, figsize=(6.9, 4.0))
    for i, task in enumerate(("A1", "A2")):
        d0 = s[s.task == task]
        base = d0[d0.method == "No update"].iloc[0]
        for j, k in enumerate(("AUROC", "Brier", "ICI")):
            ax = axes[i, j]
            ax.axhline(base[f"{k}_median"], color="#bbbbbb", lw=1.2, label="No update" if (i == 0 and j == 0) else None)
            # markers, grey levels and line styles follow Fig. 4 for the same methods (2026-10-15)
            for meth, c, ls, mk in (("Intercept recalibration", "#111111", "-", "o"),
                                    ("Logistic recalibration", "#555555", (0, (4, 2)), "s"),
                                    ("Closed testing", "#888888", (0, (1, 1)), "^"),
                                    ("Refit on C3 sample (development hyperparameters)", "#777777", (0, (2, 1)), "P"),
                                    ("Ridge refit on C3 sample (CV-tuned)", "#aaaaaa", "-", "v"),
                                    ("Pooled refit (dev + C3 sample)", "#111111", (0, (5, 1, 1, 1)), "D")):
                d = d0[d0.method == meth].sort_values("n")
                if len(d):
                    ax.plot(np.log10(d.n), d[f"{k}_median"], color=c, ls=ls, marker=mk, ms=2.6, mfc="white", lw=0.9,
                            label=meth if (i == 0 and j == 0) else None)
            ns = sorted(d0[d0.n > 0].n.unique())
            ax.set_xticks(np.log10(ns))
            ax.set_xticklabels([str(int(n)) for n in ns])
            ax.set_title(f"{TASK_TITLE[task]}: {k}", loc="left")
            if i == 1:
                ax.set_xlabel("C3 records used for updating")
    fig.tight_layout()
    _legend_outside(fig, list(axes.flat), *axes[0, 0].get_legend_handles_labels(), ncol=3, side="above")
    _save(fig, out, "figS_withinC3_updating")


def figS_attribution(out):
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.0))
    for k, task in enumerate(("A1", "A2")):
        t = _t(out, f"attribution_{task}.csv").head(12).iloc[::-1]
        ax = axes[k]
        yy = np.arange(len(t))
        ax.plot(t.deployed_on_dev, yy, "o", color="#111111", ms=3.5, label="Deployed model, development data")
        ax.plot(t.deployed_on_C3, yy, "s", mfc="white", mec="#111111", ms=3.5, label="Deployed model, C3 data")
        pos = t.refit_on_C3 > 0
        ax.plot(t.refit_on_C3[pos], yy[pos.values], "^", color="#888888", ms=3.2, label="Refitted on C3")
        if (~pos).any():   # 2026-10-15: a zero cannot be drawn on a log axis; show it at the axis floor
            floor = min(t[["deployed_on_dev", "deployed_on_C3"]].min().min(), t.refit_on_C3[pos].min()) / 1.6
            ax.plot(np.full((~pos).sum(), floor), yy[(~pos).values], "<", mfc="white", mec="#888888", ms=3.4,
                    label="Refitted on C3: zero (drawn at axis floor)")
        ax.set_yticks(yy)
        ax.set_yticklabels([_lab(VAR_LABELS.get(v, v)) for v in t.variable], fontsize=6.0)
        ax.set_xscale("log")
        ax.set_xlabel("Mean |attribution| (log-odds, log scale)")
        ax.set_title(f"{TASK_TITLE[task]} ({SHORT.get(t.model.iloc[0], t.model.iloc[0])})", loc="left")
    fig.tight_layout()
    # shared legend above both panels; an inside legend covered the lowest-ranked variables of the retention panel
    h0, l0 = axes[0].get_legend_handles_labels()
    h1, l1 = axes[1].get_legend_handles_labels()
    extra = [(h, l) for h, l in zip(h1, l1) if l not in l0]   # the zero marker may exist in one panel only
    _legend_outside(fig, axes, h0 + [h for h, _ in extra], l0 + [l for _, l in extra], ncol=4 if extra else 3,
                    side="above")
    _save(fig, out, "figS_attribution")


def make_all(out):
    fig1_framework(out)
    fig2_performance(out)
    fig3_shift(out)
    fig4_updating(out)
    figS_fairness(out)
    figS_dca(out)
    figS_withinc3(out)
    figS_attribution(out)
