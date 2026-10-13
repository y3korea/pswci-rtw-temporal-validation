"""Figures for revision v2, drawn only from aggregated CSV tables (greyscale-readable).

Layout pass 2026-10-13 (no change to what is plotted): legends that sat on marks were moved outside the axes
(_legend_outside) or into a corner verified to be empty, the "all (n)" x tick label of Fig. 4b-c is staggered so
it cannot collide with "1000", the smallest text was raised to 6 pt, and the within-C3 series got distinct markers.
Second pass (same day): the Fig. S1 legend is now a figure legend built from the Pooled panel, which carries all six
strategies (the in-panel legend of A1 lacked "Status rule"), "Flag none" got a dash-dot linestyle so that it is not
colour-only versus "Parsimonious LR", and the Fig. 2 calibration legends were tightened so they clear the CI bands.
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
SHORT = {"Null": "Null", "Status": "Status-only", "LR-small": "Parsimonious LR", "LR": "LR", "RidgeOH": "Ridge-OH",
         "RF": "RF", "XGB": "XGBoost", "LGBM": "LightGBM", "CatBoost": "CatBoost", "EBM": "EBM"}


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
    fig, ax = plt.subplots(figsize=(6.9, 2.1))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    steps = [("1 Harmonise", "three national\ncohorts\n(2013/18/23)"),
             ("2 Develop", "C1+C2; seven\nlearners;\nrepeated-CV\nselection"),
             ("3 Validate", "C3: sample and\npopulation\n(weighted)"),
             ("4 Diagnose", "adversarial\nAUROC, cohort\neffects, DISDE\n(X vs Y|X)"),
             ("5 Update", "conditional IW\nrecalibration;\nguarded labelled\nupdating"),
             ("6 Monitor", "net benefit,\nsubgroup errors,\nattributions")]
    w, h, gap, y0 = 0.143, 0.62, 0.022, 0.3
    for i, (title, body) in enumerate(steps):
        x = 0.005 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.005,rounding_size=0.02",
                                    fc="#f0f0f0" if i % 2 == 0 else "white", ec="#222222", lw=0.8))
        ax.text(x + w / 2, y0 + h - 0.1, title, ha="center", va="center", fontsize=6.8, fontweight="bold")
        ax.text(x + w / 2, y0 + h / 2 - 0.08, body, ha="center", va="center", fontsize=6.0, linespacing=1.25)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + w + gap, y0 + h / 2), xytext=(x + w, y0 + h / 2),
                        arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#222222"))
    x_from = 0.005 + 5 * (w + gap) + w / 2
    x_to = 0.005 + 1 * (w + gap) + w / 2
    ax.annotate("", xy=(x_to, y0 - 0.02), xytext=(x_from, y0 - 0.02),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#777777", ls="--",
                                connectionstyle="arc3,rad=-0.12"))
    ax.text(0.5, 0.02, "re-develop when diagnostics or monitoring indicate Y|X shift", ha="center", fontsize=6.2,
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
            ax.set_title(f"({'abcd'[2 * j + i]}) {TASK_TITLE[task]}, {lab}", loc="left")
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
                                label=lab if (task == "A1" and m == "LR-small") else None)
            ylab.append((-y, f"{task}: {SHORT[m]}"))
            y += 1
        y += 0.6
    ax.set_yticks([p for p, _ in ylab])
    ax.set_yticklabels([l for _, l in ylab], fontsize=6.0)
    ax.set_xlabel("AUROC in C3 (95% CI)")
    ax.set_title("(e) Discrimination", loc="left")
    # lower-left corner is empty (pooled-task AUROCs and their CIs all lie above 0.7); lower right covered P: EBM
    ax.legend(frameon=False, loc="lower left", fontsize=6.0)
    fig.tight_layout()
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
    ax.set_xlabel("Adjusted log-odds of employment")
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
            xs.append((k, {"C1 -> C3": "C1", "C2 -> C3": "C2", "C1+C2 -> C3": "C1+2"}[des] + f"\n{task}"))
            k += 1
        k += 0.6
    ax.axhline(0, color="#888888", lw=0.6)
    ax.set_xticks([p for p, _ in xs])
    ax.set_xticklabels([l for _, l in xs], fontsize=6.0)
    ax.set_ylabel("Change in employment prevalence")
    ax.set_xlabel("Source cohort(s) → C3")
    ax.set_title("(b) DISDE decomposition of outcome change", loc="left")
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
    ylab = []
    y = 0
    for task in ("A1", "A2"):
        prim = _primary(out, task)
        for des in ("C1 -> C3", "C2 -> C3", "C1+C2 -> C3", "C1 -> C2"):
            sub = iw[(iw.task == task) & (iw.design == des) & (iw.model == prim) & (iw.metric == "Slope")]
            if len(sub) == 0:
                sub = iw[(iw.task == task) & (iw.design == des) & (iw.model == "XGB") & (iw.metric == "Slope")]
            for j, (meth, mk, fc) in enumerate(meths):
                r = sub[sub.method == meth]
                if len(r):
                    r = r.iloc[0]
                    ax.errorbar(r.estimate, -y + (1 - j) * 0.18, xerr=np.clip(np.array([[r.estimate - r.lo], [r.hi - r.estimate]], dtype=float), 0, None),
                                fmt=mk, ms=2.8, mfc=fc, mec="#111111", ecolor="#777777", elinewidth=0.6,
                                label=meth.replace(" (unweighted)", "").replace("recalibration", "recal.") if y == 0 else None)
            ylab.append((-y, f"{task}: {des.replace(' -> ', '→')}"))
            y += 1
        y += 0.5
    ax.axvline(1, color="#888888", lw=0.6)
    ax.set_yticks([p for p, _ in ylab])
    ax.set_yticklabels([l for _, l in ylab], fontsize=6.0)
    ax.set_xlabel("Calibration slope in target")
    ax.set_title("(a) Label-free recalibration", loc="left")
    # lower-right corner is empty (the bottom rows' CIs end below slope 2); upper right touched the A1 C1->C3 CI
    ax.legend(frameon=False, fontsize=6.0, loc="lower right")
    sty = {"Intercept recalibration": ("#111111", "-", "o"), "Logistic recalibration": ("#555555", (0, (4, 2)), "s"),
           "Closed testing": ("#888888", (0, (1, 1)), "^"), "Pooled refit (C1 + C2 sample)": ("#111111", (0, (5, 1, 1, 1)), "D"),
           "Ridge refit on C2 sample only": ("#aaaaaa", "-", "v")}
    for k, task in enumerate(("A1", "A2")):
        ax = axes[k + 1]
        s = lg[lg.task == task]
        base = s[s.method == "No update (C1 model)"].iloc[0]
        ax.axhline(base["ICI_median"], color="#bbbbbb", lw=1.2, label="No update (C1 model)")
        for meth, (c, ls, mk) in sty.items():
            d = s[(s.method == meth)].sort_values("n")
            if len(d) == 0:
                continue
            x = np.log10(d.n)
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
    _save(fig, out, "fig4_updating")


def figS_fairness(out):
    f = _t(out, "fairness.csv")
    for task in ("A1", "A2"):
        sub = f[(f.task == task) & (f.weighting == "unweighted")]
        prim = _primary(out, task)
        rows = sub[(sub.level != sub.reference) & sub.gap_TPR.notna()]
        order = rows[rows.model == prim][["group", "level"]].drop_duplicates().values.tolist()
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
        axes[0].set_yticklabels([f"{g}: {lv} vs {ref[g]}" for g, lv in order], fontsize=6.0)
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
            ax.set_title(f"{TASK_TITLE[task]}, {lab}", loc="left")
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
            ax.axhline(base[f"{k}_median"], color="#bbbbbb", lw=1.2)
            # markers follow Fig. 4 for the same methods so that the series stay distinct in greyscale
            for meth, c, ls, mk in (("Intercept recalibration", "#111111", "-", "o"),
                                    ("Logistic recalibration", "#555555", (0, (4, 2)), "s"),
                                    ("Closed testing", "#888888", (0, (1, 1)), "^"),
                                    ("Refit on C3 sample (development hyperparameters)", "#aaaaaa", "-", "P"),
                                    ("Ridge refit on C3 sample (CV-tuned)", "#777777", (0, (5, 1, 1, 1)), "v"),
                                    ("Pooled refit (dev + C3 sample)", "#333333", (0, (2, 1)), "D")):
                d = d0[d0.method == meth].sort_values("n")
                if len(d):
                    ax.plot(np.log10(d.n), d[f"{k}_median"], color=c, ls=ls, marker=mk, ms=2.6, mfc="white", lw=0.9,
                            label=meth if (i == 0 and j == 0) else None)
            ns = sorted(d0[d0.n > 0].n.unique())
            ax.set_xticks(np.log10(ns))
            ax.set_xticklabels([str(int(n)) for n in ns])
            ax.set_title(f"{TASK_TITLE[task]}: {k}", loc="left")
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
        ax.plot(t.refit_on_C3, yy, "^", color="#888888", ms=3.2, label="Refitted on C3")
        ax.set_yticks(yy)
        ax.set_yticklabels([VAR_LABELS.get(v, v) for v in t.variable], fontsize=6.0)
        ax.set_xscale("log")
        ax.set_xlabel("Mean |attribution| (log-odds, log scale)")
        ax.set_title(f"{TASK_TITLE[task]} ({SHORT.get(t.model.iloc[0], t.model.iloc[0])})", loc="left")
    fig.tight_layout()
    # shared legend above both panels; an inside legend covered the lowest-ranked variables of the retention panel
    _legend_outside(fig, axes, *axes[0].get_legend_handles_labels(), ncol=3, side="above")
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
