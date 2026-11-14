"""Fig. S9: unweighted vs survey-weighted calibration curves of the primary (CatBoost) models in C3,
drawn from the saved aggregate tables calcurve_<task>.csv (no refit).
Usage: python src/make_fig_s9.py <tables folder> <output png>"""
import sys, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
T, out = sys.argv[1], sys.argv[2]
names = {"A1": "Retention (A1)", "A2": "Re-entry (A2)", "P": "Pooled (P)"}
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.6))
for ax, t in zip(axs, ["A1", "A2", "P"]):
    d = pd.read_csv(f"{T}/calcurve_{t}.csv"); d = d[d.model == "CatBoost"]
    for w, c, ls, lab in (("unweighted", "#1f4e79", "-", "Sample"), ("weighted", "#c0392b", "--", "Survey-weighted population")):
        e = d[d.weighting == w]
        ax.fill_between(e.pred, e.lo, e.hi, color=c, alpha=0.12, lw=0)
        ax.plot(e.pred, e.obs, ls, color=c, lw=1.8, label=lab)
    lo = min(d.pred.min(), 0); ax.plot([0, 1], [0, 1], ":", color="grey", lw=1)
    ax.set_xlim(d.pred.min() - 0.02, d.pred.max() + 0.02); ax.set_ylim(0, 1)
    ax.set_title(names[t], fontsize=10); ax.set_xlabel("Predicted probability of employment"); ax.grid(alpha=0.3)
axs[0].set_ylabel("Observed proportion employed"); axs[0].legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(out, dpi=300)
