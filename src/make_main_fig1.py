"""Main-text Fig. 1: model-free DISDE-type decomposition of the change in employment prevalence from each source
design to C3, re-entry (A2) and retention (A1), at the JKSCI column width.

Reads only the aggregate table shift_disde.csv (model-free rows, metric Prevalence), the same rows as Supplementary
Table S15 and Fig. S3b. Plotted: total change and Y|X component with their 95% intervals (200 resamples), and the
X-shift component as the sum of the source-side and target-side components (point only; the table gives each with its
interval). The three components add up to the total (checked below). Width 22,530 HWPUNIT = 79.5 mm, the column width
of the journal template, so the 7-pt labels print at their size.

Usage: python src/make_main_fig1.py <tables folder> <output png> [dpi, default 600]
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

T, OUT = sys.argv[1], sys.argv[2]
DPI = int(sys.argv[3]) if len(sys.argv) > 3 else 600
CM = 1 / 2.54
WIDTH_CM = 22530 / 283.465 / 10
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 0,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})

D = {}
for r in csv.DictReader(open(os.path.join(T, "shift_disde.csv"), encoding="utf-8")):
    if r["model"] == "model-free" and r["metric"] == "Prevalence":
        D[(r["task"], r["design"], r["component"])] = tuple(float(r[k]) for k in ("estimate", "lo", "hi"))

DESIGNS = ["C1 -> C3", "C2 -> C3", "C1+C2 -> C3"]
LAB = {"C1 -> C3": "C1 → C3", "C2 -> C3": "C2 → C3", "C1+C2 -> C3": "C1+C2 → C3"}
GROUPS = [("A2", "Re-entry (A2)", [5.6, 4.6, 3.6]), ("A1", "Retention (A1)", [2.0, 1.0, 0.0])]

fig, ax = plt.subplots(figsize=(WIDTH_CM * CM, 5.6 * CM))
ax.axvline(0, color="0.35", lw=0.6, zorder=1)
ax.grid(axis="x", color="0.9", lw=0.5, zorder=0)
ticks, labels, trace = [], [], []
for task, head, ys in GROUPS:
    for y, des in zip(ys, DESIGNS):
        tot, yx = D[(task, des, "total")], D[(task, des, "YgivenX_shift")]
        xs = D[(task, des, "X_shift_source_side")][0] + D[(task, des, "X_shift_target_side")][0]
        assert abs(xs + yx[0] - tot[0]) < 1e-9, ("components do not add up", task, des)
        ax.plot([tot[1], tot[2]], [y + 0.24] * 2, color="0.5", lw=0.8, solid_capstyle="butt", zorder=2)
        ax.plot(tot[0], y + 0.24, "D", ms=3.6, mfc="white", mec="black", mew=0.7, zorder=4)
        ax.plot([yx[1], yx[2]], [y] * 2, color="black", lw=1.0, solid_capstyle="butt", zorder=2)
        ax.plot(yx[0], y, "o", ms=3.8, mfc="black", mec="black", zorder=4)
        ax.plot(xs, y - 0.24, "s", ms=3.3, mfc="0.7", mec="0.25", mew=0.6, zorder=4)
        ticks.append(y)
        labels.append(LAB[des])
        trace.append((task, des, tot, yx, xs))
    ax.text(-0.268, ys[0] + 0.62, head, fontsize=7, fontweight="bold", ha="left", va="bottom")
ax.set_yticks(ticks)
ax.set_yticklabels(labels)
ax.set_xlim(-0.27, 0.06)
ax.set_ylim(-0.55, 6.75)
ax.set_xticks([-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05])
ax.set_xticklabels(["−0.25", "−0.20", "−0.15", "−0.10", "−0.05", "0", "0.05"])
ax.set_xlabel("Change in employment prevalence, source → C3")
ax.tick_params(axis="y", length=0, pad=2)
hand = [Line2D([], [], marker="D", ls="-", color="0.5", lw=0.8, mfc="white", mec="black", mew=0.7, ms=3.6, label="Total change"),
        Line2D([], [], marker="o", ls="-", color="black", lw=1.0, mfc="black", mec="black", ms=3.8, label="Y|X shift"),
        Line2D([], [], marker="s", ls="", mfc="0.7", mec="0.25", mew=0.6, ms=3.3, label="X shift (both sides)")]
ax.legend(handles=hand, loc="lower left", frameon=False, fontsize=6.4, handlelength=1.6, handletextpad=0.4,
          borderaxespad=0.15, labelspacing=0.3)
fig.tight_layout(pad=0.25)
fig.savefig(OUT, dpi=DPI)
plt.close(fig)
for task, des, tot, yx, xs in trace:
    print("%s %-12s total %.3f (%.3f, %.3f)  Y|X %.3f (%.3f, %.3f)  X %.3f" % (task, des, *tot, *yx, xs))
print("main Fig. 1 written:", OUT)
