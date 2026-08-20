"""Build the four figures used in project/report.tex.

Every number here comes from an artifact committed to this repository:

  * results/model_comparison.csv, results/baseline.csv, results/shap_importance.csv
    and results/persona_profiles.csv are read directly;
  * the remaining tables (targeting capture, accuracy by income decile, peer
    excess by category, within-decile persona attainment, SHAP attribution by
    feature group) are transcribed from the walkthrough documents, which are the
    published output of the notebooks. Each block cites its source.

The IHDS-II microdata cannot be redistributed (ICPSR terms), so this script is
deliberately independent of dataset/: re-running the notebooks regenerates the
source tables, and re-running this script regenerates the figures from them.

    python project/figures/make_figures.py
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from matplotlib.patches import Patch, PathPatch

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent

# --- Design tokens ---------------------------------------------------------
# Chart surface, ink and hairline chrome; categorical slots in fixed order.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CONTEXT = "#cfcec7"          # de-emphasised bars

BLUE = "#2a78d6"             # slot 1
ORANGE = "#eb6834"           # slot 2
AQUA = "#1baf7a"             # slot 3
YELLOW = "#eda100"           # slot 4
MAGENTA = "#e87ba4"          # slot 5
RED = "#e34948"              # diverging pole opposite BLUE

COL = 3.45                   # IEEE single-column width, inches
WIDE = 7.16                  # IEEE two-column width, inches

mpl.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans",
    "font.size": 7.0,
    "axes.labelsize": 7.0,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "figure.dpi": 400,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def strip(ax, grid_axis="x"):
    """Hairline grid on one axis only; no top/right spines."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
    ax.tick_params(length=2, width=0.5, colors=MUTED, labelcolor=INK_2)


def _units_per_point(ax):
    """Data units per typographic point, separately in x and y."""
    fig = ax.figure
    box = ax.get_position()
    w_pt = fig.get_size_inches()[0] * box.width * 72.0
    h_pt = fig.get_size_inches()[1] * box.height * 72.0
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    return (x1 - x0) / w_pt, (y1 - y0) / h_pt


def rounded_barh(ax, y, x0, x1, height, color, radius_pt=2.0, zorder=3):
    """Horizontal bar with the data end rounded and the baseline end square.

    The corner radius is given in points, so it stays a constant physical size
    however the two axes are scaled. Call after the axis limits are set.
    """
    ux, uy = _units_per_point(ax)
    sign = 1.0 if x1 >= x0 else -1.0
    span = abs(x1 - x0)
    rx = min(radius_pt * ux, span / 2 if span > 0 else 0.0)
    ry = min(radius_pt * uy, height / 2)
    y0, y1 = y - height / 2, y + height / 2
    xe = x1 - sign * rx
    verts = [
        (x0, y0), (xe, y0),
        (x1, y0), (x1, y0 + ry),         # quadratic corner
        (x1, y1 - ry),
        (x1, y1), (xe, y1),              # quadratic corner
        (x0, y1), (x0, y0),
    ]
    codes = [
        MplPath.MOVETO, MplPath.LINETO,
        MplPath.CURVE3, MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CURVE3, MplPath.CURVE3,
        MplPath.LINETO, MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color,
                           edgecolor="none", zorder=zorder))


# ---------------------------------------------------------------------------
# Figure 1 — what each ingredient of the pipeline is actually worth
# ---------------------------------------------------------------------------
def fig_margin():
    base = pd.read_csv(RESULTS / "baseline.csv").set_index("model")["F1 (macro)"]
    comp = pd.read_csv(RESULTS / "model_comparison.csv").set_index("model")["CV F1 (macro)"]

    rungs = [
        ("Majority class", base["Majority class (always 'not met')"], CONTEXT),
        ("Income only (LR)", base["income only"], CONTEXT),
        ("+ Groceries_Share", base["income + groceries share"], CONTEXT),
        ("All 28 features (LR)", base["all 28 engineered features"], CONTEXT),
        ("XGBoost (tuned)", comp["XGBoost"], BLUE),
    ]
    income_rule = 0.7425          # depth-1 tree on income, walkthrough/phase3.md

    fig, ax = plt.subplots(figsize=(COL, 1.85))
    ys = np.arange(len(rungs))[::-1]
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-1.5, len(rungs) + 0.05)
    ax.set_yticks(ys, [r[0] for r in rungs], fontsize=6.6)
    ax.set_xlabel("Macro-F1")
    strip(ax)
    ax.spines["left"].set_visible(False)

    for y, (label, val, colour) in zip(ys, rungs):
        rounded_barh(ax, y, 0, val, 0.5, colour)
        ax.text(val + 0.014, y, f"{val:.3f}", va="center", ha="left",
                fontsize=6.4, color=INK if colour == BLUE else INK_2)

    # the honest reference: one threshold on income alone
    ax.axvline(income_rule, color=ORANGE, linewidth=1.0, zorder=4,
               ymin=0.0, ymax=0.93)
    ax.text(income_rule - 0.02, len(rungs) - 0.28,
            "single income threshold  0.743", ha="right", va="center",
            fontsize=6.3, color=ORANGE)

    # the increment that everything past the income rule buys
    ax.annotate("", xy=(rungs[-1][1], -0.78), xytext=(income_rule, -0.78),
                arrowprops=dict(arrowstyle="<->", color=INK_2, linewidth=0.7,
                                shrinkA=0, shrinkB=0))
    ax.text(rungs[-1][1] + 0.02, -0.78, "+0.095", ha="left", va="center",
            fontsize=6.4, color=INK_2)
    ax.tick_params(axis="y", length=0)
    fig.savefig(OUT / "fig1_honest_margin.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — SHAP: what the model actually uses
# ---------------------------------------------------------------------------
def fig_shap():
    imp = pd.read_csv(RESULTS / "shap_importance.csv", index_col=0)["mean_abs_shap"]
    top = imp.sort_values(ascending=False).head(10)

    # Attribution by feature group, walkthrough/phase5.md Cell 6.
    # Debt (3.3), education (3.2) and participation indicators (4.0) are folded
    # into "Other" so the bar stays readable at part-to-whole glance.
    groups = [
        ("Income", 38.4, BLUE),
        ("Spending mix", 26.7, ORANGE),
        ("Social/geo", 13.3, AQUA),
        ("Household", 11.2, YELLOW),
        ("Other", 10.5, MAGENTA),
    ]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(COL, 2.6), gridspec_kw=dict(height_ratios=[7.4, 1.0], hspace=0.5))

    ys = np.arange(len(top))[::-1]
    ax1.set_xlim(0, 3.25)
    ax1.set_ylim(-0.7, len(top) - 0.3)
    ax1.set_yticks(ys, [n.replace("_", " ") for n in top.index], fontsize=6.3)
    ax1.set_xlabel("Mean |SHAP| (log-odds)")
    strip(ax1)
    ax1.spines["left"].set_visible(False)
    for y, (name, val) in zip(ys, top.items()):
        colour = BLUE if name == "Log_Income" else CONTEXT
        rounded_barh(ax1, y, 0, val, 0.52, colour)
        ax1.text(val + 0.05, y, f"{val:.2f}", va="center", ha="left",
                 fontsize=6.1, color=INK if name == "Log_Income" else INK_2)
    ax1.tick_params(axis="y", length=0)
    ax1.text(1.12, 7.85, "4.5$\\times$ the next feature", fontsize=6.2, color=INK_2)

    ax2.set_xlim(0, 100)
    ax2.set_ylim(-0.75, 0.75)
    ax2.axis("off")
    left = 0.0
    for name, share, colour in groups:
        rounded_barh(ax2, 0, left, left + share - 0.6, 1.0, colour, radius_pt=1.5)
        r, g, b = (int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        ax2.text(left + (share - 0.6) / 2, 0, f"{share:.1f}%", ha="center",
                 va="center", fontsize=5.6, zorder=5,
                 color="#ffffff" if lum < 0.5 else INK)
        left += share
    handles = [Patch(facecolor=c, edgecolor="none", label=n) for n, _, c in groups]
    leg = ax2.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.35),
                     ncol=3, frameon=False, fontsize=5.8, handlelength=0.9,
                     handleheight=0.9, handletextpad=0.4, columnspacing=1.0,
                     labelspacing=0.4, borderpad=0.0)
    for text in leg.get_texts():
        text.set_color(INK_2)

    fig.savefig(OUT / "fig2_shap.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — what the model is worth to a campaign
# ---------------------------------------------------------------------------
def fig_business():
    # Targeting capture, walkthrough/phase7.md Cell 3.
    budgets = np.array([0, 10, 25, 50, 100])
    model = np.array([0, 14.7, 36.6, 70.8, 100])
    income = np.array([0, 14.4, 35.0, 65.8, 100])
    random = np.array([0, 10.0, 25.1, 49.8, 100])

    # Accuracy by income decile, walkthrough/phase7.md Cells 9-10 (published deciles).
    dec = np.array([0, 3, 5, 6, 7, 9])
    acc = np.array([0.979, 0.870, 0.799, 0.788, 0.779, 0.887])

    # Median excess budget share vs same-income on-track peers (pp),
    # walkthrough/phase7.md Cells 6-7.
    cats = ["Healthcare", "Education", "Transport", "Miscellaneous",
            "Utilities", "Groceries"]
    excess = [2.53, 1.41, 1.33, -0.21, -2.77, -8.79]

    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 1.62))
    ax_a, ax_b, ax_c = axes
    fig.subplots_adjust(wspace=0.42)

    # (a) capture curve
    ax_a.plot(budgets, random, color=MUTED, linewidth=1.0, zorder=2)
    ax_a.plot(budgets, income, color=ORANGE, linewidth=1.4, zorder=3)
    ax_a.plot(budgets, model, color=BLUE, linewidth=1.6, zorder=4)
    for x, y, c in ((25, 36.6, BLUE), (25, 35.0, ORANGE), (25, 25.1, MUTED)):
        ax_a.plot([x], [y], "o", color=c, markersize=3.2,
                  markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=5)
    ax_a.text(52, 74.5, "Model", color=BLUE, fontsize=6.4, fontweight="bold")
    ax_a.text(52, 60.0, "Income rule", color=ORANGE, fontsize=6.4)
    ax_a.text(52, 43.5, "Random", color=MUTED, fontsize=6.4)
    ax_a.annotate("36.6% vs 35.0%\nat a 25% budget", xy=(25, 36.0), xytext=(27, 12),
                  fontsize=6.2, color=INK_2,
                  arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=0.6))
    ax_a.set_xlim(0, 100)
    ax_a.set_ylim(0, 100)
    ax_a.set_xlabel("Households contacted (%)")
    ax_a.set_ylabel("At-risk households reached (%)")
    ax_a.set_title("(a) Barely better than an income rule",
                   loc="left", fontsize=6.8, color=INK, pad=6)
    strip(ax_a, grid_axis="both")

    # (b) accuracy by income decile
    ax_b.axvspan(4.6, 7.4, color="#f0efec", zorder=1)
    ax_b.plot(dec, acc, color=BLUE, linewidth=1.5, marker="o", markersize=3.4,
              markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=4)
    ax_b.axhline(0.861, color=MUTED, linewidth=0.8, zorder=2)
    ax_b.text(0.0, 0.856, "overall 0.861", ha="left", va="top",
              fontsize=6.2, color=MUTED)
    ax_b.text(6.0, 0.744, "targeting is\ncontested here", ha="center", va="bottom",
              fontsize=6.2, color=INK_2, linespacing=1.25)
    ax_b.set_xlim(-0.5, 9.5)
    ax_b.set_ylim(0.725, 1.02)
    ax_b.set_xticks([0, 3, 5, 6, 7, 9])
    ax_b.set_xlabel("Income decile")
    ax_b.set_ylabel("Accuracy")
    ax_b.set_title("(b) Weakest where it matters",
                   loc="left", fontsize=6.8, color=INK, pad=6)
    strip(ax_b, grid_axis="y")

    # (c) peer excess, diverging
    ys = np.arange(len(cats))[::-1]
    ax_c.set_xlim(-11.5, 5.4)
    ax_c.set_ylim(-1.0, len(cats) - 0.3)
    ax_c.set_yticks(ys, cats, fontsize=6.4)
    ax_c.set_xlabel("Budget share vs same-income peers (pp)")
    ax_c.set_title("(c) Food runs the wrong way",
                   loc="left", fontsize=6.8, color=INK, pad=6)
    strip(ax_c)
    ax_c.spines["left"].set_visible(False)
    ax_c.tick_params(axis="y", length=0)
    for y, (name, val) in zip(ys, zip(cats, excess)):
        rounded_barh(ax_c, y, 0, val, 0.48, RED if val > 0 else BLUE)
        right = val > 0 or abs(val) < 0.5   # keep tiny bars clear of the tick label
        ax_c.text(val + (0.3 if right else -0.3), y, f"{val:+.2f}", va="center",
                  ha="left" if right else "right", fontsize=6.1, color=INK_2)
    ax_c.axvline(0, color=AXIS, linewidth=0.8, zorder=2)
    ax_c.text(-11.2, -0.9, "spends less", fontsize=6.0, color=BLUE, ha="left")
    ax_c.text(5.1, -0.9, "spends more", fontsize=6.0, color=RED, ha="right")

    fig.savefig(OUT / "fig3_business.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — the persona effect income was hiding
# ---------------------------------------------------------------------------
def fig_personas():
    prof = pd.read_csv(RESULTS / "persona_profiles.csv").set_index("Persona")

    # Goal attainment by persona within income decile, walkthrough/phase6.md
    # Cell 12 (published deciles).
    dec = np.array([0, 3, 5, 7, 9])
    series = {
        0: ([0.051, 0.261, 0.472, 0.789, 0.911], BLUE, "P0 · no transport spend"),
        1: ([0.009, 0.105, 0.249, 0.466, 0.835], ORANGE, "P1 · spends on everything"),
        2: ([0.030, 0.231, 0.368, 0.569, 0.811], AQUA, "P2 · no healthcare spend"),
    }

    fig, ax = plt.subplots(figsize=(COL, 1.85))
    for pid, (vals, colour, label) in series.items():
        share = prof.loc[pid, "pct_of_sample"]
        ax.plot(dec, vals, color=colour, linewidth=1.5, marker="o", markersize=3.4,
                markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=4,
                label=f"{label} ({share:.1f}% of sample)")

    # the gap income was hiding
    ax.annotate("", xy=(7, 0.789), xytext=(7, 0.466),
                arrowprops=dict(arrowstyle="<->", color=INK_2, linewidth=0.7,
                                shrinkA=1.5, shrinkB=1.5))
    ax.text(6.62, 0.63, "32 pp", ha="right", va="center", fontsize=6.4, color=INK_2)
    ax.text(9.3, 0.02, "Unconditional spread across personas: 9.2 pp",
            fontsize=6.2, color=MUTED, ha="right", va="bottom")

    ax.set_xticks(dec)
    ax.set_xlim(-0.4, 9.4)
    ax.set_ylim(0, 1.06)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Income decile")
    ax.set_ylabel("Share meeting the 20% benchmark")
    leg = ax.legend(loc="upper left", frameon=False, handlelength=1.4,
                    borderpad=0.2, labelspacing=0.3, fontsize=6.2)
    for text in leg.get_texts():
        text.set_color(INK_2)
    strip(ax, grid_axis="y")
    fig.savefig(OUT / "fig4_personas.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_margin()
    fig_shap()
    fig_business()
    fig_personas()
    print("wrote:", *(p.name for p in sorted(OUT.glob("fig*.png"))))
