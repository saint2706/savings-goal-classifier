"""Build the four figures used in project/report.tex.

Every number is read from an aggregate the notebooks write to results/ (nothing
is transcribed by hand):

  * results/baseline.csv, model_comparison.csv, shap_importance.csv,
    persona_profiles.csv;
  * results/leakage.json (NB01), explain.json (NB05), personas.json (NB06) and
    business.json (NB07).

The IHDS-II microdata cannot be redistributed (ICPSR terms), so this script is
deliberately independent of dataset/: re-running the notebooks regenerates the
source files, and re-running this script regenerates the figures from them.

    uv run python project/figures/make_figures.py
"""

import json
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


def result(name):
    return json.loads((RESULTS / f"{name}.json").read_text())

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
# Figure 1 — what each ingredient is worth, and what the oracle gets for free
# ---------------------------------------------------------------------------
def fig_margin():
    base = pd.read_csv(RESULTS / "baseline.csv").set_index("model")["ROC-AUC"]
    comp = pd.read_csv(RESULTS / "model_comparison.csv")
    auc = comp.set_index(["feature set", "model"])["ROC-AUC"]
    oracle = result("leakage")["t3"]["auc"]

    rungs = [
        ("Single income threshold", base["Single income threshold (depth-1 tree)"], CONTEXT),
        ("Income only (LR)", base["income only"], CONTEXT),
        ("Income + demographics (LR)", base["deployable (income, demographics, debt)"], CONTEXT),
        ("Headline: XGBoost, deployable", auc[("deployable", "XGBoost (tuned)")], BLUE),
        ("XGBoost + spending shares", auc[("full", "XGBoost (tuned)")], ORANGE),
    ]

    fig, ax = plt.subplots(figsize=(COL, 1.85))
    ys = np.arange(len(rungs))[::-1]
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(-0.6, len(rungs) + 0.5)
    ax.set_yticks(ys, [r[0] for r in rungs], fontsize=6.4)
    ax.set_xlabel("ROC-AUC (PSU-grouped 5-fold CV, training split)")
    strip(ax)
    ax.spines["left"].set_visible(False)
    for y, (label, val, colour) in zip(ys, rungs):
        rounded_barh(ax, y, 0.5, val, 0.5, colour)
        ax.text(val + 0.006, y, f"{val:.3f}", va="center", ha="left", zorder=6,
                fontsize=6.3, color=INK if colour != CONTEXT else INK_2,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.4))
    ax.axvline(oracle, color=RED, linewidth=1.0, zorder=4)
    ax.text(oracle - 0.005, len(rungs) - 0.05, f"model-free oracle {oracle:.3f}\n(size × food share × income)",
            ha="right", va="center", fontsize=5.8, color=RED)
    ax.tick_params(axis="y", length=0)
    fig.savefig(OUT / "fig1_honest_margin.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — SHAP for the headline model
# ---------------------------------------------------------------------------
FAMILY_COLOURS = {
    "Income": BLUE, "Household size family": ORANGE, "Social/geo (categoricals)": AQUA,
    "Head age & education": YELLOW, "Debt": MAGENTA,
}
FAMILY_LABELS = {
    "Income": "Income", "Household size family": "Household size", "Social/geo (categoricals)": "Social/geo",
    "Head age & education": "Head age & education", "Debt": "Debt",
}


def fig_shap():
    imp = pd.read_csv(RESULTS / "shap_importance.csv")
    top = (imp[imp["model"] == "deployable"].set_index("feature")["mean_abs_shap"]
           .sort_values(ascending=False).head(10))
    fam = pd.DataFrame(result("explain")["deployable"]["families"]).set_index("family")
    shares = (fam["share of grouped total"] * 100).sort_values(ascending=False)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(COL, 2.6), gridspec_kw=dict(height_ratios=[7.4, 1.0], hspace=0.5))
    ys = np.arange(len(top))[::-1]
    ax1.set_xlim(0, top.max() * 1.25)
    ax1.set_ylim(-0.7, len(top) - 0.3)
    ax1.set_yticks(ys, [n.replace("_", " ") for n in top.index], fontsize=6.0)
    ax1.set_xlabel("Mean |SHAP| (log-odds), held-out PSUs")
    strip(ax1)
    ax1.spines["left"].set_visible(False)
    for y, (name, val) in zip(ys, top.items()):
        colour = BLUE if name == "Log_Income" else CONTEXT
        rounded_barh(ax1, y, 0, val, 0.52, colour)
        ax1.text(val + top.max() * 0.015, y, f"{val:.2f}", va="center", ha="left",
                 fontsize=6.0, color=INK if name == "Log_Income" else INK_2)
    ax1.tick_params(axis="y", length=0)

    ax2.set_xlim(0, 100)
    ax2.set_ylim(-0.75, 0.75)
    ax2.axis("off")
    left = 0.0
    for name, share in shares.items():
        colour = FAMILY_COLOURS.get(name, CONTEXT)
        rounded_barh(ax2, 0, left, left + share - 0.6, 1.0, colour, radius_pt=1.5)
        r, g, b = (int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        if share > 6:
            ax2.text(left + (share - 0.6) / 2, 0, f"{share:.1f}%", ha="center", va="center",
                     fontsize=5.6, zorder=5, color="#ffffff" if lum < 0.5 else INK)
        left += share
    handles = [Patch(facecolor=FAMILY_COLOURS.get(n, CONTEXT), edgecolor="none",
                     label=f"{FAMILY_LABELS.get(n, n)} {s:.1f}%") for n, s in shares.items()]
    leg = ax2.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.35),
                     ncol=3, frameon=False, fontsize=5.6, handlelength=0.9,
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
    bus = result("business")
    curves = bus["capture_curve"]
    frac = np.linspace(0, 100, len(next(iter(curves.values()))))
    prevalence = bus["at_risk_rate"]
    cap = pd.DataFrame(bus["capture"])
    c25 = cap[cap["budget"] == 0.25].set_index("strategy")
    H, INC = "Model (headline, deployable)", "Income rule (poorest first)"
    dec = pd.DataFrame(bus["deciles"])
    bench = pd.DataFrame(bus["benchmarks"]).set_index("category")

    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 1.75))
    ax_a, ax_b, ax_c = axes
    fig.subplots_adjust(wspace=0.55)

    # (a) capture curve against its ceiling
    ax_a.plot(frac, np.minimum(100, frac / prevalence), color=INK_2, linewidth=0.7,
              linestyle="--", zorder=2)
    ax_a.plot(frac, np.array(curves["Random contact"]) * 100, color=MUTED, linewidth=1.0, zorder=2,
              label="Random")
    ax_a.plot(frac, np.array(curves[INC]) * 100, color=ORANGE, linewidth=1.4, zorder=3, label="Income rule")
    ax_a.plot(frac, np.array(curves[H]) * 100, color=BLUE, linewidth=1.6, zorder=4, label="Model")
    ax_a.text(40, 93, "ceiling", color=INK_2, fontsize=6.0, ha="right")
    leg = ax_a.legend(loc="lower right", frameon=False, fontsize=5.8, handlelength=1.2, borderpad=0.1,
                      labelspacing=0.25)
    for text in leg.get_texts():
        text.set_color(INK_2)
    ax_a.annotate(f"{c25.loc[H, 'pct_of_ceiling']:.1%} vs {c25.loc[INC, 'pct_of_ceiling']:.1%}\n"
                  "of ceiling at 25%", xy=(25, c25.loc[H, "capture"] * 100), xytext=(2, 70),
                  fontsize=6.0, color=INK_2,
                  arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=0.6))
    ax_a.set_xlim(0, 100)
    ax_a.set_ylim(0, 102)
    ax_a.set_xlabel("Households contacted (%)")
    ax_a.set_ylabel("At-risk reached (%)")
    ax_a.set_title("(a) Near the ceiling, like the income rule", loc="left", fontsize=6.8, pad=6)
    strip(ax_a, grid_axis="both")

    # (b) accuracy by income decile, all ten
    overall = bus["overall_accuracy_headline"]
    ax_b.plot(dec["Income_Decile"], dec["accuracy (headline)"], color=BLUE, linewidth=1.5,
              marker="o", markersize=3.0, markeredgecolor=SURFACE, markeredgewidth=0.7, zorder=4)
    ax_b.axhline(overall, color=MUTED, linewidth=0.8, zorder=2)
    ax_b.text(-0.3, overall - 0.008, f"overall {overall:.3f}", ha="left", va="top",
              fontsize=6.0, color=MUTED)
    worst = dec.loc[dec["accuracy (headline)"].idxmin()]
    ax_b.annotate(f"decile {int(worst['Income_Decile'])}: {worst['accuracy (headline)']:.3f}",
                  xy=(worst["Income_Decile"], worst["accuracy (headline)"]),
                  xytext=(worst["Income_Decile"] - 3.5, worst["accuracy (headline)"] - 0.06),
                  fontsize=6.0, color=INK_2, arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=0.6))
    lo = dec["accuracy (headline)"].min()
    ax_b.set_xlim(-0.5, 9.5)
    ax_b.set_ylim(lo - 0.09, 1.01)
    ax_b.set_xticks(range(10))
    ax_b.set_xlabel("Income decile")
    ax_b.set_ylabel("Accuracy (out-of-fold)")
    ax_b.set_title("(b) Weakest in the middle deciles", loc="left", fontsize=6.8, pad=6)
    strip(ax_b, grid_axis="y")

    # (c) signed composition excess (pp of budget), with 95% PSU-bootstrap intervals
    b = bench.sort_values("mean excess share (pp)")
    ys = np.arange(len(b))
    x = b["mean excess share (pp)"]
    colours = [RED if lo_ > 0 else BLUE if hi_ < 0 else CONTEXT
               for lo_, hi_ in zip(b["share CI low"], b["share CI high"])]
    ax_c.barh(ys, x, height=0.55, color=colours, zorder=3)
    ax_c.errorbar(x, ys, xerr=[x - b["share CI low"], b["share CI high"] - x],
                  fmt="none", ecolor=INK_2, elinewidth=0.6, capsize=1.2, zorder=4)
    ax_c.axvline(0, color=AXIS, linewidth=0.8, zorder=2)
    ax_c.set_yticks(ys, [c.replace("_", " ") for c in b.index], fontsize=5.6)
    ax_c.set_xlabel("Budget share vs same-income peers (pp)")
    ax_c.set_title("(c) Signed composition excess, 95% CI", loc="left", fontsize=6.8, pad=6)
    strip(ax_c)
    ax_c.spines["left"].set_visible(False)
    ax_c.tick_params(axis="y", length=0)

    fig.savefig(OUT / "fig3_business.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — cluster attainment within income decile
# ---------------------------------------------------------------------------
def fig_personas():
    per = result("personas")
    prof = pd.read_csv(RESULTS / "persona_profiles.csv").set_index("Persona")
    within = pd.DataFrame(per["within_decile"]).set_index("Income_Decile")
    colours = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, RED]
    inter = per["interaction"]

    fig, ax = plt.subplots(figsize=(COL, 1.85))
    for i, col in enumerate(within.columns):
        share = prof.loc[int(col), "pct_of_sample"]
        ax.plot(within.index, within[col], color=colours[i % len(colours)], linewidth=1.4,
                marker="o", markersize=2.8, markeredgecolor=SURFACE, markeredgewidth=0.6, zorder=4,
                label=f"C{col} · {per['labels'][col]} ({share:.1f}%)")
    ax.text(-0.2, 0.68, f"Cluster | decile: LR p = {inter['p persona | decile']:.1g}\n"
            f"pseudo-R² {inter['pseudo R2 decile only']:.3f} → {inter['pseudo R2 + persona']:.3f}",
            fontsize=5.6, color=MUTED, ha="left", va="top")
    ax.set_xticks(range(10))
    ax.set_xlim(-0.4, 9.4)
    ax.set_ylim(0, 1.06)
    ax.set_xlabel("Income decile")
    ax.set_ylabel("Share meeting the 20% benchmark")
    leg = ax.legend(loc="upper left", frameon=False, handlelength=1.4, borderpad=0.2,
                    labelspacing=0.3, fontsize=5.6)
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
