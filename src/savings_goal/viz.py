"""Plot styling shared by the notebooks and ``project/figures/make_figures.py``.

Colours come from one validated categorical order (blue, orange, aqua,
yellow), checked for colour-vision-deficiency separation on the light
surface. Series colours are assigned by entity in that fixed order, never by
rank. Magnitude uses a one-hue blue ramp; signed quantities use a blue/red
diverging map with a grey midpoint. Marks are thin, gridlines are solid
hairlines, and text is always ink, never a series colour.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib as mpl
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CONTEXT = "#cfcec7"  # de-emphasised marks
MIDPOINT = "#f0efec"  # diverging neutral

BLUE = "#2a78d6"  # slot 1
ORANGE = "#eb6834"  # slot 2
AQUA = "#1baf7a"  # slot 3
YELLOW = "#eda100"  # slot 4
RED = "#e34948"  # diverging pole opposite blue

CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
RED_RAMP = ["#fbd9d8", "#f3a9a8", "#ea7776", "#e34948", "#b8302f"]

SEQUENTIAL = LinearSegmentedColormap.from_list("sgc_blue", BLUE_RAMP)
DIVERGING = LinearSegmentedColormap.from_list(
    "sgc_blue_red", [*reversed(BLUE_RAMP[1:6]), MIDPOINT, *RED_RAMP[1:]]
)

COL = 3.45  # IEEE single-column width, inches
WIDE = 7.16  # IEEE two-column width, inches


def use(report: bool = False) -> None:
    """Apply the shared rcParams; ``report=True`` sizes text for an IEEE column."""
    size = 7.0 if report else 10.0
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "DejaVu Sans",
            "font.size": size,
            "axes.labelsize": size,
            "axes.titlesize": size + 0.5,
            "axes.titleweight": "normal",
            "axes.titlelocation": "left",
            "xtick.labelsize": size - 0.5,
            "ytick.labelsize": size - 0.5,
            "legend.fontsize": size - 0.5,
            "legend.frameon": False,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.prop_cycle": cycler(color=CATEGORICAL),
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "lines.markersize": 5.0,
            "patch.edgecolor": SURFACE,
            "figure.dpi": 400 if report else 110,
            "savefig.dpi": 400 if report else 150,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02 if report else 0.1,
            "image.cmap": "sgc_blue",
        }
    )


mpl.colormaps.register(SEQUENTIAL, force=True)
mpl.colormaps.register(DIVERGING, force=True)


def strip(ax: Axes, grid_axis: str | None = "x") -> None:
    """Hairline grid on one axis only; no top/right spines."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.5, zorder=0)  # type: ignore[arg-type]
        ax.set_axisbelow(True)
    ax.tick_params(length=2, width=0.5, colors=MUTED, labelcolor=INK_2)


def _units_per_point(ax: Axes) -> tuple[float, float]:
    fig = ax.get_figure(root=True)
    assert fig is not None
    box = ax.get_position()
    w_in, h_in = fig.get_size_inches()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    return (x1 - x0) / (w_in * box.width * 72.0), (y1 - y0) / (h_in * box.height * 72.0)


def rounded_barh(
    ax: Axes,
    y: float,
    x0: float,
    x1: float,
    height: float,
    color: str,
    radius_pt: float = 2.0,
    zorder: int = 3,
) -> None:
    """Horizontal bar, data end rounded, baseline end square. Call after limits are set."""
    ux, uy = _units_per_point(ax)
    sign = 1.0 if x1 >= x0 else -1.0
    span = abs(x1 - x0)
    rx = min(radius_pt * ux, span / 2 if span > 0 else 0.0)
    ry = min(radius_pt * uy, height / 2)
    y0, y1 = y - height / 2, y + height / 2
    xe = x1 - sign * rx
    verts = [
        (x0, y0),
        (xe, y0),
        (x1, y0),
        (x1, y0 + ry),
        (x1, y1 - ry),
        (x1, y1),
        (xe, y1),
        (x0, y1),
        (x0, y0),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", zorder=zorder))


def barh(
    ax: Axes,
    labels: Sequence[str],
    values: Sequence[float],
    colors: Sequence[str] | str = BLUE,
    height: float = 0.55,
    value_fmt: str | None = "{:.3f}",
    xlim: tuple[float, float] | None = None,
    base: float = 0.0,
) -> None:
    """Rounded horizontal bars, first label at the top, values at the tips in ink."""
    n = len(values)
    cols = [colors] * n if isinstance(colors, str) else list(colors)
    lo = min(base, *values)
    hi = max(base, *values)
    pad = (hi - lo) * 0.12 or 1.0
    ax.set_xlim(xlim or (lo - (pad if lo < base else 0), hi + pad))
    ax.set_ylim(-0.6, n - 0.4)
    ys = list(range(n))[::-1]
    ax.set_yticks(ys, list(labels))
    strip(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    off = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.012
    for y, v, c in zip(ys, values, cols, strict=True):
        rounded_barh(ax, y, base, v, height, c)
        if value_fmt:
            ax.text(
                v + (off if v >= base else -off),
                y,
                value_fmt.format(v),
                va="center",
                ha="left" if v >= base else "right",
                color=INK_2,
                fontsize=mpl.rcParams["ytick.labelsize"],
            )


def end_label(ax: Axes, x: float, y: float, text: str, dx: float = 0.0, dy: float = 0.0) -> None:
    """Direct label at a line end, in secondary ink."""
    ax.annotate(
        text,
        (x, y),
        xytext=(4 + dx, dy),
        textcoords="offset points",
        va="center",
        color=INK_2,
        fontsize=mpl.rcParams["legend.fontsize"],
    )


def legend(ax: Axes, **kwargs: Any) -> None:
    leg = ax.legend(frameon=False, handlelength=1.4, borderpad=0.2, labelspacing=0.3, **kwargs)
    for t in leg.get_texts():
        t.set_color(INK_2)


def dot(ax: Axes, x: Any, y: Any, color: str, size: float = 5.0, **kwargs: Any) -> None:
    """Markers carry a surface-coloured ring so they stay legible on crossings."""
    ax.plot(
        x,
        y,
        "o",
        color=color,
        markersize=size,
        markeredgecolor=SURFACE,
        markeredgewidth=1.0,
        **kwargs,
    )
