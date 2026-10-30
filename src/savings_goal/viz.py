"""Swiss Editorial plot system shared by the notebooks and ``project/figures/make_figures.py``.

The look is a warm off-white surface, near-black ink, a ladder of neutral greys and
exactly one vermilion accent. Colour is reserved for meaning: the accent marks the
primary model, benchmark or finding, everything else recedes into grey. Where several
series genuinely need telling apart the neutral palette is backed by line style and
marker so no chart depends on hue alone. Type is a hybrid: a serif display title, a
sans for headings, labels and every number (all candidate sans faces have tabular
digits), and small muted sans for notes. No font is required; DejaVu, bundled with
Matplotlib, is the last fallback.

The module is layered, top to bottom:

* **tokens** - colour, type and size constants, ``PRESETS``, colormaps;
* **style** - ``use()`` / ``context()`` apply rcParams, ``save()`` exports;
* **primitives** - ``strip``, ``barh``, ``dot``, ``legend``, ``refline``, ``ci_bars``,
  ``title``, ``source``, ``kpi``, ``callout``, ``direct_labels``, ``axis_format``;
* **chart families** - ROC/PR/confusion/calibration, SHAP, clustering, regression
  diagnostics and business curves, all drawing onto an ``Axes`` you own.

Every helper takes the target ``Axes`` first and leaves figure creation, layout and
limits to the caller, so they compose with plain Matplotlib. The pre-redesign API
(``use``, ``strip``, ``rounded_barh``, ``barh``, ``end_label``, ``legend``, ``dot`` and
the colour constants) is unchanged; the legacy palette names keep their slot but now
hold the editorial values (``BLUE`` is the primary ink, ``ORANGE`` the accent).
"""

from __future__ import annotations

import textwrap
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib as mpl
import numpy as np
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.backend_bases import RendererBase
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.text import Annotation
from matplotlib.ticker import FuncFormatter, PercentFormatter
from matplotlib.transforms import Bbox, blended_transform_factory
from numpy.typing import ArrayLike, NDArray

Floats = NDArray[np.float64]

# ---------------------------------------------------------------------------
# Colour tokens
# ---------------------------------------------------------------------------
SURFACE = "#F8F7F3"  # warm off-white page
INK = "#171717"  # titles, key numbers
INK_2 = "#484848"  # tick labels, annotations
MUTED = "#707070"  # notes, tick marks (4.6:1 on SURFACE, so still legible as text)
GRID = "#E5E3DE"  # hairline gridlines
AXIS = "#B9B7B1"  # spines, zero lines
CONTEXT = "#C8C6C0"  # de-emphasised marks
MIDPOINT = "#ECEAE4"  # diverging neutral

ACCENT = "#D94A3A"  # the one accent: primary model, benchmark, finding
ACCENT_DEEP = "#A82E22"  # accent as text or as the dark pole of a ramp

# Legacy palette slots. Names are kept for existing callers; values are editorial.
BLUE = "#2B2B2B"  # slot 1: primary ink series; negative pole of diverging maps
ORANGE = ACCENT  # slot 2: the accent
AQUA = "#7A7A7A"  # slot 3: mid grey
YELLOW = "#B3AB9C"  # slot 4: warm light grey
RED = ACCENT_DEEP  # positive pole opposite BLUE

CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW]
BLUE_RAMP = ["#DAD7CF", "#C2BFB7", "#A9A6A0", "#8F8C87", "#74726E", "#5A5855", "#3F3E3C", "#262625"]
RED_RAMP = ["#F6D3CD", "#EE9F94", "#E4715F", ACCENT, ACCENT_DEEP]

# Series that must be told apart without the accent (accent reserved for ``focus``).
NEUTRALS = [BLUE, AQUA, YELLOW, CONTEXT]

# Redundant encodings per categorical slot: colour, line style, marker.
_SERIES: tuple[dict[str, Any], ...] = (
    {"color": BLUE, "linestyle": "-", "marker": "o"},
    {"color": ORANGE, "linestyle": "-", "marker": "s"},
    {"color": AQUA, "linestyle": "--", "marker": "^"},
    {"color": YELLOW, "linestyle": ":", "marker": "D"},
)
_NEUTRAL_SERIES: tuple[dict[str, Any], ...] = (
    {"color": BLUE, "linestyle": "-", "marker": "o"},
    {"color": AQUA, "linestyle": "--", "marker": "s"},
    {"color": YELLOW, "linestyle": "-.", "marker": "^"},
    {"color": CONTEXT, "linestyle": ":", "marker": "D"},
)

SEQUENTIAL = LinearSegmentedColormap.from_list("sgc_sequential", BLUE_RAMP)
ACCENT_SEQUENTIAL = LinearSegmentedColormap.from_list("sgc_accent", RED_RAMP)
# 3 + mid + 3 stops: the neutral colour sits at exactly 0.5.
DIVERGING = LinearSegmentedColormap.from_list(
    "sgc_diverging",
    [BLUE_RAMP[6], BLUE_RAMP[4], BLUE_RAMP[2], MIDPOINT, RED_RAMP[1], RED_RAMP[3], RED_RAMP[4]],
)
# Feature value low -> high on a SHAP beeswarm; deliberately not the SHAP-sign colours.
FEATURE_CMAP = LinearSegmentedColormap.from_list(
    "sgc_feature", [BLUE_RAMP[6], BLUE_RAMP[2], ACCENT]
)


def _register(cmap: Colormap, name: str | None = None) -> None:
    """Register ``cmap`` once; re-imports and reloads neither warn nor duplicate.

    An identical map already under the name is left alone. A different one (the palette
    was edited and the module reloaded) is replaced, silently: that overwrite is intended.
    """
    key = name or cmap.name
    if key in mpl.colormaps and mpl.colormaps[key] == cmap:
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mpl.colormaps.register(cmap, name=key, force=True)


for _cmap in (SEQUENTIAL, ACCENT_SEQUENTIAL, DIVERGING, FEATURE_CMAP):
    _register(_cmap)
# Names the pre-redesign module registered; strings in old notebooks keep resolving.
_register(SEQUENTIAL, "sgc_blue")
_register(DIVERGING, "sgc_blue_red")

# ---------------------------------------------------------------------------
# Typography and size tokens
# ---------------------------------------------------------------------------
# Serif faces with lining figures only (Georgia's old-style digits look wrong in titles
# that carry numbers). Sans faces all set digits at a fixed width, so columns of
# numbers align without OpenType features.
FONT_SERIF = (
    "Source Serif 4",
    "Source Serif Pro",
    "Charter",
    "Bitstream Charter",
    "Cambria",
    "Liberation Serif",
    "Times New Roman",
    "DejaVu Serif",
)
FONT_SANS = (
    "Helvetica Neue",
    "Helvetica",
    "Arial",
    "Segoe UI",
    "Liberation Sans",
    "DejaVu Sans",
)
_MINUS = chr(0x2212)  # typographic minus
_RUPEE = chr(0x20B9)
_SERIF = ["serif", "DejaVu Serif"]  # per-text family list: generic first, glyph fallback second

COL = 3.45  # IEEE single-column width, inches
WIDE = 7.16  # IEEE two-column width, inches
SLIDE = 10.0  # full-width presentation figure, inches
SIZES = {"col": COL, "wide": WIDE, "slide": SLIDE}


@dataclass(frozen=True)
class Preset:
    """Output-medium tokens: body size, line weights and raster settings."""

    name: str
    base: float  # body text, pt; every other size derives from it
    dpi: int  # on-screen
    export_dpi: int  # savefig
    pad: float  # savefig.pad_inches
    line: float  # data line width, pt
    marker: float  # marker size, pt


PRESETS: dict[str, Preset] = {
    "notebook": Preset("notebook", 10.0, 110, 150, 0.1, 1.8, 5.0),
    "report": Preset("report", 7.0, 400, 400, 0.02, 1.4, 3.5),
    "presentation": Preset("presentation", 14.0, 120, 200, 0.15, 2.6, 8.0),
}
PRESETS["publication"] = PRESETS["report"]  # alias


def _preset(name: str) -> Preset:
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}") from None


def _rc(p: Preset, bundled: bool) -> Any:
    # Any: Matplotlib's stubs type rcParams keys as a closed Literal set, not str.
    s = p.base
    return {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": ["sans-serif", "DejaVu Sans"],
        "font.sans-serif": ["DejaVu Sans"] if bundled else [*FONT_SANS],
        "font.serif": ["DejaVu Serif"] if bundled else [*FONT_SERIF],
        "font.size": s,
        "axes.labelsize": s,
        "axes.titlesize": s + 0.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": s * 0.8,
        "axes.labelpad": 4.0,
        "xtick.labelsize": s - 0.5,
        "ytick.labelsize": s - 0.5,
        "legend.fontsize": s - 0.5,
        "legend.title_fontsize": s - 0.5,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.grid": False,
        "axes.prop_cycle": cycler(color=CATEGORICAL),
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.pad": 3.0,
        "ytick.major.pad": 3.0,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "lines.linewidth": p.line,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.markersize": p.marker,
        "errorbar.capsize": 2.0,
        "patch.edgecolor": SURFACE,
        "figure.dpi": p.dpi,
        "savefig.dpi": p.export_dpi,
        "savefig.bbox": "tight",
        "savefig.pad_inches": p.pad,
        "image.cmap": "sgc_sequential",
        # Real text in vector exports: TrueType in PDF/PS, outlines in SVG.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "path",
    }


def use(report: bool = False, *, preset: str | None = None, bundled_fonts: bool = False) -> None:
    """Apply the shared rcParams globally.

    ``report=True`` sizes text for an IEEE column (the ``"report"`` preset); ``preset``
    overrides it with ``"notebook"``, ``"report"``/``"publication"`` or
    ``"presentation"``. ``bundled_fonts=True`` pins DejaVu for output that must be
    byte-identical across machines.
    """
    name = preset or ("report" if report else "notebook")
    mpl.rcParams.update(_rc(_preset(name), bundled_fonts))


def context(
    report: bool = False, *, preset: str | None = None, bundled_fonts: bool = False
) -> AbstractContextManager[None]:
    """Scoped version of :func:`use` for ``with`` blocks; rcParams are restored on exit."""
    name = preset or ("report" if report else "notebook")
    return mpl.rc_context(rc=_rc(_preset(name), bundled_fonts))


def figsize(width: float | str = "wide", aspect: float = 0.62) -> tuple[float, float]:
    """Figure size in inches: ``width`` is a ``SIZES`` key or inches, height = width * aspect."""
    w = SIZES[width] if isinstance(width, str) else float(width)
    return (w, w * aspect)


def save(fig: Figure, path: str | Path, preset: str | None = None, **kwargs: Any) -> Path:
    """Export ``fig`` on the surface colour with tight margins; the format follows the suffix.

    With a ``preset`` its export dpi and padding are applied (``"report"`` for
    publication, ``"presentation"`` for slides); without one the dpi and padding
    already set by :func:`use` apply. Extra ``kwargs`` go to ``Figure.savefig``.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    opts: dict[str, Any] = {"facecolor": SURFACE, "bbox_inches": "tight"}
    if preset is not None:
        p = _preset(preset)
        opts.update(dpi=p.export_dpi, pad_inches=p.pad)
    opts.update(kwargs)
    fig.savefig(target, **opts)
    return target


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------
def _base() -> float:
    return float(mpl.rcParams["font.size"])


def _lw() -> float:
    return float(mpl.rcParams["lines.linewidth"])


def _fig(target: Axes | Figure) -> Figure:
    fig = target if isinstance(target, Figure) else target.get_figure(root=True)
    assert fig is not None
    return fig


def _arr(x: ArrayLike) -> Floats:
    return np.asarray(x, dtype=float)


def _finite_pairs(x: ArrayLike, y: ArrayLike) -> tuple[Floats, Floats]:
    xa, ya = _arr(x), _arr(y)
    if xa.shape != ya.shape:
        raise ValueError(f"x and y must have the same shape, got {xa.shape} and {ya.shape}")
    keep = np.isfinite(xa) & np.isfinite(ya)
    return xa[keep], ya[keep]


def _luminance(color: Any) -> float:
    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(c) for c in to_rgb(color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _text_on(color: Any) -> str:
    """Ink or white, whichever reads better on ``color``."""
    lum = _luminance(color)
    return INK if (lum + 0.05) / (_luminance(INK) + 0.05) >= 1.05 / (lum + 0.05) else "#FFFFFF"


def _empty(ax: Axes, message: str = "No data") -> None:
    ax.text(
        0.5,
        0.5,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=MUTED,
        fontsize=_base() * 0.9,
    )


def _spread(
    ys: Sequence[float], gap: float, lo: float | None = None, hi: float | None = None
) -> list[float]:
    """Shifts that push sorted neighbours at least ``gap`` apart.

    The block is centred on its original positions, then slid to stay within ``[lo, hi]``
    (the bottom wins if the block is taller than the range).
    """
    if not ys:
        return []
    order = sorted(range(len(ys)), key=ys.__getitem__)
    placed = [ys[order[0]]]
    for i in order[1:]:
        placed.append(max(ys[i], placed[-1] + gap))
    shift = -sum(p - ys[i] for p, i in zip(placed, order, strict=True)) / len(ys)
    if hi is not None and placed[-1] + shift > hi:
        shift = hi - placed[-1]
    if lo is not None and placed[0] + shift < lo:
        shift = lo - placed[0]
    shifts = [0.0] * len(ys)
    for p, i in zip(placed, order, strict=True):
        shifts[i] = p + shift - ys[i]
    return shifts


def _points_y(ax: Axes, x: float, y: float) -> float:
    """Vertical display position of a data point, in points."""
    fig = _fig(ax)
    return float(ax.transData.transform((x, y))[1]) * 72.0 / fig.dpi


def _y_range_points(ax: Axes) -> tuple[float, float]:
    """Bottom and top of the axes box, in points, for clamping label stacks."""
    fig = _fig(ax)
    return float(ax.bbox.y0) * 72.0 / fig.dpi, float(ax.bbox.y1) * 72.0 / fig.dpi


def _signed(text: str) -> str:
    """Typographic minus for value labels, and no sign on a rounded zero ("-0.00", "+0.0" -> "0.0")."""
    t = text.replace("-", _MINUS)
    return t if any(c in "123456789" for c in t) else t.lstrip(_MINUS + "+")


def _fit_labels(ax: Axes, items: Sequence[tuple[float, str, int]]) -> None:
    """Widen the x-limits so value labels set beside their marks stay clear of the frame.

    ``items`` are ``(x, text, side)`` with side +1 for a label to the right of ``x`` and
    -1 for one to the left. Widths are estimated from the character count, so call this
    after the axes box is final.
    """
    if not items:
        return
    for _ in range(2):  # widening shrinks points-per-unit, so settle in a second pass
        ux, _unused = _units_per_point(ax)
        lo, hi = ax.get_xlim()
        for x, text, side in items:
            reach = (len(text) * 0.62 * _base() + 10.0) * ux
            if side < 0:
                lo = min(lo, x - reach)
            else:
                hi = max(hi, x + reach)
        ax.set_xlim(lo, hi)


# ---------------------------------------------------------------------------
# Axes primitives (legacy API)
# ---------------------------------------------------------------------------
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
    """Horizontal bar, data end rounded, baseline end square. Call after limits are set.

    ``radius_pt=0`` draws a plain rectangle (the editorial default of :func:`barh`).
    """
    y0, y1 = y - height / 2, y + height / 2
    ux, uy = _units_per_point(ax)
    sign = 1.0 if x1 >= x0 else -1.0
    span = abs(x1 - x0)
    rx = min(radius_pt * ux, span / 2 if span > 0 else 0.0)
    ry = min(radius_pt * uy, height / 2)
    if radius_pt <= 0 or rx <= 0 or ry <= 0:
        verts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO, MplPath.CLOSEPOLY]
    else:
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
    radius_pt: float = 0.0,
) -> None:
    """Horizontal bars, first label at the top, values at the tips in ink.

    Bars are square-ended (Swiss); pass ``radius_pt`` to round the data end.
    Negative bars grow left of ``base`` with a baseline drawn at ``base``.
    """
    n = len(values)
    if n == 0:
        _empty(ax)
        return
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
    if lo < base:
        ax.axvline(base, color=AXIS, linewidth=0.8, zorder=2)
    if value_fmt and xlim is None:
        _fit_labels(ax, [(v, _signed(value_fmt.format(v)), 1 if v >= base else -1) for v in values])
    off = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.012
    for y, v, c in zip(ys, values, cols, strict=True):
        rounded_barh(ax, y, base, v, height, c, radius_pt=radius_pt)
        if value_fmt:
            ax.text(
                v + (off if v >= base else -off),
                y,
                _signed(value_fmt.format(v)),
                va="center",
                ha="left" if v >= base else "right",
                color=INK_2,
                fontsize=mpl.rcParams["ytick.labelsize"],
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.8},
                zorder=5,
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


def legend(ax: Axes, **kwargs: Any) -> Legend | None:
    """Frameless legend in secondary ink; returns ``None`` when nothing is labelled.

    Any ``ax.legend`` keyword overrides the defaults, including ``frameon``.
    """
    if not ax.get_legend_handles_labels()[0] and "handles" not in kwargs:
        return None
    opts: dict[str, Any] = {"frameon": False, "handlelength": 1.4, "borderpad": 0.2}
    opts.update(labelspacing=0.3)
    opts.update(kwargs)
    leg = ax.legend(**opts)
    for t in leg.get_texts():
        t.set_color(INK_2)
    leg.get_title().set_color(INK_2)
    return leg


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


# ---------------------------------------------------------------------------
# Editorial furniture: titles, notes, callouts, reference lines
# ---------------------------------------------------------------------------
def _bounds(target: Axes | Figure, renderer: RendererBase) -> Bbox:
    """Union of the axes boxes and their tick/label extents, in display pixels."""
    axes = [target] if isinstance(target, Axes) else list(target.axes)
    parts: list[Bbox] = []
    for a in axes:
        parts.append(a.bbox)
        for axis in (a.yaxis, a.xaxis):
            tb = axis.get_tightbbox(renderer)
            if tb is not None:
                parts.append(tb)
    return Bbox.union(parts)


def _place(
    target: Axes | Figure, text: str, corner: Literal["top", "bottom"], offset: float, **kw: Any
) -> Annotation:
    """Text flush with the left edge of the labelled plot, ``offset`` pt above/below it."""
    top = corner == "top"
    ann = Annotation(
        text,
        xy=(0.0, 1.0 if top else 0.0),
        xycoords=lambda r: _bounds(target, r),
        xytext=(0.0, offset),
        textcoords="offset points",
        ha="left",
        va="bottom" if top else "top",
        annotation_clip=False,
        **kw,
    )
    _fig(target).add_artist(ann)
    return ann


def _wrap(text: str, size: float, fig: Figure) -> str:
    chars = max(16, int(fig.get_figwidth() * 72 * 0.92 / (size * 0.52)))
    return "\n".join(textwrap.fill(line, chars) for line in text.split("\n"))


def eyebrow(target: Axes | Figure, text: str, pad: float | None = None) -> Annotation:
    """Small uppercase section label flush with the top-left of the plot."""
    s = _base() * 0.78
    return _place(
        target,
        text.upper(),
        "top",
        _base() * 0.9 if pad is None else pad,
        fontsize=s,
        fontweight="bold",
        color=ACCENT_DEEP,
    )


def title(
    target: Axes | Figure,
    text: str,
    subtitle: str | None = None,
    *,
    eyebrow: str | None = None,
    pad: float | None = None,
    wrap: bool = True,
) -> None:
    """Serif display title, optional sans subtitle and eyebrow, stacked above the plot.

    The stack is flush with the left edge of the y tick labels. ``target`` may be an
    ``Axes`` or a ``Figure`` (then the stack spans all axes). Titles wrap to the figure
    width unless ``wrap=False``; pass explicit ``\\n`` to control breaks.
    """
    fig = _fig(target)
    base = _base()
    offset = base * 0.9 if pad is None else pad
    line = 1.28

    def lay(s: str, size: float, **kw: Any) -> None:
        nonlocal offset
        body = _wrap(s, size, fig) if wrap else s
        _place(target, body, "top", offset, fontsize=size, **kw)
        offset += (body.count("\n") + 1) * size * line + base * 0.35

    if subtitle:
        lay(subtitle, base * 0.95, color=INK_2)
    lay(text, base * 1.55, fontfamily=_SERIF, fontweight="bold", color=INK)
    if eyebrow:
        lay(eyebrow.upper(), base * 0.78, fontweight="bold", color=ACCENT_DEEP)


def source(target: Axes | Figure, text: str, *, prefix: str = "Source: ") -> Annotation:
    """Muted note flush with the bottom-left of the plot; repeat calls stack downwards.

    Use ``prefix="Method: "`` for a methodology note.
    """
    size = _base() * 0.75
    body = _wrap(prefix + text, size, _fig(target))
    earlier = [
        a for a in _fig(target).artists if isinstance(a, Annotation) and a.get_gid() == _SOURCE
    ]
    offset = -_base() * 0.8 - sum((a.get_text().count("\n") + 1) * size * 1.4 for a in earlier)
    note = _place(target, body, "bottom", offset, fontsize=size, color=MUTED, linespacing=1.3)
    note.set_gid(_SOURCE)
    return note


def panel_label(ax: Axes, text: str) -> Annotation:
    """Bold panel letter ("a", "b") hung just above-left of the axes."""
    return ax.annotate(
        text,
        (0.0, 1.0),
        xycoords="axes fraction",
        xytext=(0, _base() * 0.6),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=_base() * 1.05,
        fontweight="bold",
        color=INK,
        annotation_clip=False,
    )


def kpi(
    ax: Axes,
    value: str,
    label: str,
    *,
    xy: tuple[float, float] = (0.0, 1.0),
    ha: Literal["left", "right"] = "left",
    accent: bool = False,
    note: str | None = None,
) -> None:
    """Statistic callout in axes-fraction coordinates: big figure, label, optional muted note."""
    base = _base()
    big = base * 2.4
    pad = {"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5}
    ax.text(
        xy[0],
        xy[1],
        value,
        transform=ax.transAxes,
        ha=ha,
        va="top",
        fontsize=big,
        fontweight="bold",
        color=ACCENT_DEEP if accent else INK,
        bbox=pad,
        zorder=6,
    )
    parts = [(label, INK_2, base * 0.85)] + ([(note, MUTED, base * 0.78)] if note else [])
    drop = big * 1.3
    for text, colour, size in parts:
        ax.annotate(
            text,
            xy,
            xycoords="axes fraction",
            xytext=(0, -drop),
            textcoords="offset points",
            ha=ha,
            va="top",
            fontsize=size,
            color=colour,
            bbox=pad,
            zorder=6,
        )
        drop += size * 1.35


def callout(
    ax: Axes,
    xy: tuple[float, float],
    text: str,
    *,
    offset: tuple[float, float] = (14.0, 14.0),
    accent: bool = False,
    ha: Literal["left", "right", "center"] = "left",
) -> Annotation:
    """Text with a hairline leader to the data point ``xy``; ``accent`` for the key finding."""
    colour = ACCENT if accent else MUTED
    return ax.annotate(
        text,
        xy,
        xytext=offset,
        textcoords="offset points",
        ha=ha,
        va="center",
        fontsize=_base() * 0.85,
        color=ACCENT_DEEP if accent else INK_2,
        fontweight="bold" if accent else "normal",
        bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.0},
        arrowprops={
            "arrowstyle": "-",
            "color": colour,
            "linewidth": 0.7,
            "shrinkA": 1,
            "shrinkB": 3,
        },
        zorder=7,
    )


_REF_STYLE: dict[str, dict[str, Any]] = {
    "baseline": {"color": AXIS, "linestyle": "-", "linewidth": 0.8},
    "reference": {"color": MUTED, "linestyle": (0, (4, 3)), "linewidth": 0.8},
    "threshold": {"color": INK_2, "linestyle": (0, (5, 2)), "linewidth": 1.0},
    "benchmark": {"color": ACCENT, "linestyle": "-", "linewidth": 1.3},
}
RefKind = Literal["baseline", "reference", "threshold", "benchmark"]
_DIAGONAL = "sgc-diagonal"
_SOURCE = "sgc-source"


def refline(
    ax: Axes,
    value: float,
    *,
    axis: Literal["x", "y"] = "y",
    kind: RefKind = "reference",
    label: str | None = None,
    label_at: float = 0.98,
) -> Line2D:
    """Reference line with optional inline label.

    ``kind``: ``baseline`` (zero line), ``reference`` (no-skill, overall rate),
    ``threshold`` (decision or quality cut-off) or ``benchmark`` (the line to beat,
    drawn in the accent). ``label_at`` is the position along the line, 0 to 1.
    """
    style = _REF_STYLE[kind]
    line = (
        ax.axhline(value, zorder=2, **style)
        if axis == "y"
        else ax.axvline(value, zorder=2, **style)
    )
    if label:
        right = label_at >= 0.5
        if axis == "y":
            xy, off = (label_at, value), (-2.0 if right else 2.0, 3.0)
            tr = blended_transform_factory(ax.transAxes, ax.transData)
            va = "bottom"
        else:
            xy, off = (value, label_at), (3.0, -2.0 if right else 2.0)
            tr = blended_transform_factory(ax.transData, ax.transAxes)
            va = "top" if right else "bottom"
        ax.annotate(
            label,
            xy,
            xycoords=tr,
            xytext=off,
            textcoords="offset points",
            ha=("right" if right else "left") if axis == "y" else "left",
            va=va,
            fontsize=_base() * 0.8,
            color=ACCENT_DEEP if kind == "benchmark" else INK_2,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.0, "alpha": 0.9},
            annotation_clip=False,
            zorder=6,
        )
    return line


def diagonal(ax: Axes, label: str | None = None) -> Line2D:
    """The y = x reference (chance for ROC, perfect for calibration), drawn once per axes."""
    for existing in ax.get_lines():
        if existing.get_gid() == _DIAGONAL:
            return existing
    line = ax.axline((0, 0), (1, 1), zorder=1, **_REF_STYLE["reference"])
    line.set_gid(_DIAGONAL)
    if label:
        ax.text(
            0.86,
            0.86,
            label,
            transform=ax.transData,
            rotation=45,
            rotation_mode="anchor",
            transform_rotates_text=True,
            ha="center",
            va="top",
            fontsize=_base() * 0.8,
            color=MUTED,
        )
    return line


def ci_band(
    ax: Axes, x: ArrayLike, lo: ArrayLike, hi: ArrayLike, color: str = INK_2, alpha: float = 0.16
) -> None:
    """Translucent confidence band under a line."""
    ax.fill_between(_arr(x), _arr(lo), _arr(hi), color=color, alpha=alpha, linewidth=0, zorder=1.5)


def ci_bars(
    ax: Axes,
    pos: ArrayLike,
    center: ArrayLike,
    lo: ArrayLike,
    hi: ArrayLike,
    *,
    orient: Literal["h", "v"] = "h",
    color: str = INK_2,
    linewidth: float = 0.9,
) -> None:
    """Capped confidence intervals around ``center`` at ``pos`` (``h``: along x, ``v``: along y)."""
    c, lo_, hi_ = _arr(center), _arr(lo), _arr(hi)
    err = [np.clip(c - lo_, 0, None), np.clip(hi_ - c, 0, None)]
    kw: dict[str, Any] = {
        "fmt": "none",
        "ecolor": color,
        "elinewidth": linewidth,
        "capthick": linewidth,
        "zorder": 4,
    }
    if orient == "h":
        ax.errorbar(c, _arr(pos), xerr=err, **kw)
    else:
        ax.errorbar(_arr(pos), c, yerr=err, **kw)


def highlight(
    labels: Sequence[str],
    focus: str | Iterable[str],
    *,
    color: str = ACCENT,
    base: str = CONTEXT,
) -> list[str]:
    """Per-label colours: ``color`` for the focus labels, ``base`` for the rest."""
    chosen = {focus} if isinstance(focus, str) else set(focus)
    return [color if label in chosen else base for label in labels]


def series_style(i: int) -> dict[str, Any]:
    """Colour, line style and marker for categorical slot ``i`` (cycles after four)."""
    return dict(_SERIES[i % len(_SERIES)])


def encode(keys: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Stable style per entity, in the order given: re-use the same keys across charts."""
    return {k: series_style(i) for i, k in enumerate(dict.fromkeys(keys))}


def _rupee(v: float, decimals: int) -> str:
    sign, a = (_MINUS if v < 0 else ""), abs(v)
    for scale, unit in ((1e7, " Cr"), (1e5, " L"), (1e3, "k")):
        if a >= scale:
            return f"{sign}{_RUPEE}{a / scale:.{decimals}f}{unit}".replace(".0 ", " ")
    return f"{sign}{_RUPEE}{a:,.0f}"


def axis_format(
    ax: Axes,
    axis: Literal["x", "y"] = "y",
    kind: Literal["percent", "percent_points", "number", "signed", "rupee"] = "percent",
    decimals: int = 0,
) -> None:
    """Tick labels as ``percent`` (0-1), ``percent_points`` (0-100), ``number``, ``signed``, ``rupee``."""
    fmt: Any
    if kind == "percent":
        fmt = PercentFormatter(xmax=1, decimals=decimals)
    elif kind == "percent_points":
        fmt = PercentFormatter(xmax=100, decimals=decimals)
    elif kind == "number":
        fmt = FuncFormatter(lambda v, _: f"{v:,.{decimals}f}".replace("-", _MINUS))
    elif kind == "signed":
        fmt = FuncFormatter(lambda v, _: f"{v:+.{decimals}f}".replace("-", _MINUS))
    elif kind == "rupee":
        fmt = FuncFormatter(lambda v, _: _rupee(v, decimals))
    else:
        raise ValueError(f"unknown axis format {kind!r}")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


def direct_labels(
    ax: Axes,
    lines: Sequence[Line2D] | None = None,
    *,
    dx: float = 5.0,
    min_gap: float | None = None,
) -> list[Annotation]:
    """Label each labelled line at its last point, nudging labels apart so none collide.

    Replaces a legend. Accent-coloured lines get an accent label. Call after limits are set.
    """
    pool = ax.get_lines() if lines is None else lines
    ends: list[tuple[Line2D, float, float]] = []
    for ln in pool:
        name = str(ln.get_label())
        if not name or name.startswith("_"):
            continue
        xy = np.asarray(ln.get_xydata(), dtype=float).reshape(-1, 2)
        finite = xy[np.isfinite(xy).all(axis=1)]
        if len(finite):
            ends.append((ln, float(finite[-1, 0]), float(finite[-1, 1])))
    gap = _base() * 1.2 if min_gap is None else min_gap
    bottom, top = _y_range_points(ax)
    shifts = _spread(
        [_points_y(ax, x, y) for _, x, y in ends], gap, bottom + gap / 2, top - gap / 2
    )
    out: list[Annotation] = []
    for (ln, x, y), shift in zip(ends, shifts, strict=True):
        is_accent = str(ln.get_color()).lower() == ACCENT.lower()
        out.append(
            ax.annotate(
                str(ln.get_label()),
                (x, y),
                xytext=(dx, shift),
                textcoords="offset points",
                va="center",
                fontsize=_base() * 0.85,
                color=ACCENT_DEEP if is_accent else INK_2,
                fontweight="bold" if is_accent else "normal",
                annotation_clip=False,
            )
        )
    return out


def signed_labels(ax: Axes, negative: str, positive: str, x: float = 0.0) -> None:
    """Direction captions just inside the top edge, either side of ``x`` (usually zero).

    Leave headroom above the top row; the SHAP and coefficient helpers already do.
    """
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    kw: dict[str, Any] = {
        "textcoords": "offset points",
        "xycoords": tr,
        "va": "top",
        "fontsize": _base() * 0.8,
        "color": MUTED,
        "annotation_clip": False,
    }
    ax.annotate(f"← {negative}", (x, 1.0), xytext=(-4, -3), ha="right", **kw)
    ax.annotate(f"{positive} →", (x, 1.0), xytext=(4, -3), ha="left", **kw)


# ---------------------------------------------------------------------------
# Generic comparison charts
# ---------------------------------------------------------------------------
def dotplot(
    ax: Axes,
    labels: Sequence[str],
    values: Sequence[float],
    *,
    lo: Sequence[float] | None = None,
    hi: Sequence[float] | None = None,
    colors: Sequence[str] | str | None = None,
    focus: str | Iterable[str] | None = None,
    reference: float | None = None,
    reference_label: str | None = None,
    reference_kind: RefKind = "reference",
    value_fmt: str | None = "{:.3f}",
    xlim: tuple[float, float] | None = None,
    size: float = 6.5,
) -> None:
    """Cleveland dot plot, first label on top, with optional intervals and a reference line.

    The analytical workhorse for model comparison, segment comparison and coefficients:
    ``focus`` labels take the accent, the rest stay ink; ``lo``/``hi`` add confidence
    intervals; ``reference`` draws a baseline, benchmark or threshold.
    """
    n = len(values)
    if n == 0:
        _empty(ax)
        return
    if colors is None:
        cols = [BLUE] * n if focus is None else highlight(labels, focus, base=BLUE)
    else:
        cols = [colors] * n if isinstance(colors, str) else list(colors)
    v = _arr(values)
    extent = [v]
    if lo is not None and hi is not None:
        extent += [_arr(lo), _arr(hi)]
    if reference is not None:
        extent.append(np.array([reference]))
    allv = np.concatenate(extent)
    lo_x, hi_x = float(np.nanmin(allv)), float(np.nanmax(allv))
    pad = (hi_x - lo_x) * 0.12 or 1.0
    ax.set_xlim(xlim or (lo_x - pad * 0.5, hi_x + pad))
    ax.set_ylim(-0.6, n - 0.4)
    ys = np.arange(n)[::-1]
    ax.set_yticks(ys, list(labels))
    strip(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    tip = _arr(hi) if hi is not None and lo is not None else v
    if value_fmt and xlim is None:
        _fit_labels(
            ax, [(max(x, t), _signed(value_fmt.format(x)), 1) for x, t in zip(v, tip, strict=True)]
        )
    ax.hlines(ys, ax.get_xlim()[0], ax.get_xlim()[1], color=GRID, linewidth=0.5, zorder=0)
    if reference is not None:
        refline(ax, reference, axis="x", kind=reference_kind, label=reference_label, label_at=0.02)
    if lo is not None and hi is not None:
        ci_bars(ax, ys, v, lo, hi)
    for y, x, c in zip(ys, v, cols, strict=True):
        dot(ax, [x], [y], c, size=size, zorder=5)
    if value_fmt:
        off = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.015
        for y, x, t in zip(ys, v, tip, strict=True):
            ax.text(
                max(x, t) + off,
                y,
                _signed(value_fmt.format(x)),
                va="center",
                ha="left",
                color=INK_2,
                fontsize=mpl.rcParams["ytick.labelsize"],
            )


def model_comparison(
    ax: Axes,
    names: Sequence[str],
    scores: Sequence[float],
    *,
    lo: Sequence[float] | None = None,
    hi: Sequence[float] | None = None,
    focus: str | Iterable[str] | None = None,
    baseline: float | None = None,
    baseline_label: str | None = "baseline",
    sort: bool = True,
    **kwargs: Any,
) -> None:
    """Models ranked on one metric: focus model in the accent, optional baseline and CIs."""
    idx = list(range(len(names)))
    if sort:
        idx.sort(key=lambda i: -scores[i])
    dotplot(
        ax,
        [names[i] for i in idx],
        [scores[i] for i in idx],
        lo=None if lo is None else [lo[i] for i in idx],
        hi=None if hi is None else [hi[i] for i in idx],
        focus=focus,
        reference=baseline,
        reference_label=baseline_label if baseline is not None else None,
        **kwargs,
    )


def dumbbell(
    ax: Axes,
    labels: Sequence[str],
    a: Sequence[float],
    b: Sequence[float],
    *,
    a_label: str,
    b_label: str,
    colors: tuple[str, str] = (BLUE, ACCENT),
    value_fmt: str | None = None,
) -> None:
    """Paired values per row joined by a rule: feature sets, models or periods compared."""
    n = len(a)
    if n == 0:
        _empty(ax)
        return
    if len(b) != n or len(labels) != n:
        raise ValueError("labels, a and b must have the same length")
    both = np.concatenate([_arr(a), _arr(b)])
    span = float(np.nanmax(both) - np.nanmin(both)) or 1.0
    ax.set_xlim(float(np.nanmin(both)) - span * 0.12, float(np.nanmax(both)) + span * 0.18)
    ax.set_ylim(-0.6, n - 0.4)
    ys = np.arange(n)[::-1]
    ax.set_yticks(ys, list(labels))
    strip(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    if value_fmt:
        _fit_labels(
            ax,
            [
                (v, _signed(value_fmt.format(v)), side)
                for va, vb in zip(a, b, strict=True)
                for v, side in ((va, -1 if va < vb else 1), (vb, 1 if va < vb else -1))
            ],
        )
    for y, va, vb in zip(ys, a, b, strict=True):
        ax.plot([va, vb], [y, y], color=CONTEXT, linewidth=2.0, zorder=2, solid_capstyle="butt")
    dot(ax, a, ys, colors[0], size=7, zorder=4, label=a_label)
    dot(ax, b, ys, colors[1], size=7, zorder=4, label=b_label)
    if value_fmt:
        for y, va, vb in zip(ys, a, b, strict=True):
            for v, side in ((va, -1 if va < vb else 1), (vb, 1 if va < vb else -1)):
                ax.annotate(
                    _signed(value_fmt.format(v)),
                    (v, y),
                    xytext=(side * 7, 0),
                    textcoords="offset points",
                    ha="left" if side > 0 else "right",
                    va="center",
                    fontsize=_base() * 0.8,
                    color=INK_2,
                )


def grouped_bars(
    ax: Axes,
    categories: Sequence[str],
    series: Mapping[str, Sequence[float]],
    *,
    colors: Sequence[str] | None = None,
    width: float = 0.78,
    value_fmt: str | None = None,
) -> None:
    """Vertical grouped bars, one bar per series within each category, labelled for a legend."""
    n_cat, n_ser = len(categories), len(series)
    if n_cat == 0 or n_ser == 0:
        _empty(ax)
        return
    cols = list(colors) if colors is not None else [BLUE, AQUA, YELLOW, CONTEXT][:n_ser]
    if len(cols) < n_ser:
        cols = (cols * n_ser)[:n_ser]
    x = np.arange(n_cat)
    each = width / n_ser
    for i, (name, vals) in enumerate(series.items()):
        if len(vals) != n_cat:
            raise ValueError(f"series {name!r} has {len(vals)} values for {n_cat} categories")
        pos = x - width / 2 + each * (i + 0.5)
        ax.bar(pos, vals, width=each * 0.92, color=cols[i], label=name, zorder=3)
        if value_fmt:
            for px, v in zip(pos, vals, strict=True):
                ax.annotate(
                    value_fmt.format(v),
                    (px, v),
                    xytext=(0, 3 if v >= 0 else -3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=_base() * 0.78,
                    color=INK_2,
                )
    ax.set_xticks(x, list(categories))
    ax.tick_params(axis="x", length=0)
    strip(ax, "y")
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom, top + (top - bottom) * 0.16)  # room for labels and a legend


def segment_bars(
    ax: Axes,
    labels: Sequence[str],
    values: Sequence[float],
    *,
    benchmark: float | None = None,
    benchmark_label: str | None = "Overall",
    focus: str | Iterable[str] | None = None,
    value_fmt: str | None = "{:.0%}",
    ylim: tuple[float, float] | None = None,
) -> None:
    """Segments compared on one measure against an overall benchmark.

    All bars are ink; with ``focus`` the chosen segments take the accent and the rest
    recede to grey.
    """
    n = len(values)
    if n == 0:
        _empty(ax)
        return
    cols = [BLUE] * n if focus is None else highlight(labels, focus)
    top = max(max(values), benchmark if benchmark is not None else float("-inf"), 0.0)
    ax.set_ylim(ylim or (min(0.0, *values), top * 1.15 or 1.0))
    ax.bar(range(n), values, width=0.62, color=cols, zorder=3)
    ax.set_xticks(range(n), list(labels))
    ax.tick_params(axis="x", length=0)
    strip(ax, "y")
    if benchmark is not None:
        # With a focus bar in the accent the benchmark steps back to ink so the two don't merge.
        kind: RefKind = "benchmark" if focus is None else "threshold"
        refline(ax, benchmark, axis="y", kind=kind, label=benchmark_label, label_at=0.99)
    if value_fmt:
        for x, v in zip(range(n), values, strict=True):
            ax.annotate(
                value_fmt.format(v),
                (x, v),
                xytext=(0, 3 if v >= 0 else -3),
                textcoords="offset points",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=_base() * 0.85,
                color=INK_2,
            )


def importance(
    ax: Axes,
    names: Sequence[str],
    values: Sequence[float],
    *,
    top: int = 10,
    focus: int = 1,
    value_fmt: str | None = "{:.2f}",
) -> None:
    """Feature importance bars, largest first; the leading ``focus`` bars take the accent."""
    order = sorted(range(len(values)), key=lambda i: -values[i])[:top]
    vals = [values[i] for i in order]
    barh(
        ax,
        [names[i] for i in order],
        vals,
        colors=[ACCENT if k < focus else CONTEXT for k in range(len(order))],
        value_fmt=value_fmt,
    )


# ---------------------------------------------------------------------------
# Classification and model evaluation
# ---------------------------------------------------------------------------
def _unit_frame(ax: Axes, xlabel: str, ylabel: str) -> None:
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ticks = np.linspace(0, 1, 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    strip(ax, "both")
    if not ax.get_xlabel():
        ax.set_xlabel(xlabel)
    if not ax.get_ylabel():
        ax.set_ylabel(ylabel)


def roc(
    ax: Axes,
    fpr: ArrayLike,
    tpr: ArrayLike,
    *,
    auc: float | None = None,
    label: str | None = None,
    color: str = ACCENT,
    mark: tuple[float, float] | None = None,
    mark_label: str | None = None,
) -> None:
    """ROC curve on a square frame with the chance diagonal (drawn once per axes).

    Call once per model; the first call draws the frame. ``auc`` is computed by the
    trapezoid rule when omitted. ``mark`` flags an operating point such as the chosen
    threshold. Pass ``label`` and call :func:`legend` to name the curves.
    """
    x, y = _finite_pairs(fpr, tpr)
    _unit_frame(ax, "False positive rate", "True positive rate")
    diagonal(ax, "chance")
    if x.size == 0:
        _empty(ax)
        return
    order = np.argsort(x, kind="stable")
    area = auc if auc is not None else float(np.trapezoid(y[order], x[order]))
    ax.plot(x, y, color=color, zorder=3, label=f"{label}  ·  AUC {area:.3f}" if label else None)
    if mark is not None:
        dot(ax, [mark[0]], [mark[1]], color, size=7, zorder=6)
        if mark_label:
            callout(ax, mark, mark_label, offset=(12, -16), accent=color == ACCENT)


def pr_curve(
    ax: Axes,
    recall: ArrayLike,
    precision: ArrayLike,
    *,
    ap: float | None = None,
    label: str | None = None,
    color: str = ACCENT,
    prevalence: float | None = None,
) -> None:
    """Precision-recall curve; ``prevalence`` draws the no-skill line (precision = base rate)."""
    x, y = _finite_pairs(recall, precision)
    _unit_frame(ax, "Recall", "Precision")
    if prevalence is not None:
        refline(ax, prevalence, axis="y", kind="reference", label=f"no skill ({prevalence:.2f})")
    if x.size == 0:
        _empty(ax)
        return
    name = None
    if label:
        name = f"{label}  ·  AP {ap:.3f}" if ap is not None else label
    ax.plot(x, y, color=color, zorder=3, label=name)


def confusion_matrix(
    ax: Axes,
    counts: ArrayLike,
    labels: Sequence[str] = ("Not on track", "On track"),
    *,
    normalize: Literal["true", "pred", "all"] | None = "true",
    cmap: Colormap | str | None = None,
) -> AxesImage | None:
    """Confusion matrix with counts as the headline and shares beneath.

    ``counts`` is ``[actual, predicted]`` as returned by scikit-learn. Shade and the
    small percentage use ``normalize`` (``"true"`` = share of each actual class, i.e.
    recall on the diagonal) so classes of different size stay comparable; ``None``
    shades by raw count. Correct cells (the diagonal) are outlined.
    """
    cm = _arr(counts)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"confusion matrix must be square, got shape {cm.shape}")
    if cm.size == 0:
        _empty(ax)
        return None
    if len(labels) != cm.shape[0]:
        raise ValueError(f"{len(labels)} labels for a {cm.shape[0]}x{cm.shape[1]} matrix")
    shares = np.zeros_like(cm)
    if normalize is not None:
        denom = {"true": cm.sum(1, keepdims=True), "pred": cm.sum(0, keepdims=True)}.get(
            normalize, np.full_like(cm, cm.sum())
        )
        np.divide(cm, denom, out=shares, where=denom > 0)
    shade = cm if normalize is None else shares
    colormap = mpl.colormaps[cmap] if isinstance(cmap, str) else (cmap or SEQUENTIAL)
    top = float(np.max(shade)) or 1.0
    im = heatmap(
        ax,
        shade,
        labels,
        labels,
        cmap=colormap,
        vmin=0.0,
        vmax=top,
        annotate=False,
        aspect="equal",
    )
    norm = Normalize(0.0, top)
    base = _base()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ink = _text_on(colormap(norm(shade[i, j])))
            ax.text(
                j,
                i - (0.08 if normalize else 0.0),
                f"{cm[i, j]:,.0f}",
                ha="center",
                va="center",
                color=ink,
                fontsize=base * 1.3,
                fontweight="bold",
            )
            if normalize:
                ax.text(
                    j,
                    i + 0.24,
                    f"{shares[i, j]:.0%}",
                    ha="center",
                    va="center",
                    color=ink,
                    fontsize=base * 0.85,
                )
    for i in range(cm.shape[0]):
        ax.add_patch(
            Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=1.2, zorder=4)
        )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return im


def calibration(
    ax: Axes,
    predicted: ArrayLike,
    observed: ArrayLike,
    *,
    counts: ArrayLike | None = None,
    lo: ArrayLike | None = None,
    hi: ArrayLike | None = None,
    label: str | None = None,
    color: str = ACCENT,
    ece: float | None = None,
) -> None:
    """Reliability curve against the perfect-calibration diagonal.

    ``predicted``/``observed`` are bin means; ``lo``/``hi`` give observed-rate
    intervals; ``counts`` adds a histogram of bin sizes along the floor so thinly
    populated bins are visible; ``ece`` is printed as supplied, never recomputed.
    """
    p, o = _finite_pairs(predicted, observed)
    _unit_frame(ax, "Predicted probability", "Observed rate")
    diagonal(ax, "perfect")
    if p.size == 0:
        _empty(ax)
        return
    if lo is not None and hi is not None:
        ci_bars(ax, p, o, lo, hi, orient="v", color=color)
    ax.plot(
        p, o, color=color, marker="o", markersize=4, markeredgecolor=SURFACE, zorder=4, label=label
    )
    if counts is not None:
        c = _arr(counts)
        if c.shape != _arr(predicted).shape:
            raise ValueError("counts must match predicted")
        c = c[np.isfinite(_arr(predicted)) & np.isfinite(_arr(observed))]
        tr = blended_transform_factory(ax.transData, ax.transAxes)
        width = float(np.min(np.diff(np.sort(p)))) * 0.8 if p.size > 1 else 0.05
        ax.bar(
            p,
            c / (np.max(c) or 1.0) * 0.12,
            width=width,
            bottom=0.0,
            color=CONTEXT,
            transform=tr,
            zorder=2,
        )
    if ece is not None:
        ax.text(
            0.03,
            0.97,
            f"ECE {ece:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=_base() * 0.85,
            color=INK_2,
        )


def threshold_curves(
    ax: Axes,
    thresholds: ArrayLike,
    curves: Mapping[str, ArrayLike],
    *,
    selected: float | None = None,
    focus: str | None = None,
) -> None:
    """Metrics against the decision threshold (precision, recall, F1, cost...).

    ``focus`` takes the accent and the rest cycle through the neutral palette with
    distinct line styles; ``selected`` marks the chosen cut-off and the value of each
    curve there. Curves are directly labelled.
    """
    t = _arr(thresholds)
    if t.size == 0 or not curves:
        _empty(ax)
        return
    others = 0
    for name, series in curves.items():
        y = _arr(series)
        if y.shape != t.shape:
            raise ValueError(f"curve {name!r} must match thresholds")
        if name == focus:
            ax.plot(t, y, color=ACCENT, linewidth=_lw() * 1.15, zorder=4, label=name)
        else:
            st = _NEUTRAL_SERIES[others % len(_NEUTRAL_SERIES)]
            ax.plot(t, y, color=st["color"], linestyle=st["linestyle"], zorder=3, label=name)
            others += 1
    ax.set_xlabel("Decision threshold")
    strip(ax, "both")
    if selected is not None:
        refline(
            ax,
            selected,
            axis="x",
            kind="threshold",
            label=f"threshold {selected:.2f}",
            label_at=0.02,
        )
        order = np.argsort(t)
        for name, series in curves.items():
            v = float(np.interp(selected, t[order], _arr(series)[order]))
            dot(ax, [selected], [v], ACCENT if name == focus else INK_2, size=5.5, zorder=6)
    direct_labels(ax)


# ---------------------------------------------------------------------------
# Heatmaps and explainability
# ---------------------------------------------------------------------------
def heatmap(
    ax: Axes,
    data: ArrayLike,
    row_labels: Sequence[str] | None = None,
    col_labels: Sequence[str] | None = None,
    *,
    cmap: Colormap | str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    center: float | None = None,
    fmt: str | Callable[[float], str] = "{:.2f}",
    annotate: bool = True,
    colorbar: bool = False,
    aspect: Literal["equal", "auto"] = "auto",
) -> AxesImage | None:
    """Cell grid with in-cell values whose ink flips to keep contrast on any shade.

    Without ``center`` the scale is sequential (default ``SEQUENTIAL``). With
    ``center`` (typically 0) the scale is symmetric about it on ``DIVERGING`` so the
    neutral colour is always the centre value. NaN cells are left blank.
    """
    arr = _arr(data)
    if arr.ndim != 2:
        raise ValueError(f"heatmap needs a 2-D array, got {arr.ndim}-D")
    if arr.size == 0:
        _empty(ax)
        return None
    finite = arr[np.isfinite(arr)]
    if center is not None:
        span = float(np.max(np.abs(finite - center))) if finite.size else 1.0
        lo, hi = center - (span or 1.0), center + (span or 1.0)
        default = DIVERGING
    else:
        lo = vmin if vmin is not None else (float(np.min(finite)) if finite.size else 0.0)
        hi = vmax if vmax is not None else (float(np.max(finite)) if finite.size else 1.0)
        default = SEQUENTIAL
    colormap = mpl.colormaps[cmap] if isinstance(cmap, str) else (cmap or default)
    norm = Normalize(lo, hi)
    im = ax.imshow(arr, cmap=colormap, norm=norm, aspect=aspect, interpolation="nearest")
    nr, nc = arr.shape
    ax.set_xticks(range(nc), list(col_labels) if col_labels is not None else [])
    ax.set_yticks(range(nr), list(row_labels) if row_labels is not None else [])
    ax.set_xticks(np.arange(-0.5, nc, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nr, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.6)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0, labelcolor=INK_2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if annotate:
        size = _base() * (0.85 if max(nr, nc) <= 10 else 0.68)
        for i in range(nr):
            for j in range(nc):
                v = arr[i, j]
                if not np.isfinite(v):
                    continue
                ax.text(
                    j,
                    i,
                    _signed(fmt(v) if callable(fmt) else fmt.format(v)),
                    ha="center",
                    va="center",
                    fontsize=size,
                    color=_text_on(colormap(norm(v))),
                )
    if colorbar:
        cb = _fig(ax).colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0, labelsize=_base() * 0.8, labelcolor=INK_2)
    return im


def binned_median(
    x: ArrayLike, y: ArrayLike, bins: int = 12, min_count: int = 10
) -> tuple[Floats, Floats]:
    """Median of ``y`` within equal-count bins of ``x``; sparse bins are dropped.

    Returns the bin median of ``x`` and of ``y``. Empty arrays when nothing qualifies.
    """
    xs, ys = _finite_pairs(x, y)
    if xs.size == 0:
        return xs, ys
    edges = np.unique(np.quantile(xs, np.linspace(0, 1, bins + 1)))
    which = np.clip(np.searchsorted(edges, xs, side="right") - 1, 0, max(len(edges) - 2, 0))
    bx: list[float] = []
    by: list[float] = []
    for k in range(max(len(edges) - 1, 1)):
        m = which == k
        if m.sum() >= min_count:
            bx.append(float(np.median(xs[m])))
            by.append(float(np.median(ys[m])))
    return np.array(bx), np.array(by)


def shap_summary(
    ax: Axes,
    shap_values: ArrayLike,
    X: ArrayLike,
    feature_names: Sequence[str],
    *,
    max_display: int = 10,
    colorbar: bool = True,
    seed: int = 0,
    point_size: float = 6.0,
    direction: tuple[str, str] | None = ("lowers prediction", "raises prediction"),
) -> None:
    """SHAP beeswarm: x is the contribution (right of zero raises the prediction), colour is
    the feature value, rows are ordered by mean |SHAP|.

    Pass the values for the explained class only (2-D, samples x features). Feature
    values are clipped to their 5th-95th percentile before shading, as shap does. The
    vertical spread is a deterministic density jitter, not a data axis.
    """
    sv, xv = _arr(shap_values), _arr(X)
    if sv.ndim != 2 or sv.shape != xv.shape:
        raise ValueError(f"shap_values and X must be 2-D and equal, got {sv.shape} and {xv.shape}")
    if sv.shape[1] != len(feature_names):
        raise ValueError(f"{len(feature_names)} names for {sv.shape[1]} features")
    if sv.size == 0:
        _empty(ax)
        return
    order = np.argsort(-np.nanmean(np.abs(sv), axis=0))[:max_display]
    rng = np.random.default_rng(seed)
    n = len(order)
    for rank, f in enumerate(order):
        y0 = n - 1 - rank
        s, v = sv[:, f], xv[:, f]
        ok = np.isfinite(s)
        counts, edges = np.histogram(s[ok], bins=40) if ok.any() else (np.zeros(1), np.zeros(2))
        bin_id = np.clip(np.digitize(s, edges[1:-1]), 0, len(counts) - 1)
        spread = 0.38 * counts[bin_id] / (np.max(counts) or 1.0)
        jitter = y0 + rng.uniform(-1, 1, size=s.shape) * spread
        finite_v = v[np.isfinite(v)]
        if finite_v.size and np.ptp(finite_v) > 0:
            lo, hi = np.percentile(finite_v, [5, 95])
            shade = np.clip((v - lo) / ((hi - lo) or 1.0), 0, 1)
        else:
            shade = np.full_like(v, 0.5)
        missing = ~np.isfinite(v)
        ax.scatter(
            s[ok & ~missing],
            jitter[ok & ~missing],
            c=shade[ok & ~missing],
            cmap=FEATURE_CMAP,
            vmin=0,
            vmax=1,
            s=point_size,
            linewidths=0,
            zorder=3,
            rasterized=True,
        )
        if (ok & missing).any():
            ax.scatter(
                s[ok & missing],
                jitter[ok & missing],
                color=CONTEXT,
                s=point_size,
                linewidths=0,
                zorder=3,
                rasterized=True,
            )
    ax.set_yticks(range(n), [feature_names[f] for f in order[::-1]])
    ax.set_ylim(-0.7, n - 0.3)
    strip(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.hlines(range(n), *ax.get_xlim(), color=GRID, linewidth=0.5, zorder=0)
    refline(ax, 0.0, axis="x", kind="baseline")
    ax.set_xlabel("SHAP value (contribution to the model output)")
    if direction:
        signed_labels(ax, *direction)
    if colorbar:
        sm = ScalarMappable(norm=Normalize(0, 1), cmap=FEATURE_CMAP)
        cb = _fig(ax).colorbar(sm, ax=ax, ticks=[0, 1], fraction=0.025, pad=0.02, aspect=28)
        cb.ax.set_yticklabels(["Low", "High"])
        cb.set_label("Feature value", color=INK_2, size=_base() * 0.85)
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0, labelsize=_base() * 0.8, labelcolor=INK_2)


def shap_dependence(
    ax: Axes,
    x: ArrayLike,
    shap_values: ArrayLike,
    *,
    color_by: ArrayLike | None = None,
    feature: str | None = None,
    color_label: str | None = None,
    trend: bool = True,
    bins: int = 12,
) -> None:
    """SHAP value against one feature, zero line kept, with a binned-median trend.

    ``color_by`` shades points by a second feature (interaction view). The trend uses
    equal-count bins and omits bins with fewer than ten points.
    """
    xs, ss = _finite_pairs(x, shap_values)
    if xs.size == 0:
        _empty(ax)
        return
    if color_by is None:
        ax.scatter(xs, ss, s=5, color=CONTEXT, alpha=0.55, linewidths=0, zorder=2, rasterized=True)
    else:
        c = _arr(color_by)
        c = c[np.isfinite(_arr(x)) & np.isfinite(_arr(shap_values))]
        sc = ax.scatter(
            xs, ss, s=5, c=c, cmap=FEATURE_CMAP, alpha=0.8, linewidths=0, zorder=2, rasterized=True
        )
        cb = _fig(ax).colorbar(sc, ax=ax, fraction=0.04, pad=0.02, aspect=28)
        cb.outline.set_visible(False)
        cb.set_label(color_label or "", color=INK_2, size=_base() * 0.85)
        cb.ax.tick_params(length=0, labelsize=_base() * 0.8, labelcolor=INK_2)
    refline(ax, 0.0, axis="y", kind="baseline")
    if trend:
        bx, by = binned_median(xs, ss, bins=bins)
        if bx.size > 1:
            ax.plot(bx, by, color=ACCENT, linewidth=_lw() * 1.1, zorder=4, label="binned median")
    strip(ax, "y")
    if feature:
        ax.set_xlabel(feature)
    ax.set_ylabel("SHAP value")


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def k_sweep(
    ax: Axes,
    k: ArrayLike,
    values: ArrayLike,
    *,
    chosen: int | None = None,
    threshold: float | None = None,
    threshold_label: str | None = None,
    ylabel: str | None = None,
) -> None:
    """Metric against number of clusters: elbow curve (inertia) or mean silhouette.

    ``chosen`` highlights the selected k; ``threshold`` draws a quality cut-off such as
    the 0.25 weak-structure line for silhouette. Integer k ticks.
    """
    ks, vs = _finite_pairs(k, values)
    if ks.size == 0:
        _empty(ax)
        return
    ax.plot(ks, vs, color=BLUE, zorder=3)
    dot(ax, ks, vs, BLUE, size=5.5, zorder=4)
    ax.set_xticks(ks)
    ax.set_xlabel("Number of clusters k")
    if ylabel:
        ax.set_ylabel(ylabel)
    strip(ax, "y")
    if threshold is not None:
        refline(ax, threshold, axis="y", kind="threshold", label=threshold_label)
    if chosen is not None and np.any(ks == chosen):
        yv = float(vs[ks == chosen][0])
        dot(ax, [chosen], [yv], ACCENT, size=8, zorder=6)
        end_label(ax, chosen, yv, f"k = {chosen}", dy=9)


def silhouette_knife(
    ax: Axes,
    scores: ArrayLike,
    labels: ArrayLike,
    *,
    focus: Iterable[Any] = (),
) -> None:
    """Per-sample silhouette values grouped by cluster, with the overall mean marked.

    Wide, uniform blades extending right of zero indicate coherent clusters; blades
    below the mean or reaching left of zero flag weak or mixed ones. ``focus`` clusters
    take the accent.
    """
    s = _arr(scores)
    lab = np.asarray(labels)
    if s.shape != lab.shape:
        raise ValueError("scores and labels must have the same shape")
    keep = np.isfinite(s)
    s, lab = s[keep], lab[keep]
    if s.size == 0:
        _empty(ax)
        return
    chosen = set(focus)
    gap = max(1.0, s.size * 0.02)
    y = 0.0
    mids: list[float] = []
    names: list[str] = []
    for c in sorted(np.unique(lab).tolist()):
        vals = np.sort(s[lab == c])
        ys = y + np.arange(len(vals))
        ax.fill_betweenx(ys, 0, vals, color=ACCENT if c in chosen else AQUA, linewidth=0, zorder=3)
        mids.append(float(ys.mean()))
        names.append(f"C{c}")
        y += len(vals) + gap
    ax.set_ylim(-gap, y + gap * 3)  # headroom for the mean label
    ax.set_yticks(mids, names)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Silhouette coefficient")
    strip(ax, "x")
    ax.spines["left"].set_visible(False)
    refline(ax, 0.0, axis="x", kind="baseline")
    mean = float(s.mean())
    refline(ax, mean, axis="x", kind="threshold", label=f"mean {mean:.2f}", label_at=0.99)


def cluster_profile(
    ax: Axes,
    values: ArrayLike,
    features: Sequence[str],
    clusters: Sequence[str],
    *,
    fmt: str = "{:+.1f}",
    colorbar: bool = False,
) -> AxesImage | None:
    """Cluster signatures as a diverging heatmap: rows are features, columns clusters.

    ``values`` are standardised deviations from the overall mean, so zero (the neutral
    colour) always means "typical" and the scale is symmetric.
    """
    return heatmap(ax, values, features, clusters, center=0.0, fmt=fmt, colorbar=colorbar)


# ---------------------------------------------------------------------------
# Regression diagnostics
# ---------------------------------------------------------------------------
def pred_vs_actual(
    ax: Axes,
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    stats: bool = True,
    trend: bool = True,
    hexbin_above: int = 2000,
) -> dict[str, float]:
    """Predicted against actual on equal axes with the y = x line.

    Dense data (more than ``hexbin_above`` points) is drawn as hexagonal bins. Returns
    ``n``, ``rmse`` and ``r2`` (1 - SSE/SST, so negative when worse than the mean; NaN
    for constant targets) computed on finite pairs, and prints them when ``stats``.
    """
    yt, yp = _finite_pairs(y_true, y_pred)
    out = {"n": float(yt.size), "rmse": float("nan"), "r2": float("nan")}
    if yt.size == 0:
        _empty(ax)
        return out
    sse = float(np.sum((yt - yp) ** 2))
    sst = float(np.sum((yt - yt.mean()) ** 2))
    out["rmse"] = float(np.sqrt(sse / yt.size))
    out["r2"] = 1.0 - sse / sst if sst > 0 else float("nan")
    both = np.concatenate([yt, yp])
    lo, hi = float(both.min()), float(both.max())
    pad = (hi - lo) * 0.04 or 1.0
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    if yt.size > hexbin_above:
        ax.hexbin(
            yt,
            yp,
            gridsize=36,
            mincnt=1,
            cmap=SEQUENTIAL,
            linewidths=0,
            zorder=2,
            extent=(lo - pad, hi + pad, lo - pad, hi + pad),
        )
    else:
        ax.scatter(yt, yp, s=7, color=BLUE, alpha=0.35, linewidths=0, zorder=2, rasterized=True)
    ax.axline((lo, lo), (hi, hi), color=MUTED, linestyle=(0, (4, 3)), linewidth=0.9, zorder=3)
    if trend:
        bx, by = binned_median(yt, yp)
        if bx.size > 1:
            ax.plot(bx, by, color=ACCENT, linewidth=_lw() * 1.1, zorder=5, label="binned median")
    strip(ax, "both")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    if stats:
        r2 = "" if np.isnan(out["r2"]) else f"R² {out['r2']:.2f}   "
        ax.text(
            0.03,
            0.97,
            f"{r2}RMSE {out['rmse']:.3g}   n = {yt.size:,}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=_base() * 0.85,
            color=INK_2,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5},
            zorder=7,
        )
    return out


def residual_vs_fitted(
    ax: Axes,
    fitted: ArrayLike,
    residuals: ArrayLike,
    *,
    band_sd: float = 2.0,
    trend: bool = True,
) -> None:
    """Residuals against fitted values: zero line, ±``band_sd`` sample SDs and a binned median.

    A trend that wanders from zero signals misspecification; a funnel signals
    heteroscedasticity. The SD band is descriptive (sample SD of the residuals shown).
    """
    f, r = _finite_pairs(fitted, residuals)
    if f.size == 0:
        _empty(ax)
        return
    ax.scatter(f, r, s=6, color=BLUE, alpha=0.3, linewidths=0, zorder=2, rasterized=True)
    refline(ax, 0.0, axis="y", kind="baseline")
    sd = float(r.std(ddof=1)) if r.size > 1 else 0.0
    if band_sd and sd > 0:
        for sign in (-1, 1):
            refline(
                ax,
                sign * band_sd * sd,
                axis="y",
                kind="reference",
                label=f"{'+' if sign > 0 else _MINUS}{band_sd:g} SD" if sign > 0 else None,
            )
    if trend:
        bx, by = binned_median(f, r)
        if bx.size > 1:
            ax.plot(bx, by, color=ACCENT, linewidth=_lw() * 1.1, zorder=5)
    strip(ax, "y")
    ax.set_xlabel("Fitted value")
    ax.set_ylabel("Residual")


def qq_plot(ax: Axes, residuals: ArrayLike) -> None:
    """Normal Q-Q of standardised residuals with the y = x reference.

    Tails curling away from the line mean heavier or lighter tails than a normal;
    points are the sorted residuals against normal quantiles at plotting positions
    (i - 0.5) / n.
    """
    from scipy.special import ndtri

    r = _arr(residuals)
    r = np.sort(r[np.isfinite(r)])
    if r.size < 2 or float(r.std(ddof=1)) == 0.0:
        _empty(ax, "Not enough variation")
        return
    z = (r - r.mean()) / r.std(ddof=1)
    q = ndtri((np.arange(1, r.size + 1) - 0.5) / r.size)
    lim = float(max(np.max(np.abs(q)), np.max(np.abs(z)))) * 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.axline((0, 0), (1, 1), color=MUTED, linestyle=(0, (4, 3)), linewidth=0.9, zorder=1)
    ax.scatter(q, z, s=6, color=BLUE, alpha=0.55, linewidths=0, zorder=2, rasterized=True)
    strip(ax, "both")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Standardised residuals")


def distribution(
    ax: Axes,
    values: ArrayLike,
    *,
    bins: int | Sequence[float] = 40,
    color: str = BLUE,
    weights: ArrayLike | None = None,
    density: bool = False,
    median: bool = True,
    reference: float | None = None,
    reference_label: str | None = None,
    reference_kind: RefKind = "benchmark",
) -> None:
    """Histogram with an optional median marker and reference value (a target or cut-off).

    Non-finite values are dropped. ``weights`` give survey-weighted counts; the median
    is then weighted too.
    """
    v = _arr(values)
    w = None if weights is None else _arr(weights)
    if w is not None and w.shape != v.shape:
        raise ValueError("weights must match values")
    keep = np.isfinite(v) if w is None else np.isfinite(v) & np.isfinite(w)
    v, w = v[keep], None if w is None else w[keep]
    if v.size == 0:
        _empty(ax)
        return
    ax.hist(
        v,
        bins=bins,
        weights=w,
        density=density,
        color=color,
        edgecolor=SURFACE,
        linewidth=0.5,
        zorder=3,
    )
    strip(ax, "y")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)  # two label rows above the tallest bar
    ax.set_ylabel("Density" if density else ("Weighted count" if w is not None else "Count"))
    if median:
        if w is None:
            m = float(np.median(v))
        else:
            order = np.argsort(v)
            cum = np.cumsum(w[order])
            m = float(v[order][np.searchsorted(cum, cum[-1] / 2)])
        refline(ax, m, axis="x", kind="threshold", label=f"median {m:.3g}", label_at=0.99)
    if reference is not None:
        refline(ax, reference, axis="x", kind=reference_kind, label=reference_label, label_at=0.9)


def coef_plot(
    ax: Axes,
    names: Sequence[str],
    coefs: Sequence[float],
    lo: Sequence[float],
    hi: Sequence[float],
    *,
    sort: bool = True,
    zero_label: str | None = None,
    value_fmt: str | None = None,
) -> None:
    """Coefficients with confidence intervals about a zero line.

    A coefficient is coloured only when its interval excludes zero: accent above,
    ink below; intervals that cross zero stay grey. Colour restates the interval and
    asserts nothing beyond it.
    """
    n = len(coefs)
    if not (len(names) == len(lo) == len(hi) == n):
        raise ValueError("names, coefs, lo and hi must have the same length")
    idx = sorted(range(n), key=lambda i: -abs(coefs[i])) if sort else list(range(n))
    cols = [ACCENT if lo[i] > 0 else BLUE if hi[i] < 0 else CONTEXT for i in idx]
    dotplot(
        ax,
        [names[i] for i in idx],
        [coefs[i] for i in idx],
        lo=[lo[i] for i in idx],
        hi=[hi[i] for i in idx],
        colors=cols,
        reference=0.0,
        reference_label=zero_label,
        reference_kind="baseline",
        value_fmt=value_fmt,
    )


# ---------------------------------------------------------------------------
# Business analytics
# ---------------------------------------------------------------------------
def curve(
    ax: Axes,
    x: ArrayLike,
    y: ArrayLike,
    *,
    label: str | None = None,
    color: str = BLUE,
    lo: ArrayLike | None = None,
    hi: ArrayLike | None = None,
    linewidth: float | None = None,
    **kwargs: Any,
) -> list[Line2D]:
    """Styled line with an optional confidence band: response curves, scenario paths."""
    xs, ys = _finite_pairs(x, y)
    if xs.size == 0:
        _empty(ax)
        return []
    if lo is not None and hi is not None:
        ci_band(
            ax,
            xs,
            _arr(lo)[np.isfinite(_arr(x)) & np.isfinite(_arr(y))],
            _arr(hi)[np.isfinite(_arr(x)) & np.isfinite(_arr(y))],
            color=color,
        )
    return ax.plot(
        xs, ys, color=color, linewidth=linewidth or _lw(), label=label, zorder=3, **kwargs
    )


def capture_curves(
    ax: Axes,
    fraction: ArrayLike,
    curves: Mapping[str, ArrayLike],
    *,
    focus: str | None = None,
    prevalence: float | None = None,
    budget: float | None = None,
    random: bool = True,
) -> None:
    """Share of the target reached against share of households contacted.

    ``fraction`` is the contacted share in 0-1 and each curve the cumulative share of
    the target captured. ``random`` draws the random-contact diagonal, ``prevalence``
    the perfect-targeting ceiling min(1, f / prevalence), ``budget`` a vertical line
    with each strategy's capture there, read off its curve by interpolation.
    ``focus`` takes the accent; the rest use the neutral palette with distinct dashes.
    """
    f = _arr(fraction)
    if f.size == 0 or not curves:
        _empty(ax)
        return
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    strip(ax, "both")
    axis_format(ax, "x")
    axis_format(ax, "y")
    ax.set_xlabel("Households contacted")
    ax.set_ylabel("Target reached")
    if random:
        ax.plot(
            f,
            f,
            color=MUTED,
            linestyle=(0, (4, 3)),
            linewidth=0.9,
            zorder=2,
            label="Random contact",
        )
    if prevalence:
        ax.plot(
            f,
            np.minimum(1.0, f / prevalence),
            color=INK_2,
            linestyle=(0, (1, 1.5)),
            linewidth=1.0,
            zorder=2,
            label="Ceiling",
        )
    others = 0
    plotted: list[tuple[str, Floats, str]] = []
    for name, y in curves.items():
        ys = _arr(y)
        if ys.shape != f.shape:
            raise ValueError(f"curve {name!r} must match fraction")
        if name == focus:
            ax.plot(f, ys, color=ACCENT, linewidth=_lw() * 1.2, zorder=5, label=name)
            plotted.append((name, ys, ACCENT))
        else:
            st = _NEUTRAL_SERIES[others % len(_NEUTRAL_SERIES)]
            ax.plot(f, ys, color=st["color"], linestyle=st["linestyle"], zorder=4, label=name)
            plotted.append((name, ys, st["color"]))
            others += 1
    if budget is not None:
        refline(ax, budget, axis="x", kind="threshold", label=f"budget {budget:.0%}", label_at=0.02)
        order = np.argsort(f)
        hits = [float(np.interp(budget, f[order], ys[order])) for _, ys, _ in plotted]
        bottom, top = _y_range_points(ax)
        gap = _base() * 1.15
        shifts = _spread(
            [_points_y(ax, budget, h) for h in hits], gap, bottom + gap / 2, top - gap / 2
        )
        for (name, _, colour), h, shift in zip(plotted, hits, shifts, strict=True):
            dot(ax, [budget], [h], colour, size=5.5, zorder=7)
            ax.annotate(
                f"{h:.0%}",
                (budget, h),
                xytext=(7, shift),
                textcoords="offset points",
                va="center",
                fontsize=_base() * 0.85,
                zorder=8,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.8},
                color=ACCENT_DEEP if name == focus else INK_2,
                fontweight="bold" if name == focus else "normal",
            )


def tornado(
    ax: Axes,
    labels: Sequence[str],
    low: Sequence[float],
    high: Sequence[float],
    base: float,
    *,
    low_label: str = "Low case",
    high_label: str = "High case",
    value_fmt: str | None = "{:.1f}",
) -> None:
    """One-way sensitivity: each driver's outcome at its low and high setting around ``base``.

    Rows are sorted by swing, widest on top. The low side is ink and the high side the
    accent; ``base`` is the labelled reference line.
    """
    n = len(labels)
    if not (len(low) == len(high) == n):
        raise ValueError("labels, low and high must have the same length")
    if n == 0:
        _empty(ax)
        return
    idx = sorted(range(n), key=lambda i: -abs(high[i] - low[i]))
    both = np.array([*low, *high, base], dtype=float)
    span = float(both.max() - both.min()) or 1.0
    ax.set_xlim(float(both.min()) - span * 0.06, float(both.max()) + span * 0.06)
    ax.set_ylim(-0.6, n - 0.4)
    ys = np.arange(n)[::-1]
    ax.set_yticks(ys, [labels[i] for i in idx])
    strip(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    if value_fmt:
        _fit_labels(
            ax, [(v, _signed(value_fmt.format(v)), 1 if v >= base else -1) for v in (*low, *high)]
        )
    off = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.012
    for y, i in zip(ys, idx, strict=True):
        # Longer bar first so a shorter one on the same side stays visible on top of it.
        pair = sorted(((low[i], BLUE), (high[i], ACCENT)), key=lambda t: -abs(t[0] - base))
        for v, colour in pair:
            rounded_barh(ax, float(y), base, v, 0.5, colour, radius_pt=0.0)
            if value_fmt and v != base:  # a zero-length bar is already named by the base line
                ax.text(
                    v + (off if v >= base else -off),
                    y,
                    _signed(value_fmt.format(v)),
                    ha="left" if v >= base else "right",
                    va="center",
                    color=INK_2,
                    fontsize=mpl.rcParams["ytick.labelsize"],
                )
    refline(
        ax,
        base,
        axis="x",
        kind="benchmark",
        label=f"base {_signed(value_fmt.format(base))}" if value_fmt else None,
        label_at=0.99,
    )
    ax.plot([], [], color=BLUE, linewidth=6, label=low_label)
    ax.plot([], [], color=ACCENT, linewidth=6, label=high_label)
