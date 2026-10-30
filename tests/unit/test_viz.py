from __future__ import annotations

import ast
import importlib
import inspect
import json
from collections.abc import Callable, Iterator
from itertools import pairwise
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.colors import to_hex, to_rgb
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from PIL import Image

from savings_goal import viz

# Font, glyph and layout problems all surface as UserWarning; treat them as defects.
pytestmark = pytest.mark.filterwarnings("error::UserWarning")

LEGACY_COLOURS = [
    "SURFACE",
    "INK",
    "INK_2",
    "MUTED",
    "GRID",
    "AXIS",
    "CONTEXT",
    "MIDPOINT",
    "BLUE",
    "ORANGE",
    "AQUA",
    "YELLOW",
    "RED",
]
LEGACY_FUNCTIONS = ["use", "strip", "rounded_barh", "barh", "end_label", "legend", "dot"]


@pytest.fixture(autouse=True)
def _isolated() -> Iterator[None]:
    """Every test starts from default rcParams and leaves no open figures behind."""
    with matplotlib.rc_context():
        yield
    plt.close("all")


@pytest.fixture
def ax() -> Axes:
    viz.use()
    _, axes = plt.subplots(figsize=(4, 3))
    return axes


def _rows(ax: Axes) -> list[str]:
    """Y tick labels from the top of the axes to the bottom."""
    labels = [t.get_text() for t in ax.get_yticklabels()]
    return [label for _, label in sorted(zip(ax.get_yticks(), labels, strict=True), reverse=True)]


def _colours(ax: Axes) -> dict[str, str]:
    """Marker colour keyed by the row label it sits on."""
    by_y = dict(zip(ax.get_yticks(), (t.get_text() for t in ax.get_yticklabels()), strict=True))
    return {
        by_y[ln.get_ydata()[0]]: to_hex(ln.get_color()).lower()
        for ln in ax.get_lines()
        if ln.get_marker() == "o"
    }


def _contrast(a: str, b: str) -> float:
    def lum(c: str) -> float:
        r, g, b_ = (v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in to_rgb(c))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b_

    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


# --------------------------------------------------------------------------------------
# Legacy API and backward compatibility
# --------------------------------------------------------------------------------------
def test_style_and_helpers() -> None:
    viz.use(report=True)
    assert matplotlib.rcParams["axes.facecolor"] == viz.SURFACE
    viz.use()
    fig, ax = plt.subplots()
    viz.barh(ax, ["a", "b"], [0.4, -0.2], colors=[viz.BLUE, viz.RED], value_fmt="{:+.1f}")
    viz.barh(ax, ["c"], [0.5], value_fmt=None, xlim=(0, 1))
    viz.dot(ax, [0, 1], [0, 1], viz.ORANGE)
    ax.plot([0, 1], [0, 1], label="line")
    viz.legend(ax)
    viz.end_label(ax, 1, 1, "end")
    viz.strip(ax, None)
    assert len(ax.patches) == 3
    assert viz.SEQUENTIAL(0.0) != viz.SEQUENTIAL(1.0)
    plt.close(fig)


@pytest.mark.parametrize("name", [*LEGACY_COLOURS, "COL", "WIDE"])
def test_legacy_constants_exist(name: str) -> None:
    assert hasattr(viz, name)


@pytest.mark.parametrize("name", LEGACY_COLOURS)
def test_colour_tokens_are_hex(name: str) -> None:
    value = getattr(viz, name)
    assert value.startswith("#") and len(value) == 7
    assert to_hex(value).lower() == value.lower()


def test_palette_collections_keep_their_shape() -> None:
    assert viz.CATEGORICAL == [viz.BLUE, viz.ORANGE, viz.AQUA, viz.YELLOW]
    assert len(viz.BLUE_RAMP) == 8
    assert len(viz.RED_RAMP) == 5
    assert (viz.COL, viz.WIDE) == (3.45, 7.16)


def test_editorial_tokens() -> None:
    assert (viz.SURFACE, viz.INK, viz.INK_2) == ("#F8F7F3", "#171717", "#484848")
    assert (viz.GRID, viz.AXIS, viz.CONTEXT, viz.ACCENT) == (
        "#E5E3DE",
        "#B9B7B1",
        "#C8C6C0",
        "#D94A3A",
    )
    assert viz.ORANGE == viz.ACCENT  # the legacy slot 2 is the one accent


def test_text_colours_are_legible_on_the_surface() -> None:
    assert _contrast(viz.INK, viz.SURFACE) >= 12
    assert _contrast(viz.INK_2, viz.SURFACE) >= 7
    assert _contrast(viz.MUTED, viz.SURFACE) >= 4.5  # muted notes still meet AA
    assert _contrast(viz.ACCENT_DEEP, viz.SURFACE) >= 4.5  # accent used as text
    assert _contrast(viz.ACCENT, viz.SURFACE) >= 3  # accent used as a mark
    for colour in viz.CATEGORICAL:
        assert _contrast(colour, viz.SURFACE) >= 2


def test_one_accent_everything_else_neutral() -> None:
    def chroma(c: str) -> float:
        r, g, b = to_rgb(c)
        return max(r, g, b) - min(r, g, b)

    neutral = [viz.INK, viz.INK_2, viz.MUTED, viz.GRID, viz.AXIS, viz.CONTEXT, viz.SURFACE]
    neutral += [viz.BLUE, viz.AQUA, viz.YELLOW, *viz.BLUE_RAMP]
    assert all(chroma(c) < 0.12 for c in neutral)
    assert chroma(viz.ACCENT) > 0.5


@pytest.mark.parametrize(
    ("name", "defaults"),
    [
        ("use", {"report": False}),
        ("strip", {"grid_axis": "x"}),
        ("rounded_barh", {"radius_pt": 2.0, "zorder": 3}),
        (
            "barh",
            {
                "colors": viz.BLUE,
                "height": 0.55,
                "value_fmt": "{:.3f}",
                "xlim": None,
                "base": 0.0,
            },
        ),
        ("end_label", {"dx": 0.0, "dy": 0.0}),
        ("dot", {"size": 5.0}),
    ],
)
def test_legacy_signature_defaults(name: str, defaults: dict[str, Any]) -> None:
    params = inspect.signature(getattr(viz, name)).parameters
    for key, value in defaults.items():
        assert params[key].default == value, f"{name}({key})"


@pytest.mark.parametrize(
    ("name", "positional"),
    [
        ("rounded_barh", ["ax", "y", "x0", "x1", "height", "color"]),
        ("barh", ["ax", "labels", "values", "colors", "height", "value_fmt", "xlim", "base"]),
        ("end_label", ["ax", "x", "y", "text", "dx", "dy"]),
        ("dot", ["ax", "x", "y", "color", "size"]),
        ("strip", ["ax", "grid_axis"]),
    ],
)
def test_legacy_positional_order_is_unchanged(name: str, positional: list[str]) -> None:
    params = list(inspect.signature(getattr(viz, name)).parameters)
    assert params[: len(positional)] == positional


@pytest.mark.parametrize("name", LEGACY_FUNCTIONS)
def test_legacy_functions_are_callable(name: str) -> None:
    assert callable(getattr(viz, name))


def test_use_report_and_presets() -> None:
    viz.use(report=True)
    assert matplotlib.rcParams["font.size"] == 7.0
    assert matplotlib.rcParams["figure.dpi"] == 400
    viz.use()
    assert matplotlib.rcParams["font.size"] == 10.0
    viz.use(preset="presentation")
    assert matplotlib.rcParams["font.size"] == 14.0
    viz.use(report=False, preset="publication")  # preset wins, alias for report
    assert matplotlib.rcParams["font.size"] == 7.0
    assert matplotlib.rcParams["axes.facecolor"] == viz.SURFACE
    assert matplotlib.rcParams["axes.spines.top"] is False
    assert matplotlib.rcParams["axes.prop_cycle"].by_key()["color"] == viz.CATEGORICAL


def test_unknown_preset_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown preset"):
        viz.use(preset="poster")
    with pytest.raises(ValueError, match="unknown preset"):
        viz.save(plt.figure(), "unused.png", preset="poster")


def test_context_restores_rcparams() -> None:
    before = matplotlib.rcParams["font.size"]
    with viz.context(preset="presentation"):
        assert matplotlib.rcParams["font.size"] == 14.0
    assert matplotlib.rcParams["font.size"] == before


def test_bundled_fonts_pin_dejavu() -> None:
    viz.use(bundled_fonts=True)
    assert matplotlib.rcParams["font.sans-serif"] == ["DejaVu Sans"]
    assert matplotlib.rcParams["font.serif"] == ["DejaVu Serif"]


def test_type_stack_has_a_bundled_last_resort() -> None:
    assert viz.FONT_SANS[-1] == "DejaVu Sans"
    assert viz.FONT_SERIF[-1] == "DejaVu Serif"
    viz.use()
    assert matplotlib.rcParams["font.family"][-1] == "DejaVu Sans"  # glyph fallback


# --------------------------------------------------------------------------------------
# Colormaps
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name",
    ["sgc_sequential", "sgc_accent", "sgc_diverging", "sgc_feature", "sgc_blue", "sgc_blue_red"],
)
def test_colormaps_are_registered(name: str) -> None:
    assert name in matplotlib.colormaps
    assert matplotlib.colormaps[name](0.5) is not None


def test_legacy_colormap_names_resolve_to_the_new_maps() -> None:
    assert matplotlib.colormaps["sgc_blue"](0.3) == viz.SEQUENTIAL(0.3)
    assert matplotlib.colormaps["sgc_blue_red"](0.8) == viz.DIVERGING(0.8)


def test_reimport_does_not_duplicate_registration() -> None:
    importlib.reload(viz)
    importlib.reload(viz)
    assert viz.SEQUENTIAL(0.0) != viz.SEQUENTIAL(1.0)


def test_diverging_is_centred_and_signed() -> None:
    assert to_rgb(viz.DIVERGING(0.5)) == pytest.approx(to_rgb(viz.MIDPOINT), abs=0.02)
    low, high = viz.DIVERGING(0.0), viz.DIVERGING(1.0)
    assert low[0] == pytest.approx(low[1], abs=0.05)  # negative pole is a neutral ink
    assert high[0] > high[2] + 0.3  # positive pole is the red accent


def test_sequential_runs_light_to_dark() -> None:
    assert sum(viz.SEQUENTIAL(0.0)[:3]) > sum(viz.SEQUENTIAL(1.0)[:3])


# --------------------------------------------------------------------------------------
# Legacy helpers: bars, legend, strip
# --------------------------------------------------------------------------------------
def test_barh_positive_and_negative_bars(ax: Axes) -> None:
    viz.barh(ax, ["up", "down"], [0.5, -0.25], value_fmt="{:+.2f}")
    assert len(ax.patches) == 2
    texts = [t.get_text() for t in ax.texts]
    assert texts == ["+0.50", "\u22120.25"]  # typographic minus
    lo, hi = ax.get_xlim()
    assert lo < -0.25 and hi > 0.5
    assert _rows(ax) == ["up", "down"]


def test_barh_first_label_is_on_top(ax: Axes) -> None:
    viz.barh(ax, ["first", "second", "third"], [1, 2, 3])
    assert _rows(ax) == ["first", "second", "third"]


def test_barh_value_labels_stay_clear_of_tick_labels(ax: Axes) -> None:
    viz.barh(ax, ["long label here", "x"], [-3.0, 1.0], value_fmt="{:.1f}")
    ax.figure.canvas.draw()  # type: ignore[union-attr]
    renderer = ax.figure.canvas.get_renderer()  # type: ignore[union-attr,attr-defined]
    frame = ax.get_window_extent(renderer)
    assert all(t.get_window_extent(renderer).x0 >= frame.x0 - 1 for t in ax.texts)


def test_barh_single_element_and_zero_values(ax: Axes) -> None:
    viz.barh(ax, ["only"], [0.0])
    assert len(ax.patches) == 1
    assert ax.get_xlim()[1] > 0


def test_barh_empty_input_draws_placeholder(ax: Axes) -> None:
    viz.barh(ax, [], [])
    assert not ax.patches
    assert [t.get_text() for t in ax.texts] == ["No data"]


def test_barh_default_bars_are_square_and_rounding_is_opt_in() -> None:
    fig, (flat, soft) = plt.subplots(1, 2)
    viz.barh(flat, ["a"], [1.0])
    viz.barh(soft, ["a"], [1.0], radius_pt=3.0)
    assert len(flat.patches[0].get_path().vertices) == 5  # rectangle
    assert len(soft.patches[0].get_path().vertices) == 9  # curved data end
    plt.close(fig)


def test_barh_custom_xlim_and_base(ax: Axes) -> None:
    viz.barh(ax, ["a", "b"], [0.8, 0.9], xlim=(0.5, 1.0), base=0.5)
    assert ax.get_xlim() == (0.5, 1.0)


def test_rounded_barh_rounds_the_data_end_only(ax: Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    viz.rounded_barh(ax, 0.5, 0.0, 0.8, 0.3, viz.BLUE, radius_pt=4.0)
    verts = ax.patches[0].get_path().vertices
    assert verts[0][0] == 0.0 and verts[-1][0] == 0.0  # baseline end square
    assert verts.max(axis=0)[0] == pytest.approx(0.8)


def test_rounded_barh_zero_length_bar_does_not_fail(ax: Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    viz.rounded_barh(ax, 0.5, 0.2, 0.2, 0.3, viz.BLUE)
    assert len(ax.patches) == 1


def test_legend_styles_text_and_returns_the_legend(ax: Axes) -> None:
    ax.plot([0, 1], [0, 1], label="series")
    leg = viz.legend(ax, title="Group")
    assert isinstance(leg, Legend)
    assert to_hex(leg.get_texts()[0].get_color()) == viz.INK_2.lower()
    assert not leg.get_frame_on()


def test_legend_without_labelled_artists_returns_none(ax: Axes) -> None:
    ax.plot([0, 1], [0, 1])
    assert viz.legend(ax) is None


def test_legend_accepts_overriding_defaults(ax: Axes) -> None:
    ax.plot([0, 1], [0, 1], label="series")
    leg = viz.legend(ax, frameon=True, loc="lower right")  # legacy raised TypeError here
    assert leg is not None and leg.get_frame_on()


def test_strip_grid_axes(ax: Axes) -> None:
    viz.strip(ax, "y")
    assert not ax.spines["top"].get_visible() and not ax.spines["right"].get_visible()
    assert ax.yaxis.get_gridlines()[0].get_visible()
    assert not ax.xaxis.get_gridlines()[0].get_visible()
    viz.strip(ax, "both")
    assert ax.xaxis.get_gridlines()[0].get_visible()


def test_end_label_and_dot(ax: Axes) -> None:
    viz.end_label(ax, 1, 1, "end", dx=2, dy=3)
    viz.dot(ax, [0.5], [0.5], viz.ORANGE, size=7, label="pt")
    assert ax.texts[0].get_text() == "end"
    line = ax.get_lines()[0]
    assert line.get_markersize() == 7
    assert to_hex(line.get_markeredgecolor()).lower() == viz.SURFACE.lower()


# --------------------------------------------------------------------------------------
# Editorial furniture
# --------------------------------------------------------------------------------------
def _draw(fig: Figure) -> Any:
    fig.canvas.draw()
    return fig.canvas.get_renderer()  # type: ignore[attr-defined]


def test_title_stack_sits_above_the_plot_in_serif(ax: Axes) -> None:
    ax.plot([0, 1], [0, 1])
    viz.title(ax, "Income explains most of it", "ROC-AUC by feature set", eyebrow="Models")
    renderer = _draw(ax.figure)  # type: ignore[arg-type]
    anns = list(ax.figure.artists)  # type: ignore[union-attr]
    assert len(anns) == 3
    top = ax.get_window_extent(renderer).y1
    assert all(a.get_window_extent(renderer).y0 >= top - 1 for a in anns)
    headline = next(a for a in anns if a.get_text().startswith("Income"))
    assert headline.get_fontfamily()[0] == "serif"
    assert headline.get_fontweight() == "bold"
    eyebrow = next(a for a in anns if a.get_text() == "MODELS")
    assert to_hex(eyebrow.get_color()).lower() == viz.ACCENT_DEEP.lower()


def test_title_is_flush_with_the_tick_labels(ax: Axes) -> None:
    viz.barh(ax, ["a rather long category name"], [1.0])
    viz.title(ax, "Headline")
    renderer = _draw(ax.figure)  # type: ignore[arg-type]
    labels_x0 = min(t.get_window_extent(renderer).x0 for t in ax.get_yticklabels())
    headline = ax.figure.artists[0]  # type: ignore[union-attr]
    assert headline.get_window_extent(renderer).x0 == pytest.approx(labels_x0, abs=2)


def test_title_wraps_long_text_to_the_figure_width() -> None:
    viz.use()
    fig, ax = plt.subplots(figsize=(3, 2))
    viz.title(ax, "A very long headline that cannot possibly fit on one narrow line " * 2)
    renderer = _draw(fig)
    assert "\n" in fig.artists[0].get_text()
    assert fig.artists[0].get_window_extent(renderer).width <= fig.get_figwidth() * fig.dpi * 1.05
    plt.close(fig)


def test_title_on_a_figure_spans_all_axes() -> None:
    viz.use()
    fig, (a, b) = plt.subplots(1, 2)
    viz.title(fig, "Two panels", wrap=False)
    renderer = _draw(fig)
    assert fig.artists[0].get_window_extent(renderer).y0 >= max(a.bbox.y1, b.bbox.y1) - 1
    plt.close(fig)


def test_source_note_is_below_the_axis_labels(ax: Axes) -> None:
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel("x label")
    viz.source(ax, "IHDS-II, 2011-12")
    viz.source(ax, "grouped five-fold CV", prefix="Method: ")
    renderer = _draw(ax.figure)  # type: ignore[arg-type]
    notes = ax.figure.artists  # type: ignore[union-attr]
    assert [n.get_text() for n in notes] == [
        "Source: IHDS-II, 2011-12",
        "Method: grouped five-fold CV",
    ]
    first, second = (n.get_window_extent(renderer) for n in notes)
    assert first.y1 <= ax.xaxis.label.get_window_extent(renderer).y0
    assert second.y1 <= first.y0 + 1  # repeat calls stack instead of overprinting
    assert to_hex(notes[0].get_color()).lower() == viz.MUTED.lower()


def test_eyebrow_and_panel_label(ax: Axes) -> None:
    ann = viz.eyebrow(ax, "section")
    assert ann.get_text() == "SECTION"
    label = viz.panel_label(ax, "a")
    assert label.get_text() == "a" and label.get_fontweight() == "bold"


def test_kpi_callout_stacks_value_label_and_note(ax: Axes) -> None:
    viz.kpi(ax, "84.2%", "ROC-AUC", accent=True, note="95% CI 82.6 to 85.8")
    texts = [t.get_text() for t in ax.texts]
    assert texts == ["84.2%", "ROC-AUC", "95% CI 82.6 to 85.8"]
    assert to_hex(ax.texts[0].get_color()).lower() == viz.ACCENT_DEEP.lower()


def test_callout_points_at_the_data(ax: Axes) -> None:
    ann = viz.callout(ax, (0.5, 0.5), "peak", accent=True)
    assert ann.xy == (0.5, 0.5)
    assert ann.get_fontweight() == "bold"
    assert viz.callout(ax, (0.2, 0.2), "plain").get_fontweight() == "normal"


@pytest.mark.parametrize("kind", ["baseline", "reference", "threshold", "benchmark"])
def test_refline_kinds(ax: Axes, kind: Any) -> None:
    line = viz.refline(ax, 0.5, kind=kind, label=f"{kind} line")
    assert line.get_ydata()[0] == 0.5
    assert ax.texts[-1].get_text() == f"{kind} line"
    assert (to_hex(line.get_color()).lower() == viz.ACCENT.lower()) == (kind == "benchmark")


def test_refline_vertical_without_label(ax: Axes) -> None:
    line = viz.refline(ax, 2.0, axis="x")
    assert line.get_xdata()[0] == 2.0
    assert not ax.texts


def test_diagonal_is_drawn_once_and_survives_autoscale(ax: Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    first = viz.diagonal(ax, "chance")
    assert viz.diagonal(ax) is first
    assert sum(ln.get_gid() == "sgc-diagonal" for ln in ax.get_lines()) == 1
    assert ax.get_xlim() == (0, 1)


def test_ci_helpers_draw_intervals(ax: Axes) -> None:
    viz.ci_band(ax, [0, 1, 2], [0, 0, 0], [1, 1, 2])
    viz.ci_bars(ax, [0, 1], [0.5, 0.6], [0.4, 0.5], [0.7, 0.9])
    viz.ci_bars(ax, [0, 1], [0.5, 0.6], [0.4, 0.5], [0.7, 0.9], orient="v")
    assert len(ax.collections) >= 3  # band plus two error-bar line collections
    plt.close(ax.figure)  # type: ignore[arg-type]


def test_ci_bars_clip_inverted_intervals(ax: Axes) -> None:
    viz.ci_bars(ax, [0], [0.5], [0.6], [0.4])  # lo above centre: never a negative error
    assert len(ax.collections) == 1


def test_highlight_and_encoding_helpers() -> None:
    assert viz.highlight(["a", "b", "c"], "b") == [viz.CONTEXT, viz.ACCENT, viz.CONTEXT]
    assert viz.highlight(["a", "b"], {"a", "b"}, color="#000000") == ["#000000"] * 2
    assert viz.highlight(["a"], "z", base="#111111") == ["#111111"]
    style = viz.series_style(1)
    assert style["color"] == viz.ACCENT and {"linestyle", "marker"} <= set(style)
    assert viz.series_style(5) == viz.series_style(1)  # cycles
    enc = viz.encode(["m1", "m2", "m1", "m3"])
    assert list(enc) == ["m1", "m2", "m3"]
    assert len({(s["color"], s["linestyle"], s["marker"]) for s in enc.values()}) == 3
    assert viz.series_style(0) is not viz.series_style(0)  # callers may mutate their copy


@pytest.mark.parametrize(
    ("kind", "value", "decimals", "expected"),
    [
        ("percent", 0.25, 0, "25%"),
        ("percent", 0.255, 1, "25.5%"),
        ("percent_points", 25, 0, "25%"),
        ("number", 12345.6, 0, "12,346"),
        ("signed", 0.5, 1, "+0.5"),
        ("signed", -0.5, 1, "\u22120.5"),
        ("rupee", 950, 0, "₹950"),
        ("rupee", 12000, 0, "₹12k"),
        ("rupee", 250000, 1, "₹2.5 L"),
        ("rupee", 30000000, 0, "₹3 Cr"),
        ("rupee", -2000, 0, "\u2212₹2k"),
    ],
)
def test_axis_format(ax: Axes, kind: Any, value: float, decimals: int, expected: str) -> None:
    viz.axis_format(ax, "y", kind, decimals)
    assert ax.yaxis.get_major_formatter()(value) == expected


def test_axis_format_x_and_unknown_kind(ax: Axes) -> None:
    viz.axis_format(ax, "x", "percent")
    assert ax.xaxis.get_major_formatter()(0.5) == "50%"
    with pytest.raises(ValueError, match="unknown axis format"):
        viz.axis_format(ax, "y", "bogus")  # type: ignore[arg-type]


def test_direct_labels_never_collide_and_stay_inside(ax: Axes) -> None:
    x = np.linspace(0, 1, 20)
    for i, end in enumerate([0.50, 0.505, 0.51, 0.001]):  # three lines ending together
        ax.plot(x, x * end, label=f"series {i}")
    ax.plot(x, x, label="_hidden")
    ax.set_xlim(0, 1.4)
    anns = viz.direct_labels(ax)
    renderer = _draw(ax.figure)  # type: ignore[arg-type]
    boxes = sorted((a.get_window_extent(renderer) for a in anns), key=lambda b: b.y0)
    assert len(anns) == 4
    assert all(lo.y1 <= hi.y0 + 1 for lo, hi in pairwise(boxes))
    frame = ax.get_window_extent(renderer)
    assert all(frame.y0 - 1 <= b.y0 and b.y1 <= frame.y1 + 1 for b in boxes)


def test_direct_labels_accent_line_gets_accent_text(ax: Axes) -> None:
    ax.plot([0, 1], [0, 1], color=viz.ACCENT, label="Model")
    ax.plot([0, 1], [1, 0.2], color=viz.BLUE, label="Other")
    anns = {a.get_text(): a for a in viz.direct_labels(ax)}
    assert to_hex(anns["Model"].get_color()).lower() == viz.ACCENT_DEEP.lower()
    assert to_hex(anns["Other"].get_color()).lower() == viz.INK_2.lower()


def test_direct_labels_skip_empty_and_nan_lines(ax: Axes) -> None:
    ax.plot([], [], label="empty")
    ax.plot([0, 1], [np.nan, np.nan], label="nan")
    assert viz.direct_labels(ax) == []


def test_spread_keeps_gap_and_bounds() -> None:
    shifts = viz._spread([10.0, 10.5, 11.0], gap=5.0)
    placed = sorted(y + s for y, s in zip([10.0, 10.5, 11.0], shifts, strict=True))
    assert placed[1] - placed[0] >= 5 and placed[2] - placed[1] >= 5
    clamped = viz._spread([0.0, 0.5], gap=4.0, lo=2.0, hi=20.0)
    assert min(y + s for y, s in zip([0.0, 0.5], clamped, strict=True)) == pytest.approx(2.0)
    assert viz._spread([], 3.0) == []


def test_signed_labels_and_signed_text(ax: Axes) -> None:
    viz.signed_labels(ax, "lowers", "raises")
    assert [t.get_text() for t in ax.texts] == ["← lowers", "raises →"]
    assert viz._signed("-0.00") == "0.00"
    assert viz._signed("+0.0") == "0.0"
    assert viz._signed("-0.31") == "\u22120.31"
    assert viz._signed("+0.31") == "+0.31"


def test_text_on_picks_the_readable_ink() -> None:
    assert viz._text_on(viz.INK) == "#FFFFFF"
    assert viz._text_on(viz.SURFACE) == viz.INK
    assert viz._text_on(viz.BLUE_RAMP[-1]) == "#FFFFFF"
    assert viz._text_on(viz.BLUE_RAMP[0]) == viz.INK


# --------------------------------------------------------------------------------------
# Specialised charts
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def data() -> dict[str, Any]:
    rng = np.random.default_rng(3)
    n = 800
    x = rng.normal(size=(n, 4))
    y = (x[:, 0] + 0.4 * x[:, 1] + rng.normal(0, 1, n) > 0).astype(int)
    score = 1 / (1 + np.exp(-(x[:, 0] + 0.4 * x[:, 1])))
    order = np.argsort(-score)
    tpr = np.cumsum(y[order]) / y.sum()
    fpr = np.cumsum(1 - y[order]) / (1 - y).sum()
    return {"X": x, "y": y, "score": score, "fpr": fpr, "tpr": tpr, "names": list("abcd")}


def test_roc_keeps_diagonal_and_square_frame(ax: Axes, data: dict[str, Any]) -> None:
    viz.roc(ax, data["fpr"], data["tpr"], label="Model")
    viz.roc(ax, data["fpr"], data["tpr"] ** 2, label="Weaker", color=viz.CONTEXT, auc=0.5)
    assert sum(ln.get_gid() == "sgc-diagonal" for ln in ax.get_lines()) == 1
    assert ax.get_aspect() == 1.0
    assert ax.get_xlim()[0] < 0 < 1 < ax.get_xlim()[1]
    labels = ax.get_legend_handles_labels()[1]
    assert labels[0].startswith("Model") and "AUC 0." in labels[0]
    assert labels[1].endswith("AUC 0.500")  # supplied value is shown, not recomputed
    assert ax.get_xlabel() == "False positive rate"


def test_roc_auc_is_the_trapezoid_area(ax: Axes) -> None:
    viz.roc(ax, [0, 0.5, 1], [0, 1, 1], label="m")
    assert ax.get_legend_handles_labels()[1] == ["m  ·  AUC 0.750"]


def test_roc_marks_an_operating_point(ax: Axes, data: dict[str, Any]) -> None:
    viz.roc(ax, data["fpr"], data["tpr"], mark=(0.2, 0.7), mark_label="threshold 0.5")
    assert any(t.get_text() == "threshold 0.5" for t in ax.texts)


def test_roc_empty_keeps_frame(ax: Axes) -> None:
    viz.roc(ax, [], [])
    assert any(ln.get_gid() == "sgc-diagonal" for ln in ax.get_lines())
    assert any(t.get_text() == "No data" for t in ax.texts)


def test_pr_curve_draws_the_no_skill_line(ax: Axes) -> None:
    viz.pr_curve(ax, [0, 0.5, 1], [1, 0.7, 0.45], ap=0.71, label="Model", prevalence=0.4)
    assert ax.get_legend_handles_labels()[1] == ["Model  ·  AP 0.710"]
    assert any("no skill (0.40)" in t.get_text() for t in ax.texts)
    assert any(ln.get_ydata()[0] == 0.4 for ln in ax.get_lines() if len(ln.get_ydata()) == 2)


def test_pr_curve_without_prevalence_or_ap(ax: Axes) -> None:
    viz.pr_curve(ax, [0, 1], [1, 0.5], label="m")
    assert ax.get_legend_handles_labels()[1] == ["m"]


def test_confusion_matrix_counts_and_row_shares(ax: Axes) -> None:
    im = viz.confusion_matrix(ax, [[80, 20], [10, 90]])
    texts = [t.get_text() for t in ax.texts]
    assert texts[0::2] == ["80", "20", "10", "90"]
    assert texts[1::2] == ["80%", "20%", "10%", "90%"]
    assert im is not None and im.get_array().max() == pytest.approx(0.9)
    assert ax.get_xlabel() == "Predicted" and ax.get_ylabel() == "Actual"
    assert len(ax.patches) == 2  # the correct (diagonal) cells are outlined


def test_confusion_matrix_normalisations(ax: Axes) -> None:
    cm = [[30, 10], [20, 40]]
    fig, axes = plt.subplots(1, 3)
    shares = {}
    for a, mode in zip(axes, ("true", "pred", "all"), strict=True):
        viz.confusion_matrix(a, cm, normalize=mode)  # type: ignore[arg-type]
        shares[mode] = a.texts[1].get_text()
    assert shares == {"true": "75%", "pred": "60%", "all": "30%"}
    plt.close(fig)


def test_confusion_matrix_raw_counts_and_zero_rows(ax: Axes) -> None:
    viz.confusion_matrix(ax, [[0, 0], [5, 5]], normalize="true")  # empty class: no NaN/inf
    assert [t.get_text() for t in ax.texts][1] == "0%"
    fig, other = plt.subplots()
    viz.confusion_matrix(other, [[3, 1], [2, 4]], normalize=None)
    assert len(other.texts) == 4  # counts only
    plt.close(fig)


def test_confusion_matrix_validation(ax: Axes) -> None:
    with pytest.raises(ValueError, match="square"):
        viz.confusion_matrix(ax, [[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError, match="labels"):
        viz.confusion_matrix(ax, [[1, 2], [3, 4]], labels=("only one",))


def test_confusion_matrix_text_is_readable_on_every_cell(ax: Axes) -> None:
    viz.confusion_matrix(ax, [[900, 5], [50, 45]], normalize=None)
    im = ax.images[0]
    for text in ax.texts:
        row, col = round(text.get_position()[1]), round(text.get_position()[0])
        shade = to_hex(im.cmap(im.norm(im.get_array()[row, col])))
        assert _contrast(to_hex(text.get_color()), shade) >= 3


def test_calibration_has_reference_intervals_and_counts(ax: Axes) -> None:
    p = np.linspace(0.1, 0.9, 9)
    viz.calibration(
        ax, p, p * 0.9, counts=np.arange(9) + 1, lo=p * 0.9 - 0.05, hi=p * 0.9 + 0.05, ece=0.0314
    )
    assert any(ln.get_gid() == "sgc-diagonal" for ln in ax.get_lines())
    assert any(t.get_text() == "ECE 0.031" for t in ax.texts)
    assert len(ax.patches) == 9  # bin-size histogram along the floor


def test_calibration_count_mismatch_and_empty(ax: Axes) -> None:
    with pytest.raises(ValueError, match="counts"):
        viz.calibration(ax, [0.1, 0.2], [0.1, 0.3], counts=[1])
    fig, empty = plt.subplots()
    viz.calibration(empty, [], [])
    assert any(t.get_text() == "No data" for t in empty.texts)
    plt.close(fig)


def test_threshold_curves_mark_the_selected_cutoff(ax: Axes) -> None:
    t = np.linspace(0, 1, 11)
    ax.set_xlim(0, 1.3)
    viz.threshold_curves(
        ax, t, {"Precision": t, "Recall": 1 - t, "F1": t * (1 - t)}, selected=0.5, focus="F1"
    )
    assert any(a.get_text() == "threshold 0.50" for a in ax.texts)
    assert {a.get_text() for a in ax.texts} >= {"Precision", "Recall", "F1"}
    f1 = next(ln for ln in ax.get_lines() if ln.get_label() == "F1")
    assert to_hex(f1.get_color()).lower() == viz.ACCENT.lower()


def test_threshold_curves_validate_lengths(ax: Axes) -> None:
    with pytest.raises(ValueError, match="must match"):
        viz.threshold_curves(ax, [0, 1], {"x": [1, 2, 3]})


def test_dotplot_focus_intervals_and_reference(ax: Axes) -> None:
    viz.dotplot(
        ax,
        ["a", "b", "c"],
        [0.7, 0.8, 0.75],
        lo=[0.65, 0.75, 0.7],
        hi=[0.75, 0.85, 0.8],
        focus="b",
        reference=0.5,
        reference_label="chance",
    )
    assert _colours(ax) == {"a": viz.BLUE.lower(), "b": viz.ACCENT.lower(), "c": viz.BLUE.lower()}
    assert any(t.get_text() == "chance" for t in ax.texts)
    assert _rows(ax) == ["a", "b", "c"]


def test_model_comparison_sorts_and_keeps_intervals_aligned(ax: Axes) -> None:
    viz.model_comparison(
        ax,
        ["weak", "strong", "mid"],
        [0.6, 0.9, 0.75],
        lo=[0.55, 0.85, 0.7],
        hi=[0.65, 0.95, 0.8],
        focus="mid",
        baseline=0.5,
    )
    assert _rows(ax) == ["strong", "mid", "weak"]
    assert _colours(ax)["mid"] == viz.ACCENT.lower()  # the interval stays with its model
    assert any(t.get_text() == "baseline" for t in ax.texts)


def test_dotplot_empty(ax: Axes) -> None:
    viz.dotplot(ax, [], [])
    assert [t.get_text() for t in ax.texts] == ["No data"]


def test_dumbbell_pairs_and_validation(ax: Axes) -> None:
    viz.dumbbell(
        ax,
        ["x", "y"],
        [0.3, 0.5],
        [0.4, 0.45],
        a_label="before",
        b_label="after",
        value_fmt="{:.0%}",
    )
    assert ax.get_legend_handles_labels()[1] == ["before", "after"]
    assert sorted(t.get_text() for t in ax.texts) == ["30%", "40%", "45%", "50%"]
    with pytest.raises(ValueError, match="same length"):
        viz.dumbbell(ax, ["x"], [1, 2], [1, 2], a_label="a", b_label="b")


def test_dumbbell_empty_and_fit_labels_noop(ax: Axes) -> None:
    viz.dumbbell(ax, [], [], [], a_label="a", b_label="b")
    assert [t.get_text() for t in ax.texts] == ["No data"]
    before = ax.get_xlim()
    viz._fit_labels(ax, [])
    assert ax.get_xlim() == before


def test_grouped_bars(ax: Axes) -> None:
    viz.grouped_bars(
        ax, ["r", "u"], {"unweighted": [0.4, 0.5], "weighted": [0.35, 0.55]}, value_fmt="{:.0%}"
    )
    assert len(ax.patches) == 4
    assert ax.get_legend_handles_labels()[1] == ["unweighted", "weighted"]
    assert ax.get_ylim()[1] > 0.55  # headroom for labels and a legend
    with pytest.raises(ValueError, match="2 categories"):
        viz.grouped_bars(ax, ["r", "u"], {"bad": [1.0]})


def test_grouped_bars_empty_and_many_series(ax: Axes) -> None:
    viz.grouped_bars(ax, [], {})
    assert [t.get_text() for t in ax.texts] == ["No data"]
    fig, many = plt.subplots()
    viz.grouped_bars(many, ["a"], {f"s{i}": [i + 1.0] for i in range(6)}, colors=[viz.BLUE])
    assert len(many.patches) == 6
    plt.close(fig)


def test_segment_bars_benchmark_and_focus(ax: Axes) -> None:
    viz.segment_bars(ax, ["C0", "C1", "C2"], [0.6, 0.3, 0.45], benchmark=0.4, focus="C0")
    colours = [to_hex(p.get_facecolor()).lower() for p in ax.patches]
    assert colours == [viz.ACCENT.lower(), viz.CONTEXT.lower(), viz.CONTEXT.lower()]
    assert any(t.get_text() == "Overall" for t in ax.texts)
    bench = next(ln for ln in ax.get_lines() if ln.get_ydata()[0] == 0.4)
    assert to_hex(bench.get_color()).lower() == viz.INK_2.lower()  # steps back beside an accent bar


def test_segment_bars_without_focus_uses_the_accent_benchmark() -> None:
    fig, ax = plt.subplots()
    viz.segment_bars(ax, ["a", "b"], [0.2, 0.3], benchmark=0.25)
    bench = next(ln for ln in ax.get_lines() if ln.get_ydata()[0] == 0.25)
    assert to_hex(bench.get_color()).lower() == viz.ACCENT.lower()
    plt.close(fig)


def test_importance_ranks_and_accents_the_leader(ax: Axes) -> None:
    viz.importance(ax, ["a", "b", "c", "d"], [0.1, 0.5, 0.3, 0.05], top=3)
    assert _rows(ax) == ["b", "c", "a"]
    colours = [to_hex(p.get_facecolor()).lower() for p in ax.patches]
    assert colours == [viz.ACCENT.lower(), viz.CONTEXT.lower(), viz.CONTEXT.lower()]


def test_heatmap_sequential_text_contrast_and_nan(ax: Axes) -> None:
    arr = np.array([[0.0, 0.5], [1.0, np.nan]])
    im = viz.heatmap(ax, arr, ["r0", "r1"], ["c0", "c1"], fmt="{:.1f}")
    assert im is not None
    assert [t.get_text() for t in ax.texts] == ["0.0", "0.5", "1.0"]  # NaN cell left blank
    assert im.get_cmap().name == "sgc_sequential"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["c0", "c1"]


def test_heatmap_diverging_is_symmetric_about_centre(ax: Axes) -> None:
    im = viz.heatmap(ax, np.array([[-2.0, 0.0], [0.5, 1.0]]), center=0.0, fmt="{:+.1f}")
    assert im is not None
    assert im.get_clim() == (-2.0, 2.0)
    assert im.get_cmap().name == "sgc_diverging"
    assert to_rgb(im.cmap(im.norm(0.0))) == pytest.approx(to_rgb(viz.MIDPOINT), abs=0.02)
    assert [t.get_text() for t in ax.texts][1] == "0.0"  # no "+0.0"/"-0.0" noise


def test_heatmap_options_and_validation(ax: Axes) -> None:
    im = viz.heatmap(
        ax, [[1, 2], [3, 4]], cmap="sgc_accent", fmt=lambda v: f"<{v:.0f}>", colorbar=True
    )
    assert im is not None and len(ax.figure.axes) == 2  # type: ignore[union-attr]
    assert ax.texts[0].get_text() == "<1>"
    with pytest.raises(ValueError, match="2-D"):
        viz.heatmap(ax, [1, 2, 3])
    fig, empty = plt.subplots()
    assert viz.heatmap(empty, np.empty((0, 3))) is None
    assert viz.heatmap(plt.subplots()[1], [[np.nan, np.nan]]) is not None  # all-NaN is tolerated
    plt.close(fig)


def test_binned_median() -> None:
    x = np.arange(200.0)
    bx, by = viz.binned_median(x, 2 * x, bins=5)
    assert len(bx) == 5 and np.all(np.diff(bx) > 0)
    assert by == pytest.approx(2 * bx, abs=1.0)
    sx, _ = viz.binned_median(x[:5], x[:5])  # too sparse to estimate
    assert sx.size == 0
    ex, ey = viz.binned_median([], [])
    assert ex.size == ey.size == 0
    nx, _ = viz.binned_median([np.nan] * 30, np.arange(30.0))
    assert nx.size == 0


def test_shap_summary_orders_by_importance_and_keeps_sign_semantics(
    ax: Axes, data: dict[str, Any]
) -> None:
    rng = np.random.default_rng(0)
    sv = np.column_stack(
        [rng.uniform(0.2, 1.0, 300), rng.normal(0, 0.1, 300), rng.normal(0, 2.0, 300)]
    )
    x = rng.normal(size=sv.shape)
    viz.shap_summary(ax, sv, x, ["small", "tiny", "big"], max_display=3)
    assert _rows(ax) == ["big", "small", "tiny"]
    points = ax.collections[1].get_offsets()  # the "small" feature: all contributions positive
    assert np.all(points[:, 0] > 0)
    assert any(ln.get_xdata()[0] == 0.0 for ln in ax.get_lines() if len(ln.get_xdata()) == 2)
    assert any("raises" in t.get_text() for t in ax.texts)
    assert len(ax.figure.axes) == 2  # colourbar  # type: ignore[union-attr]


def test_shap_summary_is_deterministic_and_handles_constants(ax: Axes) -> None:
    sv = np.random.default_rng(1).normal(size=(60, 2))
    x = np.column_stack([np.ones(60), np.where(np.arange(60) % 7 == 0, np.nan, np.arange(60.0))])
    viz.shap_summary(ax, sv, x, ["const", "gappy"], colorbar=False, direction=None)
    fig, other = plt.subplots()
    viz.shap_summary(other, sv, x, ["const", "gappy"], colorbar=False, direction=None)
    assert np.array_equal(ax.collections[0].get_offsets(), other.collections[0].get_offsets())
    scatters = [c for c in other.collections if isinstance(c, PathCollection)]
    assert len(scatters) == 3  # constant, scaled, and grey missing-value points
    plt.close(fig)


def test_shap_summary_validation_and_empty(ax: Axes) -> None:
    with pytest.raises(ValueError, match="equal"):
        viz.shap_summary(ax, np.zeros((3, 2)), np.zeros((3, 3)), ["a", "b"])
    with pytest.raises(ValueError, match="names"):
        viz.shap_summary(ax, np.zeros((3, 2)), np.zeros((3, 2)), ["a"])
    viz.shap_summary(ax, np.empty((0, 2)), np.empty((0, 2)), ["a", "b"])
    assert any(t.get_text() == "No data" for t in ax.texts)


def test_shap_dependence_with_trend_and_colour(ax: Axes) -> None:
    x = np.linspace(0, 1, 200)
    viz.shap_dependence(ax, x, 2 * x - 1, color_by=x[::-1], feature="income", color_label="size")
    assert ax.get_xlabel() == "income" and ax.get_ylabel() == "SHAP value"
    assert ax.get_legend_handles_labels()[1] == ["binned median"]
    assert len(ax.figure.axes) == 2  # type: ignore[union-attr]
    assert any(ln.get_ydata()[0] == 0.0 for ln in ax.get_lines() if len(ln.get_ydata()) == 2)


def test_shap_dependence_plain_and_empty(ax: Axes) -> None:
    viz.shap_dependence(ax, [1, 2, 3], [0.1, 0.2, 0.3], trend=False)
    assert not ax.get_legend_handles_labels()[0]
    fig, empty = plt.subplots()
    viz.shap_dependence(empty, [], [])
    assert any(t.get_text() == "No data" for t in empty.texts)
    plt.close(fig)


def test_k_sweep_highlights_the_chosen_k(ax: Axes) -> None:
    viz.k_sweep(
        ax,
        [2, 3, 4, 5],
        [0.3, 0.27, 0.26, 0.2],
        chosen=4,
        threshold=0.25,
        threshold_label="weak",
        ylabel="silhouette",
    )
    assert list(ax.get_xticks()) == [2, 3, 4, 5]
    assert {t.get_text() for t in ax.texts} >= {"weak", "k = 4"}
    assert ax.get_ylabel() == "silhouette"


def test_k_sweep_chosen_missing_and_empty(ax: Axes) -> None:
    viz.k_sweep(ax, [2, 3], [1.0, 0.5], chosen=9)
    assert not any("k = 9" in t.get_text() for t in ax.texts)
    fig, empty = plt.subplots()
    viz.k_sweep(empty, [], [])
    assert any(t.get_text() == "No data" for t in empty.texts)
    plt.close(fig)


def test_silhouette_knife_blades_and_mean(ax: Axes) -> None:
    rng = np.random.default_rng(2)
    scores = np.concatenate([rng.uniform(0.2, 0.6, 50), rng.uniform(-0.1, 0.3, 30)])
    labels = np.repeat([0, 1], [50, 30])
    viz.silhouette_knife(ax, scores, labels, focus=[1])
    assert sorted(_rows(ax)) == ["C0", "C1"]
    assert len(ax.collections) == 2
    assert any(f"mean {scores.mean():.2f}" == t.get_text() for t in ax.texts)
    with pytest.raises(ValueError, match="same shape"):
        viz.silhouette_knife(ax, [0.1, 0.2], [0])


def test_silhouette_knife_ignores_nan_and_empty(ax: Axes) -> None:
    viz.silhouette_knife(ax, [np.nan, np.nan], [0, 1])
    assert any(t.get_text() == "No data" for t in ax.texts)


def test_cluster_profile_is_a_symmetric_diverging_heatmap(ax: Axes) -> None:
    im = viz.cluster_profile(ax, [[1.5, -0.5], [-0.2, 0.3]], ["f1", "f2"], ["C0", "C1"])
    assert im is not None and im.get_clim() == (-1.5, 1.5)
    assert [t.get_text() for t in ax.texts][:2] == ["+1.5", "\u22120.5"]


def test_pred_vs_actual_reports_honest_statistics(ax: Axes) -> None:
    rng = np.random.default_rng(4)
    truth = rng.normal(size=300)
    pred = truth * 0.8 + rng.normal(0, 0.3, 300)
    stats = viz.pred_vs_actual(ax, truth, pred)
    sse = float(np.sum((truth - pred) ** 2))
    assert stats["n"] == 300
    assert stats["rmse"] == pytest.approx(np.sqrt(sse / 300))
    assert stats["r2"] == pytest.approx(1 - sse / np.sum((truth - truth.mean()) ** 2))
    assert ax.get_aspect() == 1.0
    assert ax.get_xlim() == ax.get_ylim()
    assert any("R²" in t.get_text() and "n = 300" in t.get_text() for t in ax.texts)


def test_pred_vs_actual_negative_r2_constant_and_dirty_data(ax: Axes) -> None:
    stats = viz.pred_vs_actual(ax, [1, 2, 3, 4], [4, 3, 2, np.nan])
    assert stats["n"] == 3  # non-finite pairs dropped
    assert stats["r2"] < 0  # worse than predicting the mean is reported, not clipped
    fig, flat = plt.subplots()
    constant = viz.pred_vs_actual(flat, [2, 2, 2], [1, 2, 3])
    assert np.isnan(constant["r2"])
    assert not any("R²" in t.get_text() for t in flat.texts)
    empty = viz.pred_vs_actual(plt.subplots()[1], [], [])
    assert empty["n"] == 0 and np.isnan(empty["rmse"])
    plt.close(fig)


def test_pred_vs_actual_uses_hexbins_for_dense_data(ax: Axes) -> None:
    x = np.random.default_rng(5).normal(size=3000)
    viz.pred_vs_actual(ax, x, x + 0.1)
    assert len(ax.collections) == 1 and ax.collections[0].get_array() is not None
    with pytest.raises(ValueError, match="same shape"):
        viz.pred_vs_actual(ax, [1, 2], [1])


def test_residual_vs_fitted_keeps_statistical_references(ax: Axes) -> None:
    rng = np.random.default_rng(6)
    fitted = rng.normal(size=400)
    resid = rng.normal(0, 0.5, 400)
    viz.residual_vs_fitted(ax, fitted, resid)
    levels = sorted(ln.get_ydata()[0] for ln in ax.get_lines() if len(ln.get_ydata()) == 2)
    sd = resid.std(ddof=1)
    assert levels == pytest.approx([-2 * sd, 0.0, 2 * sd])
    assert any("SD" in t.get_text() for t in ax.texts)
    assert ax.get_xlabel() == "Fitted value"


def test_residual_vs_fitted_degenerate_inputs(ax: Axes) -> None:
    viz.residual_vs_fitted(ax, [1.0], [0.5])
    fig, empty = plt.subplots()
    viz.residual_vs_fitted(empty, [], [])
    assert any(t.get_text() == "No data" for t in empty.texts)
    plt.close(fig)


def test_qq_plot_normal_data_hugs_the_reference(ax: Axes) -> None:
    resid = np.random.default_rng(7).normal(size=1000)
    viz.qq_plot(ax, resid)
    pts = ax.collections[0].get_offsets()
    assert np.corrcoef(pts[:, 0], pts[:, 1])[0, 1] > 0.995
    assert ax.get_aspect() == 1.0
    fig, heavy = plt.subplots()
    viz.qq_plot(heavy, np.random.default_rng(7).standard_t(2, 1000))
    p = heavy.collections[0].get_offsets()
    assert abs(p[-1, 1]) > abs(p[-1, 0])  # heavy tail overshoots the normal quantile
    plt.close(fig)


def test_qq_plot_constant_residuals(ax: Axes) -> None:
    viz.qq_plot(ax, [1.0, 1.0, 1.0])
    assert any(t.get_text() == "Not enough variation" for t in ax.texts)


def test_distribution_median_and_reference(ax: Axes) -> None:
    v = np.r_[np.arange(100.0), np.nan, np.inf]
    viz.distribution(ax, v, bins=10, reference=25, reference_label="target")
    assert len(ax.patches) == 10
    labels = {t.get_text() for t in ax.texts}
    assert labels == {"median 49.5", "target"}
    assert ax.get_ylabel() == "Count"


def test_distribution_weighted_and_density(ax: Axes) -> None:
    viz.distribution(ax, [1, 2, 3, 10], weights=[1, 1, 1, 100], bins=4, median=True)
    assert any(t.get_text() == "median 10" for t in ax.texts)  # weighted median
    assert ax.get_ylabel() == "Weighted count"
    fig, dens = plt.subplots()
    viz.distribution(dens, [1, 2, 3], density=True, median=False)
    assert dens.get_ylabel() == "Density"
    with pytest.raises(ValueError, match="weights"):
        viz.distribution(dens, [1, 2], weights=[1])
    viz.distribution(plt.subplots()[1], [])
    plt.close(fig)


def test_coef_plot_colours_only_what_the_interval_supports(ax: Axes) -> None:
    viz.coef_plot(
        ax,
        ["pos", "neg", "flat"],
        [0.3, -0.2, 0.05],
        [0.1, -0.35, -0.1],
        [0.5, -0.05, 0.2],
        zero_label="no effect",
    )
    assert _colours(ax) == {
        "pos": viz.ACCENT.lower(),
        "neg": viz.BLUE.lower(),
        "flat": viz.CONTEXT.lower(),
    }
    assert any(t.get_text() == "no effect" for t in ax.texts)
    with pytest.raises(ValueError, match="same length"):
        viz.coef_plot(ax, ["a"], [1, 2], [0], [1])


def test_curve_with_band(ax: Axes) -> None:
    x = np.linspace(0, 1, 10)
    lines = viz.curve(ax, x, x, lo=x - 0.1, hi=x + 0.1, label="model", color=viz.ACCENT)
    assert len(lines) == 1 and len(ax.collections) == 1
    assert viz.curve(plt.subplots()[1], [], []) == []


def test_curve_drops_non_finite_points_with_their_band(ax: Axes) -> None:
    x = np.array([0.0, 1.0, np.nan, 3.0])
    viz.curve(ax, x, [0, 1, 2, 3], lo=[-1, 0, 1, 2], hi=[1, 2, 3, 4])
    assert len(ax.get_lines()[0].get_xdata()) == 3


def test_capture_curves_reference_ceiling_and_budget(ax: Axes) -> None:
    f = np.linspace(0, 1, 101)
    viz.capture_curves(
        ax,
        f,
        {"Model": np.minimum(1, f / 0.4) ** 0.9, "Rule": f**0.8},
        focus="Model",
        prevalence=0.4,
        budget=0.25,
    )
    names = ax.get_legend_handles_labels()[1]
    assert names == ["Random contact", "Ceiling", "Model", "Rule"]
    model = next(ln for ln in ax.get_lines() if ln.get_label() == "Model")
    assert to_hex(model.get_color()).lower() == viz.ACCENT.lower()
    texts = {t.get_text() for t in ax.texts}
    assert "budget 25%" in texts
    expected = f"{float(np.interp(0.25, f, np.minimum(1, f / 0.4) ** 0.9)):.0%}"
    assert expected in texts
    assert ax.yaxis.get_major_formatter()(0.5) == "50%"
    assert (ax.get_xlim(), ax.get_ylim()[0]) == ((0, 1), 0)


def test_capture_curves_validation_and_empty(ax: Axes) -> None:
    with pytest.raises(ValueError, match="must match"):
        viz.capture_curves(ax, [0, 1], {"x": [0, 0.5, 1]})
    fig, empty = plt.subplots()
    viz.capture_curves(empty, [], {})
    assert any(t.get_text() == "No data" for t in empty.texts)
    plain = plt.subplots()[1]
    viz.capture_curves(plain, [0, 0.5, 1], {"only": [0, 0.7, 1]}, random=False)
    assert plain.get_legend_handles_labels()[1] == ["only"]
    plt.close(fig)


def test_tornado_sorts_by_swing_and_labels_the_base(ax: Axes) -> None:
    viz.tornado(
        ax,
        ["small", "large", "mid"],
        [9.5, 5.0, 8.0],
        [10.5, 15.0, 12.0],
        base=10.0,
    )
    assert _rows(ax) == ["large", "mid", "small"]
    assert any(t.get_text() == "base 10.0" for t in ax.texts)
    assert ax.get_legend_handles_labels()[1] == ["Low case", "High case"]
    assert len(ax.patches) == 6


def test_tornado_edge_cases(ax: Axes) -> None:
    viz.tornado(ax, ["flat"], [10.0], [10.0], base=10.0)  # zero swing: no value labels
    assert [t.get_text() for t in ax.texts] == ["base 10.0"]
    with pytest.raises(ValueError, match="same length"):
        viz.tornado(ax, ["a"], [1, 2], [3], 2.0)
    fig, empty = plt.subplots()
    viz.tornado(empty, [], [], [], 0.0)
    assert any(t.get_text() == "No data" for t in empty.texts)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Edge cases across the toolkit
# --------------------------------------------------------------------------------------
EMPTY_CALLS: dict[str, Callable[[Axes], Any]] = {
    "barh": lambda a: viz.barh(a, [], []),
    "dotplot": lambda a: viz.dotplot(a, [], []),
    "grouped_bars": lambda a: viz.grouped_bars(a, [], {}),
    "segment_bars": lambda a: viz.segment_bars(a, [], []),
    "roc": lambda a: viz.roc(a, [], []),
    "pr_curve": lambda a: viz.pr_curve(a, [], []),
    "calibration": lambda a: viz.calibration(a, [], []),
    "threshold_curves": lambda a: viz.threshold_curves(a, [], {}),
    "confusion_matrix": lambda a: viz.confusion_matrix(a, np.empty((0, 0)), labels=()),
    "heatmap": lambda a: viz.heatmap(a, np.empty((0, 0))),
    "shap_summary": lambda a: viz.shap_summary(a, np.empty((0, 1)), np.empty((0, 1)), ["f"]),
    "shap_dependence": lambda a: viz.shap_dependence(a, [], []),
    "k_sweep": lambda a: viz.k_sweep(a, [], []),
    "silhouette_knife": lambda a: viz.silhouette_knife(a, [], []),
    "pred_vs_actual": lambda a: viz.pred_vs_actual(a, [], []),
    "residual_vs_fitted": lambda a: viz.residual_vs_fitted(a, [], []),
    "qq_plot": lambda a: viz.qq_plot(a, []),
    "distribution": lambda a: viz.distribution(a, []),
    "curve": lambda a: viz.curve(a, [], []),
    "capture_curves": lambda a: viz.capture_curves(a, [], {}),
    "tornado": lambda a: viz.tornado(a, [], [], [], 0.0),
}


@pytest.mark.parametrize("name", sorted(EMPTY_CALLS))
def test_empty_inputs_render_a_placeholder_instead_of_failing(name: str) -> None:
    fig, axes = plt.subplots()
    EMPTY_CALLS[name](axes)
    fig.canvas.draw()
    assert any(t.get_text() in {"No data", "Not enough variation"} for t in axes.texts)
    plt.close(fig)


def test_single_point_inputs_render() -> None:
    viz.use()
    fig, axes = plt.subplots(2, 3, figsize=(9, 5), layout="constrained")
    a = axes.ravel()
    viz.dotplot(a[0], ["only"], [0.5], lo=[0.4], hi=[0.6])
    viz.segment_bars(a[1], ["only"], [0.5], benchmark=0.5)
    viz.k_sweep(a[2], [3], [0.1], chosen=3)
    viz.distribution(a[3], [1.0])
    viz.heatmap(a[4], [[1.0]])
    viz.roc(a[5], [0.0], [0.0])
    fig.canvas.draw()
    plt.close(fig)


def test_headless_backend() -> None:
    assert matplotlib.get_backend().lower() == "agg"


# --------------------------------------------------------------------------------------
# Rendering and export
# --------------------------------------------------------------------------------------
def test_full_toolkit_renders_without_warnings(data: dict[str, Any]) -> None:
    viz.use()
    fig, axes = plt.subplots(2, 3, figsize=viz.figsize("wide", 0.7), layout="constrained")
    a = axes.ravel()
    viz.roc(a[0], data["fpr"], data["tpr"], label="m")
    viz.legend(a[0], loc="lower right")
    viz.confusion_matrix(a[1], [[50, 10], [5, 35]])
    viz.calibration(a[2], [0.2, 0.5, 0.8], [0.25, 0.45, 0.85], counts=[10, 20, 10])
    viz.shap_summary(a[3], data["X"] * 0.3, data["X"], data["names"])
    viz.pred_vs_actual(a[4], data["X"][:, 0], data["X"][:, 0] * 0.9)
    viz.tornado(a[5], ["a", "b"], [1.0, 2.0], [3.0, 2.5], 2.0)
    fig.canvas.draw()
    plt.close(fig)


def test_figsize_helper() -> None:
    assert viz.figsize("col", 1.0) == (viz.COL, viz.COL)
    assert viz.figsize(5, 0.5) == (5.0, 2.5)
    assert viz.figsize()[0] == viz.WIDE


@pytest.mark.parametrize("suffix", ["png", "pdf", "svg"])
def test_save_exports_every_format(tmp_path: Path, suffix: str) -> None:
    viz.use()
    fig, ax = plt.subplots(figsize=(3, 2))
    viz.barh(ax, ["a", "b"], [1.0, 2.0])
    viz.title(ax, "Export check")
    path = viz.save(fig, tmp_path / "nested" / f"fig.{suffix}", preset="report")
    assert path.exists() and path.stat().st_size > 500
    plt.close(fig)


def test_save_presets_set_resolution_and_surface(tmp_path: Path) -> None:
    viz.use()
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot([0, 1], [0, 1])
    sizes = {}
    for preset in ("notebook", "report", "presentation"):
        p = viz.save(fig, tmp_path / f"{preset}.png", preset=preset)
        with Image.open(p) as im:
            sizes[preset] = im.size
            assert im.convert("RGB").getpixel((1, 1)) == tuple(
                round(c * 255) for c in to_rgb(viz.SURFACE)
            )
    assert sizes["report"][0] > sizes["presentation"][0] > sizes["notebook"][0]
    plain = viz.save(fig, tmp_path / "default.png", dpi=50)  # kwargs reach savefig
    with Image.open(plain) as im:
        assert im.size[0] < sizes["notebook"][0]
    plt.close(fig)


def test_serif_title_renders_in_headless_mode(tmp_path: Path) -> None:
    viz.use(bundled_fonts=True)
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.plot([0, 1], [0, 1])
    viz.title(ax, "AUC 0.847 → 0.861", "Out-of-fold, ₹ figures")
    viz.save(fig, tmp_path / "t.png")
    assert (tmp_path / "t.png").stat().st_size > 1000
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Existing callers: notebooks and the report-figure script must keep working
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]


def _is_viz(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "viz"


def _viz_nodes(source: str) -> Iterator[ast.Attribute]:
    """Every ``viz.<name>`` attribute access in a code cell (IPython magics skipped)."""
    code = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith(("%", "!"))
    )
    try:
        tree = ast.parse(code)
    except SyntaxError:  # cell relying on IPython-only syntax
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and _is_viz(node.value):
            yield node


def _viz_calls(source: str) -> Iterator[ast.Call]:
    code = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith(("%", "!"))
    )
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and _is_viz(node.func.value)
        ):
            yield node


def _notebook_code() -> Iterator[tuple[str, str]]:
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
            if cell["cell_type"] == "code":
                yield path.name, "".join(cell["source"])


def test_notebooks_use_only_existing_viz_names_and_valid_keywords() -> None:
    calls = 0
    for name, source in _notebook_code():
        for call in _viz_calls(source):
            assert isinstance(call.func, ast.Attribute)
            fn = getattr(viz, call.func.attr, None)
            assert fn is not None, f"{name}: viz.{call.func.attr} no longer exists"
            if any(isinstance(a, ast.Starred) for a in call.args) or any(
                k.arg is None for k in call.keywords
            ):
                continue
            sig = inspect.signature(fn)
            try:
                sig.bind(*[None] * len(call.args), **{k.arg: None for k in call.keywords if k.arg})
            except TypeError as err:
                pytest.fail(f"{name}: viz.{call.func.attr} no longer accepts this call: {err}")
            calls += 1
    assert calls > 20 or not (ROOT / "notebooks").exists()


def test_notebook_attribute_access_exists() -> None:
    seen = {node.attr for _, source in _notebook_code() for node in _viz_nodes(source)}
    assert {"use", "barh", "strip", "SEQUENTIAL", "DIVERGING", "CATEGORICAL"} <= seen
    missing = sorted(name for name in seen if not hasattr(viz, name))
    assert not missing, f"notebooks use viz names that no longer exist: {missing}"


def test_report_figure_script_imports_still_resolve() -> None:
    script = (ROOT / "project" / "figures" / "make_figures.py").read_text(encoding="utf-8")
    tree = ast.parse(script)
    wanted = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "savings_goal.viz"
        for alias in node.names
    ]
    assert wanted
    assert all(hasattr(viz, name) for name in wanted)


def test_report_figure_script_regenerates_all_four_figures(tmp_path: Path) -> None:
    script = ROOT / "project" / "figures" / "make_figures.py"
    if not (ROOT / "results" / "business.json").exists():
        pytest.skip("committed results/ not available")
    redirect = "OUT = Path(__file__).resolve().parent"
    source = script.read_text(encoding="utf-8")
    assert source.count(redirect) == 1
    source = source.replace(redirect, f"OUT = Path({str(tmp_path)!r})")
    exec(compile(source, str(script), "exec"), {"__file__": str(script), "__name__": "__main__"})
    assert sorted(p.name for p in tmp_path.glob("fig*.png")) == [
        "fig1_feature_ladder.png",
        "fig2_shap.png",
        "fig3_business.png",
        "fig4_personas.png",
    ]
