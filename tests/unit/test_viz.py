from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from savings_goal import viz


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
