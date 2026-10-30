"""Outreach what-if: what a contact budget buys under explicit cost and response assumptions.

IHDS-II is observational and records no outreach, so nothing here is identified from
data: who *responds* to a nudge is an assumption, stated per scenario. What the survey
does supply is the household's actual savings rate, hence how far it sits below the
benchmark. The simulation combines the two:

* a contacted household responds with probability ``r_i`` (a :class:`Scenario`);
* a responder raises its savings rate by ``delta`` (a share of income, e.g. 0.05);
* the household is moved to "on track" only if it was within ``delta`` of the
  benchmark, so the poorest households are at risk but out of reach;
* every contact costs ``cost`` rupees.

Strategies differ only in the ranking they contact in. A risk ranking and a
reachability ranking pick different households, which is the point of the exercise:
it shows how much the value of a targeting rule depends on an assumption the data
cannot test. All outputs are expected values, not draws.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from savings_goal.config import THRESHOLD
from savings_goal.evaluation.metrics import top_by_budget

Array = NDArray[Any]


@dataclass(frozen=True)
class Scenario:
    """A named assumption about P(respond | contacted) for each household."""

    name: str
    response: Callable[[pd.DataFrame], Array]


def uniform(r: float = 0.20) -> Scenario:
    """Everyone responds alike; the implicit assumption of a pure risk ranking."""
    return Scenario(f"uniform r={r:.2f}", lambda df: np.full(len(df), r))


def constrained_poor(lo: float = 0.05, hi: float = 0.35) -> Scenario:
    """Response rises with income rank: access constraints bind the poor (Dupas-Robinson)."""

    def f(df: pd.DataFrame) -> Array:
        pct = df["INCOME"].rank(pct=True).to_numpy()
        return lo + (hi - lo) * pct

    return Scenario(f"income-gradient {lo:.2f}-{hi:.2f}", f)


def debt_burdened(no_debt: float = 0.30, debt: float = 0.10) -> Scenario:
    """Indebted households divert extra income to repayment, so respond less."""

    def f(df: pd.DataFrame) -> Array:
        return np.where(df["Has_Debt"].to_numpy() == 1, debt, no_debt)

    return Scenario(f"debt-burdened {no_debt:.2f}/{debt:.2f}", f)


SCENARIOS: tuple[Scenario, ...] = (uniform(), constrained_poor(), debt_burdened())


def reachable(df: pd.DataFrame, delta: float, threshold: float = THRESHOLD) -> Array:
    """True where a ``delta`` rise in savings rate would lift the household over the benchmark."""
    sr = df["Savings_Rate"].to_numpy(dtype=float)
    return (sr < threshold) & (sr >= threshold - delta)


def simulate(
    df: pd.DataFrame,
    scores: Mapping[str, Array],
    scenario: Scenario,
    budgets: Sequence[float],
    delta: float,
    cost: float,
    weights: Array | None = None,
) -> pd.DataFrame:
    """Expected outcome of contacting the top ``budget`` of each ranking.

    Columns: contacts; flips (expected households moved to on track) per 1,000
    contacts; extra savings per contact (Rs/yr); cost per household moved (Rs).
    With ``weights`` the figures describe the survey population, not the sample.
    """
    r = scenario.response(df)
    flip = r * reachable(df, delta)
    extra = r * delta * df["INCOME"].to_numpy(dtype=float)
    rows = []
    for name, s in scores.items():
        for b in budgets:
            top, w = top_by_budget(s, b, weights)
            contacts = float(w[top].sum())
            flips = float((flip[top] * w[top]).sum())
            rows.append(
                {
                    "scenario": scenario.name,
                    "strategy": name,
                    "budget": b,
                    "contacts": contacts,
                    "flips_per_1000": 1000 * flips / contacts,
                    "extra_savings_per_contact": float((extra[top] * w[top]).sum()) / contacts,
                    "cost_per_flip": cost * contacts / flips if flips > 0 else float("inf"),
                }
            )
    return pd.DataFrame(rows)


def flips_per_1000(
    df: pd.DataFrame,
    score: Array,
    scenario: Scenario,
    budget: float,
    delta: float,
    idx: Array | None = None,
) -> float:
    """Expected households moved per 1,000 contacts for one ranking (for bootstrap)."""
    sub = df if idx is None else df.iloc[idx]
    sc = np.asarray(score)[np.arange(len(df)) if idx is None else idx]
    flip = scenario.response(sub) * reachable(sub, delta)
    top, _ = top_by_budget(sc, budget)
    return float(1000 * flip[top].mean()) if len(top) else float("nan")


def reach_score(p_on_track_at_relaxed: Array, p_on_track_at_benchmark: Array) -> Array:
    """P(0.20 - delta <= savings rate < 0.20): the model's estimate of being within reach.

    Both inputs are out-of-fold P(on track) from models trained on the benchmark and on
    the benchmark relaxed by ``delta``; their difference is the probability mass in the
    band a nudge could carry a household across.
    """
    out: Array = np.asarray(p_on_track_at_relaxed) - np.asarray(p_on_track_at_benchmark)
    return out
