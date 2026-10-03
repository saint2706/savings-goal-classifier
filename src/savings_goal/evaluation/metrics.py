"""Threshold-free metrics, targeting capture, and grouped bootstrap intervals.

Class convention: ``Goal_Met = 1`` (on track) is the positive class of every
model, prevalence ~0.319. The business acts on the *at-risk* class
(``1 - Goal_Met``, prevalence ~0.681), scored as ``1 - p``. Functions say which
class they are about in their name.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)

Array = NDArray[Any]


def expected_calibration_error(y: Array, p: Array, n_bins: int = 10) -> float:
    """Equal-width-bin ECE: sum_b (n_b / n) * |mean(y_b) - mean(p_b)|."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def probability_metrics(y: Array, p: Array) -> dict[str, float]:
    """All threshold-free metrics, labelled by class."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    return {
        "ROC-AUC": float(roc_auc_score(y, p)),
        "PR-AUC (on-track)": float(average_precision_score(y, p)),
        "PR-AUC (at-risk)": float(average_precision_score(1 - y, 1 - p)),
        "Brier": float(brier_score_loss(y, p)),
        "Log-loss": float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15))),
        "ECE": expected_calibration_error(y, p),
    }


def fold_metrics(with_proba: bool = True) -> dict[str, Callable[[Array, Array, Array], float]]:
    """Metric callables for ``cv.cross_validate_grouped`` (y_true, score, y_pred)."""
    m: dict[str, Callable[[Array, Array, Array], float]] = {
        "accuracy": lambda y, s, yp: float((y == yp).mean()),
        "F1 (macro)": lambda y, s, yp: float(f1_score(y, yp, average="macro")),
        "ROC-AUC": lambda y, s, yp: float(roc_auc_score(y, s)),
        "PR-AUC (at-risk)": lambda y, s, yp: float(average_precision_score(1 - y, -s)),
    }
    if with_proba:
        m.update(
            {
                "Brier": lambda y, s, yp: float(brier_score_loss(y, s)),
                "Log-loss": lambda y, s, yp: float(log_loss(y, np.clip(s, 1e-15, 1 - 1e-15))),
                "ECE": lambda y, s, yp: expected_calibration_error(y, s),
            }
        )
    return m


def capture_at_budget(
    at_risk: Array, score: Array, budget: float, weights: Array | None = None
) -> tuple[float, float]:
    """Contact the top ``budget`` fraction by ``score``: (share of at-risk reached, precision).

    With ``weights`` the budget is a share of the weighted population (households
    are taken in score order until their weights reach ``budget`` of the total),
    and both returned quantities are survey-weighted. Ties in ``score`` are
    broken by a stable sort, so pass a continuous score.
    """
    at_risk = np.asarray(at_risk, dtype=float)
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    if weights is None:
        w = np.ones_like(at_risk)
        n_top = round(len(order) * budget)
    else:
        w = np.asarray(weights, dtype=float)
        cum = np.cumsum(w[order])
        n_top = int(np.searchsorted(cum, budget * cum[-1], side="right"))
    top = order[:n_top]
    if n_top == 0:
        return 0.0, float("nan")
    captured = (at_risk[top] * w[top]).sum()
    return float(captured / (at_risk * w).sum()), float(captured / w[top].sum())


def capture_vs_ceiling(
    at_risk: Array,
    scores: Mapping[str, Array],
    budgets: Sequence[float] = (0.10, 0.25, 0.50, 0.65),
    weights: Array | None = None,
) -> pd.DataFrame:
    """Capture for each strategy at each budget, against the best achievable.

    The ceiling at budget b is min(1, b / prevalence): with 68% of households
    at risk, a 25% budget can reach at most 25/68 = 36.7% of them, so raw
    capture numbers look small for every strategy, perfect ones included.
    """
    at_risk = np.asarray(at_risk, dtype=float)
    w = np.ones_like(at_risk) if weights is None else np.asarray(weights, dtype=float)
    prevalence = (at_risk * w).sum() / w.sum()
    rows = []
    for b in budgets:
        ceiling = min(1.0, b / prevalence)
        for name, s in scores.items():
            cap, prec = capture_at_budget(at_risk, s, b, weights)
            rows.append(
                {
                    "strategy": name,
                    "budget": b,
                    "capture": cap,
                    "ceiling": ceiling,
                    "pct_of_ceiling": cap / ceiling,
                    "precision": prec,
                }
            )
    return pd.DataFrame(rows)


def bootstrap_indices(
    groups: Array, n_boot: int = 1000, seed: int = 42
) -> Iterator[NDArray[np.intp]]:
    """Yield row indices of PSU-clustered bootstrap samples (whole groups, with replacement)."""
    groups = np.asarray(groups)
    uniq, inverse = np.unique(groups, return_inverse=True)
    rows_by_group = [np.flatnonzero(inverse == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        yield np.concatenate([rows_by_group[g] for g in pick])


def grouped_bootstrap(
    statistic: Callable[[NDArray[np.intp]], float],
    groups: Array,
    n_boot: int = 1000,
    seed: int = 42,
) -> NDArray[np.float64]:
    """The statistic on each PSU-clustered bootstrap replicate.

    ``statistic`` receives the row indices of the resampled data set.
    """
    return np.array([statistic(idx) for idx in bootstrap_indices(groups, n_boot, seed)])


def percentile_ci(draws: Array, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    lo, hi = np.nanquantile(np.asarray(draws, dtype=float), [a, 1 - a])
    return float(lo), float(hi)


def weighted_mean(x: Array, w: Array) -> float:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    m = ~np.isnan(x)
    return float((x[m] * w[m]).sum() / w[m].sum())
