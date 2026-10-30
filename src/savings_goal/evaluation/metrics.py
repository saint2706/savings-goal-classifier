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


def top_by_budget(
    score: Array, budget: float, weights: Array | None = None
) -> tuple[NDArray[np.intp], Array]:
    """Indices of the highest-scoring households that fit in ``budget``, and the weights used.

    Unweighted, the budget is a share of rows. With ``weights`` it is a share of the
    weighted population: households are taken in score order until their weights
    reach ``budget`` of the total. Ties in ``score`` are broken by a stable sort,
    so pass a continuous score.
    """
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    if weights is None:
        return order[: round(len(order) * budget)], np.ones(len(order))
    w = np.asarray(weights, dtype=float)
    cum = np.cumsum(w[order])
    return order[: int(np.searchsorted(cum, budget * cum[-1], side="right"))], w


def capture_at_budget(
    at_risk: Array, score: Array, budget: float, weights: Array | None = None
) -> tuple[float, float]:
    """Contact the top ``budget`` fraction by ``score``: (recall of at-risk, precision).

    Recall is the share of all at-risk households that were contacted; precision is
    the share of contacted households that are at risk. With ``weights`` both are
    survey-weighted (population estimates); the model that produced ``score`` is
    still fitted unweighted.
    """
    at_risk = np.asarray(at_risk, dtype=float)
    top, w = top_by_budget(score, budget, weights)
    if len(top) == 0:
        return 0.0, float("nan")
    captured = np.float64((at_risk[top] * w[top]).sum())
    return float(captured / np.float64((at_risk * w).sum())), float(
        captured / np.float64(w[top].sum())
    )


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
    prevalence = np.float64((at_risk * w).sum()) / np.float64(w.sum())
    rows = []
    for b in budgets:
        ceiling = np.float64(min(1.0, np.float64(b) / prevalence))
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


def targeting_table(
    at_risk: Array,
    scores: Mapping[str, Array],
    budgets: Sequence[float] = (0.10, 0.25, 0.50, 0.65),
    weights: Array | None = None,
) -> pd.DataFrame:
    """Recall, precision and ceiling capture as three separate columns.

    ``ceiling`` is the best recall any strategy can reach, min(1, budget / prevalence).
    While ``budget`` is below the at-risk prevalence, ``ceiling_capture`` is
    recall / ceiling = hits / (budget x N), which is ``precision`` up to the rounding
    of the contact count: the two are the same number there and only diverge once the
    ceiling saturates at 1.
    Reporting recall (how many at-risk households are reached), precision (how few
    contacts are wasted) and ceiling capture (how close to perfect ranking) side by
    side makes that explicit instead of presenting one number under two names.
    """
    t = capture_vs_ceiling(at_risk, scores, budgets, weights)
    t = t.rename(columns={"capture": "recall", "pct_of_ceiling": "ceiling_capture"})
    return t[["strategy", "budget", "recall", "precision", "ceiling", "ceiling_capture"]]


def operating_point(
    at_risk: Array, risk_score: Array, threshold: float, weights: Array | None = None
) -> dict[str, float]:
    """Contact everyone with ``risk_score >= threshold``: contacted share, recall, precision.

    This is the threshold view (a deployed classifier), as opposed to the budget view
    of :func:`targeting_table`; the two coincide when the budget equals ``contacted``.
    """
    at_risk = np.asarray(at_risk, dtype=float)
    w = np.ones_like(at_risk) if weights is None else np.asarray(weights, dtype=float)
    sel = np.asarray(risk_score, dtype=float) >= threshold
    hits = np.float64((at_risk[sel] * w[sel]).sum())
    return {
        "contacted": float(np.float64(w[sel].sum()) / np.float64(w.sum())),
        "recall": float(hits / np.float64((at_risk * w).sum())),
        "precision": float(hits / np.float64(w[sel].sum())) if sel.any() else float("nan"),
    }


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
    return float(np.float64((x[m] * w[m]).sum()) / np.float64(w[m].sum()))
