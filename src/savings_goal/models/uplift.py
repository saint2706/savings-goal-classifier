"""T-learner uplift estimator, ready for intervention data IHDS cannot supply.

IHDS-II is observational and records no outreach, nudge or product offer, so
nothing in this repository can estimate who *responds* to an intervention. This
module exists so that a team with randomised (or logged-propensity) campaign
data can rank households by estimated treatment effect instead of by risk.
It is exercised only on synthetic data in the test suite.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, clone


class TLearner(BaseEstimator):  # type: ignore[misc]
    """Fit separate outcome models on treated and control; uplift = p1(x) - p0(x)."""

    def __init__(self, base_estimator: Any) -> None:
        self.base_estimator = base_estimator

    def fit(self, X: Any, treatment: NDArray[Any], y: NDArray[Any]) -> TLearner:
        t = np.asarray(treatment).astype(bool)
        if t.all() or (~t).all():
            raise ValueError("need both treated and control units")
        Xa = np.asarray(X)
        ya = np.asarray(y)
        self.model_treated_ = clone(self.base_estimator).fit(Xa[t], ya[t])
        self.model_control_ = clone(self.base_estimator).fit(Xa[~t], ya[~t])
        return self

    def predict_uplift(self, X: Any) -> NDArray[np.float64]:
        Xa = np.asarray(X)
        p1 = self.model_treated_.predict_proba(Xa)[:, 1]
        p0 = self.model_control_.predict_proba(Xa)[:, 1]
        out: NDArray[np.float64] = p1 - p0
        return out


def qini_auc(uplift: NDArray[Any], treatment: NDArray[Any], y: NDArray[Any]) -> float:
    """Area between the Qini curve and the random-targeting line (higher is better)."""
    order = np.argsort(-np.asarray(uplift, dtype=float), kind="stable")
    t = np.asarray(treatment, dtype=float)[order]
    yy = np.asarray(y, dtype=float)[order]
    n_t = np.cumsum(t)
    n_c = np.cumsum(1 - t)
    y_t = np.cumsum(yy * t)
    y_c = np.cumsum(yy * (1 - t))
    with np.errstate(divide="ignore", invalid="ignore"):
        qini = y_t - np.where(n_c > 0, y_c * n_t / n_c, 0.0)
    random_line = np.linspace(0, qini[-1], len(qini))
    return float(np.mean(qini - random_line))
