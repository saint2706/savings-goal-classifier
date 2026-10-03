"""Robust / quantile regression on the continuous savings rate.

The binary label throws away how far a household is from the benchmark. Here
the conditional *median* savings rate is modelled instead (robust to the long
left tail of under-reported incomes), and households are ranked by predicted
rupee shortfall: ``max(0, threshold - q50) * INCOME``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline, make_pipeline

from savings_goal.config import SEED, THRESHOLD
from savings_goal.models.pipeline import preprocessor


def quantile_model(
    numeric: list[str], categorical: list[str] | None = None, quantile: float = 0.5
) -> Pipeline:
    return make_pipeline(
        preprocessor(numeric, categorical),
        HistGradientBoostingRegressor(
            loss="quantile", quantile=quantile, max_iter=300, learning_rate=0.08, random_state=SEED
        ),
    )


def rupee_shortfall(
    pred_rate: NDArray[np.float64], income: NDArray[np.float64], threshold: float = THRESHOLD
) -> NDArray[np.float64]:
    return np.clip(threshold - pred_rate, 0, None) * income


def pinball_loss(y: NDArray[Any], q: NDArray[np.float64], tau: float = 0.5) -> float:
    diff = np.asarray(y, dtype=float) - q
    return float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))


def decile_of(x: pd.Series) -> pd.Series:
    return pd.qcut(x, 10, labels=False, duplicates="drop")
