"""Grouped cross-validation: households in one PSU never straddle train and test.

IHDS samples households in clusters (villages / urban blocks). Neighbours share
prices, labour markets and interviewers, so a household-level random split lets
the model see a test household's neighbours during training. Every split in the
project therefore groups by ``IDPSU``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

from savings_goal.config import N_SPLITS, SEED


def grouped_cv(n_splits: int = N_SPLITS, seed: int = SEED) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def grouped_test_mask(
    y: pd.Series, groups: pd.Series, n_splits: int = N_SPLITS, seed: int = SEED
) -> NDArray[np.bool_]:
    """Hold out one stratified-grouped fold (~1/n_splits of rows) as the test set."""
    cv = grouped_cv(n_splits, seed)
    _, test_idx = next(cv.split(np.zeros(len(y)), y, groups))
    mask = np.zeros(len(y), dtype=bool)
    mask[test_idx] = True
    return mask


def oof_predict_proba(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    cv: StratifiedGroupKFold | None = None,
) -> NDArray[np.float64]:
    """Out-of-fold P(y=1) under grouped CV."""
    cv = cv or grouped_cv()
    out = np.full(len(y), np.nan)
    for tr, te in cv.split(X, y, groups):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        out[te] = m.predict_proba(X.iloc[te])[:, 1]
    return out


def cross_validate_grouped(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    metrics: Mapping[str, Callable[[NDArray[Any], NDArray[np.float64], NDArray[Any]], float]],
    cv: StratifiedGroupKFold | None = None,
    on_fold: Callable[[int, Any, Iterable[int]], None] | None = None,
) -> pd.DataFrame:
    """Fit per fold and score with ``metric(y_true, proba, y_pred)``; one row per fold."""
    cv = cv or grouped_cv()
    rows = []
    for k, (tr, te) in enumerate(cv.split(X, y, groups)):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        yt = y.iloc[te].to_numpy()
        proba = (
            m.predict_proba(X.iloc[te])[:, 1]
            if hasattr(m, "predict_proba")
            else m.decision_function(X.iloc[te])
        )
        pred = m.predict(X.iloc[te])
        rows.append({name: fn(yt, proba, pred) for name, fn in metrics.items()})
        if on_fold is not None:
            on_fold(k, m, tr)
    return pd.DataFrame(rows)


def summarise_folds(folds: pd.DataFrame) -> pd.Series:
    """Mean and sd across folds, flattened to one Series."""
    mean = folds.mean().add_prefix("")
    sd = folds.std().add_suffix(" sd")
    return pd.concat([mean, sd])
