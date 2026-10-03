"""Transformers and compositional-data helpers used inside model pipelines."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin

FloatArray = NDArray[np.float64]


class QuantileClipper(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Winsorise each column at quantiles learned on the training fold only.

    Replaces the full-sample ``clip(upper=p99)`` that used to run before the
    train/test split. NaNs pass through untouched (impute afterwards).
    """

    def __init__(self, lower: float | None = None, upper: float | None = 0.99) -> None:
        self.lower = lower
        self.upper = upper

    def fit(self, X: Any, y: Any = None) -> QuantileClipper:
        A = np.asarray(X, dtype=float)
        self.n_features_in_ = A.shape[1]
        self.lower_ = (
            np.nanquantile(A, self.lower, axis=0)
            if self.lower is not None
            else np.full(A.shape[1], -np.inf)
        )
        self.upper_ = (
            np.nanquantile(A, self.upper, axis=0)
            if self.upper is not None
            else np.full(A.shape[1], np.inf)
        )
        return self

    def transform(self, X: Any) -> FloatArray:
        A = np.asarray(X, dtype=float)
        return np.clip(A, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features: Any = None) -> NDArray[np.object_]:
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        return np.asarray(input_features, dtype=object)


def zero_replacement_deltas(X: pd.DataFrame, factor: float = 0.65) -> FloatArray:
    """Per-part delta = ``factor`` x the smallest observed positive value."""
    return np.array([X[c][X[c] > 0].min() * factor for c in X.columns], dtype=float)


def multiplicative_replacement(X: pd.DataFrame | FloatArray, deltas: FloatArray) -> FloatArray:
    """Martin-Fernandez multiplicative zero replacement on a (sub)composition.

    Rows are first re-closed to sum to 1; zeros become ``delta`` and the
    non-zero parts shrink by the mass handed out, so rows still sum to 1.
    Rows with no positive part come back as NaN.
    """
    A = np.asarray(X, dtype=float).copy()
    row_sum = A.sum(axis=1, keepdims=True)
    A = np.divide(A, row_sum, out=np.full_like(A, np.nan), where=row_sum > 0)
    zeros = A == 0
    wide = np.broadcast_to(deltas, A.shape)
    lost = (zeros * wide).sum(axis=1, keepdims=True)
    return np.where(zeros, wide, A * (1 - lost))


def clr(closed: FloatArray) -> FloatArray:
    """Centred log-ratio. Rows sum to zero, so the columns are exactly singular."""
    logs = np.log(closed)
    return logs - logs.mean(axis=1, keepdims=True)


def helmert_basis(D: int) -> FloatArray:
    """Orthonormal D x (D-1) contrast matrix orthogonal to the 1-vector."""
    V = np.zeros((D, D - 1))
    for i in range(D - 1):
        V[: i + 1, i] = 1.0 / (i + 1)
        V[i + 1, i] = -1.0
        V[:, i] *= np.sqrt((i + 1) / (i + 2))
    return V


def ilr(closed: FloatArray) -> FloatArray:
    """Isometric log-ratio coordinates: full-rank, and Euclidean = Aitchison distance."""
    return clr(closed) @ helmert_basis(closed.shape[1])
