from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from savings_goal.config import CATEGORICALS, GROUP_COL, SHARE_COLS
from savings_goal.features.engineer import engineer, spec_from_frame, zero_rates
from savings_goal.features.transforms import (
    QuantileClipper,
    clr,
    helmert_basis,
    ilr,
    multiplicative_replacement,
    zero_replacement_deltas,
)


def test_split_is_grouped_and_stratified(features: pd.DataFrame) -> None:
    test_groups = set(features.loc[features["Is_Test"], GROUP_COL])
    train_groups = set(features.loc[~features["Is_Test"], GROUP_COL])
    assert not test_groups & train_groups
    assert 0.1 < features["Is_Test"].mean() < 0.35


def test_feature_table(features: pd.DataFrame, households: pd.DataFrame) -> None:
    spec = spec_from_frame(features)
    assert set(spec.features) <= set(features.columns)
    assert not features[CATEGORICALS].isna().any().any()
    assert "Goal_Met" in features and "Savings_Rate" in features
    assert set(spec.deployable_numeric).isdisjoint(SHARE_COLS)
    assert np.isfinite(features[spec.clr_cols].to_numpy()).all()
    assert features["Has_Debt"].isin([0, 1]).all()
    assert len(zero_rates(features)) == len(SHARE_COLS)
    _, spec2 = engineer(households)
    assert spec2.indicators == spec.indicators and spec2.deltas


def test_quantile_clipper_learns_on_fit_data_only() -> None:
    train = np.arange(100, dtype=float).reshape(-1, 1)
    clip = QuantileClipper(upper=0.9).fit(train)
    assert clip.transform(np.array([[1e6]]))[0, 0] == pytest.approx(np.quantile(train, 0.9))
    both = QuantileClipper(lower=0.1, upper=None).fit(train)
    assert both.transform(np.array([[-5.0]]))[0, 0] == pytest.approx(np.quantile(train, 0.1))
    assert list(clip.get_feature_names_out(["a"])) == ["a"]
    assert list(clip.get_feature_names_out()) == ["x0"]


def test_compositional_helpers() -> None:
    X = pd.DataFrame({"a": [0.5, 0.0, 0.2], "b": [0.5, 0.6, 0.0], "c": [0.0, 0.4, 0.8]})
    deltas = zero_replacement_deltas(X)
    closed = multiplicative_replacement(X, deltas)
    np.testing.assert_allclose(closed.sum(axis=1), 1.0)
    assert (closed > 0).all()
    np.testing.assert_allclose(clr(closed).sum(axis=1), 0.0, atol=1e-12)
    V = helmert_basis(3)
    np.testing.assert_allclose(V.T @ V, np.eye(2), atol=1e-12)
    # ILR is an isometry of CLR space.
    z, c = ilr(closed), clr(closed)
    np.testing.assert_allclose(np.linalg.norm(z[0] - z[1]), np.linalg.norm(c[0] - c[1]))
    empty = multiplicative_replacement(np.zeros((1, 3)), deltas)
    assert np.isnan(empty).all()
