from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from savings_goal.config import CATEGORICALS, SHARE_COLS
from savings_goal.features.engineer import spec_from_frame
from savings_goal.io import load_features, load_households, read_result, write_result
from savings_goal.models import personas, regression
from savings_goal.models.pipeline import load_model_final, save_model_final, tuned_xgb_params
from savings_goal.models.uplift import TLearner, qini_auc


def test_model_final_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "model_final.json"
    save_model_final({"best_params": {"max_depth": 4}, "best_score": 0.8}, path)
    assert load_model_final(path)["best_score"] == 0.8
    assert tuned_xgb_params(path) == {"max_depth": 4}


def test_results_io(tmp_path: Path) -> None:
    payload = {
        "a": np.float64(1.5),
        "b": np.int64(2),
        "c": np.arange(3),
        "d": pd.Series({"x": 1.0}),
        "e": pd.DataFrame({"y": [1]}),
        "f": np.float64("nan"),
        "g": [True, "text", None, (1, 2)],
    }
    write_result("demo", payload, tmp_path)
    back = read_result("demo", tmp_path)
    assert back["c"] == [0, 1, 2] and back["f"] is None and back["e"] == [{"y": 1}]
    assert back["g"] == [True, "text", None, [1, 2]]
    frame = pd.DataFrame({"x": [1, 2]})
    frame.to_parquet(tmp_path / "t.parquet")
    pd.testing.assert_frame_equal(load_households(tmp_path / "t.parquet"), frame)
    pd.testing.assert_frame_equal(load_features(tmp_path / "t.parquet"), frame)
    with pytest.raises(TypeError):
        write_result("bad", {"x": object()}, tmp_path)


def test_personas(features: pd.DataFrame) -> None:
    spec = spec_from_frame(features)
    shares = features[spec.core_shares]
    X = personas.standardise(personas.ilr_coordinates(shares))
    sweep = personas.kmeans_sweep(X, range(2, 4), sample=200)
    assert {"silhouette", "davies_bouldin", "calinski_harabasz"} <= set(sweep.columns)
    stab = personas.bootstrap_stability(X, 2, n_boot=3)
    assert ((stab >= -1) & (stab <= 1)).all()
    sens = personas.delta_sensitivity(shares, 2, factors=(0.1, 0.65))
    assert sens.loc[sens["delta factor"] == 0.65, "ARI vs baseline delta"].iloc[0] == 1.0
    df = features.copy()
    df["Persona"] = personas.fit_kmeans(X, 2).labels_
    df["Income_Decile"] = pd.qcut(df["INCOME"], 3, labels=False)
    assoc = personas.income_association(df["Persona"], df["INCOME"], n_bins=3)
    assert 0 <= assoc["NMI vs income decile"] <= 1
    inter = personas.persona_income_interaction(df)
    assert 0 <= inter["p persona | decile"] <= 1


def test_quantile_regression(features: pd.DataFrame) -> None:
    spec = spec_from_frame(features)
    model = regression.quantile_model(spec.deployable_numeric)
    model.fit(
        features[[*spec.deployable_numeric, *CATEGORICALS]],
        features["Savings_Rate"],
    )
    q = np.array([0.1, 0.3])
    np.testing.assert_allclose(regression.rupee_shortfall(q, np.array([100.0, 100.0])), [10.0, 0.0])
    assert regression.pinball_loss(np.array([0.0, 1.0]), np.array([0.5, 0.5])) == 0.25
    assert regression.decile_of(pd.Series(np.arange(100.0))).nunique() == 10
    assert set(SHARE_COLS).isdisjoint(spec.deployable_numeric)


def test_t_learner_recovers_uplift() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(size=(n, 1))
    t = rng.integers(0, 2, n)
    effect = np.where(x[:, 0] > 0, 0.3, 0.0)
    y = (rng.random(n) < 0.3 + t * effect).astype(int)
    learner = TLearner(LogisticRegression()).fit(x, t, y)
    uplift = learner.predict_uplift(x)
    assert uplift[x[:, 0] > 1].mean() > uplift[x[:, 0] < -1].mean()
    assert qini_auc(uplift, t, y) > qini_auc(-uplift, t, y)
    with pytest.raises(ValueError):
        TLearner(LogisticRegression()).fit(x, np.ones(n), y)


def test_json_is_strict(tmp_path: Path) -> None:
    path = write_result("inf", {"x": 1.0}, tmp_path)
    assert json.loads(path.read_text()) == {"x": 1.0}
