"""Targeting metrics, the outreach what-if, the transport tests and the leakage simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from savings_goal.evaluation import leakage_sim, outreach, transport
from savings_goal.evaluation.metrics import (
    operating_point,
    targeting_table,
    top_by_budget,
)


def _toy(n: int = 4000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    risk = rng.random(n)
    at_risk = (rng.random(n) < risk * 0.9 + 0.05).astype(float)
    return at_risk, risk


def test_ceiling_capture_equals_precision_below_prevalence() -> None:
    at_risk, risk = _toy()
    t = targeting_table(at_risk, {"m": risk}, budgets=(0.1, 0.25, 0.5))
    assert at_risk.mean() > 0.5
    np.testing.assert_allclose(t["ceiling_capture"], t["precision"])
    assert (t["recall"] < t["precision"]).all()  # recall is a different quantity


def test_ceiling_capture_equals_recall_once_ceiling_saturates() -> None:
    at_risk, risk = _toy()
    t = targeting_table(at_risk, {"m": risk}, budgets=(0.95,))
    assert t["ceiling"].iloc[0] == 1.0
    assert t["ceiling_capture"].iloc[0] == pytest.approx(t["recall"].iloc[0])


def test_weighted_budget_is_a_share_of_the_weighted_population() -> None:
    score = np.array([4.0, 3.0, 2.0, 1.0])
    w = np.array([1.0, 1.0, 6.0, 2.0])
    top, used = top_by_budget(score, 0.2, w)
    assert list(top) == [0, 1]
    assert used is w or np.array_equal(used, w)


def test_operating_point_matches_hand_count() -> None:
    at_risk = np.array([1, 1, 0, 1, 0, 0])
    score = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.1])
    op = operating_point(at_risk, score, 0.5)
    assert op == {
        "contacted": 0.5,
        "recall": pytest.approx(2 / 3),
        "precision": pytest.approx(2 / 3),
    }


def _hh(n: int = 2000, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    income = rng.lognormal(11.5, 0.8, n)
    return pd.DataFrame(
        {
            "INCOME": income,
            "Savings_Rate": np.clip(rng.normal(-0.1, 0.5, n), -3, 0.9),
            "Has_Debt": rng.integers(0, 2, n),
        }
    )


def test_poorest_are_at_risk_but_out_of_reach() -> None:
    df = _hh()
    delta = 0.05
    reach = outreach.reachable(df, delta)
    assert not reach[df["Savings_Rate"] < 0.0].any()
    assert not reach[df["Savings_Rate"] >= 0.2].any()
    near = df["Savings_Rate"].between(0.15, 0.2, inclusive="left")
    assert reach[near.to_numpy()].all()


def test_reachability_ranking_beats_risk_ranking_on_flips() -> None:
    df = _hh()
    scen = outreach.uniform(0.2)
    scores = {
        "risk": -df["Savings_Rate"].to_numpy(),
        "reach": -np.abs(df["Savings_Rate"] - 0.17).to_numpy(),
    }
    t = outreach.simulate(df, scores, scen, budgets=[0.1], delta=0.05, cost=100).set_index(
        "strategy"
    )
    assert t.loc["reach", "flips_per_1000"] > t.loc["risk", "flips_per_1000"]
    # expected flips = r x reachable share among the contacted: bounded by r
    assert t["flips_per_1000"].max() <= 1000 * 0.2 + 1e-9


def test_debt_scenario_lowers_response_for_indebted() -> None:
    df = _hh()
    r = outreach.debt_burdened().response(df)
    assert r[df["Has_Debt"].to_numpy() == 1].max() < r[df["Has_Debt"].to_numpy() == 0].min()


def test_zero_flips_gives_infinite_cost() -> None:
    df = _hh()
    scen = outreach.Scenario("none", lambda d: np.zeros(len(d)))
    t = outreach.simulate(df, {"s": df["INCOME"].to_numpy()}, scen, [0.2], 0.05, 100)
    assert np.isinf(t["cost_per_flip"].iloc[0])


def test_oracle_exact_when_food_is_proportional_to_size() -> None:
    assert leakage_sim.oracle_auc(0.0, 1.0, n=5000) == pytest.approx(1.0)


def test_oracle_auc_falls_as_food_decouples_from_size() -> None:
    aucs = [leakage_sim.oracle_auc(s, 0.63, n=8000) for s in (0.1, 0.5, 1.5)]
    assert aucs[0] > aucs[1] > aucs[2] > 0.5


def test_residual_sd_recovers_the_generating_elasticity() -> None:
    rng = np.random.default_rng(3)
    size = pd.Series(rng.integers(1, 9, 5000).astype(float))
    food = 100 * size**0.7 * np.exp(0.3 * rng.standard_normal(5000))
    fit = leakage_sim.residual_sd(food, size)
    assert fit["elasticity"] == pytest.approx(0.7, abs=0.03)
    assert fit["sigma"] == pytest.approx(0.3, abs=0.03)


def test_leave_group_out_folds_partition_the_rows(features: pd.DataFrame) -> None:
    df = features.copy()
    df["STATEID"] = df["IDPSU"] % 5
    cols = ["Log_Income", "Household_Size"]
    out = transport.leave_group_out(
        df.dropna(subset=cols), "STATEID", LogisticRegression(), cols, 3
    )
    assert len(out) == 3
    assert out["n_test"].sum() == pytest.approx(len(df.dropna(subset=cols)))
    assert out["ROC-AUC"].between(0, 1).all()


def test_oof_sample_weight_is_forwarded_to_the_last_step(features: pd.DataFrame) -> None:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from savings_goal.evaluation.cv import grouped_cv, oof_predict_proba

    df = features.dropna(subset=["Log_Income", "Household_Size"])
    X, y, g = df[["Log_Income", "Household_Size"]], df["Goal_Met"], df["IDPSU"]
    model = make_pipeline(StandardScaler(), LogisticRegression())
    cv = grouped_cv(n_splits=3)
    plain = oof_predict_proba(model, X, y, g, cv)
    w = pd.Series(np.where(y == 1, 20.0, 1.0), index=df.index)
    weighted = oof_predict_proba(model, X, y, g, cv, sample_weight=w)
    assert weighted.mean() > plain.mean()  # up-weighting on-track pulls P(on track) up


def test_income_gradient_response_rises_with_income() -> None:
    df = _hh()
    r = outreach.constrained_poor(0.05, 0.35).response(df)
    order = np.argsort(df["INCOME"].to_numpy())
    assert r[order[0]] == pytest.approx(0.05, abs=1e-3) and r[order[-1]] == pytest.approx(0.35)
    assert (np.diff(r[order]) >= 0).all()


def test_flips_per_1000_agrees_with_simulate_and_supports_bootstrap_rows() -> None:
    df = _hh()
    scen, score = outreach.uniform(0.2), -df["Savings_Rate"].to_numpy()
    sim = outreach.simulate(df, {"s": score}, scen, [0.25], 0.05, 100.0)["flips_per_1000"].iloc[0]
    assert outreach.flips_per_1000(df, score, scen, 0.25, 0.05) == pytest.approx(sim)
    idx = np.arange(len(df))[::-1]  # a resample is just a row selection
    assert outreach.flips_per_1000(df, score, scen, 0.25, 0.05, idx) == pytest.approx(sim)
