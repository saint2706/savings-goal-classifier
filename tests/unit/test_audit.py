"""Unit tests for subgroup audits, measurement-error robustness, and nominal scaling shift."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from savings_goal.evaluation import robustness, subgroup, transport


def _audit_frame(n: int = 1200, seed: int = 19) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    income = rng.lognormal(11.3, 0.7, n)
    cototal = income * rng.uniform(0.4, 2.6, n)
    sr = (income - cototal) / income
    y = (sr >= 0.20).astype(int)
    logit = 1.6 * (np.log(income) - 11.3) - 0.8 * (cototal / income - 1.0)
    p = 1.0 / (1.0 + np.exp(-logit))
    df = pd.DataFrame(
        {
            "INCOME": income,
            "Log_Income": np.log(income),
            "COTOTAL": cototal,
            "Savings_Rate": sr,
            "Goal_Met": y,
            "IDPSU": rng.integers(0, 40, n),
            "WT": rng.uniform(0.5, 2.0, n),
            "Caste_Group": rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.35, 0.15]),
            "Religion": rng.choice(["R1", "R2", "Tiny"], size=n, p=[0.7, 0.28, 0.02]),
            "Occupation": rng.choice(["Farm", "Salaried", "Business"], size=n),
            "Has_Fixed_Deposit": (rng.random(n) < np.clip(p * 0.4, 0.02, 0.9)).astype(int),
            "Has_Bank_Savings": (rng.random(n) < np.clip(p * 0.6 + 0.2, 0.05, 0.95)).astype(int),
        }
    )
    return df, y, p


def test_subgroup_audit_computes_errors_calibration_and_selection() -> None:
    df, y, p = _audit_frame()
    reach = np.clip(1.0 - np.abs(p - 0.5), 0.0, 1.0)
    out = subgroup.subgroup_audit(
        df,
        y,
        p,
        group_cols=["Caste_Group", "Religion"],
        budget=0.25,
        weights=df["WT"].to_numpy(),
        reach_score=reach,
        min_n=50,
    )
    assert set(out["group_col"]) == {"Caste_Group", "Religion"}
    # Tiny (~24 rows) is filtered out by min_n=50
    assert "Tiny" not in set(out["subgroup"])
    expected_cols = {
        "n",
        "weighted_share",
        "at_risk_rate",
        "weighted_at_risk_rate",
        "ROC-AUC",
        "CITL",
        "weighted_CITL",
        "ECE",
        "at_risk_FNR",
        "at_risk_FPR",
        "selection_rate",
        "reach_selection_rate",
    }
    assert expected_cols <= set(out.columns)
    assert out["ROC-AUC"].between(0.0, 1.0).all()
    assert out["ECE"].between(0.0, 1.0).all()
    assert out["at_risk_FNR"].between(0.0, 1.0).all()
    assert out["at_risk_FPR"].between(0.0, 1.0).all()

    # Without reach_score or weights still works cleanly
    out_plain = subgroup.subgroup_audit(df, y, p, group_cols=["Caste_Group"], min_n=50)
    assert "reach_selection_rate" not in out_plain.columns
    assert out_plain["weighted_share"].sum() == pytest.approx(1.0)


def test_subgroup_audit_handles_constant_class_subgroup() -> None:
    df, _, p = _audit_frame(400)
    df.loc[df["Caste_Group"] == "C", "Goal_Met"] = 0
    y_mod = df["Goal_Met"].to_numpy()
    out = subgroup.subgroup_audit(df, y_mod, p, group_cols=["Caste_Group"], min_n=10)
    row_c = out[out["subgroup"] == "C"].iloc[0]
    assert np.isnan(row_c["ROC-AUC"])


def test_sensitive_ablation_summary_returns_bootstrap_cis() -> None:
    df, y, p = _audit_frame()
    p_alt = np.clip(p + np.random.default_rng(7).normal(0, 0.03, len(p)), 0.01, 0.99)
    res = subgroup.sensitive_ablation_summary(
        y,
        p,
        p_alt,
        df["IDPSU"].to_numpy(),
        budget=0.25,
        weights=df["WT"].to_numpy(),
        n_boot=25,
        seed=42,
    )
    assert res["diff_auc_ci"][0] <= res["diff_auc_ci"][1]
    assert res["diff_recall@25%_ci"][0] <= res["diff_recall@25%_ci"][1]
    assert "weighted_recall@25%" in res["with_sensitive"]
    assert "weighted_precision@25%" in res["without_sensitive"]

    # Also test without weights and with degenerate bootstrap skip branch
    y_tiny = np.zeros(20)
    y_tiny[0] = 1.0
    p_tiny = np.linspace(0.1, 0.9, 20)
    g_tiny = np.arange(20)
    res_plain = subgroup.sensitive_ablation_summary(
        y_tiny, p_tiny, p_tiny, g_tiny, budget=0.25, weights=None, n_boot=30, seed=1
    )
    assert "weighted_recall@25%" not in res_plain["with_sensitive"]
    assert res_plain["diff_auc"] == pytest.approx(0.0)


def test_decile_allocation_contrasts_risk_and_reachability() -> None:
    df, _, p = _audit_frame()
    reach = 1.0 - np.abs(df["Savings_Rate"].to_numpy() - 0.175)
    alloc = subgroup.decile_allocation(
        df["INCOME"].to_numpy(),
        1.0 - p,
        reach,
        budget=0.25,
        weights=df["WT"].to_numpy(),
    )
    assert len(alloc) == 10
    # Poorest decile has much higher risk selection than richest decile
    assert alloc.iloc[0]["risk_selection_rate"] > alloc.iloc[-1]["risk_selection_rate"]
    assert alloc["risk_selection_rate"].between(0.0, 1.0).all()
    assert alloc["reach_selection_rate"].between(0.0, 1.0).all()


def test_dissaver_sensitivity_and_underreporting_stress() -> None:
    df, _, p = _audit_frame()
    inc_score = -df["INCOME"].to_numpy()

    dis = robustness.dissaver_sensitivity(
        df,
        p,
        inc_score,
        cutoffs=(None, 2.0, 1.5),
        budget=0.25,
        groups=df["IDPSU"].to_numpy(),
        n_boot=20,
        seed=42,
    )
    assert len(dis) == 3
    assert dis.iloc[0]["n"] == len(df)
    assert dis.iloc[1]["n"] < dis.iloc[0]["n"]
    assert (dis["diff_ci_low"] <= dis["diff_ci_high"]).all()

    dis_no_boot = robustness.dissaver_sensitivity(df, p, inc_score, cutoffs=(None, 2.0))
    assert "diff_ci_low" not in dis_no_boot.columns

    mults = {
        "1.0x": np.ones(len(df)),
        "1.2x": np.full(len(df), 1.2),
        "hetero": np.where(df["Occupation"] == "Farm", 1.25, 1.05),
    }
    stress = robustness.underreporting_stress(df, p, inc_score, mults)
    assert len(stress) == 3
    # Raising income reduces at-risk prevalence
    assert stress.iloc[1]["at_risk_rate"] < stress.iloc[0]["at_risk_rate"]
    assert stress["ROC-AUC"].between(0.5, 1.0).all()


def test_external_criterion_check() -> None:
    df, _, p = _audit_frame()
    ext = robustness.external_criterion_check(
        df,
        scores={"headline": p, "income": df["INCOME"].to_numpy()},
        target_cols=["Has_Fixed_Deposit", "Has_Bank_Savings"],
    )
    assert list(ext["target"]) == ["Has_Fixed_Deposit", "Has_Bank_Savings"]
    assert ext["auc_headline"].between(0.0, 1.0).all()
    assert ext["auc_income"].between(0.0, 1.0).all()


def test_nominal_scaling_shift_inflates_predicted_on_track_rate(features: pd.DataFrame) -> None:
    df = features.dropna(subset=["Log_Income", "Household_Size"]).copy()
    train = (df["IDPSU"] % 2 == 0).to_numpy()
    test = ~train
    cols = ["Log_Income", "Household_Size"]
    res = transport.nominal_scaling_shift(
        df, train, test, LogisticRegression(), cols, scale_factor=2.4
    )
    assert res["scale_factor"] == pytest.approx(2.4)
    assert res["pred_on_track_after"] > res["pred_on_track_before"]
    assert res["citl_after"] > res["citl_before"]
    assert res["ece_after"] > res["ece_before"]
