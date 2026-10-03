from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from savings_goal.config import CATEGORICALS, GROUP_COL, SHARE_COLS
from savings_goal.evaluation import business, explain, leakage
from savings_goal.evaluation.cv import (
    cross_validate_grouped,
    grouped_cv,
    oof_predict_proba,
    summarise_folds,
)
from savings_goal.evaluation.metrics import (
    capture_at_budget,
    capture_vs_ceiling,
    expected_calibration_error,
    fold_metrics,
    grouped_bootstrap,
    percentile_ci,
    probability_metrics,
    weighted_mean,
)
from savings_goal.features.engineer import spec_from_frame
from savings_goal.models.pipeline import logistic, model_zoo, preprocessor, xgb_pipeline


def test_ece_and_probability_metrics() -> None:
    y = np.array([0, 0, 1, 1])
    assert expected_calibration_error(y, np.array([0.0, 0.0, 1.0, 1.0])) == 0
    assert expected_calibration_error(y, np.full(4, 0.9)) == pytest.approx(0.4)
    m = probability_metrics(y, np.array([0.1, 0.4, 0.35, 0.8]))
    assert m["ROC-AUC"] == pytest.approx(0.75)
    assert set(m) >= {"PR-AUC (on-track)", "PR-AUC (at-risk)", "Brier", "Log-loss", "ECE"}


def test_capture_respects_ceiling() -> None:
    at_risk = np.array([1, 1, 1, 0] * 25)
    perfect = at_risk + np.linspace(0, 0.1, 100)
    cap, prec = capture_at_budget(at_risk, perfect, 0.25)
    assert prec == 1.0
    assert cap == pytest.approx(25 / 75)
    table = capture_vs_ceiling(at_risk, {"perfect": perfect}, budgets=(0.25, 0.9))
    assert table["pct_of_ceiling"].tolist() == pytest.approx([1.0, 75 / 75])
    w = np.ones(100)
    weighted = capture_vs_ceiling(at_risk, {"perfect": perfect}, budgets=(0.25,), weights=w)
    assert weighted["capture"].iloc[0] == pytest.approx(cap)


def test_grouped_bootstrap_resamples_groups() -> None:
    x = np.repeat([0.0, 10.0], 50)
    groups = np.repeat([0, 1], 50)
    draws = grouped_bootstrap(lambda idx: float(x[idx].mean()), groups, n_boot=200)
    assert set(np.round(draws, 6)) <= {0.0, 5.0, 10.0}
    lo, hi = percentile_ci(draws)
    assert lo <= 5.0 <= hi
    assert weighted_mean(np.array([1.0, np.nan, 3.0]), np.array([1.0, 1.0, 3.0])) == 2.5


def test_grouped_cv_and_models(features: pd.DataFrame) -> None:
    spec = spec_from_frame(features)
    X, y, g = features[spec.features], features["Goal_Met"], features[GROUP_COL]
    folds = cross_validate_grouped(
        xgb_pipeline(spec.numeric), X, y, g, fold_metrics(), grouped_cv(3)
    )
    assert len(folds) == 3 and folds["ROC-AUC"].between(0, 1).all()
    assert "ROC-AUC sd" in summarise_folds(folds)
    zoo = model_zoo(spec.numeric)
    assert len(zoo) == 7
    svm = cross_validate_grouped(
        zoo["Linear SVM"], X, y, g, fold_metrics(with_proba=False), grouped_cv(3)
    )
    assert svm["ROC-AUC"].notna().all()
    oof = oof_predict_proba(zoo["Logistic Regression"], X, y, g, grouped_cv(3))
    assert np.isfinite(oof).all()
    seen: list[int] = []
    cross_validate_grouped(
        logistic(),
        X[["Log_Income"]],
        y,
        g,
        {},
        grouped_cv(3),
        on_fold=lambda k, m, tr: seen.append(k),
    )
    assert seen == [0, 1, 2]


def test_linear_preprocessor_drops_reference_share(features: pd.DataFrame) -> None:
    spec = spec_from_frame(features)
    lin = preprocessor(spec.numeric, linear=True).fit(features[spec.features])
    names = list(lin.get_feature_names_out())
    assert "Groceries_Share" not in names and "Rent_Share" in names
    keep = preprocessor(spec.numeric, linear=True, drop_reference=False).fit(features[spec.features])
    assert "Groceries_Share" in keep.get_feature_names_out()
    tree = preprocessor(spec.numeric).fit(features[spec.features])
    assert "Groceries_Share" in tree.get_feature_names_out()
    only_num = preprocessor(["Debt_To_Income"], []).fit(features[["Debt_To_Income"]])
    assert list(only_num.get_feature_names_out()) == ["Debt_To_Income"]


def test_leakage_tests(features: pd.DataFrame, households: pd.DataFrame) -> None:
    spec = spec_from_frame(features)
    sets = leakage.ablation_sets(spec.features, spec.indicators, SHARE_COLS)
    assert "Household_Size" not in sets["no size family"]
    assert not set(SHARE_COLS) & set(sets["no shares or indicators (deployable)"])

    def make(cols: list[str]) -> object:
        return xgb_pipeline(
            [c for c in cols if c not in CATEGORICALS], [c for c in cols if c in CATEGORICALS]
        )

    t1 = leakage.t1_ablation(features, features["Goal_Met"], features[GROUP_COL], sets, make)
    assert t1.loc["all features", "AUC retained"] == 1.0
    t2 = leakage.t2_size_r2(households)
    assert 0 <= t2["r2"] <= 1
    hh = households.copy()
    hh.loc[hh.index[0], "Groceries_Share"] = 0.0
    t3 = leakage.t3_oracle(hh)
    assert t3.n_excluded == 1 and t3.n_used == len(hh) - 1
    assert 0 <= t3.auc <= 1


def test_explain_helpers(features: pd.DataFrame) -> None:
    spec = spec_from_frame(features)
    train = features[~features["Is_Test"]]
    test = features[features["Is_Test"]]
    model = xgb_pipeline(spec.numeric).fit(train[spec.features], train["Goal_Met"])
    pre, booster = model[:-1], model[-1]
    names = list(pre.get_feature_names_out())
    Xt = pd.DataFrame(pre.transform(test[spec.features]), columns=names)
    values, _, err = explain.tree_shap(booster, Xt)
    assert values.shape == Xt.shape and err < 1e-3
    fams = explain.feature_families(names)
    assert sum(len(v) for v in fams.values()) == len(names)
    g = explain.grouped_shap(values, names, fams)
    assert g["share of grouped total"].sum() == pytest.approx(1.0)
    raw = explain.raw_families(spec.features)
    assert sorted(c for v in raw.values() for c in v) == sorted(spec.features)
    from sklearn.metrics import roc_auc_score

    perm = explain.grouped_permutation_importance(
        model, test[spec.features], test["Goal_Met"], raw, roc_auc_score, n_repeats=2
    )
    assert set(perm.index) == set(raw)
    totals, pairs = explain.interaction_decomposition(booster, Xt, names, chunk=25)
    assert totals["n rows"] == len(Xt) and 0 <= totals["interaction share"] <= 1
    assert len(pairs) == len(names) * (len(names) - 1) // 2
    stab = explain.coefficient_stability(
        lambda: __import__("sklearn.pipeline").pipeline.make_pipeline(
            preprocessor(spec.numeric, linear=True), logistic()
        ),
        features[spec.features],
        features["Goal_Met"],
        n_parts=2,
    )
    assert "flips_sign" in stab
    with pytest.raises(KeyError):
        explain.feature_families(["mystery"])


def test_business_benchmarks(households: pd.DataFrame) -> None:
    df = households.copy()
    df["Income_Decile"] = pd.qcut(df["INCOME"], 4, labels=False)
    share_x, rupee_x = business.signed_excess(df)
    assert rupee_x.shape == share_x.shape
    assert len(share_x) == (df["Goal_Met"] == 0).sum()
    bench = business.benchmark_with_ci(df, n_boot=20)
    assert {"Rs CI low", "Rs CI high", "CI excludes zero (Rs)"} <= set(bench.table.columns)
    assert (bench.table["Rs CI low"] <= bench.table["Rs CI high"]).all()
    gaps = business.gap_closure(df)
    assert (gaps["gap"] > 0).all()
