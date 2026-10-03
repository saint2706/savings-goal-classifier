"""Preprocessing and model factories shared by every notebook.

Two preprocessing rules live here so that no notebook can skip them:

* Zero-inflated shares are never ``RobustScaler``-ed. Their IQR collapses, so
  a 90%-zero column gets a scale near zero and values in the hundreds. Linear
  models get ``StandardScaler``; tree models get no scaling at all.
* The 11 shares sum to one, so a linear model cannot identify all eleven
  coefficients. Linear pipelines drop a reference part (``Groceries_Share``);
  every other share coefficient is then read relative to food.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from savings_goal.config import CATEGORICALS, RESULTS, SEED, SHARE_COLS
from savings_goal.features.transforms import QuantileClipper

REFERENCE_SHARE = "Groceries_Share"
WINSORISED = ["Debt_To_Income"]

# The configuration Table 1 used to report (untuned) and the search space.
XGB_DEFAULT: dict[str, Any] = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.08,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
}
XGB_SEARCH_SPACE: dict[str, list[Any]] = {
    "max_depth": [4, 6, 8, 10],
    "learning_rate": [0.03, 0.08, 0.15],
    "n_estimators": [200, 400, 700],
    "subsample": [0.7, 0.9, 1.0],
    "min_child_weight": [1, 5, 20],
}
MODEL_FINAL_PATH = RESULTS / "model_final.json"


def preprocessor(
    numeric: list[str],
    categorical: list[str] | None = None,
    linear: bool = False,
    drop_reference: bool = True,
) -> ColumnTransformer:
    """Impute, winsorise debt per fold, one-hot categoricals; scale only for linear models."""
    categorical = CATEGORICALS if categorical is None else categorical
    drop = linear and drop_reference and _has_all_shares(numeric)
    num = [c for c in numeric if not (drop and c == REFERENCE_SHARE)]
    wins = [c for c in num if c in WINSORISED]
    rest = [c for c in num if c not in WINSORISED]

    def steps(extra: list[tuple[str, Any]]) -> Pipeline:
        tail: list[tuple[str, Any]] = [("scale", StandardScaler())] if linear else []
        return Pipeline([("impute", SimpleImputer(strategy="median")), *extra, *tail])

    transformers: list[tuple[str, Any, list[str]]] = []
    if wins:
        transformers.append(("wins", steps([("clip", QuantileClipper(upper=0.99))]), wins))
    if rest:
        transformers.append(("num", steps([]), rest))
    if categorical:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)
        )
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


def _has_all_shares(cols: list[str]) -> bool:
    return all(s in cols for s in SHARE_COLS)


def xgb(params: dict[str, Any] | None = None) -> XGBClassifier:
    return XGBClassifier(
        **(params or XGB_DEFAULT), eval_metric="logloss", random_state=SEED, n_jobs=-1
    )


def logistic(C: float = 1.0) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=3000, class_weight="balanced")


def model_zoo(numeric: list[str], categorical: list[str] | None = None) -> dict[str, Any]:
    """The seven families compared in Phase 4."""

    def lin(est: Any) -> Pipeline:
        return make_pipeline(preprocessor(numeric, categorical, linear=True), est)

    def tree(est: Any) -> Pipeline:
        return make_pipeline(preprocessor(numeric, categorical), est)

    return {
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": lin(logistic()),
        "Linear SVM": lin(LinearSVC(class_weight="balanced", max_iter=5000)),
        "Decision Tree": tree(
            DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=SEED)
        ),
        "Random Forest": tree(
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=SEED,
                n_jobs=-1,
            )
        ),
        "HistGradientBoosting": tree(HistGradientBoostingClassifier(random_state=SEED)),
        "XGBoost (untuned)": tree(xgb()),
    }


def xgb_pipeline(
    numeric: list[str], categorical: list[str] | None = None, params: dict[str, Any] | None = None
) -> Pipeline:
    return make_pipeline(preprocessor(numeric, categorical), xgb(params))


def save_model_final(payload: dict[str, Any], path: Path = MODEL_FINAL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float) + "\n")


def load_model_final(path: Path = MODEL_FINAL_PATH) -> dict[str, Any]:
    """The tuned XGBoost configuration written by NB04 (``best_params_``, ``best_score_``)."""
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def tuned_xgb_params(path: Path = MODEL_FINAL_PATH) -> dict[str, Any]:
    params: dict[str, Any] = load_model_final(path)["best_params"]
    return params
