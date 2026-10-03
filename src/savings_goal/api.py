"""Two-tier deployable scoring service.

Tier 1 (onboarding) scores from income, demographics and debt only - the
feature set A1 showed carries the defensible signal. Tier 2 adds spending
shares, but only when they were observed *before* the prediction window opens:
shares measured over the same period as the outcome reconstruct total spend
(A1, T3), so scoring with them is a restatement of the label, not a forecast.

Run with ``uv run --extra api sgc serve`` after ``sgc train`` has written the
model artifacts (git-ignored: they are fitted on IHDS microdata).
"""

from __future__ import annotations

import datetime as dt
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pandera.pandas as pa
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from savings_goal.config import CATEGORICALS, EXPENSE_CATEGORIES, ROOT, SHARE_COLS
from savings_goal.data.build import CASTE_LABELS, RELIGION_LABELS, URBAN4_LABELS
from savings_goal.data.schema import OCCUPATIONS

ARTIFACTS = ROOT / "artifacts"
SHARE_SUM_TOL = 1e-3


class Tier1Request(BaseModel):
    annual_income: float = Field(gt=0, description="annual household income, rupees")
    household_size: int = Field(ge=1, le=60)
    age_dependents: int = Field(ge=0, le=60, description="members aged 0-14 or 60+")
    head_age: float | None = Field(default=None, ge=10, le=110)
    max_adult_education: float | None = Field(default=None, ge=0, le=15)
    outstanding_debt: float | None = Field(default=None, ge=0, description="rupees; null = unknown")
    occupation: str = "No_Regular_Worker"
    area_type: str | None = None
    caste_group: str | None = None
    religion: str | None = None

    @model_validator(mode="after")
    def _dependents_fit(self) -> Tier1Request:
        if self.age_dependents > self.household_size:
            raise ValueError("age_dependents cannot exceed household_size")
        return self


class Tier2Request(Tier1Request):
    shares: dict[str, float] = Field(description="category -> share of total expenditure")
    shares_observed_through: dt.date
    prediction_window_start: dt.date

    @model_validator(mode="after")
    def _shares_precede_window(self) -> Tier2Request:
        if self.shares_observed_through >= self.prediction_window_start:
            raise ValueError(
                "spending shares must be observed strictly before the prediction window; "
                "same-period shares reconstruct the outcome"
            )
        return self


class Score(BaseModel):
    tier: int
    p_on_track: float
    at_risk_score: float
    model: str


def _categories(values: list[str]) -> list[str]:
    return [*values, "Unknown"]


TIER1_SCHEMA = pa.DataFrameSchema(
    {
        "Log_Income": pa.Column(float, pa.Check.in_range(0, 20)),
        "Household_Size": pa.Column(float, pa.Check.ge(1)),
        "Age_Dependents": pa.Column(float, pa.Check.ge(0)),
        "Dependency_Ratio": pa.Column(float, pa.Check.in_range(0, 1)),
        "Head_Age": pa.Column(float, nullable=True),
        "Max_Adult_Education": pa.Column(float, nullable=True),
        "Debt_To_Income": pa.Column(float, pa.Check.ge(0), nullable=True),
        "Has_Debt": pa.Column(int, pa.Check.isin([0, 1])),
        "Debt_Missing": pa.Column(int, pa.Check.isin([0, 1])),
        "Occupation": pa.Column(str, pa.Check.isin(OCCUPATIONS)),
        "Area_Type": pa.Column(str, pa.Check.isin(_categories(list(URBAN4_LABELS.values())))),
        "Caste_Group": pa.Column(str, pa.Check.isin(_categories(list(CASTE_LABELS.values())))),
        "Religion": pa.Column(str, pa.Check.isin(_categories(list(RELIGION_LABELS.values())))),
    },
    strict=False,
)

TIER2_SCHEMA = pa.DataFrameSchema(
    {
        **TIER1_SCHEMA.columns,
        **{c: pa.Column(float, pa.Check.in_range(0, 1)) for c in SHARE_COLS},
    },
    checks=[
        pa.Check(
            lambda d: (d[SHARE_COLS].sum(axis=1) - 1).abs() <= SHARE_SUM_TOL,
            name="shares_sum_to_one",
        )
    ],
    strict=False,
)


def tier1_frame(req: Tier1Request) -> pd.DataFrame:
    debt = req.outstanding_debt
    row: dict[str, Any] = {
        "Log_Income": float(np.log(req.annual_income)),
        "Household_Size": float(req.household_size),
        "Age_Dependents": float(req.age_dependents),
        "Dependency_Ratio": req.age_dependents / req.household_size,
        "Head_Age": req.head_age,
        "Max_Adult_Education": req.max_adult_education,
        "Debt_To_Income": None if debt is None else debt / req.annual_income,
        "Has_Debt": int(bool(debt)),
        "Debt_Missing": int(debt is None),
        "Occupation": req.occupation,
        "Area_Type": req.area_type or "Unknown",
        "Caste_Group": req.caste_group or "Unknown",
        "Religion": req.religion or "Unknown",
    }
    df = pd.DataFrame([row]).astype(
        {"Head_Age": float, "Max_Adult_Education": float, "Debt_To_Income": float}
    )
    return TIER1_SCHEMA.validate(df)


def tier2_frame(req: Tier2Request) -> pd.DataFrame:
    unknown = set(req.shares) - set(EXPENSE_CATEGORIES)
    if unknown:
        raise ValueError(f"unknown expense categories: {sorted(unknown)}")
    df = tier1_frame(req)
    for cat in EXPENSE_CATEGORIES:
        df[f"{cat}_Share"] = float(req.shares.get(cat, 0.0))
    for cat in ("Eating_Out", "Rent", "Education", "Entertainment", "Insurance"):
        df[f"Spends_On_{cat}"] = int(df[f"{cat}_Share"].iloc[0] > 0)
    return TIER2_SCHEMA.validate(df)


@lru_cache(maxsize=2)
def load_tier(tier: int, artifacts: Path = ARTIFACTS) -> Any:
    path = artifacts / f"tier{tier}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `sgc train` first")
    return joblib.load(path)


def _score(tier: int, X: pd.DataFrame) -> Score:
    try:
        bundle = load_tier(tier)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    cols = bundle["features"]
    p = float(bundle["model"].predict_proba(X[cols])[0, 1])
    return Score(tier=tier, p_on_track=p, at_risk_score=1 - p, model=bundle["name"])


def create_app() -> FastAPI:
    app = FastAPI(title="Savings-adequacy triage", version="0.2.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/score/onboarding", response_model=Score)
    def score_onboarding(req: Tier1Request) -> Score:
        try:
            X = tier1_frame(req)
        except (pa.errors.SchemaError, pa.errors.SchemaErrors) as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        return _score(1, X)

    @app.post("/score/with-spending", response_model=Score)
    def score_with_spending(req: Tier2Request) -> Score:
        try:
            X = tier2_frame(req)
        except (ValueError, pa.errors.SchemaError, pa.errors.SchemaErrors) as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        return _score(2, X)

    return app


def train_tiers(features: pd.DataFrame, artifacts: Path = ARTIFACTS) -> dict[str, float]:
    """Fit both tiers on the training split and write the joblib bundles."""
    from sklearn.metrics import roc_auc_score

    from savings_goal.features.engineer import spec_from_frame
    from savings_goal.models.pipeline import MODEL_FINAL_PATH, load_model_final, xgb_pipeline

    spec = spec_from_frame(features)
    final = load_model_final() if MODEL_FINAL_PATH.exists() else {}
    tiers = {1: spec.deployable_numeric, 2: spec.numeric}
    keys = {1: "deployable", 2: "full"}
    train = features[~features["Is_Test"]]
    test = features[features["Is_Test"]]
    artifacts.mkdir(parents=True, exist_ok=True)
    out = {}
    for tier, numeric in tiers.items():
        params = final.get(keys[tier], {}).get("best_params")
        model = xgb_pipeline(numeric, CATEGORICALS, params)
        cols = numeric + CATEGORICALS
        model.fit(train[cols], train["Goal_Met"])
        auc = roc_auc_score(test["Goal_Met"], model.predict_proba(test[cols])[:, 1])
        joblib.dump(
            {"model": model, "features": cols, "name": f"xgboost-{keys[tier]}"},
            artifacts / f"tier{tier}.joblib",
        )
        out[f"tier{tier} held-out ROC-AUC"] = float(auc)
    load_tier.cache_clear()
    return out
