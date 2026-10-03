"""Pandera contracts for the built household table and the engineered features."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.api.polars.types import PolarsData

from savings_goal.config import EXPENSE_CATEGORIES, SHARE_COLS
from savings_goal.data.build import (
    CASTE_LABELS,
    OCCUPATION_SOURCES,
    RELIGION_LABELS,
    URBAN4_LABELS,
)

SHARE_SUM_TOL = 1e-9
OCCUPATIONS = [label for _, label in OCCUPATION_SOURCES] + ["No_Regular_Worker"]


def _shares_sum_to_one(data: PolarsData) -> pl.LazyFrame:
    total = pl.sum_horizontal(SHARE_COLS)
    return data.lazyframe.select(((total - 1.0).abs() <= SHARE_SUM_TOL).alias("ok"))


def _label_matches_rate(data: PolarsData) -> pl.LazyFrame:
    # Goal_Met is a deterministic function of Savings_Rate; the build threshold
    # is not stored, so check monotonicity: every met row's rate exceeds every
    # unmet row's rate.
    lf = data.lazyframe
    lo = lf.filter(pl.col("Goal_Met") == 1).select(pl.col("Savings_Rate").min())
    hi = lf.filter(pl.col("Goal_Met") == 0).select(pl.col("Savings_Rate").max())
    return lo.join(hi, how="cross").select(
        (pl.col("Savings_Rate") > pl.col("Savings_Rate_right")).alias("ok")
    )


def _in(values: list[str]) -> pa.Check:
    return pa.Check.isin(values)


share_column = pa.Column(pl.Float64, pa.Check.in_range(0.0, 1.0), nullable=False)
money_column = pa.Column(pl.Float64, pa.Check.ge(0.0), nullable=False)

households_schema = pa.DataFrameSchema(
    {
        "IDHH": pa.Column(pl.Int64, unique=True),
        "IDPSU": pa.Column(pl.Int64),
        "WT": pa.Column(pl.Float64, pa.Check.gt(0.0)),
        "INCOME": pa.Column(pl.Float64, pa.Check.gt(0.0)),
        "Log_Income": pa.Column(pl.Float64),
        "Household_Size": pa.Column(pl.Float64, pa.Check.ge(1.0)),
        "Age_Dependents": pa.Column(pl.Float64, pa.Check.ge(0.0)),
        "Head_Age": pa.Column(pl.Float64, nullable=True),
        "Max_Adult_Education": pa.Column(pl.Float64, nullable=True),
        "Occupation": pa.Column(pl.String, _in(OCCUPATIONS)),
        "Area_Type": pa.Column(pl.String, _in(list(URBAN4_LABELS.values())), nullable=True),
        "Caste_Group": pa.Column(pl.String, _in(list(CASTE_LABELS.values())), nullable=True),
        "Religion": pa.Column(pl.String, _in(list(RELIGION_LABELS.values())), nullable=True),
        "Debt_To_Income": pa.Column(pl.Float64, pa.Check.ge(0.0), nullable=True),
        "Debt_Missing": pa.Column(pl.Int8, pa.Check.isin([0, 1])),
        **{c: share_column for c in SHARE_COLS},
        **{c: money_column for c in EXPENSE_CATEGORIES},
        "Category_Total": pa.Column(pl.Float64, pa.Check.gt(0.0)),
        "COTOTAL": pa.Column(pl.Float64, pa.Check.gt(0.0)),
        "Savings": pa.Column(pl.Float64),
        "Savings_Rate": pa.Column(pl.Float64, pa.Check.le(1.0)),
        "Goal_Met": pa.Column(pl.Int8, pa.Check.isin([0, 1])),
    },
    checks=[
        pa.Check(_shares_sum_to_one, name="shares_sum_to_one"),
        pa.Check(_label_matches_rate, name="label_monotone_in_savings_rate"),
    ],
    strict=False,
    coerce=False,
)


def validate_households(df: pl.DataFrame) -> pl.DataFrame:
    """Raise ``pandera.errors.SchemaError(s)`` if the table breaks the contract."""
    return households_schema.validate(df, lazy=True)
