"""Households table -> engineered feature table (the old NB02 export).

Everything here is row-wise or learned on the training split only. The
grouped train/test split is drawn first, before any design decision reads the
data, and stored as ``Is_Test`` so every later phase uses the same partition.
Winsorisation is no longer applied here: it lives inside the model pipeline
(``QuantileClipper``) so it is refit on each training fold.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from savings_goal.config import (
    CATEGORICALS,
    DEBT_COLS,
    DEMOGRAPHIC_NUMERIC,
    GROUP_COL,
    SEED,
    SHARE_COLS,
    TARGET,
)
from savings_goal.evaluation.cv import grouped_test_mask
from savings_goal.features.transforms import (
    clr,
    multiplicative_replacement,
    zero_replacement_deltas,
)

ZERO_INFLATED_CUTOFF = 0.30  # share zero for more than this -> participation indicator
CORE_CUTOFF = 0.20  # share zero for less than this -> part of the log-ratio core
DELTA_FACTOR = 0.65

META_COLS = ["IDHH", GROUP_COL, "STATEID", "WT", "INCOME", "Is_Test"]
OUTCOME_COLS = [TARGET, "Savings_Rate"]


@dataclass(frozen=True)
class FeatureSpec:
    """Column lists decided on the training split."""

    indicators: list[str]
    core_shares: list[str]
    clr_cols: list[str]
    deltas: dict[str, float]

    @property
    def numeric(self) -> list[str]:
        return SHARE_COLS + DEMOGRAPHIC_NUMERIC + self.indicators + DEBT_COLS

    @property
    def deployable_numeric(self) -> list[str]:
        """Income + demographics + debt: everything known at onboarding."""
        return DEMOGRAPHIC_NUMERIC + DEBT_COLS

    @property
    def features(self) -> list[str]:
        return self.numeric + CATEGORICALS


def engineer(households: pd.DataFrame, seed: int = SEED) -> tuple[pd.DataFrame, FeatureSpec]:
    df = households.copy()
    df["Is_Test"] = grouped_test_mask(df[TARGET], df[GROUP_COL], seed=seed)
    train = df.loc[~df["Is_Test"]]

    for c in CATEGORICALS:
        df[c] = df[c].fillna("Unknown")

    zero_rate = (train[SHARE_COLS] == 0).mean()
    zero_inflated = [c for c in SHARE_COLS if zero_rate[c] > ZERO_INFLATED_CUTOFF]
    indicators = []
    for c in zero_inflated:
        name = f"Spends_On_{c.removesuffix('_Share')}"
        df[name] = (df[c] > 0).astype(int)
        indicators.append(name)

    df["Has_Debt"] = (df["Debt_To_Income"].fillna(0) > 0).astype(int)
    df["Dependency_Ratio"] = df["Age_Dependents"] / df["Household_Size"]

    core = [c for c in SHARE_COLS if zero_rate[c] < CORE_CUTOFF]
    deltas = zero_replacement_deltas(train[core], DELTA_FACTOR)
    clr_vals = clr(multiplicative_replacement(df[core], deltas))
    clr_cols = [c.replace("_Share", "_CLR") for c in core]
    df[clr_cols] = clr_vals

    spec = FeatureSpec(
        indicators=indicators,
        core_shares=core,
        clr_cols=clr_cols,
        deltas=dict(zip(core, deltas.tolist(), strict=True)),
    )
    out = df[META_COLS + spec.features + clr_cols + OUTCOME_COLS].copy()
    out["Is_Test"] = out["Is_Test"].astype(bool)
    return out, spec


def spec_from_frame(features: pd.DataFrame) -> FeatureSpec:
    """Recover the column lists from a saved feature table."""
    indicators = [c for c in features.columns if c.startswith("Spends_On_")]
    clr_cols = [c for c in features.columns if c.endswith("_CLR")]
    core = [c.replace("_CLR", "_Share") for c in clr_cols]
    return FeatureSpec(indicators=indicators, core_shares=core, clr_cols=clr_cols, deltas={})


def zero_rates(features: pd.DataFrame) -> pd.Series:
    return (features[SHARE_COLS] == 0).mean().sort_values()
