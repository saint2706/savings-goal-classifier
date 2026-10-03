"""Project-wide constants: paths, column families, seeds."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset"
RESULTS = ROOT / "results"
HOUSEHOLDS_PATH = DATASET / "households.parquet"
FEATURES_PATH = DATASET / "features.parquet"

SEED = 42
THRESHOLD = 0.20
N_SPLITS = 5

EXPENSE_CATEGORIES: list[str] = [
    "Groceries",
    "Eating_Out",
    "Utilities",
    "Rent",
    "Transport",
    "Healthcare",
    "Education",
    "Entertainment",
    "Insurance",
    "Clothing_Footwear",
    "Miscellaneous",
]
SHARE_COLS: list[str] = [f"{c}_Share" for c in EXPENSE_CATEGORIES]

# IDPSU is the survey's unique primary-sampling-unit id. PSUID on its own repeats
# across districts (only 39 distinct values), so it cannot be the CV group key.
GROUP_COL = "IDPSU"
TARGET = "Goal_Met"
WEIGHT = "WT"

CATEGORICALS: list[str] = ["Occupation", "Area_Type", "Caste_Group", "Religion"]

# The size family: Groceries rupees are close to proportional to household size,
# so these columns plus Groceries_Share and income can approximately recover
# total spend. See evaluation/leakage.py.
SIZE_FAMILY: list[str] = ["Household_Size", "Age_Dependents", "Dependency_Ratio"]
DEMOGRAPHIC_NUMERIC: list[str] = [
    "Log_Income",
    "Household_Size",
    "Age_Dependents",
    "Dependency_Ratio",
    "Head_Age",
    "Max_Adult_Education",
]
DEBT_COLS: list[str] = ["Debt_To_Income", "Has_Debt", "Debt_Missing"]

# Columns that reconstruct the label. Never features.
LEAKAGE_COLS: list[str] = [
    *EXPENSE_CATEGORIES,
    "Category_Total",
    "COTOTAL",
    "Savings",
    "Savings_Rate",
]
