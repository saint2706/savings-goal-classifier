"""Build an analysis-ready household dataset from IHDS-II (ICPSR 36151, DS0002).

Replaces the synthetic Kaggle dataset with real survey data. See
``walkthrough/dataset_construction.md`` for the reasoning behind every mapping and
threshold in this file.

Usage:
    python src/build_dataset.py --tsv path/to/36151-0002-Data.tsv \
        --out dataset/households.csv

Key facts this script encodes (all verified against 36151-0002-Codebook.pdf):

* ``CO1X``-``CO33`` are 30-day recall; ``CO34``-``CO52`` are annual recall.
  ``12 * monthly + annual`` reproduces IHDS's own ``COTOTAL`` (median error 0).
* ``INCOME`` and ``COTOTAL`` are both annual rupees, so they are directly
  comparable without rescaling.
* ``CO4X`` is kerosene -- a fuel, not a food. It belongs in Utilities even
  though it sits inside the ``CO*X`` food block.
* ``CO18`` is paan/tobacco/intoxicants, not eating out. Eating out is ``CO20``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Expense category mapping: IHDS consumption items -> the project's 11 categories
# --------------------------------------------------------------------------
# Values are (monthly_items, annual_items). Monthly items are multiplied by 12
# so that every category is expressed in annual rupees, matching INCOME.
CATEGORY_MAP: dict[str, tuple[list[str], list[str]]] = {
    # Food staples (CO1X-CO14X minus kerosene) plus the non-durable food items
    "Groceries": (
        [f"CO{i}X" for i in (1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)]
        + ["CO15", "CO16", "CO17", "CO19"],
        [],
    ),
    "Eating_Out": (["CO20"], []),
    # Kerosene + household fuel + electricity + telephone/cable/internet
    "Utilities": (["CO4X", "CO21", "CO22", "CO24"], []),
    "Rent": (["CO30"], []),
    # Conveyance + petrol/maintenance (monthly), vehicle purchase (annual)
    "Transport": (["CO28", "CO29"], ["CO45"]),
    # Out-patient (monthly), in-patient + therapeutic appliances (annual)
    "Healthcare": (["CO33"], ["CO34", "CO46"]),
    "Education": ([], ["CO35", "CO36", "CO37"]),
    # Entertainment (monthly), recreation goods + vacations (annual)
    "Entertainment": (["CO23"], ["CO43", "CO51"]),
    "Insurance": ([], ["CO50"]),
    "Clothing_Footwear": ([], ["CO38", "CO39"]),
    "Miscellaneous": (
        ["CO18", "CO25", "CO26", "CO27", "CO31", "CO32"],
        ["CO40", "CO41", "CO42", "CO44", "CO47", "CO48", "CO49", "CO52"],
    ),
}

EXPENSE_CATEGORIES = list(CATEGORY_MAP)

URBAN4_LABELS = {
    0: "Metro_Urban",
    1: "Other_Urban",
    2: "Developed_Village",
    3: "Less_Developed_Village",
}

CASTE_LABELS = {
    1: "Brahmin",
    2: "Forward_Caste",
    3: "OBC",
    4: "Scheduled_Caste",
    5: "Scheduled_Tribe",
    6: "Other",
}

RELIGION_LABELS = {
    1: "Hindu",
    2: "Muslim",
    3: "Christian",
    4: "Sikh",
    5: "Buddhist",
    6: "Jain",
    7: "Tribal",
    8: "Other",
    9: "None",
}

# Worker-count columns -> occupation label. Order is the tie-break priority:
# a household with equal salaried and farm workers is classed Salaried, because
# the salaried income stream is the one that dominates household cash flow.
OCCUPATION_SOURCES = [
    ("NWKSALARY", "Salaried"),
    ("NWKBUSINESS", "Business"),
    ("NWKNONAG", "Non_Ag_Labour"),
    ("NWKFARM", "Farm"),
    ("NWKAGLAB", "Ag_Labour"),
]

ID_COLS = ["STATEID", "DISTID", "PSUID", "HHID", "HHSPLITID", "IDHH"]

DEMOGRAPHIC_COLS = [
    "NPERSONS", "NADULTM", "NADULTF", "NCHILDM", "NCHILDF",
    "NELDERM", "NELDERF", "NTEENM", "NTEENF",
    "MHEADAGE", "FHEADAGE", "HHEDUC",
    "URBAN2011", "URBAN4_2011", "METRO", "ID11", "ID13",
    "DB5", "DB9C", "DB9D", "DB9E", "DB9G", "DB9H", "DB9I",
    "POOR", "ASSETS", "WT", "INCOME", "COTOTAL", "COPC", "INCOMEPC",
] + [src for src, _ in OCCUPATION_SOURCES]

SAVINGS_INDICATORS = {
    "DB9C": "Has_Securities",
    "DB9D": "Has_Fixed_Deposit",
    "DB9E": "Has_Bank_Savings",
    "DB9G": "Has_Post_Office_Account",
    "DB9H": "Has_Pension_LIC",
    "DB9I": "Has_Gold_Jewellery",
}


def load_raw(tsv: Path) -> pd.DataFrame:
    """Read the DS0002 household file, keeping only the columns we need."""
    monthly = sorted({c for m, _ in CATEGORY_MAP.values() for c in m})
    annual = sorted({c for _, a in CATEGORY_MAP.values() for c in a})
    keep = set(monthly + annual + ID_COLS + DEMOGRAPHIC_COLS)

    df = pd.read_csv(tsv, sep="\t", usecols=lambda c: c in keep, low_memory=False)

    missing = keep - set(df.columns)
    if missing:
        raise KeyError(f"IHDS file is missing expected columns: {sorted(missing)}")

    # ICPSR encodes some missing values as non-numeric text; coerce everything.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the 52 IHDS consumption items into 11 annual-rupee categories."""
    out = pd.DataFrame(index=df.index)
    for name, (monthly, annual) in CATEGORY_MAP.items():
        total = pd.Series(0.0, index=df.index)
        if monthly:
            total = total + df[monthly].fillna(0).sum(axis=1) * 12
        if annual:
            total = total + df[annual].fillna(0).sum(axis=1)
        out[name] = total
    return out


def derive_occupation(df: pd.DataFrame) -> pd.Series:
    """Label each household by the worker type contributing the most workers."""
    counts = df[[src for src, _ in OCCUPATION_SOURCES]].fillna(0)
    labels = [label for _, label in OCCUPATION_SOURCES]
    # idxmax returns the FIRST maximum, so column order encodes the tie-break.
    winner = counts.to_numpy().argmax(axis=1)
    occupation = pd.Series([labels[i] for i in winner], index=df.index)
    return occupation.where(counts.sum(axis=1) > 0, "No_Regular_Worker")


def build(tsv: Path, threshold: float) -> pd.DataFrame:
    raw = load_raw(tsv)
    n_raw = len(raw)

    cats = build_categories(raw)
    df = pd.concat([raw[ID_COLS + DEMOGRAPHIC_COLS], cats], axis=1)

    # ---- filter to households where the target is computable ----------------
    # Income <= 0 makes a savings *rate* undefined (IHDS reports negative farm
    # income for ~9% of households, driving ~1.5% of totals to zero or below).
    # COTOTAL is missing for a small number of households.
    before = len(df)
    df = df[(df["INCOME"] > 0) & df["COTOTAL"].notna() & (df["COTOTAL"] > 0)].copy()
    dropped = before - len(df)

    # ---- target -------------------------------------------------------------
    # Normative benchmark: a household is "on track" if it retains at least
    # `threshold` of its income after all recorded consumption. Unlike the
    # synthetic dataset there is no self-declared savings goal, so the goal is
    # imposed externally and is identical for every household.
    df["Savings"] = df["INCOME"] - df["COTOTAL"]
    df["Savings_Rate"] = df["Savings"] / df["INCOME"]
    df["Goal_Met"] = (df["Savings_Rate"] >= threshold).astype(int)

    # ---- features -----------------------------------------------------------
    # Expense COMPOSITION shares, not expense-to-income ratios. The shares sum
    # to 1 by construction, so they say nothing about consumption relative to
    # income and therefore cannot reconstruct the target. Expense-to-income
    # ratios WOULD reconstruct it (sum <= 1 - threshold is exactly Goal_Met),
    # which is why they are deliberately not built here.
    total_expense = df[EXPENSE_CATEGORIES].sum(axis=1)
    for cat in EXPENSE_CATEGORIES:
        df[f"{cat}_Share"] = df[cat] / total_expense

    df["Log_Income"] = np.log(df["INCOME"])
    df["Head_Age"] = df["MHEADAGE"].fillna(df["FHEADAGE"])
    df["Dependents"] = (
        df[["NCHILDM", "NCHILDF", "NELDERM", "NELDERF"]].fillna(0).sum(axis=1)
    )
    df["Household_Size"] = df["NPERSONS"]
    df["Occupation"] = derive_occupation(df)
    df["Area_Type"] = df["URBAN4_2011"].map(URBAN4_LABELS)
    df["Caste_Group"] = df["ID13"].map(CASTE_LABELS)
    df["Religion"] = df["ID11"].map(RELIGION_LABELS)
    df["Max_Adult_Education"] = df["HHEDUC"]
    # Debt enters as a stock scaled by income. IHDS has no monthly loan-repayment
    # flow, so this is the nearest available stand-in for Loan_Repayment.
    df["Debt_To_Income"] = df["DB5"].fillna(0) / df["INCOME"]

    for src, name in SAVINGS_INDICATORS.items():
        df[name] = df[src]

    ordered = (
        ID_COLS
        + ["WT", "INCOME", "Log_Income", "Household_Size", "Dependents", "Head_Age",
           "Max_Adult_Education", "Occupation", "Area_Type", "Caste_Group", "Religion",
           "Debt_To_Income"]
        + [f"{c}_Share" for c in EXPENSE_CATEGORIES]
        + list(SAVINGS_INDICATORS.values())
        + EXPENSE_CATEGORIES
        + ["COTOTAL", "Savings", "Savings_Rate", "Goal_Met"]
    )
    df = df[ordered]

    print(f"raw households:        {n_raw:,}")
    print(f"dropped (income<=0 / missing consumption): {dropped:,}")
    print(f"final households:      {len(df):,}")
    print(f"Goal_Met rate @ {threshold:.0%}:  {df['Goal_Met'].mean():.4f}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", required=True, type=Path, help="36151-0002-Data.tsv")
    parser.add_argument("--out", default=Path("dataset/households.csv"), type=Path)
    parser.add_argument(
        "--threshold",
        default=0.20,
        type=float,
        help="Normative savings-rate benchmark defining Goal_Met (default 0.20)",
    )
    args = parser.parse_args()

    df = build(args.tsv, args.threshold)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
