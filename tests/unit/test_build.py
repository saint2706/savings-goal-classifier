from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pandera.errors
import polars as pl
import pytest

from savings_goal.config import EXPENSE_CATEGORIES, SHARE_COLS
from savings_goal.data.build import (
    CATEGORY_MAP,
    OUTPUT_COLUMNS,
    build,
    reconcile_totals,
    scan_raw,
    write,
)
from savings_goal.data.schema import validate_households


def test_filters_unusable_households(raw_tsv: Path, raw_df: pd.DataFrame) -> None:
    df, report = build(raw_tsv)
    assert report.n_raw == len(raw_df)
    assert report.dropped == 3  # negative income, missing COTOTAL, zero income
    assert df.height == len(raw_df) - 3
    assert df["INCOME"].min() > 0
    assert "raw households" in str(report)


def test_columns_and_schema(households: pd.DataFrame, raw_tsv: Path) -> None:
    assert list(households.columns) == OUTPUT_COLUMNS
    df, _ = build(raw_tsv)
    validate_households(df)


def test_annualisation_matches_category_map(households: pd.DataFrame, raw_df: pd.DataFrame) -> None:
    raw = raw_df.set_index("IDHH").loc[households["IDHH"]]
    for cat, (monthly, annual) in CATEGORY_MAP.items():
        expected = raw[monthly].fillna(0).sum(axis=1) * 12 + raw[annual].fillna(0).sum(axis=1)
        np.testing.assert_allclose(households[cat].to_numpy(), expected.to_numpy())


def test_shares_sum_to_one_and_target(households: pd.DataFrame) -> None:
    np.testing.assert_allclose(households[SHARE_COLS].sum(axis=1), 1.0)
    np.testing.assert_allclose(
        households["Category_Total"], households[EXPENSE_CATEGORIES].sum(axis=1)
    )
    rate = (households["INCOME"] - households["COTOTAL"]) / households["INCOME"]
    np.testing.assert_allclose(households["Savings_Rate"], rate)
    assert (households["Goal_Met"] == (rate >= 0.20).astype(int)).all()


def test_occupation_tie_break_and_no_worker(households: pd.DataFrame, raw_df: pd.DataFrame) -> None:
    by_id = households.set_index("IDHH")
    tie = by_id.loc[raw_df.loc[3, "IDHH"]]
    assert tie["Occupation"] == "Salaried"
    assert tie["Occupation_Tie"] == 1
    assert by_id.loc[raw_df.loc[4, "IDHH"], "Occupation"] == "No_Regular_Worker"


def test_debt_missing_is_not_zero(households: pd.DataFrame) -> None:
    missing = households["Debt_Missing"] == 1
    assert missing.any()
    assert households.loc[missing, "Debt_To_Income"].isna().all()
    assert households.loc[~missing, "Debt_To_Income"].notna().all()


def test_age_dependents_and_labels(households: pd.DataFrame, raw_df: pd.DataFrame) -> None:
    raw = raw_df.set_index("IDHH").loc[households["IDHH"]]
    deps = raw[["NCHILDM", "NCHILDF", "NELDERM", "NELDERF"]].sum(axis=1)
    np.testing.assert_array_equal(households["Age_Dependents"], deps)
    assert "No_Religion" in set(households["Religion"])
    assert households["Caste_Group"].isna().any()  # missing ID13 stays missing


def test_schema_rejects_bad_shares(raw_tsv: Path) -> None:
    df, _ = build(raw_tsv)
    bad = df.with_columns((pl.col("Rent_Share") + 0.5).alias("Rent_Share"))
    with pytest.raises((pandera.errors.SchemaError, pandera.errors.SchemaErrors)):
        validate_households(bad)


def test_missing_columns_raise(tmp_path: Path, raw_df: pd.DataFrame) -> None:
    path = tmp_path / "bad.tsv"
    raw_df.drop(columns=["CO20"]).to_csv(path, sep="\t", index=False)
    with pytest.raises(KeyError, match="CO20"):
        scan_raw(path)


def test_reconcile_totals(raw_tsv: Path) -> None:
    df, _ = build(raw_tsv)
    rec = reconcile_totals(df)
    assert rec["within_1pct"] == pytest.approx(1.0)
    assert 0.9 < rec["label_agreement"] <= 1.0


@pytest.mark.parametrize("suffix", [".parquet", ".csv"])
def test_write_streams_to_disk(raw_tsv: Path, tmp_path: Path, suffix: str) -> None:
    out = tmp_path / f"households{suffix}"
    report = write(raw_tsv, out)
    assert out.exists()
    assert report.n_final == build(raw_tsv)[1].n_final
