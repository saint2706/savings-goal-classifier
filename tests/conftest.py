"""Synthetic stand-in for ICPSR DS0002 (the real file cannot be redistributed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from savings_goal.data.build import (
    CATEGORY_MAP,
    DEMOGRAPHIC_COLS,
    ID_COLS,
    OCCUPATION_SOURCES,
    annual_items,
    monthly_items,
)

N_HOUSEHOLDS = 400


def synthetic_raw(n: int = N_HOUSEHOLDS, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(index=range(n))
    n_psu = 40
    psu = rng.integers(0, n_psu, n)
    df["STATEID"] = 1 + psu // 10
    df["DISTID"] = 1 + psu // 5
    df["PSUID"] = 1 + psu % 5
    df["IDPSU"] = 10000 + psu
    df["HHID"] = np.arange(n) * 10
    df["HHSPLITID"] = 1
    df["IDHH"] = 1_000_000 + np.arange(n)
    df["SURVEY"] = 2

    size = rng.integers(1, 9, n)
    children = rng.binomial(size, 0.3)
    teens = rng.binomial(size - children, 0.15)
    adults = size - children - teens
    elders = rng.binomial(adults, 0.2)
    df["NPERSONS"] = size
    df["NCHILDM"], df["NCHILDF"] = children // 2, children - children // 2
    df["NTEENM"], df["NTEENF"] = teens, 0
    df["NADULTM"], df["NADULTF"] = adults // 2, adults - adults // 2
    df["NELDERM"], df["NELDERF"] = elders, 0
    df["MHEADAGE"] = rng.integers(20, 80, n).astype(float)
    df["FHEADAGE"] = rng.integers(20, 80, n).astype(float)
    df.loc[rng.random(n) < 0.2, "MHEADAGE"] = np.nan
    df["HHEDUC"] = rng.integers(0, 16, n)
    df["URBAN2011"] = rng.integers(0, 2, n)
    df["URBAN4_2011"] = rng.integers(0, 4, n)
    df["METRO"] = 0
    df["ID11"] = rng.integers(1, 10, n)
    df["ID13"] = rng.integers(1, 7, n).astype(float)
    df.loc[rng.random(n) < 0.05, "ID13"] = np.nan
    for c in ("DB9C", "DB9D", "DB9E", "DB9G", "DB9H", "DB9I"):
        df[c] = rng.integers(0, 2, n)
    df["POOR"] = rng.integers(0, 2, n)
    df["ASSETS"] = rng.integers(0, 30, n)
    df["WT"] = rng.uniform(500, 15000, n)
    for src, _ in OCCUPATION_SOURCES:
        df[src] = rng.integers(0, 3, n)

    for c in monthly_items():
        base = 400 if c.endswith("X") else 150
        df[c] = np.where(rng.random(n) < 0.3, 0.0, rng.gamma(2.0, base / 2, n) * size / 4)
    for c in annual_items():
        df[c] = np.where(rng.random(n) < 0.5, 0.0, rng.gamma(1.5, 2000, n))
    df["CO1X"] = rng.gamma(4.0, 150, n) * size + 1.0  # every household eats
    total = sum(
        df[m].fillna(0).sum(axis=1) * 12 + df[a].fillna(0).sum(axis=1)
        for m, a in CATEGORY_MAP.values()
    )
    df["COTOTAL"] = total * rng.uniform(0.995, 1.005, n)
    df["INCOME"] = total * rng.lognormal(0.1, 0.5, n)
    df["COPC"] = df["COTOTAL"] / size
    df["INCOMEPC"] = df["INCOME"] / size
    df["DB5"] = np.where(rng.random(n) < 0.5, 0.0, rng.gamma(1.0, 20000, n))
    df.loc[rng.random(n) < 0.08, "DB5"] = np.nan
    # Edge cases the build must filter out.
    df.loc[0, "INCOME"] = -500.0
    df.loc[1, "COTOTAL"] = np.nan
    df.loc[2, "INCOME"] = 0.0
    # An exact occupation tie (2 salaried, 2 farm) and an all-zero household.
    for src, _ in OCCUPATION_SOURCES:
        df.loc[[3, 4], src] = 0
    df.loc[3, ["NWKSALARY", "NWKFARM"]] = 2
    # The "None" religion label that pandas used to read back as NaN.
    df.loc[5, "ID11"] = 9
    expected = set(ID_COLS + DEMOGRAPHIC_COLS + monthly_items() + annual_items())
    assert expected <= set(df.columns)
    return df


def write_tsv(df: pd.DataFrame, path: Path) -> Path:
    # ICPSR writes missing values as a single space.
    df.astype(object).where(df.notna(), " ").to_csv(path, sep="\t", index=False)
    return path


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return synthetic_raw()


@pytest.fixture(scope="session")
def raw_tsv(tmp_path_factory: pytest.TempPathFactory, raw_df: pd.DataFrame) -> Path:
    return write_tsv(raw_df, tmp_path_factory.mktemp("raw") / "36151-0002-Data.tsv")


@pytest.fixture(scope="session")
def households(raw_tsv: Path) -> pd.DataFrame:
    from savings_goal.data.build import build

    df, _ = build(raw_tsv)
    return df.to_pandas()


@pytest.fixture(scope="session")
def features(households: pd.DataFrame) -> pd.DataFrame:
    from savings_goal.features.engineer import engineer

    return engineer(households)[0]
