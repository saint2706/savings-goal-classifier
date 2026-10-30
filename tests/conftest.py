"""Fixtures built on the synthetic DS0002 in ``savings_goal.data.synthetic``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from savings_goal.data.synthetic import synthetic_raw, write_tsv


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
