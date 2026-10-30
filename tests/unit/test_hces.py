"""The HCES household table, on hand-built level files with known answers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from savings_goal.data import hces


@pytest.fixture
def levels(tmp_path: Path) -> Path:
    # hh 1: rural, 2 people, food 1000 (monthly code 129) + 700 weekly (code 169 -> x30/7 = 3000),
    # consumables 500 (monthly code 349), durables 3650 a year (-> 300 a month).
    # hh 2: urban, 4 people, food 2000 only.
    items = pd.DataFrame(
        {
            "hhid": [1, 1, 1, 1, 2],
            "questionnaire_num": ["F", "F", "C", "D", "F"],
            "item_code": [129, 169, 349, 501, 129],
            "value": [1000, 700, 500, 3650, 2000],
        }
    )
    items.to_parquet(tmp_path / "level_14.parquet")
    pd.DataFrame({"hhid": [1, 2], "hh_size": [2, 4]}).to_parquet(tmp_path / "level_03.parquet")
    pd.DataFrame({"hhid": [1, 2], "sector": [1, 2], "multiplier": [100, 300]}).to_parquet(
        tmp_path / "level_01.parquet"
    )
    return tmp_path


def test_household_table_follows_the_mpce_construction(levels: Path) -> None:
    hh = hces.household_table(levels).set_index("hhid")
    assert hh.loc[1, "food"] == pytest.approx(1000 + 700 * 30 / 7)
    assert hh.loc[1, "total"] == pytest.approx(1000 + 3000 + 500 + 300)
    assert hh.loc[2, "total"] == pytest.approx(2000)
    assert list(hh["sector"]) == ["rural", "urban"]
    assert list(hh["weight"]) == [1.0, 3.0]


def test_weighted_mpce_is_total_over_persons(levels: Path) -> None:
    mpce = hces.weighted_mpce(hces.household_table(levels))
    assert mpce["rural"] == pytest.approx(4800 / 2)
    assert mpce["urban"] == pytest.approx(2000 / 4)


def test_oracle_reconstruction_is_exact_when_food_is_proportional_to_size() -> None:
    rng = np.random.default_rng(0)
    size = pd.Series(rng.integers(1, 9, 500).astype(float))
    total = pd.Series(rng.lognormal(9, 0.5, 500))
    food = 800 * size  # k * size exactly
    rec = hces.oracle_reconstruction(food, total, size)
    assert rec["spearman"] == pytest.approx(1.0)
    assert rec["median_abs_log_error"] == pytest.approx(0.0, abs=1e-9)
