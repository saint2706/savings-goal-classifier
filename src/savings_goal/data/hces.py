"""HCES 2022-23 household table, for the external-validity check on the leakage mechanism.

HCES (NSO, MoSPI) records consumption but not income, so the savings label cannot be
rebuilt on it. What it can test is the *ingredient* of the composition leak: whether
food spend is still tightly tied to household size 11 years after IHDS-II. The input
is the level-wise parquet dump of the public microdata at
https://github.com/advaitmoharir/hces_2022 (``04_clean/``), which follows the
questionnaire's own MPCE construction: item groups below are the dump's ``2_mpce.do``.
Only ``level_01`` (sector), ``level_03`` (household size) and ``level_14`` (item values)
are read. These files are third-party copies of public data; they stay untracked.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FOOD_MONTHLY = (129, 139, 159, 179)
FOOD_WEEKLY = (169, 219, 239, 249, 199, 189, 269, 279, 289, 299)  # x 30/7
CONSUMABLES_MONTHLY = (349, 459, 479, 429, 519, 499, 439, 529)
CONSUMABLES_WEEKLY = (309, 319, 329)  # x 30/7
CONSUMABLES_ANNUAL = (409, 419, 899)  # x 30/365
# Questionnaire D (durables) is all annual, x 30/365.


def _sum_items(items: pd.DataFrame, qn: str, codes: tuple[int, ...], scale: float) -> pd.Series:
    m = (items["questionnaire_num"] == qn) & items["item_code"].isin(codes)
    return items.loc[m].groupby("hhid")["value"].sum() * scale


def household_table(directory: Path) -> pd.DataFrame:
    """One row per household: sector, size, weight, monthly food and total consumption (Rs)."""
    items = pd.read_parquet(directory / "level_14.parquet")
    parts = {
        "food": [(("F", FOOD_MONTHLY, 1.0)), ("F", FOOD_WEEKLY, 30 / 7)],
        "total_other": [
            ("C", CONSUMABLES_MONTHLY, 1.0),
            ("C", CONSUMABLES_WEEKLY, 30 / 7),
            ("C", CONSUMABLES_ANNUAL, 30 / 365),
        ],
    }
    food = sum(
        _sum_items(items, *p).reindex(items["hhid"].unique(), fill_value=0) for p in parts["food"]
    )
    other = sum(
        _sum_items(items, *p).reindex(items["hhid"].unique(), fill_value=0)
        for p in parts["total_other"]
    )
    durables = (
        items.loc[items["questionnaire_num"] == "D"].groupby("hhid")["value"].sum() * 30 / 365
    )
    hh = pd.DataFrame({"food": food, "other": other}).join(durables.rename("durables")).fillna(0.0)
    hh["total"] = hh["food"] + hh["other"] + hh["durables"]
    size = pd.read_parquet(directory / "level_03.parquet", columns=["hhid", "hh_size"])
    sec = pd.read_parquet(directory / "level_01.parquet", columns=["hhid", "sector", "multiplier"])
    out = hh.reset_index(names="hhid").merge(size, on="hhid").merge(sec, on="hhid")
    out["weight"] = out["multiplier"] / 100
    out["sector"] = out["sector"].map({1: "rural", 2: "urban"})
    return out[["hhid", "sector", "hh_size", "weight", "food", "total"]]


def weighted_mpce(hh: pd.DataFrame) -> dict[str, float]:
    """Monthly per-capita expenditure by sector: sum(w * total) / sum(w * size)."""
    return {
        str(s): float((g["weight"] * g["total"]).sum() / (g["weight"] * g["hh_size"]).sum())
        for s, g in hh.groupby("sector")
    }


def oracle_reconstruction(food: pd.Series, total: pd.Series, size: pd.Series) -> dict[str, float]:
    """How well ``k * size / food_share`` recovers total spend (no income needed).

    ``k`` is the median food spend per person. Returns the Spearman correlation of the
    implied and actual totals and the median absolute log error.
    """
    m = (food > 0) & (total > 0) & (size > 0)
    f, c, s = (x[m].to_numpy(dtype=float) for x in (food, total, size))
    k = float(np.median(f / s))
    implied = k * s / (f / c)
    rho = float(pd.Series(implied).corr(pd.Series(c), method="spearman"))
    return {"spearman": rho, "median_abs_log_error": float(np.median(np.abs(np.log(implied / c))))}
