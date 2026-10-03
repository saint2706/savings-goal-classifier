"""Peer benchmarks for "which spending is recoverable", with uncertainty.

The original analysis compared each at-risk household with the *median*
on-track household in its income decile, clipped negative excess to zero and
summed (so the total could only grow), and reported shares rather than rupees.
Here: *mean* benchmarks, signed excess (negative = spends less than peers),
benchmarks computed in rupees as well as shares, and PSU-clustered bootstrap
intervals. A category is only called a lever if its interval excludes zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from savings_goal.config import EXPENSE_CATEGORIES, GROUP_COL, SHARE_COLS, THRESHOLD
from savings_goal.evaluation.metrics import bootstrap_indices, percentile_ci


def signed_excess(
    df: pd.DataFrame, decile: str = "Income_Decile"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per at-risk household: share excess (pp) and rupee excess vs mean on-track peer.

    ``df`` needs the 11 rupee categories, the 11 shares, ``Goal_Met`` and ``decile``.
    """
    on = df[df["Goal_Met"] == 1]
    risk = df[df["Goal_Met"] == 0]
    share_bench = on.groupby(decile)[SHARE_COLS].mean().reindex(risk[decile])
    rupee_bench = on.groupby(decile)[EXPENSE_CATEGORIES].mean().reindex(risk[decile])
    share_x = (
        pd.DataFrame(
            risk[SHARE_COLS].to_numpy() - share_bench.to_numpy(),
            columns=EXPENSE_CATEGORIES,
            index=risk.index,
        )
        * 100
    )
    rupee_x = pd.DataFrame(
        risk[EXPENSE_CATEGORIES].to_numpy() - rupee_bench.to_numpy(),
        columns=EXPENSE_CATEGORIES,
        index=risk.index,
    )
    return share_x, rupee_x


@dataclass(frozen=True)
class BenchmarkTable:
    table: pd.DataFrame
    n_boot: int


def benchmark_with_ci(
    df: pd.DataFrame, n_boot: int = 500, seed: int = 42, decile: str = "Income_Decile"
) -> BenchmarkTable:
    """Mean signed excess per category with 95% PSU-bootstrap CIs (benchmarks re-estimated per draw)."""
    share_x, rupee_x = signed_excess(df, decile)
    groups = df[GROUP_COL].to_numpy()
    cats = EXPENSE_CATEGORIES

    draws = []
    for idx in bootstrap_indices(groups, n_boot, seed):
        s, r = signed_excess(df.iloc[idx], decile)
        draws.append(np.concatenate([s.mean().to_numpy(), r.mean().to_numpy()]))
    draws_arr = np.stack(draws)
    k = len(cats)
    rows = []
    for j, c in enumerate(cats):
        s_lo, s_hi = percentile_ci(draws_arr[:, j])
        r_lo, r_hi = percentile_ci(draws_arr[:, k + j])
        rows.append(
            {
                "category": c,
                "mean excess share (pp)": share_x[c].mean(),
                "share CI low": s_lo,
                "share CI high": s_hi,
                "mean excess (Rs/yr)": rupee_x[c].mean(),
                "Rs CI low": r_lo,
                "Rs CI high": r_hi,
                "share of at-risk above peer (Rs)": float((rupee_x[c] > 0).mean()),
                "sample total excess (Rs crore/yr)": rupee_x[c].sum() / 1e7,
                "CI excludes zero (Rs)": bool(r_lo > 0 or r_hi < 0),
            }
        )
    table = (
        pd.DataFrame(rows).set_index("category").sort_values("mean excess (Rs/yr)", ascending=False)
    )
    return BenchmarkTable(table=table, n_boot=n_boot)


def gap_closure(
    df: pd.DataFrame, decile: str = "Income_Decile", threshold: float = THRESHOLD
) -> pd.DataFrame:
    """Rupee gap to the benchmark vs positive peer excess, per at-risk household."""
    _, rupee_x = signed_excess(df, decile)
    risk = df.loc[rupee_x.index]
    out = pd.DataFrame(index=risk.index)
    out["gap"] = (threshold - risk["Savings_Rate"]) * risk["INCOME"]
    out["recoverable"] = rupee_x.clip(lower=0).sum(axis=1)
    out["closable"] = out["recoverable"] >= out["gap"]
    return out
