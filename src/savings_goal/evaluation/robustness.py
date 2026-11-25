"""Measurement-error sensitivity and external non-accounting criterion checks.

Tests how strongly ranking performance and the edge over poorest-first depend on
extreme dissaving (``COTOTAL > 2 * INCOME``), multiplicative income under-reporting,
and whether the savings-adequacy score tracks formal financial-instrument holdings
outside the consumption identity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

from savings_goal.config import SEED, THRESHOLD
from savings_goal.evaluation.metrics import (
    bootstrap_indices,
    capture_at_budget,
    expected_calibration_error,
    percentile_ci,
)

Array = NDArray[Any]


def dissaver_sensitivity(
    df: pd.DataFrame,
    p: Array,
    income_score: Array,
    cutoffs: Sequence[float | None] = (None, 2.0, 1.5, 1.0),
    budget: float = 0.25,
    groups: Array | None = None,
    n_boot: int = 300,
    seed: int = SEED,
) -> pd.DataFrame:
    """Evaluate discrimination and budget recall when extreme dissavers are excluded.

    ``cutoffs`` caps ``COTOTAL / INCOME``; ``None`` keeps the full sample.
    """
    ratio = df["COTOTAL"].to_numpy(dtype=float) / df["INCOME"].to_numpy(dtype=float)
    y_all = df["Goal_Met"].to_numpy(dtype=float)
    p_arr = np.asarray(p, dtype=float)
    inc_arr = np.asarray(income_score, dtype=float)

    rows: list[dict[str, Any]] = []
    for c in cutoffs:
        mask = np.ones(len(df), dtype=bool) if c is None else ratio <= c
        y = y_all[mask]
        ar = 1.0 - y
        pm = p_arr[mask]
        im = inc_arr[mask]
        r_mod, pr_mod = capture_at_budget(ar, 1.0 - pm, budget)
        r_inc, _ = capture_at_budget(ar, im, budget)
        row: dict[str, Any] = {
            "subset": "Full sample" if c is None else f"COTOTAL <= {c:g}x INCOME",
            "max_ratio": float("inf") if c is None else float(c),
            "n": int(mask.sum()),
            "share_of_sample": float(mask.mean()),
            "at_risk_rate": float(ar.mean()),
            "ROC-AUC": float(roc_auc_score(y, pm)),
            "ECE": expected_calibration_error(y, pm),
            "recall_model": r_mod,
            "precision_model": pr_mod,
            "recall_income": r_inc,
            "recall_diff_pp": 100.0 * (r_mod - r_inc),
        }
        if groups is not None:
            g_sub = np.asarray(groups)[mask]
            diffs: list[float] = []
            for idx in bootstrap_indices(g_sub, n_boot=n_boot, seed=seed):
                r1, _ = capture_at_budget(ar[idx], 1.0 - pm[idx], budget)
                r2, _ = capture_at_budget(ar[idx], im[idx], budget)
                diffs.append(100.0 * (r1 - r2))
            lo, hi = percentile_ci(np.asarray(diffs))
            row["diff_ci_low"] = lo
            row["diff_ci_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def underreporting_stress(
    df: pd.DataFrame,
    p: Array,
    income_score: Array,
    multipliers: Mapping[str, Array],
    threshold: float = THRESHOLD,
    budget: float = 0.25,
) -> pd.DataFrame:
    """Recompute the savings label under scaled true income and score the original rankings."""
    inc = df["INCOME"].to_numpy(dtype=float)
    co = df["COTOTAL"].to_numpy(dtype=float)
    p_arr = np.asarray(p, dtype=float)
    inc_arr = np.asarray(income_score, dtype=float)

    rows: list[dict[str, Any]] = []
    for name, mult in multipliers.items():
        adj_inc = inc * np.asarray(mult, dtype=float)
        adj_sr = (adj_inc - co) / adj_inc
        adj_y = (adj_sr >= threshold).astype(float)
        adj_ar = 1.0 - adj_y
        r_mod, pr_mod = capture_at_budget(adj_ar, 1.0 - p_arr, budget)
        r_inc, _ = capture_at_budget(adj_ar, inc_arr, budget)
        rows.append(
            {
                "scenario": name,
                "at_risk_rate": float(adj_ar.mean()),
                "ROC-AUC": float(roc_auc_score(adj_y, p_arr)),
                "recall_model": r_mod,
                "precision_model": pr_mod,
                "recall_income": r_inc,
                "recall_diff_pp": 100.0 * (r_mod - r_inc),
            }
        )
    return pd.DataFrame(rows)


def external_criterion_check(
    df: pd.DataFrame,
    scores: Mapping[str, Array],
    target_cols: Sequence[str],
) -> pd.DataFrame:
    """ROC-AUC of candidate scores against external non-accounting savings-instrument flags."""
    rows: list[dict[str, Any]] = []
    for col in target_cols:
        y_ext = df[col].fillna(0).astype(int).to_numpy()
        row: dict[str, Any] = {"target": col, "rate": float(y_ext.mean())}
        for name, sc in scores.items():
            row[f"auc_{name}"] = float(roc_auc_score(y_ext, np.asarray(sc, dtype=float)))
        rows.append(row)
    return pd.DataFrame(rows)
