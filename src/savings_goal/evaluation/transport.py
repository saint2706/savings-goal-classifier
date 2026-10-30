"""Does the model travel? Train on one slice of IHDS-II, score a structurally different one.

A newer survey with income and consumption for the same households does not exist in
public form, so out-of-time validation is not possible. These tests are the available
proxy for "the population changes under the model": whole states held out (different
prices, labour markets, institutions) and the rural/urban divide in both directions.
Each reports discrimination (ROC-AUC), calibration-in-the-large (mean predicted minus
observed on-track rate) and the 25%-budget at-risk recall of the model against a
poorest-first income rule on the held-out slice.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from savings_goal.evaluation.metrics import capture_at_budget, expected_calibration_error


def shift_test(
    df: pd.DataFrame,
    train: NDArray[np.bool_],
    test: NDArray[np.bool_],
    model: Any,
    cols: list[str],
    budget: float = 0.25,
) -> dict[str, float]:
    """Fit on ``train`` rows, score ``test`` rows (boolean masks over ``df``)."""
    m = clone(model).fit(df.loc[train, cols], df.loc[train, "Goal_Met"])
    p = m.predict_proba(df.loc[test, cols])[:, 1]
    y = df.loc[test, "Goal_Met"].to_numpy()
    at_risk = 1 - y
    rng = np.random.default_rng(0)  # only breaks exact income ties
    income_score = -df.loc[test, "INCOME"].to_numpy() + rng.random(len(y)) * 1e-6
    model_recall, model_prec = capture_at_budget(at_risk, 1 - p, budget)
    income_recall, _ = capture_at_budget(at_risk, income_score, budget)
    return {
        "n_train": float(train.sum()),
        "n_test": float(test.sum()),
        "at_risk_rate": float(at_risk.mean()),
        "ROC-AUC": float(roc_auc_score(y, p)),
        "ECE": expected_calibration_error(y, p),
        "calibration_in_the_large": float(p.mean() - y.mean()),
        f"recall@{budget:.0%} model": model_recall,
        f"precision@{budget:.0%} model": model_prec,
        f"recall@{budget:.0%} income rule": income_recall,
    }


def leave_group_out(
    df: pd.DataFrame, group_col: str, model: Any, cols: list[str], n_splits: int = 5
) -> pd.DataFrame:
    """Hold out whole groups (e.g. states) in ``n_splits`` balanced folds; one row per fold."""
    rows: list[dict[str, float]] = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits).split(df, groups=df[group_col])):
        train = np.zeros(len(df), dtype=bool)
        test = np.zeros(len(df), dtype=bool)
        train[tr], test[te] = True, True
        meta: dict[str, float] = {"fold": k, "groups_held_out": df.loc[test, group_col].nunique()}
        rows.append(meta | shift_test(df, train, test, model, cols))
    return pd.DataFrame(rows)
