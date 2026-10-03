"""A1: can the leakage-free feature set recover total spend, and hence the label?

Food rupees are close to proportional to household size. If so, then

    Groceries ~= k * Household_Size  and  Groceries_Share = Groceries / COTOTAL
    =>  COTOTAL ~= k * Household_Size / Groceries_Share
    =>  Savings_Rate ~= 1 - (k * Household_Size / Groceries_Share) / INCOME

so ``Household_Size`` + ``Groceries_Share`` + ``INCOME`` together approximately
reconstruct the target that the share representation was chosen to hide. The
three tests here measure how much of the model's skill comes from that route.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from savings_goal.config import SIZE_FAMILY
from savings_goal.evaluation.cv import cross_validate_grouped, grouped_cv
from savings_goal.evaluation.metrics import fold_metrics


def ablation_sets(
    features: list[str], indicators: list[str], shares: list[str]
) -> dict[str, list[str]]:
    """T1 feature sets: all / no Groceries_Share / no size family / no shares+indicators."""
    drop_shares = set(shares) | set(indicators)
    return {
        "all features": features,
        "no Groceries_Share": [c for c in features if c != "Groceries_Share"],
        "no size family": [c for c in features if c not in SIZE_FAMILY],
        "no shares or indicators (deployable)": [c for c in features if c not in drop_shares],
    }


def t1_ablation(
    df: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    feature_sets: Mapping[str, list[str]],
    make_model: Callable[[list[str]], Any],
) -> pd.DataFrame:
    """Grouped-CV ROC-AUC / macro-F1 / at-risk PR-AUC for each feature set."""
    rows = []
    for name, cols in feature_sets.items():
        folds = cross_validate_grouped(
            make_model(cols), df[cols], y, groups, fold_metrics(with_proba=False), grouped_cv()
        )
        rows.append(
            {
                "feature set": name,
                "n features": len(cols),
                "ROC-AUC": folds["ROC-AUC"].mean(),
                "ROC-AUC sd": folds["ROC-AUC"].std(),
                "F1 (macro)": folds["F1 (macro)"].mean(),
                "PR-AUC (at-risk)": folds["PR-AUC (at-risk)"].mean(),
            }
        )
    out = pd.DataFrame(rows).set_index("feature set")
    full_auc = float(out["ROC-AUC"].iloc[0])  # "all features" is the first set
    out["AUC retained"] = out["ROC-AUC"] / full_auc
    return out


def t2_size_r2(households: pd.DataFrame) -> dict[str, float]:
    """R^2 of log(Groceries) on log(Household_Size), households with positive food spend."""
    m = (households["Groceries"] > 0) & (households["Household_Size"] > 0)
    x = np.log(households.loc[m, "Household_Size"].to_numpy(dtype=float))
    yv = np.log(households.loc[m, "Groceries"].to_numpy(dtype=float))
    slope, intercept = np.polyfit(x, yv, 1)
    resid = yv - (intercept + slope * x)
    r2 = 1 - resid.var() / yv.var()
    return {"r2": float(r2), "elasticity": float(slope), "n": int(m.sum())}


@dataclass(frozen=True)
class OracleResult:
    auc: float
    k: float
    n_used: int
    n_excluded: int
    agreement_at_threshold: float


def implied_savings_rate(households: pd.DataFrame, k: float) -> pd.Series:
    share = households["Groceries_Share"]
    implied_spend = k * households["Household_Size"] / share
    return 1 - implied_spend / households["INCOME"]


def t3_oracle(households: pd.DataFrame, threshold: float = 0.20) -> OracleResult:
    """Model-free oracle: rank households by the savings rate implied by size, share, income.

    ``k`` is the median food spend per person. Rows with ``Groceries_Share == 0``
    (division by zero -> inf) or a NaN share are excluded and counted.
    """
    k = float((households["Groceries"] / households["Household_Size"]).median())
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = implied_savings_rate(households, k)
    m = np.isfinite(sr.to_numpy()) & households["Groceries_Share"].gt(0).to_numpy()
    y = households["Goal_Met"].to_numpy()
    auc = roc_auc_score(y[m], sr.to_numpy()[m])
    agree = float(((sr.to_numpy()[m] >= threshold).astype(int) == y[m]).mean())
    return OracleResult(
        auc=float(auc),
        k=k,
        n_used=int(m.sum()),
        n_excluded=int((~m).sum()),
        agreement_at_threshold=agree,
    )
