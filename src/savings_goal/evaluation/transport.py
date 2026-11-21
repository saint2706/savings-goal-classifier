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

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from savings_goal.config import GROUP_COL, SEED
from savings_goal.evaluation.metrics import capture_at_budget, expected_calibration_error

Array = NDArray[Any]


def recalibrate_intercept(
    p_cal: Array, y_cal: Array, p_eval: Array, eps: float = 1e-6
) -> tuple[Array, float]:
    """Shift predicted log-odds by a constant so the calibration mean matches ``y_cal``.

    Strictly monotone in ``p_eval``, so ROC-AUC and rankings are unchanged. Extreme
    probabilities and degenerate calibration target means are clipped to ``[eps, 1 - eps]``.
    """
    p_c = np.clip(np.asarray(p_cal, dtype=float), eps, 1.0 - eps)
    p_e = np.clip(np.asarray(p_eval, dtype=float), eps, 1.0 - eps)
    target = float(np.clip(np.mean(np.asarray(y_cal, dtype=float)), eps, 1.0 - eps))
    eta_c = np.log(p_c / (1.0 - p_c))
    eta_e = np.log(p_e / (1.0 - p_e))
    bound = max(20.0, 2.0 * float(np.log((1.0 - eps) / eps)) + 1.0)
    alpha = float(brentq(lambda a: float(np.mean(expit(eta_c + a))) - target, -bound, bound))
    return np.asarray(expit(eta_e + alpha), dtype=float), alpha


def shift_test(
    df: pd.DataFrame,
    train: NDArray[np.bool_],
    test: NDArray[np.bool_],
    model: Any,
    cols: list[str],
    budget: float = 0.25,
    seed: int = SEED,
) -> dict[str, float]:
    """Fit on ``train`` rows, score ``test`` rows (boolean masks over ``df``)."""
    m = clone(model).fit(df.loc[train, cols], df.loc[train, "Goal_Met"])
    p = m.predict_proba(df.loc[test, cols])[:, 1]
    y = df.loc[test, "Goal_Met"].to_numpy()
    at_risk = 1 - y
    rng = np.random.default_rng(seed)  # only breaks exact income ties
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


def recalibration_shift_test(
    df: pd.DataFrame,
    train: NDArray[np.bool_],
    test: NDArray[np.bool_],
    model: Any,
    cols: list[str],
    group_col: str = GROUP_COL,
    cal_frac: float = 0.10,
    seed: int = SEED,
) -> dict[str, float]:
    """Fit on ``train``, recalibrate intercept on ``cal_frac`` of target groups, score the rest."""
    m = clone(model).fit(df.loc[train, cols], df.loc[train, "Goal_Met"])
    test_df = df.loc[test]
    p = np.asarray(m.predict_proba(test_df[cols])[:, 1], dtype=float)
    y = test_df["Goal_Met"].to_numpy()
    groups = test_df[group_col].to_numpy()
    uniq = np.unique(groups)
    perm = np.random.default_rng(seed).permutation(uniq)
    n_cal_groups = max(1, min(len(uniq) - 1, round(cal_frac * len(uniq))))
    is_cal = np.isin(groups, perm[:n_cal_groups])
    is_eval = ~is_cal
    p_recal, alpha = recalibrate_intercept(p[is_cal], y[is_cal], p[is_eval])
    p_eval, y_eval = p[is_eval], y[is_eval]
    return {
        "n_cal": float(is_cal.sum()),
        "n_cal_groups": float(n_cal_groups),
        "n_eval": float(is_eval.sum()),
        "n_eval_groups": float(len(uniq) - n_cal_groups),
        "intercept_shift": alpha,
        "ROC-AUC": float(roc_auc_score(y_eval, p_eval)),
        "ece_before": expected_calibration_error(y_eval, p_eval),
        "ece_after": expected_calibration_error(y_eval, p_recal),
        "citl_before": float(p_eval.mean() - y_eval.mean()),
        "citl_after": float(p_recal.mean() - y_eval.mean()),
        "brier_before": float(brier_score_loss(y_eval, p_eval)),
        "brier_after": float(brier_score_loss(y_eval, p_recal)),
    }


def _run_group_folds(
    df: pd.DataFrame,
    group_col: str,
    n_splits: int,
    fn: Callable[[NDArray[np.bool_], NDArray[np.bool_]], dict[str, float]],
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits).split(df, groups=df[group_col])):
        train = np.zeros(len(df), dtype=bool)
        test = np.zeros(len(df), dtype=bool)
        train[tr], test[te] = True, True
        meta: dict[str, float] = {"fold": k, "groups_held_out": df.loc[test, group_col].nunique()}
        rows.append(meta | fn(train, test))
    return pd.DataFrame(rows)


def leave_group_out(
    df: pd.DataFrame,
    group_col: str,
    model: Any,
    cols: list[str],
    n_splits: int = 5,
    seed: int = SEED,
) -> pd.DataFrame:
    """Hold out whole groups (e.g. states) in ``n_splits`` balanced folds; one row per fold."""
    return _run_group_folds(
        df, group_col, n_splits, lambda tr, te: shift_test(df, tr, te, model, cols, seed=seed)
    )


def leave_group_recalibration(
    df: pd.DataFrame,
    group_col: str,
    model: Any,
    cols: list[str],
    n_splits: int = 5,
    psu_col: str = GROUP_COL,
    cal_frac: float = 0.10,
    seed: int = SEED,
) -> pd.DataFrame:
    """Recalibrate each held-out ``GroupKFold`` slice on ``cal_frac`` of its target PSUs."""
    return _run_group_folds(
        df,
        group_col,
        n_splits,
        lambda tr, te: recalibration_shift_test(
            df, tr, te, model, cols, group_col=psu_col, cal_frac=cal_frac, seed=seed
        ),
    )
