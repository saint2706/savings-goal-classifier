"""Subgroup error, calibration, sensitive-attribute ablation, and allocation audits.

Evaluates discrimination, calibration-in-the-large (CITL), expected calibration error
(ECE), error rates (at-risk FNR and FPR), and fixed-budget selection rates across
demographic and socioeconomic strata, both with and without sensitive attributes
(``Caste_Group`` and ``Religion``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

from savings_goal.config import SEED
from savings_goal.evaluation.metrics import (
    bootstrap_indices,
    capture_at_budget,
    expected_calibration_error,
    percentile_ci,
    top_by_budget,
    weighted_mean,
)

Array = NDArray[Any]


def subgroup_audit(
    df: pd.DataFrame,
    y: Array,
    p: Array,
    group_cols: Sequence[str],
    budget: float = 0.25,
    weights: Array | None = None,
    reach_score: Array | None = None,
    min_n: int = 50,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Error, calibration, and selection rates for each stratum in ``group_cols``.

    ``y`` is ``Goal_Met`` (1 = on track) and ``p`` is ``P(on track)``. Risk selection
    ranks by ``1 - p``; optional ``reach_score`` compares who gets contacted under a
    reachability ranking at the same ``budget``.
    """
    y_arr = np.asarray(y, dtype=float)
    p_arr = np.asarray(p, dtype=float)
    at_risk = 1.0 - y_arr
    w = np.ones_like(y_arr) if weights is None else np.asarray(weights, dtype=float)
    w_total = float(w.sum())

    top_risk, _ = top_by_budget(1.0 - p_arr, budget)
    sel_risk = np.zeros(len(y_arr), dtype=bool)
    sel_risk[top_risk] = True

    sel_reach: NDArray[np.bool_] | None = None
    if reach_score is not None:
        top_reach, _ = top_by_budget(np.asarray(reach_score, dtype=float), budget)
        sel_reach = np.zeros(len(y_arr), dtype=bool)
        sel_reach[top_reach] = True

    rows: list[dict[str, Any]] = []
    for col in group_cols:
        series = df[col].to_numpy()
        for val in pd.unique(series):
            idx = np.flatnonzero(series == val)
            if len(idx) < min_n:
                continue
            yg, pg, arg, wg = y_arr[idx], p_arr[idx], at_risk[idx], w[idx]
            auc = float(roc_auc_score(yg, pg)) if len(np.unique(yg)) > 1 else float("nan")
            pred_on = pg >= threshold
            fnr = float(np.sum(pred_on & (arg == 1)) / max(1.0, float(arg.sum())))
            fpr = float(np.sum((~pred_on) & (arg == 0)) / max(1.0, float((arg == 0).sum())))
            row: dict[str, Any] = {
                "group_col": col,
                "subgroup": str(val),
                "n": len(idx),
                "weighted_share": float(wg.sum() / w_total),
                "at_risk_rate": float(arg.mean()),
                "weighted_at_risk_rate": weighted_mean(arg, wg),
                "ROC-AUC": auc,
                "CITL": float(pg.mean() - yg.mean()),
                "weighted_CITL": weighted_mean(pg - yg, wg),
                "ECE": expected_calibration_error(yg, pg),
                "at_risk_FNR": fnr,
                "at_risk_FPR": fpr,
                "selection_rate": float(sel_risk[idx].mean()),
            }
            if sel_reach is not None:
                row["reach_selection_rate"] = float(sel_reach[idx].mean())
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["group_col", "subgroup"]).reset_index(drop=True)


def sensitive_ablation_summary(
    y: Array,
    p_with: Array,
    p_without: Array,
    groups: Array,
    budget: float = 0.25,
    weights: Array | None = None,
    n_boot: int = 300,
    seed: int = SEED,
) -> dict[str, Any]:
    """Compare models with and without sensitive attributes, with PSU-bootstrap CIs."""
    y_arr = np.asarray(y, dtype=float)
    pw = np.asarray(p_with, dtype=float)
    po = np.asarray(p_without, dtype=float)
    ar = 1.0 - y_arr

    def _stats(p_score: Array) -> dict[str, float]:
        r_b, pr_b = capture_at_budget(ar, 1.0 - p_score, budget)
        out = {
            "ROC-AUC": float(roc_auc_score(y_arr, p_score)),
            "ECE": expected_calibration_error(y_arr, p_score),
            f"recall@{budget:.0%}": r_b,
            f"precision@{budget:.0%}": pr_b,
        }
        if weights is not None:
            rw, prw = capture_at_budget(ar, 1.0 - p_score, budget, weights=weights)
            out[f"weighted_recall@{budget:.0%}"] = rw
            out[f"weighted_precision@{budget:.0%}"] = prw
        return out

    auc_diffs: list[float] = []
    rec_diffs: list[float] = []
    for idx in bootstrap_indices(groups, n_boot=n_boot, seed=seed):
        if len(np.unique(y_arr[idx])) < 2:
            continue
        auc_diffs.append(
            float(roc_auc_score(y_arr[idx], pw[idx]) - roc_auc_score(y_arr[idx], po[idx]))
        )
        r1, _ = capture_at_budget(ar[idx], 1.0 - pw[idx], budget)
        r2, _ = capture_at_budget(ar[idx], 1.0 - po[idx], budget)
        rec_diffs.append(r1 - r2)

    auc_lo, auc_hi = percentile_ci(np.asarray(auc_diffs))
    rec_lo, rec_hi = percentile_ci(np.asarray(rec_diffs))
    s_with, s_without = _stats(pw), _stats(po)
    return {
        "with_sensitive": s_with,
        "without_sensitive": s_without,
        "diff_auc": s_with["ROC-AUC"] - s_without["ROC-AUC"],
        "diff_auc_ci": [auc_lo, auc_hi],
        f"diff_recall@{budget:.0%}": s_with[f"recall@{budget:.0%}"]
        - s_without[f"recall@{budget:.0%}"],
        f"diff_recall@{budget:.0%}_ci": [rec_lo, rec_hi],
    }


def decile_allocation(
    income: Array,
    risk_score: Array,
    reach_score: Array,
    budget: float = 0.25,
    weights: Array | None = None,
) -> pd.DataFrame:
    """Selection rate within each income decile for risk vs reachability targeting."""
    inc = np.asarray(income, dtype=float)
    deciles = pd.qcut(inc, 10, labels=False, duplicates="drop")
    w = np.ones_like(inc) if weights is None else np.asarray(weights, dtype=float)

    top_r, _ = top_by_budget(np.asarray(risk_score, dtype=float), budget, weights)
    top_e, _ = top_by_budget(np.asarray(reach_score, dtype=float), budget, weights)
    sel_r = np.zeros(len(inc), dtype=bool)
    sel_e = np.zeros(len(inc), dtype=bool)
    sel_r[top_r] = True
    sel_e[top_e] = True

    rows: list[dict[str, Any]] = []
    for d in sorted(np.unique(deciles)):
        m = deciles == d
        wm = w[m]
        rows.append(
            {
                "Income_Decile": int(d),
                "n": int(m.sum()),
                "risk_selection_rate": weighted_mean(sel_r[m].astype(float), wm),
                "reach_selection_rate": weighted_mean(sel_e[m].astype(float), wm),
            }
        )
    return pd.DataFrame(rows)
