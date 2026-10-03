"""SHAP on grouped feature families, with a permutation-importance cross-check.

Uses ``shap.TreeExplainer`` directly: shap >= 0.50 parses XGBoost 3.x's
``base_score`` correctly, so the old ``pred_contribs`` workaround is gone.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd
import shap
from numpy.typing import NDArray
from sklearn.base import clone

from savings_goal.config import CATEGORICALS, DEBT_COLS, SHARE_COLS, SIZE_FAMILY


def feature_families(feature_names: list[str]) -> dict[str, list[str]]:
    """Map each (post-one-hot) column to a family; families partition the columns."""
    fam: dict[str, list[str]] = {
        "Income": [],
        "Household size family": [],
        "Spending mix (shares)": [],
        "Participation indicators": [],
        "Debt": [],
        "Head age & education": [],
        "Social/geo (categoricals)": [],
    }
    for c in feature_names:
        if c == "Log_Income":
            fam["Income"].append(c)
        elif c in SIZE_FAMILY:
            fam["Household size family"].append(c)
        elif c in SHARE_COLS:
            fam["Spending mix (shares)"].append(c)
        elif c.startswith("Spends_On_"):
            fam["Participation indicators"].append(c)
        elif c in DEBT_COLS:
            fam["Debt"].append(c)
        elif c in ("Head_Age", "Max_Adult_Education"):
            fam["Head age & education"].append(c)
        elif any(c.startswith(f"{k}_") for k in CATEGORICALS):
            fam["Social/geo (categoricals)"].append(c)
        else:
            raise KeyError(f"no family for column {c!r}")
    return {k: v for k, v in fam.items() if v}


def raw_families(columns: list[str]) -> dict[str, list[str]]:
    """Same families on the raw (pre-one-hot) columns, for permutation importance."""
    expanded = [f"{c}_x" if c in CATEGORICALS else c for c in columns]
    fam = feature_families(expanded)
    return {
        k: [c.removesuffix("_x") if c.removesuffix("_x") in CATEGORICALS else c for c in v]
        for k, v in fam.items()
    }


def tree_shap(model: Any, X: pd.DataFrame) -> tuple[NDArray[np.float64], float, float]:
    """SHAP values (log-odds), base value, and max additivity error vs the raw margin."""
    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(X), dtype=float)
    base = float(np.ravel(explainer.expected_value)[0])
    margin = model.predict(X, output_margin=True)
    err = float(np.abs(values.sum(axis=1) + base - margin).max())
    return values, base, err


def grouped_shap(
    values: NDArray[np.float64], names: list[str], families: Mapping[str, list[str]]
) -> pd.DataFrame:
    """Mean |sum of SHAP within family| per row: SHAP is additive, so this is the family's effect.

    Also reports the naive sum of per-column mean |SHAP| for comparison; the two
    differ when columns in a family push in opposite directions (shares do).
    """
    idx = {n: i for i, n in enumerate(names)}
    rows = []
    for fam, cols in families.items():
        cols_i = [idx[c] for c in cols]
        joint = np.abs(values[:, cols_i].sum(axis=1)).mean()
        naive = np.abs(values[:, cols_i]).mean(axis=0).sum()
        rows.append(
            {
                "family": fam,
                "n columns": len(cols),
                "grouped mean |SHAP|": joint,
                "sum of per-column mean |SHAP|": naive,
            }
        )
    out = pd.DataFrame(rows).set_index("family")
    out["share of grouped total"] = out["grouped mean |SHAP|"] / out["grouped mean |SHAP|"].sum()
    return out.sort_values("grouped mean |SHAP|", ascending=False)


def grouped_permutation_importance(
    fitted: Any,
    X: pd.DataFrame,
    y: pd.Series,
    families: Mapping[str, list[str]],
    score: Callable[[NDArray[Any], NDArray[np.float64]], float],
    n_repeats: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Drop in ``score`` when all columns of a family are permuted jointly (same row order)."""
    rng = np.random.default_rng(seed)
    base = score(y.to_numpy(), fitted.predict_proba(X)[:, 1])
    rows = []
    for fam, cols in families.items():
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[cols] = X[cols].to_numpy()[perm]
            drops.append(base - score(y.to_numpy(), fitted.predict_proba(Xp)[:, 1]))
        rows.append(
            {"family": fam, "importance": float(np.mean(drops)), "sd": float(np.std(drops))}
        )
    out = pd.DataFrame(rows).set_index("family")
    out["share"] = out["importance"].clip(lower=0) / out["importance"].clip(lower=0).sum()
    return out.sort_values("importance", ascending=False)


def interaction_decomposition(
    model: Any, X: pd.DataFrame, names: list[str], chunk: int = 2000
) -> tuple[pd.Series, pd.DataFrame]:
    """Main-effect vs interaction share of attribution over the whole of ``X``.

    Returns totals (main, interaction, share) and the ranked interaction pairs.
    Computed with XGBoost's exact TreeSHAP interaction values in chunks so the
    full test set fits in memory.
    """
    import xgboost as xgb

    booster = model.get_booster()
    d = len(names)
    abs_sum = np.zeros((d, d))
    n = 0
    for start in range(0, len(X), chunk):
        part = X.iloc[start : start + chunk]
        inter = booster.predict(xgb.DMatrix(part), pred_interactions=True)[:, :-1, :-1]
        abs_sum += np.abs(inter).sum(axis=0)
        n += len(part)
    mean_abs = abs_sum / n
    main = float(np.trace(mean_abs))
    off = mean_abs.copy()
    np.fill_diagonal(off, 0)
    inter_total = float(off.sum())
    pairs = pd.DataFrame(off, index=names, columns=names).stack().reset_index()
    pairs.columns = pd.Index(["level_0", "level_1", "mean |interaction SHAP|"])
    pairs = pairs[pairs["level_0"] < pairs["level_1"]]
    pairs["mean |interaction SHAP|"] *= 2  # both (i, j) and (j, i)
    pairs = pairs.sort_values("mean |interaction SHAP|", ascending=False).reset_index(drop=True)
    totals = pd.Series(
        {
            "main effects": main,
            "interactions": inter_total,
            "interaction share": inter_total / (main + inter_total),
            "n rows": n,
        }
    )
    return totals, pairs


def coefficient_stability(
    make_model: Callable[[], Any], X: pd.DataFrame, y: pd.Series, n_parts: int = 5, seed: int = 42
) -> pd.DataFrame:
    """Fit the same linear model on disjoint subsamples; report coefficient sd and sign flips."""
    rng = np.random.default_rng(seed)
    parts = np.array_split(rng.permutation(len(X)), n_parts)
    coefs = []
    for idx in parts:
        m = clone(make_model()).fit(X.iloc[idx], y.iloc[idx])
        est = m[-1]
        names = m[:-1].get_feature_names_out()
        coefs.append(pd.Series(est.coef_[0], index=names))
    c = pd.concat(coefs, axis=1)
    out = pd.DataFrame({"mean": c.mean(axis=1), "sd": c.std(axis=1)})
    out["sd/|mean|"] = out["sd"] / out["mean"].abs()
    out["flips_sign"] = (c > 0).any(axis=1) & (c < 0).any(axis=1)
    return out
