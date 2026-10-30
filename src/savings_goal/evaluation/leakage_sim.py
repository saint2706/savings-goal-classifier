"""When does a spending-share representation leak? A simulation anyone can rerun.

If food rupees follow ``F = k * size**e * exp(sigma * eps)`` and the budget is ``C``,
the food share is ``s = F / C`` and the model-free oracle ``C_hat = k * size / s``
equals ``C * k * size / F``. Its log error is therefore ``(1 - e) log(size) - sigma * eps``:
the oracle is accurate exactly when food spend is tightly tied to household size, which
is a property of the survey, not of this data set. This module simulates the oracle's
ROC-AUC against the label ``C / Y <= 1 - threshold`` over a grid of ``sigma`` and ``e``,
so a reader can place any survey on the curve from one regression (the residual
standard deviation of log food on log size).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from savings_goal.config import SEED, THRESHOLD


def oracle_auc(
    sigma: float,
    elasticity: float,
    n: int = 40_000,
    prevalence: float = 0.319,
    budget_sd: float = 0.6,
    seed: int = SEED,
) -> float:
    """ROC-AUC of the size x food-share x income oracle on simulated households."""
    rng = np.random.default_rng(seed)
    size = 1 + rng.poisson(3.2, n)
    log_ratio_mu = np.log(1 - THRESHOLD) - budget_sd * _z(prevalence)
    log_c_over_y = log_ratio_mu + budget_sd * rng.standard_normal(n)  # log(C / Y)
    # log(k * size / F) with k fixed at 1 and F = size**e * exp(sigma * eps)
    log_err = (1 - elasticity) * np.log(size) - sigma * rng.standard_normal(n)
    y = (log_c_over_y <= np.log(1 - THRESHOLD)).astype(int)
    # score = implied savings rate; higher implied rate <=> lower implied log(C / Y)
    return float(roc_auc_score(y, -(log_c_over_y + log_err)))


def _z(p: float) -> float:
    """Standard normal quantile of ``p``."""
    from scipy.stats import norm

    return float(norm.ppf(p))


def auc_grid(
    sigmas: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5),
    elasticities: tuple[float, ...] = (1.0, 0.63),
) -> pd.DataFrame:
    """Long table of oracle AUC over ``sigma`` x ``elasticity``."""
    return pd.DataFrame(
        [
            {"sigma": s, "elasticity": e, "oracle_auc": oracle_auc(s, e)}
            for e in elasticities
            for s in sigmas
        ]
    )


def residual_sd(food: pd.Series, size: pd.Series) -> dict[str, float]:
    """Elasticity and residual sd of log(food) on log(size): where a survey sits on the curve."""
    m = (food > 0) & (size > 0)
    x = np.log(size[m].to_numpy(dtype=float))
    y = np.log(food[m].to_numpy(dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (intercept + slope * x)
    return {
        "elasticity": float(slope),
        "sigma": float(resid.std()),
        "r2": float(1 - resid.var() / y.var()),
    }
