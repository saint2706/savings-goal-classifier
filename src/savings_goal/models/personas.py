"""Spending-pattern clusters on an ILR basis, with stability and sensitivity checks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from numpy.typing import NDArray
from scipy.stats import chi2, kruskal
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from savings_goal.config import SEED
from savings_goal.features.transforms import (
    ilr,
    multiplicative_replacement,
    zero_replacement_deltas,
)

FloatArray = NDArray[np.float64]


def ilr_coordinates(
    shares: pd.DataFrame, factor: float = 0.65, reference: pd.DataFrame | None = None
) -> FloatArray:
    """ILR of the sub-composition, zeros replaced at ``factor`` x min positive (from ``reference``)."""
    deltas = zero_replacement_deltas(shares if reference is None else reference, factor)
    return ilr(multiplicative_replacement(shares, deltas))


def standardise(X: FloatArray) -> FloatArray:
    out: FloatArray = StandardScaler().fit_transform(X)
    return out


def kmeans_sweep(
    X: FloatArray, ks: range = range(2, 9), seed: int = SEED, sample: int = 5000
) -> pd.DataFrame:
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        lab = km.labels_
        rows.append(
            {
                "k": k,
                "silhouette": silhouette_score(
                    X, lab, sample_size=min(sample, len(X)), random_state=seed
                ),
                "davies_bouldin": davies_bouldin_score(X, lab),
                "calinski_harabasz": calinski_harabasz_score(X, lab),
                "inertia": km.inertia_,
            }
        )
    return pd.DataFrame(rows)


def fit_kmeans(X: FloatArray, k: int, seed: int = SEED, n_init: int = 20) -> KMeans:
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(X)


def bootstrap_stability(X: FloatArray, k: int, n_boot: int = 30, seed: int = SEED) -> FloatArray:
    """ARI between the reference partition and partitions refit on bootstrap resamples."""
    ref = fit_kmeans(X, k, seed).labels_
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(X), len(X))
        km = KMeans(n_clusters=k, n_init=5, random_state=seed + b + 1).fit(X[idx])
        out[b] = adjusted_rand_score(ref, km.predict(X))
    return out


def zero_pattern(shares: pd.DataFrame) -> NDArray[np.intp]:
    key = (shares == 0).astype(int).astype(str).agg("".join, axis=1)
    codes: NDArray[np.intp] = key.factorize()[0]
    return codes


def delta_sensitivity(
    shares: pd.DataFrame,
    k: int,
    factors: tuple[float, ...] = (0.1, 0.25, 0.5, 0.65, 1.0),
    baseline: float = 0.65,
    seed: int = SEED,
) -> pd.DataFrame:
    """Re-cluster with the zero-replacement delta at each multiple of the min positive value."""
    base_lab = fit_kmeans(standardise(ilr_coordinates(shares, baseline)), k, seed).labels_
    zp = zero_pattern(shares)
    rows = []
    for f in factors:
        X = standardise(ilr_coordinates(shares, f))
        lab = fit_kmeans(X, k, seed).labels_
        rows.append(
            {
                "delta factor": f,
                "silhouette": silhouette_score(X, lab, sample_size=5000, random_state=seed),
                "ARI vs baseline delta": adjusted_rand_score(base_lab, lab),
                "ARI vs zero pattern": adjusted_rand_score(zp, lab),
            }
        )
    return pd.DataFrame(rows)


def income_association(persona: pd.Series, income: pd.Series, n_bins: int = 10) -> dict[str, float]:
    """Is persona membership just income? NMI with income deciles + Kruskal-Wallis on income."""
    decile = pd.qcut(income, n_bins, labels=False)
    groups = [income[persona == p].to_numpy() for p in sorted(persona.unique())]
    h, p = kruskal(*groups)
    eps2 = (h - len(groups) + 1) / (len(income) - len(groups))  # epsilon-squared effect size
    return {
        "NMI vs income decile": float(normalized_mutual_info_score(persona, decile)),
        "Kruskal-Wallis H": float(h),
        "Kruskal-Wallis p": float(p),
        "epsilon squared": float(eps2),
    }


def persona_income_interaction(df: pd.DataFrame) -> dict[str, Any]:
    """Logistic Goal_Met ~ decile + persona (+ decile x persona); likelihood-ratio tests.

    Replaces the "within-decile spread / raw spread" ratio, which is not a test.
    """
    d = df[["Goal_Met", "Income_Decile", "Persona"]].copy()
    base = smf.logit("Goal_Met ~ C(Income_Decile)", d).fit(disp=0)
    main = smf.logit("Goal_Met ~ C(Income_Decile) + C(Persona)", d).fit(disp=0)
    full = smf.logit("Goal_Met ~ C(Income_Decile) * C(Persona)", d).fit(disp=0, maxiter=200)
    lr_p = 2 * (main.llf - base.llf)
    lr_i = 2 * (full.llf - main.llf)
    df_p = main.df_model - base.df_model
    df_i = full.df_model - main.df_model
    persona_or = np.exp(main.params.filter(like="C(Persona)"))
    return {
        "LR persona | decile": float(lr_p),
        "df persona": int(df_p),
        "p persona | decile": float(chi2.sf(lr_p, df_p)),
        "LR interaction": float(lr_i),
        "df interaction": int(df_i),
        "p interaction": float(chi2.sf(lr_i, df_i)),
        "persona odds ratios (vs persona 0), income-adjusted": persona_or.to_dict(),
        "pseudo R2 decile only": float(base.prsquared),
        "pseudo R2 + persona": float(main.prsquared),
    }
