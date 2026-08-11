# Phase 2 — Feature Engineering

**Source:** [README § Phase 2 — Feature Engineering](../README.md#phase-2--feature-engineering)
**Notebook:** [`notebooks/02_feature_engineering.ipynb`](../notebooks/02_feature_engineering.ipynb)
**Builds on:** [Phase 1 — Data Understanding](phase1.md)

Phase 1 established two facts this phase acts on directly: raw expense columns are almost entirely explained by `Income` (r up to 0.99), and `Disposable_Income` / `Desired_Savings` / `Desired_Savings_Percentage` leak the target and must be excluded. Phase 2 turns those findings into an actual feature matrix — deciding how spending should be represented, how the two categorical columns should be encoded, which features need scaling, and whether the resulting feature set has any redundancy left to clean up.

---

## Research questions & answers

| # | Question | Answer |
|---|---|---|
| 1 | Do expense-to-income ratios generalize better across income levels than raw expense values? | Yes, decisively. Raw expense↔income correlation of 0.79–0.99 collapses to ≈0 once converted to `category / Income`; mean raw `Groceries` spend rises ~8× from the lowest to highest income quartile while `Groceries_Ratio` stays flat at ≈0.125 in every quartile. |
| 2 | How should `Occupation` and `City_Tier` be encoded? | Both one-hot encoded. `Occupation` is purely nominal (mean income is ~identical across all 4 groups). `City_Tier` has a real, monotonic cost-of-living order (`Rent_Ratio` steps 0.30 → 0.20 → 0.15 from Tier_1 to Tier_3) but stays one-hot by default given only 3 levels; ordinal encoding is a defensible cheaper alternative for tree-based models only. |
| 3 | Which features require scaling, and does that depend on the downstream model? | `Income` (and to a lesser extent `Age`) need scaling for linear/distance/NN models (Logistic Regression, SVM, KNN, neural nets); tree-based models (Decision Tree, Random Forest, XGBoost) are scale-invariant, so scaling is unnecessary — but harmless — for them. |
| 4 | Are any features redundant or highly collinear? | Minimal redundancy after the ratio transform — every VIF ≈ 1.0 except `Dependents`/`Education_Ratio` at ≈1.73, well below any concerning threshold (5, let alone 10). No features need to be dropped. |

The rest of this document walks through *how* the notebook arrives at each answer, cell by cell, and why each analysis step was chosen.

---

## Notebook walkthrough

### Cell 0 (markdown) — Title and scope

States the four Phase 2 questions and explicitly names the dependency on Phase 1: the leakage columns are excluded from the start, and the income↔expense correlation from Phase 1 is the direct motivation for Question 1.

### Cell 1 (code) — Imports, data load, and target reconstruction

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

pd.set_option("display.max_columns", 40)
sns.set_theme(style="whitegrid")

DATA_PATH = "../dataset/data.csv"
df = pd.read_csv(DATA_PATH)

LEAKAGE_COLS = ["Disposable_Income", "Desired_Savings", "Desired_Savings_Percentage"]
df["Goal_Met"] = (df["Disposable_Income"] >= df["Desired_Savings"]).astype(int)

expense_cols = [
    "Rent", "Loan_Repayment", "Insurance", "Groceries", "Transport", "Eating_Out",
    "Entertainment", "Utilities", "Healthcare", "Education", "Miscellaneous",
]
print(f"Shape: {df.shape}")
df.head()
```

**What it does:** Sets up the same EDA stack as Phase 1, plus `statsmodels` (needed later for the collinearity check in Question 4). Loads the raw CSV, immediately reconstructs `Goal_Met` from its Phase 0/1-verified formula, and names `LEAKAGE_COLS` explicitly as a constant so it's unambiguous which columns are off-limits for the rest of the notebook.

**Why:** Naming `LEAKAGE_COLS` as a variable right at the top — rather than just remembering to avoid three column names by convention — makes the exclusion an explicit, checkable artifact of the code instead of a rule that only lives in a comment. `statsmodels` is imported here rather than lazily later so all dependencies for the notebook are visible in one place, matching Phase 1's pattern.

### Cell 2 (markdown) — Question 1 setup

Restates the Phase 1 finding (raw expenses correlate with income at up to r = 0.99) and frames the hypothesis under test: dividing each expense by `Income` should strip out the income-driven component and leave the actual *spending mix* signal behind.

### Cell 3 (code) — Building ratio features and comparing correlation with income

```python
ratio_cols = []
for c in expense_cols:
    ratio_col = f"{c}_Ratio"
    df[ratio_col] = df[c] / df["Income"]
    ratio_cols.append(ratio_col)

raw_corr = df[expense_cols].corrwith(df["Income"]).rename("raw_corr_with_income")
ratio_corr = df[ratio_cols].corrwith(df["Income"]).rename("ratio_corr_with_income")
ratio_corr.index = expense_cols  # align for side-by-side comparison

comparison = pd.concat([raw_corr, ratio_corr], axis=1)
comparison["abs_reduction"] = comparison["raw_corr_with_income"].abs() - comparison["ratio_corr_with_income"].abs()
comparison.sort_values("raw_corr_with_income", ascending=False)
```

**What it does:** Creates one `{category}_Ratio` column per expense category (`expense / Income`), then computes each *raw* column's correlation with `Income` side by side with each *ratio* column's correlation with `Income`. The `ratio_corr.index = expense_cols` line re-labels the ratio series with the plain category names so the two series can be concatenated into one directly comparable table.

**Why:** This is the actual experiment behind Question 1, not just an assertion — rather than claiming ratios generalize better, it measures the before/after correlation with income directly. `corrwith` is used instead of building a full correlation matrix because only one relationship (each column vs. `Income`) is needed here, which is both faster and produces exactly the comparison table required without extra columns to filter out.

### Cell 4 (code) — Quartile breakdown for one representative category

```python
df["Income_Quartile"] = pd.qcut(df["Income"], 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])

quartile_means = df.groupby("Income_Quartile", observed=True)[["Groceries", "Groceries_Ratio"]].mean()
quartile_means.columns = ["Mean raw Groceries (₹)", "Mean Groceries_Ratio"]
quartile_means
```

**What it does:** Buckets individuals into four equal-sized income quartiles using `pd.qcut` (quantile-based cuts, so each bucket has ~5,000 people regardless of income's skew), then compares the mean *raw* `Groceries` spend against the mean `Groceries_Ratio` within each bucket. `observed=True` in the `groupby` avoids pandas materializing empty categorical combinations that don't apply here.

**Why:** The correlation numbers in Cell 3 are already conclusive, but a single aggregate statistic can be hard to build intuition from. Breaking it down by income quartile turns "correlation ≈ 0" into something directly readable: raw grocery spend should visibly scale up across quartiles while the ratio should look flat — a concrete, inspectable version of the abstract claim. `pd.qcut` (quantile cuts) is used rather than `pd.cut` (equal-width bins) specifically *because* `Income` is right-skewed (per Phase 1) — equal-width bins on skewed data would put almost everyone in the first bin and leave the rest nearly empty.

### Cell 5 (code) — Visualizing the quartile comparison

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.barplot(x="Income_Quartile", y="Groceries", data=df, ax=axes[0], errorbar=None)
axes[0].set_title("Raw Groceries spend by income quartile\n(scales up with income)")
sns.barplot(x="Income_Quartile", y="Groceries_Ratio", data=df, ax=axes[1], errorbar=None)
axes[1].set_title("Groceries_Ratio by income quartile\n(flat across income levels)")
fig.tight_layout()
plt.show()
```

**What it does:** Plots the same quartile comparison as two side-by-side bar charts, one for the raw column and one for the ratio column, with titles that state the expected pattern directly. `errorbar=None` turns off seaborn's default confidence-interval whiskers, since the sample size per quartile (~5,000) makes them visually negligible anyway and they'd just add clutter.

**Why:** Putting the "scales up" and "flat" charts side by side, at the same width, makes the contrast immediate — a reader can see in one glance that the left chart has a rising staircase shape and the right one doesn't, which is a much faster way to absorb "ratios generalize across income levels" than reading the quartile table in Cell 4.

### Cell 6 (markdown) — Answer to Q1

States the answer: raw expense↔income correlation of 0.79–0.99 collapses to <0.01 for every category once converted to ratios; the ~8× rise in raw grocery spend vs. the flat ~0.125 ratio across quartiles makes the mechanism concrete. Concludes that ratios should replace raw expense columns in the feature set, with `Income` kept separately.

### Cell 7 (markdown) — Question 2 setup

Introduces the encoding question for `Occupation` and `City_Tier`.

### Cell 8 (code) — Category cardinality and income association

```python
print("Occupation categories:", df["Occupation"].value_counts().to_dict())
print("City_Tier categories:", df["City_Tier"].value_counts().to_dict())

income_by_occupation = df.groupby("Occupation", observed=True)["Income"].mean().sort_values(ascending=False)
income_by_tier = df.groupby("City_Tier", observed=True)["Income"].mean().reindex(["Tier_1", "Tier_2", "Tier_3"])
print("\nMean Income by Occupation:\n", income_by_occupation)
print("\nMean Income by City_Tier:\n", income_by_tier)
```

**What it does:** Prints the value counts for both categorical columns (confirming low cardinality: 4 and 3 levels respectively, and roughly balanced group sizes), then compares mean `Income` across the groups of each. `income_by_tier` is explicitly `.reindex(["Tier_1", "Tier_2", "Tier_3"])`'d so the output prints in the categories' natural order rather than whatever order `groupby` happens to produce.

**Why:** Cardinality is the first thing that determines whether one-hot encoding is even viable — with only 3–4 levels each, the one-hot cost (extra columns) is trivial, so cardinality alone doesn't rule anything out here. The income comparison is the real test: if `Occupation` or `City_Tier` correlated strongly with `Income`, that would be worth knowing (it could imply the categorical column is partly a proxy for something already captured elsewhere). It turns out neither does — a result that's reported plainly here and feeds directly into the encoding decision in Cell 10.

### Cell 9 (code) — Testing for a monotonic relationship in `City_Tier`

```python
rent_ratio_by_tier = df.groupby("City_Tier", observed=True)["Rent_Ratio"].mean().reindex(["Tier_1", "Tier_2", "Tier_3"])
print("Mean Rent_Ratio by City_Tier:\n", rent_ratio_by_tier)

fig, ax = plt.subplots(figsize=(5, 4))
rent_ratio_by_tier.plot(kind="bar", ax=ax, color=["#4c72b0", "#55a868", "#c44e52"])
ax.set_ylabel("Mean Rent_Ratio")
ax.set_title("Rent burden rises monotonically\nfrom Tier_3 to Tier_1")
fig.tight_layout()
plt.show()
```

**What it does:** Even though `Income` doesn't differ by `City_Tier` (Cell 8), this checks whether `Rent_Ratio` — the ratio feature just engineered in Cell 3 — does, and plots the result as a bar chart.

**Why:** This is the cell that actually justifies treating `City_Tier` as more than a plain nominal category. `Income` alone showing no relationship (Cell 8) could have made it tempting to conclude `City_Tier` carries no meaningful signal at all — checking `Rent_Ratio` specifically (cost of living should show up in rent burden, not necessarily in income) is what surfaces the real, strongly monotonic pattern (0.30 → 0.20 → 0.15 from Tier_1 to Tier_3) that the answer in Cell 10 hinges on. This is also a nice validation that the ratio features from Question 1 aren't just theoretically motivated — they're already surfacing genuine structure the raw columns didn't make visible as clearly.

### Cell 10 (markdown) — Answer to Q2

States the answer: `Occupation` is one-hot encoded as a purely nominal category (no income association). `City_Tier` is also one-hot encoded by default — despite having a real, monotonic cost-of-living order evidenced by the `Rent_Ratio` pattern — because with only 3 levels the one-hot cost is trivial and it avoids forcing linear/distance-based models to assume equal spacing between tiers. Notes ordinal encoding (`Tier_1=1, Tier_2=2, Tier_3=3`) as a defensible cheaper alternative specifically for tree-based models, which aren't hurt by an imposed order.

### Cell 11 (markdown) — Question 3 setup

Introduces the scaling question.

### Cell 12 (code) — Feature scale summary

```python
feature_cols = ["Income", "Age", "Dependents"] + ratio_cols
scale_summary = df[feature_cols].agg(["mean", "std", "min", "max"]).T
scale_summary["range"] = scale_summary["max"] - scale_summary["min"]
scale_summary.sort_values("range", ascending=False)
```

**What it does:** Builds a summary table of mean, standard deviation, min, max, and range for every candidate numeric feature — `Income`, `Age`, `Dependents`, and the 11 ratio columns — sorted by range, largest first.

**Why:** Sorting by `range` makes the scale mismatch impossible to miss: `Income`'s range (from ~₹1.3k to ~₹1.08M) dwarfs `Age`'s (18–64), which in turn dwarfs the ratio columns' (each bounded within roughly 0–0.3). This table is the direct evidence for the scaling answer — rather than asserting "features are on different scales," it quantifies exactly how different, which is what determines whether that difference will actually matter to a given model.

### Cell 13 (markdown) — Answer to Q3

States the answer: `Income` (and to a lesser extent `Age`) need scaling before use with magnitude-sensitive models. Splits the model families into two groups — Logistic Regression / SVM / KNN / neural nets need scaling because they're distance- or gradient-magnitude-sensitive, while Decision Tree / Random Forest / XGBoost are scale-invariant because they split on per-feature thresholds independently of other features' units. Recommends maintaining both a scaled and an unscaled feature matrix for Phase 4 rather than assuming one preprocessing pipeline fits every model.

### Cell 14 (markdown) — Question 4 setup

Introduces the redundancy/collinearity question.

### Cell 15 (code) — Correlation heatmap of the engineered feature set

```python
corr_matrix = df[["Income", "Age", "Dependents"] + ratio_cols].corr()
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 7})
ax.set_title("Correlation matrix — engineered feature set")
fig.tight_layout()
plt.show()
```

**What it does:** Computes and visualizes the full pairwise correlation matrix across the engineered numeric feature set (`Income`, `Age`, `Dependents`, and the 11 ratio columns) — the same style of heatmap as Phase 1 Cell 16, but now over the *post-engineering* features rather than the raw ones.

**Why:** This is the direct follow-up to Phase 1's finding that raw expenses were highly collinear with each other — the natural question is whether converting to ratios (Question 1) also fixed that problem, or only fixed the income-correlation problem. Running the same diagnostic again, on the new features, is what actually answers that rather than assuming it.

### Cell 16 (code) — Variance Inflation Factor (VIF)

```python
X = df[["Income", "Age", "Dependents"] + ratio_cols].dropna()
X_with_const = sm.add_constant(X)  # VIF requires an intercept term, or scale-driven artifacts inflate every score
vif = pd.Series(
    [variance_inflation_factor(X_with_const.values, i) for i in range(1, X_with_const.shape[1])],
    index=X.columns,
    name="VIF",
).sort_values(ascending=False)
vif.to_frame()
```

**What it does:** Computes the Variance Inflation Factor for every feature — for each one, how well it can be linearly predicted from *all the other* features combined (a VIF of 1.0 means "not predictable from the others at all"; conventionally, 5 or 10 is treated as a concern threshold).

**Why VIF, and not just the correlation heatmap:** Pairwise correlation (Cell 15) only catches redundancy between *two* features at a time. A feature can look uncorrelated with every other feature individually while still being almost perfectly predictable from a *combination* of several others — VIF catches that multivariate case, which is exactly the kind of collinearity a pairwise heatmap is structurally blind to.

**Why `sm.add_constant` specifically:** `variance_inflation_factor` from `statsmodels` computes each feature's VIF by regressing it on the rest of the design matrix — and that regression needs an intercept term to be meaningful. Without `sm.add_constant`, the underlying regression is forced through the origin, which inflates every VIF score by an amount related to each feature's distance from zero — so features with large, non-zero means (like `Income`, in the tens of thousands) would show artificially huge VIFs that reflect their *scale*, not genuine collinearity. Adding the constant column removes that artifact and makes the VIF numbers a true measure of multicollinearity. This is a well-known `statsmodels` sharp edge worth calling out explicitly, since skipping it silently produces misleading results rather than an error.

### Cell 17 (markdown) — Answer to Q4

States the answer: the correlation heatmap shows only one pair worth flagging (`Dependents`/`Education_Ratio` at r ≈ 0.65), and the properly-computed VIF confirms it quantitatively — every feature scores ≈1.0 except `Dependents` and `Education_Ratio` at ≈1.73, both comfortably under the standard concern threshold. Concludes no features need to be dropped, though the `Dependents`/`Education_Ratio` link is worth watching in Phase 5's explainability step since their individual contributions may partially trade off against each other.

### Cell 18 (markdown) — Transition to building the final feature set

Summarizes how the four answers combine into one concrete decision: replace raw expenses with ratios, one-hot encode both categoricals, keep `Income`/`Age`/`Dependents` as numeric (to be scaled per model family downstream), and drop the leakage columns plus the now-redundant raw expense columns.

### Cell 19 (code) — Assembling the engineered feature matrix

```python
engineered = pd.get_dummies(
    df[["Income", "Age", "Dependents", "Occupation", "City_Tier"] + ratio_cols + ["Goal_Met"]],
    columns=["Occupation", "City_Tier"],
    drop_first=False,
)

print(f"Final engineered feature matrix shape: {engineered.shape}")
print(f"Dropped as leakage: {LEAKAGE_COLS}")
print(f"Dropped as redundant (replaced by ratios): {expense_cols}")
engineered.head()
```

**What it does:** Selects exactly the columns the four answers settled on — `Income`, `Age`, `Dependents`, the two categoricals, the 11 ratio columns, and the target — and one-hot encodes the categoricals with `pd.get_dummies`. Prints the resulting shape and explicitly restates which columns were left out and why.

**Why `drop_first=False`:** Keeping all dummy columns (rather than dropping one reference category, which is standard practice for OLS-style linear regression to avoid perfect multicollinearity with the intercept) is deliberate here — this feature set is meant to feed several model families in Phase 4, including tree-based ones that benefit from having every category as an explicit, independently-splittable column, and Question 4 already confirmed collinearity isn't a real concern for this feature set. Explicitly printing the leakage and redundant-column exclusions in the output — rather than just silently not including them — keeps the notebook self-auditing: anyone reading the output can verify what was excluded without re-reading the code.

### Cell 20 (markdown) — Summary table and handoff to Phase 3

Closes with a compact table of all four answers and a pointer to `03_baseline.ipynb`, so Phase 3 can start directly from this engineered feature matrix instead of re-deriving it.

---

## What Phase 2 sets up for later phases

| Decision made here | Where it gets used |
|---|---|
| Raw expenses replaced with income ratios | The `engineered` feature matrix built in Cell 19, used as-is starting in Phase 3 |
| `Occupation` and `City_Tier` one-hot encoded | Same `engineered` matrix; ordinal `City_Tier` noted as a tree-model-only alternative if Phase 4 wants to compare |
| `Income`/`Age` flagged as needing scaling for some models | Phase 3/4 build both a scaled and unscaled version of `engineered` depending on model family |
| No features dropped for collinearity | Phase 4's full feature set carries forward unchanged; the mild `Dependents`↔`Education_Ratio` link is flagged for Phase 5's explainability review |

**Next:** [Phase 3 — Baseline](phase3.md) (`03_baseline.ipynb`), which trains a majority-class baseline and a plain logistic regression on this feature set, informed directly by Phase 1's ~99.4%/0.6% class imbalance finding.
