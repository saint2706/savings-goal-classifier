# Phase 1 — Data Understanding

**Source:** [README § Phase 1 — Data Understanding](../README.md#phase-1--data-understanding)
**Notebook:** [`notebooks/01_eda_and_leakage_check.ipynb`](../notebooks/01_eda_and_leakage_check.ipynb)

Phase 0 fixed *what* we're predicting and *why*. Phase 1 turns to the data itself: what's actually in it, whether it's trustworthy, and — critically — whether the leakage risk flagged in the README (`Disposable_Income` / `Desired_Savings` mathematically defining `Goal_Met`) is real. Everything here is read-only exploration; no features are engineered yet (that's Phase 2) and no leakage columns are ever fed to a model.

---

## Research questions & answers

| # | Question | Answer |
|---|---|---|
| 1 | What does each column mean, and what unit/time period does it represent? | All money columns are monthly ₹ amounts; `Age`/`Dependents` are counts; `Occupation`/`City_Tier` are categorical; the dataset is a single cross-sectional snapshot, not a time series. |
| 2 | What is the distribution of income, expenses, and savings — skew, outliers, implausible values? | Every money column is strongly right-skewed (skew ≈ 4–5); an IQR check flags a smooth high-value tail rather than obvious errors; ~0.5–0.6% of rows have total expenses exceeding income (implausible / negative disposable income). |
| 3 | Are there missing values or duplicate rows, and how are they handled? | Zero missing values, zero duplicate rows across all 20,000 records — no imputation or dedup logic is needed. |
| 4 | How correlated are expense categories with income and with each other? | Every expense category correlates strongly with `Income` (r ≈ 0.79–0.99), and expense categories are highly correlated with each other (several pairs > 0.94) — raw amounts mostly re-encode income level. |
| 5 | Is the target mathematically derivable from any candidate feature (leakage check)? | Yes — `Goal_Met` is 100% reconstructable from `Disposable_Income` + `Desired_Savings` (that's its definition), and `Disposable_Income` itself is exactly `Income` minus the 11 expense columns. `Disposable_Income`, `Desired_Savings`, and `Desired_Savings_Percentage` are excluded from the feature set. |
| 6 | What is the class balance of `Goal_Met` once leakage columns are excluded? | ~99.4% `Goal_Met = 1` vs. ~0.6% `Goal_Met = 0` — a severe imbalance that constrains every later phase's metric and validation choices. |

The rest of this document walks through *how* the notebook arrives at each answer, cell by cell, and why each analysis step was chosen.

---

## Notebook walkthrough

The notebook itself carries no narrative — section headers and code only, real results in the executed output. This document is where the reasoning behind each step lives.

### Cell 0 (markdown) — Title

Points back to this document; no restated questions or narration in the notebook itself.

### Cell 1 (code) — Imports and data load

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", 40)
sns.set_theme(style="whitegrid")

DATA_PATH = "../dataset/data.csv"
df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
df.head()
```

**What it does:** Loads the standard EDA stack (`pandas`, `numpy`, `matplotlib`, `seaborn`), widens pandas' column display so all 27 columns are visible instead of being truncated with `...`, and applies a clean `whitegrid` theme so every chart in the notebook looks consistent. Then reads the CSV from `dataset/data.csv` (relative to the `notebooks/` folder) and prints its shape.

**Why:** Setting `display.max_columns` before doing anything else avoids a recurring annoyance in a table this wide (27 columns) — without it, `df.head()` and `.describe()` truncate output and hide exactly the columns you need to compare. Printing `df.shape` first is a cheap sanity check that the file loaded as expected (20,000 rows × 27 columns) before any analysis is trusted.

### Cell 2 (markdown) — "Column types and cardinality"

### Cell 3 (code) — Dtype and cardinality check

```python
dtypes_table = df.dtypes.rename("dtype").to_frame()
dtypes_table["n_unique"] = df.nunique()
dtypes_table
```

**What it does:** Builds a small reference table of each column's pandas dtype and its number of distinct values.

**Why:** This is the fastest way to confirm the data dictionary in the README actually matches what's in the file, rather than trusting documentation blindly. `n_unique` specifically distinguishes the categorical columns (`Occupation`: 4, `City_Tier`: 3) from the continuous ones (thousands of unique values) at a glance, and it's a cheap way to catch surprises — e.g. an `Age` column that turned out to have only 3 unique values would be a red flag worth investigating before going further.

**Answer to Q1:** every money column is a monthly ₹ amount, `Age`/`Dependents` are counts, `Occupation`/`City_Tier` are categorical, `Desired_Savings_Percentage` is the one percentage-of-income column, and there's no date/time column because this is a single snapshot.

### Cell 4 (markdown) — "Distributions"

### Cell 5 (code) — Descriptive statistics and skew

```python
money_cols = [
    "Income", "Rent", "Loan_Repayment", "Insurance", "Groceries", "Transport",
    "Eating_Out", "Entertainment", "Utilities", "Healthcare", "Education",
    "Miscellaneous", "Desired_Savings", "Disposable_Income",
]
desc = df[money_cols].describe().T
desc["skew"] = df[money_cols].skew()
desc
```

**What it does:** Runs `.describe()` (count, mean, std, min, quartiles, max) on every money column and appends a skewness statistic.

**Why:** `.describe()` alone tells you *where* the data sits, but not its *shape* — a mean far above the median is the classic tell of a right-skewed distribution, and the explicit `skew` column quantifies exactly how skewed. This matters concretely for later phases: strongly skewed features (skew ≈ 4–5 here) are candidates for a log transform before feeding them into scale-sensitive models like logistic regression, and skew this consistent across every money column suggests it's a structural property of income/spending data, not a fluke in one column.

### Cell 6 (code) — Distribution histograms

```python
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.ravel(), ["Income", "Rent", "Groceries", "Eating_Out", "Desired_Savings", "Disposable_Income"]):
    sns.histplot(df[col], bins=60, ax=ax, kde=True)
    ax.set_title(f"{col}  (skew={df[col].skew():.2f})")
fig.tight_layout()
plt.show()
```

**What it does:** Plots histograms (with a KDE overlay) for six representative money columns in a 2×3 grid, each titled with its skew value.

**Why:** Numbers alone (Cell 5) can hide *shape* — two distributions can share a skew statistic while looking completely different (one long smooth tail vs. a handful of extreme spikes). A visual check confirms whether the skew is a smooth, well-behaved long tail (which it is here) rather than something pathological like a bimodal distribution or a cluster of duplicate extreme values, which would need different handling.

### Cell 7 (code) — IQR-based outlier counts

```python
def iqr_outlier_count(s):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((s < lo) | (s > hi)).sum())

outlier_counts = {c: iqr_outlier_count(df[c]) for c in money_cols}
pd.Series(outlier_counts, name="n_outliers_iqr_1.5x").sort_values(ascending=False)
```

**What it does:** Applies the standard Tukey rule (flag anything beyond 1.5× the interquartile range past Q1/Q3) to every money column and counts the flagged rows.

**Why:** The IQR rule is a scale-independent, distribution-agnostic way to quantify "how many rows are unusually extreme" — useful specifically *because* it doesn't assume normality, which these skewed columns clearly violate. It's a triage step, not a decision to delete anything: a high outlier count on a right-skewed column like `Income` is expected (that's what a long tail *is*), so the number is reported for downstream phases to weigh, not acted on here.

### Cell 8 (code) — Implausible-value check (expenses exceeding income)

```python
expense_cols = [
    "Rent", "Loan_Repayment", "Insurance", "Groceries", "Transport", "Eating_Out",
    "Entertainment", "Utilities", "Healthcare", "Education", "Miscellaneous",
]
df["Total_Expenses"] = df[expense_cols].sum(axis=1)

implausible_mask = df["Total_Expenses"] > df["Income"]
n_implausible = int(implausible_mask.sum())
print(f"Rows where total expenses exceed income: {n_implausible} ({n_implausible / len(df):.2%})")
df.loc[implausible_mask, ["Income", "Total_Expenses"] + expense_cols].head()
```

**What it does:** Sums the 11 expense columns into `Total_Expenses`, then flags and counts rows where that total exceeds `Income` — i.e. rows where disposable income would be negative.

**Why:** This is a domain-specific plausibility check that a generic outlier statistic can't catch — a row can look perfectly unremarkable by IQR standards (moderate income, moderate individual expenses) and still be internally inconsistent if the expenses simply don't add up to less than the income. This check directly informs Phase 2: whether to clip, drop, or keep the ~112 rows (0.56%) where it happens.

**Answer to Q2:** strong right-skew across all money columns (skew ≈ 4–5), a smooth (not erratic) IQR tail, and ~0.5–0.6% of rows with expenses exceeding income — flagged for a Phase 2 decision rather than resolved here.

### Cell 9 (markdown) — "Missing values and duplicate rows"

### Cell 10 (code) — Missing values and duplicate rows

```python
missing = df.isnull().sum()
missing = missing[missing > 0]
print("Columns with missing values:", missing.to_dict() if len(missing) else "none")
print("Total missing cells:", int(df.isnull().sum().sum()))
print("Duplicate rows (all columns):", int(df.duplicated().sum()))
```

**What it does:** Counts nulls per column (filtering to only the columns that have any) and counts fully-duplicate rows across the entire table.

**Why:** These are the two cheapest, highest-value checks in any EDA — if either comes back non-zero, it changes everything downstream (imputation strategy, dedup logic). Checking explicitly and printing the result — rather than silently assuming a clean dataset — is what lets Phase 2's preprocessing skip imputation code entirely with a documented reason, instead of adding defensive handling for a problem that doesn't exist.

**Answer to Q3:** zero missing values, zero duplicate rows across all 20,000 records.

### Cell 11 (markdown) — "Correlation: income vs. expenses"

### Cell 12 (code) — Income–expense correlation

```python
corr_full = df[["Income"] + expense_cols].corr()
corr_full["Income"].drop("Income").sort_values(ascending=False)
```

**What it does:** Computes the Pearson correlation matrix across `Income` and the 11 expense columns, then isolates and sorts just the `Income` row.

**Why:** This is the first direct evidence for the leakage-adjacent concern that motivates Phase 2's entire ratio-conversion strategy: if expense columns are near-perfectly correlated with income, they're not really independent signal about *spending behavior* — they're mostly restating *how much this person earns*, scaled down.

### Cell 13 (code) — Correlation heatmap

```python
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr_full, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation: Income vs. expense categories")
fig.tight_layout()
plt.show()
```

**What it does:** Visualizes the same correlation matrix as a heatmap, annotated with the actual coefficient values, using a diverging colormap centered at 0.

**Why:** A 12×12 correlation matrix is hard to scan as raw numbers — the heatmap makes the *pattern* (almost everything is a warm, strongly-positive color) immediately obvious, which is the point: this isn't one or two coincidentally-correlated columns, it's a structural property of the whole expense block.

### Cell 14 (code) — Most correlated expense-expense pairs

```python
import itertools
expense_corr = df[expense_cols].corr()
pairs = sorted(
    ((a, b, expense_corr.loc[a, b]) for a, b in itertools.combinations(expense_cols, 2)),
    key=lambda t: -abs(t[2]),
)
pd.DataFrame(pairs[:8], columns=["feature_a", "feature_b", "correlation"])
```

**What it does:** Enumerates every unique pair of expense columns (`itertools.combinations` avoids double-counting `(A, B)` and `(B, A)`), looks up each pair's correlation, sorts by absolute value, and shows the top 8.

**Why:** The heatmap in Cell 13 shows the overall pattern, but reading off the *specific* highest pairs from a heatmap by eye is error-prone at this size. Extracting them programmatically gives a precise, rankable answer to "which expense categories are most redundant with each other" — directly useful for Phase 2's collinearity analysis.

**Answer to Q4:** expense categories correlate strongly with income (r ≈ 0.79–0.99) and with each other (several pairs > 0.94), which is exactly why Phase 2 converts raw expenses to income ratios.

### Cell 15 (markdown) — "Leakage check"

Arguably the most consequential section in the whole notebook, since getting it wrong would silently produce a model with meaningless, artificially perfect accuracy.

### Cell 16 (code) — Verifying `Disposable_Income` is internally consistent

```python
df["Disposable_Income_recalculated"] = df["Income"] - df["Total_Expenses"]
max_abs_diff = (df["Disposable_Income_recalculated"] - df["Disposable_Income"]).abs().max()
print(f"Max |Income - sum(expenses) - Disposable_Income| across all rows: {max_abs_diff:.2e}")
```

**What it does:** Independently recomputes `Disposable_Income` as `Income - Total_Expenses` and compares it to the `Disposable_Income` column already provided in the dataset, reporting the largest absolute discrepancy across all 20,000 rows.

**Why:** This doesn't just describe the leakage relationship — it *proves* it, row by row, rather than trusting the README's stated formula on faith. A max difference on the order of `1e-10` (floating-point noise, not a real discrepancy) confirms the provided `Disposable_Income` column really is deterministically `Income` minus expenses, which matters because it means excluding `Disposable_Income` from the feature set isn't enough on its own — a model could just as easily reconstruct it (and therefore the target) from `Income` and the expense columns *together*, which is worth keeping in mind when engineering features in Phase 2.

### Cell 17 (code) — Confirming the target formula and naming the leakage columns

```python
df["Goal_Met"] = (df["Disposable_Income"] >= df["Desired_Savings"]).astype(int)

# A trivial rule using only the two leakage columns reconstructs Goal_Met exactly.
trivial_rule = (df["Disposable_Income"] >= df["Desired_Savings"]).astype(int)
print("Trivial-rule accuracy vs Goal_Met (using only leakage columns):",
      (trivial_rule == df["Goal_Met"]).mean())

leakage_cols = ["Disposable_Income", "Desired_Savings", "Desired_Savings_Percentage"]
print("\nColumns excluded from the feature set due to leakage:", leakage_cols)
```

**What it does:** Constructs `Goal_Met` per its Phase 0 definition, then demonstrates that a "model" using nothing but a direct comparison of the two defining columns reproduces the label with 100% accuracy — by construction, since it's the literal same formula. Finally states the three columns that must be excluded from any legitimate feature set.

**Why:** This is the concrete, quantified version of the abstract warning in the README — rather than just asserting "these columns would leak," the notebook shows *exactly* what would happen (perfect, meaningless accuracy) if they were included, which is a much stronger and more convincing check to leave in the project's history. `Desired_Savings_Percentage` is included in the exclusion list even though it isn't part of the `>=` comparison directly, because `Desired_Savings` is a deterministic function of it (`Income × percentage`) — excluding the percentage but not the derived amount would just move the leak one column over.

**Answer to Q5:** `Goal_Met` is deterministically derivable from `Disposable_Income` + `Desired_Savings`, `Disposable_Income` is itself deterministic from `Income` minus expenses, and all three leakage-adjacent columns (`Disposable_Income`, `Desired_Savings`, `Desired_Savings_Percentage`) are excluded from the feature set — matching README § 3.

### Cell 18 (markdown) — "Class balance of `Goal_Met`"

Scoped explicitly to *after* leakage columns are excluded — the label itself is still legitimately computed from them; only feeding them to the model as *inputs* is forbidden.

### Cell 19 (code) — Class balance table

```python
balance = df["Goal_Met"].value_counts().rename({0: "Goal not met (0)", 1: "Goal met (1)"})
balance_pct = df["Goal_Met"].value_counts(normalize=True).rename({0: "Goal not met (0)", 1: "Goal met (1)"})
pd.DataFrame({"count": balance, "proportion": balance_pct.round(4)})
```

**What it does:** Reports both the raw counts and normalized proportions of the two `Goal_Met` classes side by side.

**Why:** Reporting proportion alongside raw count avoids ambiguity when the imbalance is severe — "19,888 vs. 112" and "99.4% vs. 0.6%" are the same fact, but the percentage form is what directly tells Phase 3/4 that accuracy alone will be a useless metric (a model that always predicts `1` already scores ~99.4%).

### Cell 20 (code) — Class balance chart

```python
fig, ax = plt.subplots(figsize=(5, 4))
sns.countplot(x="Goal_Met", data=df, ax=ax)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Not met (0)", "Met (1)"])
ax.set_title("Class balance: Goal_Met")
for p in ax.patches:
    ax.annotate(f"{p.get_height():,}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom")
fig.tight_layout()
plt.show()
```

**What it does:** Bar-charts the class counts with the exact count annotated above each bar. `ax.set_xticks([0, 1])` is set explicitly before `set_xticklabels` so matplotlib doesn't warn about relabeling ticks it hasn't been told the positions of.

**Why:** A picture of a 99.4/0.6 split communicates the severity of the imbalance faster than a table for a Phase 8 stakeholder audience — one bar is visibly a hairline next to the other. This is exactly the kind of chart the README's Phase 8 question ("which visualizations communicate findings fastest to a non-technical reader?") is asking for, produced here as a byproduct of the Phase 1 analysis.

**Answer to Q6:** ~99.4%/0.6% split. Consequence for later phases: accuracy isn't a usable metric on its own, stratified validation is required, and resampling or imbalance-aware metrics are likely necessary before model comparison is meaningful.

---

## What Phase 1 sets up for later phases

| Finding | Where it gets used |
|---|---|
| Expense columns are ~0.8–0.99 correlated with `Income` | Motivates Phase 2's conversion to expense-to-income ratios |
| `Disposable_Income`, `Desired_Savings`, `Desired_Savings_Percentage` leak the target | Excluded from the feature set from Phase 2 onward |
| No missing values, no duplicates | Phase 2 skips imputation/dedup logic entirely |
| Money columns are strongly right-skewed | Candidates for scaling/log-transform in Phase 2's model-specific preprocessing |
| `Goal_Met` is ~99.4%/0.6% imbalanced | Drives Phase 3's baseline metric choice and Phase 4's stratified validation strategy |

**Next:** [Phase 2 — Feature Engineering](phase2.md), which builds the ratio features and encoding decisions this phase's findings point toward.
