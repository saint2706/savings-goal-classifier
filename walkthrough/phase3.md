# Phase 3 — Baseline

**Source:** [README § Phase 3 — Baseline](../README.md#phase-3--baseline)
**Notebook:** [`notebooks/03_baseline.ipynb`](../notebooks/03_baseline.ipynb)
**Builds on:** [Phase 2 — Feature Engineering](phase2.md)

Phase 2 produced a clean, leakage-free feature matrix — income ratios in place of raw expenses, `Occupation`/`City_Tier` one-hot encoded. Phase 3 asks the question that has to be answered before any "real" model comparison is meaningful: what does the _worst_ reasonable model score, and what does the _simplest_ real model score, without any of the imbalance-handling machinery Phase 4 will bring in? Both numbers exist to be beaten — Phase 4 only means something once there's a floor to compare against.

---

## Research questions & answers

| #   | Question                                                                              | Answer                                                                                                                                                                                                                                   |
| --- | ------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | What accuracy/F1 does a majority-class or simple single-rule baseline achieve?        | ~99.4% accuracy, but 0.0 precision/recall/F1 on the minority class (`Goal_Met = 0`) and ~0.499 macro-F1. Accuracy alone is not a usable metric for this problem.                                                                         |
| 2   | What does plain logistic regression achieve using only income and 1–2 expense ratios? | ~99.4% accuracy, ~0.499 macro-F1 — statistically indistinguishable from the baseline. The model learns the right coefficient signs but its default 0.5 threshold almost never predicts the minority class under this level of imbalance. |

The rest of this document walks through _how_ the notebook arrives at each answer, cell by cell, and why each analysis step was chosen.

---

## Notebook walkthrough

The notebook carries only section headers and code; the reasoning behind each step lives here. Its purpose is to establish a floor, not find a winning model — "plain" is deliberately minimal on two axes, a small feature set and no imbalance handling, so those omissions are intentional rather than oversights to fix later in this same notebook.

### Cell 0 (markdown) — Title

### Cell 1 (code) — Imports, data load, and rebuilding the Phase 2 feature matrix

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

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
ratio_cols = []
for c in expense_cols:
    ratio_col = f"{c}_Ratio"
    df[ratio_col] = df[c] / df["Income"]
    ratio_cols.append(ratio_col)

engineered = pd.get_dummies(
    df[["Income", "Age", "Dependents", "Occupation", "City_Tier"] + ratio_cols + ["Goal_Met"]],
    columns=["Occupation", "City_Tier"],
    drop_first=False,
)
```

**What it does:** Adds the modeling stack (`scikit-learn`'s `train_test_split`, `DummyClassifier`, `LogisticRegression`, `StandardScaler`, and the metrics functions) to the same EDA imports used in Phases 1–2. Then reloads the raw CSV and reconstructs exactly the same `Goal_Met` target and `engineered` feature matrix that Phase 2 built — ratio columns replacing raw expenses, both categoricals one-hot encoded, leakage columns excluded.

**Why rebuild from scratch:** Reloading from `dataset/data.csv` rather than depending on Phase 2's live notebook state keeps this notebook runnable on its own, matching the pattern already set by `01_eda_and_leakage_check.ipynb` and `02_feature_engineering.ipynb` — each phase's notebook is self-contained and reproducible independently of the others.

### Cell 2 (markdown) — "Train/test split"

### Cell 3 (code) — Train/test split

```python
X_full = engineered.drop(columns=["Goal_Met"])
y = engineered["Goal_Met"]

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, stratify=y, random_state=42
)
```

**What it does:** Splits the engineered matrix 80/20 into train and test sets, with `stratify=y` so both splits preserve the ~99.4%/0.6% class balance, and a fixed `random_state` for reproducibility. Both splits' class balances are printed and confirmed to match the population proportion (~99.4%/0.6% in both).

**Why stratify:** With only 112 minority-class rows in the full 20,000-row dataset, a plain random split risks an unlucky test set with very few (or unusually many) `Goal_Met = 0` cases by chance alone — that would make a single evaluation noisy and unreliable for comparing models. Stratifying removes that source of noise, so any difference between the baseline and logistic regression later in the notebook reflects the models themselves, not which rows happened to land in the test set. This split (and its stratification strategy) is reused for every model in this notebook, and is the same principle Phase 4 will extend into stratified k-fold cross-validation.

### Cell 4 (markdown) — "Majority-class baseline"

### Cell 5 (code) — Fitting the baseline

```python
def evaluate(name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {
        "model": name,
        "accuracy": accuracy,
        "precision_0": precision[0], "recall_0": recall[0], "f1_0": f1[0],
        "precision_1": precision[1], "recall_1": recall[1], "f1_1": f1[1],
        "f1_macro": f1_macro,
    }

dummy = DummyClassifier(strategy="most_frequent", random_state=42)
dummy.fit(X_train, y_train)
pred_dummy = dummy.predict(X_test)

results = [evaluate("Majority-class baseline", y_test, pred_dummy)]
```

**What it does:** Defines a reusable `evaluate()` helper — used for every model in this notebook — that reports accuracy plus precision/recall/F1 broken out _per class_, alongside macro-F1. Fits `scikit-learn`'s `DummyClassifier(strategy="most_frequent")`, which is exactly the "simple single-rule baseline" the question asks about: with a 99.4%/0.6% split, "predict the majority class" and "always predict `Goal_Met = 1`" are the same rule.

**Why report per-class metrics, not just accuracy:** Accuracy alone (~99.4%) makes this do-nothing baseline look almost perfect — exactly the trap Phase 1's class-balance finding warned about. Breaking metrics out by class shows what accuracy hides: class 0 (`Goal not met`) precision, recall, and F1 are all _zero_ — the baseline never once predicts the minority class, making it structurally incapable of catching the exact at-risk customers the Phase 0 business decision cares about. Macro-F1 (which weights both classes' F1 equally, regardless of class size) collapses this failure into one number that a weighted average or plain accuracy would hide.

### Cell 6 (code) — Confusion matrix

Visualizes the same result as a heatmap: the entire "Pred 0" column is empty, confirming the metrics table isn't a computation mistake and making the failure visible at a glance.

**Answer to Q1:** ~99.4% accuracy but 0.0 precision/recall/F1 on the minority class and ~0.499 macro-F1 — barely above the ~0.5 floor a coin flip would produce on a two-class problem. Accuracy is not a usable metric for this project; every Phase 4 model is judged on macro-F1 (or an equivalent class-balanced metric) instead.

### Cell 7 (markdown) — "Plain logistic regression"

### Cell 8 (code) — Choosing the minimal feature set

```python
candidate_cols = ["Income"] + ratio_cols
train_df = df.loc[X_train.index]  # training rows only — the test set must stay unseen for this feature-selection step too
target_corr = train_df[candidate_cols + ["Goal_Met"]].corr()["Goal_Met"].drop("Goal_Met")
target_corr.sort_values(key=lambda s: -s.abs())
```

**What it does:** Correlates `Income` and every expense ratio with `Goal_Met`, computed on the training rows only (`df.loc[X_train.index]`), sorted by absolute correlation strength.

**Why restrict this to the training split:** Choosing which features to feed the model is itself a modeling decision, not a passive lookup — computing this correlation over the full `df` (test rows included) would let the held-out set influence which features the logistic regression gets to use, a subtler form of the same test-set leakage the scaler-fitting step later in this notebook is careful to avoid. With only 22 minority-class rows in the test set, that influence wouldn't be negligible, so restricting the correlation to `X_train`'s rows keeps the test set genuinely unseen until evaluation.

**Why these two ratios:** The question asks for logistic regression on "income and 1–2 expense ratios" without specifying which ratios — rather than picking two arbitrarily, this cell lets the (training) data decide. `Loan_Repayment_Ratio` (r ≈ -0.152) and `Rent_Ratio` (r ≈ -0.109) come out clearly ahead of the rest, which matches domain intuition: rent and loan repayments are typically the largest, least-discretionary expense categories, so a higher share of income going to either one directly squeezes the disposable income `Goal_Met` depends on. `Income` itself barely correlates with `Goal_Met` (r ≈ -0.008) — a first hint that income level alone says little about whether someone hits their own self-defined savings goal, since the goal itself scales with income.

### Cell 9 (code) — Fitting and evaluating plain logistic regression

```python
baseline_features = ["Income", "Loan_Repayment_Ratio", "Rent_Ratio"]

X_train_min = X_train[baseline_features]
X_test_min = X_test[baseline_features]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_min)
X_test_scaled = scaler.transform(X_test_min)

logreg = LogisticRegression(random_state=42)
logreg.fit(X_train_scaled, y_train)
pred_logreg = logreg.predict(X_test_scaled)

results.append(evaluate("Plain LogisticRegression (Income + 2 ratios)", y_test, pred_logreg))
```

**What it does:** Fits `scikit-learn`'s default `LogisticRegression` — no `class_weight`, default L2 regularization — on just the three chosen features, after scaling with a `StandardScaler` fit on the training split only. Appends its scores to the same `results` list as the baseline so both rows sit side by side in one comparison table.

**Why fit the scaler on training data only:** `scaler.fit_transform(X_train_min)` learns the mean/std from the training split; `scaler.transform(X_test_min)` (no second `fit`) reuses those exact statistics on the test split rather than recomputing them from test data. Letting the scaler see the test set's own distribution would leak information about it into preprocessing — a subtler cousin of the label leakage Phase 1 checked for, and worth guarding against explicitly even in a deliberately "plain" baseline model.

**Why scale `Income` but not the ratios:** Consistent with Phase 2's Q3 answer — `Income` ranges from roughly ₹1.3k to ₹1.08M, while the ratio columns are already small and bounded (roughly 0–0.3), so only `Income` risks dominating a gradient-based linear model's optimization if left unscaled.

### Cell 10 (code) — Comparison chart

Plots accuracy and macro-F1 for both models side by side, next to the logistic regression's own confusion matrix (same layout as the baseline's, for direct visual comparison).

**Answer to Q2:** plain logistic regression scores ~99.4% accuracy and ~0.499 macro-F1 — statistically indistinguishable from the majority-class baseline. Its confusion matrix shows it predicts class 0 for at most one of the 22 true at-risk test individuals. The fitted coefficients for `Loan_Repayment_Ratio` and `Rent_Ratio` are both strongly negative — the model _did_ learn the right relationship — but with no `class_weight` or resampling, its default 0.5 probability threshold is dominated by the 99.4% majority class and almost never crosses into predicting the minority one.

This notebook already shows _why_ Phase 4 can't skip imbalance handling: a plain, imbalance-unaware classifier — however reasonable its learned coefficients — cannot clear the baseline floor on its own.

---

## What Phase 3 sets up for later phases

| Finding                                                                                            | Where it gets used                                                                                                          |
| -------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Majority-class baseline: ~99.4% accuracy, ~0.499 macro-F1, 0.0 recall on the minority class        | The floor every Phase 4 model must clear; accuracy is retired as a decision metric in favor of macro-F1                     |
| Plain logistic regression does not beat the baseline under this imbalance                          | Justifies Phase 4's use of `class_weight`, resampling, and/or threshold tuning rather than treating them as optional extras |
| `Loan_Repayment_Ratio` and `Rent_Ratio` are the two expense ratios most associated with `Goal_Met` | A candidate signal for Phase 5's explainability review, to check whether the winning model agrees                           |
| The stratified 80/20 train/test split (`random_state=42`)                                          | Reused as the basis for Phase 4's stratified k-fold cross-validation                                                        |

**Next:** [Phase 4 — Model Comparison](phase4.md) (`04_model_comparison.ipynb`), which compares 5–7 model families under stratified cross-validation and imbalance-aware handling, evaluated against the macro-F1 floor this phase established.
