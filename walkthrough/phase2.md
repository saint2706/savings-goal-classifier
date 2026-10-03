# Phase 2: Feature Engineering

**Source:** [README § Phase 2: Feature Engineering](../README.md#phase-2-feature-engineering)
**Notebook:** [`notebooks/02_feature_engineering.ipynb`](../notebooks/02_feature_engineering.ipynb)
**Builds on:** [Phase 1](phase1.md), [Dataset construction](dataset_construction.md)
**Artifacts:** `results/feature_engineering.json`; the feature table `dataset/features.parquet` (41,518 households × 43 columns) is written by `sgc build` through `savings_goal.features.engineer.engineer`

The feature table is built by the package, not by this notebook. The notebook rebuilds it, checks the rebuild matches the saved file exactly, and then tests each design choice behind it. Every test runs on the training split only: one PSU-grouped fold (8,299 households, 472 PSUs) is held out as the test set before any choice is made, leaving 33,219 households in 1,989 PSUs.

> **In plain terms: feature engineering.** Raw survey columns rarely go straight into a model. Feature engineering is the step that turns them into model inputs: dividing, logging, adding yes/no flags, deciding what to do with blanks. Each choice here is settled by building both versions and scoring them, not by convention.

> **In plain terms: why decide on the training split only.** If the test households influenced which features were built, the final test score would be slightly flattering, because the design was tuned to them. Holding the test fold out first keeps it unseen until Phase 4 scores the finished model on it.

---

## Research questions & answers

| # | Question | Answer |
| --- | --- | --- |
| 1 | Do expense-to-income ratios generalise better across income levels than raw expense values? | No. Trained on one income half and scored on the other (target `Has_Bank_Savings`, p99 winsorising inside the pipeline), mean ROC-AUC is 0.603 for ratios, 0.628 for raw rupees and 0.618 for shares. Without winsorising: 0.581, 0.620, 0.614. Ratios transfer worst either way. |
| 2 | How should the categorical features be encoded, and how should missing categories be handled? | One-hot encoding with an explicit `Unknown` level. Only `Caste_Group` has missing values in the training split (69 households); their goal rate is 0.333 against 0.319, so the missingness carries no visible signal. Levels: `Occupation` 6, `Area_Type` 4, `Caste_Group` 7 (including `Unknown`), `Religion` 9. |
| 3 | Should the majority-zero expense categories get explicit participation indicators? | Yes, for the five shares zero for more than 30% of training households (rent, insurance, eating out, entertainment, education). Logistic regression ROC-AUC rises from 0.917 to 0.919 and macro-F1 from 0.814 to 0.818. The indicators belong to the diagnostic full set only. |
| 4 | Which features require scaling, and does that depend on the downstream model? | Linear models get `StandardScaler`; tree models get none. `RobustScaler` is not used on shares: for zero-inflated shares the interquartile range is close to zero, and `Entertainment_Share` scales to values up to 154. The ROC-AUC difference between the two scalers is 0.0001 (0.9200 vs 0.9199). Debt is winsorised at p99 inside the pipeline; log1p ties with it (0.920 both, logistic). The centred log-ratio transform loses to raw shares (0.911 vs 0.920). |
| 5 | Are any features redundant or highly collinear? | The 11 shares are exactly collinear because they sum to one (VIF about 10¹⁵). With `Groceries_Share` dropped as a reference, as the linear pipelines do, the largest VIF is `Age_Dependents` 8.92, then `Household_Size` 5.20 and `Dependency_Ratio` 4.60; no share exceeds 1.86. |

Final sets: the full (diagnostic) set has 29 features, 25 numeric plus 4 categorical. The headline deployable set has 13: `Log_Income`, `Household_Size`, `Age_Dependents`, `Dependency_Ratio`, `Head_Age`, `Max_Adult_Education`, `Debt_To_Income`, `Has_Debt`, `Debt_Missing`, and the four categoricals.

> **In plain terms: one-hot encoding.** A model does arithmetic, so a text column like `Occupation = "Farm"` must become numbers. Numbering the categories 1 to 6 would imply an order and spacing that do not exist. One-hot encoding replaces the column with one yes/no column per category, exactly one of which is 1.

---

## Notebook walkthrough

### Cell 1: load, rebuild check, training split

Loads the household table (`savings_goal.io.load_households`) and the saved feature table (`savings_goal.io.load_features`), then calls `savings_goal.features.engineer.engineer` on the households and asserts the output equals the saved table column for column. If anyone edits the build code without rebuilding, the notebook fails here.

`engineer` does the following, in order:

1. Draws the test fold with `savings_goal.evaluation.cv.grouped_test_mask` (the first fold of a stratified, PSU-grouped 5-fold split, seed 42) and stores it as `Is_Test`.
2. Fills missing categoricals with `Unknown`.
3. Measures each share's zero rate on the training rows and adds a `Spends_On_*` indicator for every share zero more than 30% of the time.
4. Adds `Has_Debt` and `Dependency_Ratio` (`Age_Dependents / Household_Size`).
5. Picks the log-ratio core (shares zero less than 20% of the time), replaces their zeros and writes six `*_CLR` columns. These are not model features; they record the log-ratio view of the core for inspection.

The cell selects the training rows (`Is_Test == False`), prints 33,219 households in 1,989 PSUs with 8,299 held out, and defines `cv_score`, which runs `savings_goal.evaluation.cv.cross_validate_grouped` with `grouped_cv()` (`StratifiedGroupKFold`, 5 folds, grouped by `IDPSU`) and the metrics from `savings_goal.evaluation.metrics.fold_metrics`.

> **In plain terms: stratified, grouped folds.** Grouped means no PSU is split between training and scoring. Stratified means each fold gets roughly the same share of on-track households (about 32%), so no fold is scored on an unusual mix.

### Cell 3: ratios vs raw rupees vs shares (Q1)

Ratios are often recommended because they make households of different incomes comparable. The test cannot use `Goal_Met`: Phase 1 showed raw rupees and ratios reconstruct it for 99.75% of households, so they would win by construction. The neutral target is `Has_Bank_Savings`, a survey-reported fact outside the consumption arithmetic (33,112 training households with a recorded answer; 57.7% hold a bank account with savings).

Each representation is fitted with logistic regression (median imputation, optional `savings_goal.features.transforms.QuantileClipper` at p99, `StandardScaler`) on the households at or below median income and scored on those above it, then the reverse.

| Representation | Winsorised | High → low | Low → high | Mean |
| --- | --- | --- | --- | --- |
| Raw rupees | no | 0.601 | 0.639 | 0.620 |
| Raw rupees | p99 | 0.606 | 0.650 | 0.628 |
| Share of expenditure | no | 0.604 | 0.624 | 0.614 |
| Share of expenditure | p99 | 0.606 | 0.629 | 0.618 |
| Expense / income | no | 0.577 | 0.585 | 0.581 |
| Expense / income | p99 | 0.590 | 0.615 | 0.603 |

Winsorising helps the ratios most (+0.022), as expected for a heavy-tailed representation, but they still transfer worst. Income is the under-reported side of this survey (Phase 1, Q2), which makes it a noisy divisor. Shares remain the spending representation because they do not reconstruct the target on their own, not because they transfer better than raw rupees; raw rupees transfer slightly better and are excluded as leakage.

> **In plain terms: this transfer test.** A random split lets the model see rich and poor households during training. Training only on the poorer half and scoring on the richer half (then the reverse) asks whether what was learned about one income level carries to another, which is the claim made for ratios.

### Cell 5: categorical encoding and missing caste (Q2)

Compares the goal rate of households with and without a recorded `Caste_Group` in the training split: 69 missing, rate 0.333 against 0.319. On 69 households that gap is well inside chance variation. `Unknown` is kept as its own level anyway, so the model never assigns a caste a household did not report. The other three categoricals have no missing values in the training split.

Encoding happens inside `savings_goal.models.pipeline.preprocessor` with `OneHotEncoder(handle_unknown="ignore")`, so a category unseen in a training fold is encoded as all zeros instead of raising an error.

> **In plain terms: inside chance variation.** Two groups of households will differ a little by luck, the way two handfuls of coins rarely give the same number of heads. With 69 households, a 1.4-point gap in the goal rate is the size luck alone produces, so the reading is "no evidence of a difference".

### Cell 7: participation indicators (Q3)

Zero rates on the training split:

| Share | Zero rate |
| --- | --- |
| `Rent_Share` | 0.903 |
| `Insurance_Share` | 0.737 |
| `Eating_Out_Share` | 0.724 |
| `Entertainment_Share` | 0.695 |
| `Education_Share` | 0.359 |
| `Healthcare_Share` | 0.193 |
| `Transport_Share` | 0.114 |
| `Clothing_Footwear_Share` | 0.014 |
| `Utilities_Share` | 0.002 |
| `Miscellaneous_Share` | 0.001 |
| `Groceries_Share` | 0.001 |

The five above 30% get indicators: `Spends_On_Rent`, `Spends_On_Insurance`, `Spends_On_Eating_Out`, `Spends_On_Entertainment`, `Spends_On_Education`. A share alone cannot tell "spends nothing" from "spends a little", and for a straight-line model the step from zero to any spending is a change of kind.

Logistic regression on shares plus demographics, grouped CV:

| Feature set | ROC-AUC | Macro-F1 |
| --- | --- | --- |
| Shares only | 0.917 | 0.814 |
| Shares + indicators | 0.919 | 0.818 |

A small gain, and it applies only to the diagnostic full set; the deployable headline set has no shares or indicators.

### Cell 9: debt handling (Q4)

`Debt_Missing` covers 7.57% of training households. Their goal rate is 0.367 against 0.315 when debt is recorded, a 5-point gap on about 2,500 households, so the flag is kept as a feature instead of imputing the blank as zero debt.

`Debt_To_Income` reaches 1,300 times annual income. Two treatments are compared with logistic regression and a 150-tree random forest:

| Variant | Logistic ROC-AUC | Random forest ROC-AUC |
| --- | --- | --- |
| log1p | 0.9204 | 0.9130 |
| p99 winsorised per fold | 0.9200 | 0.9128 |

The difference is 0.0004 at most, so the choice does not matter for performance. The pipeline uses winsorising: `savings_goal.models.pipeline.preprocessor` routes `Debt_To_Income` through `QuantileClipper(upper=0.99)`, which learns the cap on each training fold (about 11 times income on the full training split) and applies it to the scored fold. A capped ratio in units of annual income is easier to read in Phase 5 than a log.

> **In plain terms: winsorising and log1p.** Winsorising at p99 finds the value only 1% of households exceed and caps everyone above it at that value. log1p replaces x with log(1 + x), which compresses the tail (1,300 becomes about 7.2) and keeps zero at zero. Learning the cap inside each training fold means the scored households never influence where it sits.

> **In plain terms: logistic regression and random forest.** Logistic regression weighs each feature, adds the weights and turns the total into a probability. A random forest grows many decision trees on resampled data and averages their votes. They fail in different ways, so a change that helps both is more convincing than one that helps one.

### Cell 11: log-ratio transform and scaler choice (Q5)

First comparison: the full numeric set with raw shares, against the same set with the six core shares replaced by their centred log-ratio (CLR) columns.

| Representation | Logistic ROC-AUC |
| --- | --- |
| Raw shares | 0.920 |
| CLR core + zero-inflated shares | 0.911 |

The CLR loses by 0.009 and is not used for classification. It expresses each share relative to the household's own typical share, which discards the absolute level (food at 70% of the budget against 35%) that Engel's law makes predictive. The six `*_CLR` columns stay in the feature table. Phase 6 clusters on the same six core shares with the same zero replacement, expressed in an isometric log-ratio (ILR) basis by `savings_goal.models.personas.ilr_coordinates`. Two households spend nothing in any core category; their CLR values are left missing.

> **In plain terms: centred log-ratio.** For each household, take the geometric mean of its core shares, then replace each share with the log of share ÷ that mean. The result says "food is large or small relative to this household's other categories". Zeros have no logarithm, so the core is limited to the six categories zero for under 20% of households, and their remaining zeros are replaced by 0.65 × the smallest positive value seen in training (`savings_goal.features.transforms.multiplicative_replacement`). CLR values sum to zero in every row, which is why Phase 6 drops one dimension with the ILR.

Second, why shares never go through `RobustScaler`. That scaler divides by the interquartile range, which collapses for zero-inflated shares:

| Share | Zero rate | IQR | Max \|scaled value\| |
| --- | --- | --- | --- |
| `Entertainment_Share` | 0.695 | 0.0042 | 154.5 |
| `Insurance_Share` | 0.737 | 0.0073 | 107.3 |
| `Eating_Out_Share` | 0.724 | 0.0072 | 104.1 |
| `Clothing_Footwear_Share` | 0.014 | 0.0353 | 23.2 |
| `Education_Share` | 0.359 | 0.0624 | 15.6 |
| `Groceries_Share` | 0.001 | 0.2093 | 2.5 |
| `Rent_Share` | 0.903 | 0.0000 | 1.0 (scale set to 1) |

Third, the two scalers head to head on the full logistic pipeline: `StandardScaler` ROC-AUC 0.9200, log-loss 0.3605; `RobustScaler` 0.9199, 0.3607. Performance is the same. `StandardScaler` is kept for linear models because it does not produce the values in the hundreds shown above. Tree models are not scaled, since a split only asks whether a value is above a cut-point. Both rules are built into `savings_goal.models.pipeline.preprocessor`.

> **In plain terms: scaling.** Income is in the tens of thousands, shares lie between 0 and 1. Linear models compare coefficient sizes across columns, so columns are put on a common footing first. `StandardScaler` subtracts the mean and divides by the standard deviation; `RobustScaler` subtracts the median and divides by the spread of the middle 50%.

### Cell 13: collinearity, and the saved results (Q6)

Computes the variance inflation factor (VIF) for each numeric feature on standardised training data, first with all 11 shares and then with `Groceries_Share` removed.

| Feature | VIF, all shares | VIF, `Groceries_Share` dropped |
| --- | --- | --- |
| Any of the 11 shares | ≈ 1.0 × 10¹⁵ | 1.11 to 1.85 |
| `Age_Dependents` | 8.92 | 8.92 |
| `Household_Size` | 5.20 | 5.20 |
| `Dependency_Ratio` | 4.60 | 4.60 |
| `Spends_On_Insurance` | 1.89 | 1.89 |
| `Log_Income` | 1.57 | 1.57 |
| `Debt_To_Income` | 1.03 | 1.03 |

With all 11 shares the design is singular: any one share equals one minus the sum of the other ten. Dropping one share as a reference removes that exactly, which is what `preprocessor(..., linear=True)` does whenever all 11 shares are present. Every remaining share coefficient is then read relative to food. The size family (`Age_Dependents`, `Household_Size`, `Dependency_Ratio`) is the only other cluster, and it stays below the conventional threshold of 10. Tree models never invert this matrix, so they are unaffected.

> **In plain terms: VIF.** VIF asks how well the other features predict this one. 1 means not at all; above about 10 is usually treated as a problem; a value near 10¹⁵ means another combination of features reproduces this one exactly. When that happens, a linear model has many equally good ways to split credit between the columns and its individual coefficients mean nothing.

> **In plain terms: the reference share.** Dropping one share is like quoting salaries relative to the entry grade instead of in rupees. With food as the reference, a positive coefficient on insurance means "a larger insurance share, with a correspondingly smaller food share, goes with a higher chance of meeting the goal".

The cell writes every table above, both VIF series and the final column lists to `results/feature_engineering.json` with `savings_goal.io.write_result`, and prints the final sizes: 29 features (25 numeric + 4 categorical), deployable subset 13.

---

## What this means for later phases

| Phase | Consequence |
| --- | --- |
| 3: Baseline | Read `dataset/features.parquet` and use the stored `Is_Test` split; score baselines on the training split under the same grouped CV. |
| 4: Model comparison | Headline models use the 13 deployable features; the 29-feature full set is reported as a diagnostic. Build every pipeline with `savings_goal.models.pipeline.preprocessor`, so debt is winsorised per fold, linear models drop `Groceries_Share` and get `StandardScaler`, and trees are unscaled. |
| 5: Explainability | Share coefficients from linear models are relative to food. Use SHAP on the tree model for attributions; it does not depend on inverting the collinear design. |
| 6: Clustering | Cluster on the six core shares in an ILR basis (five coordinates), with the same zero replacement; drop the 2 households with no core spending. |
