# Phase 2 — Feature Engineering

**Source:** [README § Phase 2 — Feature Engineering](../README.md#phase-2--feature-engineering)
**Notebook:** [`notebooks/02_feature_engineering.ipynb`](../notebooks/02_feature_engineering.ipynb)
**Builds on:** [Phase 1](phase1.md), [Dataset construction](dataset_construction.md)
**Output:** `dataset/features.csv` — 41,518 households × 28 features

Phase 1 left four decisions open. This phase settles each one **empirically** rather than by convention, and two of the four go against the textbook answer — including one of my own proposals.

---

## Research questions & answers

| # | Question | Answer |
| --- | --- | --- |
| 1 | Do expense-to-income ratios generalise better across income levels than raw expense values? | **No — they generalise worst.** Fitting on one half of the income distribution and scoring on the other: raw rupees 0.6245, composition shares 0.6175, expense/income ratios **0.5893**. The intuition that dividing by income aids transfer does not hold on this data. |
| 2 | How should `Occupation` and the other categoricals be encoded? | One-hot, with an explicit `Unknown` level. All four are low-cardinality (4–8 levels), so one-hot costs little and keeps every coefficient interpretable. Missingness is **not** informative (`Caste_Group` missing: Goal_Met 0.3176 vs 0.3193 present), so the `Unknown` level is a transparency choice rather than a signal-preserving one. |
| 3 | Which features require scaling, and does that depend on the downstream model? | Barely matters here: ROC-AUC is 0.9212 / 0.9212 / 0.9211 for RobustScaler / StandardScaler / none under logistic regression, and 0.9162 / 0.9160 / 0.9160 under the forest — despite a **658×** spread in feature standard deviations. Scaling is still specified for the linear models because it makes coefficients comparable and helps convergence, not because it moves the score. |
| 4 | Are any features redundant or highly collinear? | **Yes, and structurally so.** The 11 shares have **VIF = ∞** because they sum to 1 — they are exactly singular as a set. Beyond that: `Household_Size` × `Dependents` r = 0.714 (VIF 8.98 / 5.23), and each `*_Share` × `Spends_On_*` pair r ≈ 0.45–0.67 by construction. |

---

## The four open decisions

### Decision 1 — Categorical missingness: explicit `Unknown`

`Caste_Group` is missing for 85 households (0.21%) and `Religion` for 12 (0.03%). The question was whether refusal to state caste is itself informative.

**It is not.** Goal_Met among households with missing `Caste_Group` is 0.3176 against 0.3193 for those with it recorded — a 0.17 percentage point difference on 85 households, indistinguishable from noise. Either imputation or an explicit level is therefore defensible on predictive grounds.

`Unknown` is used anyway, because at this rate the choice cannot affect the model and an explicit level does not silently assert a caste the household declined to give. That is a reporting-integrity reason, not a statistical one, and it is worth being clear that the evidence did not force it.

### Decision 2 — Zero-inflation: participation indicators, kept

Phase 1 found five categories zero for more than 30% of households. A single share column conflates "spends nothing on this" with "spends a little", which are different states. Binary `Spends_On_*` indicators were added for all five and tested.

**Result:** ROC-AUC 0.9183 → **0.9204**, macro-F1 0.8160 → 0.8195. A small gain, but consistent, and the univariate breakdown shows why:

| Indicator | Goal_Met if no | Goal_Met if yes | Lift |
| --- | --- | --- | --- |
| `Spends_On_Insurance` | 0.2938 | 0.3906 | **+0.0968** |
| `Spends_On_Eating_Out` | 0.3067 | 0.3515 | +0.0448 |
| `Spends_On_Entertainment` | 0.3125 | 0.3346 | +0.0221 |
| `Spends_On_Rent` | 0.3198 | 0.3148 | −0.0050 |
| `Spends_On_Education` | 0.3721 | 0.2897 | **−0.0825** |

The two large effects point in opposite directions and both are interpretable. Paying insurance premiums at all is a marker of financial slack — households already saving are the ones who buy insurance. Paying school fees at all costs **8.3 percentage points** of goal attainment: it is a large, non-negotiable claim on income that falls on households with dependents. That is a genuine finding about dependent burden, and it is the sort of thing a single continuous share would have blurred, because the difference between ₹0 and ₹1 of school fees is categorical, not marginal.

`Spends_On_Rent` is essentially flat (−0.005) — paying rent at all does not predict the outcome, which is unsurprising given Phase 1's finding that 90.4% of households record no rent.

### Decision 3 — `Debt_To_Income`: winsorised at the 99th percentile

Phase 1 flagged a tail reaching 1,300× annual income. Three variants were compared across both model families:

| Variant | logreg ROC-AUC | rf ROC-AUC | logreg F1 |
| --- | --- | --- | --- |
| Raw | 0.9210 | 0.9172 | 0.8201 |
| Winsorised (p99 = 10.234) | 0.9212 | 0.9168 | 0.8202 |
| `log1p` | **0.9215** | 0.9172 | **0.8203** |

**The honest answer is that this decision does not matter** — the spread is 0.0005 ROC-AUC across all three, far inside fold-to-fold noise. `log1p` is nominally best on the linear model and the raw form is nominally best on the forest, which is itself a sign that the differences are noise rather than signal.

Winsorisation at p99 is used (affecting 416 households) because when nothing separates the options on performance, the tiebreaker should be interpretability: a coefficient on a variable bounded at 10× income is readable, one on a variable reaching 1,300× is not. `Has_Debt` is carried alongside it, since roughly half of households have no debt at all and the same zero-inflation argument as Decision 2 applies.

### Decision 4 — The compositional transform: **proposed, tested, rejected**

Phase 1 argued that the shares live on a simplex, so Euclidean methods are not well-founded on them, and that a centred log-ratio (CLR) transform is the standard fix. That was my recommendation going into this phase. The notebook implemented it properly and the evidence rejected it.

**Implementation.** The transform was restricted to a **core sub-composition** — the six parts zero for under 20% of households (`Groceries`, `Utilities`, `Transport`, `Healthcare`, `Clothing_Footwear`, `Miscellaneous`). Forcing `Rent_Share` through a log when it is zero for 90.4% of households would produce a column that is mostly imputed constant. Zeros in the core were handled by multiplicative replacement at 0.65× the smallest observed positive value in each part (Martín-Fernández et al.), which is the standard method:

| Part | δ |
| --- | --- |
| `Groceries_Share` | 2.230×10⁻³ |
| `Miscellaneous_Share` | 6.776×10⁻⁴ |
| `Utilities_Share` | 2.505×10⁻⁴ |
| `Clothing_Footwear_Share` | 1.129×10⁻⁴ |
| `Healthcare_Share` | 8.302×10⁻⁵ |
| `Transport_Share` | 6.738×10⁻⁵ |

The result satisfies the defining property of a CLR — rows sum to zero, max |row sum| = 7.99×10⁻¹⁵.

**Result:**

| Feature set | logreg ROC-AUC | rf ROC-AUC | logreg F1 |
| --- | --- | --- | --- |
| Raw shares | **0.9212** | **0.9168** | **0.8202** |
| CLR core + zero-inflated shares | 0.9117 | 0.9095 | 0.8097 |

**Why the theoretically-correct transform lost.** Two reasons, and the second is decisive:

1. It discards information. The CLR is scale-free *within* the composition, which is exactly the property that makes it appropriate for clustering — but the absolute level of a share (`Groceries_Share` = 0.70 vs 0.35) is directly informative about the standard of living, and Phase 1 showed that is Engel's law doing real predictive work. The log-ratio deliberately throws that away.
2. **CLR components sum to zero by construction, so their covariance matrix is exactly singular.** The first run recorded VIF up to 3×10⁵ for the CLR columns. A CLR basis can never be used safely in an unregularised linear model. The standard remedy is an **isometric log-ratio (ILR)** basis, which drops one dimension and is therefore full-rank.

**Outcome:** CLR is excluded from the classification feature set. The six CLR columns are still written to `features.csv` for **Phase 6** to evaluate, because clustering is the step that genuinely needs a metric on the simplex — but Phase 6 should test ILR rather than CLR, for the singularity reason above.

**A first-run bug worth recording.** The initial implementation used a single flat `δ = 10⁻⁵`, which is 10–30× smaller than the smallest genuine values in these parts, and manufactured artificial extremes at ±9.59 (the CLR of a household where one part is ~1 and the rest are imputed δ). It also produced `NaN` for the **2 households that spend nothing on any core category**, whose composition is undefined. Both were fixed before the comparison above; the corrected CLR ranges are ±2.7 to ±7.3 rather than ±9.6, and the 2 undefined households are left as `NaN` rather than silently filled.

---

## Q1 in detail — why the ratio intuition fails here

The question of whether expense-to-income ratios generalise better than raw values **cannot be tested against `Goal_Met`**: Phase 1 showed raw rupee categories reconstruct the target with 99.75% agreement, so the leaky representation would win by construction and the result would be meaningless.

The comparison is instead run against **`Has_Bank_Savings`** — a survey-reported behaviour that is not an accounting function of the expense columns, so all three representations face it on equal terms. The real question is generalisation *across income levels*, so the test fits on one half of the income distribution and scores on the other:

| Representation | high → low | low → high | mean |
| --- | --- | --- | --- |
| Raw rupees | 0.6063 | 0.6428 | **0.6245** |
| Share of expenditure | 0.6075 | 0.6275 | 0.6175 |
| Expense / income | 0.5828 | 0.5957 | 0.5893 |

**Expense-to-income ratios transfer worst.** Raw rupees and shares are within 0.007 of each other; the ratio form loses 3.5 points.

**Why the intuition fails.** Dividing by income is worthwhile when the raw columns are largely restatements of income — the division removes a redundancy that would otherwise crowd out everything else. Phase 1 measured that correlation at only **0.09–0.43** here, so there is little redundancy to remove. What the division does instead is inject the noise in a poorly-measured denominator: recall from Phase 1 that income is the *under-reported* side of this survey, so it is the worst available choice of divisor.

**This does not undo the feature-set design.** Composition shares remain the right representation, but for a specific reason: they are **not reconstructable from the target** (Phase 1, Test C), not because they generalise better than raw values — which they marginally do not. Raw rupees generalise slightly better and are unusable for leakage reasons. That trade is now explicit rather than assumed.

---

## Q4 in detail — the collinearity that cannot be engineered away

```text
Groceries_Share            inf
Eating_Out_Share           inf
Utilities_Share            inf
... all 11 shares          inf
Dependents                8.98
Household_Size            5.23
Dependency_Ratio          4.61
Log_Income                1.68
```

**The 11 shares sum to exactly 1, so one is a perfect linear combination of the other ten.** This is the same singularity that disqualified CLR, present in the raw shares as well — it is a property of compositional data, not of either transform.

**Why the pipeline still works:** both models in use are robust to it. L2-regularised logistic regression has a unique solution even with a singular design matrix, and tree ensembles never invert one. The measured performance (0.9212) is real.

**What later phases must not do:**

- **Do not fit an unregularised linear model** (plain OLS/`LinearRegression`, or `LogisticRegression(penalty=None)`) on the full share set. It is rank-deficient and the coefficients will be arbitrary.
- **Phase 5 must not read individual share coefficients as identified effects.** With a singular set, the split of credit between shares is determined by the regulariser, not the data. Either drop one share as an explicit reference part (the compositional analogue of a dummy baseline) before interpreting, or use SHAP on the tree model, which does not depend on invertibility.

`Dependents` at VIF 8.98 sits just under the conventional threshold of 10 and is explained by `Household_Size` (r = 0.714) and `Dependency_Ratio` (r = 0.678), which is arithmetic — the ratio is built from the other two. Keeping all three is defensible for the tree models; a linear model should take `Dependency_Ratio` and `Household_Size` and drop the raw count.

---

## Final feature set — 28 features

**Numeric (24):** 11 `*_Share`, `Log_Income`, `Household_Size`, `Dependents`, `Dependency_Ratio`, `Head_Age`, `Max_Adult_Education`, 5 `Spends_On_*`, `Debt_To_Income_W`, `Has_Debt`

**Categorical (4):** `Occupation`, `Area_Type`, `Caste_Group`, `Religion`

**Also exported, not features:** 6 `*_CLR` columns (for Phase 6 only), `Goal_Met`, `Savings_Rate`, `IDHH`, `WT`

| Model | ROC-AUC | F1 (macro) |
| --- | --- | --- |
| Logistic Regression | **0.9212** | **0.8203** |
| Random Forest | 0.9160 | 0.8162 |

Notably the linear model is **ahead** of the forest on both metrics, which is unusual on tabular data of this size. The participation indicators are the likely reason: they hand the linear model the threshold effects it cannot otherwise express, which is exactly what a tree would have had to spend splits discovering.

---

## What this changes for later phases

| Phase | Consequence |
| --- | --- |
| **3 — Baseline** | Load `dataset/features.csv` directly. Majority baseline: 0.681 accuracy / 0.405 macro-F1. |
| **4 — Model comparison** | Scaling barely moves the score, but keep `RobustScaler` for linear models given skew up to 7.4. **Do not** include an unregularised linear model. Logistic regression is the one to beat at 0.9212. |
| **5 — Explainability** | Individual share coefficients are not identified (VIF = ∞). Use SHAP on a tree model, or drop a reference part first. |
| **6 — Clustering** | Use the exported `*_CLR` columns, but prefer an **ILR** basis — CLR is singular. The 2 all-zero-core households must be excluded or handled explicitly. |
| **7 — Business translation** | `Spends_On_Education` (−8.3pp) and `Spends_On_Insurance` (+9.7pp) are the two most actionable univariate signals found so far. |
