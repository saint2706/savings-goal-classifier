# Phase 2 — Feature Engineering

**Source:** [README § Phase 2 — Feature Engineering](../README.md#phase-2--feature-engineering)
**Notebook:** [`notebooks/02_feature_engineering.ipynb`](../notebooks/02_feature_engineering.ipynb)
**Builds on:** [Phase 1](phase1.md), [Dataset construction](dataset_construction.md)
**Output:** `dataset/features.csv` — 41,518 households × 28 features

Phase 1 left four decisions open. This phase settles each one **empirically** rather than by convention, and two of the four go against the textbook answer — including one of my own proposals.

> **In plain terms — feature engineering, and "empirically".** **Feature engineering** is the work of turning raw survey columns into the inputs a model actually sees: combining them, rescaling them, splitting a single messy column into two clean ones, deciding what to do with blanks. It is where most of the judgement in a project like this lives.
>
> Settling a decision **empirically** means building both versions, scoring both, and keeping whichever wins — rather than following the textbook rule. The phrase matters here because two of the four textbook answers lose.

---

## Research questions & answers

| # | Question | Answer |
| --- | --- | --- |
| 1 | Do expense-to-income ratios generalise better across income levels than raw expense values? | **No — they generalise worst.** Fitting on one half of the income distribution and scoring on the other: raw rupees 0.6245, composition shares 0.6175, expense/income ratios **0.5893**. The intuition that dividing by income aids transfer does not hold on this data. |
| 2 | How should `Occupation` and the other categoricals be encoded? | One-hot, with an explicit `Unknown` level. All four are low-cardinality (4–8 levels), so one-hot costs little and keeps every coefficient interpretable. Missingness is **not** informative (`Caste_Group` missing: Goal_Met 0.3176 vs 0.3193 present), so the `Unknown` level is a transparency choice rather than a signal-preserving one. |
| 3 | Which features require scaling, and does that depend on the downstream model? | Barely matters here: ROC-AUC is 0.9212 / 0.9212 / 0.9211 for RobustScaler / StandardScaler / none under logistic regression, and 0.9162 / 0.9160 / 0.9160 under the forest — despite a **658×** spread in feature standard deviations. Scaling is still specified for the linear models because it makes coefficients comparable and helps convergence, not because it moves the score. |
| 4 | Are any features redundant or highly collinear? | **Yes, and structurally so.** The 11 shares have **VIF = ∞** because they sum to 1 — they are exactly singular as a set. Beyond that: `Household_Size` × `Dependents` r = 0.714 (VIF 8.98 / 5.23), and each `*_Share` × `Spends_On_*` pair r ≈ 0.45–0.67 by construction. |

> **In plain terms — the four pieces of jargon in that table.**
> - **Encoding / one-hot.** Models do arithmetic, so a text column like `Occupation = "Farm"` has to become numbers. Numbering the categories 1–6 would be a disaster, because the model would infer that Farm is twice Salaried and that the midpoint between them means something. **One-hot encoding** avoids that by replacing one text column with one yes/no column per category: `Is_Farm`, `Is_Salaried`, and so on, exactly one of which is 1. **Cardinality** is just how many distinct categories a column has — with only 4–8 here, one-hot adds a handful of columns and costs nothing. With thousands of categories it would be unusable.
> - **Scaling.** Different columns arrive on wildly different scales: income in hundreds of thousands, shares between 0 and 1. Some model families treat a big-numbered column as automatically important, so columns are **scaled** to a common footing first — **StandardScaler** using the average and standard deviation, **RobustScaler** using the median and the middle 50% (the version that shrugs off outliers, see [Phase 1](phase1.md) on skew). Tree-based models are immune to all of this, since they only ever ask "is this value above or below a cut-point?"
> - **Collinear / redundant.** Two features are **collinear** when they carry nearly the same information — household size and number of dependents, for instance. The model still works, but it can no longer tell which of the two deserves the credit.
> - **VIF** (variance inflation factor) puts a number on that. It answers "how well can the other features predict this one?" 1 means not at all (perfectly independent); above about 10 is conventionally considered a problem; **∞ means another feature reproduces this one exactly**, which is what the eleven-slices-of-one-pie constraint guarantees. Q4 below explains why that is survivable for prediction and fatal for interpretation.

---

## The four open decisions

### Decision 1 — Categorical missingness: explicit `Unknown`

`Caste_Group` is missing for 85 households (0.21%) and `Religion` for 12 (0.03%). The question was whether refusal to state caste is itself informative.

**It is not.** Goal_Met among households with missing `Caste_Group` is 0.3176 against 0.3193 for those with it recorded — a 0.17 percentage point difference on 85 households, indistinguishable from noise. Either imputation or an explicit level is therefore defensible on predictive grounds.

> **In plain terms — "indistinguishable from noise".** Any two groups of households will differ *a little* by pure chance, the way two handfuls of coins rarely give identical head counts. **Noise** is that meaningless random variation. On only 85 households a swing of a few tenths of a percentage point is well inside what chance alone produces, so the honest reading is "we found nothing", not "we found a small effect". This phrase recurs throughout the project, and it is usually the reason a decision gets made on grounds *other* than the score.

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

> **In plain terms — the three variants and the two models.**
> - **Raw** = leave the runaway values alone. **Winsorised at p99** = find the value only 1% of households exceed (here 10.234× income) and cap everyone above it at that number. **`log1p`** = replace each value `x` with `log(1 + x)`, which squashes the tail — 1,300 becomes about 7.2 while 0.5 stays about 0.4. The "1 +" is there so that a debt of zero maps to zero instead of to the undefined log of 0.
> - **logreg** is **logistic regression**, the standard straight-line classifier: it weighs each feature, adds the weights up, and converts the total into a probability. **rf** is a **random forest**: hundreds of yes/no decision trees, each grown on a slightly different slice of the data, voting together. They are included as a pair deliberately — they fail in different ways, so a change that helps both is real, while one that helps only one is usually noise.

**The honest answer is that this decision does not matter** — the spread is 0.0005 ROC-AUC across all three, far inside fold-to-fold noise. `log1p` is nominally best on the linear model and the raw form is nominally best on the forest, which is itself a sign that the differences are noise rather than signal.

Winsorisation at p99 is used (affecting 416 households) because when nothing separates the options on performance, the tiebreaker should be interpretability: a coefficient on a variable bounded at 10× income is readable, one on a variable reaching 1,300× is not. `Has_Debt` is carried alongside it, since roughly half of households have no debt at all and the same zero-inflation argument as Decision 2 applies.

### Decision 4 — The compositional transform: **proposed, tested, rejected**

Phase 1 argued that the shares live on a simplex, so Euclidean methods are not well-founded on them, and that a centred log-ratio (CLR) transform is the standard fix. That was my recommendation going into this phase. The notebook implemented it properly and the evidence rejected it.

> **In plain terms — what a CLR actually does.** Recall the problem from [Phase 1](phase1.md): the eleven shares are slices of one pie, so they are not free to vary independently and ordinary straight-line distance mismeasures how different two budgets are.
>
> The **centred log-ratio** transform is the textbook repair. For each household, work out the typical size of its slices (specifically the **geometric mean**, an average that suits ratios); then replace every share with the logarithm of that share divided by the typical size. In words, each number stops saying *"food is 45% of the budget"* and starts saying *"food is unusually large or small relative to this household's other categories."* Once expressed that way, the values are free to move independently again and normal geometry applies.
>
> Two consequences follow, and both bite later. First, the transform is deliberately blind to the absolute level of a share — that is exactly what "relative to this household's own budget" buys, and exactly what turns out to be predictive information we cannot afford to lose. Second, the transformed values now sum to **zero** for every household instead of to one, which swaps one constraint for another rather than removing it.

**Implementation.** The transform was restricted to a **core sub-composition** — the six parts zero for under 20% of households (`Groceries`, `Utilities`, `Transport`, `Healthcare`, `Clothing_Footwear`, `Miscellaneous`). Forcing `Rent_Share` through a log when it is zero for 90.4% of households would produce a column that is mostly imputed constant. Zeros in the core were handled by multiplicative replacement at 0.65× the smallest observed positive value in each part (Martín-Fernández et al.), which is the standard method:

> **In plain terms — sub-composition and zero replacement.** A **sub-composition** is a subset of the slices, re-scaled to be a pie of its own. Six of the eleven categories are used here because the transform needs logarithms, **and the logarithm of zero does not exist** — so any category that is zero for most households cannot go through it.
>
> Even the six chosen categories have some zeros, so those have to be replaced with something small but positive. **Multiplicative replacement** is the established recipe: for each category, take the smallest genuine non-zero value anyone recorded and use 0.65 of it as the stand-in (this stand-in is written **δ**, the Greek letter delta). It is called *multiplicative* because the other slices are then shrunk proportionally so the pie still totals one. The point of tying δ to each category's own smallest real value — rather than picking one convenient number like 0.00001 — is that the substitute stays on the same scale as the data. The bug recorded at the end of this section is precisely what happens when that is not respected.

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

> **In plain terms — singular, full-rank, and why ILR fixes it.** A **covariance matrix** is the grid of how the features vary together (the correlation matrix from [Phase 1](phase1.md), in unstandardised form). Fitting a linear model requires, in effect, dividing by that grid — and **singular** means the division is impossible, the matrix equivalent of dividing by zero. It happens whenever one feature can be reconstructed exactly from the others, which the sum-to-zero constraint guarantees. **Full-rank** is the healthy opposite: no feature is a rebuild of the rest, and the division works.
>
> The **ILR (isometric log-ratio)** basis solves this by keeping the same geometry but expressing it in one fewer column — five coordinates instead of six. That is not a loss: because the six always summed to zero, only five of them were ever carrying independent information. Removing the redundant one removes the singularity. Note the CLR is not abandoned; [Phase 6](phase6.md) builds its ILR by transforming the CLR values, since clustering is the task that genuinely needs simplex geometry.

**Outcome:** CLR is excluded from the classification feature set. The six CLR columns are still written to `features.csv` for **Phase 6** to evaluate, because clustering is the step that genuinely needs a metric on the simplex — but Phase 6 should test ILR rather than CLR, for the singularity reason above.

**A first-run bug worth recording.** The initial implementation used a single flat `δ = 10⁻⁵`, which is 10–30× smaller than the smallest genuine values in these parts, and manufactured artificial extremes at ±9.59 (the CLR of a household where one part is ~1 and the rest are imputed δ). It also produced `NaN` for the **2 households that spend nothing on any core category**, whose composition is undefined. Both were fixed before the comparison above; the corrected CLR ranges are ±2.7 to ±7.3 rather than ±9.6, and the 2 undefined households are left as `NaN` rather than silently filled.

> **In plain terms — why too small a δ is worse than none.** Logarithms magnify small numbers ferociously: the gap between 0.001 and 0.00001 is a hundredfold, and after taking logs it looks like an enormous distance. So a stand-in chosen far below the real data does not quietly fill a hole — it **fabricates households at the extremes**, sitting further from everyone else than any genuine household does. Every later method that measures distance would then organise itself around a value nobody actually reported. That is why δ is anchored to each category's own smallest observed value.
>
> **`NaN`** is the "not a number" marker software uses for an undefined result. Two households recorded no spending at all in any of the six core categories, so their budget shape is genuinely undefined — there is no pie to slice. They are left marked as undefined rather than filled with a fake value, and [Phase 6](phase6.md) drops them explicitly and says so.

---

## Q1 in detail — why the ratio intuition fails here

The question of whether expense-to-income ratios generalise better than raw values **cannot be tested against `Goal_Met`**: Phase 1 showed raw rupee categories reconstruct the target with 99.75% agreement, so the leaky representation would win by construction and the result would be meaningless.

The comparison is instead run against **`Has_Bank_Savings`** — a survey-reported behaviour that is not an accounting function of the expense columns, so all three representations face it on equal terms. The real question is generalisation *across income levels*, so the test fits on one half of the income distribution and scores on the other:

> **In plain terms — generalisation, and this particular test.** **Generalising** means working on households the model has never seen. Scoring a model on the same rows it learned from proves nothing — it can memorise. So the data is always split: learn on one part, score on another.
>
> This test uses a deliberately harsh split. Instead of a random division, it trains only on the **richer half** of households and scores on the **poorer half**, then reverses. That asks a much tougher question — does what the model learned about rich households transfer to poor ones? — and it is the right question here, because "dividing by income makes households comparable across income levels" is precisely the claim being tested. A random split would have let the model see both halves and would never have exposed the failure.

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

> **In plain terms — regularisation, and why it rescues a broken system.** With the shares summing to 1, there is no single right answer for how much credit each share deserves: you could add 1 to one coefficient and subtract 1 from another and get *identical* predictions, forever. The system has infinitely many equally good solutions.
>
> **L2 regularisation** (also called a **penalty** or *ridge*) breaks the tie by adding a rule: among all solutions that fit equally well, prefer the one whose coefficients are smallest overall. That is enough to single out exactly one answer, so the model becomes computable and its predictions are perfectly sound. But read what just happened — **the tie-break was chosen by the penalty, not by the data.** The predictions are trustworthy; the individual coefficients are an arbitrary pick from an infinite set. That distinction is the whole reason [Phase 5](phase5.md) refuses to interpret them.
>
> **Tree ensembles** (random forests, gradient boosting) never face the problem at all, because they only ever compare a value against a cut-point and never solve a system of equations.

**What later phases must not do:**

- **Do not fit an unregularised linear model** (plain OLS/`LinearRegression`, or `LogisticRegression(penalty=None)`) on the full share set. It is rank-deficient and the coefficients will be arbitrary.

  > **In plain terms.** **Unregularised** means no tie-break rule — so the model is handed the infinite set of equally-good answers with no way to choose, and returns whichever one the arithmetic stumbles into, which can change completely on a slightly different sample. **OLS (ordinary least squares)** is the plainest such fit, the classic straight-line-through-the-points method. **Rank-deficient** is the same condition as *singular*: the features contain fewer genuinely independent pieces of information than there are columns.

- **Phase 5 must not read individual share coefficients as identified effects.** With a singular set, the split of credit between shares is determined by the regulariser, not the data. Either drop one share as an explicit reference part (the compositional analogue of a dummy baseline) before interpreting, or use SHAP on the tree model, which does not depend on invertibility.

  > **In plain terms — "identified", and the reference-part idea.** A coefficient is **identified** when the data pins it to one value. Here it is not, for the reason above. One standard escape is to nominate one share as the **reference part** and quote every other coefficient *relative to it* — the same trick as reporting salaries "compared to the graduate-entry grade" rather than in absolute terms. Comparisons against a fixed baseline are well-defined even when the absolute levels are not. [Phase 5](phase5.md) tries this and reports that it only partly works.

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
