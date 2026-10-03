# Phase 1: Data Understanding and the leakage check

**Source:** [README § Phase 1: Data Understanding](../README.md#phase-1-data-understanding)
**Notebook:** [`notebooks/01_eda_and_leakage_check.ipynb`](../notebooks/01_eda_and_leakage_check.ipynb)
**Builds on:** [Dataset construction](dataset_construction.md), [Phase 0](phase0.md)
**Artifacts:** `results/phase1_eda.json`, `results/leakage.json`, `results/phase1_eda.png`, `results/phase1_share_correlations.png`

Six questions answered on 41,518 IHDS-II households (42,152 raw, 634 dropped upstream for non-positive income or missing consumption) in 2,461 primary sampling units. Most of this page is description. The exception is the leakage check in Q5, which decides which feature set the rest of the project treats as the headline model.

> **In plain terms: what exploratory analysis is for.** Before fitting any model you look at the data: what each column holds, what shape the numbers take, what is missing, what looks broken. No predictions come out of this stage. Its value is in the decisions it forces later. The vocabulary from [Dataset construction](dataset_construction.md) (feature, target, leakage, median, correlation) is assumed here.

---

## Research questions & answers

| # | Question | Answer |
| --- | --- | --- |
| 1 | What does each column mean, and what unit/time period does it represent? | 54 columns. All money is annual rupees and the grain is a household. Roles: 7 identifiers, 1 survey weight (`WT`), 1 target, 15 leakage columns (the 11 rupee categories, `Category_Total`, `COTOTAL`, `Savings`, `Savings_Rate`), 6 `Has_*` savings-instrument columns kept for external validation, 1 build diagnostic (`Occupation_Tie`), 23 feature columns (`INCOME` enters as `Log_Income`). One cross-section, 2011-12. |
| 2 | Distribution of income, expenses and savings: skew, outliers, implausible values? | Heavily skewed: `INCOME` skew 15.8, `Clothing_Footwear` 112.3, `Utilities` 48.0. 55.89% of households report consumption above income (57.69% survey-weighted) and 22.01% spend more than twice their income. Median savings rate is −10.8%. 90.45% record no rent. |
| 3 | Missing values or duplicate rows? | `Debt_To_Income` is missing for 3,150 households (7.59%), flagged by `Debt_Missing`. Everything else is under 0.5% (`Has_*` 0.32–0.42%, `Caste_Group` 0.21%, `Max_Adult_Education` 6 rows). Zero duplicate household identifiers. |
| 4 | How correlated are expense categories with income and with each other? | Spearman correlation of raw rupee categories with income runs from 0.09 (healthcare) to 0.58 (groceries). Among the shares, `Groceries_Share` falls with income (−0.280) and `Insurance_Share` (+0.332) and `Transport_Share` (+0.328) rise. 56% of share pairs correlate negatively because the shares sum to one. |
| 5 | Is the target derivable from any candidate feature, singly or jointly? | Raw rupee categories and expense/income ratios reconstruct `Goal_Met` for 99.75% of households and are excluded. No single share carries the label (largest \|marginal AUC − 0.5\| = 0.10, healthcare). Jointly, the shares do: log food spend on log household size gives R² 0.29 (elasticity 0.63), and the size × food-share oracle reaches ROC-AUC 0.895 with no model. In the ablation, removing all shares and participation indicators keeps 94.3% of the full model's ROC-AUC (0.878 vs 0.931). The headline model therefore uses income, demographics and debt only (13 features); the 29-feature set with spending shares is diagnostic. |
| 6 | Class balance of `Goal_Met`? | 13,256 met vs 28,262 not met: 31.93% positive, 2.13 : 1. Survey-weighted 30.47%. The positive rate moves from 44.1% at a 0% threshold to 18.8% at 40%. Attainment rises from less-developed villages (26.7%) to metro areas (42.1%), and the gradient holds weighted. |

---

## Notebook walkthrough

The notebook holds section headers and code. All logic lives in the `savings_goal` package; the notebook calls it and prints aggregates.

### Cell 1: imports and load

Loads `dataset/households.parquet` with `savings_goal.io.load_households` and `dataset/features.parquet` with `savings_goal.io.load_features`, then recovers the column lists (participation indicators, log-ratio columns) from the saved feature table with `savings_goal.features.engineer.spec_from_frame`. Column families come from `savings_goal.config`: `EXPENSE_CATEGORIES`, `SHARE_COLS`, `LEAKAGE_COLS`, `CATEGORICALS` and `GROUP_COL`.

It prints 41,518 households, 54 columns and 2,461 PSUs. The PSU key is `IDPSU`. `PSUID` on its own repeats across districts (39 distinct values), so it cannot identify a village or urban block.

> **In plain terms: PSU.** IHDS does not sample households one at a time. It picks villages and urban blocks (primary sampling units) and interviews several households in each. Neighbours share prices, jobs and often the same interviewer, so they look alike. Every train/test split in this project keeps a PSU's households together.

### Cell 3: column inventory (Q1)

Assigns each column a role and unit and counts them:

| Role | Columns |
| --- | --- |
| Feature | 22 |
| Feature (as `Log_Income`) | 1 |
| Leakage, excluded | 15 |
| Identifier | 7 |
| External validation, excluded | 6 |
| Survey weight | 1 |
| Target | 1 |
| Build diagnostic | 1 |

The leakage role is taken from `savings_goal.config.LEAKAGE_COLS`, so the notebook cannot drift from the package's definition. Two unit facts matter for anyone reusing the data. Money is annual, not monthly: IHDS reports `INCOME` and `COTOTAL` as annual rupees, so a monthly threshold is off by a factor of 12. And the grain is a household: `Household_Size` and `Age_Dependents` are counts within the household, so per-person framing needs `INCOMEPC`/`COPC` from the raw file.

### Cell 5: distributions and implausible values (Q2)

Prints mean, sd, quartiles, max and skew for the 14 money columns, the savings-rate percentiles, and a table of awkward values (unweighted and weighted with `savings_goal.evaluation.metrics.weighted_mean`).

| Column | Skew | Column | Skew |
| --- | --- | --- | --- |
| `Clothing_Footwear` | 112.3 | `Insurance` | 31.5 |
| `Utilities` | 48.0 | `Entertainment` | 26.8 |
| `Healthcare` | 35.4 | `Rent` | 24.3 |
| `Transport` | 17.8 | `INCOME` | 15.8 |

> **In plain terms: skew.** Skew measures how lopsided a distribution is. Zero is symmetric. A large positive value means most households sit at the low end and a few stretch far to the right. Above 5 is severe; 112 means one or two households reported a single enormous purchase in an annual-recall category. Averages and standard deviations are dominated by those few, which is why income enters models as `Log_Income`.

The savings-rate mean (−1.16) and sd (13.08) are unreadable because the left tail reaches −1,647 (a household consuming about 1,648 times its reported income). The percentiles carry the information: 1st −15.73, 25th −0.84, median −0.108, 75th +0.305, 99th +0.803.

| Check | Households | Unweighted | Weighted |
| --- | --- | --- | --- |
| Consumption exceeds income | 23,204 | 55.89% | 57.69% |
| Spends more than 2× income | 9,137 | 22.01% | 23.80% |
| No rent recorded | 37,553 | 90.45% | 91.01% |
| Occupation tie broken by priority | 7,491 | 18.04% | 17.93% |
| Debt (DB5) missing | 3,150 | 7.59% | 7.51% |
| Debt above 10× income | 430 | 1.04% | 1.08% |
| Head age below 18 | 14 | 0.03% | 0.04% |

Consumption above income for most of the sample is a known property of Indian household surveys, so these rows are kept. Income is under-reported (irregular, informal and in-kind earnings are recalled poorly, and IHDS records negative farm income for some households) while consumption is collected item by item with short recall windows. Dropping 56% of households would leave a sample biased toward salaried, formally employed ones. The consequence is that `Goal_Met` levels are biased downward at every threshold, so later phases lean on relative comparisons rather than "X% of Indian households save enough".

### Cell 7: missing values and duplicates (Q3)

| Column | Missing | % |
| --- | --- | --- |
| `Debt_To_Income` | 3,150 | 7.59 |
| `Has_Gold_Jewellery` | 176 | 0.42 |
| `Has_Pension_LIC` | 155 | 0.37 |
| `Has_Post_Office_Account` | 142 | 0.34 |
| `Has_Fixed_Deposit` | 138 | 0.33 |
| `Has_Bank_Savings` | 133 | 0.32 |
| `Has_Securities` | 133 | 0.32 |
| `Caste_Group` | 85 | 0.21 |
| `Max_Adult_Education` | 6 | 0.01 |

No household identifier is duplicated. The missing debt values come from the debt question (DB5) not being answered; the build keeps them as missing and adds `Debt_Missing` rather than reading a blank as zero debt. Phase 2 tests whether that flag carries signal. The `Has_*` gaps cluster at 0.3–0.4%, consistent with a few households skipping the savings block; since those columns are only used for validation, the affected households are dropped from validation comparisons rather than counted as "no".

> **In plain terms: imputation.** A model cannot take a blank cell, so each gap needs a rule. Imputation fills it with a guess such as the median. The alternative is to mark the gap ("Unknown", or a 0/1 missing flag) so the fact that it was blank stays visible. Which is right depends on whether households with blanks differ from the rest.

### Cell 9: correlation with income and between shares (Q4)

Computes Spearman correlations, which use ranks and so are not driven by the extreme values from Cell 5.

Raw rupee categories against income: groceries 0.581, utilities 0.547, transport 0.504, miscellaneous 0.472, clothing 0.437, insurance 0.358, education 0.281, entertainment 0.250, eating out 0.219, rent 0.113, healthcare 0.086.

Shares against log income:

| Share | Spearman ρ with income |
| --- | --- |
| `Insurance_Share` | +0.332 |
| `Transport_Share` | +0.328 |
| `Entertainment_Share` | +0.220 |
| `Education_Share` | +0.193 |
| `Healthcare_Share` | −0.122 |
| `Groceries_Share` | −0.280 |

Food taking a smaller slice of the budget as income rises is Engel's law. Nothing in the pipeline was built to produce it, which is some evidence the shares measure real budget structure.

> **In plain terms: Engel's law.** As households get richer they spend more rupees on food but a smaller proportion of their budget on it. It is one of the oldest regularities in economics (1857).

The cell then draws `results/phase1_share_correlations.png`: the lower triangle of the share-by-share Spearman matrix, annotated with each value. 56% of the off-diagonal pairs are negative. The strongest negative pair is groceries with transport (−0.32); the strongest positive is eating out with entertainment (+0.24).

> **In plain terms: compositional closure.** The eleven shares are slices of one pie that always totals 1. A slice can only grow by taking room from the others, so negative correlations appear even if households chose at random. This is why the share pairs lean negative, and why a share's effect is always "more of X and correspondingly less of everything else". Data like this is called compositional; it needs log-ratio tools for distance-based methods (Phase 2 and Phase 6).

### Cell 11: the leakage check, single representations (Q5, part 1)

Four steps.

1. `savings_goal.data.build.reconcile_totals` compares the survey's `COTOTAL` with the sum of the 11 categories (`Category_Total`). Pearson r 0.9932, median difference Rs 0, 97.72% within 1% and 98.29% within 5%. Recomputing the label from the category sum agrees with `Goal_Met` for 99.75% of households. The target uses `COTOTAL` (the official aggregate); the shares are computed against `Category_Total` so they sum to exactly one.
2. Test A: raw rupee categories plus `INCOME` reproduce `Goal_Met` for 99.75% of households. Excluded.
3. Test B: expense-to-income ratios give the same 99.75%, since they are the same quantity divided by income. Excluded.
4. Test C: for each share, Spearman correlation with `Savings_Rate` and the ROC-AUC of the share used alone as a score.

| Share | Spearman vs savings rate | Marginal ROC-AUC | \|AUC − 0.5\| |
| --- | --- | --- | --- |
| `Healthcare_Share` | −0.209 | 0.400 | 0.100 |
| `Utilities_Share` | +0.181 | 0.581 | 0.081 |
| `Transport_Share` | +0.117 | 0.563 | 0.063 |
| `Insurance_Share` | +0.115 | 0.551 | 0.051 |
| `Groceries_Share` | +0.138 | 0.544 | 0.044 |
| `Clothing_Footwear_Share` | +0.097 | 0.536 | 0.036 |
| `Education_Share` | −0.048 | 0.468 | 0.032 |
| `Eating_Out_Share` | +0.064 | 0.525 | 0.025 |
| `Entertainment_Share` | +0.037 | 0.516 | 0.016 |
| `Miscellaneous_Share` | −0.021 | 0.492 | 0.008 |
| `Rent_Share` | +0.008 | 0.499 | 0.001 |

No share on its own is a strong predictor. A naive reading would stop here and call the shares safe. The next two cells test them jointly with household size and income, which is where they do leak.

> **In plain terms: ROC-AUC.** Take one household that met the goal and one that did not, at random. ROC-AUC is the probability that the score ranks the first above the second. 0.5 is a coin flip, 1.0 is perfect ranking. A score below 0.5 ranks backwards, which is why the table reports the distance from 0.5: healthcare at 0.400 is as informative as a score of 0.600 used the other way round.

### Cell 12: the food–size fit and the size × food-share oracle (Q5, part 2)

The concern, written out in the docstring of `savings_goal.evaluation.leakage`: if food spending is roughly proportional to household size, then `Groceries ≈ k · Household_Size`, and because `Groceries_Share = Groceries / total spend`, total spend ≈ `k · Household_Size / Groceries_Share`. Add `INCOME` and you have an approximate savings rate, the quantity the share representation was meant to hide.

`savings_goal.evaluation.leakage.food_size_fit` regresses log food spend on log household size for the 41,494 households with positive food spend: R² 0.292, elasticity 0.630. Food spend rises with size, but less than proportionally (doubling the household raises food spend by about 55%, not 100%).

> **In plain terms: R² and elasticity.** R² is the fraction of the variation in one quantity that a fitted line explains: 0.29 means household size accounts for under a third of the spread in food spending. An elasticity of 0.63 on a log-log fit means a 1% larger household spends about 0.63% more on food.

`savings_goal.evaluation.leakage.size_share_oracle` then scores each household by the implied savings rate

`implied_sr = 1 − (k · Household_Size / Groceries_Share) / INCOME`

with `k` = median food spend per person = Rs 8,848 per year. The 24 households with a zero or missing food share (0.058%) are excluded because the formula divides by the share; 41,494 are scored. The result is ROC-AUC 0.895 with no fitting, no cross-validation and three columns plus one median. Thresholding `implied_sr` at 0.20 agrees with `Goal_Met` for 82.2% of households.

### Cell 13: the ablation (Q5, part 3)

`savings_goal.evaluation.leakage.ablation_sets` builds four feature sets and `savings_goal.evaluation.leakage.ablation` scores each with the untuned default XGBoost from `savings_goal.models.pipeline.xgb_pipeline` (400 trees, depth 6, learning rate 0.08) under PSU-grouped 5-fold cross-validation on all households.

| Feature set | Features | ROC-AUC (sd) | Macro-F1 | At-risk PR-AUC | ROC-AUC retained |
| --- | --- | --- | --- | --- | --- |
| All features | 29 | 0.931 (0.003) | 0.837 | 0.966 | 100% |
| No `Groceries_Share` | 28 | 0.929 (0.003) | 0.835 | 0.965 | 99.8% |
| No size family (`Household_Size`, `Age_Dependents`, `Dependency_Ratio`) | 26 | 0.921 (0.003) | 0.824 | 0.961 | 98.9% |
| No shares or indicators | 13 | 0.878 (0.007) | 0.775 | 0.940 | 94.3% |

Removing the food share alone costs almost nothing because the other shares and size columns carry the same route. Removing the size family costs 0.010. Removing every share and participation indicator costs 0.053 ROC-AUC and leaves a model built only from income, demographics, debt and the four categoricals.

> **In plain terms: grouped cross-validation.** Cross-validation splits the data into five parts, trains on four and scores on the fifth, five times over, and averages. "Grouped" means every household from one PSU lands in the same part, so the model is never scored on the neighbours of households it trained on. Splitting households at random would let village-level similarities inflate the score.

> **In plain terms: macro-F1 and PR-AUC.** F1 balances how many flagged households were right (precision) against how many of the true cases were found (recall). Macro-F1 averages F1 over the two classes, so the smaller "met" class counts equally. PR-AUC summarises precision against recall across all thresholds for one class; here it is computed for the at-risk class (68.1% of households), so its floor is about 0.68 rather than 0.5.

### Cell 14: the decision and `results/leakage.json`

The notebook applies a fixed rule: reframe the project if the oracle reaches ROC-AUC 0.85 or the share-free model keeps at least 90% of the full model's ROC-AUC. Both hold (0.895 and 94.3%).

The headline model is therefore the 13-feature "deployable" set: `Log_Income`, `Household_Size`, `Age_Dependents`, `Dependency_Ratio`, `Head_Age`, `Max_Adult_Education`, `Debt_To_Income`, `Has_Debt`, `Debt_Missing`, and the four categoricals. These are all known when a household is first onboarded. The 29-feature "full" set with spending shares stays in the project as a diagnostic: with size and income the shares approximately rebuild total spend, so their extra 0.053 ROC-AUC is largely the label restated. All test results, the reconciliation, the food–size fit, the oracle, the ablation table and the decision are written to `results/leakage.json` with `savings_goal.io.write_result`.

### Cell 16: class balance and threshold sensitivity (Q6)

| `Goal_Met` | Households | % |
| --- | --- | --- |
| 0 (at risk) | 28,262 | 68.07 |
| 1 (on track) | 13,256 | 31.93 |

Imbalance 2.13 : 1. With 13,256 households in the smaller class, resampling is not needed; the metrics above (ROC-AUC, macro-F1, at-risk PR-AUC) work as they are. Survey-weighted, 30.47% meet the goal, so the sample slightly over-represents savers.

> **In plain terms: survey weights.** IHDS over-samples some groups so they have enough interviews. `WT` says how many households in India each sampled household stands for. Unweighted numbers describe the sample; weighted numbers estimate the national population.

The 20% benchmark is a convention, so the cell recomputes the label at other thresholds:

| Threshold | Positive rate | Weighted | Agreement with 20% label |
| --- | --- | --- | --- |
| 0% | 0.441 | 0.423 | 0.878 |
| 5% | 0.414 | 0.397 | 0.905 |
| 10% | 0.383 | 0.367 | 0.936 |
| 15% | 0.351 | 0.335 | 0.969 |
| 20% | 0.319 | 0.305 | 1.000 |
| 25% | 0.287 | 0.274 | 0.968 |
| 30% | 0.253 | 0.240 | 0.934 |
| 40% | 0.188 | 0.176 | 0.868 |

A result that holds from 10% to 30% describes households; one that appears only at 20% describes the threshold.

By area type:

| Area | Households | Rate | Weighted |
| --- | --- | --- | --- |
| Metro urban | 3,063 | 0.421 | 0.432 |
| Other urban | 11,410 | 0.370 | 0.361 |
| Developed village | 12,635 | 0.309 | 0.287 |
| Less-developed village | 14,410 | 0.267 | 0.257 |

The gradient is monotone in both columns. It is unconditional: metro households also earn more, so this table cannot separate place from income. Phase 5 looks at area with income held constant.

As a check that the normative label tracks real behaviour, the cell also records fixed-deposit ownership by class: 15.2% of on-track households hold one against 8.2% of at-risk households. These figures, the class balance, the sensitivity table and the area breakdown go to `results/phase1_eda.json`.

### Cell 18: summary figure

Writes `results/phase1_eda.png`, four panels:

- Income: histogram of log annual income. Raw income has skew 15.8; after the log the distribution is close to symmetric.
- Savings rate: histogram clipped to [−2, 1] with the 20% benchmark marked. Median −10.8%; 44% of households save a positive amount.
- Zero spend: share of households with zero spend in each category, sorted. Four categories (rent, insurance, eating out, entertainment) are zero for more than half of households; education is zero for 36%.
- Area type: unweighted and weighted attainment rates for the four area types, with the overall 31.9% and weighted 30.5% in the title.

> **In plain terms: zero-inflation.** A column is zero-inflated when a large block of rows are exactly zero rather than small. Ninety per cent of households pay no rent because they own their home; that is a different state from paying a little. Phase 2 adds a yes/no "spends on X" column for each category that is zero for more than 30% of households.

---

## What this means for later phases

| Phase | Consequence |
| --- | --- |
| 2: Feature engineering | Raw rupees and expense/income ratios are leakage and never features. Decide the handling of `Debt_Missing`, missing caste, zero-inflated shares and the log-ratio transform on the training split only. |
| 3: Baseline | Majority-class macro-F1 is 0.405 at a 68.1% at-risk rate. The single income threshold is the baseline that matters. |
| 4: Model comparison | Tune and report the 13-feature deployable set as the headline; report the 29-feature full set alongside it as a diagnostic, never as the result. All validation is grouped by `IDPSU`. |
| 5: Explainability | Read share effects relative to the other shares (closure). Check whether the area gradient survives once income is held constant. |
| 6: Clustering | Euclidean distance on raw shares is not meaningful; use a log-ratio basis, with care for the zero-inflated categories. |
| 7 and 8: Business translation and reporting | Lean on relative comparisons, since 55.9% of households report spending above income. Apply `WT` to any population-level figure. |
