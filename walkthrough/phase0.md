# Phase 0: Framing

**Source:** [README § Phase 0: Framing](../README.md#phase-0-framing)
**Notebook:** none. Phase 0 is answered in prose; the numbers it cites come from later phases
**Builds on:** [Dataset construction](dataset_construction.md)
**Artifacts:** none of its own. Cited figures come from `results/phase1_eda.json`, `results/leakage.json`, `results/baseline.json`, `results/model_final.json`, `results/model_test.json`, `results/business.json` and `results/savings_rate_regression.json`

Phase 0 fixes what is predicted and why, before any modelling. One constraint shapes the rest: a household survey does not record what its respondents intend to save. IHDS-II measures income and consumption, not aspirations, so the target is an external benchmark rather than a personal goal. This document sets that benchmark, says what it can and cannot support, and lists the constraints later phases inherit. Where results from later phases bear on the framing, they are cited.

> **In plain terms: normative vs customer-relative.** A **normative** benchmark is set from outside and applied to everyone the same way: save at least 20%, whoever you are. A **customer-relative** benchmark would compare each household with its own stated intention: are you on track for the goal you set? A savings app would want the second, but no survey asks households what they intend to save, so only the first can be built here. Most caveats in this document follow from that substitution.

---

## Research questions & answers

| # | Question | Answer |
| --- | --- | --- |
| 1 | What business decision does this project inform? | Whether a savings-product or financial-inclusion team should treat a household as on track or at risk against a normative 20% savings benchmark, and so which households to contact first. It is a triage decision, not a savings forecast. At a 25% contact budget the headline model reaches 98.5% of the best possible at-risk capture, against 95.4% for a poorest-first income rule. |
| 2 | What is the precise definition of the target? | `Goal_Met = 1` if `(INCOME − COTOTAL) / INCOME >= 0.20`, else 0, both in annual rupees from IHDS-II. 31.93% of 41,518 households meet it (30.47% survey-weighted). |
| 3 | Classification or regression? | Binary classification is the primary framing, because the decision is binary and the continuous savings rate has an extreme left tail (minimum −1,647, standard deviation 13.08). Notebook 08 adds a median (quantile) regression on the savings rate: ranked by predicted rupee shortfall, it reaches 45.5% of the total rupee gap at a 25% budget, against 37.0% for the classifier and 28.7% for the income rule. |

---

## Q1. What business decision does this project inform?

The decision is whether a savings-product team, financial-inclusion programme or microfinance lender should treat a household as on track or at risk against a normative savings benchmark, and therefore which households to prioritise for a low-cost intervention such as a savings nudge, a commitment-savings product or a budgeting tool.

The benchmark is normative, and that changes who gets flagged. Under a customer-relative rule, a household that wanted to save 5% and did would be on track. Under this one, the same household is at risk. The model therefore measures something closer to savings capacity than goal alignment, and later phases show it:

- A single income threshold, re-learned in each fold (mean Rs 122,249), reaches macro-F1 0.742 ([Phase 3](phase3.md)).
- The headline model reaches CV macro-F1 0.782 and held-out macro-F1 0.781 ([Phase 4](phase4.md)), a gain of about 0.04 over the income rule.
- Income accounts for 58.0% of the headline model's grouped SHAP attribution and 82% of its permutation importance ([Phase 5](phase5.md)).

Most of what the classifier does is separate households that are not poor from households that are.

> **In plain terms: macro-F1.** Two ideas about a yes/no prediction:
> - Precision: of the households flagged, what fraction really were what we said?
> - Recall: of the households that really were, what fraction did we flag?
>
> Flag everybody and recall is perfect while precision is poor. F1 combines the two into one number that stays low unless both are decent. Macro-F1 computes F1 separately for on-track and at-risk households and averages the two, so each class counts equally however many households it has.
>
> That matters here because 68% of households are at risk. A model that always says "at risk" is right 68% of the time (accuracy 0.681) and is useless. Its macro-F1 is 0.405, because it scores zero on the class it never predicts. [Phase 3](phase3.md) shows this on the baselines.

> **In plain terms: `Log_Income`.** Annual income with a logarithm applied. It pulls in the long tail of very high incomes so that going from Rs 20,000 to Rs 40,000 counts the same as going from Rs 200,000 to Rs 400,000.

### Why the decision is still worth supporting

1. Finding households with structurally inadequate savings is the question a financial-inclusion programme asks, even when income is most of the answer. The model adds to income: at a 25% contact budget it reaches 98.5% of the at-risk capture ceiling against 95.4% for the poorest-first rule, a gain of 1.1 percentage points (95% CI 1.0 to 1.3) ([Phase 7](phase7.md)). The gap is small because the income rule is already close to the ceiling.
2. Household size, debt, education and location carry information beyond income. Interactions account for 31.6% of the headline model's SHAP attribution, the largest being income × household size ([Phase 5](phase5.md)): the same income means something different for a household of two and a household of eight.

> **In plain terms: "beyond income".** A feature carries independent signal if it still helps once the model already knows income. The test is incremental: score a model on income alone, add the feature, and see whether the score moves. Many features look predictive on their own only because rich and poor households differ in them; those add nothing once income is in.

What the project cannot claim is that it identifies households failing at a goal they set. Nothing in IHDS-II records intentions, so customer-goal language in any write-up would misdescribe the target.

### The constraint this places on later phases

55.9% of IHDS households report consumption above income (57.7% weighted), a known effect of income being under-reported relative to item-by-item consumption ([Phase 1](phase1.md)). `Goal_Met` is therefore biased downward at every threshold. The project supports relative prioritisation ("which households first") and not absolute prevalence ("X% of Indian households save too little").

---

## Q2. What is the precise definition of the target?

```text
Savings      = INCOME − COTOTAL                   (annual rupees)
Savings_Rate = Savings / INCOME
Goal_Met     = 1 if Savings_Rate >= 0.20 else 0
```

`Goal_Met = 1` when a household keeps at least 20% of its annual income after all recorded consumption.

The 20% is a convention, roughly the rate implied by common personal-finance guidance and close to India's household savings rate in the survey period. It is the default of `uv run sgc build --threshold` and can be changed. [Phase 1](phase1.md) publishes the sensitivity curve:

| Threshold | Goal_Met rate | Survey-weighted | Imbalance (at risk : on track) |
| --- | --- | --- | --- |
| 0% | 0.4411 | 0.4231 | 1.27 : 1 |
| 10% | 0.3833 | 0.3669 | 1.61 : 1 |
| 20% | 0.3193 | 0.3047 | 2.13 : 1 |
| 30% | 0.2532 | 0.2396 | 2.95 : 1 |

> **In plain terms: imbalance.** The imbalance ratio counts how many households are in the larger class for each one in the smaller. At 20%, there are 2.13 at-risk households for every on-track one. Ratios like 100 : 1 (fraud, rare disease) need resampling and special metrics; 2.13 : 1 does not. The ratio worsens quickly as the threshold rises, which is one reason not to lean on a single threshold.

A finding that holds only at 20% is a finding about the threshold, not about households.

Two consequences of the definition:

- The unit is a household. Per-person statements would need `INCOMEPC` or `COPC` and are not what the model predicts.
- `Savings` is an exact identity over `INCOME` and `COTOTAL`, and an approximate one over the 11 rebuilt expense categories, which match `COTOTAL` within 1% for 97.72% of households ([Dataset construction](dataset_construction.md)). Anything from which spending relative to income can be computed is the answer in disguise.

  > **In plain terms: identity.** An identity is an equation true by definition: savings is income minus spending. Nothing is estimated. A model given the pieces of an identity reproduces it and learns nothing about households.

### The leakage constraint

Never usable as features: the 11 raw rupee categories, `Category_Total`, `COTOTAL`, `Savings`, `Savings_Rate`. Raw categories with income, and expense-to-income ratios, each reproduce `Goal_Met` for 99.75% of households.

The constraint reaches further than those columns. Spending shares (each category over total spending) pass every one-at-a-time test, but household size predicts food rupees well enough that size, food share and income together back out total spend. The size × food-share oracle, a formula with no model, ranks households with ROC-AUC 0.895 ([Phase 1](phase1.md)). The headline model therefore uses income, demographics and debt only (13 features, the deployable set). The 29-feature full set with spending shares reaches ROC-AUC 0.929 in CV and 0.932 held out, and is reported as a diagnostic, because part of its extra skill is a restatement of the label.

The six `Has_*` columns (bank savings, fixed deposit, pension/LIC, securities, post office account, gold) record which savings instruments a household holds. They are outside the consumption arithmetic, so they are not leakage, and they are not features either: they are kept back to check the normative target against reported saving behaviour.

---

## Q3. Is the primary task classification or regression?

Binary classification is the primary framing. A median regression on the savings rate complements it in [Phase 8](phase8.md).

> **In plain terms: classification vs regression.**
> - Classification predicts which bucket something falls in. Binary classification has two buckets, here on track and at risk, usually with a probability attached.
> - Regression predicts a number on a sliding scale, here the savings rate itself: −0.34, +0.07, +0.41.
>
> The choice follows from the decision the output supports. Here the label was made from a number by cutting it at 20%, so regression is available and the case for classification has to be argued.

### The case against classification

`Savings_Rate` is a continuous outcome and the 20% cut is imposed. Thresholding discards information: a household at 19% and one at −150% are both "not met", though their situations differ enormously. The target is not binary by nature.

### Why classification is still the primary framing

1. The decision is binary. A team either includes a household in an outreach campaign or does not. A predicted savings rate would be thresholded anyway; doing it inside the model keeps the threshold explicit. [Phase 4](phase4.md) tunes the operating threshold on the at-risk score: at 0.425 the held-out at-risk recall is 0.913 with precision 0.826.
2. The continuous target is badly behaved. `Savings_Rate` has a median of −0.108, a minimum of −1,647 and a standard deviation of 13.08. A least-squares regression would be pulled toward a long left tail of households with tiny reported incomes, the least reliable part of the data. The binary label treats a household at −150% and one at −15% the same way, as at risk, and does not ask the model to separate two numbers that are mostly measurement error.

   > **In plain terms: why −1,647 breaks an ordinary regression.** A savings rate of −1,647 means a household reported consuming 1,648 times its stated income, almost certainly because income was recorded far too low. Ordinary regression minimises squared error, so being wrong by 1,000 costs a million times more than being wrong by 1, and one absurd row can drag the fitted line toward itself. Classification sidesteps it: −1,647 and −0.15 are both "no".

3. A binary flag is easy to inspect. A stakeholder can read the confusion matrix, the operating threshold and the cost of a miss.

   > **In plain terms: confusion matrix and operating threshold.** A confusion matrix is a 2×2 tally of predictions against reality: correctly flagged, wrongly flagged, correctly cleared, wrongly cleared. It shows the two kinds of mistake separately. The operating threshold is the cut-off applied to the model's probability to turn it into an action; moving it trades one mistake for the other.

### What classification gives up, and how Phase 8 recovers it

A yes/no flag cannot say how far a household is from adequacy, and for budgeting outreach the size of the gap matters. [Phase 8](phase8.md) (notebook 08) fits a median regression of `Savings_Rate` on the 13 deployable features, using gradient boosting with a quantile loss (`savings_goal.models.regression.quantile_model`), out-of-fold under the same PSU-grouped folds. It then ranks households by predicted rupee shortfall, `max(0, 0.20 − predicted rate) × INCOME` (`savings_goal.models.regression.rupee_shortfall`).

> **In plain terms: quantile regression.** Ordinary regression predicts the average outcome for households like this one. A median (quantile) regression predicts the middle outcome instead: half of similar households save more, half save less. Because the median ignores how extreme the extremes are, a household at −1,647 pulls on it no harder than one at −2. Its error measure is the pinball loss, which for the median is the average absolute miss.

| Measure (out-of-fold) | Value |
| --- | --- |
| Pinball loss, median model | 0.474 |
| Pinball loss, constant median | 0.739 |
| Spearman correlation, predicted vs actual savings rate | 0.779 |

Share of the total rupee gap below the 20% benchmark reached by each ranking:

| Contact budget | Predicted rupee shortfall | Classifier at-risk score | Income rule (poorest first) |
| --- | --- | --- | --- |
| 10% | 23.7% | 17.0% | 12.4% |
| 25% | 45.5% | 37.0% | 28.7% |
| 50% | 69.6% | 64.6% | 54.7% |

The classifier still captures slightly more at-risk households by count (36.2% vs 35.0% of at-risk households at a 25% budget), so the two answer different questions. The classifier is the better tool for "who is at risk"; the shortfall ranking is the better tool for "where is the largest rupee gap". Classification stays primary because the decision is binary, and the regression is the companion view for sizing the gap.

---

## What this means for later phases

| Commitment | Consequence |
| --- | --- |
| Normative target, not self-set goals | No "their own savings goal" language anywhere. |
| Household grain | No per-individual claims. |
| Relative prioritisation only | No absolute prevalence claims, because 55.9% of households report consumption above income. |
| Leakage list | Raw categories, `Category_Total`, `COTOTAL`, `Savings`, `Savings_Rate` are never features; spending shares are diagnostic only. |
| `Has_*` reserved | External validation only, never a feature. |
| Binary classification, macro-F1 | Accuracy is gameable at 0.681 by always predicting "at risk". |
| At-risk class (`Goal_Met = 0`) is the majority | Lift over random contact is capped; see below. |
| 20% is a convention | The threshold sensitivity is published in Phase 1. |
| Distance matters for budgeting | Phase 8 adds a median regression and a rupee-shortfall ranking. |

> **In plain terms: why a majority target caps lift.** Targeting models usually hunt for something rare, such as fraud or churn, and a good one concentrates the rare cases at the top of its ranking. **Lift** is the factor by which contacting model-selected households beats contacting households at random. Here the households to reach are 68% of everyone, so a random 25% of the population is already 67.8% at risk, and no ranking can do better than 100%, a lift of about 1.47. The headline model reaches 98.5% precision at that budget, close to the cap. [Phase 7](phase7.md) works through it.
