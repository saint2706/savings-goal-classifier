# Phase 0 — Framing

**Source:** [README § Phase 0 — Framing](../README.md#phase-0--framing)
**Notebook:** none — Phase 0 is a framing exercise, answered in prose.
**Builds on:** [Dataset construction](dataset_construction.md)

Phase 0 fixes *what* we are predicting and *why*, before any modelling. One constraint shapes everything that follows: **a household survey does not record what its respondents intend to save.** IHDS-II measures income and consumption, not aspirations, so the target has to be an externally imposed benchmark rather than a personal goal. This document sets that benchmark, states what it can and cannot support, and fixes the constraints later phases inherit.

Where the analysis later constrained the framing, this document says so rather than presenting the outcome as foresight.

> **In plain terms — normative vs customer-relative.** A **normative** benchmark is a standard set from outside and applied identically to everyone: *save at least 20%, whoever you are.* A **customer-relative** benchmark would compare each household against its own stated intention: *are you on track for the goal you personally set?* The second is what a savings app would want and what the project's title suggests — but no survey asks households what they intend to save, so only the first is buildable here. Nearly every caveat in this document flows from that substitution.

---

## Research questions & answers

| # | Question | Answer |
| --- | --- | --- |
| 1 | What real business decision does this project inform? | Whether a savings-product or financial-inclusion team should treat a **household** as on-track or at-risk against a **normative** savings adequacy benchmark, and therefore which households to prioritise for outreach. The decision is a triage/prioritisation call, not a savings forecast. |
| 2 | What is the precise definition of the target variable? | `Goal_Met = 1` if `(INCOME − COTOTAL) / INCOME >= 0.20`, else `0`, where both are annual rupees from IHDS-II. In words: a household is on track if it retains at least 20% of its annual income after all recorded consumption. |
| 3 | Is the primary task classification or regression, and why? | **Binary classification** — but this is a closer call than it looks. The target is a *thresholded continuous quantity*, so a robust regression on `Savings_Rate` is a genuinely defensible alternative. The reasoning is set out honestly below. |

---

## Q1. What real business decision does this project inform?

**Answer:** whether a savings-product team, financial-inclusion programme, or microfinance lender should treat a household as **on-track** or **at-risk** against a normative savings-adequacy benchmark, and therefore which households to prioritise for a low-cost intervention — a savings nudge, a commitment-savings product, or a budgeting tool.

**The benchmark is normative, and that choice has consequences.** A **customer-relative** framing — is this household on track for the goal *it* set — would respect each household's own intention, and a household saving 5% would be "on track" if 5% was all it wanted. IHDS records no such intention, so the benchmark is imposed from outside and is identical for every household.

That changes who gets flagged. A low-income household with modest ambitions is on-track under a customer-relative rule and at-risk under this one. The model is therefore measuring something closer to *savings capacity* than *goal alignment* — and Phases 3 and 4 confirmed that empirically. A single threshold on income alone recovers macro-F1 0.7425 (Phase 3), and `Log_Income` carries the largest share of model importance. **A large part of what this classifier does is identify households that are not poor.**

> **In plain terms — macro-F1, and why it is the headline number.** Start with two ideas about a yes/no prediction:
> - **Precision** — of the households we flagged, what fraction really were what we said? (How much of our shouting was wasted?)
> - **Recall** — of the households that really were, what fraction did we flag? (How many did we miss?)
>
> These trade off: flag everybody and recall is perfect while precision is terrible. **F1** combines them into one number that stays low unless *both* are decent. **Macro-F1** then computes F1 separately for the "on track" households and for the "at risk" households and averages the two, giving each class equal say regardless of how many households are in it.
>
> That last part is the whole point. Because 68% of households are at risk, a model that predicts "at risk" for everyone, always, is right 68% of the time — 0.68 **accuracy**, which sounds respectable and is worth nothing. Macro-F1 refuses to be fooled: that same do-nothing model scores 0.405, because it is hopeless on the class it never predicts. Scores run 0 to 1, higher is better. [Phase 3](phase3.md) demonstrates this on real baselines.
>
> **`Log_Income`** is simply annual income with a logarithm applied — a rescaling that pulls in the long tail of very high incomes so that the step from ₹20,000 to ₹40,000 counts the same as the step from ₹200,000 to ₹400,000. Models handle that shape far better than raw rupees.

**Why the decision is still worth supporting despite that.** Two honest defences, and one thing this project cannot claim:

1. Identifying households with structurally inadequate savings *is* the actual question for a financial-inclusion programme, even if income is most of the answer. The model adds +0.095 macro-F1 over the income-only rule (Phase 4) — real incremental value in identifying which similar-income households are and are not saving.
2. The spending-mix features carry independent signal on top of income: `Groceries_Share` alone adds +0.041 ROC-AUC *after* income is known (Phase 3), because two households with identical income and very different budget shapes are in materially different positions.

> **In plain terms — "independent signal", and why *after* is the word doing the work.** A feature carries **signal** if it helps predict the target. It carries *independent* signal if it still helps **once the model already knows income** — that is, it tells you something income was not already telling you. The test is incremental: measure the model on income alone, add the new feature, and see whether the score moves. Plenty of features look predictive on their own purely because rich households differ from poor ones; those add nothing here and are exposed as redundant. Grocery share survives the test, which is why it earns a place in the story. [Phase 3](phase3.md) runs exactly this progression, and finds features that fail it.

**What it cannot claim:** that it identifies households failing at a goal *they* set. Nothing in IHDS-II records household intentions, so any write-up using customer-goal language would be misrepresenting the target.

**The constraint this places on every later phase.** Phase 1 found that **55.9% of IHDS households report consumption exceeding income**, a documented consequence of income being under-reported relative to item-by-item consumption. So `Goal_Met` is biased downward at every threshold. The decision this project informs must therefore be a **relative prioritisation** ("which households first"), never an absolute claim ("X% of Indian households save inadequately"). Ranking is defensible; levels are not.

---

## Q2. What is the precise, one-sentence definition of the target variable?

```text
Savings      = INCOME − COTOTAL                   (annual rupees)
Savings_Rate = Savings / INCOME
Goal_Met     = 1 if Savings_Rate >= 0.20 else 0
```

**`Goal_Met = 1` when a household retains at least 20% of its annual income after all recorded consumption expenditure.**

**Why 20%.** It is a convention, not a measurement — roughly the savings rate implied by common personal-finance guidance and close to India's household savings rate in the survey period. **It was chosen before the results were seen, and it is configurable** (`--threshold` in `src/build_dataset.py`). Phase 1 published the full sensitivity curve precisely so no reader has to take the choice on trust:

| Threshold | Goal_Met rate | Imbalance |
| --- | --- | --- |
| 0% | 0.4411 | 1.27 : 1 |
| 10% | 0.3833 | 1.61 : 1 |
| **20%** | **0.3193** | **2.13 : 1** |
| 30% | 0.2532 | 2.95 : 1 |

> **In plain terms — imbalance.** The **imbalance ratio** counts how many households fall in the larger class for each one in the smaller. At the 20% threshold, 31.93% of households meet the goal, so there are 2.13 "not met" for every "met" — written 2.13 : 1. This number matters because it decides how much special handling a classification problem needs. Ratios like 100 : 1 (fraud, rare disease) force resampling, reweighting and unusual metrics. **2.13 : 1 is mild**, which is why [Phase 4](phase4.md) can use ordinary methods and ordinary metrics — but notice how quickly the ratio worsens as the threshold rises, which is itself an argument for not over-trusting one threshold.

Any finding that holds only at 20% is a finding about the threshold, not about households, and Phase 7 must check its headline claims at 10% and 30% before resting on them.

**Two definitional consequences that are easy to miss:**

- **The unit is a household, not an individual.** Every claim is about households; per-person statements require `INCOMEPC`/`COPC` and are not what this model predicts.
- **The target is an exact accounting identity** over the expense columns (Phase 1 verified `Savings = INCOME − COTOTAL` to 5.8×10⁻¹¹). This is the source of the leakage constraint below, and it is why the feature set uses composition shares rather than expense-to-income ratios.

  > **In plain terms — "identity", and that tiny number.** An **identity** is an equation that is true by definition rather than by discovery — savings *is* income minus spending, in the same way that a bachelor *is* an unmarried man. Nothing was estimated. **5.8×10⁻¹¹** is scientific notation for 0.000000000058, and it is the largest disagreement found anywhere in 41,518 households: not a real gap but the rounding dust computers leave behind when they store decimals. Quoting it is a way of saying "we checked, and the equation holds perfectly." The problem is that anything the model can compute this equation from is the answer in disguise — hence the exclusion list.

### The leakage constraint

**Never usable as features:** the 11 raw rupee expense categories, `COTOTAL`, `Savings`, `Savings_Rate`.

Phase 1 measured it: raw categories reconstruct `Goal_Met` with 99.75% agreement, and expense-to-income ratios with the identical 99.75% — because they are the same quantity divided by income. Composition shares (each category over *total expenditure*) sum to exactly 1 and so carry no information about consumption relative to income; they are safe, and they are what the pipeline uses.

**A separate reserved set:** the six `Has_*` columns (bank savings, fixed deposit, pension/LIC, securities, post office, gold) are survey-reported behaviour, outside the consumption arithmetic. They are **not features** and **not leakage** — they are held back so the normative target can be checked against real saving behaviour.

---

## Q3. Is the primary task classification or regression?

**Answer: binary classification — but this is a closer call than it was, and the honest reasoning is different.**

> **In plain terms — classification vs regression.** Two ways of framing a prediction problem:
> - **Classification** predicts which bucket something falls in. **Binary** classification means two buckets — here, on track / not on track. The output is a label, usually with a probability attached.
> - **Regression** predicts a number on a sliding scale — here, the savings rate itself: −0.34, +0.07, +0.41.
>
> Neither is "correct" in the abstract; the choice follows from what decision the output has to support. The awkwardness here is that our label was *manufactured* from a number by cutting it at 20%, so regression is genuinely available and the argument has to be made rather than assumed.

**The argument against classification, stated first.** `Savings_Rate` is a perfectly good continuous outcome, and the 20% cut is something *we* imposed rather than something the data hands us. Thresholding discards information, and here that information is real — the difference between a household at 19% and one at −150% is enormous, and the binary label erases it. Any claim that the target "is binary by construction" would be false.

**The three reasons classification is still the right primary framing:**

1. **The decision is binary.** The team either includes a household in an outreach campaign or does not. A predicted savings rate would have to be thresholded anyway, and doing that inside the model keeps the threshold explicit and auditable — which is exactly what Phase 4 used when it tuned for 95% recall on the at-risk class.
2. **The continuous target is badly behaved.** `Savings_Rate` has a median of −0.108, a minimum of −1647, and a standard deviation of 13.08 (Phase 1). Regression on that would be dominated by a long left tail of households with tiny reported incomes — the least reliable part of the data. The binary label is *robust to exactly the measurement error that most afflicts this survey*: a household at −150% and one at −15% are both, correctly, "not on track", and the model is not penalised for failing to distinguish two numbers that are mostly noise.

   > **In plain terms — why a −1647 breaks a regression.** A minimum of −1647 means one household reportedly consumed 1,648 times its stated income (almost certainly because the income was recorded far too low, not because it truly spent that). Regression is normally fitted by minimising **squared** error, so being wrong by 1,000 is punished a million times harder than being wrong by 1 — a single absurd row can therefore pull the whole fitted relationship toward itself. **Robust** methods (a **robust loss**, or fitting only on plausible incomes) exist precisely to blunt that. Classification sidesteps it entirely: −1647 and −0.15 are both simply "no", so the nonsense value carries no more weight than any other row.

3. **A binary flag is auditable.** A stakeholder can inspect the confusion matrix, the operating threshold, and the cost of a miss. A predicted savings rate would have to be converted into the same decision anyway, with the conversion happening somewhere less visible.

   > **In plain terms — confusion matrix and operating threshold.** A **confusion matrix** is a 2×2 tally of predictions against reality: correctly flagged, wrongly flagged, correctly cleared, wrongly cleared. It is the most honest one-glance summary of a classifier, because it shows the two kinds of mistake separately instead of blending them into a single score. The **operating threshold** is the cut-off applied to the model's probability to turn it into an action — flag everyone above 0.50, or above 0.37, or wherever. Moving it trades the two mistakes against each other, and [Phase 4](phase4.md) tunes it deliberately.

**What is given up, stated plainly:** ranking households by *how far* they are from adequacy is more actionable than a yes/no flag, and this framing cannot do it. A regression on `Savings_Rate`, fitted on the subset with plausible income (or with a robust loss), is the clearest follow-up this project leaves on the table. Phase 4's threshold-tuning partially recovers the loss — the predicted probability is a usable ranking even though the label is binary — but that is a workaround, not a substitute.

---

## What Phase 0 fixes for the phases that follow

| Commitment | Consequence |
| --- | --- |
| Normative, not self-referential target | Phase 7 may not use "their own savings goal" language. |
| Household grain | No per-individual claims anywhere. |
| Relative prioritisation only | No absolute prevalence claims, because of the 55.9% under-reporting. |
| Leakage list | Raw categories, `COTOTAL`, `Savings`, `Savings_Rate` are never features. |
| `Has_*` reserved | External validation only — never a feature. |
| Binary classification | Metric is macro-F1 (Phase 3 showed accuracy is gameable at 0.68). |
| At-risk class is `Goal_Met = 0` | The actionable class is the **majority** one (68%), so lift-based reasoning is structurally weak. |

> **In plain terms — why the majority class being the actionable one is awkward.** Targeting models usually hunt for something rare: fraud, churn, a disease. In that setting, a model that concentrates the rare cases into the top slice of its ranking looks spectacular — this is called **lift**, the factor by which contacting model-selected people beats contacting people at random. But lift has a ceiling set by how rare the target is. Here the households we want to reach are 68% of everyone, so contacting a random quarter of the population already reaches roughly a quarter of them, and no model can be more than about 1.5× better. The model is not weak; the arithmetic of a majority target simply denies it the impressive-looking number. [Phase 7](phase7.md) shows this playing out.
| 20% is a convention | Headline findings must be checked at 10% and 30%. |
