# Phase 0 (IHDS-II) — Framing

**Source:** [README § Phase 0 — Framing](../README.md#phase-0--framing)
**Notebook:** none — Phase 0 is a framing exercise, answered in prose.
**Builds on:** [Migration — Synthetic to IHDS-II](dataset_construction.md)
**Replaces:** [`phase0.md`](phase0.md)

Phase 0 is normally written before touching the data. This one is not, and that has to be said plainly: it is a **re-framing after a dataset change**, written with Phases 1–4 already complete. That order carries a real risk — it is easy to retrofit a business question to the results you happen to have got. Where the evidence has constrained the framing, this document says so explicitly rather than presenting the outcome as foresight.

The re-framing is forced by one fact: **no real household survey collects a self-declared savings goal.** The synthetic dataset's `Desired_Savings` column is the kind of variable a generator invents and a survey does not ask, so the original target — "is this person on track for the goal they set themselves" — cannot be reproduced on any real data. Everything below follows from replacing it.

---

## Research questions & answers

| # | Question | Answer |
| --- | --- | --- |
| 1 | What real business decision does this project inform? | Whether a savings-product or financial-inclusion team should treat a **household** as on-track or at-risk against a **normative** savings adequacy benchmark, and therefore which households to prioritise for outreach. The decision is a triage/prioritisation call, not a savings forecast. |
| 2 | What is the precise definition of the target variable? | `Goal_Met = 1` if `(INCOME − COTOTAL) / INCOME >= 0.20`, else `0`, where both are annual rupees from IHDS-II. In words: a household is on track if it retains at least 20% of its annual income after all recorded consumption. |
| 3 | Is the primary task classification or regression, and why? | **Binary classification**, but the justification is weaker than in the synthetic project and is stated honestly below. The target is now a *thresholded continuous quantity*, so regression on `Savings_Rate` is a genuinely defensible alternative that was not available before. |

---

## Q1. What real business decision does this project inform?

**Answer:** whether a savings-product team, financial-inclusion programme, or microfinance lender should treat a household as **on-track** or **at-risk** against a normative savings-adequacy benchmark, and therefore which households to prioritise for a low-cost intervention — a savings nudge, a commitment-savings product, or a budgeting tool.

**What changed from the synthetic framing, and why it matters.** The original decision was *"is this person on track for the goal they set themselves."* That is a **customer-relative** question: it respects each individual's own stated intention, and a household saving 5% of income was "on track" if 5% was all they wanted to save. The new decision is **normative**: the benchmark is imposed from outside and is identical for every household.

This is not a cosmetic difference, and it changes who gets flagged. Under the old framing, a low-income household with modest goals could be on-track; under the new one it almost certainly is not. The model is now measuring something closer to *savings capacity* than *goal alignment* — and Phases 3 and 4 confirmed that empirically. A single threshold on income alone recovers macro-F1 0.7425 (Phase 3), and `Log_Income` carries the largest share of model importance. **A large part of what this classifier does is identify households that are not poor.**

**Why the decision is still worth supporting despite that.** Two honest defences, and one thing this project cannot claim:

1. Identifying households with structurally inadequate savings *is* the actual question for a financial-inclusion programme, even if income is most of the answer. The model adds +0.095 macro-F1 over the income-only rule (Phase 4) — real incremental value in identifying which similar-income households are and are not saving.
2. The spending-mix features carry independent signal on top of income: `Groceries_Share` alone adds +0.041 ROC-AUC *after* income is known (Phase 3), because two households with identical income and very different budget shapes are in materially different positions.

**What it cannot claim:** that it identifies households failing at a goal *they* set. Nothing in IHDS-II supports that, and any write-up that borrows the synthetic project's customer-goal language on top of IHDS results would be misrepresenting the target.

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

Any finding that holds only at 20% is a finding about the threshold, not about households, and Phase 7 must check its headline claims at 10% and 30% before resting on them.

**Two definitional consequences that are easy to miss:**

- **The unit is a household, not an individual.** Every claim is about households; per-person statements require `INCOMEPC`/`COPC` and are not what this model predicts.
- **The target is an exact accounting identity** over the expense columns (Phase 1 verified `Savings = INCOME − COTOTAL` to 5.8×10⁻¹¹). This is the source of the leakage constraint below, and it is why the feature set uses composition shares rather than expense-to-income ratios.

### The leakage constraint (replaces the synthetic project's)

**Never usable as features:** the 11 raw rupee expense categories, `COTOTAL`, `Savings`, `Savings_Rate`.

Phase 1 measured it: raw categories reconstruct `Goal_Met` with 99.75% agreement, and expense-to-income ratios with the identical 99.75% — because they are the same quantity divided by income. Composition shares (each category over *total expenditure*) sum to exactly 1 and so carry no information about consumption relative to income; they are safe, and they are what the pipeline uses.

**A separate reserved set:** the six `Has_*` columns (bank savings, fixed deposit, pension/LIC, securities, post office, gold) are survey-reported behaviour, outside the consumption arithmetic. They are **not features** and **not leakage** — they are held back for external validation, which the synthetic dataset made impossible.

---

## Q3. Is the primary task classification or regression?

**Answer: binary classification — but this is a closer call than it was, and the honest reasoning is different.**

**Why the original argument no longer works.** The synthetic project argued that the target "is already binary by construction" — `Goal_Met` came from comparing two given quantities, and framing it as regression would have meant predicting a continuous value only to re-threshold it. That argument is now void: `Savings_Rate` is a perfectly good continuous outcome, and the 20% cut is something *we* imposed. Thresholding a continuous target discards information, and here that information is real — the difference between a household at 19% and one at −150% is enormous and the binary label erases it.

**The three reasons classification is still the right primary framing:**

1. **The decision is binary.** The team either includes a household in an outreach campaign or does not. A predicted savings rate would have to be thresholded anyway, and doing that inside the model keeps the threshold explicit and auditable — which is exactly what Phase 4 used when it tuned for 95% recall on the at-risk class.
2. **The continuous target is badly behaved.** `Savings_Rate` has a median of −0.108, a minimum of −1647, and a standard deviation of 13.08 (Phase 1). Regression on that would be dominated by a long left tail of households with tiny reported incomes — the least reliable part of the data. The binary label is *robust to exactly the measurement error that most afflicts this survey*: a household at −150% and one at −15% are both, correctly, "not on track", and the model is not penalised for failing to distinguish two numbers that are mostly noise.
3. **Comparability with the synthetic track.** Keeping the task type fixed means the two tracks' differences are attributable to the data, not to a change of problem — which is the entire point of the migration.

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
| At-risk class is `Goal_Met = 0` | The actionable class is the majority one — the inverse of the synthetic project. |
| 20% is a convention | Headline findings must be checked at 10% and 30%. |
