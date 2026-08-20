# Dataset Construction

**Build script:** [`src/build_dataset.py`](../src/build_dataset.py)
**Source:** India Human Development Survey-II, 2011-12 — ICPSR study 36151, dataset DS0002 (Household)
**Output:** `dataset/households.csv` (41,518 households × 50 columns)
**Acquisition:** see [`dataset/README.md`](../dataset/README.md)

This document records how the analysis dataset is built from the raw survey file: what IHDS provides, how its 52 consumption items map onto 11 expense categories, how the target is defined, and the structural problem that definition creates.

> **How to read this walkthrough.** Every technical or mathematical term is explained in plain language, in a quoted note like this one, the first time it comes up. This is the first document in the reading order, so most of the shared vocabulary is introduced here and the later phases assume it.
>
> Three words used constantly from here on:
>
> - **Feature** — an input column the model is allowed to look at (income, household size, share of budget spent on food).
> - **Target** (or **label**) — the single column the model is trying to predict. Here it is `Goal_Met`, which is 1 or 0 for every household.
> - **Household** — one row of the dataset, and the unit every claim is about. Never one person.

---

## Why IHDS-II

This project's entire feature set is built from **income and category-level expenditure measured on the same household**. That requirement rules out most household microdata: India's NSS-lineage surveys, including the current HCES rounds, deliberately collect consumption *without* income.

IHDS-II is the strongest freely available survey that measures both:

| | |
| --- | --- |
| Households | 42,152, nationally representative |
| Reference period | 2011-12 |
| Income | `INCOME` (total, annual rupees) plus eight components `INCFARM`…`INCOTHER`, from queries across 50+ income sources |
| Expenditure | 52 consumption items, `CO1X`–`CO52` |
| Cost | Free; ICPSR registration only |
| Formats | Stata, SPSS, SAS, R, TSV |

The main alternative, if institutional access exists, is **CMIE's Consumer Pyramids (CPHS)** — 174,000+ households, monthly panel since 2014, 150+ expense heads — which is a better dataset in every respect except that it requires a paid subscription.

---

## Unit handling — the check to run before anything else

IHDS uses **two different recall windows**, and mixing them silently inflates food by 12× relative to durables:

- `CO1X`–`CO33` — 30-day recall (questionnaire blocks HQ23/HQ24)
- `CO34`–`CO52` — annual recall (block HQ25)

The build script annualises with `12 × monthly + annual` and validates that formula against IHDS's own `COTOTAL`:

| | |
| --- | --- |
| Correlation with `COTOTAL` | 0.9931 |
| Median difference | 0.00 |
| 25th / 75th percentile difference | 0.00 / 0.00 |
| Within 1% of `COTOTAL` | 97.7% |

> **In plain terms — correlation, median, percentiles.**
> - **Correlation** (written `r`) is a single number between −1 and +1 saying how tightly two columns move together. +1 means they rise in perfect lockstep, 0 means knowing one tells you nothing about the other, −1 means one rises exactly as the other falls. **0.9931 is very nearly perfect agreement** between our rebuilt total and IHDS's own.
> - The **median** is the middle value: line all 41,518 households up in order and read off the one in the middle. It is used instead of the mean (the average) throughout this project because a handful of enormous values drag a mean around and leave the median untouched.
> - A **percentile** is the same idea at another position in the queue: the 25th percentile is the value a quarter of the way up, the 75th is three quarters of the way up. Median difference 0.00 with both quartiles also 0.00 means the typical household matches exactly, and so do the households on either side of typical.

`INCOME` (mean ₹127,760) and `COTOTAL` (mean ₹118,055) are both annual, so they are directly comparable without rescaling. The **mean** is the ordinary average — add everything up, divide by the count.

**Why this check matters more than it looks:** the ICPSR codebook is the only reliable source for which items sit in which block. The IHDS project website's summary of the consumption module lists the items in a different order from the actual variable labels — following it would have put house rent at `CO27` (really soap and detergents) and eating out at `CO18` (really paan/tobacco/intoxicants). Every mapping below was taken from `36151-0002-Codebook.pdf`, not from the summary.

---

## Category mapping

The 11 expense categories, rebuilt from IHDS items. All amounts are annual rupees.

| Category | IHDS items | Notes |
| --- | --- | --- |
| `Groceries` | `CO1X`–`CO3X`, `CO5X`–`CO14X`, `CO15`–`CO17`, `CO19` | Food staples plus salt/spices, tea/coffee, processed foods, fruits/nuts |
| `Eating_Out` | `CO20` | Restaurants/eating out |
| `Utilities` | `CO4X`, `CO21`, `CO22`, `CO24` | Kerosene + household fuel + electricity + telephone/cable/internet |
| `Rent` | `CO30` | House rent/society charges |
| `Transport` | `CO28`, `CO29`, `CO45` | Conveyance + petrol/maintenance + vehicle purchase |
| `Healthcare` | `CO33`, `CO34`, `CO46` | Out-patient + in-patient + therapeutic appliances |
| `Education` | `CO35`–`CO37` | Fees + private tuition + school books |
| `Entertainment` | `CO23`, `CO43`, `CO51` | Entertainment + recreation goods + vacations |
| `Insurance` | `CO50` | Insurance premiums |
| `Clothing_Footwear` | `CO38`, `CO39` | Large enough that folding it into Miscellaneous would distort the mix |
| `Miscellaneous` | `CO18`, `CO25`–`CO27`, `CO31`, `CO32`, `CO40`–`CO42`, `CO44`, `CO47`–`CO49`, `CO52` | Paan/tobacco, toiletries, soap, taxes/fees, services, furniture, crockery, appliances, jewellery, personal care, repairs, social functions |

**Two mappings that are easy to get wrong:**

- **`CO4X` is kerosene**, and it sits inside the `CO*X` food block because IHDS collects it on the same quantity-and-price schedule as staples. It is a fuel and belongs in `Utilities` — leaving it in `Groceries` would overstate food and understate energy for exactly the poorest households that use kerosene most.
- **`CO18` is paan/tobacco/intoxicants**, not eating out. It is routed to `Miscellaneous` rather than to a discretionary category, because it is closer to habitual than elective spending.

### What the survey does not record

**There is no loan-repayment flow.** The debt module (`DB1`–`DB9I`) records debt as a **stock**, not a monthly obligation: outstanding household debt (`DB5`), largest loan amount, purpose, source, and interest rates — but no instalment.

The build script substitutes `Debt_To_Income` (`DB5 / INCOME`), which is the nearest available construct but is not the same thing — a stock of debt is not a claim on this month's cash flow. Any interpretation of that feature has to carry the caveat.

---

## The target

**A household survey does not record what its respondents intend to save.** IHDS measures income and consumption, not aspirations, so savings adequacy has to be defined against an externally imposed benchmark:

```text
Savings      = INCOME - COTOTAL
Savings_Rate = Savings / INCOME
Goal_Met     = 1 if Savings_Rate >= 0.20 else 0
```

> **In plain terms — the three lines above.** Take what the household earned in a year and subtract everything it spent; what's left is **savings** (a negative number if it spent more than it earned). Divide that by income to get the **savings rate**, a proportion rather than a rupee amount, so a rich and a poor household can be compared on the same scale — 0.20 means "kept a fifth of what came in". Then **`Goal_Met`** just answers yes-or-no: did the savings rate reach 0.20? A cut like this that turns a sliding scale into a yes/no is called a **threshold**, and where it is placed is a choice, not a fact about households.

The threshold is configurable via `--threshold`; 20% is the default and is a **convention, not a measurement**. [`phase1.md`](phase1.md) publishes the full sensitivity curve — a **sensitivity curve** simply re-runs the calculation at several different thresholds so a reader can see how much the answer depends on the one that was picked.

### Sample construction

| Step | Households |
| --- | --- |
| Raw DS0002 | 42,152 |
| Dropped: `INCOME <= 0` or `COTOTAL` missing/zero | 634 (1.5%) |
| **Final** | **41,518** |

Income of zero or below makes a savings *rate* undefined. IHDS reports negative farm income for about 9% of households, which drives roughly 1.5% of *total* income to zero or below — a documented property of the survey, not a data error.

### What the target looks like

| Threshold | Goal_Met rate |
| --- | --- |
| ≥ 0% | 0.4411 |
| ≥ 5% | 0.4138 |
| ≥ 10% | 0.3833 |
| **≥ 20%** | **0.3193** |
| ≥ 30% | 0.2532 |

**The median household savings rate is −10.8%**, and only 44% of households have positive savings at all. This is a well-known feature of IHDS rather than a defect: household income is under-reported relative to consumption in most Indian surveys. It is stated in the limitations rather than corrected away.

---

## The leakage constraint

> **In plain terms — leakage.** **Leakage** is when a feature secretly contains the answer. If you let the model see a column from which the label can be recomputed by arithmetic, the model will score brilliantly and will have learned nothing: it is copying, not predicting. On real, unlabelled households it would be useless, because the copied column would not be available or would not mean the same thing. Detecting and excluding leaky columns is therefore not a nicety — it is the difference between a real result and a fake one.

A fixed benchmark makes the target an **exact arithmetic function of the expense-to-income ratios**:

```text
Goal_Met = 1  ⟺  Savings_Rate >= 0.20  ⟺  COTOTAL/INCOME <= 0.80
                                       ⟺  sum(expense ratios) <= 0.80
```

> **In plain terms.** The `⟺` symbol means "these statements are the same statement written differently" — each line is a rearrangement of the one above, the way `a − b ≥ 0` is a rearrangement of `a ≥ b`. So "the household met the goal" and "its expense-to-income ratios add up to no more than 0.80" are not two related facts; they are one fact in two costumes. Anyone holding the ratios already holds the answer.

Measured on the built dataset, `sum(expense/income) <= 0.80` reproduces `Goal_Met` with **99.75% agreement** — i.e. that rule alone gets the right yes/no for 99.75 of every 100 households, without any model. Feeding the ratio features into a model would be reconstructing the label, not predicting it.

**The consequence is that expense-to-income ratios — the intuitive representation — are unusable here.** This is a property of any normative savings threshold, not of IHDS: whenever the benchmark is a constant, knowing what fraction of income each category consumes is knowing the answer.

**Resolution — composition shares instead of income ratios:**

```text
Groceries_Share = Groceries / COTOTAL      (not / INCOME)
```

> **In plain terms — why swapping the denominator fixes it.** Dividing food spending by *income* answers "how much of what they earned went on food" — which is part of the answer we are trying to predict. Dividing by *total spending* instead answers "of the money they spent, what fraction went on food" — the **shape** of the budget, with the size of the budget divided out. Two households, one spending ₹50,000 and one ₹500,000, can both have a 40% food share. Because the shape says nothing about whether the budget was large or small next to income, it cannot give the answer away.

The eleven shares sum to exactly 1.0 for every household (verified: min = mean = max = 1.0), so they describe only the *shape* of spending and carry no information about spending relative to income. They therefore cannot reconstruct `COTOTAL/INCOME`.

> **In plain terms — "sum to exactly 1".** The eleven shares are eleven slices of one pie, so they must add up to the whole pie, every time. That is a guarantee, not a measurement — and the check that `min = mean = max = 1.0` confirms the code delivers it for all 41,518 households rather than on average. This property is convenient here and becomes a genuine nuisance later: [Phase 1](phase1.md) shows it forces the shares to move against each other, and [Phase 2](phase2.md) shows it breaks certain models outright.

### Feature set

| Group | Columns |
| --- | --- |
| Spending mix | 11 `*_Share` columns |
| Income | `INCOME`, `Log_Income` |
| Household | `Household_Size`, `Dependents`, `Head_Age`, `Max_Adult_Education` |
| Categorical | `Occupation` (derived), `Area_Type`, `Caste_Group`, `Religion` |
| Debt | `Debt_To_Income` |

**Leakage-excluded** (present in the CSV for EDA, never as features): the 11 raw rupee category columns, `COTOTAL`, `Savings`, `Savings_Rate`. The raw columns must stay out for the same reason the ratios do — they sum to `COTOTAL`, which with `INCOME` recovers the target.

**Derived columns worth noting:**

- `Occupation` — IHDS has no household occupation variable. It is derived as the worker type contributing the most workers (≥240 hrs/yr) from `NWKSALARY`, `NWKBUSINESS`, `NWKNONAG`, `NWKFARM`, `NWKAGLAB`, with that list as the tie-break order and `No_Regular_Worker` when all are zero.
- `Area_Type` — from `URBAN4_2011`: `Metro_Urban` / `Other_Urban` / `Developed_Village` / `Less_Developed_Village`.
- `Head_Age` — `MHEADAGE`, falling back to `FHEADAGE` for female-headed households.

### Reserved for validation, never features

IHDS asks separately whether the household holds specific savings instruments (`DB9C`–`DB9I`). These are survey-reported *behaviour*, entirely independent of the consumption arithmetic that defines `Goal_Met`, so they are a genuine external check on whether the normative target tracks real saving. They are carried as `Has_*` columns.

| Indicator | Rate among `not_met` | Rate among `met` | Lift |
| --- | --- | --- | --- |
| Has bank savings | 0.5568 | 0.6165 | +0.0597 |
| Has fixed deposit | 0.0822 | 0.1518 | +0.0696 |
| Has pension/LIC | 0.1478 | 0.2189 | +0.0711 |
| Has securities | 0.0117 | 0.0252 | +0.0135 |

> **In plain terms — lift.** **Lift** here is just the gap between two rates. 55.68% of households the benchmark calls "not met" hold a bank savings account, against 61.65% of those it calls "met" — a gap of about 6 percentage points, so the households our arithmetic labels as savers really are a bit more likely to have somewhere to put savings. A positive gap in all four rows means the benchmark is pointing in the right direction. (Careful: a **percentage point** gap is a subtraction of two percentages, not a percentage change. 55.68% → 61.65% is +6.0 percentage points but only about +11% in relative terms.)

All four point the right way. A model's predicted probability of `Goal_Met` achieves ROC-AUC 0.566 against observed bank savings — **weak, but above chance**, and honestly so: a normative accounting threshold is related to, but clearly not the same as, whether a household actually holds savings instruments.

> **In plain terms — ROC-AUC.** This is the project's main measure of **ranking** quality, and it has an unusually concrete meaning. Pick one household at random from each group — one that really does hold bank savings, one that does not. **ROC-AUC is the probability that the model gives the first a higher score than the second.** So 0.50 is a coin flip (the model has learned nothing), 1.00 is perfect ordering, and **0.566 means it wins that comparison 56.6 times out of 100** — better than guessing, but only just. Note what it does *not* measure: it judges the order the model puts households in, never whether its actual yes/no calls are right. That is a separate question, handled by the accuracy and F1 measures introduced in [Phase 0](phase0.md) and [Phase 3](phase3.md).

---

## Limitations of this dataset

- **Vintage.** 2011-12. Fine for methodology; not current for any business claim. IHDS-3 fieldwork is complete but public microdata was not released as of this work.
- **Grain.** One row is a **household**, not an individual. Per-capita variables (`INCOMEPC`, `COPC`) exist in the raw file where per-person framing is needed.
- **Income under-reporting.** A median savings rate of −10.8% reflects the survey's known tendency to under-capture income relative to consumption. It biases the `Goal_Met` rate downward at any threshold.
- **Survey weights.** `WT` is carried in the CSV and is **not** applied anywhere. National-level descriptive claims should be weighted; model fitting on unweighted data is defensible but should be stated.

  > **In plain terms — survey weights.** A survey cannot interview everyone, and it deliberately does not sample everyone with equal probability — some regions and groups are over-sampled so there are enough of them to analyse. A **weight** is the correction: it says how many real households this row stands for. Counting rows without applying weights describes *the sample*; applying them describes *the country*. That is why any national-sounding figure needs `WT` and a model fitted for ranking does not.

- **No repayment flow.** `Debt_To_Income` is a stock-based stand-in for a cash-flow burden.
- **`COTOTAL` vs the reconstruction.** The build sums the 52 items with `fillna(0)`; IHDS's own `COTOTAL` imputes some unanswered items. The target uses `COTOTAL` (the official aggregate) while the shares use the reconstruction (so they sum to exactly 1). The measured cost is **0.25 percentage points** on the `Goal_Met` rate.
- **`Rent_Share` is zero for most households** (median 0.000, mean 0.010). Most Indian households own their homes. Any analysis that leans on rent must condition on `Rent_Share > 0` or it will describe a ~10% urban subsample while appearing to describe the whole.
- **`Debt_To_Income` has a very long right tail** (mean 0.88, sd 11.75, max 1300). These are genuine responses from households with near-zero reported income, not coding errors, but they dominate distance-based and linear methods. The build leaves the raw value so the winsorisation decision stays visible in [`phase2.md`](phase2.md).

  > **In plain terms — long right tail, standard deviation, winsorisation.**
  > - **Standard deviation (sd)** measures spread: roughly, how far a typical value sits from the average. Here the average is 0.88 and the sd is 11.75 — the spread is thirteen times the average, which is the signature of a column where most values are small and a few are astronomically large.
  > - A **long right tail** is exactly that picture: nearly everyone bunched near zero, and a thin trail of extreme values running off to the right (here, up to 1,300× annual income).
  > - Those few extremes cause real damage to methods that measure distance or fit straight lines, because one household 1,300 units away outweighs thousands of ordinary ones. **Winsorising** is the standard blunt fix: pick a cut-off (say the 99th percentile) and pull everything above it down to that cut-off. Nothing is deleted, and the households stay in the data — their values are just capped so they stop dominating. [Phase 2](phase2.md) tests whether it actually matters.

- **`Head_Age` ranges 11–99.** The low end reflects genuinely child-headed households.
