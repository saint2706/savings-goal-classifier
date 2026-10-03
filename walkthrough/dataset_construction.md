# Dataset construction

**Source:** [README § 2. Dataset](../README.md#2-dataset) and [§ 3. Target Variable](../README.md#3-target-variable); India Human Development Survey-II, 2011-12, ICPSR study 36151, dataset DS0002 (Household)
**Notebook:** none. The build is a package module: [`src/savings_goal/data/build.py`](../src/savings_goal/data/build.py), validated by [`src/savings_goal/data/schema.py`](../src/savings_goal/data/schema.py), run through the `sgc build` command
**Builds on:** nothing; this is the first document in the reading order
**Artifacts:** `dataset/households.parquet` (41,518 households × 54 columns), `dataset/features.parquet` (engineered features plus the grouped train/test split); the reconciliation figures are also stored in `results/leakage.json`
**Acquisition:** see [`dataset/README.md`](../dataset/README.md)

This document covers how the analysis table is built from the raw survey file: what IHDS provides, how its 52 consumption items map onto 11 expense categories, how the target is defined, why most spending information cannot be used as model input, and what the dataset cannot tell you.

> **How to read this walkthrough.** Technical terms are explained in quoted notes like this one the first time they come up. Later phases assume the vocabulary introduced here. Three words recur everywhere:
>
> - Feature: an input column the model is allowed to look at (income, household size, the occupation of the main earners).
> - Target (or label): the single column the model predicts. Here it is `Goal_Met`, which is 1 or 0 for every household.
> - Household: one row of the dataset, and the unit every claim is about. Never one person.

---

## Research questions & answers

| # | Question | Answer |
| --- | --- | --- |
| 1 | Which survey, and why? | IHDS-II (2011-12), because it measures income and item-level consumption on the same household. 42,152 households in the raw file. |
| 2 | How are the 52 consumption items turned into annual rupees? | 30-day items `CO1X`–`CO33` × 12, plus annual items `CO34`–`CO52`. The 11-category total correlates 0.9932 with IHDS's own `COTOTAL`, median difference Rs 0, within 1% for 97.72% of households. |
| 3 | How many households survive the build? | 41,518. 634 (1.5%) are dropped because `INCOME <= 0` (618 households) or `COTOTAL` is missing or zero. |
| 4 | What is the target? | `Goal_Met = 1` if `(INCOME − COTOTAL) / INCOME >= 0.20`. 31.93% of households meet it (30.47% survey-weighted); 68.07% are at risk. |
| 5 | Which columns would give the answer away? | Raw rupee categories and expense-to-income ratios each reproduce the label for 99.75% of households. No single spending share does (largest marginal ROC-AUC gap from 0.5 is 0.10, healthcare share). |
| 6 | Are the spending shares safe, then? | Not jointly. Household size predicts food rupees (R² 0.29, elasticity 0.63), so size and food share together back out total spend. A three-column formula with no model reaches ROC-AUC 0.895. Shares are therefore diagnostic only; the headline model uses income, demographics and debt. |
| 7 | How many sampling clusters? | 2,461 primary sampling units, keyed by `IDPSU`. `PSUID` alone takes only 39 values because it repeats across districts. |

---

## Why IHDS-II

The feature set needs income and category-level expenditure measured on the same household. That rules out most Indian household microdata: the NSS-lineage surveys, including the current HCES rounds, collect consumption without income.

IHDS-II is the largest freely available survey that measures both:

| | |
| --- | --- |
| Households | 42,152, nationally representative |
| Reference period | 2011-12 |
| Income | `INCOME` (total, annual rupees), assembled by IHDS from more than 50 income sources, with components such as `INCAG`, `INCBUS`, `INCSALARY`, `INCBENEFITS`, `INCREMIT` |
| Expenditure | 52 consumption items, `CO1X`–`CO52`, plus the survey's own total `COTOTAL` |
| Cost | Free; ICPSR registration only |
| Formats | Stata, SPSS, SAS, R, TSV (the build reads the TSV) |

The main alternative, for anyone with institutional access, is CMIE's Consumer Pyramids (CPHS): 174,000+ households, a monthly panel since 2014, 150+ expense heads, behind a paid subscription.

---

## Running the build

```bash
uv sync
uv run sgc build --tsv path/to/36151-0002-Data.tsv
```

`sgc build` (in [`src/savings_goal/cli.py`](../src/savings_goal/cli.py)) calls `savings_goal.data.build.write`, validates the result with `savings_goal.data.schema.validate_households`, prints a build report and the totals reconciliation, and by default also writes the engineered feature table (`--no-features` skips that step). `--threshold` changes the savings-rate benchmark; the default is `0.20`. Expected output:

```text
raw households:        42,152
dropped (income<=0 / missing consumption): 634
final households:      41,518
Goal_Met rate @ 20%:  0.3193
occupation ties broken by priority: 7,491
debt (DB5) missing:    3,150
COTOTAL vs 11-category sum: within 1% for 97.72%; label agreement 0.9975
```

## Build module walkthrough

The module is a lazy Polars pipeline: nothing is computed until the result is written, and only the columns the build needs are read from the 758-column TSV.

### `scan_raw`: read only what is needed

Reads the TSV header, checks that every required column exists (raising `KeyError` with the missing names if not), and scans just those columns. ICPSR writes missing values as a single space, so every column is read as text, stripped, and cast to a float non-strictly: anything unparseable becomes null rather than an error.

### `CATEGORY_MAP` and `_category_exprs`: from 52 items to 11 categories

IHDS uses two recall windows, and mixing them inflates food twelvefold relative to durables:

- `CO1X`–`CO33`: 30-day recall (questionnaire blocks HQ23/HQ24)
- `CO34`–`CO52`: annual recall (block HQ25)

`CATEGORY_MAP` stores each category as a pair (monthly items, annual items). `_category_exprs` fills missing items with 0, multiplies the monthly sum by 12 and adds the annual sum. All amounts below are annual rupees.

| Category | IHDS items | Contents |
| --- | --- | --- |
| `Groceries` | `CO1X`–`CO3X`, `CO5X`–`CO14X`, `CO15`–`CO17`, `CO19` | Food staples, salt and spices, tea and coffee, processed foods, fruits and nuts |
| `Eating_Out` | `CO20` | Restaurants and eating out |
| `Utilities` | `CO4X`, `CO21`, `CO22`, `CO24` | Kerosene, household fuel, electricity, telephone/cable/internet |
| `Rent` | `CO30` | House rent and society charges |
| `Transport` | `CO28`, `CO29` (monthly); `CO45` (annual) | Conveyance, petrol and maintenance, vehicle purchase |
| `Healthcare` | `CO33` (monthly); `CO34`, `CO46` (annual) | Out-patient, in-patient, therapeutic appliances |
| `Education` | `CO35`–`CO37` (annual) | Fees, private tuition, school books |
| `Entertainment` | `CO23` (monthly); `CO43`, `CO51` (annual) | Entertainment, recreation goods, vacations |
| `Insurance` | `CO50` (annual) | Insurance premiums |
| `Clothing_Footwear` | `CO38`, `CO39` (annual) | Clothing and footwear; large enough that folding it into Miscellaneous would distort the mix |
| `Miscellaneous` | `CO18`, `CO25`–`CO27`, `CO31`, `CO32` (monthly); `CO40`–`CO42`, `CO44`, `CO47`–`CO49`, `CO52` (annual) | Paan/tobacco, toiletries, soap, taxes and fees, services, furniture, crockery, appliances, jewellery, personal care, repairs, social functions |

Two mappings are easy to get wrong:

- `CO4X` is kerosene. It sits in the `CO*X` food block because IHDS collects it on the same quantity-and-price schedule as staples. It is a fuel, so it goes to `Utilities`. Leaving it in `Groceries` would overstate food and understate energy for the poorest households, who use the most kerosene.
- `CO18` is paan/tobacco/intoxicants, not eating out (that is `CO20`). It goes to `Miscellaneous` because it is closer to habitual than elective spending.

Every mapping comes from `36151-0002-Codebook.pdf`. The IHDS project website's summary of the consumption module lists the items in a different order from the variable labels; following it would put house rent at `CO27` (really soap and detergents) and eating out at `CO18`.

### Filtering the sample

`build_lazy` sums the 11 categories into `Category_Total`, then keeps households with `INCOME > 0` and a positive, non-missing `COTOTAL`.

| Step | Households |
| --- | --- |
| Raw DS0002 | 42,152 |
| Dropped: `INCOME <= 0` or `COTOTAL` missing/zero | 634 (1.5%) |
| Final | 41,518 |

A savings rate is undefined when income is zero or negative. In the raw file, net agricultural income (`INCAG`) is negative for 10.7% of households, and for 618 of them that pulls total income to zero or below. This is a documented property of how IHDS nets farm costs against farm revenue, not a data error.

### Totals: `COTOTAL` for the target, `Category_Total` for the shares

Two spending totals exist and they are kept side by side:

- `COTOTAL` is IHDS's own aggregate. It imputes some unanswered items, and the target is built from it.
- `Category_Total` is the sum of the 11 rebuilt categories (missing items counted as zero). The composition shares divide by it, so they sum to exactly 1.

`savings_goal.data.build.reconcile_totals` measures how far apart they are:

| Check | Value |
| --- | --- |
| Pearson correlation | 0.9932 |
| Median difference | Rs 0 |
| Within 1% of `COTOTAL` | 97.72% |
| Within 5% of `COTOTAL` | 98.29% |
| Label rebuilt from `Category_Total` agrees with `Goal_Met` | 99.75% |

So savings is an exact identity over `INCOME` and `COTOTAL`, and an approximate one over the 11 categories. `INCOME` (mean Rs 129,977) and `COTOTAL` (mean Rs 117,939) are both annual, so they compare directly.

> **In plain terms: correlation, median, percentiles.**
> - Correlation (`r`) is a number between −1 and +1 saying how tightly two columns move together. +1 means perfect lockstep, 0 means no relationship. 0.9932 means the rebuilt total and the survey's own total almost always agree.
> - The median is the middle value: line all households up in order and read off the one in the middle. This project prefers it to the mean (the ordinary average) because a few enormous values drag a mean around and leave the median alone.
> - A percentile is the same idea at another position: the 25th percentile is a quarter of the way up the queue.

### Derived columns

The same `with_columns` step builds the household descriptors:

- `Savings = INCOME − COTOTAL`, then `Savings_Rate = Savings / INCOME`, then `Goal_Met`.
- The 11 `*_Share` columns: each category divided by `Category_Total`.
- `Log_Income`: the natural log of `INCOME`.
- `Household_Size`: `NPERSONS`.
- `Age_Dependents`: members aged 0–14 plus members aged 60+ (`NCHILDM + NCHILDF + NELDERM + NELDERF`). The codebook defines `NADULT*` as 21+ (elders included), `NTEEN*` as 15–20, `NCHILD*` as 0–14 and `NELDER*` as 60+. This is the standard age-dependency count; 15- to 20-year-olds count as working age.
- `Head_Age`: `MHEADAGE`, falling back to `FHEADAGE` for female-headed households.
- `Max_Adult_Education`: `HHEDUC`.
- `Area_Type` from `URBAN4_2011`: `Metro_Urban`, `Other_Urban`, `Developed_Village`, `Less_Developed_Village`.
- `Caste_Group` from `ID13` and `Religion` from `ID11`. Religion code 9 is stored as `No_Religion` (12 households), because the string `None` is read back as missing by pandas.
- `Debt_To_Income = DB5 / INCOME`, with `Debt_Missing = 1` when `DB5` is blank. A blank `DB5` (3,150 households, 7.59%) is unknown debt, not zero debt, and it carries information: 37.1% of those households meet the goal against 31.5% of households with recorded debt.
- The six `Has_*` savings-instrument columns, copied from `DB9C`–`DB9I` (see below).

### `_occupation_exprs`: a household occupation

IHDS has no household occupation variable. `Occupation` is the worker type with the most members (each counted at 240+ hours a year) among `NWKSALARY`, `NWKBUSINESS`, `NWKNONAG`, `NWKFARM`, `NWKAGLAB`, or `No_Regular_Worker` when all five are zero. Ties are broken by that list order, from the most regular cash income to the least: salaried, business, non-agricultural labour, farm, agricultural labour. The reasoning is that the more regular income stream dominates a household's ability to plan saving. 7,491 households (18.04%) are ties, flagged in `Occupation_Tie`.

| Occupation | Households |
| --- | --- |
| Salaried | 10,560 |
| Farm | 9,547 |
| Non_Ag_Labour | 8,751 |
| Business | 5,391 |
| Ag_Labour | 3,825 |
| No_Regular_Worker | 3,444 |

### `write` and `BuildReport`

`write` streams the lazy result to Parquet (or CSV if the output path ends in `.csv`), reads it back and returns a `BuildReport` with the raw and final counts, the `Goal_Met` rate, the number of occupation ties and the number of missing debt values. The output keeps identifiers (`STATEID`, `DISTID`, `PSUID`, `IDPSU`, `HHID`, `HHSPLITID`, `IDHH`), the survey weight `WT`, the descriptors, the shares, the `Has_*` columns, the raw categories, both totals, and the three outcome columns: 54 columns in all.

### `schema.py`: the contract every build must pass

`households_schema` is a Pandera schema checked on every Parquet build. It enforces:

- `IDHH` unique; `INCOME`, `COTOTAL`, `Category_Total` and `WT` strictly positive; `Household_Size >= 1`; rupee categories non-negative; `Savings_Rate <= 1`.
- Each share in [0, 1], and the 11 shares summing to 1 within 1e-9 for every row (`shares_sum_to_one`). On the built table the row sums have min = max = 1.0.
- Categorical columns restricted to their known labels.
- `Goal_Met` monotone in the savings rate (`label_monotone_in_savings_rate`): every household with `Goal_Met = 1` has a higher savings rate than every household with `Goal_Met = 0`. The threshold itself is not stored, so this is how the schema confirms the label is a cut of the rate.

Validation runs with `lazy=True`, so a failing build reports every broken check at once.

> **In plain terms: "sum to exactly 1".** The 11 shares are slices of one pie, so they must add up to the whole pie for every household. The schema checks this row by row rather than on average. The property becomes a nuisance later: [Phase 1](phase1.md) shows it forces the shares to move against each other, and [Phase 2](phase2.md) shows how it breaks models that assume independent inputs.

---

## The target

A household survey does not record what its respondents intend to save. IHDS measures income and consumption, not aspirations, so savings adequacy is defined against an external benchmark:

```text
Savings      = INCOME - COTOTAL
Savings_Rate = Savings / INCOME
Goal_Met     = 1 if Savings_Rate >= 0.20 else 0
```

> **In plain terms: the three lines above.** Take what the household earned in a year and subtract everything it spent; the remainder is savings (negative if it spent more than it earned). Dividing by income gives the savings rate, a proportion, so a rich and a poor household are on the same scale: 0.20 means "kept a fifth of what came in". `Goal_Met` answers yes or no: did the rate reach 0.20? A cut that turns a sliding scale into yes/no is a **threshold**, and where it sits is a choice, not a fact about households.

20% is the default of `--threshold` and a convention, not a measurement. [Phase 1](phase1.md) publishes the sensitivity curve, which re-computes the rate at several thresholds:

| Threshold | Goal_Met rate | Survey-weighted |
| --- | --- | --- |
| 0% | 0.4411 | 0.4231 |
| 5% | 0.4138 | 0.3966 |
| 10% | 0.3833 | 0.3669 |
| 20% | 0.3193 | 0.3047 |
| 30% | 0.2532 | 0.2396 |
| 40% | 0.1876 | 0.1757 |

The median savings rate is −10.8%, and only 44.1% of households save anything at all; 55.9% report consumption above income. This is a known feature of Indian household surveys: income is under-reported relative to item-by-item consumption. It is stated as a limitation rather than corrected.

---

## The leakage constraint

> **In plain terms: leakage.** **Leakage** is when a feature secretly contains the answer. If the model can recompute the label from a column by arithmetic, it will score brilliantly and learn nothing: it is copying, not predicting. On new households the copied column would not be available, or would not mean the same thing.

### Raw rupees and income ratios rebuild the label

With a fixed benchmark, the target is an arithmetic function of spending relative to income:

```text
Goal_Met = 1  ⟺  Savings_Rate >= 0.20  ⟺  COTOTAL / INCOME <= 0.80
                                       ⟺  sum(expense / INCOME) <= 0.80   (approximately)
```

> **In plain terms.** `⟺` means "these are the same statement written differently", the way `a − b ≥ 0` is a rearrangement of `a ≥ b`. "The household met the goal" and "its expense-to-income ratios add up to at most 0.80" are one fact written two ways.

On the built table, `sum(expense / INCOME) <= 0.80` reproduces `Goal_Met` for 99.75% of households, and the raw rupee categories with `INCOME` do the same (99.75%). Both are excluded. This holds for any constant savings benchmark, not just this survey.

### Shares look safe one at a time

Dividing each category by total spending instead of income gives the shape of the budget with its size divided out:

```text
Groceries_Share = Groceries / Category_Total      (not / INCOME)
```

> **In plain terms: why the denominator matters.** Food divided by income answers "how much of what they earned went on food", which is part of the answer. Food divided by total spending answers "of what they spent, what fraction was food". A household spending Rs 50,000 and one spending Rs 500,000 can both have a 40% food share.

Tested one at a time, no share predicts the label well. The strongest is `Healthcare_Share` with a marginal ROC-AUC of 0.400 (0.10 from chance); `Utilities_Share` follows at 0.581.

> **In plain terms: ROC-AUC.** Pick one household that met the goal and one that did not. ROC-AUC is the probability that a score ranks the first above the second. 0.50 is a coin flip, 1.00 is perfect ordering. A value below 0.5, like 0.400 for healthcare share, means the score ranks in the wrong direction: higher health spending goes with missing the goal. Its distance from 0.5 is what measures strength.

### Size × food share rebuilds total spend

A naive reading of the one-at-a-time test would suggest the shares are leakage-free. The data shows otherwise once they are combined with household size and income. Food spending grows with household size: regressing log food rupees on log household size gives R² 0.29 and an elasticity of 0.63 (`savings_goal.evaluation.leakage.food_size_fit`, 41,494 households with positive food spend). So `k × Household_Size` approximates food rupees, and dividing by the food share recovers total spend:

```text
implied total spend  ≈ k × Household_Size / Groceries_Share
implied savings rate = 1 − implied total spend / INCOME
```

`savings_goal.evaluation.leakage.size_share_oracle` sets `k` to the median food rupees per person (Rs 8,848 a year), excludes the 24 households with a zero food share, and ranks the rest by the implied rate. With no model, no training and three columns plus a median, it reaches ROC-AUC 0.895, and cutting the implied rate at 0.20 agrees with `Goal_Met` for 82.2% of households.

> **In plain terms: elasticity and R².** An **elasticity** of 0.63 means a household 10% larger spends about 6.3% more on food. **R²** is the share of the variation in food spending that household size explains: 0.29, or 29%. That is enough to turn the food share into a rough meter of total spending.

The ablation in [Phase 1](phase1.md) confirms it under PSU-grouped cross-validation with XGBoost:

| Feature set | Features | ROC-AUC | Share of full ROC-AUC |
| --- | --- | --- | --- |
| All features | 29 | 0.931 | 100% |
| No `Groceries_Share` | 28 | 0.929 | 99.8% |
| No size family (`Household_Size`, `Age_Dependents`, `Dependency_Ratio`) | 26 | 0.921 | 98.9% |
| No shares or participation indicators (deployable) | 13 | 0.878 | 94.3% |

Removing one route barely moves the score because the other shares and size columns carry the same information. Removing all spending-mix columns costs 0.053 ROC-AUC, which is the part of the full model's skill that comes from reconstructing total spend.

### Resulting feature sets

| Set | Columns | Role |
| --- | --- | --- |
| Deployable (13) | `Log_Income`, `Household_Size`, `Age_Dependents`, `Dependency_Ratio`, `Head_Age`, `Max_Adult_Education`, `Debt_To_Income`, `Has_Debt`, `Debt_Missing`, `Occupation`, `Area_Type`, `Caste_Group`, `Religion` | Headline model: everything known at onboarding without a spending diary |
| Full (29) | Deployable plus the 11 `*_Share` columns and 5 `Spends_On_*` participation indicators | Diagnostic only |
| Never features | 11 raw rupee categories, `Category_Total`, `COTOTAL`, `Savings`, `Savings_Rate` (listed in `savings_goal.config.LEAKAGE_COLS`) | Leakage |
| Reserved | 6 `Has_*` savings-instrument columns | External validation |

`Dependency_Ratio`, `Has_Debt` and the `Spends_On_*` indicators are added by `savings_goal.features.engineer.engineer`, which `sgc build` runs after the household build.

### Reserved for validation: savings instruments

IHDS asks separately whether the household holds specific savings instruments (`DB9C`–`DB9I`). These are reported behaviour, independent of the consumption arithmetic behind `Goal_Met`, so they are an outside check on whether the benchmark tracks real saving. They are never features.

| Indicator | Rate when not met | Rate when met | Gap (pp) |
| --- | --- | --- | --- |
| `Has_Bank_Savings` | 0.557 | 0.616 | +6.0 |
| `Has_Pension_LIC` | 0.148 | 0.219 | +7.1 |
| `Has_Fixed_Deposit` | 0.082 | 0.152 | +7.0 |
| `Has_Post_Office_Account` | 0.110 | 0.134 | +2.5 |
| `Has_Gold_Jewellery` | 0.095 | 0.120 | +2.5 |
| `Has_Securities` | 0.012 | 0.025 | +1.3 |

All six point the same way: households the benchmark calls savers are more likely to hold each instrument. The gaps are small, so the accounting benchmark is related to holding savings instruments without being the same thing.

> **In plain terms: percentage points.** A **percentage point** (pp) gap is a subtraction of two percentages. 55.7% to 61.6% is +6.0 pp, which is about +11% in relative terms.

---

## Limitations of this dataset

- **Vintage.** 2011-12. Suitable for methodology; not current for any business claim. IHDS-3 fieldwork is complete but its public microdata was not released at the time of this work.
- **Grain.** One row is a household. Per-capita variables (`INCOMEPC`, `COPC`) exist in the raw file but are not what the model predicts.
- **Income under-reporting.** A median savings rate of −10.8% reflects under-captured income. It pushes the `Goal_Met` rate down at every threshold, so absolute prevalence figures are not reliable; rankings are more defensible.
- **Survey weights.** `WT` is carried in the table. Phase 1 reports descriptive rates both unweighted and weighted (31.93% vs 30.47% for `Goal_Met`); models are fitted unweighted.

  > **In plain terms: survey weights.** IHDS deliberately over-samples some regions and groups so there are enough of them to analyse. A **weight** says how many real households a row stands for. Unweighted counts describe the sample; weighted counts describe the country.

- **Clustered sample.** Households are sampled in clusters (PSUs), and neighbours share prices, wages and survey teams. Cross-validation groups by `IDPSU` (2,461 clusters) so that no village or urban block appears on both sides of a split.
- **No repayment flow.** The debt module (`DB1`–`DB9I`) records outstanding debt as a stock (`DB5`), the largest loan, its purpose, source and interest, but no instalment. `Debt_To_Income` is the nearest available construct and is not a claim on this month's cash flow.
- **Long tail in debt.** `Debt_To_Income` has mean 0.96, standard deviation 12.2, a 99th percentile of 11.0 and a maximum of 1,300. These are recorded responses from households with near-zero reported income. The build keeps the raw value; the model pipeline caps it at the 99th percentile, re-learned inside each training fold.

  > **In plain terms: long tail and winsorising.** The **standard deviation** measures spread; here it is 13 times the mean, the signature of a column where most values are small and a few are enormous. Those few dominate methods that measure distance or fit straight lines. **Winsorising** caps everything above a cut-off (here the 99th percentile) at the cut-off: no household is removed, but none can dominate.

- **Two totals.** The target uses `COTOTAL`, the shares use `Category_Total`. A label built from `Category_Total` would differ for 0.25% of households.
- **Rent is mostly absent.** `Rent_Share` is zero for 90.4% of households (mean 0.010), because most Indian households own their homes. An analysis that leans on rent describes a mostly urban 9.6% subsample.
- **Child-headed households.** `Head_Age` ranges from 11 to 99; 14 households have a head under 18.

---

## What this means for later phases

| Fact fixed here | Consequence |
| --- | --- |
| `Goal_Met` is a 20% cut of a continuous, under-reported savings rate | Rankings are defensible, absolute rates are not; Phase 1 publishes the threshold sensitivity |
| Raw rupees and income ratios rebuild the label (99.75%) | Never features, in any phase |
| Shares with size and income rebuild total spend (oracle ROC-AUC 0.895) | The headline model is the 13-feature deployable set; the 29-feature full set is reported as a diagnostic ceiling |
| `IDPSU` is the cluster key | All cross-validation and the held-out split are grouped by it |
| Shares sum to exactly 1 | Phases 1, 2 and 6 handle the compositional constraint (log-ratio transforms, ILR for clustering) |
| `Has_*` reserved | Used only as an external check on the benchmark |
| Debt has a long tail and 7.6% missing | Missing is kept as its own signal; capping happens inside the model pipeline |
