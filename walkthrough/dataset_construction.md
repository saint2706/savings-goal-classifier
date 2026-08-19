# Dataset Construction

**Build script:** [`src/build_dataset.py`](../src/build_dataset.py)
**Source:** India Human Development Survey-II, 2011-12 — ICPSR study 36151, dataset DS0002 (Household)
**Output:** `dataset/households.csv` (41,518 households × 50 columns)
**Acquisition:** see [`dataset/README.md`](../dataset/README.md)

This document records how the analysis dataset is built from the raw survey file: what IHDS provides, how its 52 consumption items map onto 11 expense categories, how the target is defined, and the structural problem that definition creates.

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

`INCOME` (mean ₹127,760) and `COTOTAL` (mean ₹118,055) are both annual, so they are directly comparable without rescaling.

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

The threshold is configurable via `--threshold`; 20% is the default and is a **convention, not a measurement**. [`phase1.md`](phase1.md) publishes the full sensitivity curve.

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

A fixed benchmark makes the target an **exact arithmetic function of the expense-to-income ratios**:

```text
Goal_Met = 1  ⟺  Savings_Rate >= 0.20  ⟺  COTOTAL/INCOME <= 0.80
                                       ⟺  sum(expense ratios) <= 0.80
```

Measured on the built dataset, `sum(expense/income) <= 0.80` reproduces `Goal_Met` with **99.75% agreement**. Feeding the ratio features into a model would be reconstructing the label, not predicting it.

**The consequence is that expense-to-income ratios — the intuitive representation — are unusable here.** This is a property of any normative savings threshold, not of IHDS: whenever the benchmark is a constant, knowing what fraction of income each category consumes is knowing the answer.

**Resolution — composition shares instead of income ratios:**

```text
Groceries_Share = Groceries / COTOTAL      (not / INCOME)
```

The eleven shares sum to exactly 1.0 for every household (verified: min = mean = max = 1.0), so they describe only the *shape* of spending and carry no information about spending relative to income. They therefore cannot reconstruct `COTOTAL/INCOME`.

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

All four point the right way. A model's predicted probability of `Goal_Met` achieves ROC-AUC 0.566 against observed bank savings — **weak, but above chance**, and honestly so: a normative accounting threshold is related to, but clearly not the same as, whether a household actually holds savings instruments.

---

## Limitations of this dataset

- **Vintage.** 2011-12. Fine for methodology; not current for any business claim. IHDS-3 fieldwork is complete but public microdata was not released as of this work.
- **Grain.** One row is a **household**, not an individual. Per-capita variables (`INCOMEPC`, `COPC`) exist in the raw file where per-person framing is needed.
- **Income under-reporting.** A median savings rate of −10.8% reflects the survey's known tendency to under-capture income relative to consumption. It biases the `Goal_Met` rate downward at any threshold.
- **Survey weights.** `WT` is carried in the CSV and is **not** applied anywhere. National-level descriptive claims should be weighted; model fitting on unweighted data is defensible but should be stated.
- **No repayment flow.** `Debt_To_Income` is a stock-based stand-in for a cash-flow burden.
- **`COTOTAL` vs the reconstruction.** The build sums the 52 items with `fillna(0)`; IHDS's own `COTOTAL` imputes some unanswered items. The target uses `COTOTAL` (the official aggregate) while the shares use the reconstruction (so they sum to exactly 1). The measured cost is **0.25 percentage points** on the `Goal_Met` rate.
- **`Rent_Share` is zero for most households** (median 0.000, mean 0.010). Most Indian households own their homes. Any analysis that leans on rent must condition on `Rent_Share > 0` or it will describe a ~10% urban subsample while appearing to describe the whole.
- **`Debt_To_Income` has a very long right tail** (mean 0.88, sd 11.75, max 1300). These are genuine responses from households with near-zero reported income, not coding errors, but they dominate distance-based and linear methods. The build leaves the raw value so the winsorisation decision stays visible in [`phase2.md`](phase2.md).
- **`Head_Age` ranges 11–99.** The low end reflects genuinely child-headed households.
