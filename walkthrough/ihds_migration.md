# Migration — From Synthetic Kaggle Data to IHDS-II

**Builds on:** all phases (this changes the dataset every phase runs on)
**Build script:** [`src/build_ihds_dataset.py`](../src/build_ihds_dataset.py)
**Output:** `dataset/ihds2_households.csv` (41,518 households × 40 columns)
**Source:** India Human Development Survey-II, 2011-12 — ICPSR study 36151, dataset DS0002 (Household)

The bonus extensions ([`bonus.md`](bonus.md)) established that the Kaggle dataset's non-target columns were largely generated independently of one another: `Rent_Ratio` is a fixed 0.30/0.20/0.15 lookup on `City_Tier`, `Occupation` is pure noise, and discretionary spending is a fixed ~7% of income regardless of age. Those findings mean the core model's near-perfect scores demonstrate correct methodology on a well-behaved dataset rather than anything about Indian household finance.

This document records the move to real survey data: what IHDS-II provides, how its 52 consumption items map onto the project's 11 categories, why the target had to be redefined, and the one structural problem that redefinition created.

---

## Why IHDS-II

The binding constraint is that this project's entire Phase 2 feature set is `expense / income`, so it needs **income and category-level expenditure measured on the same household**. That eliminates most real consumption microdata: India's NSS-lineage surveys, including the current HCES 2022-23 and 2023-24 rounds, deliberately collect consumption *without* income.

IHDS-II is the strongest freely available survey that measures both:

| | |
| --- | --- |
| Households | 42,152, nationally representative |
| Reference period | 2011-12 |
| Income | `INCOME` (total, annual rupees) plus eight components `INCFARM`…`INCOTHER`, from queries across 50+ income sources |
| Expenditure | 52 consumption items, `CO1X`–`CO52` |
| Cost | Free; ICPSR registration only |
| Formats | Stata, SPSS, SAS, R, TSV |

The alternative worth pursuing if institutional access exists is **CMIE's Consumer Pyramids (CPHS)** — 174,000+ households, monthly panel since 2014, 150+ expense heads — which is a better dataset in every respect except that it requires a paid subscription.

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

The project's 11 expense categories, rebuilt from IHDS items. All amounts are annual rupees.

| Project category | IHDS items | Notes |
| --- | --- | --- |
| `Groceries` | `CO1X`–`CO3X`, `CO5X`–`CO14X`, `CO15`–`CO17`, `CO19` | Food staples plus salt/spices, tea/coffee, processed foods, fruits/nuts |
| `Eating_Out` | `CO20` | Restaurants/eating out — exact match |
| `Utilities` | `CO4X`, `CO21`, `CO22`, `CO24` | Kerosene + household fuel + electricity + telephone/cable/internet |
| `Rent` | `CO30` | House rent/society charges — exact match |
| `Transport` | `CO28`, `CO29`, `CO45` | Conveyance + petrol/maintenance + vehicle purchase |
| `Healthcare` | `CO33`, `CO34`, `CO46` | Out-patient + in-patient + therapeutic appliances |
| `Education` | `CO35`–`CO37` | Fees + private tuition + school books |
| `Entertainment` | `CO23`, `CO43`, `CO51` | Entertainment + recreation goods + vacations |
| `Insurance` | `CO50` | Insurance premiums — exact match |
| `Clothing_Footwear` | `CO38`, `CO39` | New category; large enough that folding it into Miscellaneous would distort the mix |
| `Miscellaneous` | `CO18`, `CO25`–`CO27`, `CO31`, `CO32`, `CO40`–`CO42`, `CO44`, `CO47`–`CO49`, `CO52` | Paan/tobacco, toiletries, soap, taxes/fees, services, furniture, crockery, appliances, jewellery, personal care, repairs, social functions |

**Two mappings that are easy to get wrong:**

- **`CO4X` is kerosene**, and it sits inside the `CO*X` food block because IHDS collects it on the same quantity-and-price schedule as staples. It is a fuel and belongs in `Utilities` — leaving it in `Groceries` would overstate food and understate energy for exactly the poorest households that use kerosene most.
- **`CO18` is paan/tobacco/intoxicants**, not eating out. It is routed to `Miscellaneous` rather than to a discretionary category, because it is closer to habitual than elective spending; treating it as discretionary would change the Q3-style lifecycle analysis.

### The category with no counterpart

**`Loan_Repayment` does not exist in IHDS.** The debt module (`DB1`–`DB9I`) records debt as a **stock**, not a monthly flow: outstanding household debt (`DB5`), largest loan amount, purpose, source, and interest rates — but no repayment instalment.

This matters more than the missing column suggests, because `Loan_Repayment_Ratio` was Phase 5's single strongest SHAP feature, more than 2× the next. The build script substitutes `Debt_To_Income` (`DB5 / INCOME`), which is the nearest available construct but is not the same thing — a stock of debt is not a claim on this month's cash flow.

---

## The target had to be redefined

**No real survey collects `Desired_Savings`.** A self-declared savings target is exactly the kind of variable a synthetic generator invents and a household survey does not ask, so `Goal_Met = Disposable_Income >= Desired_Savings` cannot be reproduced.

The chosen replacement is a **normative benchmark**:

```text
Savings      = INCOME - COTOTAL
Savings_Rate = Savings / INCOME
Goal_Met     = 1 if Savings_Rate >= 0.20 else 0
```

The threshold is configurable via `--threshold`; 20% is the default.

**What changes conceptually:** the original target was *self-referential* — measured against each individual's own stated goal. The new one is *externally imposed* and identical for everyone. Phase 0's framing shifts from "is this person on track for the goal they set themselves" to "is this household retaining a normatively adequate share of income." That is a defensible business question for a savings-product team, but it is a different question, and the README's Phase 0 answer needs updating to say so.

### Sample construction

| Step | Households |
| --- | --- |
| Raw DS0002 | 42,152 |
| Dropped: `INCOME <= 0` or `COTOTAL` missing/zero | 634 (1.5%) |
| **Final** | **41,518** |

Income of zero or below makes a savings *rate* undefined. IHDS reports negative farm income for about 9% of households, which drives roughly 1.5% of *total* income to zero or below — a documented property of the survey, not a data error.

### What the target looks like on real data

| Threshold | Goal_Met rate |
| --- | --- |
| ≥ 0% | 0.4411 |
| ≥ 5% | 0.4138 |
| ≥ 10% | 0.3833 |
| **≥ 20%** | **0.3193** |
| ≥ 30% | 0.2532 |

**The median household savings rate is −10.8%**, and only 44% of households have positive savings at all. This is a well-known feature of IHDS rather than a defect: household income is under-reported relative to consumption in most Indian surveys. It should be stated in the limitations rather than corrected away.

**Class balance changes completely.** The synthetic target had a 0.56% minority class; the new one is 31.9% positive. Phase 4's entire narrative — extreme imbalance, `class_weight="balanced"`, selecting on `recall_0`, the SVM winning on perfect minority recall — was a response to that imbalance and no longer applies. This is now an ordinary, near-balanced classification problem.

---

## The leakage problem the new target created, and how it is resolved

A fixed benchmark makes the target an **exact arithmetic function of the expense-to-income ratios**:

```text
Goal_Met = 1  ⟺  Savings_Rate >= 0.20  ⟺  COTOTAL/INCOME <= 0.80
                                       ⟺  sum(expense ratios) <= 0.80
```

Measured on the built dataset, `sum(expense/income) <= 0.80` reproduces `Goal_Met` with **99.75% agreement**. Feeding the Phase 2 ratio features into a model would be reconstructing the label, not predicting it — precisely what Phase 1's leakage check exists to prevent.

**Why the synthetic dataset did not have this problem:** there, `Desired_Savings_Percentage` varied per individual and was excluded from the feature set, so the threshold itself was unknown to the model. A *constant* benchmark removes that unknown. The leakage is a property of the normative target, not of IHDS.

**Resolution — composition shares instead of income ratios:**

```text
Groceries_Share = Groceries / COTOTAL      (not / INCOME)
```

The eleven shares sum to exactly 1.0 for every household (verified: min = mean = max = 1.0), so they describe only the *shape* of spending and carry no information about spending relative to income. They therefore cannot reconstruct `COTOTAL/INCOME`. The "spending mix predicts savings outcomes" narrative that Phases 2, 5, and 6 are built around survives intact; only the denominator changes.

### Feature set

| Group | Columns |
| --- | --- |
| Spending mix | 11 `*_Share` columns |
| Income | `INCOME`, `Log_Income` |
| Household | `Household_Size`, `Dependents`, `Head_Age`, `Max_Adult_Education` |
| Categorical | `Occupation` (derived), `Area_Type`, `Caste_Group`, `Religion` |
| Debt | `Debt_To_Income` |

**Leakage-excluded** (present in the CSV for EDA, never as features — mirroring how the original dataset carried `Disposable_Income`): the 11 raw rupee category columns, `COTOTAL`, `Savings`, `Savings_Rate`.

The raw category columns must stay out of the feature set for the same reason the ratios do — they sum to `COTOTAL`, which with `INCOME` recovers the target.

**Derived columns worth noting:**

- `Occupation` — IHDS has no household occupation variable. It is derived as the worker type contributing the most workers (≥240 hrs/yr) from `NWKSALARY`, `NWKBUSINESS`, `NWKNONAG`, `NWKFARM`, `NWKAGLAB`, with that list as the tie-break order and `No_Regular_Worker` when all are zero. Unlike the synthetic `Occupation`, this one is real and load-bearing.
- `Area_Type` — from `URBAN4_2011`: `Metro_Urban` / `Other_Urban` / `Developed_Village` / `Less_Developed_Village`. This is the `City_Tier` analogue.
- `Head_Age` — `MHEADAGE`, falling back to `FHEADAGE` for female-headed households.

---

## Validation

### The problem is learnable, and not through leakage

Five-fold stratified cross-validation, 41,518 households:

| Model | CV Accuracy | CV F1 (macro) | CV ROC-AUC |
| --- | --- | --- | --- |
| Majority baseline | 0.6807 | 0.4050 | 0.5000 |
| Logistic Regression | 0.8321 | 0.8166 | **0.9187** |
| Random Forest | **0.8501** | **0.8188** | 0.9179 |

Out-of-fold, the random forest reaches precision 0.819 / recall 0.681 on the `met` class and 0.861 / 0.930 on `not_met`.

**Why a score of 0.85 is a better result than the synthetic dataset's 0.998.** The synthetic pipeline reached near-perfect accuracy because its generator made the target nearly deterministic in the features. Real households retain a large irreducible component that spending mix and demographics cannot explain. A model that lands well above baseline but far from perfect, with a confusion matrix that makes real errors in both directions, is measuring something. Perfect separation on real survey data would be the signal to go looking for a leak.

### Feature importances invert the synthetic story

| Feature | Importance |
| --- | --- |
| `Log_Income` | 0.3358 |
| `Groceries_Share` | 0.0671 |
| `Utilities_Share` | 0.0557 |
| `Healthcare_Share` | 0.0496 |
| `Miscellaneous_Share` | 0.0491 |
| `Debt_To_Income` | 0.0439 |

On real data, **income level dominates** — five times the next feature. The synthetic dataset's headline finding was the opposite: that raw income mattered far less than *how* income was spent. That conclusion was an artifact of a generator that set every expense category as a fixed percentage of income, leaving no genuine variation in spending mix for income to compete with.

`Groceries_Share` ranking second is Engel's law appearing unprompted: food's share of the budget falls as households get richer, so it proxies for the standard of living that determines whether saving is possible at all.

### External validation — something the synthetic data could never support

IHDS asks separately whether the household holds specific savings instruments (`DB9C`–`DB9I`). These are survey-reported *behaviour*, entirely independent of the consumption arithmetic that defines `Goal_Met`, so they are a genuine external check on whether the normative target tracks real saving. They are carried in the CSV as `Has_*` columns and **must not be used as features**.

| Indicator | Rate among `not_met` | Rate among `met` | Lift |
| --- | --- | --- | --- |
| Has bank savings | 0.5568 | 0.6165 | +0.0597 |
| Has fixed deposit | 0.0822 | 0.1518 | +0.0696 |
| Has pension/LIC | 0.1478 | 0.2189 | +0.0711 |
| Has securities | 0.0117 | 0.0252 | +0.0135 |

All four point the right way. The model's predicted probability of `Goal_Met` achieves ROC-AUC 0.566 against observed bank savings — **weak, but above chance**, and honestly so: a normative accounting threshold is related to, but clearly not the same as, whether a household actually holds savings instruments. Reporting that gap is more useful than hiding it, and it is a validity check the synthetic dataset made impossible.

### Area type: the geography result reverses

| Area type | Goal_Met rate | n |
| --- | --- | --- |
| Metro urban | 0.4212 | 3,063 |
| Other urban | 0.3698 | 11,410 |
| Developed village | 0.3091 | 12,635 |
| Less developed village | 0.2666 | 14,410 |

A clean monotone gradient — and it points **opposite** to the synthetic finding. There, all 112 at-risk individuals lived in Tier-1 cities, and Phase 7's headline recommendation was to target Tier-1 residents. In real data, metro households are the *most* likely to save, by 15.5 percentage points over the least-developed villages. Phase 7's central business recommendation does not survive the move to real data.

### One diagnostic worth carrying forward

`Debt_To_Income` correlates −0.595 with `Savings_Rate`, which looks strong enough to be suspicious — both quantities have `INCOME` in the denominator, which can manufacture correlation. Checking within income deciles:

- Bottom decile: −0.596 (the artifact concentrates here)
- All other nine deciles: −0.137 to −0.261

So the raw figure is inflated by shared-denominator effects at the bottom of the income distribution, but a genuine moderate relationship remains throughout. The quintile breakdown confirms real signal at the top of the range: households in the highest `Debt_To_Income` quintile (median 1.43× annual income) meet the goal 15.7% of the time versus 33–36% elsewhere. Keep the feature; do not quote the −0.595.

---

## What each phase needs

| Phase | Status |
| --- | --- |
| **0 — Framing** | **Rewrite required.** Target is now normative, not self-referential. |
| **1 — EDA/leakage** | **Rerun required.** New leakage list (raw categories, `COTOTAL`, `Savings*`); new distributions; real missingness (`Max_Adult_Education` 6, `Caste_Group` 85, `Religion` 12); negative savings rates. |
| **2 — Feature engineering** | **Changed.** Composition shares replace income ratios; the reasoning is here rather than in `phase2.md`. |
| **3 — Baseline** | Rerun. Majority baseline is now 0.681, not 0.994. |
| **4 — Model comparison** | **Rewrite required.** The imbalance narrative no longer applies at 31.9% positive. |
| **5 — Explainability** | Rerun. `Log_Income` now dominates; `Loan_Repayment_Ratio` no longer exists. |
| **6 — Clustering** | Rerun. `Rent_Ratio` is no longer a deterministic tier encoding, so the personas should be genuinely multivariate this time — the low silhouette scores may not recur. |
| **7 — Business translation** | **Rewrite required.** The Tier-1 targeting recommendation reverses. |
| **8 — Reporting** | Rewrite after the above. |
| **Bonus** | Q1–Q3 were findings *about the synthetic generator* and become moot; Q4 is worth rerunning as a real question. |

---

## Limitations specific to IHDS-II

- **Vintage.** 2011-12. Fine for methodology; not current for any business claim. IHDS-3 fieldwork is complete but public microdata was not released as of this migration.
- **Grain change.** One row is a **household**, not an individual. Per-capita variables (`INCOMEPC`, `COPC`) exist where per-person framing is needed.
- **Income under-reporting.** Median savings rate of −10.8% reflects the survey's known tendency to under-capture income relative to consumption. It biases the `Goal_Met` rate downward at any threshold.
- **Survey weights.** `WT` is carried in the CSV and is **not** applied anywhere yet. National-level descriptive claims should be weighted; model fitting on unweighted data is defensible but should be stated.
- **Missing repayment flow.** `Debt_To_Income` is a stock-based stand-in for the strongest feature in the original analysis.
- **Threshold sensitivity.** 20% is a convention, not a finding. Results should be checked at 10% and 30% before any recommendation rests on the boundary.
- **`Rent_Share` is zero for most households** (median 0.000, mean 0.010). Most Indian households — overwhelmingly the rural majority of the sample — own their homes and record no rent. This is the sharpest contrast with the synthetic data, where `Rent_Ratio` was a non-zero deterministic tier key present for everyone. Any analysis that leans on rent must condition on `Rent_Share > 0` or it will be describing a ~10% urban subsample while appearing to describe the whole.
- **`Debt_To_Income` has a very long right tail** (mean 0.88, sd 11.75, max 1300 — a household owing 1,300× its annual income). These are genuine survey responses from households with near-zero reported income, not coding errors, but they will dominate any distance-based or linear method. Phase 1 should decide on a winsorisation or log transform; the build script deliberately leaves the raw value so that decision stays visible rather than baked in.
- **`Head_Age` ranges 11–99.** The low end reflects genuinely child-headed households in the survey. Small in number, but they should be looked at in Phase 1 rather than assumed to be errors.
