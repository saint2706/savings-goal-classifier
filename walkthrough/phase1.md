# Phase 1 (IHDS-II) — Data Understanding

**Source:** [README § Phase 1 — Data Understanding](../README.md#phase-1--data-understanding)
**Notebook:** [`notebooks/01_eda_and_leakage_check.ipynb`](../notebooks/01_eda_and_leakage_check.ipynb)
**Builds on:** [Migration — Synthetic to IHDS-II](dataset_construction.md)
**Replaces:** [`phase1.md`](phase1.md), which analysed the synthetic Kaggle dataset

Same six research questions as the original Phase 1, re-answered on 41,518 real IHDS-II households. Every answer changes, and three of them change enough to invalidate a downstream decision. Everything here is read-only exploration; no features are engineered and no leakage column is ever fed to a model.

---

## Research questions & answers

| # | Question | Answer |
| --- | --- | --- |
| 1 | What does each column mean, and what unit/time period does it represent? | All money columns are **annual** ₹ (the synthetic data was monthly). The grain is a **household**, not an individual. Of 50 columns: 6 identifiers, 1 unapplied survey weight, 19 features, 6 behavioural columns reserved for external validation, 13 leakage-excluded, 1 target. Single cross-section, 2011-12. |
| 2 | Distribution of income, expenses, and savings — skew, outliers, implausible values? | Far more extreme than the synthetic data. `INCOME` skew **15.8** (synthetic: ~4–5); `Clothing_Footwear` skew **112.3**. The headline: **55.9% of households report consumption exceeding income**, and 22.0% spend more than twice their income — against ~0.5% in the synthetic data. Median savings rate is **−10.8%**. Categories are heavily zero-inflated: 90.5% record no rent, 73.7% no insurance, 72.0% no eating out. |
| 3 | Missing values or duplicate rows, and how are they handled? | Missingness now exists but is negligible: `Has_*` columns 0.32–0.42%, `Caste_Group` 0.21%, `Religion` 0.03%, `Max_Adult_Education` 0.01%. **Zero duplicate households, zero duplicate rows.** 634 households (1.5%) were dropped upstream for `INCOME <= 0` or missing consumption. |
| 4 | How correlated are expense categories with income and with each other? | **Dramatically weaker than the synthetic data**, which is the single most important finding here. Raw categories correlate with income at r = 0.09–0.43 (synthetic: 0.79–0.99), and mean \|r\| *between* categories is **0.143** (synthetic: several pairs > 0.94). Among the composition shares, `Groceries_Share` correlates **−0.266** with log income — Engel's law, appearing unprompted. |
| 5 | Is the target mathematically derivable from any candidate feature (leakage check)? | **Yes, from two representations, and both are excluded.** `Savings = INCOME − COTOTAL` holds exactly (max error 5.8×10⁻¹¹). Raw rupee categories reconstruct `Goal_Met` with **99.75%** agreement, and expense-to-income ratios with the same **99.75%**. The composition shares do not: they sum to exactly 1.000 for every household, and the strongest correlation between any share and `Savings_Rate` is **0.056**. |
| 6 | Class balance once leakage columns are excluded? | **31.93% positive, a 2.13:1 imbalance** — against 178:1 in the synthetic data. Survey-weighted the rate is 30.47%. The balance is highly sensitive to the arbitrary threshold, running from 44.1% at 0% to 18.8% at 40%. |

---

## Notebook walkthrough

The notebook carries only section headers and code; the reasoning lives here.

### Cell 1 (code) — Imports, load, and the column-role constants

Loads `dataset/households.csv` and declares four column groups up front: the 11 expense categories, their `_Share` counterparts, the behavioural `Has_*` columns, and `LEAKAGE_COLS`.

**Why the leakage list is a named constant in the first cell rather than derived later:** the original Phase 1 discovered its leakage columns partway through and excluded them afterwards. Here the exclusion set is known before any analysis runs — the migration document already proved which representations reconstruct the target — so declaring it first makes it impossible for an intermediate cell to accidentally treat a leakage column as a feature. Cell 13 then *re-verifies* the list rather than establishing it.

### Cell 3 (code) — Column inventory (Q1)

Builds a table assigning every column a role and unit.

**Why this cell is longer than the synthetic equivalent:** the original dataset had one obvious role per column and 28 of them. This has 50 columns in six distinct roles, including a category the synthetic data had no equivalent for — the `Has_*` behavioural indicators, which are neither features nor leakage but a reserved validation set. Writing the roles down as data (rather than prose in the walkthrough) means later phases can import the grouping instead of re-deriving it.

**Two unit changes that will break any copied code:**

- **Money is annual, not monthly.** IHDS reports `INCOME` and `COTOTAL` as annual rupees. Any threshold, axis label, or business figure carried over from the synthetic analysis is off by 12×.
- **The grain is a household, not an individual.** `Household_Size` and `Dependents` are counts within the unit of observation, not attributes of a person. Per-capita framing requires `INCOMEPC`/`COPC` from the raw IHDS file.

### Cell 5 (code) — Distributions (Q2, part 1)

Reports mean/sd/quartiles/max plus skew for all 14 money columns, then the savings-rate percentiles.

**Result — the skew is on a different scale entirely:**

| Column | Skew | Column | Skew |
| --- | --- | --- | --- |
| `Clothing_Footwear` | 112.3 | `Insurance` | 31.5 |
| `Utilities` | 48.0 | `Entertainment` | 26.8 |
| `Healthcare` | 35.4 | `Rent` | 24.3 |
| `INCOME` | 15.8 | `Transport` | 17.8 |

The synthetic dataset's money columns had skew ≈ 4–5 and a smooth tail. These have skews up to 112, produced by a handful of households reporting a single very large purchase in an annual-recall category. This is normal for survey expenditure data and is why `Log_Income` exists as a feature — but it means any distance-based or linear method in Phases 4 and 6 needs either the log transform or robust scaling, not the `StandardScaler` default that sufficed before.

**Why savings-rate *percentiles* are reported rather than just the mean and sd:** the mean (−1.16) and standard deviation (13.08) are uninterpretable here, because the distribution has a long left tail reaching −1647 (a household consuming 1,648× its reported income). The percentiles are readable and tell the actual story: the 1st percentile is −15.73, the median is −0.108, the 75th is +0.305.

### Cell 6 (code) — Implausible values and zero-inflation (Q2, part 2)

**The headline data-quality finding:**

| Check | Households | Share |
| --- | --- | --- |
| Consumption exceeds income | 23,204 | **55.89%** |
| Spends more than 2× income | 9,137 | 22.01% |
| Spends more than 6× income | 1,635 | 3.94% |

In the synthetic dataset this same check flagged ~0.5% of rows as implausible. Here it flags the **majority of the sample**.

**Why this is not treated as an error to clean:** it is the documented behaviour of Indian household surveys. Income is under-reported relative to consumption — respondents recall irregular, informal, and in-kind earnings poorly, while consumption is asked item-by-item with short recall windows and is captured much more completely. IHDS itself reports negative farm income for about 9% of households. Deleting or winsorising 56% of the sample to make the arithmetic look tidier would discard the survey's actual population and bias every downstream estimate toward richer, more formally employed households.

**What it does mean:** the `Goal_Met` rate is biased *downward* at every threshold, and no absolute figure from this analysis should be quoted as "X% of Indian households save adequately." Relative comparisons — between area types, occupations, spending profiles — are far more defensible than levels, and Phase 7 should be written to lean only on the former.

**Zero-inflation** is the other structural feature:

| Feature | Zero |
| --- | --- |
| `Rent_Share` | 90.5% |
| `Insurance_Share` | 73.7% |
| `Eating_Out_Share` | 72.0% |
| `Entertainment_Share` | 69.1% |
| `Education_Share` | 35.9% |

**Why `Rent_Share` at 90.5% zero deserves special attention:** rent was the *deterministic key* to `City_Tier` in the synthetic data — non-zero for everybody, and exactly 0.30/0.20/0.15 by tier. In reality most Indian households own their homes, so rent is absent for nine in ten. Any analysis leaning on rent is describing a ~10% urban subsample. This single fact retires the entire BONUS-Q1 finding and much of Phase 6's persona structure.

Four of the eleven features being majority-zero also means these are **semi-continuous** variables — a binary "does this household spend on X at all" mixed with a continuous amount. Phase 2 should consider modelling that explicitly (a paired indicator plus amount) rather than feeding a spike-at-zero distribution to a linear model.

### Cell 8 (code) — Missingness and duplicates (Q3)

**Result:** nine columns have missing values, all under 0.5%; zero duplicate household identifiers; zero duplicate rows.

**Why this is still worth a cell when the answer is "almost none":** the original Phase 1 could report exactly zero missing values, and that is itself a tell of synthetic data — real surveys always have refusals and non-response. The pattern here is informative rather than alarming: the six `Has_*` columns cluster at 0.32–0.42% missing, which is the signature of a small number of households that skipped the debt-and-investment block entirely, not of random item non-response. Because those columns are reserved for validation rather than used as features, the missingness needs no imputation strategy — but the affected households must be dropped from validation comparisons rather than treated as "no".

`Caste_Group` (0.21%) and `Religion` (0.03%) do need a decision in Phase 2, since they are model inputs. At this rate, either most-frequent imputation or an explicit `Unknown` level is defensible; an explicit level is preferable because refusal to state caste is plausibly informative rather than random.

### Cell 10 (code) — Correlation with income (Q4, part 1)

**The most consequential comparison in this document:**

| | Synthetic | IHDS-II |
| --- | --- | --- |
| Raw category vs `Income`, range | r = 0.79 – 0.99 | r = 0.09 – 0.43 |
| Mean \|r\| between raw categories | several pairs > 0.94 | **0.143** |

The synthetic dataset's expense columns were near-perfect restatements of income, because the generator set each as a fixed percentage of it. Real spending is only loosely tied to income (strongest: `Groceries` at 0.43; weakest: `Rent` at 0.09).

**Why this changes the justification for Phase 2's central decision:** the original Phase 2 converted expenses to ratios primarily to *remove redundancy* — the raw columns were so collinear with income that they carried little independent signal. That argument no longer applies; the raw columns here are genuinely informative and mutually distinct. Ratios and shares are still the right representation, but now for a different reason: comparability across a 100× income range, and (for shares specifically) leakage avoidance. Phase 2's write-up should state the new justification rather than inherit the old one.

**The share correlations recover a real economic law:**

| Feature | r with `Log_Income` |
| --- | --- |
| `Transport_Share` | **+0.273** |
| `Insurance_Share` | +0.251 |
| `Education_Share` | +0.179 |
| `Healthcare_Share` | −0.128 |
| `Groceries_Share` | **−0.266** |

Food's budget share falling as income rises is **Engel's law**, one of the oldest empirical regularities in economics, and it appears here without being engineered in. The synthetic dataset could not have produced it: `Groceries_Ratio` there was flat at 0.125 across every tier and income level. This is the clearest single piece of evidence that the new dataset contains real economic structure.

### Cell 11 (code) — Correlation among the shares, and the closure problem

**Result:** mean off-diagonal correlation **−0.064**, with **65.5% of all pairs negative**. Strongest negative: `Groceries_Share` × `Healthcare_Share` (−0.362). Strongest positive: `Eating_Out_Share` × `Entertainment_Share` (+0.110).

**Why a predominantly negative correlation matrix is expected rather than a finding:** the shares sum to exactly 1 for every household. That constraint — *compositional closure* — forces the components to be negatively correlated on average, because one share can only rise if others fall. The negative correlations are therefore partly an artifact of the representation, not evidence that households trade groceries off against healthcare.

**What this obliges later phases to do:**

- **Phase 5 (explainability):** a share's coefficient or SHAP value is never "the effect of spending more on X." It is the effect of spending more on X *and correspondingly less on everything else*. Every interpretive sentence must carry that relative framing.
- **Phase 6 (clustering):** Euclidean distance is not well-defined on compositional data — the space is a simplex, not ℝ¹¹. Standard KMeans on raw shares will find structure partly driven by the closure constraint. A centred log-ratio transform before clustering is the standard fix, and the 4 zero-inflated features complicate it (the log of zero is undefined), which is a real design problem Phase 6 must solve rather than ignore.

The `Eating_Out_Share` × `Entertainment_Share` positive pair is worth noting precisely because it survives the closure pressure: two discretionary categories moving together against a background that pushes everything apart is a genuine behavioural signal.

### Cell 13 (code) — The leakage check (Q5)

Four tests, in increasing subtlety.

**Identity 1 — `Savings = INCOME − COTOTAL`:** max absolute error 5.82×10⁻¹¹, i.e. floating point. The target is definitionally an accounting identity, exactly as in the synthetic data.

**Identity 2 — does `COTOTAL` equal the sum of the 11 categories?** Median difference 0, 97.7% within 1%, but the maximum gap is ₹960,000. The gap is strictly one-directional (`COTOTAL ≥ reconstruction` for every household, never below) and affects 2.7% of the sample.

**Why the gap exists and why it is left in place:** the build script sums the 52 IHDS consumption items with `fillna(0)`, treating an unanswered item as zero spend. IHDS's own `COTOTAL` imputes some of those. The target uses IHDS's `COTOTAL` — the official aggregate, comparable to the published literature — while the shares are computed against the reconstruction so that they sum to exactly 1. Using one denominator for both would break one property or the other. The measured cost of this choice is **0.25 percentage points**: recomputing `Goal_Met` from the reconstruction gives 32.18% versus the published 31.93%, agreeing on 99.75% of households. That is small, but it is a real design decision and it is recorded here rather than left to be rediscovered.

**Test A — raw categories + `INCOME`:** reconstruct `Goal_Met` with **99.75%** agreement. Excluded.

**Test B — expense-to-income ratios:** identical **99.75%** agreement, because they are the same quantity divided through by income. Excluded. This is the test that forced the whole feature-set redesign: these were the Phase 2 features in the original project.

**Test C — composition shares:** sum to exactly 1.000000 (min = max), so they contain no information about the level of consumption relative to income and cannot reconstruct the target by construction. Empirically, the strongest correlation between any share and `Savings_Rate` is **0.056**. Safe.

**Why Test C reports both the algebraic argument and the empirical correlation:** the algebra proves reconstruction is impossible; the correlation shows the shares are not even a strong *statistical* proxy. A feature can be non-reconstructing but still so predictive that it amounts to leakage in practice. Checking both closes that gap.

**Test D — the behavioural `Has_*` columns:** these are survey-reported facts about holding savings instruments, entirely outside the consumption arithmetic. They are neither features nor leakage but a reserved external-validation set — the thing the synthetic dataset made impossible.

### Cell 15 (code) — Class balance and threshold sensitivity (Q6)

**Result:** 28,262 not-met vs 13,256 met — **31.93% positive, 2.13:1**.

**Why this single number invalidates most of Phase 4:** the synthetic target was 178:1 imbalanced with only 112 minority cases. Everything Phase 4 concluded followed from that: `class_weight="balanced"` everywhere, model selection on `recall_0`, the SVM winning because it was one of only two models reaching perfect minority recall, and the warning that `recall_0 = 1.0` rested on 112 rows. At 2.13:1 with 13,256 minority cases, none of that reasoning applies. Class weighting becomes optional rather than essential, macro-F1 and ROC-AUC become straightforwardly usable, and the winning-model argument must be re-made from scratch.

**Threshold sensitivity** — the benchmark is a convention, not a measurement:

| Threshold | Goal_Met rate | Imbalance |
| --- | --- | --- |
| 0% | 0.4411 | 1.27 : 1 |
| 10% | 0.3833 | 1.61 : 1 |
| **20%** | **0.3193** | **2.13 : 1** |
| 30% | 0.2532 | 2.95 : 1 |
| 40% | 0.1876 | 4.33 : 1 |

**Why this table is Phase 1 work rather than a Phase 4 robustness check:** the threshold is the one part of the target definition that was chosen rather than measured. Establishing early how much the problem changes across a plausible range tells later phases how much weight any single-threshold result can bear. A finding that holds at 10%, 20%, and 30% is a finding about households; one that only appears at 20% is a finding about the threshold.

**Survey weights:** unweighted 31.93%, weighted **30.47%**. The 1.5-point gap means the sample modestly over-represents goal-meeting households relative to the national population. Small, but it is the reason `WT` is carried in the dataset, and any nationally-framed claim in Phase 7 or 8 must apply it.

**The two gradients that will drive Phase 7:**

| Area type | Goal_Met | | Occupation | Goal_Met |
| --- | --- | --- | --- | --- |
| Metro urban | 0.4212 | | Salaried | **0.4410** |
| Other urban | 0.3698 | | Non-ag labour | 0.3171 |
| Developed village | 0.3091 | | Business | 0.2870 |
| Less developed village | 0.2666 | | Farm | 0.2570 |
| | | | Ag labour | 0.2512 |
| | | | No regular worker | 0.2503 |

Both are clean and monotone, and **both contradict the synthetic project's conclusions**. Geography runs the opposite way — metro households are the *most* likely to save, not the least. And occupation, which BONUS-Q2 proved was pure noise in the synthetic data, is here the strongest single categorical signal: salaried households meet the goal at 1.76× the rate of agricultural labourers.

---

## What this changes for later phases

| Phase | Consequence |
| --- | --- |
| **2 — Feature engineering** | Restate the ratio justification: collinearity with income is no longer the reason (r ≤ 0.43). Decide on `Unknown` vs imputation for `Caste_Group`. Consider explicit zero-indicators for the four majority-zero shares. Decide winsorisation for `Debt_To_Income`. |
| **3 — Baseline** | Majority baseline is **0.681 accuracy / 0.405 macro-F1**, not 0.994 / 0.499. |
| **4 — Model comparison** | **Rewrite.** The imbalance narrative is gone. Re-select a winning model on macro-F1/ROC-AUC without the `recall_0` constraint. Skew up to 112 means robust or log scaling, not plain standardisation. |
| **5 — Explainability** | Every share must be interpreted *relatively* because of compositional closure. |
| **6 — Clustering** | Euclidean KMeans on shares is not well-founded; needs a log-ratio transform, complicated by zero-inflation. |
| **7 — Business translation** | Both headline gradients reverse or newly appear. Lean on relative comparisons, not levels, because of the 55.9% income under-reporting. Apply `WT` for any national claim. |
| **Bonus** | BONUS-Q1 (rent as a deterministic tier key) and BONUS-Q2 (occupation is noise) are both retired — they described the generator, and the real data behaves oppositely on each. |
