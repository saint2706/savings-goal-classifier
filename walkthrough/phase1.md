# Phase 1 — Data Understanding

**Source:** [README § Phase 1 — Data Understanding](../README.md#phase-1--data-understanding)
**Notebook:** [`notebooks/01_eda_and_leakage_check.ipynb`](../notebooks/01_eda_and_leakage_check.ipynb)
**Builds on:** [Dataset construction](dataset_construction.md)

Six research questions answered on 41,518 IHDS-II households. Three of the answers constrain decisions later phases would otherwise make wrongly. Everything here is read-only exploration; no features are engineered and no leakage column is ever fed to a model.

> **In plain terms — what "EDA" is for.** **Exploratory data analysis** is the stage of looking at the data before modelling it: what is in each column, what shape the numbers take, what is missing, what is broken. It produces no model and no prediction. Its value is entirely in the decisions it prevents later — nearly every "we cannot do X" in Phases 2 through 7 traces back to something measured on this page. The technical vocabulary introduced in [Dataset construction](dataset_construction.md) (feature, target, leakage, median, correlation, ROC-AUC) is assumed from here on.

---

## Research questions & answers

| # | Question | Answer |
| --- | --- | --- |
| 1 | What does each column mean, and what unit/time period does it represent? | All money columns are **annual** ₹. The grain is a **household**, not an individual. Of 50 columns: 6 identifiers, 1 unapplied survey weight, 19 features, 6 behavioural columns reserved for external validation, 13 leakage-excluded, 1 target. Single cross-section, 2011-12. |
| 2 | Distribution of income, expenses, and savings — skew, outliers, implausible values? | Extreme. `INCOME` skew **15.8**; `Clothing_Footwear` skew **112.3**. The headline: **55.9% of households report consumption exceeding income**, and 22.0% spend more than twice their income. Median savings rate is **−10.8%**. Categories are heavily zero-inflated: 90.5% record no rent, 73.7% no insurance, 72.0% no eating out. |
| 3 | Missing values or duplicate rows, and how are they handled? | Missingness now exists but is negligible: `Has_*` columns 0.32–0.42%, `Caste_Group` 0.21%, `Religion` 0.03%, `Max_Adult_Education` 0.01%. **Zero duplicate households, zero duplicate rows.** 634 households (1.5%) were dropped upstream for `INCOME <= 0` or missing consumption. |
| 4 | How correlated are expense categories with income and with each other? | **Weakly, and that is the single most important finding here.** Raw categories correlate with income at only r = 0.09–0.43, and mean \|r\| *between* categories is **0.143**. Among the composition shares, `Groceries_Share` correlates **−0.266** with log income — Engel's law, appearing unprompted. |
| 5 | Is the target mathematically derivable from any candidate feature (leakage check)? | **Yes, from two representations, and both are excluded.** `Savings = INCOME − COTOTAL` holds exactly (max error 5.8×10⁻¹¹). Raw rupee categories reconstruct `Goal_Met` with **99.75%** agreement, and expense-to-income ratios with the same **99.75%**. The composition shares do not: they sum to exactly 1.000 for every household, and the strongest correlation between any share and `Savings_Rate` is **0.056**. |
| 6 | Class balance once leakage columns are excluded? | **31.93% positive, a 2.13:1 imbalance.** Survey-weighted the rate is 30.47%. The balance is highly sensitive to the arbitrary threshold, running from 44.1% at 0% to 18.8% at 40%. |

---

## Notebook walkthrough

The notebook carries only section headers and code; the reasoning lives here.

### Cell 1 (code) — Imports, load, and the column-role constants

Loads `dataset/households.csv` and declares four column groups up front: the 11 expense categories, their `_Share` counterparts, the behavioural `Has_*` columns, and `LEAKAGE_COLS`.

**Why the leakage list is a named constant in the first cell rather than derived later:** [dataset construction](dataset_construction.md) already established which representations reconstruct the target, so the exclusion set is known before any analysis runs. Declaring it up front makes it impossible for an intermediate cell to accidentally treat a leakage column as a feature. Cell 13 then *re-verifies* the list rather than establishing it.

### Cell 3 (code) — Column inventory (Q1)

Builds a table assigning every column a role and unit.

**Why the roles are written down as data:** there are 50 columns in six distinct roles, including one that is easy to mishandle — the `Has_*` behavioural indicators, which are neither features nor leakage but a reserved validation set. Writing the roles down as data (rather than prose in the walkthrough) means later phases can import the grouping instead of re-deriving it.

**Two unit changes that will break any copied code:**

- **Money is annual, not monthly.** IHDS reports `INCOME` and `COTOTAL` as annual rupees, so any threshold or axis label assuming a monthly figure is off by 12×.
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

> **In plain terms — skew.** **Skew** measures how lopsided a distribution is. Zero means symmetric — as many values above the middle as below, at similar distances. A positive skew means the values pile up on the left with a thin tail stretching right: most households spend a little, a few spend enormously. As a rough guide, anything above about 1 is noticeably lopsided and above 5 is severe. **112 is extraordinary** — it means the picture is essentially "everyone here, plus a handful out there." One household buying a motorbike inside an annual-recall category is enough to produce it.
>
> Why it matters practically: several methods assume values are spread out roughly evenly. **Standardisation** (rescaling a column by its average and spread so all columns are comparable) works fine on symmetric data and badly here, because the average and the spread are both being set by the outliers. The fixes are a **log transform** (compress the tail, as `Log_Income` does) or **robust scaling** (rescale using the median and the middle 50% instead, so the extremes cannot distort the scale). [Phase 2](phase2.md) tests which is needed.

Skews up to 112 are produced by a handful of households reporting a single very large purchase in an annual-recall category. This is normal for survey expenditure data and is why `Log_Income` exists as a feature — but it means any distance-based or linear method in Phases 4 and 6 needs either the log transform or robust scaling rather than plain standardisation.

**Why savings-rate *percentiles* are reported rather than just the mean and sd:** the mean (−1.16) and standard deviation (13.08) are uninterpretable here, because the distribution has a long left tail reaching −1647 (a household consuming 1,648× its reported income). The percentiles are readable and tell the actual story: the 1st percentile is −15.73, the median is −0.108, the 75th is +0.305.

### Cell 6 (code) — Implausible values and zero-inflation (Q2, part 2)

**The headline data-quality finding:**

| Check | Households | Share |
| --- | --- | --- |
| Consumption exceeds income | 23,204 | **55.89%** |
| Spends more than 2× income | 9,137 | 22.01% |
| Spends more than 6× income | 1,635 | 3.94% |

This check flags the **majority of the sample** — which is why it is treated below as a property of the survey rather than as rows to clean.

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

> **In plain terms — zero-inflation.** A column is **zero-inflated** when a large block of rows are exactly zero rather than merely small. That is not the same as a low average: 90.5% of households do not pay a *little* rent, they pay *none*, because they own their home. The distribution is really two populations glued together — a group where the question does not apply at all, and a group with a genuine amount.
>
> This causes trouble for models that assume a smooth sliding scale, because the step from ₹0 to ₹1 of school fees is a change of *kind* (this household now pays fees) while the step from ₹5,000 to ₹5,001 is a change of *degree*. Averages across such a column describe nobody. [Phase 2](phase2.md)'s fix is to split the two questions apart: a yes/no column for "does this household spend on X at all", alongside the amount.

**Why `Rent_Share` at 90.5% zero deserves special attention:** most Indian households own their homes, so rent is absent for nine in ten. Housing cost is the intuitive first place to look for budget strain, and here it is simply not observable for the overwhelming majority. Any analysis leaning on rent is describing a ~10% urban subsample while appearing to describe the whole.

Four of the eleven features being majority-zero also means these are **semi-continuous** variables — a binary "does this household spend on X at all" mixed with a continuous amount. Phase 2 should consider modelling that explicitly (a paired indicator plus amount) rather than feeding a spike-at-zero distribution to a linear model.

> **In plain terms — the words in that sentence.** **Binary** = takes only two values, yes or no. **Continuous** = takes any value on a sliding scale. **Semi-continuous** = both at once, which is the awkward case described above. A **spike at zero** is what that looks like when drawn: a tall bar sitting on zero, then a low spread of real amounts beside it. A **linear model** is one that assumes the outcome moves by a fixed step for each unit of a feature — a straight-line relationship — and a spike at zero is precisely the shape a straight line cannot describe.

### Cell 8 (code) — Missingness and duplicates (Q3)

**Result:** nine columns have missing values, all under 0.5%; zero duplicate household identifiers; zero duplicate rows.

**Why this is still worth a cell when the answer is "almost none":** the *pattern* of missingness is informative even when the volume is negligible. the six `Has_*` columns cluster at 0.32–0.42% missing, which is the signature of a small number of households that skipped the debt-and-investment block entirely, not of random item non-response. Because those columns are reserved for validation rather than used as features, the missingness needs no imputation strategy — but the affected households must be dropped from validation comparisons rather than treated as "no".

`Caste_Group` (0.21%) and `Religion` (0.03%) do need a decision in Phase 2, since they are model inputs. At this rate, either most-frequent imputation or an explicit `Unknown` level is defensible; an explicit level is preferable because refusal to state caste is plausibly informative rather than random.

> **In plain terms — missingness and imputation.** Models cannot be handed a blank cell, so every gap needs a decision. **Imputation** means filling the gap with a guess — most commonly the column's most frequent value or its average. The alternative is to stop treating the gap as a gap and make **"Unknown" its own category**, which keeps the fact of the refusal visible instead of overwriting it with a caste the household declined to give.
>
> The choice hinges on *why* the value is missing. If it is missing at random, imputing is harmless. If the missingness itself carries meaning — people who refuse to answer differ from people who answer — then imputing destroys real information. [Phase 2](phase2.md) measures which case this is rather than assuming.

### Cell 10 (code) — Correlation with income (Q4, part 1)

**The most consequential comparison in this document:**


Household spending is only loosely tied to income here (strongest: `Groceries` at 0.43; weakest: `Rent` at 0.09), so the raw columns carry substantial information that income does not.

**Why this constrains Phase 2's central decision:** converting expenses to ratios is often justified as *removing redundancy* — when raw columns are so collinear with income that they carry little independent signal. That argument does not apply here; the raw columns are genuinely informative and mutually distinct. Ratios and shares remain the right representation, but the justification has to be comparability across a 100× income range and, for shares specifically, leakage avoidance.

**The share correlations recover a real economic law:**

| Feature | r with `Log_Income` |
| --- | --- |
| `Transport_Share` | **+0.273** |
| `Insurance_Share` | +0.251 |
| `Education_Share` | +0.179 |
| `Healthcare_Share` | −0.128 |
| `Groceries_Share` | **−0.266** |

> **In plain terms — reading these signs.** A **positive** correlation means the two rise together: richer households put a larger slice of their budget into transport (+0.273) and insurance (+0.251). A **negative** correlation means one rises as the other falls: richer households put a *smaller* slice into food (−0.266). The size (0.27, not 0.9) says the relationship is real but loose — a tendency across thousands of households, not a rule you could apply to any individual one.
>
> **Engel's law** is the observation, first published in 1857, that as a household's income grows, the *proportion* of its budget spent on food falls — even though the rupees spent on food usually rise. It is one of the most reliably reproduced findings in economics. Nothing in this pipeline was designed to produce it; it simply appears when the shares are computed, which is good evidence the shares are measuring something economically real rather than an artefact of the arithmetic.

Food's budget share falling as income rises is **Engel's law**, one of the oldest empirical regularities in economics, and it appears here without being engineered in. It is the clearest single piece of evidence that the composition shares carry real economic structure rather than arithmetic noise.

### Cell 11 (code) — Correlation among the shares, and the closure problem

**Result:** mean off-diagonal correlation **−0.064**, with **65.5% of all pairs negative**. Strongest negative: `Groceries_Share` × `Healthcare_Share` (−0.362). Strongest positive: `Eating_Out_Share` × `Entertainment_Share` (+0.110).

> **In plain terms — correlation matrix, off-diagonal.** A **correlation matrix** is a grid holding the correlation of every column with every other column. Down its diagonal sits each column paired with itself, which is always exactly 1 and tells you nothing — so the informative entries are the **off-diagonal** ones, and "mean off-diagonal correlation" is just the average over all the genuine pairs.

**Why a predominantly negative correlation matrix is expected rather than a finding:** the shares sum to exactly 1 for every household. That constraint — *compositional closure* — forces the components to be negatively correlated on average, because one share can only rise if others fall. The negative correlations are therefore partly an artifact of the representation, not evidence that households trade groceries off against healthcare.

> **In plain terms — compositional closure.** Think of the eleven shares as slices of one pie. Because the pie is always exactly one whole, **any slice can only grow by taking room from the others** — that is *closure*. So if you measured thousands of pies you would find slice sizes moving against each other on average, even if the bakers were cutting them completely at random. The negative correlations here are largely that geometry, not a behavioural discovery: they do not show households choosing food over healthcare. Data of this kind — proportions that must sum to a fixed whole — is called **compositional data**, and it needs its own toolkit, which is what the next paragraph is warning about.

**What this obliges later phases to do:**

- **Phase 5 (explainability):** a share's coefficient or SHAP value is never "the effect of spending more on X." It is the effect of spending more on X *and correspondingly less on everything else*. Every interpretive sentence must carry that relative framing.
- **Phase 6 (clustering):** Euclidean distance is not well-defined on compositional data — the space is a simplex, not ℝ¹¹. Standard KMeans on raw shares will find structure partly driven by the closure constraint. A centred log-ratio transform before clustering is the standard fix, and the 4 zero-inflated features complicate it (the log of zero is undefined), which is a real design problem Phase 6 must solve rather than ignore.

> **In plain terms — the vocabulary in that second bullet.**
> - A **coefficient** is the number a linear model attaches to a feature: "each extra unit of this moves the prediction by that much."
> - **SHAP** is a method for splitting a single prediction into per-feature contributions — "income pushed this household up, family size pulled it down." [Phase 5](phase5.md) explains it properly.
> - **Clustering** means letting an algorithm group similar households together without being told what the groups are. **KMeans** is the standard workhorse: it puts households near each other into the same group.
> - "Near each other" requires a definition of distance. **Euclidean distance** is the everyday one — straight-line distance, Pythagoras extended to many columns. **ℝ¹¹** is notation for "ordinary space with eleven free axes", where every combination of eleven numbers is possible.
> - A **simplex** is the restricted space our shares actually live in: eleven numbers that must be non-negative *and* must sum to 1. That is not ordinary space — the eleventh value is fixed the moment the first ten are known, so one of the eleven axes is not free. Straight-line distance measured as if all eleven were free therefore mismeasures how different two budgets are.
> - A **log-ratio transform** is the standard repair: instead of the shares themselves, work with the logarithms of ratios between them, which converts the constrained simplex into an ordinary space where straight-line distance behaves. Its snag is that **the logarithm of zero does not exist**, and four of our categories are zero for most households — which is the real design problem [Phase 6](phase6.md) has to solve.

The `Eating_Out_Share` × `Entertainment_Share` positive pair is worth noting precisely because it survives the closure pressure: two discretionary categories moving together against a background that pushes everything apart is a genuine behavioural signal.

### Cell 13 (code) — The leakage check (Q5)

Four tests, in increasing subtlety.

**Identity 1 — `Savings = INCOME − COTOTAL`:** max absolute error 5.82×10⁻¹¹, i.e. floating point. The target is definitionally an accounting identity, which is the root of every leakage risk below.

**Identity 2 — does `COTOTAL` equal the sum of the 11 categories?** Median difference 0, 97.7% within 1%, but the maximum gap is ₹960,000. The gap is strictly one-directional (`COTOTAL ≥ reconstruction` for every household, never below) and affects 2.7% of the sample.

**Why the gap exists and why it is left in place:** the build script sums the 52 IHDS consumption items with `fillna(0)`, treating an unanswered item as zero spend. IHDS's own `COTOTAL` imputes some of those. The target uses IHDS's `COTOTAL` — the official aggregate, comparable to the published literature — while the shares are computed against the reconstruction so that they sum to exactly 1. Using one denominator for both would break one property or the other. The measured cost of this choice is **0.25 percentage points**: recomputing `Goal_Met` from the reconstruction gives 32.18% versus the published 31.93%, agreeing on 99.75% of households. That is small, but it is a real design decision and it is recorded here rather than left to be rediscovered.

**Test A — raw categories + `INCOME`:** reconstruct `Goal_Met` with **99.75%** agreement. Excluded.

**Test B — expense-to-income ratios:** identical **99.75%** agreement, because they are the same quantity divided through by income. Excluded. This is the test that forced the whole feature-set redesign: these were the Phase 2 features in the original project.

**Test C — composition shares:** sum to exactly 1.000000 (min = max), so they contain no information about the level of consumption relative to income and cannot reconstruct the target by construction. Empirically, the strongest correlation between any share and `Savings_Rate` is **0.056**. Safe.

**Why Test C reports both the algebraic argument and the empirical correlation:** the algebra proves reconstruction is impossible; the correlation shows the shares are not even a strong *statistical* proxy. A feature can be non-reconstructing but still so predictive that it amounts to leakage in practice. Checking both closes that gap.

> **In plain terms — two different kinds of proof.** The **algebraic** argument is a proof on paper: because the shares always total 1, no formula built from them can recover how large spending was relative to income. That settles the matter for *every possible* dataset. The **empirical** check looks at these particular 41,518 households and asks whether the shares happen to track the answer closely anyway — they do not, the strongest link being a feeble 0.056. The distinction matters because a feature can be innocent in theory and still be a giveaway in practice (a **proxy** — a stand-in that carries nearly the same information by a different route). Running both tests is how you rule out each failure separately.

**Test D — the behavioural `Has_*` columns:** these are survey-reported facts about holding savings instruments, entirely outside the consumption arithmetic. They are neither features nor leakage but a reserved external-validation set, letting the normative target be checked against real saving behaviour.

### Cell 15 (code) — Class balance and threshold sensitivity (Q6)

**Result:** 28,262 not-met vs 13,256 met — **31.93% positive, 2.13:1**.

**Why this single number shapes Phase 4:** at 2.13:1 with 13,256 minority cases, this is an ordinary near-balanced problem. Class weighting is optional rather than essential, and macro-F1 and ROC-AUC are straightforwardly usable — none of the contortions a severe imbalance would force (resampling, selecting on minority recall alone, treating a handful of minority rows as the whole result) are needed here.

> **In plain terms — the imbalance toolkit we do *not* need.** When one class is genuinely rare, models tend to ignore it (predicting "no" always is nearly free), and practitioners reach for workarounds: **class weighting** tells the model that mistakes on the rare class hurt more; **resampling** duplicates rare rows or discards common ones to even the counts. Both distort the data or the fitted probabilities, so they are a cost, not a free improvement. With 13,256 households in the smaller class — a large number in absolute terms — there is no starvation to fix here, and [Phase 4](phase4.md) can mostly leave these tools alone.

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

Both are clean and monotone, and both run against a common intuition. Metro households are the *most* likely to save, not the least — urban living does not strain the budget once income is in view. And occupation is the strongest single categorical signal: salaried households meet the goal at 1.76× the rate of agricultural labourers.

> **In plain terms — monotone, and a warning about these two tables.** **Monotone** means the values move in one direction the whole way down without doubling back: 0.42, 0.37, 0.31, 0.27. Ragged orderings usually mean noise; a clean staircase like this one usually means a real underlying gradient.
>
> But note what these tables are *not*. Each column is **unconditional** — it compares metro households with village households as they actually are, differences in income included. It cannot tell you whether living in a metro *causes* higher saving, because metro households also earn more. Separating those requires holding income constant, which is exactly what [Phase 5](phase5.md) does — and it finds the geographic gradient is largely the income gradient wearing a different label.

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
