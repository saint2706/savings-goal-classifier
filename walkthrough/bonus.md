# Bonus — Extension Questions

**Source:** [README § Bonus / extension questions](../README.md#bonus--extension-questions-not-required-for-core-deliverable)
**Notebook:** [`notebooks/08_bonus_extensions.ipynb`](../notebooks/08_bonus_extensions.ipynb)
**Builds on:** [Phase 1 — Data Understanding](phase1.md), [Phase 2 — Feature Engineering](phase2.md), [Phase 5 — Explainability](phase5.md)

Phases 0–8 answer one question — who is at risk of missing their savings goal, and why. The four bonus questions point in a different direction: they ask what *else* the dataset knows. Three of them are inverted prediction problems (can a demographic column be recovered from spending behaviour?), and one is a controlled-comparison question (does city tier matter for its own sake, or only because Tier-1 residents differ in some other way?).

These are labelled "not required for the core deliverable" in the README, and none of them changes the recommended model. But two of the four turn out to be diagnostics on the dataset itself, and one of those materially changes how Phase 5's explainability results should be read — so this document ends with what the bonus work sends back upstream.

**A note on what "answered" means here.** Three of the four questions get a *negative or trivial* answer. That is a real finding, not a failed analysis: it tells you the dataset's generating process is simpler than the questions assume. The work below is therefore aimed as much at establishing *why* each answer is what it is as at reporting the score.

---

## Research questions & answers

| #   | Question                                                                      | Answer                                                                                                                                                                                                                                                                                                                                                                        |
| --- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| BONUS-Q1 | Can `City_Tier` be predicted from spending mix alone?                    | **Yes — perfectly, and trivially.** Both a multinomial logistic regression and a random forest reach **1.0000 CV accuracy and 1.0000 macro-F1** (majority baseline: 0.5034). This is not a discovery about spending behaviour: `Rent_Ratio` is a *deterministic lookup* on tier — exactly 0.30 in Tier-1, 0.20 in Tier-2, 0.15 in Tier-3, with zero within-tier variance. `Rent_Ratio` alone scores 1.0000; the other ten ratios together score **0.4925, below the majority baseline**. |
| BONUS-Q2 | Can `Occupation` be predicted from income + expense pattern?             | **No — indistinguishable from chance.** Best model (random forest) reaches **0.2590 CV accuracy** against a 0.2510 majority baseline on four near-balanced classes; the out-of-fold confusion matrix is essentially uniform. Adding `Age` and `Dependents` does not help (0.2555). `Occupation` is assigned independently of every financial and demographic variable in this dataset — retirees and students span the full 18–64 age range alike. |
| BONUS-Q3 | Does a lifecycle pattern exist between age and discretionary spending?    | **No.** Discretionary share (`(Eating_Out + Entertainment) / Income`) is flat at **~7.0% in every age band** from 18–25 to 56–64. Regressing it on age gives **R² = 0.000015**; the quadratic term that would capture an inverted-U lifecycle profile is not significant (**p = 0.503**, nested F = 0.45), and Spearman ρ = −0.0030 (p = 0.67). Adding income, dependents, tier, and occupation controls leaves adjusted R² **negative**. |
| BONUS-Q4 | Does `City_Tier` still affect disposable income after controlling for income? | **Yes — strongly, and the mechanism is rent alone.** Adding tier to an income-only model lifts R² from **0.777 to 0.816** (nested F = 2140, p < 10⁻³⁰⁰); Tier-2 and Tier-3 residents hold **+₹4,164** and **+₹6,313** more disposable income than Tier-1 residents at the same income. On the scale-free outcome the effect is cleaner still: tier shifts the *share of income kept* by **+10.0 pp** (Tier-2) and **+15.2 pp** (Tier-3) while `log(Income)` is insignificant (p = 0.21). Decomposing the gap, **`Rent_Ratio` accounts for 98.7%** of the Tier-1 → Tier-3 expense difference; no other category differs by more than 0.2 pp. |

**Artifacts:** [`results/bonus_extension_summary.csv`](../results/bonus_extension_summary.csv), [`results/bonus_extensions.png`](../results/bonus_extensions.png).

---

## Notebook walkthrough

The notebook carries only section headers and code; the reasoning behind each step lives here.

### Cell 0 (markdown) — Title

### Cell 1 (code) — Imports, data load, and derived columns

Rebuilds the Phase 2 expense-to-income ratio columns (`{category}_Ratio` for each of the 11 expense categories) directly from `dataset/data.csv`, matching every prior phase's self-contained, rebuild-from-raw-CSV pattern, and adds three columns the bonus questions need:

- `Discretionary` / `Discretionary_Ratio` — `Eating_Out + Entertainment`, absolute and as a share of income (Q3).
- `Disposable_Ratio` — `Disposable_Income / Income`, the scale-free version of Q4's outcome.
- `Age_Band` — five bands (18–25, 26–35, 36–45, 46–55, 56–64) for the non-parametric part of Q3.

**Why `Goal_Met` is not rebuilt here:** none of the four bonus questions involves the target. That also means the project's leakage constraint doesn't bind in this notebook — `Disposable_Income` is used freely in Q4 because it is Q4's *outcome*, not a feature predicting `Goal_Met`. This is worth stating explicitly, because using that column anywhere in Phases 3–7 would be a leakage bug; here it is the thing being explained.

**Why discretionary spending is defined as eating out plus entertainment:** these are the two categories in the dataset that are unambiguously *elective*. Transport and utilities scale with a household but aren't chosen month to month; groceries are a necessity whose level (not existence) is discretionary. Restricting the definition to the two clearly-elective categories is the conservative choice — if a lifecycle effect exists anywhere, it should show up most strongly here. Q3 also tests each of the two components separately, so the answer doesn't depend on the decision to combine them.

### Cell 2 (code) — The `benchmark()` helper

```python
def benchmark(X, y, tag, with_dummies=True):
    models = [("Majority baseline", DummyClassifier(strategy="most_frequent")),
              ("Stratified baseline", DummyClassifier(strategy="stratified", random_state=42)),
              ("Multinomial LogReg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
              ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))]
```

**What it does:** runs 5-fold stratified cross-validation of a categorical target against two chance baselines and two real models, reporting mean accuracy and macro-F1 with their fold-to-fold standard deviations.

**Why two baselines and not one:** Q1 and Q2 both have imbalanced-or-balanced multiclass targets, and the two dummies fail in opposite directions. `most_frequent` sets the accuracy floor (0.5034 for `City_Tier`, whose largest class is Tier-2) but has a terrible macro-F1 (0.2232) because it never predicts the other two classes at all. `stratified` has a lower accuracy (0.3828) but a much higher macro-F1 (0.3338) because it does predict every class, just at random. Quoting only one of them would make a real model look better than it is on whichever metric that baseline happens to be weak on. Reporting both means a model has to beat the harder of the two on *each* metric to count as having learned anything.

**Why both a linear and a tree model:** the two questions have different failure modes to rule out. If a linear model fails but a random forest succeeds, the relationship is non-linear or interactive; if both fail, there is no learnable relationship of either kind. Q2's answer depends on that distinction — one model at chance would be inconclusive, both at chance is not.

**Why macro-F1 alongside accuracy:** `City_Tier` is imbalanced (5,934 / 10,068 / 3,998), so accuracy alone can be inflated by getting the majority class right. Macro-F1 weights all three tiers equally, which is the right question here — "can spending mix identify a tier" means *any* tier, not just the common one. This is the same reasoning Phase 4 uses to select the winning classifier on macro-F1 rather than accuracy.

**Why `StandardScaler` only for the logistic regression:** it is fitted inside a `Pipeline`, so the scaler is refit on each training fold rather than on the whole dataset — scaling on all 20,000 rows before cross-validation would leak fold-level distribution information into the training set. Tree models are invariant to monotone rescaling of individual features, so the forest needs no scaler.

### Cell 3 (markdown) — "BONUS-Q1 — Can `City_Tier` be predicted from spending mix alone?"

### Cell 4 (code) — Q1: the headline benchmark

Runs `benchmark()` with all 11 ratio columns predicting `City_Tier`, then produces an out-of-fold confusion matrix via `cross_val_predict` and the random forest's feature importances.

**Why "spending mix" is operationalised as the 11 ratios and not the raw rupee amounts:** the question asks about the *mix* — the shape of someone's spending — not its scale. Raw expense columns confound the two: a Tier-1 resident paying more rent in rupees might simply earn more. Dividing by income removes the level and leaves the composition, which is exactly what the question is about. It's also the Phase 2 feature form, so the answer speaks to the feature set the project actually models on.

**Result:** both real models score **1.0000 accuracy and 1.0000 macro-F1, with zero standard deviation across all five folds**, and the out-of-fold confusion matrix is perfectly diagonal — 5,934 / 10,068 / 3,998 correct, zero errors of any kind.

**Why a perfect score is a red flag rather than a result:** this project has already been through this once. Phase 1's leakage check exists precisely because a perfect or near-perfect score usually means the label has been reconstructed rather than predicted. A cross-validated 1.0000 with zero fold variance, from a *linear* model, on a three-class problem, is not what a genuine behavioural signal looks like — real signals produce fold-to-fold wobble and at least some confusions between adjacent classes. The `Rent_Ratio` importance of 0.9738 says where to look next.

### Cell 5 (markdown) — "Q1 follow-up — where does the perfect score come from?"

### Cell 6 (code) — Establishing the mechanism

Three checks, in increasing order of directness:

1. **Mean ratio by tier.** Every category is flat across tiers to three decimal places — `Groceries_Ratio` is 0.1253 / 0.1251 / 0.1255, `Transport_Ratio` is 0.0650 / 0.0650 / 0.0651 — except `Rent_Ratio`, which is 0.3000 / 0.2000 / 0.1500.
2. **Within-tier distribution of `Rent_Ratio`.** Standard deviation is **0.0** in all three tiers; min = 25th percentile = median = 75th percentile = max.
3. **Distinct values per tier.** Exactly one value each: `[0.3]`, `[0.2]`, `[0.15]`.

**Why this progression rather than jumping straight to the third check:** the first check is what you would run without a hypothesis, and it is what generates the hypothesis. The third check confirms it as a hard fact rather than an approximate tendency. Reporting the intermediate step matters because "Tier-1 residents *tend* to spend more of their income on rent" and "`Rent_Ratio` *is* an encoding of `City_Tier`" are very different claims with very different consequences, and only the second one is true here.

Two follow-up benchmarks then quantify the split:

- **Q1b — `Rent_Ratio` alone:** 1.0000 accuracy and macro-F1 for both models. One feature reproduces the entire result.
- **Q1c — the other ten ratios, `Rent_Ratio` removed:** logistic regression 0.5034 (identical to the majority baseline, to four decimals), random forest **0.4925 — *below* the majority baseline**.

**Why the forest scoring below the baseline is the strongest evidence here, not an anomaly:** when features carry genuinely zero signal, a flexible model does not tie the baseline — it loses to it. The forest has enough capacity to fit fold-specific noise, and that fitted noise generalises worse than the constant "always predict Tier-2" rule. A model that underperforms a constant predictor is a much sharper statement of "there is nothing here" than a non-significant p-value, because it demonstrates the cost of trying rather than merely failing to reject a null.

**What Q1's answer actually is:** yes, `City_Tier` is perfectly recoverable from spending mix — but only because the dataset's generator assigned rent as a fixed percentage of income by tier. The honest answer is "yes by construction, no behaviourally": strip the one deterministic column and the remaining spending mix carries no information about where someone lives at all.

### Cell 7 (markdown) — "BONUS-Q2 — Can `Occupation` be predicted from income + expense pattern?"

### Cell 8 (code) — Q2: the headline benchmark

Same structure as Q1, with `Income` added to the 11 ratios because the question explicitly names income as an input.

**Why `Income` enters as a raw level while the expenses enter as ratios:** the question is "income *and* expense pattern," which is two distinct pieces of information — how much someone earns, and how they allocate it. Keeping income as a level preserves the first; the ratios carry the second. Using raw expense amounts instead would make the eleven expense features largely restatements of income, and the model's failure would then be ambiguous between "occupation isn't predictable" and "the features were redundant."

**Result:** majority baseline 0.2510, stratified baseline 0.2496, logistic regression 0.2496, random forest 0.2590. The best model beats chance by 0.8 percentage points, against a fold-to-fold standard deviation of 0.0061 — roughly one standard deviation, on four near-balanced classes (5,011 / 5,019 / 4,967 / 5,003).

The out-of-fold confusion matrix makes this concrete: every cell sits near 1,250, the value expected if predictions were assigned at random. Per-class F1 ranges from 0.241 to 0.272 — all four occupations are equally unpredictable, so there is no single class carrying a usable signal that the aggregate is hiding.

**Why the feature importances are reported even though the model failed:** they diagnose *how* it failed. Nine of the twelve features sit between 0.0935 and 0.0947 — essentially 1/12 each, which is what impurity importance converges to when no feature is more informative than any other. The three below that (`Education_Ratio` 0.0806, `Loan_Repayment_Ratio` 0.0458, `Rent_Ratio` 0.0267) are lower simply because they have less within-dataset variance to split on — `Rent_Ratio`, as Q1 established, takes only three distinct values in the entire dataset. A near-uniform importance profile is the fingerprint of a model that found nothing, as opposed to one that found something weak.

### Cell 9 (markdown) — "Q2 follow-up — is the signal hiding in a demographic the question excluded?"

### Cell 10 (code) — Adding `Age` and `Dependents`

**Why run a model the question didn't ask for:** the most likely reason a real occupation classifier would fail on income and spending alone is that occupation is mostly a *demographic* fact — in any real population, "Student" and "Retired" are nearly determined by age. If adding `Age` rescued the model, the correct answer to Q2 would be "not from financial data, but occupation is recoverable from the dataset overall," which is a materially different conclusion. Testing it is what separates "the chosen features were wrong" from "the target is unpredictable."

**Result:** random forest 0.2555, logistic regression 0.2512 — no better, and the forest is marginally *worse* than without the demographics. The reason is in the table below it: mean age is 41.2 / 41.3 / 40.8 / 40.9 across Professional / Retired / Self-Employed / Student, and **every occupation spans the full 18–64 range**. The dataset contains 18-year-old retirees and 64-year-old students. Mean income is likewise flat (₹40,667–₹42,020).

**What Q2's answer actually is:** no. And the diagnostic upgrades the answer from "our features couldn't find it" to "there is nothing to find" — `Occupation` was drawn independently of every other column, so no feature set built from this dataset can predict it. This also retroactively explains a Phase 5 result: occupation dummies contributed essentially nothing to the SHAP rankings, and that was a property of the data, not a shortcoming of the model.

### Cell 11 (markdown) — "BONUS-Q3 — Does a lifecycle pattern exist between age and discretionary spending?"

### Cell 12 (code) — Non-parametric first pass

Reports discretionary spending by age band, then runs Spearman correlation (monotone trend against age as a continuous variable) and Kruskal–Wallis (any difference in distribution between the five bands) for four outcomes: absolute discretionary spending, the discretionary ratio, and each of its two components separately.

**Why non-parametric tests before the regression:** Phase 1 established that income in this dataset is heavily right-skewed (max ₹1.08M against a median near ₹41k). Any absolute-rupee outcome inherits that skew, which violates the normality assumption behind an OLS t-test. Spearman and Kruskal–Wallis rank-transform the data and so are unaffected — if they and the regression agree, the conclusion doesn't rest on a distributional assumption the data doesn't satisfy.

**Why four outcomes rather than one:** a lifecycle effect could plausibly show up in absolute spending but not in share of income (if older people simply earn more), or in eating out but not entertainment. Testing all four closes those escape routes, so a null result can't be attributed to having picked the wrong operationalisation.

**Result:** all four are null on both tests. Spearman |ρ| ≤ 0.0043 with p ranging 0.54–0.77; Kruskal–Wallis H between 1.10 and 3.98, p ranging 0.41–0.90. Band means are flat: the discretionary ratio is 0.0702, 0.0701, 0.0698, 0.0699, 0.0700 across the five bands — a total spread of 0.0004, or four hundredths of a percentage point.

### Cell 13 (code) — Parametric confirmation and the lifecycle-shape test

Fits three nested OLS models and an F-test between the first two.

**Why the quadratic term is the model that matters:** the economic hypothesis behind Q3 is not "discretionary spending rises with age" — it is that discretionary spending is *hump-shaped*: low for students, peaking in mid-career, falling in retirement. A linear model cannot represent that shape at all, so a null linear coefficient would not be evidence against a lifecycle pattern. Adding `Age²` gives the model the ability to express exactly that inverted U, which makes its coefficient the direct test of the hypothesis as stated.

**Result:**

| Model | R² | adj. R² | Verdict |
| --- | --- | --- | --- |
| `Discretionary_Ratio ~ Age` | 0.000015 | −0.000035 | Age coefficient p = 0.58 |
| `+ I(Age**2)` | 0.000037 | −0.000063 | `Age²` p = 0.503; nested F = 0.45 |
| `+ log(Income) + Dependents + C(City_Tier) + C(Occupation)` | 0.000392 | −0.000058 | Every coefficient p > 0.10 |

**Why the negative adjusted R² is the most informative number in the table:** adjusted R² penalises each added parameter, and it goes negative when a model explains less variance than the penalty for its own complexity. All three models are here — even the one with income, dependents, tier, and occupation controls. The full model explains 0.04% of the variation in discretionary share; a model with no predictors at all would do better once the parameter cost is charged. This isn't a weak effect that a larger sample would resolve: with n = 20,000 the study is well powered, and the confidence interval on `Age` (−0.000119 to +0.000052) rules out any effect larger than roughly a hundredth of a percentage point of income per year of age.

**What Q3's answer actually is:** no lifecycle pattern exists in this dataset. Consistent with Q1 and Q2, the reason is generative rather than behavioural — discretionary spending was drawn as a fixed ~7% share of income independent of age. The substantive economics question ("do people spend differently on leisure at different life stages?") is a real and well-studied one, but **it cannot be answered with this dataset**, and the flat result should be reported as a limitation of the data rather than as evidence about people.

### Cell 14 (markdown) — "BONUS-Q4 — Does `City_Tier` still affect disposable income after controlling for income level?"

### Cell 15 (code) — Nested regressions

Reports raw means by tier, then fits four models of increasing specification and an F-test between the first two.

**Why the raw means come first:** they establish that there is something to explain, and they pre-empt the obvious confound. Mean income is essentially identical across tiers (₹41,068 / ₹41,713 / ₹42,031) while mean disposable income differs enormously (₹7,156 / ₹11,486 / ₹13,718). Tier-1 residents earn the same and keep 40% less. That framing matters: the control for income is not expected to explain the gap away, and knowing that in advance makes the regression a confirmation rather than a fishing expedition.

**Result:**

| Model | R² | Tier-2 effect | Tier-3 effect |
| --- | --- | --- | --- |
| `~ Income` | 0.777074 | — | — |
| `~ Income + C(City_Tier)` | 0.816381 | +₹4,163.62 | +₹6,313.27 |
| `~ Income + C(City_Tier) + Dependents + Age + C(Occupation)` | 0.821733 | +₹4,147.54 | +₹6,301.23 |
| `~ Income * C(City_Tier)` | 0.854749 | slope +0.1055 | slope +0.1573 |

Nested F-test of tier over income alone: **F(2, 19996) = 2140.2, p ≈ 0**.

**Why the demographics model is the one that answers the question:** the question asks whether tier matters *for its own sake*. The threat to that claim is that tier is a proxy for something else — that Tier-1 residents have more dependents, are older, or work in different occupations. Adding those controls moves the Tier-3 coefficient from ₹6,313 to ₹6,301, a change of 0.2%. The tier effect is not a demographic effect in disguise. (`Dependents` is separately significant at −₹605 each, and `Age` and `Occupation` are not — consistent with Q2's finding that occupation is noise in this dataset.)

**Why the interaction model is included:** the first three models constrain tier to shift disposable income by a fixed rupee amount regardless of earnings, which is the wrong functional form if tier acts on a *proportion* of income. The interaction lets each tier have its own slope, and it fits substantially better (R² 0.855 vs 0.816). The slopes — **0.172 in Tier-1, 0.277 in Tier-2, 0.329 in Tier-3** — say that a Tier-3 resident keeps 33 paise of each additional rupee earned where a Tier-1 resident keeps 17. Note that the tier *intercepts* become insignificant once the interaction is present (p = 0.09 and 0.11): the effect is entirely multiplicative, which is the signature of a percentage-of-income mechanism and sets up the next cell.

### Cell 16 (code) — Scale-free model, mechanism decomposition, and the identity check

**The scale-free model** (`Disposable_Ratio ~ log(Income) + C(City_Tier)`) restates the finding in the units the interaction model implied: Tier-2 keeps **+10.02 pp** and Tier-3 **+15.19 pp** more of their income than Tier-1 (both p < 10⁻³⁰⁰), while `log(Income)` is **insignificant (p = 0.212)**. Once the outcome is expressed as a share, income level stops mattering entirely and tier explains all of the systematic variation.

**Why this is the cleanest statement of Q4's answer:** the question is "does tier still matter after controlling for income." Here the control is not merely insignificant — it is *irrelevant*, because the mechanism operates in percentage terms and the ratio outcome has already removed the scale. A reader who wants one number should take +15.2 pp, not +₹6,313, because the rupee figure is an average over a skewed income distribution and doesn't transfer to any individual.

**The mechanism decomposition** subtracts Tier-3's mean expense ratios from Tier-1's, category by category. `Rent_Ratio` differs by **0.15000**; the next largest is `Loan_Repayment_Ratio` at 0.00162, and eight of the eleven categories differ by less than 0.0003 in either direction. Rent is **98.7%** of the total gap.

**The identity check** confirms `Disposable_Income == Income − Σ(expenses)` to a maximum absolute error of **8.73 × 10⁻¹¹** — floating-point noise.

**Why the identity check belongs in the answer and not just the appendix:** it bounds what the regression is entitled to claim. Because disposable income is an exact accounting function of the expense columns, the tier effect cannot be a discovered causal relationship — it is arithmetic propagating from wherever tier enters the expense side. The decomposition then locates that entry point precisely: rent, and essentially nothing else. So the defensible statement of Q4's answer is not "living in a Tier-1 city causes lower disposable income" but "**in this dataset, tier affects disposable income entirely and exclusively through rent's share of income, by construction**" — which is the same generative fact Q1 uncovered from the other direction.

### Cell 17 (markdown) — "Summary table and figure"

### Cell 18 (code) — The summary artifact

Builds a four-row table (question id, question, one-line answer, key evidence) and writes it to `results/bonus_extension_summary.csv`. Every numeric value in the `key_evidence` column is interpolated from the fitted objects rather than typed in, so the CSV cannot drift out of sync with the notebook if the analysis is re-run — the same rule the earlier phases follow for `results/model_comparison.csv` and `results/persona_profiles.csv`.

### Cell 19 (code) — The four-panel figure

Writes `results/bonus_extensions.png`, one panel per question, each chosen to make its answer legible without reading a table.

- **Q1 — strip plot of `Rent_Ratio` by tier.** A strip plot rather than a box plot or bar chart because the point *is* the absence of spread: three flat lines with no width show determinism in a way a box plot (which would render as three degenerate lines anyway, but reads as "a summary of a distribution") does not.
- **Q2 — horizontal bars with a chance reference line.** The models are plotted alongside the dummies against a dashed line at 0.25. The visual claim is that the bars are indistinguishable, so both baselines are kept in frame rather than removed as uninteresting.
- **Q3 — age-band means with 95% confidence intervals.** The y-axis is deliberately zoomed to 0.065–0.075 rather than starting at zero. A zero-based axis would render the series as a flat line by construction and prove nothing; zooming to a range where a real lifecycle effect *would* be visible, and showing the error bars overlapping anyway, is the stronger null result.
- **Q4 — raw vs income-adjusted disposable share by tier, paired bars.** Adjusted values are the scale-free model's predictions at mean log-income. The two bars per tier being visually identical is the whole finding: controlling for income changes nothing.

---

## What the bonus work sends back upstream

Three of the four answers are negative or trivial, but two of them are load-bearing for how the core deliverable should be read.

**1. `Rent_Ratio` and `City_Tier` are the same variable (from Q1).** This is the most consequential finding in this notebook. `Rent_Ratio` is a deterministic three-valued encoding of `City_Tier`, which means any model given both features is given one piece of information twice. Three downstream consequences:

- **Phase 5's SHAP rankings should be read with this collinearity in mind.** `City_Tier_Tier_1` appears third in the SHAP importance ranking, but its explanatory content is perfectly duplicated by `Rent_Ratio`. SHAP splits credit between exactly-collinear features rather than assigning it to one, so the true importance of "living in a Tier-1 city" is the *sum* of those two contributions, and each individually understates it. The dominance of `Loan_Repayment_Ratio` is unaffected — it is not collinear with either — but the gap between first and third place is narrower than the chart suggests.
- **Phase 6's persona result gains a mechanical explanation.** Phase 6 already notes that its clusters are driven almost entirely by `Education_Ratio` and `Rent_Ratio`, and that the personas are "a demographic/geographic split already present in the raw data." Q1 supplies the reason: clustering on a feature set containing `Rent_Ratio` is *partly clustering directly on `City_Tier`*, because that feature takes exactly three values. The low silhouette scores (< 0.1) and the near-categorical persona boundaries are what a three-valued feature does inside a distance-based method.
- **Phase 7's headline targeting rule is safe, but for a narrower reason than it appears.** "All 112 at-risk individuals live in Tier-1 cities" remains true, and it remains actionable. But Q1 and Q4 together show that "Tier-1" in this dataset means precisely "pays 30% of income in rent" and nothing else — no other spending category varies by tier. The recommendation should be understood as targeting a rent burden, which happens to be labelled as a city tier, rather than as a finding about urban living generally.

**2. `Occupation` is noise (from Q2).** It was drawn independently of every financial and demographic column. This confirms rather than challenges Phase 5 — occupation dummies contributed almost nothing to the SHAP rankings, and Q2 shows that was a property of the data. It also justifies Phase 7's decision to target by city tier rather than occupation on grounds stronger than the observed at-risk composition: occupation *cannot* carry targeting signal in this dataset, regardless of sample size or model.

**3. Q3 is a limitation of the data, not a finding about people (from Q3).** Discretionary spending was generated as a fixed share of income independent of age. Reporting "no lifecycle pattern exists" without that qualifier would misrepresent a generative artifact as a behavioural result.

**Taken together,** all four bonus questions point at the same underlying fact: this dataset's non-target columns were largely generated independently of one another, with expense categories set as fixed percentages of income and only `City_Tier` (via rent) and `Dependents` (via education) introducing systematic structure. That sharpens the README's existing "may be synthetically generated" limitation from a caveat about *absolute figures* into a specific, evidenced claim about *which relationships in the data are real* — and it is the reason the core model's predictive success should be read as a demonstration of correct methodology on a well-behaved dataset, not as evidence about Indian household finance.
