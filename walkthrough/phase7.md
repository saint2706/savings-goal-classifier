# Phase 7: Business Translation

**Source:** [README § Phase 7: Business Translation](../README.md#phase-7-business-translation)
**Notebook:** [`notebooks/07_business_translation.ipynb`](../notebooks/07_business_translation.ipynb)
**Builds on:** [Phase 1](phase1.md) (leakage check, headline feature set), [Phase 3](phase3.md) (income rule), [Phase 4](phase4.md) (tuned parameters in `model_final.json`)
**Artifacts:** `results/business.json`, `results/business_recommendations.csv`, `results/business_translation.png`

This phase turns model scores into statements a savings-product or financial-inclusion team could act on. Every household gets an out-of-fold score, targeting is measured against the best any ranking could achieve and against a plain income rule, and every peer comparison carries a PSU-bootstrap interval. The result: the headline model ranks households slightly better than "poorest first", the probabilities are well calibrated, and the peer-spending comparison cannot tell a team how much of the shortfall a budgeting product could close.

---

## Research questions & answers

| # | Question | Answer |
| --- | --- | --- |
| 1 | Which 3–5 findings should a team act on, stated as recommendations? | Five, in `results/business_recommendations.csv` and listed [below](#the-five-recommendations). The lead one: prioritise by income and let the model re-rank. At a 25% contact budget the headline model reaches 36.2% of at-risk households (98.5% of the 36.7% ceiling) against 35.0% for the income rule (95.4%): +1.1 pp, 95% CI 1.0 to 1.3. |
| 2 | Which expense category carries the most recoverable spend? | None can be named from this data. In rupees, at-risk households out-spend same-income on-track peers in all 11 categories, which follows from the label. By budget share the largest positive excesses are healthcare +4.4 pp (CI 3.8 to 4.9), education +2.8 (2.7 to 3.0), miscellaneous +2.1 (1.7 to 2.5) and transport +1.8 (1.5 to 2.1); groceries run the other way at −9.0 pp (−9.7 to −8.3). Whether the gap is "closable" depends on the benchmark: 92.8% (rupee) vs 22.3% (share). |
| 3 | Where does the model fail, and what must a stakeholder be told? | Calibration holds across the range (out-of-fold ECE 0.006, max bin error 0.013). Accuracy is 0.978 in the poorest decile, falls to 0.662 in decile 7 and recovers to 0.833 in the richest. 22.0% of households report spending more than twice their income and make up 32.3% of the at-risk group, so the at-risk count is not a prevalence estimate. |

---

## Notebook walkthrough

### Cell 1: Out-of-fold scores for every household

Loads the engineered features (`savings_goal.io.load_features`), the feature spec (`savings_goal.features.engineer.spec_from_frame`) and the tuned parameters from Phase 4 (`savings_goal.models.pipeline.load_model_final`). The 11 raw rupee categories are joined back from the household file because the peer benchmarks later need rupees as well as shares.

Two XGBoost pipelines are built with `savings_goal.models.pipeline.xgb_pipeline`: the headline model on the 13 deployable features (income, demographics, debt, social group) and, for comparison, the full model on all 29 features including spending shares. `savings_goal.evaluation.cv.oof_predict_proba` scores each household with a model fitted on folds that excluded its whole PSU, using the same `StratifiedGroupKFold(5)` grouped by `IDPSU` as Phases 3 to 5.

Four ranking strategies are then defined: the headline model's at-risk score (1 − P(on track)), the full model's, income ascending (poorest first, with a 1e-6 random jitter so ties break evenly) and random contact. Of 41,518 households, 28,262 are at risk: 68.1% unweighted, 69.5% survey-weighted.

> **In plain terms: out-of-fold scores.** If a model scores households it was trained on, it has already seen their answers and its ranking looks better than it will be in practice. Here the data is cut into five parts; each part is scored by a model trained on the other four. Every household ends up with one score from a model that never saw it, and because the cuts follow survey villages (PSUs), never saw its neighbours either.

### Cell 3: Capture against the ceiling, and the model's edge over the income rule (Q1)

`savings_goal.evaluation.metrics.capture_vs_ceiling` ranks households by each score, contacts the top b% and records the share of all at-risk households reached (capture), the share of contacts who were at risk (precision), and capture as a percentage of the ceiling min(1, b / 0.681). It is run unweighted and with survey weights.

| Strategy | Capture @10% | @25% | @50% | @65% | % of ceiling @25% | Precision @25% |
| --- | --- | --- | --- | --- | --- | --- |
| Ceiling | 14.7% | 36.7% | 73.5% | 95.5% | 100% | 100% |
| Full model (with shares) | 14.7% | 36.6% | 70.8% | 87.0% | 99.6% | 99.6% |
| Headline model | 14.7% | 36.2% | 68.5% | 83.6% | 98.5% | 98.5% |
| Income rule (poorest first) | 14.4% | 35.0% | 65.8% | 80.9% | 95.4% | 95.4% |
| Random contact | 10.0% | 24.9% | 49.9% | 65.1% | 67.8% | 67.8% |

> **In plain terms: capture, and the ceiling on it.** A campaign can only contact so many households. Fix a budget, say 25% of households; each strategy ranks everyone, you contact its top 25%, and capture is the fraction of all at-risk households you reached. Here 68% of households are at risk, so a 25% budget can reach at most 25/68 = 36.7% of them even with a perfect list. Raw capture looks small for every strategy for that reason, so the table also reports each strategy as a percentage of that ceiling. Below the ceiling, capture and precision move together: at 25% the headline model's 98.5% of ceiling and its 98.5% precision are the same number.

The income rule is the comparison that matters. Random contact is far behind every informed strategy because the at-risk class is the majority, which makes "lift over random" a weak argument. The model's edge over poorest-first is measured with a PSU-clustered bootstrap (`savings_goal.evaluation.metrics.bootstrap_indices`, 500 resamples, `percentile_ci`):

| Budget | Capture: model − income rule (pp) | 95% CI | Precision: model − income rule (pp) | 95% CI |
| --- | --- | --- | --- | --- |
| 10% | +0.29 | 0.22 to 0.36 | +1.95 | 1.50 to 2.46 |
| 25% | +1.14 | 1.00 to 1.32 | +3.10 | 2.72 to 3.57 |
| 50% | +2.70 | 2.44 to 2.98 | +3.68 | 3.35 to 4.05 |
| 65% | +2.71 | 2.46 to 3.05 | +2.84 | 2.57 to 3.18 |

Every interval excludes zero, so the model does better than the income rule at every budget, and the gap widens as the budget grows into the middle of the income distribution, where income alone ranks households least well.

> **In plain terms: a PSU-bootstrap interval.** The survey sampled villages and urban blocks (PSUs), then households within them, so neighbouring households are not independent draws. To see how much the +1.1 pp could move under a different sample, the notebook redraws 500 pseudo-samples by picking whole PSUs at random with replacement, recomputes the difference each time, and reports the middle 95% of those results. Resampling whole PSUs keeps the interval from looking narrower than the survey design allows.

The survey-weighted version (`capture_weighted` in `business.json`) counts the budget in population weight: households are taken in score order until their weights reach the budget share of all households. The at-risk share is 69.5% weighted, so the ceiling at 25% is 36.0%. The weighted picture matches the unweighted one: at 25% the model reaches 98.5% of that ceiling and the income rule 95.1%, with precision 98.5% against 95.1%.

### Cell 5: Where the at-risk households are, and where the model is reliable

Households are cut into income deciles and the headline model's out-of-fold calls at 0.5 are scored per decile.

| Income decile | Median income (Rs/yr) | At-risk rate | Weighted | Accuracy (headline) |
| --- | --- | --- | --- | --- |
| 0 | 13,360 | 97.8% | 97.0% | 0.978 |
| 1 | 28,100 | 94.5% | 93.7% | 0.946 |
| 2 | 40,230 | 91.7% | 90.1% | 0.919 |
| 3 | 53,430 | 85.0% | 83.4% | 0.857 |
| 4 | 67,500 | 78.4% | 77.2% | 0.810 |
| 5 | 84,737 | 70.5% | 67.8% | 0.750 |
| 6 | 108,780 | 61.9% | 59.8% | 0.700 |
| 7 | 145,730 | 48.7% | 47.5% | 0.662 |
| 8 | 216,000 | 34.5% | 33.2% | 0.687 |
| 9 | 410,000 | 16.9% | 16.6% | 0.833 |

Overall out-of-fold accuracy is 0.815. In the bottom three deciles accuracy is almost exactly the at-risk rate (0.978 vs 97.8% in decile 0): the model calls nearly everyone at risk there, which is correct but adds nothing a team would not already assume. Accuracy is lowest in deciles 6 to 8, where the at-risk rate passes through 50% and the outreach decision is contested.

The same cell bins the out-of-fold probabilities into ten equal-width bins and compares predicted with observed on-track rates. The largest bin error is 0.013. `savings_goal.evaluation.metrics.expected_calibration_error` gives 0.0055 for the headline model and 0.0074 for the full model.

> **In plain terms: calibration and ECE.** A model is calibrated when its probabilities mean what they say: of the households it scores at 0.70, about 70% turn out on track. The check sorts households into ten bins by predicted probability and compares the average prediction with the observed rate in each bin. ECE (expected calibration error) is the average of those gaps, weighted by how many households fall in each bin. At 0.006, predicted and observed rates differ by well under one percentage point on average, so a team can use the scores for expected-count arithmetic ("10,000 contacts at an average 80% risk is about 8,000 at-risk households") as well as for ranking.

### Cell 7: Peer benchmarks and gap closure (Q2)

IHDS does not record which spending a household could have avoided. The notebook therefore compares each at-risk household with the mean on-track household in the same income decile, using `savings_goal.evaluation.business.benchmark_with_ci`. That function computes signed excess per category (negative means the household spends less than its peers; nothing is clipped before averaging), in budget-share points and in rupees, and re-estimates the peer means inside each of 300 PSU-bootstrap draws.

| Category | Share excess (pp) | 95% CI | Rupee excess (Rs/yr) | Sample total (Rs crore/yr) |
| --- | --- | --- | --- | --- |
| Healthcare | +4.37 | +3.78 to +4.86 | 11,764 | 33.2 |
| Education | +2.83 | +2.67 to +2.98 | 5,976 | 16.9 |
| Miscellaneous | +2.09 | +1.69 to +2.49 | 18,160 | 51.3 |
| Transport | +1.80 | +1.45 to +2.14 | 7,080 | 20.0 |
| Insurance | +0.48 | +0.41 to +0.53 | 1,405 | 4.0 |
| Rent | +0.36 | +0.16 to +0.54 | 1,191 | 3.4 |
| Entertainment | +0.33 | +0.29 to +0.37 | 833 | 2.4 |
| Eating out | +0.13 | +0.01 to +0.21 | 650 | 1.8 |
| Clothing & footwear | −0.49 | −0.66 to −0.31 | 2,570 | 7.3 |
| Utilities | −2.90 | −3.39 to −2.40 | 4,773 | 13.5 |
| Groceries | −8.99 | −9.73 to −8.28 | 21,720 | 61.4 |

> **In plain terms: peer benchmarking and signed excess.** Labelling categories "essential" or "optional" by hand would write our own assumptions into the answer. Instead, each at-risk household is compared with on-track households at the same income: if they put 10% of their budget into transport and it puts 12%, its excess is +2 points. Comparing within an income decile matters because against the whole population a poor household would look "short" on almost everything, and the exercise would only rediscover that it is poor. Keeping the sign matters too: averaging only the positive differences would make every category look like overspending.

The rupee column cannot identify a lever. Every rupee interval is above zero, in all 11 categories, and that is forced by the label: at a given income, being at risk means a larger total budget, so the at-risk household spends more rupees than its on-track peer on nearly everything (94% of at-risk households exceed their peers on groceries in rupees). The levers are judged on budget composition instead. Eight categories take a significantly larger share of at-risk budgets; miscellaneous ranks third of those eight. Groceries take 9.0 points less of the budget, so a food-spending campaign would be aimed at the wrong households.

> **In plain terms: crore, and sample totals.** A crore is 10 million rupees. The "sample total" column adds rupee excess over the 28,262 surveyed at-risk households. It is not scaled by survey weights and is not a national figure.

`savings_goal.evaluation.business.gap_closure` then asks, household by household, whether matching peers' spending where the household spends more would cover its rupee gap to the 20% benchmark. The median gap is Rs 37,114 a year. Matching peers' rupee spending would close the gap for 92.8% of at-risk households (91.8% weighted); matching peers' budget shares, applied to the household's own total, would close it for 22.3%. The two answers differ because the rupee benchmark restates the label. The split between a "structural" and a "behavioural" shortfall here comes from the choice of benchmark, and the data cannot settle it.

### Cell 9: Measurement caveats

22.0% of households (23.8% weighted) report spending more than twice their reported income; they are 32.3% of the at-risk group. Indian household surveys tend to under-report income relative to item-by-item consumption, so part of the at-risk group reflects measurement rather than observed distress. Moving the benchmark changes the at-risk share: 61.7% at a 10% savings threshold, 68.1% at 20%, 74.7% at 30%. Phase 1 found 93.6% and 93.4% label agreement with the 20% definition at those two alternatives, so rankings are stable while the level moves.

### Cell 11: The recommendations

Each recommendation is built from f-strings over the values computed above (and the Phase 3 income threshold from `results/baseline.json`), written to `results/business_recommendations.csv`, and printed. The cell then writes `results/business.json` with the capture tables (unweighted and weighted), the bootstrap edge, decile table, calibration bins, ECE, the benchmark table, the gap-closure figures, the measurement caveats and a 41-point capture curve per strategy that Phase 8's figure script reads.

### Cell 12: Figure

`results/business_translation.png` has three panels: capture curves for the four strategies with the ceiling line, the signed composition excess per category with 95% intervals, and the calibration curve of the headline model against the diagonal. `project/figures/fig3_business.png` redraws the same content for the report, with accuracy by decile in place of calibration.

---

## The five recommendations

1. Prioritise by income first; the onboarding model re-ranks within income bands. The reference is a single income threshold (about Rs 122,249 a year, re-learned per fold). At a 25% budget the headline model reaches 98.5% of the ceiling against 95.4% for poorest-first: +1.1 pp of capture (CI 1.0 to 1.3). Caveat: the model refines an income ranking; it does not find a hidden at-risk segment.
2. Judge targeting against the ceiling and the income rule, not against random contact. With 68.1% at risk, no list reaches more than 36.7% of them with a 25% budget. Precision at 25%: 98.5% for the model, 95.4% for the income rule (+3.1 pp, CI 2.7 to 3.6), 67.8% at random. Caveat: lift over random looks large only because the target class is the majority.
3. Do not size a budgeting product from peer benchmarks. The share of at-risk households that could close the gap by spending like peers is 92.8% or 22.3% depending on whether the benchmark is in rupees or budget shares. Median gap Rs 37,114 a year. Caveat: IHDS cannot separate structural from behavioural shortfall.
4. Miscellaneous is not the lever; composition differs most in healthcare and education. Healthcare +4.4 pp, education +2.8, miscellaneous +2.1, transport +1.8, groceries −9.0. Caveat: excess over peers is a hypothesis about slack. Healthcare and education are hard to cut, and the rupee totals (miscellaneous Rs 51 crore a year) are sample totals.
5. Report relative priority only, never an absolute prevalence figure. 32.3% of the at-risk group report spending more than twice their income, and the at-risk share moves from 61.7% to 74.7% across the 10% to 30% thresholds. Caveat: a statement that "X% of Indian households save inadequately" is not supported by this work.

---

## What this means for later phases

| Finding | Used in |
| --- | --- |
| The headline model is close to the capture ceiling and 1.1 pp ahead of poorest-first at 25% | Phase 8 executive summary; fig 3(a) |
| Accuracy dips to 0.662 in income decile 7 | Phase 8 "what a stakeholder must be told"; fig 3(b) |
| Composition excess, not rupee excess, is the only usable peer comparison, and it cannot size a product | Phase 8 recommendations; fig 3(c) |
| A binary label says who misses the benchmark, not by how much | Phase 8 notebook: quantile regression on the savings rate and ranking by predicted rupee shortfall |
| The capture curves and decile table in `business.json` | `project/figures/make_figures.py` |
