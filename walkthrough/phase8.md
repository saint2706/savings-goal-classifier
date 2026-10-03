# Phase 8: Distance from Adequacy and Reporting

**Source:** [README § Phase 8: Reporting and distance from adequacy](../README.md#phase-8-reporting-and-distance-from-adequacy)
**Notebook:** [`notebooks/08_savings_rate_regression.ipynb`](../notebooks/08_savings_rate_regression.ipynb)
**Builds on:** [Phase 1](phase1.md) (leakage check, deployable feature set), [Phase 4](phase4.md) (tuned headline model), [Phase 7](phase7.md) (capture at a budget, income rule)
**Artifacts:** `results/savings_rate_regression.json`, `results/savings_rate_regression.png`; report figures `project/figures/fig1`–`fig4`

Phase 8 has two parts. Part A is a short notebook that models how far each household is from the 20% benchmark rather than only whether it misses it, and asks whether ranking by the predicted rupee shortfall reaches more of the total gap than the classifier. Part B is the stakeholder write-up: every number in it was computed by Phases 1 to 8 and stored in `results/`.

---

## Research questions & answers

| # | Question | Answer |
| --- | --- | --- |
| 1 | Does ranking by predicted rupee shortfall (quantile regression on `Savings_Rate`) reach more of the gap than a classifier? | Yes. At a 25% contact budget, ranking by the median model's predicted shortfall reaches 45.5% of the total rupee gap, against 37.0% for the classifier and 28.7% for the income rule. It reaches about as many at-risk households as the income rule (35.0%, vs 36.2% for the classifier). |
| 2 | Does the write-up explain reasoning rather than just reporting numbers? | Yes: [Why each decision was made](#why-each-decision-was-made) maps each material decision to the phase that made it and the evidence behind it. |
| 3 | Which 3–4 visualisations communicate the findings fastest to a non-technical reader? | The four report figures in `project/figures/`: the ROC-AUC ladder with the model-free oracle (fig 1), SHAP for the headline model (fig 2), the campaign view (fig 3) and cluster attainment within income deciles (fig 4). See the [visual appendix](#visual-appendix). |

---

## Part A: notebook walkthrough

### Cell 1: Out-of-fold median savings rate

The binary label records whether a household falls short of a 20% savings rate, not by how much. This cell models the savings rate itself, using only the 13 deployable features (`spec.deployable_numeric` plus the four categoricals: income, household size and composition, head age, adult education, debt, occupation, area type, caste group, religion). Spending shares are left out for the reason Phase 1 gave: with household size and income they reconstruct total spending, and so the label.

`savings_goal.models.regression.quantile_model` builds a pipeline of the shared preprocessor and a `HistGradientBoostingRegressor` with quantile loss at τ = 0.5 (300 iterations, learning rate 0.08). It is fitted inside the same PSU-grouped folds as earlier phases (`savings_goal.evaluation.cv.grouped_cv`), so each household's predicted median comes from a model that did not see its PSU. A constant benchmark, the training folds' median savings rate, is computed alongside.

| Measure | Value |
| --- | --- |
| Out-of-fold pinball loss, model | 0.474 |
| Out-of-fold pinball loss, constant median | 0.739 |
| Spearman (predicted median, actual savings rate) | 0.779 |

The model reduces pinball loss by 36% relative to predicting the same median for everyone, and its predictions order households by savings rate with a rank correlation of 0.78.

> **In plain terms: why the median, not the mean.** The savings rate has a very long left tail: the median household's rate is −10.8%, and the bottom 1% sit below −1,573%, mostly households whose reported income is far below their recorded spending. An ordinary regression predicts the average, and a few extreme values drag the average around. Quantile regression at τ = 0.5 predicts the conditional median instead: for households like this one, the rate that half fall above and half below. The extremes barely move it.

> **In plain terms: pinball loss.** Pinball loss (`savings_goal.models.regression.pinball_loss`) is the error measure that matches a median prediction: at τ = 0.5 it is half the average absolute gap between predicted and actual savings rate. Lower is better. A loss of 0.474 means the typical miss is still large in absolute terms, which is expected with a tail this long; the comparison that matters is against the 0.739 a constant median scores.

> **In plain terms: Spearman correlation.** Spearman's correlation compares the order of two lists, not their values. 1 means the predicted ranking matches the actual ranking exactly, 0 means no relation. Ranking is what a contact list needs, so it is the more useful check here than how close each predicted rate is.

### Cell 3: Ranking by predicted rupee shortfall vs the classifier and the income rule

The predicted rate becomes a rupee figure with `savings_goal.models.regression.rupee_shortfall`: max(0, 0.20 − predicted median) × income. The actual shortfall is the same formula applied to the observed rate. The headline classifier's out-of-fold scores are recomputed with `oof_predict_proba` and the Phase 4 parameters, so the classifier rows match Phase 7. Each strategy is then scored with `savings_goal.evaluation.metrics.capture_at_budget` twice: once on the at-risk indicator (share of at-risk households reached) and once on the actual rupee shortfall (share of the total rupee gap reached).

| Strategy | At-risk reached @10% | @25% | @50% | Rupee gap reached @10% | @25% | @50% |
| --- | --- | --- | --- | --- | --- | --- |
| Predicted rupee shortfall (median model) | 14.3% | 35.0% | 67.1% | 23.7% | 45.5% | 69.6% |
| Classifier at-risk score | 14.7% | 36.2% | 68.5% | 17.0% | 37.0% | 64.6% |
| Income rule (poorest first) | 14.4% | 35.0% | 65.8% | 12.4% | 28.7% | 54.7% |

Precision at 25% is 95.4% for the shortfall ranking, 98.5% for the classifier and 95.4% for the income rule.

The two rankings answer different questions. The classifier is best at finding households below the line; the shortfall ranking gives up about a point of at-risk capture and puts households with the largest rupee gaps first. Its advantage over the classifier is 6.7 pp of the gap at 10%, 8.5 pp at 25% and 5.0 pp at 50%.

Among at-risk households only, the Spearman correlation between predicted and actual rupee shortfall is 0.31. The model orders the size of the gap far less well than it orders the savings rate across all households (0.78), so individual shortfall predictions are rough even when the ranking gains in aggregate.

> **In plain terms: "share of the total rupee gap reached".** Add up, over every household, how many rupees short of 20% savings it is. A strategy that contacts 25% of households "reaches" the part of that total belonging to the households it contacted. Two households can both be at risk while one is Rs 2,000 short and the other Rs 80,000 short; capture of at-risk households counts them equally, this measure does not.

One identity to keep in mind: (0.20 − savings rate) × income equals consumption − 0.8 × income. The rupee gap is therefore largest where recorded consumption is far above reported income, which is where Phase 1 found income under-reporting concentrated (22.0% of households report spending more than twice their income). A rupee-gap ranking leans further into that group than the classifier does.

### Cell 4: Figure and results

`results/savings_rate_regression.png` plots, for 50 contact budgets from 1% to 100%, the share of the total rupee gap each strategy reaches, with a dotted random-contact diagonal. The shortfall curve sits above the classifier's and the income rule's over the lower half of budgets, and the curves converge as the budget approaches everyone. The cell writes `results/savings_rate_regression.json` with both pinball losses, both Spearman correlations, the ranking table and the feature list.

---

## Part B: the stakeholder write-up

Written for a savings-product or financial-inclusion team deciding which households to prioritise, with no assumed background in machine learning.

### Executive summary

The survey covers 41,518 Indian households interviewed in 2011–12 (India Human Development Survey, round II). 68.1% of them (69.5% after survey weighting) keep less than 20% of their annual income after recorded consumption; we call these households at risk.

A model that uses only what a team could ask at onboarding (income, household size and composition, age and education of the head and adults, debt, occupation, area, social group) ranks households well. Shown a random on-track household and a random at-risk household from villages it never saw, it scores the on-track one higher 88.6% of the time (ROC-AUC 0.886), and its probabilities match observed rates closely (calibration error 0.009).

Its value over a simple rule is modest. Contacting the poorest households first already reaches 95.4% of the at-risk households any 25% contact list could reach; the model reaches 98.5%, a gain of 1.1 percentage points of capture (95% interval 1.0 to 1.3). Income does most of the work.

Spending patterns would appear to add a lot (ROC-AUC 0.93), but measured over the same year as the outcome they mostly restate it: with household size and income, food spending reconstructs the size of the household budget, and a formula with no model reaches 0.895 from those three facts alone.

If the goal is rupees rather than headcount, ranking households by predicted rupee shortfall reaches 45.5% of the total gap with a 25% contact budget, against 37.0% for the risk score and 28.7% for poorest-first.

Recommended use: prioritise by income, let the onboarding model re-rank within income bands, report relative priority rather than prevalence, and do not size a budgeting product from these data.

> **In plain terms: the measures used in this summary.**
> - ROC-AUC: pick one household that is on track and one that is not; ROC-AUC is how often the model gives the first a higher score. A coin flip scores 0.50, a perfect ranking 1.00. It measures the order of the list, which is what a contact list needs.
> - Macro-F1: how good the model's yes/no calls are, scoring "right about at-risk" and "right about on-track" equally. A model that calls everyone at risk is right 68% of the time and scores 0.405; the income rule scores 0.742 and the headline model 0.781.
> - Calibration error (ECE): the average gap between the predicted and observed rate. At 0.009, a household scored at 70% belongs to a group where about 70% are on track.
> - Percentage points (pp): the difference between two percentages. 98.5% vs 95.4% is a gap of 3.1 pp.

### The business question

A team cannot review 41,518 households by hand. The project builds a triage score that flags households unlikely to keep an adequate share of their income, so that outreach goes first to those who need it most.

The target is normative. No large household survey asks people for their own savings goal, so "on track" here means keeping at least 20% of annual income. That measures savings capacity, not whether a household meets goals it set for itself, and nothing in this report is a claim about personal goals.

### What we found

1. Income dominates. Risk falls from 97.8% in the poorest tenth of households to 16.9% in the richest. In the headline model income accounts for 58.0% of the explanation (SHAP) and 82% of permutation importance (the drop in score when a feature group is shuffled). A single income cut-off, around Rs 122,249 a year, already reaches macro-F1 0.742.
2. The model refines an income ranking. Its edge over poorest-first is +0.3 pp of capture at a 10% budget, +1.1 at 25% and +2.7 at 50% and 65%, all with intervals above zero. Its precision edge at 25% is 3.1 pp (98.5% vs 95.4%). After survey weighting, its capture at 10% and 25% is level with the income rule (39.0% vs 39.1% at 25%); the precision edge remains.
3. About a third of the model's behaviour comes from combinations of features (31.6% of attribution is interactions), led by income with household size. The same income means different things for a family of three and a family of eight.
4. Spending shares cannot be used to score households in the same period as the outcome. Food spending relative to household size, divided by income, ranks households at ROC-AUC 0.895 with no model at all; the ablation shows the model loses only 0.002 without the food share but 0.053 without all shares and spending indicators. The spending model (0.929 in cross-validation, 0.932 on held-out villages) is a diagnostic, not a forecast.
5. Peer comparisons cannot size a budgeting product. At the same income, at-risk households spend more rupees than on-track households in all 11 categories, because "at risk" means a larger budget. In budget shares they spend more on healthcare (+4.4 pp), education (+2.8), miscellaneous (+2.1) and transport (+1.8), and 9.0 pp less on food. Whether matching peers would close the gap depends on how "matching" is defined: 92.8% of at-risk households under a rupee benchmark, 22.3% under a share benchmark.
6. The spending clusters are groups defined by which categories a household records as zero: 69.7% record spending in all core categories, 18.6% record no healthcare spending and 11.7% record no transport spending. They agree with the raw zero pattern at an adjusted Rand index of 0.858 and are nearly independent of income. They describe the data; they are not behavioural personas to target.
7. Where the goal is rupees, a different ranking helps. Ranking by predicted rupee shortfall reaches 45.5% of the total gap at a 25% budget (risk score 37.0%, income rule 28.7%) while reaching 35.0% of at-risk households.

| Recommendation | Evidence |
| --- | --- |
| Prioritise by income; use the onboarding model to re-rank within income bands | +1.1 pp capture over poorest-first at 25% (CI 1.0 to 1.3); +3.1 pp precision |
| Judge any targeting result against the ceiling and the income rule, not random contact | With 68.1% at risk, a 25% list reaches at most 36.7% of them; random contact reaches 24.9% |
| Rank by predicted rupee shortfall when the programme is sized in rupees | 45.5% of the gap at 25% vs 37.0% (risk score) and 28.7% (income rule) |
| Do not size a budgeting product from peer benchmarks | 92.8% vs 22.3% "closable" depending on the benchmark |
| Report relative priority only, never a prevalence figure | 32.3% of the at-risk group report spending over twice their income; at-risk share is 61.7% to 74.7% across 10%–30% thresholds |

### What a stakeholder must be told before acting

- The model is least accurate where the decision is hardest. Accuracy is 0.978 in the poorest tenth, where almost everyone is at risk anyway, and 0.662 in the eighth tenth (decile 7), where the at-risk rate is 48.7%. Overall accuracy (0.815) hides that dip.
- The at-risk count is not a population estimate. 22.0% of households report spending more than twice their income and they make up 32.3% of the at-risk group. Indian household surveys under-report income relative to itemised consumption. The ranking is usable; the count is not.
- The 20% line is a convention. At 10% the at-risk share is 61.7%, at 30% it is 74.7%. About 93.5% of households keep the same label either way, so rankings hold and levels do not.
- The probabilities can be used for planning arithmetic. Out-of-fold calibration error is 0.006 and no probability bin is off by more than 1.3 points.
- Spending data can only be used if it was recorded before the period being predicted. The scoring service enforces this (below).
- Nothing here says who would respond to outreach. IHDS records no intervention, so risk is not the same as benefit from contact.
- The data is from 2011–12: sound for method, not current for market sizing.

### Scoring service and uplift scaffold

`src/savings_goal/api.py` serves the score as a FastAPI app (`sgc train` fits the models, `sgc serve` runs the service) with two endpoints:

- `/score/onboarding`: scores from income, demographics and debt only, the 13 deployable features. Requests are validated (for example, dependants cannot exceed household size; unknown debt stays unknown rather than becoming zero).
- `/score/with-spending`: adds spending shares, and accepts them only when `shares_observed_through` is strictly before `prediction_window_start`. Shares from the same period as the outcome reconstruct it, so the service rejects them with a 422 error. Shares must also sum to one.

Who to contact is a different question from who would respond. Answering it needs data in which some households were contacted (ideally at random) and others were not, and IHDS has none. `savings_goal.models.uplift` provides a T-learner (separate outcome models for contacted and not-contacted households; uplift is the difference) and a Qini-curve metric. It is tested on synthetic data in `tests/unit/test_models.py` and ready for campaign data; nothing in this project estimates uplift on IHDS.

---

## Why each decision was made

| Decision | Phase | Evidence |
| --- | --- | --- |
| Normative 20% target, with the threshold configurable | Phase 0, Phase 1 | No survey records a self-set goal. Threshold sensitivity is published: at-risk share 61.7% at 10%, 74.7% at 30%, with 93.6% and 93.4% label agreement. |
| Group all validation by survey PSU (`IDPSU`) | Dataset construction, Phase 1 | Households in a PSU share prices and local shocks. 2,461 PSUs; `PSUID` alone repeats across districts (39 distinct values). One grouped fold (8,299 households, 472 PSUs) is held out first. |
| Exclude raw rupee categories and expense-to-income ratios | Phase 1 | Either one, with income, reconstructs the label for 99.75% of households. |
| Headline model on the 13 deployable features; spending shares diagnostic only | Phase 1 (leakage check, ablation) | No single share carries the label (max marginal AUC distance from 0.5 is 0.10), but jointly with size and income they do: food spend scales with size at elasticity 0.63 (R² 0.29), and the size × food-share oracle reaches ROC-AUC 0.895. Dropping all shares and indicators keeps 94.3% of the full model's AUC (0.878 vs 0.931). |
| Keep missing debt as missing; winsorise debt at the 99th percentile inside each fold | Dataset construction, Phase 2 | 7.6% of households have no debt answer, and they meet the benchmark more often (0.367 vs 0.315). Fitting the cap inside each fold keeps held-out values from setting it. |
| Raw shares over the CLR log-ratio transform | Phase 2 | Logistic regression ROC-AUC 0.920 on raw shares vs 0.911 with CLR. |
| Macro-F1 as the selection metric, and a single income threshold as the reference | Phase 3 | The majority class scores 0.681 accuracy and 0.405 macro-F1 while never finding an on-track household. The income threshold (re-learned per fold, mean Rs 122,249) scores 0.742. |
| Tuned XGBoost (depth 4, learning rate 0.08, 200 trees, subsample 0.7) | Phase 4 | Grouped-CV macro-F1 0.782 and ROC-AUC 0.882, the best of the candidate models; ECE 0.011 vs 0.103 for logistic regression. Held out: ROC-AUC 0.886, macro-F1 0.781, at-risk PR-AUC 0.944, Brier 0.127, ECE 0.009. |
| Explain with grouped SHAP and permutation importance, not coefficients | Phases 2, 5 | The 11 shares sum to one, so their coefficients are not identified (VIF about 10¹⁵). Income: 58.0% of grouped SHAP, 82% of permutation importance; interactions 31.6%. |
| Describe the k = 3 clusters as zero-pattern groups, not personas | Phase 6 | ILR basis; all three selection indices pick k = 3. Adjusted Rand index 0.858 with the zero pattern; NMI 0.012 with income decile. |
| Measure targeting against the ceiling and the income rule, with PSU-bootstrap intervals | Phase 7 | 98.5% vs 95.4% of ceiling at 25%; +1.1 pp (CI 1.0 to 1.3). Random contact is not a meaningful comparison when 68% are at risk. |
| Judge spending levers by signed budget-share excess against mean peers | Phase 7 | Rupee excess is positive in all 11 categories by construction. Share excess: healthcare +4.4, education +2.8, miscellaneous +2.1, transport +1.8, food −9.0 pp. |
| Do not split shortfall into structural and behavioural | Phase 7 | 92.8% closable under a rupee benchmark, 22.3% under a share benchmark: the answer belongs to the benchmark. |
| Model the savings rate with median regression and rank by rupee shortfall | Phase 8 | Pinball loss 0.474 vs 0.739 for a constant; 45.5% of the rupee gap at 25% vs 37.0% for the classifier. |
| Two-tier scoring service with a date check on spending data | Phase 8 | Same-period shares reconstruct the label (Phase 1), so the spending tier accepts only shares observed before the prediction window. |

> **In plain terms: the technical terms in the table.**
> - Leakage: a feature that contains the answer. Spending categories plus income add up to the savings rate, so a model given them scores near-perfectly and predicts nothing new.
> - Oracle: a formula with no fitted model, here 1 − (Rs 8,848 × household size / food share) / income. If it scores higher than a model, the model's inputs are doing arithmetic, not prediction.
> - Grouped cross-validation: testing on whole villages or urban blocks the model never saw, so neighbours cannot leak information between training and testing.
> - SHAP: splits each prediction into contributions from each feature ("income pushed this household toward on-track, household size pulled it back").
> - ILR and CLR: log-ratio transforms for data that must add up to 100%, such as budget shares.
> - Adjusted Rand index: agreement between two groupings, 1 for identical and 0 for chance.

---

## Visual appendix

Four report figures, built by `project/figures/make_figures.py` from `results/*.json` and `results/*.csv` (no numbers are typed by hand):

| Figure | Question it answers | What it shows |
| --- | --- | --- |
| `project/figures/fig1_feature_ladder.png` | How much does each ingredient add? | ROC-AUC from grouped CV for a ladder of models: single income threshold 0.739, income-only logistic regression 0.835, income plus demographics 0.876, headline XGBoost 0.882, XGBoost with spending shares 0.929. A vertical line marks the model-free oracle at 0.895, above every model that does not see spending shares. |
| `project/figures/fig2_shap.png` | What drives the score? | The ten largest mean absolute SHAP values for the headline model on held-out PSUs (income 1.95, household size 0.50, debt-to-income 0.30), and a bar splitting attribution by family: income 58.0%, household size 14.5%, social and geographic 10.9%, debt 8.7%, head age and education 8.0%. |
| `project/figures/fig3_business.png` | What is the model worth to a campaign? | (a) capture curves for the model, the income rule and random contact against the ceiling, annotated 98.5% vs 95.4% of ceiling at 25%; (b) accuracy by all ten income deciles with the overall 0.815 line and the decile 7 low of 0.662; (c) signed budget-share excess per category with 95% intervals. |
| `project/figures/fig4_personas.png` | Do the clusters matter once income is known? | Share meeting the benchmark within each income decile for the three clusters, with the likelihood-ratio test for cluster given decile and the pseudo-R² gain from 0.265 to 0.280. |

Supporting figures from the notebooks, for a technical reviewer:

| File | Notebook | Content |
| --- | --- | --- |
| `results/phase1_eda.png` | 01 | Income before and after the log, savings-rate distribution with the 20% line, zero-spend rates by category, attainment by area type |
| `results/phase1_share_correlations.png` | 01 | Correlations between the 11 expense shares |
| `results/baseline.png` | 03 | Baselines under grouped CV, majority class to income threshold |
| `results/model_comparison.png` | 04 | Candidate models on the deployable and full feature sets |
| `results/shap_summary.png`, `results/shap_summary_full.png` | 05 | SHAP summaries for the headline and full models |
| `results/shap_dependence.png` | 05 | Family attribution and the food-share SHAP against implied spend relative to income |
| `results/cluster_selection.png` | 06 | Silhouette, Davies-Bouldin and Calinski-Harabasz across k |
| `results/personas.png` | 06 | Cluster spending signatures, attainment by cluster, attainment within income deciles |
| `results/business_translation.png` | 07 | Capture curves, signed composition excess, calibration curve |
| `results/savings_rate_regression.png` | 08 | Share of the total rupee gap reached by budget for the three rankings |

---

## Limitations

- Income under-reporting: 55.9% of households report consumption above income and 22.0% report more than twice their income. Relative rankings hold; absolute prevalence does not.
- Vintage: 2011–12 data. IHDS-3 microdata was not public at the time of this work.
- Household grain: no claim about individuals is supported.
- Weights: the survey weight `WT` is applied to population figures (prevalence, weighted capture) but not to model fitting. Rupee totals in `results/` are sample totals unless labelled otherwise. Survey-weighted, the model's capture edge over the income rule at small budgets disappears.
- Spending shares leak jointly with household size and income. They can be used to score only if measured before the outcome period.
- Debt is a stock: IHDS records outstanding debt but no repayment instalment, so `Debt_To_Income` stands in for a cash-flow burden it does not measure.
- The at-risk class is the majority (68%), so capture must be read against its ceiling.
- Peer benchmarks cannot separate structural from behavioural shortfall.
- The clusters are keyed on which categories are recorded as zero, which may reflect a healthy year or a recall gap rather than a budgeting style.
- The rupee-shortfall ranking favours households whose consumption far exceeds reported income, the group most affected by under-reporting; within at-risk households its predicted gap sizes correlate with actual ones at only 0.31.
- No intervention data: nothing here estimates who would respond to outreach.

---

## What this means for later work

| Next step | Needs |
| --- | --- |
| Deploy the onboarding tier | Fitted artifacts from `sgc train`; a decision on budget and operating threshold (held-out: threshold 0.425 gives 91.3% at-risk recall at 82.6% precision) |
| Use the spending tier | Spending recorded before the prediction window |
| Rank by expected benefit instead of risk | A randomised or logged campaign, then `savings_goal.models.uplift` |
| Current prevalence or market sizing | IHDS-3 or another recent survey, with weights applied |
