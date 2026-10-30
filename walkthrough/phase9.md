# Phase 9: Outreach What-If

**Source:** [README § Phase 9: Outreach what-if](../README.md#phase-9-outreach-what-if)
**Notebook:** [`notebooks/09_outreach_simulation.ipynb`](../notebooks/09_outreach_simulation.ipynb)
**Builds on:** [Phase 7](phase7.md) (out-of-fold scores in `dataset/oof.parquet`, capture against a budget), [Phase 4](phase4.md) (tuned parameters)
**Artifacts:** `results/outreach.json`, `results/outreach.png`; code in `src/savings_goal/evaluation/outreach.py`

An outreach simulation needs explicit costs and a response to a nudge that differs across households, and the survey cannot say who would benefit from one. IHDS-II records no outreach, so response cannot be estimated from it. This phase therefore does not estimate a benefit. It asks how much the *targeting decision* depends on an assumption the data cannot test, and states every assumption in one place.

---

## Research questions & answers

| # | Question | Answer |
| --- | --- | --- |
| 1 | Does the household a risk model ranks first coincide with the household a nudge could move? | No. A 5 pp nudge can only carry a household across the benchmark if it sits within 5 pp of it, which is 4.6% of at-risk households. The most at-risk are the deepest dissavers. At a 25% budget and uniform response r = 0.20, the risk model moves 0.8 households per 1,000 contacts, the income rule 2.2, random contact 6.2, a reachability model 10.1 and an oracle that knows the true band 25.0. |
| 2 | Is that ordering an artefact of one scenario? | The reachability model beats the risk model by 9.3 per 1,000 (95% PSU-bootstrap CI 8.4 to 10.1) at uniform response, 11.3 (10.3 to 12.2) when response rises with income rank (access constraints), and 10.1 (8.9 to 11.0) when indebted households respond less. It holds at effect sizes of 2.5 pp (4.4 vs 0.3) and 10 pp (21.3 vs 2.0), and survey-weighted (9.9 vs 0.5). |
| 3 | What does a moved household cost? | At an *assumed* Rs 150 per contact: Rs 14,885 for the reachability model, Rs 24,028 for random contact, Rs 199,615 for the risk model. A programme breaks even only if a moved household is worth more than the relevant figure; the cost scales linearly, so any other cost is a rescaling. |

---

## Assumptions, all in one place

| Assumption | Value | Where it lives |
| --- | --- | --- |
| Effect of a nudge on a responder's savings rate | +5 pp of income (2.5 and 10 pp as sensitivity) | `DELTA` in the notebook |
| Cost per contact | Rs 150 | `COST` in the notebook |
| Who responds | `uniform(0.20)`, `constrained_poor(0.05, 0.35)` (rises with income rank), `debt_burdened(0.30, 0.10)` | `outreach.SCENARIOS` |
| A household is moved only if | `0.20 - delta <= Savings_Rate < 0.20` | `outreach.reachable` |

None of these is estimated. The first three are the knobs; the fourth is arithmetic.

## What each ranking sees

All rankings use only deployable features, out of fold under PSU-grouped CV. The **reachability model** fits the headline XGBoost twice, to `Goal_Met` and to the label relaxed by `delta`; the difference in P(on track) is the model's estimate of P(0.20 - delta <= Savings_Rate < 0.20). The **oracle** uses the true savings rate and is an upper bound, not a strategy. The **risk model** is the Phase 7 ranking, `1 - P(on track)`.

## Reading the result correctly

- The result is a statement about sensitivity, not benefit. It says that the choice between a risk ranking and a reachability ranking changes the outcome by an order of magnitude *under these assumptions*, and that the survey cannot adjudicate between them.
- The outcome is households moved across the benchmark. Counting rupees of gap closed instead would favour the largest gaps, which come from the least reliable part of the data (22.0% of households report spending more than twice their income), so it is not reported.
- Weighted figures (population) use the survey weights for selection and averaging; the models are fitted unweighted ([Phase 7](phase7.md) checks that weighting the fit changes nothing material).
- `savings_goal.models.uplift.TLearner` is the estimator to use once randomised campaign data exists; this phase is what to run before that.

## What this means for later work

The recommendation from [Phase 7](phase7.md) stands for *capturing* at-risk households. For a nudge, the budget is better spent near the benchmark, and the cheapest way to learn whether that is true is a small randomised pilot stratified by distance to the benchmark.
