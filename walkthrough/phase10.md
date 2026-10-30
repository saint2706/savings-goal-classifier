# Phase 10: External Validity

**Source:** [README § Phase 10: External validity](../README.md#phase-10-external-validity)
**Notebook:** [`notebooks/10_external_validity.ipynb`](../notebooks/10_external_validity.ipynb)
**Builds on:** [Phase 1](phase1.md) (the size × food-share oracle), [Phase 4](phase4.md) (headline model)
**Artifacts:** `results/external_validity.json`, `results/leakage_curve.png`; code in `src/savings_goal/data/hces.py` and `src/savings_goal/evaluation/{leakage_sim,transport}.py`

IHDS-II is from 2011-12, so the open question is how far the result survives to newer data. No public survey records both income and consumption for the same households after 2011-12, so the classifier cannot be tested out of time, and this phase does not pretend otherwise. It tests three things that can be tested.

---

## Research questions & answers

| # | Question | Answer |
| --- | --- | --- |
| 1 | Does the leakage mechanism still hold in a newer survey? | Yes. In HCES 2022-23 (261,746 households), the elasticity of food spend on household size is 0.565 against 0.630 in IHDS-II, the residual sd is 0.472 against 0.485, the mean household food share is 0.465 against 0.466, and the simulated oracle ROC-AUC is 0.868 in both. The rank correlation between the implied and actual budget is lower (0.556 against 0.670). |
| 2 | When does a spending-share representation leak at all? | When food spend is tightly tied to household size. The simulation puts any survey on one curve: oracle AUC falls from 1.00 (food exactly proportional to size) to 0.89 at residual sd 0.4, 0.83 at 0.6 and 0.74 at 1.0, for elasticity 0.63. IHDS-II (0.485) and HCES (0.472) sit near 0.87 (the empirical IHDS-II oracle is 0.895, higher because real food spend also moves with the budget). |
| 3 | Does the model transport across states and sectors? | Discrimination yes, calibration no. Leaving whole states out (5 folds), ROC-AUC is 0.871 to 0.890 (grouped CV 0.882) and the model beats a poorest-first rule at a 25% budget in every fold (+0.7 to +1.6 pp recall). Mean predicted minus observed on-track rate ranges from -7.1 to +3.9 pp across folds; rural-to-urban is 0.866 with +9.1 pp, urban-to-rural 0.882 with -4.8 pp. |

---

## The HCES data

HCES 2022-23 (NSO, MoSPI) records consumption but not income, so the savings label cannot be rebuilt on it. The loader (`hces.household_table`) uses a third-party level-wise parquet dump of the public microdata, <https://github.com/advaitmoharir/hces_2022> (`04_clean/level_01.parquet`, `level_03.parquet`, `level_14.parquet`). It follows the questionnaire's own MPCE construction (item groups from the dump's `2_mpce.do`) and **reproduces the published MPCE exactly: Rs 3,773 rural and Rs 6,459 urban per month**. That check is the evidence that the loader is right. Files go in `dataset/hces/`, which is gitignored; only aggregates are committed.

Definitions differ between the surveys (IHDS "Groceries" against the HCES food block, a total that includes durables in HCES), so levels are compared loosely and the *elasticity and residual spread* carry the argument.

## What nominal growth implies

Weighted MPCE rose from Rs 1,600 (rural) and Rs 2,572 (urban) in IHDS-II to Rs 3,773 and Rs 6,459, factors of 2.36 and 2.51, before any price deflator (neither file carries a CPI). The Rs 122,249 income threshold that Phase 3 learns is a 2011-12 rupee figure; a deployment today must re-base it.

## Reading the result correctly

- A stable mechanism means the *warning* travels: spending shares should still be treated as leaky wherever food tracks household size. It does not mean the 0.895 or 0.882 carry over.
- Transport tests show that ranking quality is robust to a change of population but absolute probabilities are not. Use the model to rank, recalibrate before reading a probability.
- Leave-states-out is a proxy for a changing population, not for time. A change in the income distribution over eleven years is bigger than the difference between states.

## What this means for later work

Anyone with income and consumption for a newer sample can repeat the check: compute the elasticity and residual sd (`leakage_sim.residual_sd`), read the implied AUC off `leakage_sim.oracle_auc`, and confirm with `leakage.size_share_oracle`.
