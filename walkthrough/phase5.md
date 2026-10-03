# Phase 5: Explainability

**Source:** [README § Phase 5: Explainability](../README.md#phase-5-explainability)
**Notebook:** [`notebooks/05_explainability.ipynb`](../notebooks/05_explainability.ipynb)
**Builds on:** [Phase 4](phase4.md) (both tuned models, stored in `results/model_final.json`), [Phase 2](phase2.md), [Phase 1](phase1.md) (the leakage check)
**Artifacts:** `results/explain.json`, `results/shap_importance.csv`, `results/shap_summary.png`, `results/shap_summary_full.png`, `results/shap_dependence.png`

This phase explains two models on the 8,299 held-out households (472 PSUs). The headline model is the tuned XGBoost on the deployable feature set: income, household demographics and debt, 13 raw features that one-hot encoding expands to 35 columns. The full model adds the 11 expense shares and 5 participation indicators (29 raw features, 51 encoded columns). Phase 1's leakage check showed that household size plus the food share recovers most of total spending, so the full model is explained as a diagnostic: its attributions show how the shares carry label information, not how households behave.

> **In plain terms: what explainability is for.** A model that predicts well but cannot say why is hard to trust and hard to act on. Explainability extracts reasons at two levels: which inputs drive predictions across all households, and why one particular household got its score. The obvious route, reading the coefficients of a linear model, does not work for the expense shares (the first section below shows why), so the phase uses SHAP instead.

---

## Research questions & answers

| # | Question | Answer |
| --- | --- | --- |
| 1 | Which features matter most globally? | In the headline model, income: mean \|SHAP\| 1.948 for `Log_Income`, 3.9 times the next column (`Household_Size`, 0.495). Summed by family, income is 58.0% of attribution, the household-size family 14.5%, social and geographic categoricals 10.9%, debt 8.7%, head age and education 8.0%. Permuting each family on the held-out set gives the same order, with income at 82% of the ROC-AUC loss (0.385 of 0.471). In the full model the share family is 19.1% of SHAP attribution and 16.0% of permutation importance; income is still first (51.8% and 73.8%). |
| 2 | Are there notable interaction effects? | Yes. Over all 8,299 held-out households, interactions are 31.6% of total attribution in the headline model and 42.8% in the full model. Every top pair in the headline model involves income: size × income (0.242), education × income (0.125), debt × income (0.064). In the full model the top pair is food share × income (0.338). |
| 3 | Can individual predictions be explained in plain business language? | Yes. SHAP values add up to the model's output (largest additivity error 6.5e-06 on the log-odds scale), so each household's explanation is a short list of signed pushes. The notebook prints three cases (most confident on track, most confident at risk, borderline) as contributions only, without any household's raw values. |

A fourth result frames the rest: no coefficient on the expense shares can be read as an effect. With all 11 shares in a logistic regression, 3 of 10 share coefficients flip sign across five disjoint training subsamples. Dropping `Groceries_Share` as a reference part leaves 1 flip, but the median coefficient spread barely changes (ratio 0.97) and 6 of 10 shares get less stable.

---

## Notebook walkthrough

### Cell 1: fit both tuned models and check they reproduce Phase 4

The notebook loads the engineered features (`savings_goal.io.load_features`), rebuilds the column lists with `savings_goal.features.engineer.spec_from_frame`, and reads the tuned hyperparameters with `savings_goal.models.pipeline.load_model_final`. For each feature set it fits `savings_goal.models.pipeline.xgb_pipeline` on the training split (33,219 households), which applies median imputation, debt winsorisation at the 99th percentile, and one-hot encoding before XGBoost.

| Model | Feature set | Encoded columns | Tuned parameters | Held-out ROC-AUC |
| --- | --- | --- | --- | --- |
| Headline | deployable | 35 | depth 4, lr 0.08, 200 trees, subsample 0.7, min_child_weight 1 | 0.886 |
| Diagnostic | full | 51 | depth 4, lr 0.15, 200 trees, subsample 0.9, min_child_weight 5 | 0.932 |

Both held-out scores match Phase 4, which confirms the same models were rebuilt. The cell also transforms the held-out rows through the fitted preprocessing step and keeps the encoded column names, because SHAP attributes to the 35 or 51 encoded columns rather than to the raw features.

> **In plain terms: encoded columns.** A categorical feature such as `Area_Type` cannot enter a tree model as text. One-hot encoding turns it into one yes/no column per level (`Area_Type_Metro_Urban`, `Area_Type_Other_Urban`, and so on). The four categoricals expand into 26 such columns, which is why 13 features become 35.

### Cell 3: why no coefficient on the shares can be read

The 11 expense shares sum to 1 for every household, so any one of them is an exact linear combination of the other ten (Phase 2). A linear model with all 11 has infinitely many coefficient sets that predict identically. The cell fits the same weakly penalised logistic regression (`savings_goal.models.pipeline.logistic` with C = 100, standard-scaled inputs from `savings_goal.models.pipeline.preprocessor(linear=True)`) on 5 disjoint training subsamples using `savings_goal.evaluation.explain.coefficient_stability`, twice: once with all 11 shares, once with `Groceries_Share` dropped as the reference part.

| Share | sd (all 11) | sd (food dropped) | Flips sign (all 11) | Flips sign (dropped) | Zero in |
| --- | --- | --- | --- | --- | --- |
| Healthcare | 0.051 | 0.070 | no | no | 19% |
| Miscellaneous | 0.043 | 0.049 | no | no | 0% |
| Utilities | 0.028 | 0.031 | no | no | 0% |
| Transport | 0.055 | 0.062 | no | no | 11% |
| Insurance | 0.041 | 0.044 | yes | no | 74% |
| Clothing & footwear | 0.052 | 0.053 | no | no | 1% |
| Eating out | 0.074 | 0.073 | yes | yes | 72% |
| Rent | 0.087 | 0.085 | yes | no | 90% |
| Entertainment | 0.017 | 0.015 | no | no | 69% |
| Education | 0.023 | 0.017 | no | no | 36% |

With all 11 shares, 3 of 10 coefficients flip sign between subsamples (eating out, rent, insurance). Dropping the reference part removes two of those flips, but it does not tighten the estimates: the median ratio of spreads is 0.97, six shares get slightly less stable (healthcare is the worst, at 1.4 times the spread), and no share changes by more than a factor of 1.4 either way. The three that flip sign are among the shares that are zero for most households.

> **In plain terms: reproducible is not the same as identified.** Fitting the same model on five separate groups of households and comparing coefficients is a quick test: a coefficient that measures something real should come out similar each time. A sign flip means the feature appears to help in one group and hurt in another. The test can only show the problem, though. A penalised regression breaks the tie between equivalent coefficient sets with a fixed rule, so five fits could agree perfectly and still be reporting the rule's choice rather than the data's. A bathroom scale that always reads 5 kg heavy is consistent and wrong. The algebra (shares that sum to 1 have no unique coefficients) is the argument; the table shows where it leaks through.

The phase therefore explains the tree model with SHAP, which needs no matrix inversion and is defined whether or not features are collinear.

> **In plain terms: SHAP.** For one household, SHAP answers: how much did each feature push this prediction up or down, compared with the average household? It comes from cooperative game theory. Treat the features as a team that jointly produced a score and split the credit by averaging, over every order in which the features could have been added, what each one contributed when it arrived. The contributions for a household add up exactly to the model's output for that household. TreeSHAP is an exact shortcut for tree models that makes this fast enough for thousands of households.

### Cell 5: global importance per column, per family, and by permutation

For each model the cell calls `savings_goal.evaluation.explain.tree_shap`, which runs `shap.TreeExplainer` and checks additivity: SHAP values plus the base value reproduce the raw model margin to within 6.5e-06 (headline) and 8.7e-06 (full).

> **In plain terms: the additivity check.** The base value is the model's starting point before any feature is considered. The margin is the model's output before it is converted to a probability. SHAP promises base value plus contributions equals margin for every household. The check adds them up and compares; a gap of a few millionths is rounding error.

Top columns in the headline model by mean |SHAP|:

| Column | Mean \|SHAP\| |
| --- | --- |
| `Log_Income` | 1.948 |
| `Household_Size` | 0.495 |
| `Debt_To_Income` | 0.301 |
| `Max_Adult_Education` | 0.260 |
| `Occupation_Non_Ag_Labour` | 0.158 |
| `Dependency_Ratio` | 0.125 |
| `Area_Type_Less_Developed_Village` | 0.109 |

> **In plain terms: mean |SHAP|.** Each household gets one SHAP value per feature, positive or negative. Dropping the signs and averaging over households measures how far a feature moves predictions, regardless of direction.

Per-column rankings split a feature like `Area_Type` across four columns and score each separately. `savings_goal.evaluation.explain.grouped_shap` instead sums SHAP within each family for each household (families from `savings_goal.evaluation.explain.feature_families`), takes the absolute value, then averages. As a cross-check, `savings_goal.evaluation.explain.grouped_permutation_importance` shuffles all columns of a family together on the held-out set (5 repeats, families from `savings_goal.evaluation.explain.raw_families`) and records the drop in ROC-AUC.

Headline model:

| Family | Columns | Grouped mean \|SHAP\| | Share of SHAP | ROC-AUC drop when permuted | Share of permutation |
| --- | --- | --- | --- | --- | --- |
| Income | 1 | 1.948 | 58.0% | 0.385 | 81.7% |
| Household size family | 3 | 0.487 | 14.5% | 0.039 | 8.3% |
| Social/geo (categoricals) | 26 | 0.364 | 10.9% | 0.022 | 4.6% |
| Debt | 3 | 0.291 | 8.7% | 0.014 | 3.0% |
| Head age & education | 2 | 0.268 | 8.0% | 0.011 | 2.4% |

Full model:

| Family | Columns | Share of SHAP | ROC-AUC drop when permuted | Share of permutation |
| --- | --- | --- | --- | --- |
| Income | 1 | 51.8% | 0.414 | 73.8% |
| Spending mix (shares) | 11 | 19.1% | 0.090 | 16.0% |
| Household size family | 3 | 10.9% | 0.033 | 5.9% |
| Social/geo (categoricals) | 26 | 7.1% | 0.014 | 2.4% |
| Head age & education | 2 | 4.4% | 0.004 | 0.7% |
| Debt | 3 | 3.5% | 0.004 | 0.7% |
| Participation indicators | 5 | 3.3% | 0.003 | 0.5% |

The two methods agree on the order in both models. Permutation gives income a larger share because it measures lost ranking ability: without income, the headline model's held-out ROC-AUC falls from 0.886 to about 0.50, while the other families each cost 0.04 or less.

For the shares, the grouped value (0.973) is about half the sum of the 11 per-column values (1.815). The share columns push in opposite directions for the same household, because a larger food share forces the others down, so their contributions partly cancel. Adding per-column scores would overstate the family by almost a factor of two.

> **In plain terms: permutation importance.** Shuffle one family's values across households, so each household gets someone else's income (say), and see how much worse the model ranks households. A big drop means the model relied on that information. It measures the same thing as SHAP from a different direction, which is why agreement between the two is reassuring.

### Cell 6: SHAP summary plots

One beeswarm plot per model (`shap_summary.png` for the headline model, `shap_summary_full.png` for the full model), 15 columns each. Each dot is a held-out household, placed by its SHAP value and coloured by the feature's value. In the headline plot, high income pushes toward on track and low income toward at risk, over a range of roughly −6 to +5 log-odds. Large households and high debt-to-income push toward at risk. Some smaller effects run against intuition: higher adult education and metro location push slightly toward at risk, while agricultural and non-agricultural labour households push slightly toward on track. These are conditional on income already being in the model and should not be read as the effect of education or occupation on saving.

### Cell 8: interactions over the whole held-out set

`savings_goal.evaluation.explain.interaction_decomposition` computes XGBoost's exact TreeSHAP interaction values for every held-out household, in chunks of 2,000 rows so the full 8,299 × 35 × 35 array never sits in memory at once. The diagonal holds each column's main effect; the off-diagonal entries hold pairwise interactions.

| Model | Main effects | Interactions | Interaction share | Top pairs (mean \|interaction SHAP\|) |
| --- | --- | --- | --- | --- |
| Headline | 3.870 | 1.790 | 31.6% | size × income 0.242; education × income 0.125; debt-to-income × income 0.064; less-developed village × income 0.046 |
| Full | 6.444 | 4.827 | 42.8% | food share × income 0.338; size × income 0.188; utilities share × income 0.172; education × income 0.089 |

In the headline model all of the top six pairs involve `Log_Income`, so income changes how much household size, education and debt matter. The size × income pair fits the arithmetic of the target: a savings rate is one minus spending over income, and spending grows with household size, so an extra member costs a low-income household a larger fraction of its income.

The full model's larger interaction share comes from the shares. Food share × income is its top pair, ahead of size × income, which is what the leakage check predicts: food share, size and income together approximately rebuild total spending relative to income.

> **In plain terms: an interaction.** Two features interact when the effect of one depends on the value of the other. An extra household member means something different on Rs 40,000 a year than on Rs 400,000. A model that gives each feature one fixed effect (a logistic regression without interaction terms) cannot express this; a tree can, because a branch reached only by low-income households can behave differently from its high-income neighbour. These figures describe how this fitted model reaches its answers. A model restricted to depth 4 can only represent fairly shallow interactions.

### Cell 10: dependence plots, and what the food-share pattern means

`shap_dependence.png` has two panels.

The left panel plots each held-out household's SHAP value for `Log_Income` in the headline model against its log income, one colour per area type. The curve is an S-shape: flat at about −5.5 below log income 9 (about Rs 8,000 a year), steep between 10 and 13, crossing zero near 11.5 (about Rs 100,000), and flattening near +4 to +5 above 14. The four area types lie on top of each other, so the model treats a rupee of income the same way in a metro and in a less-developed village.

> **In plain terms: a dependence plot, and the log scale.** One dot per household: its value for a feature along the bottom, that feature's SHAP contribution up the side. It shows which way and how hard the model pushes as the feature rises. The axis is in natural-log rupees: each step of 1 multiplies income by about 2.7, so 9 is about Rs 8,100, 11.5 about Rs 99,000, and 14 about Rs 1.2 million.

The right panel plots the full model's SHAP value for `Groceries_Share` against the share itself, coloured by log(household size / food share / income). SHAP rises from about −3 at a 10% food share to about +1 above 60%, crossing zero near 45%. A naive reading would be that, at a given income, food-heavy budgets are prudent budgets. The data supports a mechanical reading instead. Food spending tracks household size (Phase 1: elasticity 0.63, R² 0.29), so at a given size, total spending is roughly food spending divided by the food share. A higher food share therefore implies a smaller total budget, and with income known, a smaller budget relative to income means a higher savings rate. The colour in the panel is that implied budget relative to income. The food-share SHAP value rises as it falls, with a Spearman correlation of 0.27 between the SHAP value and the negative of the implied ratio. The size × food-share oracle from Phase 1 uses the same route and reaches ROC-AUC 0.895 with no model at all.

> **In plain terms: conditional and unconditional.** Across all households, high food shares go with low income (poorer households spend proportionally more on food). Inside a model that already knows income and household size, a high food share mostly tells the model that total spending is small. The same column answers two different questions depending on what else is held fixed. A recommendation such as "target households with high food shares" would need the second reading, and the second reading here is arithmetic about the label rather than behaviour.

### Cell 12: individual predictions in plain language

`explain_row` prints the five largest contributions for one held-out household, with sign and direction, and the predicted probability. Raw feature values are not printed. Positive values push toward on track and negative values toward at risk, in log-odds.

| Case | Predicted P(on track) | Actual | Largest contributions |
| --- | --- | --- | --- |
| Most confident on track | 98.9% | on track | income +4.48; adult education +0.41; dependency ratio +0.28; business occupation −0.17; debt-to-income +0.16 |
| Most confident at risk | 0.0% | at risk | income −5.29; debt-to-income −1.09; household size −0.78; adult education +0.20; head age −0.17 |
| Borderline | 50.0% | at risk | income +2.15; Sikh religion −0.66; household size −0.54; adult education −0.31; debt-to-income +0.20 |

The confident cases read as one sentence each: "very high income, with smaller pushes from adult education and the dependency ratio" or "very low income, heavy debt relative to income, and a large household". The borderline case shows the limits: a moderately positive income push is cancelled by several smaller negatives, and one of the larger ones is a religion indicator, which describes the household's group rather than anything it could change. An explanation can be legible and still give the household nothing to act on.

> **In plain terms: reading an individual explanation.** Start from the average household's score, then add each listed push. Income +4.48 means income alone moved this household a long way toward on track. The final probability is what the summed score becomes after conversion to the 0 to 1 scale. Only the five largest pushes are shown, so the listed numbers do not add up to the full score.

### Cell 13: write results

Writes the top 20 columns per model by mean |SHAP| to `results/shap_importance.csv`, and to `results/explain.json` (via `savings_goal.io.write_result`): the family tables, permutation tables, interaction totals and top 10 pairs, top 15 columns per model, the coefficient-stability table, and the food-share SHAP correlation (0.27).

---

## What this means for later phases

| Phase | Consequence |
| --- | --- |
| 6: Spending clusters | Spending mix carries about a fifth of the full model's attribution, and part of that is the size × food-share route to the label. Clusters built on spending shares should be checked against income before being read as behaviour, and any link they show with `Goal_Met` may run through the same budget-size arithmetic. |
| 7: Business translation | Lead with income, which is 58% of the headline model's SHAP attribution and 82% of its permutation importance. Area-type effects are small once income is known and income's effect does not vary by area. A high food share is not a behavioural signal to target. Any statement about a share has to be relative ("food taking a larger part of the budget, and everything else less"). |
| 8: Reporting | `shap_summary.png` and `shap_dependence.png` are the explainability figures. The interaction share (31.6% headline) shows the model is not additive, mostly through income. |
