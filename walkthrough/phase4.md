# Phase 4: Model Comparison

**Source:** [README § Phase 4: Model Comparison](../README.md#phase-4-model-comparison)
**Notebook:** [`notebooks/04_model_comparison.ipynb`](../notebooks/04_model_comparison.ipynb)
**Builds on:** [Phase 3](phase3.md)
**Artifacts:** `results/model_comparison.csv`, `results/model_final.json`, `results/model_test.json`, `results/model_comparison.png`

Seven model families are compared on two feature sets under PSU-grouped cross-validation on the training split. One configuration is tuned, chosen, and then scored once on the held-out PSUs.

The two feature sets:

- **Deployable (headline), 13 features**: log income, household size, age dependents, dependency ratio, head age, highest adult education, debt-to-income, has-debt, debt-missing, and the four categoricals. All of these are known when a household is onboarded.
- **Full (diagnostic), 29 features**: the deployable set plus the 11 spending shares and 5 participation indicators. The leakage check in [Phase 1](phase1.md) showed that household size, the food share and income together approximate total spending, and the ablation showed that dropping the shares and indicators lowers grouped-CV ROC-AUC from 0.931 to 0.878. The full set is therefore an upper bound on what a model could do with spending composition, not the model to deploy.

At 31.9% positive (2.13:1), the classes are imbalanced but not severely, so macro-F1 and ROC-AUC can be used directly.

---

## Research questions & answers

| # | Question | Answer |
| --- | --- | --- |
| 1 | Which model families are appropriate given the data? | Seven: majority baseline, logistic regression, linear SVM, decision tree, random forest, histogram gradient boosting, XGBoost, plus the single income threshold from Phase 3. On the deployable set, tuned XGBoost (depth 4, learning rate 0.08, 200 trees, subsample 0.7, min_child_weight 1) has the best CV macro-F1 (0.782) and ROC-AUC (0.882). HistGradientBoosting (0.778 / 0.880) is within one fold standard deviation. |
| 2 | What validation strategy fits the class balance found in Phase 1? | `StratifiedGroupKFold(5, shuffle=True, random_state=42)` grouped by `IDPSU`. One grouped fold (8,299 households, 472 PSUs) is held out before any design decision; the remaining 33,219 households in 1,989 PSUs are used for every comparison and search. Fold-to-fold macro-F1 sd is 0.004 to 0.009. |
| 3 | What hyperparameter search method is used, and what parameters move performance most? | 15-iteration randomised search over five XGBoost parameters, refit on macro-F1, plus a 5-value sweep of logistic regression's `C`. On the deployable set tuning adds +0.007 macro-F1 (0.774 to 0.782) and +0.006 ROC-AUC; `max_depth` moves the score most (sd 0.0085 across levels). On the full set tuning adds nothing (0.836 both ways). |
| 4 | Is precision or recall more important given the business framing? | Recall on the at-risk class, because a missed at-risk household costs more than an unneeded nudge. The at-risk threshold tuned on out-of-fold training scores is 0.425 (precision 0.827, recall 0.915). Applied once to the held-out set it gives precision 0.826, recall 0.913. Reaching 95% recall costs precision 0.789 (out-of-fold). |

Held-out result for the headline model: ROC-AUC 0.886, macro-F1 0.781 at the 0.5 threshold, accuracy 0.814, at-risk PR-AUC 0.944, Brier 0.127, ECE 0.009. The full model reaches ROC-AUC 0.929 in CV and 0.932 held out.

> **In plain terms: cross-validation, the held-out set and hyperparameters.** Cross-validation scores a model on households it was not trained on by rotating which fifth of the training data is held back. Here the fifths are cut along survey clusters (PSUs), so whole villages or urban blocks are held back together.
>
> The spread of the five fold scores matters as much as their average. A fold-to-fold standard deviation of 0.006 means a rerun with a different split would move the score by about that much. A gap between two models smaller than that is not evidence that one is better.
>
> The held-out set is a separate group of 8,299 households in 472 PSUs, set aside before any choice was made and scored once at the end. Cross-validation guides the choices, so its scores are slightly optimistic; the held-out score is the one measurement no choice was tuned to.
>
> A hyperparameter is a setting chosen before training, such as how deep a tree may grow. It is not learned from the data, so the usual approach is to try many combinations and keep the best.

---

## Notebook walkthrough

### Cell 1: load the split and define the feature sets

Loads the feature table (`savings_goal.io.load_features`), recovers the column lists (`savings_goal.features.engineer.spec_from_frame`), and separates training and held-out rows on the stored `Is_Test` flag. The held-out fold was drawn when the features were built (`savings_goal.evaluation.cv.grouped_test_mask`, the first fold of the same grouped splitter), so every phase uses the same partition and no PSU appears on both sides.

`FEATURE_SETS` maps "deployable" to `spec.deployable_numeric` and "full" to `spec.numeric`; the four categoricals are added to both. The printout confirms 33,219 training households in 1,989 PSUs, 8,299 held-out households in 472 PSUs, and prevalence 0.319 on track / 0.681 at risk. Two extra metrics, precision and recall on the on-track class, are added to `savings_goal.evaluation.metrics.fold_metrics`.

### Cell 3: seven families on both feature sets (Q1, Q2)

`savings_goal.models.pipeline.model_zoo` builds each family behind the shared preprocessor (`savings_goal.models.pipeline.preprocessor`): median imputation, `Debt_To_Income` winsorised at the 99th percentile inside each training fold, one-hot categoricals, and scaling for the linear models only. Logistic regression, the linear SVM, the decision tree (depth 8) and the random forest (300 trees) use `class_weight="balanced"`; the two boosters do not. XGBoost runs at its untuned defaults (`XGB_DEFAULT`: 400 trees, depth 6, learning rate 0.08, subsample 0.9, colsample 0.9). The single income threshold from Phase 3 is added as an eighth row, fitted on `Log_Income` alone.

Each model goes through `savings_goal.evaluation.cv.cross_validate_grouped` on the same five grouped folds. Besides macro-F1 and ROC-AUC, every family is scored on at-risk PR-AUC, Brier score, log-loss and expected calibration error. The linear SVM has no probabilities, so those three are left blank for it.

Deployable set, ranked by ROC-AUC (the tuned XGBoost row is added in cell 6):

| Model | Accuracy | Macro-F1 | F1 sd | ROC-AUC | At-risk PR-AUC | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost (tuned) | 0.815 | 0.782 | 0.006 | 0.882 | 0.942 | 0.128 | 0.011 |
| HistGradientBoosting | 0.811 | 0.778 | 0.007 | 0.880 | 0.941 | 0.129 | 0.012 |
| Logistic regression | 0.791 | 0.773 | 0.007 | 0.876 | 0.938 | 0.144 | 0.103 |
| Linear SVM | 0.791 | 0.774 | 0.007 | 0.876 | 0.938 | n/a | n/a |
| XGBoost (untuned) | 0.808 | 0.774 | 0.006 | 0.875 | 0.938 | 0.132 | 0.023 |
| Random forest | 0.798 | 0.777 | 0.005 | 0.870 | 0.934 | 0.142 | 0.082 |
| Decision tree | 0.768 | 0.751 | 0.009 | 0.850 | 0.913 | 0.159 | 0.104 |
| Single income threshold | 0.778 | 0.742 | 0.004 | 0.739 | 0.808 | 0.166 | 0.003 |
| Majority baseline | 0.681 | 0.405 | 0.000 | 0.500 | 0.681 | 0.319 | 0.319 |

Full set, same models:

| Model | Macro-F1 | ROC-AUC | At-risk PR-AUC | Brier | ECE |
| --- | --- | --- | --- | --- | --- |
| XGBoost (tuned) | 0.836 | 0.929 | 0.965 | 0.098 | 0.014 |
| XGBoost (untuned) | 0.836 | 0.929 | 0.965 | 0.099 | 0.023 |
| HistGradientBoosting | 0.835 | 0.928 | 0.965 | 0.099 | 0.015 |
| Logistic regression | 0.819 | 0.920 | 0.960 | 0.115 | 0.082 |
| Linear SVM | 0.817 | 0.920 | 0.960 | n/a | n/a |
| Random forest | 0.823 | 0.915 | 0.957 | 0.116 | 0.088 |
| Decision tree | 0.784 | 0.882 | 0.928 | 0.138 | 0.085 |

> **In plain terms: the seven families.** Ordered roughly from least to most flexible:
> - **Majority baseline**: always answers "not on track".
> - **Logistic regression**: gives each feature a weight, adds them up, and turns the total into a probability. Straight-line boundaries only.
> - **Linear SVM**: also a straight-line boundary, placed to leave the widest margin between the classes. It outputs a score, not a probability.
> - **Decision tree**: one flowchart of yes/no questions, here up to 8 deep.
> - **Random forest**: 300 trees, each grown on a random sample of rows and columns, voting together.
> - **HistGradientBoosting and XGBoost**: trees built one after another, each fitted to the errors the earlier trees still make. Two implementations of the same idea, usually the strongest family on tabular data.
>
> Kernel SVMs, which can draw curved boundaries, compare every household with every other one. At 33,219 training rows that is about a billion pairs per fit, so they were left out on cost.

What the tables show:

- Against the income threshold, tuned XGBoost on the deployable set gains +0.040 macro-F1 and +0.143 ROC-AUC. Against deployable logistic regression the gain is +0.009 macro-F1 and +0.006 ROC-AUC. Most of the step from the income rule to the best model is already available from a linear model on the same 13 features.
- The deployable boosters are close together. Tuned XGBoost and HistGradientBoosting differ by 0.004 macro-F1 and 0.002 ROC-AUC, against fold standard deviations of about 0.006. XGBoost is kept because it scored highest and its tuned configuration is stored for later phases; HistGradientBoosting would be a reasonable substitute.
- The full set adds 0.03 to 0.05 ROC-AUC to every family (tuned XGBoost 0.882 to 0.929). Phase 1's leakage check links that lift to the shares rebuilding total spending together with household size and income.
- Calibration splits the families. The boosters have ECE 0.011 to 0.023; logistic regression (0.103), the random forest (0.082) and the decision tree (0.104) are off by 8 to 10 points on average. Those three use balanced class weights, which push predicted probabilities toward "on track" by design. Their ranking is fine (ROC-AUC 0.870 to 0.876 for the two strongest); their probabilities are not usable as stated without recalibration.
- The income threshold has the lowest ECE in the table (0.003). A depth-1 tree predicts the observed on-track rate on each side of the cut, so its two probabilities are calibrated by construction; it just has only two of them.

> **In plain terms: calibration, Brier score and ECE.** A model is calibrated if, among households it gives a 30% chance of being on track, about 30% are. Expected calibration error (ECE) groups predictions into ten bins by probability and averages the gap between predicted and observed rates, weighted by bin size: 0.011 means predictions are off by about one percentage point on average. The Brier score is the mean squared difference between the predicted probability and the 0/1 outcome; lower is better, and it rewards both good ranking and good calibration. Calibration matters whenever a probability is read as a probability, for example when a team sets a budget from expected counts.

### Cell 5: hyperparameter search (Q3)

`RandomizedSearchCV` draws 15 combinations from `savings_goal.models.pipeline.XGB_SEARCH_SPACE` (`max_depth` 4/6/8/10, `learning_rate` 0.03/0.08/0.15, `n_estimators` 200/400/700, `subsample` 0.7/0.9/1.0, `min_child_weight` 1/5/20), scores each on macro-F1, ROC-AUC and log-loss over the grouped folds, and refits on macro-F1. This runs once per feature set, through `savings_goal.models.pipeline.xgb_pipeline`.

| Feature set | Best macro-F1 | ROC-AUC | Chosen parameters |
| --- | --- | --- | --- |
| Deployable | 0.781 | 0.882 | depth 4, lr 0.08, 200 trees, subsample 0.7, min_child_weight 1 |
| Full | 0.837 | 0.930 | depth 4, lr 0.15, 200 trees, subsample 0.9, min_child_weight 5 |

Sensitivity is the standard deviation, across the levels of one parameter, of the mean macro-F1 of the runs at that level:

| Parameter | Deployable | Full |
| --- | --- | --- |
| `max_depth` | 0.0085 | 0.0015 |
| `n_estimators` | 0.0042 | 0.0010 |
| `subsample` | 0.0026 | 0.0014 |
| `learning_rate` | 0.0010 | 0.0042 |
| `min_child_weight` | 0.0009 | 0.0002 |

On the deployable set, depth is the parameter that matters, and the search picks the shallowest depth in the grid (4, against the default 6). Fewer trees (200 against 400) and stronger row subsampling (0.7) point the same way: with 13 features, the default configuration has more capacity than the signal needs and gives up a little macro-F1 to overfitting. On the full set every parameter has sensitivity below 0.005.

A second search sweeps logistic regression's `C` over 0.01, 0.1, 1, 10 and 100 on the deployable set. The best is `C = 0.01` at macro-F1 0.774, against 0.773 at the default `C = 1`: the linear model is insensitive to the strength of its penalty here.

> **In plain terms: randomised search and the five settings.** A randomised search tries a fixed number of random combinations and keeps the best, instead of trying every combination (324 here, each needing five fits). Usually only one or two settings matter, so a random sample finds a near-best combination for a fraction of the compute.
> - **`n_estimators`**: how many trees to build.
> - **`learning_rate`**: how much of each new tree's correction to apply. Smaller steps need more trees but overfit less.
> - **`max_depth`**: how many questions deep each tree may go. Deeper trees can capture combinations such as "large household and low income".
> - **`subsample`**: the fraction of households each tree sees. Below 1 it adds randomness that helps the ensemble generalise.
> - **`min_child_weight`**: how much data a branch needs before it may split again. Higher values stop the tree from carving out tiny, noisy groups.
>
> Overfitting means learning the training households' quirks so closely that performance on new households gets worse. The last three settings exist to limit it.

> **In plain terms: `C`.** `C` sets how strongly logistic regression is pushed to keep its weights small. Small `C` means a strong penalty and a simpler model; large `C` means a weak penalty. The sweep covers a factor of 10,000, and macro-F1 moves by about 0.001 across it.

### Cell 6: score the tuned configurations on the same folds and save

The search's best score comes from scikit-learn's own scorer. To put the tuned model in the same table as everything else, the cell refits each tuned configuration through `cross_validate_grouped` with the same folds and metric functions as cell 3, and appends it as "XGBoost (tuned)". On the deployable set that gives macro-F1 0.782 and ROC-AUC 0.882; the search reported 0.781. The table, the text and the held-out evaluation all describe this one configuration.

`savings_goal.models.pipeline.save_model_final` writes `results/model_final.json`: the headline set name, the untuned defaults, the search space, the CV description, and for each feature set the chosen parameters, search score, grouped-CV metrics, parameter sensitivity and feature count. Phases 5, 7 and 8 read the tuned parameters from this file. The full comparison, including fold standard deviations of macro-F1 and ROC-AUC, goes to `results/model_comparison.csv`.

Tuning gain on the headline set: +0.007 macro-F1 (0.774 to 0.782), about one fold standard deviation. On the full set the tuned and untuned XGBoost are tied at 0.836.

### Cell 8: precision or recall, from the at-risk side (Q4)

Every model treats `Goal_Met = 1` ("on track") as the positive class, so precision and recall in the tables above describe how well the model finds households that are doing fine. The households a savings nudge would target are the at-risk ones, `Goal_Met = 0`. This cell computes the at-risk view explicitly.

> **In plain terms: why the class matters.** Software has to call one class "positive", and that choice is bookkeeping. Precision and recall are not symmetric between classes, so a recall figure for "on track" says nothing direct about how many at-risk households are found. Reporting the on-track figure to someone who will contact at-risk households answers a question they did not ask.

`savings_goal.evaluation.cv.oof_predict_proba` produces out-of-fold probabilities for the tuned deployable model on the training split. The at-risk score is `1 - p`. From the precision-recall curve the cell takes the threshold that maximises at-risk F1, then, for recall targets of 80%, 90% and 95%, the highest threshold that still reaches each target.

| Operating point (out-of-fold, training split) | At-risk threshold | Precision | Recall |
| --- | --- | --- | --- |
| Max F1 (F1 0.869) | 0.425 | 0.827 | 0.915 |
| Recall at least 80% | 0.654 | 0.893 | 0.800 |
| Recall at least 90% | 0.464 | 0.838 | 0.900 |
| Recall at least 95% | 0.316 | 0.789 | 0.950 |

> **In plain terms: out-of-fold scores.** Each training household is scored by a model trained on the other four folds, which never saw its PSU. Choosing a threshold on these scores uses only training data, so the held-out set can still check the result.

> **In plain terms: threshold tuning.** The model outputs a probability. The threshold turns it into a decision: flag every household whose at-risk score is above 0.425, or 0.316, or wherever you choose. A lower threshold flags more households, so fewer at-risk ones are missed (recall rises) and more of those flagged were fine (precision falls). The table is a set of settings for the same model; nothing is retrained between rows. Which row to use depends on the cost of a contact against the cost of a miss, which is a business decision.

The case for favouring at-risk recall: a false positive is one low-cost nudge to a household that was already saving, while a false negative is a household heading for a shortfall that outreach never reaches. Moving from 80% to 95% recall costs 10.4 points of precision (0.893 to 0.789). At 95% recall, about four in five flagged households are at risk. Because the at-risk class is 68% of households, even a random list would be 68% precise; Phase 7 measures capture against that floor and against the best possible list.

### Cell 10: the single held-out evaluation

Both tuned configurations are refit on the full training split and scored once on the 8,299 held-out households, using `savings_goal.evaluation.metrics.probability_metrics` for the threshold-free numbers.

Headline model (tuned XGBoost, deployable set), threshold 0.5:

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| At risk | 0.849 | 0.883 | 0.866 | 5,648 |
| On track | 0.727 | 0.667 | 0.696 | 2,651 |
| Macro average | 0.788 | 0.775 | 0.781 | 8,299 |

| True \ predicted | At risk | On track |
| --- | --- | --- |
| At risk | 4,985 | 663 |
| On track | 884 | 1,767 |

> **In plain terms: the confusion matrix.** Rows are the truth, columns are the model's call. The diagonal holds the correct calls: 4,985 at-risk households flagged and 1,767 on-track households cleared. Off the diagonal, 663 at-risk households were cleared (the costly miss) and 884 on-track households were flagged (the cheap one). Precision, recall, F1 and accuracy are all arithmetic on these four counts.

At the at-risk threshold chosen on out-of-fold training scores (0.425), the held-out set gives precision 0.826 and recall 0.913, with 6,243 of 8,299 households flagged. The training estimate was 0.827 / 0.915, so the threshold carries over to unseen PSUs.

Threshold-free metrics on the held-out set:

| Feature set | ROC-AUC | On-track PR-AUC | At-risk PR-AUC | Brier | Log-loss | ECE | Macro-F1 @0.5 | Accuracy @0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Deployable (headline) | 0.886 | 0.775 | 0.944 | 0.127 | 0.391 | 0.009 | 0.781 | 0.814 |
| Full (diagnostic) | 0.932 | 0.873 | 0.967 | 0.097 | 0.306 | 0.013 | 0.836 | 0.861 |

The held-out numbers match the grouped-CV estimates closely (deployable ROC-AUC 0.886 against 0.882, macro-F1 0.781 against 0.782), so tuning on the grouped folds did not produce a visibly optimistic estimate. `savings_goal.io.write_result` stores these metrics, the tuned at-risk threshold, the out-of-fold recall targets and the held-out size in `results/model_test.json`.

One caveat travels with every figure here. Phase 1 found that 55.9% of IHDS households report consumption above income, so `Goal_Met` counts saving from a noisy income-minus-consumption difference. The precision and recall above are measured against that label. They support ranking households and choosing a threshold; they do not mean that 83% of flagged households are in financial distress.

### Cell 11: plot

`results/model_comparison.png` is a horizontal bar chart of grouped-CV ROC-AUC for every model on both feature sets, sorted by the deployable score, with the tuned XGBoost values (0.882 deployable, 0.929 full) in the title. It shows the gap between the two feature sets as a constant offset across families and the small spread among the strong models within each set.

---

## What this means for later phases

| Phase | Consequence |
| --- | --- |
| 5: Explainability | Explain the tuned XGBoost on the deployable set with SHAP, reading parameters from `results/model_final.json`. Do not read logistic-regression coefficients as effects. Expect income to dominate (Phase 3: income alone gives ROC-AUC 0.835). The full model can be explained alongside as a diagnostic. |
| 6: Clustering | Independent of model choice. |
| 7: Business translation | Use the deployable model's out-of-fold or held-out scores, and compare targeting with the income rule and with the ceiling at each budget. Its probabilities are well calibrated (held-out ECE 0.009), so expected counts can be read from them directly. The at-risk threshold of 0.425 (held-out precision 0.826, recall 0.913) is one available operating point. |
| 8: Savings-rate regression | Reuses the same split, grouped folds and deployable feature set, so its results line up with this phase's. |
