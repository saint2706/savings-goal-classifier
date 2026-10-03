# Phase 3: Baseline

**Source:** [README § Phase 3: Baseline](../README.md#phase-3-baseline)
**Notebook:** [`notebooks/03_baseline.ipynb`](../notebooks/03_baseline.ipynb)
**Builds on:** [Phase 2](phase2.md)
**Artifacts:** `results/baseline.csv`, `results/baseline.json`, `results/baseline.png`

This phase measures how much of the problem is solved by very simple predictors before any tuned model is built. Everything runs on the training split (33,219 households, 1,989 PSUs; Goal_Met rate 0.3192) under PSU-grouped 5-fold cross-validation. The held-out fold is not touched.

The short answer: a single cut-off on income reaches macro-F1 0.742. That number, not the majority class, is the bar [Phase 4](phase4.md) has to clear.

> **In plain terms: what a baseline is for.** A score means little on its own. "Macro-F1 0.78" is only good or bad relative to something, and that something has to be fixed before the real model is built, so the comparison cannot be picked afterwards to flatter the result.
>
> A baseline is a deliberately simple predictor used as that yardstick. This phase builds two kinds:
> - Chance baselines use no information about the household: always give the most common answer, or guess at random in the right proportions. They set the floor.
> - Simple-rule baselines use one obvious variable in the crudest way, here a single cut-off on income. This is the harder comparison, because it is what someone with a spreadsheet could do in an afternoon.

---

## Research questions & answers

| # | Question | Answer |
| --- | --- | --- |
| 1 | What accuracy/F1 does a majority-class or simple single-rule baseline achieve? | Majority class: accuracy 0.681, macro-F1 0.405, and precision and recall on "on track" both 0. Stratified random guessing: accuracy 0.570, macro-F1 0.502. A single income threshold (re-learned in each fold, mean Rs 122,249 per year, range Rs 110,377 to 126,545): accuracy 0.778, macro-F1 0.742, ROC-AUC 0.739. |
| 2 | What does plain logistic regression achieve using only income and 1-2 expense shares? | Income alone: ROC-AUC 0.835, macro-F1 0.732. Adding `Groceries_Share`: ROC-AUC 0.875 (+0.041). Income + household size + `Groceries_Share`: 0.894, higher than the 13-feature deployable set (0.876). All 29 features: 0.920. The food share adds +0.044 once household size is present, which is the food-size reconstruction of total spend found by the leakage check in [Phase 1](phase1.md), so the share-based rows are diagnostic, not deployable. |

---

## Notebook walkthrough

### Cell 1: load, metrics and the shared evaluation helper

Loads the engineered feature table with `savings_goal.io.load_features` and recovers the column lists with `savings_goal.features.engineer.spec_from_frame`. Only rows with `Is_Test == False` are kept. The grouping column is `IDPSU` (`savings_goal.config.GROUP_COL`).

The metric set comes from `savings_goal.evaluation.metrics.fold_metrics(with_proba=False)` (accuracy, macro-F1, ROC-AUC, at-risk PR-AUC) plus precision and recall for the positive class, `Goal_Met = 1` ("on track"). The helper `evaluate` passes a model and its columns to `savings_goal.evaluation.cv.cross_validate_grouped` with the folds from `savings_goal.evaluation.cv.grouped_cv` (`StratifiedGroupKFold(5, shuffle=True, random_state=42)`), and returns the fold means plus the standard deviation of macro-F1.

Every baseline in this notebook uses the same folds as the model comparison in Phase 4, so the rows in both phases are directly comparable.

> **In plain terms: grouped cross-validation.** Cross-validation estimates how a model does on households it has not seen. The training households are split into 5 parts; the model is trained on 4 and scored on the fifth, five times over, and the scores are averaged. IHDS samples households in clusters (a village or an urban block, a PSU). Neighbours share prices, jobs and interviewers, so a household's neighbours can give away its answer. Grouping keeps every household of a PSU in the same fold, so the model is always scored on villages and blocks it never trained on. "Stratified" means each fold keeps roughly the same 32% / 68% class mix.

### Cell 3: chance baselines and the single income threshold (Q1)

Four rows, all under grouped CV:

| Baseline | Accuracy | Macro-F1 | ROC-AUC | At-risk PR-AUC | Precision (on track) | Recall (on track) |
| --- | --- | --- | --- | --- | --- | --- |
| Majority class (always "not met") | 0.681 | 0.405 | 0.500 | 0.681 | 0.000 | 0.000 |
| Stratified random | 0.570 | 0.502 | 0.502 | 0.682 | 0.322 | 0.316 |
| Single income threshold (depth-1 tree) | 0.778 | 0.742 | 0.739 | 0.808 | 0.660 | 0.632 |
| Income only (depth-3 tree) | 0.781 | 0.727 | 0.827 | 0.887 | 0.717 | 0.526 |

The two chance baselines are scikit-learn `DummyClassifier`s. The majority classifier always answers "not met", so it reaches 0.681 accuracy while never identifying a single on-track household. The stratified dummy has lower accuracy (0.570) but higher macro-F1 (0.502), because it at least predicts both classes. Quoting only one of them would flatter a real model on whichever metric that dummy is weak on, so both are reported.

> **In plain terms: macro-F1, and why accuracy is not the headline.** For one class, precision is "of the households I flagged, how many were right" and recall is "of the households that really are in this class, how many did I find". F1 combines the two into one number. Macro-F1 computes F1 separately for "on track" and "at risk" and averages them, so a model that ignores one class scores badly. Accuracy has no such protection: 68% of households are at risk, so a constant answer gets 0.68 accuracy and learns nothing.

The single-rule baseline is a depth-1 decision tree on `Log_Income` alone, with a median imputer in front. A callback (`record_threshold`) reads the split point from each fitted fold and converts it back to rupees. The cut-off is re-learned inside every training fold, so no fold is scored with a threshold that saw it: across the five folds it lands between Rs 110,377 and Rs 126,545 a year (mean Rs 122,249).

> **In plain terms: a depth-1 decision tree.** A decision tree is a flowchart of yes/no questions that ends in a prediction. Depth 1 allows one question, so the whole model is a single line on the income axis: households above it are called on track, households below it at risk. The only thing learned is where to draw the line. A depth-3 tree asks three nested questions and can cut income into up to eight bands.

The depth-3 tree ranks households better (ROC-AUC 0.827 against 0.739) but has lower macro-F1 (0.727 against 0.742). Its finer bands give a better ordering, while its default 0.5 decision line sits further toward the majority class, so on-track recall drops to 0.526.

> **In plain terms: ranking versus deciding.** A classifier does two jobs, scored by different numbers.
> - Ranking: put households in order, most likely to save first. ROC-AUC measures only this. It is threshold-free: wherever you put the cut-off, ROC-AUC does not change. 0.5 is a coin flip; 1.0 is a perfect ordering.
> - Deciding: turn that order into yes/no by drawing a line (by default at probability 0.5). Macro-F1 and accuracy measure this and depend on where the line sits.
>
> A model can order households better and still make worse calls if its line is in the wrong place. When the two metrics disagree, the usual cause is the threshold. [Phase 4](phase4.md) tunes that threshold on purpose.

> **In plain terms: at-risk PR-AUC.** Precision-recall AUC summarises, across every possible cut-off, how precise the flagged list stays as you try to find more of the at-risk households. A random ordering scores the at-risk share itself, 0.681 here, so that is the floor to compare against, not 0.

### Cell 5: logistic regression on income plus a few features (Q2)

Fits logistic regression (`savings_goal.models.pipeline.logistic`, with `class_weight="balanced"`) behind `savings_goal.models.pipeline.preprocessor(..., linear=True)`, which imputes medians, winsorises `Debt_To_Income` at the 99th percentile inside each training fold, and standardises. The four categoricals (occupation, area type, caste group, religion) are one-hot encoded only for the two larger sets. In the 29-feature set the preprocessor drops `Groceries_Share` as the reference share, because the 11 shares sum to one.

| Feature set | Accuracy | Macro-F1 | ROC-AUC | At-risk PR-AUC |
| --- | --- | --- | --- | --- |
| Income only | 0.752 | 0.732 | 0.835 | 0.910 |
| Income + groceries share | 0.788 | 0.770 | 0.875 | 0.931 |
| Income + household size | 0.770 | 0.750 | 0.850 | 0.922 |
| Income + size + groceries share | 0.806 | 0.789 | 0.894 | 0.944 |
| Deployable (income, demographics, debt; 13 features) | 0.791 | 0.773 | 0.876 | 0.938 |
| All 29 features | 0.835 | 0.819 | 0.920 | 0.960 |

Three readings:

1. Income-only logistic regression has a much better ranking than the income threshold (ROC-AUC 0.835 against 0.739) but slightly lower macro-F1 (0.732 against 0.742). The balanced class weights move its decision line toward "on track" (recall 0.750, precision 0.587). This is the ranking-versus-deciding gap again.
2. `Groceries_Share` adds +0.041 ROC-AUC on top of income alone and +0.044 on top of income and household size. A naive reading would treat this as an independent behavioural signal. The leakage check in Phase 1 shows why it is larger with size present: food spending in rupees grows with household size (R² 0.29, elasticity 0.63), so size divided by food share approximates total spending, and income minus total spending is the savings rate the label is built from. A three-variable model with the food share (0.894) outranks the full 13-feature deployable set (0.876) for that reason.
3. The deployable set, which uses only what is known about a household at onboarding (income, demographics, debt and the four categoricals), reaches ROC-AUC 0.876 and macro-F1 0.773 with a linear model. That is the realistic starting point for Phase 4. The 29-feature row (0.920) includes the spending shares and is kept as a diagnostic upper bound.

> **In plain terms: reading a progression table.** Each row adds something to a simpler row, so the change in ROC-AUC between rows shows what that addition bought. The comparisons that matter here are "income only" to "income + groceries share" (+0.041) and "income + household size" to "income + size + groceries share" (+0.044). The same column adds more when size is already in the model. That is a sign of two features working together to rebuild a quantity, not of one feature carrying its own information.

### Cell 6: save and plot

Concatenates both tables, writes `results/baseline.csv`, and writes `results/baseline.json` with `savings_goal.io.write_result` (the per-fold income thresholds, their mean, the full table and the split description). The figure, `results/baseline.png`, is a horizontal bar chart of accuracy, macro-F1 and ROC-AUC for every row, with a reference line at 0.5. It puts the majority class's 0.68 accuracy next to its 0.41 macro-F1 so the gap is visible at a glance.

---

## What this means for later phases

| Phase | Consequence |
| --- | --- |
| 4: Model comparison | The bar is the income threshold (macro-F1 0.742, ROC-AUC 0.739) and the deployable logistic regression (macro-F1 0.773, ROC-AUC 0.876), on the same grouped folds. Report margins over these, not over the majority class. |
| 4: Feature sets | Share-based rows partly rebuild total spending through household size, so the headline model uses the deployable set and the full set is reported alongside as a diagnostic. |
| 5: Explainability | Income alone gives ROC-AUC 0.835 of the deployable logistic model's 0.876, so income should dominate the attributions of any deployable model. |
| 7: Business translation | Any targeting result has to be compared with the income rule, not with random contact. |
