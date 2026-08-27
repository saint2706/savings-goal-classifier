# Phase 4 — Model Comparison

**Source:** [README § Phase 4 — Model Comparison](../README.md#phase-4--model-comparison)
**Notebook:** [`notebooks/04_model_comparison.ipynb`](../notebooks/04_model_comparison.ipynb)
**Builds on:** [Phase 3](phase3.md)
**Artifacts:** `results/model_comparison.csv`, `results/model_comparison.png`

At **2.13:1 with 13,256 minority cases** (Phase 1), this is an ordinary near-balanced classification problem. Class weighting is therefore optional rather than essential, and macro-F1 and ROC-AUC are straightforwardly usable without the contortions a severe imbalance would force.

---

## Research questions & answers

| # | Question | Answer |
| --- | --- | --- |
| 1 | Which model families are appropriate given the data? | Seven: majority baseline, logistic regression, linear SVM, decision tree, random forest, histogram gradient boosting, and XGBoost. At n = 41,518 with 28 mixed features, kernel SVMs are impractical (O(n²)) and were excluded on cost, not principle. **XGBoost wins** at CV macro-F1 **0.8371** / ROC-AUC **0.9306**. |
| 2 | What validation strategy fits the class balance found in Phase 1? | 5-fold **stratified** k-fold on an 80% training split, with a 20% test set (8,304 households) held out and touched exactly once. Stratification still matters at 2.13:1, but as insurance rather than necessity — fold-to-fold macro-F1 sd is 0.0018–0.0052, so the estimates are stable. |
| 3 | What hyperparameter search method is used, and what parameters move performance most? | 15-iteration randomised search over 5 XGBoost parameters, plus a 5-point sweep on logistic regression's `C`. **The search is worth +0.0006 macro-F1** (0.8371 → 0.8377) — statistically nothing. `learning_rate` matters most (score sd 0.0040 across levels); `max_depth` matters least (0.0003). |
| 4 | Is precision or recall more important given the business framing? | **Recall on the at-risk class**, because the intervention is a low-cost nudge and a missed at-risk household is a customer who silently fails their goal. The good news is the trade-off is cheap here: at the default threshold the at-risk class already gets precision 0.887 / recall 0.913, and pushing recall to **95%** costs only ~3 points of precision (0.856). |

> **In plain terms — cross-validation, the held-out set, and hyperparameters.** Three ideas run through this whole phase.
>
> **Cross-validation** answers "how well would this model do on households it has never seen?" without wasting data. Split the training households into 5 equal parts (**folds**); train on 4 and score on the 1 held back; repeat 5 times so each fold takes a turn as the scorer; average the five scores. Every household gets predicted exactly once by a model that never saw it. **Stratified** means each fold is built to contain the same 32%/68% mix as the whole — otherwise a fold could land with an unrepresentative share of on-track households and give a misleading score.
>
> The **spread** across those five scores is as informative as the average. A fold-to-fold standard deviation of 0.002 means that if you reran everything with a different random split, the score would wobble by about that much. **Any difference between two models smaller than that wobble is not a real difference** — a rule applied repeatedly below, and the reason several apparent improvements in this phase are called nothing.
>
> The **held-out test set** is a further 20% of households (8,304) locked away before anything begins and looked at exactly once, at the very end. Cross-validation guides choices, so its scores are gently optimistic — after enough decisions have been steered by those folds, some tuning to their quirks has crept in. The test set is the only truly untouched measurement, and it is spent once. Peeking early and then continuing to tune quietly destroys it.
>
> A **hyperparameter** is a setting you choose *before* training, as opposed to the coefficients the model learns *during* training — how deep the trees may grow, how large a step each round takes. They are not learned from the data, so the usual approach is to try many combinations and keep the best, which is what "hyperparameter search" means.

---

## Notebook walkthrough

### Cell 1 (code) — Load, and split off the test set immediately

The 20% test split is made in the **first cell**, before any model is defined, and is not referenced again until the final cell.

**Why up front rather than at the end:** it makes accidental leakage structurally difficult. Every intermediate decision in this notebook — model choice, hyperparameter search, threshold tuning — reads `X_train` only. If the split happened later, any of those steps could have quietly seen the test data.

### Cell 3 (code) — The seven-family comparison (Q1, Q2)

| Model | CV accuracy | **CV macro-F1** | CV precision | CV recall | F1 sd | CV ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| **XGBoost** | 0.8614 | **0.8371** | 0.8064 | 0.7446 | 0.0018 | **0.9306** |
| HistGradientBoosting | 0.8606 | 0.8360 | 0.8065 | 0.7412 | 0.0021 | 0.9306 |
| Random Forest | 0.8512 | 0.8249 | 0.7914 | 0.7253 | 0.0032 | 0.9170 |
| Logistic Regression | 0.8343 | 0.8186 | 0.6984 | **0.8466** | 0.0037 | 0.9213 |
| Linear SVM | 0.8332 | 0.8177 | 0.6961 | 0.8479 | 0.0039 | — |
| Decision Tree | 0.8067 | 0.7889 | 0.6614 | 0.8091 | 0.0052 | 0.8847 |
| Majority baseline | 0.6807 | 0.4050 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |

> **In plain terms — the seven contenders.** They differ in how much structure they can express, and they are lined up roughly from simplest to most flexible:
> - **Majority baseline** — always answers "not on track". Learns nothing; marks the floor.
> - **Logistic regression** — weighs each feature, adds up the weights, converts the total to a probability. Straight lines only.
> - **Linear SVM** (support vector machine) — also draws a single dividing line, but positions it to leave the widest possible margin between the two classes.
> - **Decision tree** — one flowchart of yes/no questions, as in [Phase 3](phase3.md) but allowed to grow deep. Flexible, and prone to memorising its training data.
> - **Random forest** — hundreds of trees, each grown on a random slice of rows and columns, all voting. The randomness cancels out any single tree's memorisation.
> - **Histogram gradient boosting** and **XGBoost** — trees again, but built in sequence rather than in parallel: each new tree is fitted to the mistakes the current ensemble is still making, so the model corrects itself step by step. These two are different implementations of the same idea, and this family is the usual winner on tabular data like this.
>
> **Why kernel SVMs were excluded.** A **kernel** SVM can draw curved boundaries, but it works by comparing every household against every other one. That is **O(n²)** — "order n squared", meaning the work grows with the *square* of the number of rows. At 41,518 households that is roughly 1.7 billion comparisons, so it was dropped on cost, not because it would have failed.

**Why macro-F1 is the ranking metric and accuracy is not:** the majority baseline scores 0.6807 accuracy while never predicting the positive class. Macro-F1 weights both classes equally and so cannot be gamed by ignoring one.

**Why the linear models look worse than they are.** Logistic regression and the linear SVM have the *highest recall* in the table (0.847, 0.848) and the lowest precision (0.698, 0.696). That is `class_weight="balanced"` doing exactly what it is asked: reweighting the loss to favour the minority class, which shifts the decision boundary toward predicting "on track" more often.

> **In plain terms — what "balanced" did, and why it makes this comparison unfair.** A model is fitted by minimising its **loss** — a running tally of how wrong it is, which training tries to make as small as possible. `class_weight="balanced"` tells it that mistakes on the smaller class count for more, so the model becomes readier to say "on track". That catches more genuinely on-track households (**recall** rises) at the cost of more false alarms (**precision** falls).
>
> The two linear models were given this instruction and the boosted models were not — so they are not simply better or worse, they are **aimed differently**. Comparing them on macro-F1 partly compares the settings rather than the models. ROC-AUC is the fair column precisely because it ignores where the line is drawn (see [Phase 3](phase3.md) on ranking vs deciding), and there the gap shrinks from what macro-F1 suggests. The boosted models were fitted **without** class weighting, because at 2.13:1 they do not need it, and they land on a more even precision/recall split. The comparison is therefore between differently-calibrated models, and the fair reading is the ROC-AUC column — which is threshold-free. There, logistic regression (0.9213) sits much closer to XGBoost (0.9306) than macro-F1 suggests.

**The honest margin.** Against the **single income threshold** from Phase 3 (macro-F1 0.7425), XGBoost is worth **+0.095**. Against the majority baseline it is worth +0.432, but that comparison flatters the model and Phase 3 exists to prevent it being quoted.

**Why the two boosting implementations are treated as tied:** 0.8371 vs 0.8360 is a gap of 0.0011 against a fold-to-fold sd of 0.0018–0.0021 — well inside noise, and their ROC-AUCs are identical to four decimals. XGBoost is selected because it scored marginally higher, not because it is meaningfully better; `HistGradientBoosting` would be a defensible swap and has the advantage of being a scikit-learn built-in with no extra dependency.

**Excluded families and why:** RBF-kernel SVM scales roughly quadratically in n and is impractical at 41,518 rows. A neural net (MLP) was excluded because Phase 3 showed steep diminishing returns — the marginal gain over the boosted trees would not justify the tuning surface. Both exclusions are cost decisions, and neither is likely to change the ranking.

### Cell 5 (code) — Hyperparameter search (Q3)

15-iteration randomised search over `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `min_child_weight`, scored on macro-F1.

> **In plain terms — randomised search, and the five knobs.** A **randomised search** picks 15 random combinations of settings, cross-validates each, and keeps the best. The alternative — a **grid search** trying every combination — would need thousands of fits here; random sampling finds nearly as good a combination for a fraction of the compute, because usually only one or two settings matter much.
>
> The five knobs, all governing how the sequence of corrective trees is built:
> - **`n_estimators`** — how many trees to build in total.
> - **`learning_rate`** — how much of each new tree's correction to actually apply. Small values mean cautious steps, requiring more trees but overfitting less.
> - **`max_depth`** — how many questions deep each tree may go. Deeper trees can capture combinations of features ("high food share *and* low income"); shallow ones mostly capture features acting separately.
> - **`subsample`** — what fraction of households each tree sees. Below 1 it injects deliberate randomness, which helps the ensemble generalise.
> - **`min_child_weight`** — how much data a branch must contain before the tree is allowed to split it further. Higher values stop the tree from carving out tiny groups that are really noise.
>
> **Overfitting**, referred to throughout: learning the training households' quirks so precisely that performance on new households gets worse. It is the failure these last three knobs exist to prevent.

**Best parameters:** `learning_rate=0.15`, `n_estimators=200`, `max_depth=4`, `min_child_weight=5`, `subsample=0.9`
**Best score: 0.8377** — against **0.8371** for the untuned defaults.

**The search bought +0.0006 macro-F1, which is a third of one fold's standard deviation.** This is the phase's most useful negative result, and it confirms the prediction Phase 3 made from the feature-progression curve: the problem's difficulty lives in the data, not in the model configuration. A larger search would have been compute spent to move a number that does not move.

> **In plain terms — "a third of one fold's standard deviation".** The tuned model beat the default by 0.0006, while the score already wobbles by about 0.002 just from which households land in which fold. The improvement is smaller than the measurement's own jitter — so it is not an improvement that was detected, it is noise that happened to point upward. Rerun with a different random seed and it could as easily point down. This is the same "smaller than the wobble" rule from the top of this phase, and it is why more search would have been wasted compute.

**Which parameters matter, by sd of mean score across each level:**

| Parameter | Sensitivity |
| --- | --- |
| `learning_rate` | 0.00397 |
| `subsample` | 0.00090 |
| `n_estimators` | 0.00082 |
| `min_child_weight` | 0.00054 |
| `max_depth` | 0.00033 |

`learning_rate` dominates by 4×, which is the standard result for gradient boosting — it is the one parameter that trades directly against `n_estimators` and controls how much the ensemble can overfit. That **`max_depth` matters least** is more telling: the model does not need deep interactions, which is consistent with Phase 3's finding that two features (income and grocery share) already reach AUC 0.876 of the final 0.931.

**Note that the tuned model chose `max_depth=4`, shallower than the default 6.** Combined with the flat sensitivity, this says the signal is close to additive.

> **In plain terms — "additive", and the sensitivity numbers.** The sensitivity column asks: as this one knob is varied while the others move around it, how much does the score swing? A big swing means the setting matters; 0.0003 for `max_depth` means it barely does.
>
> A signal is **additive** when the features mostly contribute independently — income pushes the prediction one way, family size another, and you can add up the pushes. It is **interactive** when a feature's meaning depends on another's value: a high food share might mean one thing for a poor household and the opposite for a rich one. Depth is what lets a tree capture interactions, so a model that gains nothing from depth is telling you the interactions it needs are shallow. Worth flagging now, because [Phase 5](phase5.md) measures interactions directly and finds them to be 41.9% of the model's behaviour — a genuine tension with this reading, which Phase 5 addresses head-on.

### Cell 6 (code) — Logistic regression's regularisation sweep

`C` from 0.01 to 100: macro-F1 moves from 0.8159 to **0.8190** — a range of 0.003 across four orders of magnitude, monotonically increasing.

> **In plain terms — `C`.** `C` sets how hard the L2 penalty from [Phase 2](phase2.md) pushes: **small `C` = strong penalty** (coefficients forced small, model kept simple), **large `C` = weak penalty** (coefficients free to grow, model follows the data more closely). "Four **orders of magnitude**" means the setting was varied by a factor of 10,000 — from 0.01 to 100. That the score moved by only 0.003 across that entire range is the finding: the model is essentially indifferent to the penalty's strength, which is exactly what the next paragraph interprets.

**Why this is worth reporting rather than skipping:** Phase 2 found the 11 shares are **exactly singular** (VIF = ∞, they sum to 1), which means the model is only identifiable because of the L2 penalty. If the fit were badly conditioned, performance would be sharply sensitive to `C`. It is not — and that the best `C` is the *weakest* regularisation tested (100.0) says the penalty is doing structural work (making the solution unique) rather than being needed to control variance. The singularity is real but benign for prediction, exactly as Phase 2 predicted. **It remains fatal for coefficient interpretation**, which is Phase 5's problem.

### Cell 8 (code) — Precision vs recall for the business decision (Q4)

**The framing correction this cell makes:** the positive class is `Goal_Met = 1` ("on track"), but the class the business acts on is `Goal_Met = 0` — the at-risk households a savings nudge would target. Every precision/recall figure in the comparison table is for the *wrong class* for decision-making purposes, so the at-risk view is computed explicitly.

> **In plain terms — why "the wrong class" is a real problem.** Software has to nominate one class as the **positive** class, and that choice is arbitrary bookkeeping — here it fell to `Goal_Met = 1`, "on track". Every precision and recall figure printed so far therefore describes how well the model finds households that are *doing fine*.
>
> But nobody acts on those. The households the team would contact are the at-risk ones. Precision and recall are **not symmetric** — they say completely different things about each class — so the at-risk view has to be computed on purpose. This is an easy and consequential mistake: reporting the wrong class's recall to a stakeholder answers a question they did not ask, in the language of the one they did.

Out-of-fold, XGBoost at the default threshold:

> **In plain terms — "out-of-fold".** Every figure here comes from cross-validation, so each household was scored by a model trained without it. These are honest out-of-sample numbers, not the model grading its own homework. **Support** in the last column is simply how many households are in that class.

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| **not on track** | 0.885 | 0.916 | 0.900 | 22,609 |
| on track | 0.807 | 0.746 | 0.775 | 10,605 |

**The at-risk class is the easier one**, because it is the majority at 68% — the opposite of the usual situation, where the class you want to act on is the rare one.

Tuning the threshold on the at-risk score:

| Target | Precision | Recall | Threshold |
| --- | --- | --- | --- |
| Max F1 | 0.873 | 0.933 | 0.440 |
| Recall 80% | 0.941 | 0.800 | 0.757 |
| Recall 90% | 0.896 | 0.900 | 0.554 |
| **Recall 95%** | **0.856** | 0.950 | 0.368 |

> **In plain terms — what threshold tuning is.** The model outputs a probability, not a verdict. The **threshold** is the cut-off that turns it into one: flag every household scoring above 0.50, or above 0.37, or wherever you choose. Lowering the threshold flags more households, so you miss fewer at-risk ones (**recall** climbs) but include more who were fine (**precision** slips).
>
> The table reads as a menu of operating points on that single dial, all from the *same model* — nothing was retrained. Want to catch 95% of at-risk households? Set the cut-off at 0.368 and accept that 14.4% of those you contact were already saving. **This is a business choice, not a modelling one**, which is why [Phase 0](phase0.md) argued that keeping the threshold visible was a reason to prefer classification: the trade-off is stated on the page rather than buried in a default.

**The recommendation: optimise for recall on the at-risk class.** The asymmetry is in the costs. A false positive is one unnecessary low-cost nudge to a household that was already saving — mildly wasteful. A false negative is a household heading for a shortfall that the outreach never reaches, which is the failure the project exists to prevent (Phase 0).

> **In plain terms — false positive, false negative.** A **false positive** here is flagging a household as at risk when it was saving fine: the cost is one wasted nudge. A **false negative** is failing to flag a household that really is heading for a shortfall: the cost is that the programme never reaches someone it exists to help, and nobody ever finds out. Because the second mistake is much more expensive than the first, the threshold is deliberately set to make many more of the cheap mistake in order to make fewer of the expensive one.

**And the trade is unusually cheap here.** Moving from 80% to 95% recall costs 8.5 points of precision (0.941 → 0.856). Even at 95% recall, roughly six in seven flagged households are genuinely at risk — a direct consequence of the at-risk class being the majority.

**A caveat that must travel with any of these numbers:** Phase 1 established that 55.9% of IHDS households report consumption exceeding income, so `Goal_Met` is biased downward. These precision/recall figures are measured against a target that under-counts saving. They are sound for *ranking* households and for choosing an operating threshold; they should not be read as "89% of the households we flag are genuinely in financial distress."

### Cell 10 (code) — The single held-out evaluation

**XGBoost, 8,304 households, evaluated once:**

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| not on track | 0.887 | 0.913 | 0.900 | 5,653 |
| on track | 0.802 | 0.751 | 0.776 | 2,651 |
| **macro avg** | 0.845 | 0.832 | **0.838** | 8,304 |

| | pred not on track | pred on track |
| --- | --- | --- |
| **not on track** | 5,163 | 490 |
| **on track** | 660 | 1,991 |

> **In plain terms — reading the confusion matrix.** Rows are the truth, columns are what the model said. The diagonal is the successes: **5,163** at-risk households correctly flagged and **1,991** on-track households correctly cleared. Off the diagonal are the two mistakes: **490** at-risk households the model wrongly cleared (the expensive miss — outreach never reaches them) and **660** on-track households it wrongly flagged (the cheap miss — a wasted nudge). Every headline metric in this project is arithmetic on these four numbers.

**Test macro-F1 is 0.838 against a cross-validated 0.838 — identical to three decimals.** No overfitting, and no optimism in the CV estimate. That is the expected outcome given the untuned and tuned models differed by 0.0006, but it is worth confirming rather than assuming: a model whose hyperparameters were selected on the same folds used to report its score would normally be slightly optimistic, and here the effect is unmeasurable because the search found nothing to overfit to.

---

## What this changes for later phases

| Phase | Consequence |
| --- | --- |
| **5 — Explainability** | Use SHAP on the **XGBoost** model. Do **not** read logistic-regression coefficients on the shares — Phase 2 showed VIF = ∞, and Cell 6 confirmed the fit is only identified by the L2 penalty. Expect `Log_Income` and `Groceries_Share` to dominate; anything else on top contradicts Phase 3. |
| **6 — Clustering** | Unaffected by model choice; still needs the ILR-vs-CLR decision from Phase 2. |
| **7 — Business translation** | Operate at the **95% at-risk recall** threshold (0.368): precision 0.856. Report the model's margin over the single income rule (+0.095 macro-F1), not over the majority baseline. |
