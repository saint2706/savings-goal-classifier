# Phase 4 (IHDS-II) — Model Comparison

**Source:** [README § Phase 4 — Model Comparison](../README.md#phase-4--model-comparison)
**Notebook:** [`notebooks/04_model_comparison.ipynb`](../notebooks/04_model_comparison.ipynb)
**Builds on:** [Phase 3 (IHDS-II)](phase3.md)
**Artifacts:** `results/model_comparison.csv`, `results/model_comparison.png`
**Replaces:** [`phase4.md`](phase4.md) — which is void on real data, for the reason below

The original Phase 4 was built entirely around a 178:1 class imbalance with only 112 minority cases. Every decision it made followed from that: `class_weight="balanced"` on everything, model selection on `recall_0`, and the linear SVM winning because it was one of only two models achieving perfect minority recall. At **2.13:1 with 13,256 minority cases**, none of that reasoning applies and the comparison has to be re-made from scratch.

---

## Research questions & answers

| # | Question | Answer |
| --- | --- | --- |
| 1 | Which model families are appropriate given the data? | Seven: majority baseline, logistic regression, linear SVM, decision tree, random forest, histogram gradient boosting, and XGBoost. At n = 41,518 with 28 mixed features, kernel SVMs are impractical (O(n²)) and were excluded on cost, not principle. **XGBoost wins** at CV macro-F1 **0.8371** / ROC-AUC **0.9306**. |
| 2 | What validation strategy fits the class balance found in Phase 1? | 5-fold **stratified** k-fold on an 80% training split, with a 20% test set (8,304 households) held out and touched exactly once. Stratification still matters at 2.13:1, but as insurance rather than necessity — fold-to-fold macro-F1 sd is 0.0018–0.0052, so the estimates are stable. |
| 3 | What hyperparameter search method is used, and what parameters move performance most? | 15-iteration randomised search over 5 XGBoost parameters, plus a 5-point sweep on logistic regression's `C`. **The search is worth +0.0006 macro-F1** (0.8371 → 0.8377) — statistically nothing. `learning_rate` matters most (score sd 0.0040 across levels); `max_depth` matters least (0.0003). |
| 4 | Is precision or recall more important given the business framing? | **Recall on the at-risk class**, because the intervention is a low-cost nudge and a missed at-risk household is a customer who silently fails their goal. The good news is the trade-off is cheap here: at the default threshold the at-risk class already gets precision 0.887 / recall 0.913, and pushing recall to **95%** costs only ~3 points of precision (0.856). |

---

## Notebook walkthrough

### Cell 1 (code) — Load, and split off the test set immediately

The 20% test split is made in the **first cell**, before any model is defined, and is not referenced again until the final cell.

**Why up front rather than at the end:** it makes accidental leakage structurally difficult. Every intermediate decision in this notebook — model choice, hyperparameter search, threshold tuning — reads `X_train` only. If the split happened later, any of those steps could have quietly seen the test data. The original Phase 4 made the same choice for the same reason, and it is the one thing from it that transfers unchanged.

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

**Why macro-F1 is the ranking metric and accuracy is not:** the majority baseline scores 0.6807 accuracy while never predicting the positive class. Macro-F1 weights both classes equally and so cannot be gamed by ignoring one.

**Why the linear models look worse than they are.** Logistic regression and the linear SVM have the *highest recall* in the table (0.847, 0.848) and the lowest precision (0.698, 0.696). That is `class_weight="balanced"` doing exactly what it is asked: reweighting the loss to favour the minority class, which shifts the decision boundary toward predicting "on track" more often. The boosted models were fitted **without** class weighting, because at 2.13:1 they do not need it, and they land on a more even precision/recall split. The comparison is therefore between differently-calibrated models, and the fair reading is the ROC-AUC column — which is threshold-free. There, logistic regression (0.9213) sits much closer to XGBoost (0.9306) than macro-F1 suggests.

**The honest margin.** Against the **single income threshold** from Phase 3 (macro-F1 0.7425), XGBoost is worth **+0.095**. Against the majority baseline it is worth +0.432, but that comparison flatters the model and Phase 3 exists to prevent it being quoted.

**Why the two boosting implementations are treated as tied:** 0.8371 vs 0.8360 is a gap of 0.0011 against a fold-to-fold sd of 0.0018–0.0021 — well inside noise, and their ROC-AUCs are identical to four decimals. XGBoost is selected because it scored marginally higher, not because it is meaningfully better; `HistGradientBoosting` would be a defensible swap and has the advantage of being a scikit-learn built-in with no extra dependency.

**Excluded families and why:** RBF-kernel SVM scales roughly quadratically in n and is impractical at 41,518 rows. A neural net (MLP) was excluded because Phase 3 showed steep diminishing returns — the marginal gain over the boosted trees would not justify the tuning surface. Both exclusions are cost decisions, and neither is likely to change the ranking.

### Cell 5 (code) — Hyperparameter search (Q3)

15-iteration randomised search over `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `min_child_weight`, scored on macro-F1.

**Best parameters:** `learning_rate=0.15`, `n_estimators=200`, `max_depth=4`, `min_child_weight=5`, `subsample=0.9`
**Best score: 0.8377** — against **0.8371** for the untuned defaults.

**The search bought +0.0006 macro-F1, which is a third of one fold's standard deviation.** This is the phase's most useful negative result, and it confirms the prediction Phase 3 made from the feature-progression curve: the problem's difficulty lives in the data, not in the model configuration. A larger search would have been compute spent to move a number that does not move.

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

### Cell 6 (code) — Logistic regression's regularisation sweep

`C` from 0.01 to 100: macro-F1 moves from 0.8159 to **0.8190** — a range of 0.003 across four orders of magnitude, monotonically increasing.

**Why this is worth reporting rather than skipping:** Phase 2 found the 11 shares are **exactly singular** (VIF = ∞, they sum to 1), which means the model is only identifiable because of the L2 penalty. If the fit were badly conditioned, performance would be sharply sensitive to `C`. It is not — and that the best `C` is the *weakest* regularisation tested (100.0) says the penalty is doing structural work (making the solution unique) rather than being needed to control variance. The singularity is real but benign for prediction, exactly as Phase 2 predicted. **It remains fatal for coefficient interpretation**, which is Phase 5's problem.

### Cell 8 (code) — Precision vs recall for the business decision (Q4)

**The framing correction this cell makes:** the positive class is `Goal_Met = 1` ("on track"), but the class the business acts on is `Goal_Met = 0` — the at-risk households a savings nudge would target. Every precision/recall figure in the comparison table is for the *wrong class* for decision-making purposes, so the at-risk view is computed explicitly.

Out-of-fold, XGBoost at the default threshold:

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| **not on track** | 0.885 | 0.916 | 0.900 | 22,609 |
| on track | 0.807 | 0.746 | 0.775 | 10,605 |

**The at-risk class is the easier one**, because it is the majority at 68%. This inverts the synthetic project entirely, where the at-risk class was 0.56% and catching it was the whole difficulty.

Tuning the threshold on the at-risk score:

| Target | Precision | Recall | Threshold |
| --- | --- | --- | --- |
| Max F1 | 0.873 | 0.933 | 0.440 |
| Recall 80% | 0.941 | 0.800 | 0.757 |
| Recall 90% | 0.896 | 0.900 | 0.554 |
| **Recall 95%** | **0.856** | 0.950 | 0.368 |

**The recommendation: optimise for recall on the at-risk class.** The asymmetry is in the costs. A false positive is one unnecessary low-cost nudge to a household that was already saving — mildly wasteful. A false negative is a household heading for a shortfall that the outreach never reaches, which is the failure the project exists to prevent (Phase 0).

**And the trade is unusually cheap here.** Moving from 80% to 95% recall costs 8.5 points of precision (0.941 → 0.856). Even at 95% recall, roughly six in seven flagged households are genuinely at risk. Contrast the synthetic project, where achieving perfect minority recall cost so much precision that roughly one in eight flags was real.

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

**Test macro-F1 is 0.838 against a cross-validated 0.838 — identical to three decimals.** No overfitting, and no optimism in the CV estimate. That is the expected outcome given the untuned and tuned models differed by 0.0006, but it is worth confirming rather than assuming: a model whose hyperparameters were selected on the same folds used to report its score would normally be slightly optimistic, and here the effect is unmeasurable because the search found nothing to overfit to.

---

## What this changes for later phases

| Phase | Consequence |
| --- | --- |
| **5 — Explainability** | Use SHAP on the **XGBoost** model. Do **not** read logistic-regression coefficients on the shares — Phase 2 showed VIF = ∞, and Cell 6 confirmed the fit is only identified by the L2 penalty. Expect `Log_Income` and `Groceries_Share` to dominate; anything else on top contradicts Phase 3. |
| **6 — Clustering** | Unaffected by model choice; still needs the ILR-vs-CLR decision from Phase 2. |
| **7 — Business translation** | Operate at the **95% at-risk recall** threshold (0.368): precision 0.856. Report the model's margin over the single income rule (+0.095 macro-F1), not over the majority baseline. |
| **8 — Reporting** | The winning model, its metric, and the imbalance narrative all differ from `phase4.md`. That document describes the synthetic track and should not be cited for IHDS results. |
