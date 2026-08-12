# Phase 4 — Model Comparison

**Source:** [README § Phase 4 — Model Comparison](../README.md#phase-4--model-comparison)
**Notebook:** [`notebooks/04_model_comparison.ipynb`](../notebooks/04_model_comparison.ipynb)
**Builds on:** [Phase 3 — Baseline](phase3.md)

Phase 3 established two things this notebook takes as given: **accuracy is not a usable metric** for this project's ~99.4%/0.6% class imbalance, so every model here is judged on macro-F1 instead; and a **plain, imbalance-unaware classifier cannot beat the do-nothing baseline** (~0.499 macro-F1). Phase 4's job is to find out whether *any* model, once given proper imbalance handling and real hyperparameter tuning, can clear that floor — and, just as importantly, which of several model families does it best under the actual cost trade-off the business decision implies.

**A methodological point that shapes this notebook's whole structure:** with six candidate models, comparing all of them on `X_test` and reporting whichever scores best is itself a form of test-set leakage — it lets the test set influence *which model gets reported*, biasing that model's own final score upward. This notebook instead selects the winning model using **cross-validated, out-of-fold predictions only** (`cross_val_predict`), and touches `X_test` exactly once, at the very end, for a single evaluation of the model already chosen.

---

## Research questions & answers

| # | Question | Answer |
|---|---|---|
| 1 | Which 5–7 model families are appropriate given the data shape? | Logistic Regression, Decision Tree, Random Forest, XGBoost, SVM (linear kernel), and Neural Net (MLP) — six families spanning linear, tree/ensemble, and non-linear model types, each paired with an imbalance-handling mechanism appropriate to that family. |
| 2 | What validation strategy fits the class balance found in Phase 1? | Stratified 5-fold cross-validation (`StratifiedKFold`), used for hyperparameter tuning, for selecting the winning model, and for threshold tuning — `X_test` is touched exactly once, at the end, for the single already-selected model. |
| 3 | What hyperparameter search method is used, and what moves performance most? | `GridSearchCV` for the two single-hyperparameter models (Logistic Regression, SVM); `RandomizedSearchCV` (n_iter 8–15) for the four multi-hyperparameter models. Performance was driven far more by *which imbalance-handling mechanism* each model family used than by any individual hyperparameter — linear models' `class_weight` reliably reached `recall_0=1.0` under cross-validation; tree ensembles and the oversampled MLP did not. |
| 4 | Is precision or recall more important, given the business framing? | **Recall on the minority (`Goal_Met = 0`, at-risk) class** — a false negative means an at-risk customer is never flagged for outreach at all, which is costlier than the low-friction cost of an unnecessary nudge (a false positive). |

The rest of this document walks through *how* the notebook arrives at each answer, cell by cell, and why each design decision was made.

---

## Notebook walkthrough

### Cell 0 (markdown) — Title, scope, and the leakage-avoidance methodology

States the four Phase 4 questions and, up front, the notebook's core discipline: comparing several already-fitted models against each other using their test-set scores is a subtle leakage risk (the winner is partly chosen *because* it did well on those specific 4,000 rows), so this notebook selects the winner from cross-validated, out-of-fold predictions instead, reserving `X_test` for exactly one evaluation at the end.

### Cell 1 (code) — Imports and rebuilding Phase 2/3's feature matrix and split

Rebuilds `engineered` and reproduces Phase 3's exact 80/20 stratified split (`random_state=42`). `X_test`/`y_test` are set aside here and not referenced again until the notebook's final evaluation section.

### Cell 3 (markdown) — Answer to Question 1: model family selection

The data shape after Phase 2's encoding — n = 20,000, 21 numeric features (3 continuous, 11 expense ratios, 7 one-hot categoricals), and the severe ~99.4%/0.6% imbalance — rules out anything that doesn't scale to 20,000 rows or can't be given an explicit imbalance-handling mechanism. Six families are chosen:

| Model | Why included | Imbalance handling |
|---|---|---|
| Logistic Regression | Re-runs Phase 3's baseline model with imbalance handling added | `class_weight="balanced"` |
| Decision Tree | Non-linear splits/interactions a linear model can't capture; directly explainable structure | `class_weight="balanced"` |
| Random Forest | Ensemble of trees, usually reduces a single tree's variance | `class_weight="balanced"` |
| XGBoost | Typically the strongest tabular-data performer; named explicitly in the README's results table | `scale_pos_weight` (tuned) |
| SVM (linear kernel) | Named in the README; **linear kernel is deliberate** — n=20,000 makes an RBF kernel's O(n²)–O(n³) training cost impractical, and Phase 3 already found a linear boundary captures the key signal | `class_weight="balanced"` |
| Neural Net (MLP) | Non-linear model with a genuinely different imbalance mechanism, since `MLPClassifier` has no `class_weight` parameter | Random oversampling |

### Cell 4 (markdown) — Answer to Question 2: validation strategy

**`StratifiedKFold(n_splits=5)`**, used for three distinct jobs, all without touching `X_test`: (1) scoring every hyperparameter combination in each model's search below, (2) selecting which model family wins via `cross_val_predict`, and (3) tuning the winning model's decision threshold. With only ~90 minority-class rows in the training set, a single split (even a stratified one) means every reported number rests on how a handful of minority rows happened to land — cross-validation pools the result across 5 folds instead. Stratification specifically matters because a non-stratified fold could land with zero or very few `Goal_Met = 0` rows by chance.

### Cell 5 (code) — Shared evaluation helper and cross-validator

```python
def evaluate(name, y_true, y_pred): ...
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_models = {}
```

Defines the reused `evaluate()` helper and the shared cross-validator every search and comparison in this notebook uses. `best_models` collects each tuned pipeline for the cross-validated comparison section that follows all six searches — notably, **no test-set evaluation happens here or in any of the six model-fitting cells that follow**; each cell prints only its search's best hyperparameters and aggregate CV macro-F1 (`best_score_`).

### Cells 7–24 — Fitting and searching all six models

Each model gets a markdown cell (rationale + search method) and a code cell (pipeline + grid + search), printing only `best_params_` and `best_score_` — no per-model test-set touch. In fitting order: **Logistic Regression** (`GridSearchCV`, `C`, CV macro-F1 ~0.896), **Decision Tree** (`RandomizedSearchCV`, ~0.803), **Random Forest** (`RandomizedSearchCV`, ~0.843), **XGBoost** (`RandomizedSearchCV`, including `scale_pos_weight` as a searched value, ~0.820), **SVM/LinearSVC** (`GridSearchCV`, ~0.923 — the best aggregate CV score), **Neural Net/MLP** (`RandomizedSearchCV`, inside a pipeline with `RandomOverSampler`, ~0.861).

**The XGBoost cell's finding deserves its own note:** because `Goal_Met = 1` (met) is the *majority* class here, the textbook `scale_pos_weight` formula (`count(negative) / count(positive)`) comes out *less than 1*, deliberately down-weighting the abundant class. The search's own `cv_results_` show **`scale_pos_weight=1` (no correction) scoring highest** — pushing toward the textbook value traded away more precision than it gained in recall.

### Cells 25–29 — Selecting the winning model from cross-validated predictions

```python
majority_pred_cv = np.full(shape=y_train.shape, fill_value=y_train.mode()[0])
cv_rows = [evaluate("Majority baseline", y_train, majority_pred_cv)]

for name, model in best_models.items():
    oof_pred = cross_val_predict(model, X_train, y_train, cv=cv, n_jobs=-1)
    cv_rows.append(evaluate(name, y_train, oof_pred))

cv_comparison = pd.DataFrame(cv_rows).set_index("model")
```

**What it does:** `cross_val_predict` clones and refits each already-tuned pipeline fresh within each of the 5 folds, producing one out-of-fold prediction per training row. The majority baseline is computed directly (a constant-prediction rule needs no fitting, so it's identical either way).

**Why this table, not a test-set table:** every number in it comes from predictions on rows each model never saw during that fold's fit — the same guarantee `X_test` would offer, but obtained 5 times over and pooled, without spending the one held-out set this project has.

**Resulting per-class picture (the numbers Question 3 and 4 are built on):**

| Model | CV macro-F1 | CV recall₀ | CV precision₀ |
|---|---|---|---|
| SVM (Linear) | 0.922 | **1.000** | 0.732 |
| Logistic Regression | 0.894 | **1.000** | 0.652 |
| Neural Net (MLP) | 0.858 | 0.933 | 0.583 |
| Random Forest | 0.844 | 0.689 | 0.689 |
| XGBoost | 0.826 | 0.511 | 0.902 |
| Decision Tree | 0.803 | 0.733 | 0.520 |
| Majority baseline | 0.499 | 0.000 | 0.000 |

Two charts follow: accuracy vs. macro-F1 for all seven rows, and precision/recall on the minority class specifically — the one that makes the tree-ensemble/MLP shortfall visible at a glance, since only Logistic Regression and SVM's `recall_0` bars reach 1.0.

### Cell 29 (markdown) — Answer to Question 3

`GridSearchCV` for the single-hyperparameter models, `RandomizedSearchCV` otherwise. What moved performance most wasn't any individual hyperparameter — it was **which imbalance-handling mechanism each model family used**: `class_weight="balanced"` on linear models reliably reached `recall_0=1.0` across every fold (few enough degrees of freedom that reweighting the loss function alone moves the boundary); on tree models it only partially worked (recall₀ 0.73/0.69 — reweighting the split criterion doesn't manufacture more minority examples to split on); XGBoost's own search left `scale_pos_weight` unweighted, leaving recall₀ at 0.51; and oversampling on the MLP came close but fell short of perfect recall (0.93) — direct exposure to duplicated examples helps, but not as reliably as a boundary simple enough for weighting alone to move.

### Cells 30–31 — Question 4: business framing and model selection

Phase 0's framing makes the cost asymmetry concrete: a **false negative** (predicting on-track for someone at-risk) means that person is never flagged for outreach at all — the project's entire purpose is lost for them; a **false positive** just means an unnecessary, low-cost nudge. **Recall on `Goal_Met = 0` matters more than precision on it.**

Applying that to the cross-validated table: exactly **two** models reach `recall_0 = 1.0` — **Logistic Regression** and **SVM (Linear)**. Between them, SVM (Linear) has the higher `precision_0` (0.732 vs. 0.652) and higher macro-F1 (0.922 vs. 0.894), so **SVM (Linear), `C=10`, is selected as Phase 4's winning model**.

**A cautionary callout in this same cell directly demonstrates why the CV-based selection matters:** the Neural Net also reaches `recall_0 = 1.0` when evaluated on `X_test` directly — which would make it look like a third contender (or even the winner) under a naive test-set comparison. But its *cross-validated* `recall_0` is only 0.93 — its apparently perfect test-set recall is plausibly an artifact of which 22 minority rows happen to sit in this particular test split, not a reliable model property. This is the exact failure mode a reviewer flagged during development of this notebook, and the reason it's structured around cross-validated selection rather than test-set comparison.

### Cells 32–34 — Threshold tuning, using cross-validated predictions only

```python
oof_scores = cross_val_predict(svm_best, X_train, y_train, cv=cv, method="decision_function", n_jobs=-1)
for threshold in np.arange(-3.0, 3.01, 0.1):
    pred_t = (oof_scores >= threshold).astype(int)
    ...
```

**What it does:** Gets out-of-fold decision-function scores for the tuned SVM via `cross_val_predict` (still no test-set involvement), sweeps a threshold, and picks the one with the best `precision_0` among all thresholds keeping `recall_0 = 1.0` — entirely from `X_train`.

**Result:** the CV-selected threshold is **t ≈ -0.4** — raising `precision_0` from ~0.73 (default `t=0`) to **~0.93** in cross-validation, at no cost to `recall_0`.

### Cells 35–37 — Final, single evaluation on the held-out test set

```python
test_scores = svm_best.decision_function(X_test)
for threshold, label in [(0.0, "default"), (chosen_threshold, "CV-selected")]:
    pred_t = (test_scores >= threshold).astype(int)
    ...
```

**What it does:** This is the **only place in the entire notebook where `y_test` is compared against predictions** — a single, honest look at held-out performance for the one model and threshold already chosen using `X_train` alone.

**Result:** at the CV-selected threshold, **accuracy ≈ 0.999, macro-F1 ≈ 0.968, `recall_0 = 1.0`** (catches all 22 true at-risk individuals in the test set), and **`precision_0 ≈ 0.88`** — better than even the CV estimate (~0.93 was the *out-of-fold* estimate; the held-out test result of 0.88 is a plausible, unremarkable generalization gap, not a red flag).

### Cells 38–39 — Persisting results

Saves `results/model_comparison.csv` (the cross-validated comparison table — the one actually used to select the winner), `results/final_test_evaluation.csv` (the single test-set evaluation), and `results/model_comparison.png`.

### Cell 40 (markdown) — Summary table and handoff to Phase 5

States the winning model plainly: **SVM (Linear)** — `C=10`, `class_weight="balanced"`, decision threshold `t≈-0.4` — final single test evaluation: accuracy ≈ 0.999, macro-F1 ≈ 0.968, `recall_0 = 1.0`, `precision_0 ≈ 0.88`. Notes that because the winner is linear, Phase 5 can use SHAP's exact `LinearExplainer` rather than a sampled approximation.

---

## What Phase 4 sets up for later phases

| Finding | Where it gets used |
|---|---|
| SVM (Linear), `C=10`, `class_weight="balanced"`, threshold `t≈-0.4` — macro-F1 ≈ 0.968, `recall_0 = 1.0` | The exact model and threshold Phase 5 retrains and explains |
| Model selection must use cross-validated, not test-set, comparison | The reason Phase 5's explanation is of a linear model — a direct consequence of the CV-based selection surfacing a different winner than a naive test-set comparison would have |
| The winning model is linear | Enables Phase 5 to use SHAP's exact `LinearExplainer` (no sampling/approximation needed) instead of a permutation-based explainer |
| Recall on `Goal_Met = 0` is the metric the business actually needs optimized | Frames Phase 5's explainability work around *why the model catches at-risk customers*, not just around accuracy |
| Threshold `t≈-0.4` (found via CV) improves precision at no recall cost | A candidate operational recommendation for Phase 7 — Business Translation |

**Next:** Phase 5 — Explainability (`05_explainability.ipynb`), which uses SHAP to explain the winning model's predictions — both globally (which features matter most) and for individual predictions, in language a non-technical stakeholder can act on.
