# Phase 4 — Model Comparison

**Source:** [README § Phase 4 — Model Comparison](../README.md#phase-4--model-comparison)
**Notebook:** [`notebooks/04_model_comparison.ipynb`](../notebooks/04_model_comparison.ipynb)
**Builds on:** [Phase 3 — Baseline](phase3.md)

Phase 3 established two things this notebook takes as given: **accuracy is not a usable metric** for this project's ~99.4%/0.6% class imbalance, so every model here is judged on macro-F1 instead; and a **plain, imbalance-unaware classifier cannot beat the do-nothing baseline** (~0.499 macro-F1). Phase 4's job is to find out whether *any* model, once given proper imbalance handling and real hyperparameter tuning, can clear that floor — and, just as importantly, which of several model families does it best under the actual cost trade-off the business decision implies.

---

## Research questions & answers

| # | Question | Answer |
|---|---|---|
| 1 | Which 5–7 model families are appropriate given the data shape? | Logistic Regression, Decision Tree, Random Forest, XGBoost, SVM (linear kernel), and Neural Net (MLP) — six families spanning linear, tree/ensemble, and non-linear model types, each paired with an imbalance-handling mechanism appropriate to that family. |
| 2 | What validation strategy fits the class balance found in Phase 1? | Stratified 5-fold cross-validation (`StratifiedKFold`), scored on macro-F1 — required because only ~90 minority-class rows exist in the 16,000-row training set. |
| 3 | What hyperparameter search method is used, and what moves performance most? | `GridSearchCV` for the two single-hyperparameter models (Logistic Regression, SVM); `RandomizedSearchCV` (n_iter 8–15) for the four multi-hyperparameter models. Performance was driven far more by *which imbalance-handling mechanism* each model family used than by any individual hyperparameter — and XGBoost's search revealed the textbook `scale_pos_weight` formula actively hurting macro-F1 on this dataset. |
| 4 | Is precision or recall more important, given the business framing? | **Recall on the minority (`Goal_Met = 0`, at-risk) class** — a false negative means an at-risk customer is never flagged for outreach at all, which is costlier than the low-friction cost of an unnecessary nudge (a false positive). |

The rest of this document walks through *how* the notebook arrives at each answer, cell by cell, and why each design decision was made.

---

## Notebook walkthrough

### Cell 0 (markdown) — Title and scope

States the four Phase 4 questions and frames the notebook's relationship to Phase 3: the same engineered feature matrix and the same stratified 80/20 split (`random_state=42`) are rebuilt from the raw CSV, so every model here is judged on the exact same held-out rows Phase 3 used for its baseline and plain-logistic-regression numbers — every macro-F1 in this notebook is directly comparable to Phase 3's ~0.499 floor.

### Cell 1 (code) — Imports and rebuilding Phase 2/3's feature matrix and split

```python
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
...
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler
import xgboost as xgb
...
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, stratify=y, random_state=42
)
```

**What it does:** Adds the Phase 4 modeling stack (cross-validation and hyperparameter-search tools, `imbalanced-learn`'s pipeline and `RandomOverSampler`, `xgboost`) to the same feature-engineering imports Phases 1–3 used, then rebuilds `engineered` and reproduces Phase 3's exact 80/20 stratified split.

**Why the split is reproducible, not just "similar":** `train_test_split` with a fixed `random_state` and identical inputs is deterministic — this is *the same row-level split* Phase 3 used, not a fresh one with matching proportions. That's what makes every macro-F1 number in this notebook directly comparable to Phase 3's floor, without needing to depend on Phase 3's live kernel state.

### Cell 3 (markdown) — Answer to Question 1: model family selection

The data shape after Phase 2's encoding — n = 20,000, 21 numeric features (3 continuous, 11 expense ratios, 7 one-hot categoricals), and the severe ~99.4%/0.6% imbalance — rules out anything that doesn't scale to 20,000 rows or can't be given an explicit imbalance-handling mechanism. Six families are chosen:

| Model | Why included | Imbalance handling |
|---|---|---|
| Logistic Regression | Re-runs Phase 3's baseline model with imbalance handling added, isolating whether that resolves the earlier "right coefficients, wrong threshold" finding | `class_weight="balanced"` |
| Decision Tree | Non-linear splits/interactions a linear model can't capture; directly explainable structure | `class_weight="balanced"` |
| Random Forest | Ensemble of trees, usually reduces a single tree's variance | `class_weight="balanced"` |
| XGBoost | Typically the strongest tabular-data performer; named explicitly in the README's results table | `scale_pos_weight` (tuned) |
| SVM (linear kernel) | Named in the README; **linear kernel is deliberate** — n=20,000 makes an RBF kernel's O(n²)–O(n³) training cost impractical, and Phase 3 already found a linear boundary captures the key signal | `class_weight="balanced"` |
| Neural Net (MLP) | Non-linear model with a genuinely different imbalance mechanism, since `MLPClassifier` has no `class_weight` parameter | Random oversampling |

Random Forest and XGBoost being both tree ensembles is intentional: comparing bagging (`class_weight`) against boosting (`scale_pos_weight`) tests whether the imbalance *mechanism* matters as much as the ensemble *strategy* — a question Question 3's answer below returns to directly.

### Cell 4 (markdown) — Answer to Question 2: validation strategy

**`StratifiedKFold(n_splits=5)`**, scored with `scoring="f1_macro"` in every search. Two dataset-specific facts make this necessary rather than a stylistic choice:

- With only ~90 minority-class rows in the training set, a single evaluation split (even Phase 3's stratified one) means every reported number rests on how ~18 minority test rows happened to land. Cross-validation averages across 5 folds instead, so no single lucky/unlucky fold drives which hyperparameters "win."
- **Stratification specifically** (not plain `KFold`) matters because a non-stratified fold could easily land with zero or very few `Goal_Met = 0` rows by chance, making that fold's precision/recall for the minority class undefined or meaningless.

### Cell 5 (code) — Shared evaluation helper, cross-validator, and recomputed majority baseline

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
dummy = DummyClassifier(strategy="most_frequent", random_state=42)
...
results = [evaluate("Majority baseline", y_test, pred_dummy)]
```

**What it does:** Reuses Phase 3's `evaluate()` helper, defines the shared `cv` object every search below uses, and recomputes the majority-class baseline on this notebook's own (identical) split so it sits in the same comparison table as every model that follows — no need to scroll back to Phase 3 to see the floor.

**Why recompute instead of hardcode:** cheap, self-verifying (the numbers should — and do — match Phase 3's ~99.4% accuracy / ~0.499 macro-F1 exactly), and keeps the notebook runnable standalone.

### Cells 7–24 — Fitting and searching all six models

Each model gets a markdown cell (rationale + search method) and a code cell (pipeline + grid + search + evaluation), in this order: **Logistic Regression** (`GridSearchCV`, `C`), **Decision Tree** (`RandomizedSearchCV`, depth/leaf-size/criterion), **Random Forest** (`RandomizedSearchCV`, ensemble size/depth/leaf-size/feature-subsampling), **XGBoost** (`RandomizedSearchCV`, including `scale_pos_weight` as a searched value), **SVM/LinearSVC** (`GridSearchCV`, `C`), **Neural Net/MLP** (`RandomizedSearchCV`, hidden-layer architecture/`alpha`, inside a pipeline with `RandomOverSampler`).

**What each result showed, in order fit:**

| Model | CV macro-F1 | Test recall₀ | Note |
|---|---|---|---|
| Logistic Regression | ~0.896 | 1.000 | `class_weight="balanced"` alone resolves most of Phase 3's imbalance gap; search prefers the *least* regularization (`C=10`) |
| Decision Tree | ~0.803 | 0.591 | `class_weight` reweights the split criterion but can't manufacture more minority rows to split on |
| Random Forest | ~0.843 | 0.500 | Averaging reweighted trees smooths some variance but doesn't fix the underlying scarcity problem |
| XGBoost | ~0.820 | 0.500 | Search picked `scale_pos_weight=1` (no reweighting) over the textbook formula — see the counter-intuitive finding below |
| SVM (Linear) | ~0.923 | 1.000 | Best of the first five; maximum-margin boundary generalizes slightly better than logistic regression's log-loss fit here |
| Neural Net (MLP) | ~0.861 | 1.000 | Best **test-set** macro-F1 overall (~0.931) — oversampling gives it direct exposure to (duplicated) minority rows rather than relying on reweighting alone |

**The XGBoost cell's specific finding (Cell 17–18) deserves its own note:** because `Goal_Met = 1` (met) is the *majority* class here — not the rare "positive" event `scale_pos_weight` is usually written for — the textbook formula `count(negative) / count(positive)` comes out *less than 1*, deliberately down-weighting the abundant class. The search compared that value, half of it, and `1` (no correction), and its own `cv_results_` show **`scale_pos_weight=1` scoring highest** — the reweighted settings gained recall but lost more precision than that was worth to macro-F1. This is exactly what a hyperparameter search is for: catching that the "correct" formula for a parameter isn't automatically the best choice for a specific metric and class distribution.

### Cells 25–29 — Comparison table, charts, and Answer to Question 3

```python
comparison_sorted = comparison.sort_values("f1_macro", ascending=False).round(4)
```

Two charts follow: accuracy vs. macro-F1 for all seven rows (majority baseline included, with a dashed line at its ~0.499 floor), and a second chart isolating **precision/recall on the minority class specifically** — this is the one that makes the tree-ensemble weakness visible at a glance, since Decision Tree and Random Forest's `recall_0` bars are visibly shorter than the three models (Logistic Regression, SVM, MLP) that reach `recall_0 = 1.0`.

**Answer to Question 3** (Cell 29, markdown): `GridSearchCV` for the two single-hyperparameter models, `RandomizedSearchCV` otherwise. What moved performance most wasn't any individual hyperparameter — it was **which imbalance-handling mechanism each model family used**: `class_weight="balanced"` worked well on linear models (few enough degrees of freedom that reweighting the loss function alone moves the boundary), only partially on tree models (reweighting the split criterion doesn't manufacture more minority examples to split on), actively hurt XGBoost when pushed to the textbook value, and oversampling worked best of all for the MLP (direct exposure beats reweighting when there are only ~90 real minority rows to work with).

### Cells 30–33 — Question 4: business framing and threshold-tuning demo

**Answer to Question 4** (Cell 30, markdown): Phase 0 framed the decision as flagging at-risk individuals for marketing outreach. A **false negative** (predicting `Goal_Met = 1` for someone actually at risk) means that person is never flagged at all — the project's entire purpose is lost for them. A **false positive** (predicting `Goal_Met = 0` for someone on-track) just means an unnecessary, low-cost nudge. **Recall on `Goal_Met = 0` matters more than precision on it.** Three models (Logistic Regression, SVM, MLP) already reach `recall_0 = 1.0` at the default threshold; among them, the **Neural Net (MLP)** has the highest `precision_0` and macro-F1, so it's selected as **Phase 4's winning model**.

```python
proba_test = best_mlp.predict_proba(X_test)[:, 1]
for threshold in np.arange(0.05, 0.96, 0.05):
    pred_t = (proba_test >= threshold).astype(int)
    ...
```

**What it does:** Sweeps the decision threshold used to call `Goal_Met = 0` from 0.05 to 0.95 and recomputes `precision_0`/`recall_0` at each one — directly testing whether the section's own conclusion (maximize recall, then precision) has room to improve beyond the 0.5 default.

**Result (Cell 33, markdown):** `recall_0` reaches and holds at 1.0 once the threshold is ≥0.35. Within that safe range, **lowering the threshold from 0.5 to 0.35 raises `precision_0` from ~0.759 to ~0.786** — same perfect recall, fewer false alarms — a small, free improvement the default threshold leaves on the table.

### Cells 34–35 — Persisting results

```python
os.makedirs("../results", exist_ok=True)
comparison_sorted.to_csv("../results/model_comparison.csv")
...
fig_summary.savefig("../results/model_comparison.png", dpi=150)
```

**What it does:** Saves the final comparison table and its summary chart to `results/`, matching the repository structure the README documents as planned — Phase 4's numbers are now available as plain files, not only inside notebook output cells.

### Cell 36 (markdown) — Summary table and handoff to Phase 5

States the winning model plainly: **Neural Net (MLP)** — `hidden_layer_sizes=(64,)`, `alpha=0.001`, `RandomOverSampler` inside the pipeline — test accuracy ~99.8%, macro-F1 ~0.931, `recall_0 = 1.0`, `precision_0 ≈ 0.759` (improvable to ~0.786 at threshold 0.35 with no recall cost). Points to Phase 5 — Explainability, which needs to explain *why* this specific model makes its predictions.

---

## What Phase 4 sets up for later phases

| Finding | Where it gets used |
|---|---|
| Neural Net (MLP), `hidden_layer_sizes=(64,)`, `alpha=0.001`, oversampled — macro-F1 ~0.931, `recall_0 = 1.0` | The exact model and hyperparameters Phase 5 retrains and explains |
| Imbalance-handling *mechanism* matters more than any individual hyperparameter | Explains why Phase 5's model has no simple coefficient/split-based explanation and needs a model-agnostic tool (SHAP) |
| `Loan_Repayment_Ratio`-driven linear models (LogReg, SVM) already reach `recall_0 = 1.0` | A sanity check for Phase 5: the winning model's top features should plausibly include the same ones |
| Recall on `Goal_Met = 0` is the metric the business actually needs optimized | Frames Phase 5's explainability work around *why the model catches at-risk customers*, not just around accuracy |
| Threshold 0.35 (vs. default 0.5) gets slightly better precision at no recall cost | A candidate operational recommendation for Phase 7 — Business Translation |

**Next:** Phase 5 — Explainability (`05_explainability.ipynb`), which uses SHAP to explain the winning model's predictions — both globally (which features matter most) and for individual predictions, in language a non-technical stakeholder can act on.
