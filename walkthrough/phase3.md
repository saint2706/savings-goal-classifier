# Phase 3 (IHDS-II) — Baseline

**Source:** [README § Phase 3 — Baseline](../README.md#phase-3--baseline)
**Notebook:** [`notebooks/03_baseline.ipynb`](../notebooks/03_baseline.ipynb)
**Builds on:** [Phase 2 (IHDS-II)](phase2.md)
**Artifacts:** `results/baseline.csv`, `results/baseline.png`

A baseline exists to answer one question before any modelling effort is justified: **how much of this problem is solved by doing almost nothing?** The answer here sets a far higher bar for Phase 4 than the majority class alone would suggest — a single threshold on income recovers most of the achievable performance.

---

## Research questions & answers

| # | Question | Answer |
| --- | --- | --- |
| 1 | What accuracy/F1 does a majority-class or simple single-rule baseline achieve? | Majority class: **0.6807 accuracy but 0.4050 macro-F1** — it never predicts the positive class at all, so precision and recall on "on track" are both exactly 0. The interesting result is the single-rule baseline: **one threshold on income** (predict "on track" if annual income > **₹121,685**) reaches **0.7753 accuracy and 0.7425 macro-F1**. |
| 2 | What does plain logistic regression achieve using only income and 1–2 expense shares? | Income alone: ROC-AUC **0.8348**. Adding `Groceries_Share` — a single feature — lifts it to **0.8756** (+0.041). Adding `Spends_On_Insurance` adds almost nothing (+0.0015). The full 28-feature set reaches **0.9212** (+0.044 over the three-feature model). |

---

## Notebook walkthrough

### Cell 1 (code) — Load and define the feature split

Loads `dataset/features.csv` from Phase 2 rather than rebuilding features from the raw household file. **This breaks the repo's usual self-contained-notebook convention deliberately:** Phase 2's feature set was the product of four tested decisions (participation indicators, winsorisation, the rejected CLR transform), and re-deriving that chain in every later notebook would invite the phases to drift apart. The 6 `*_CLR` columns exported by Phase 2 are explicitly dropped here — Phase 2 rejected them for classification and they are reserved for Phase 6.

### Cell 3 (code) — Chance baselines (Q1, part 1)

| Baseline | Accuracy | Macro-F1 | ROC-AUC | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| Majority class | 0.6807 | 0.4050 | 0.5000 | **0.0000** | **0.0000** |
| Stratified random | 0.5649 | 0.4948 | 0.4949 | 0.3122 | 0.3013 |
| Prior probability | 0.6807 | 0.4050 | 0.5000 | 0.0000 | 0.0000 |

**Why three dummies rather than one:** they fail differently, and quoting only one flatters a real model on whichever metric that dummy happens to be weak on. The majority classifier scores a respectable-looking 0.68 accuracy while being completely useless — it never identifies a single on-track household. The stratified dummy has *worse* accuracy (0.56) but much better macro-F1 (0.49), because it at least predicts both classes. A model has to beat the harder of the two on each metric to have learned anything.

**The reason accuracy is retired as the headline metric from here on:** 0.68 accuracy is obtainable with a constant. The gap between 0.4050 and 0.4948 macro-F1 between two *equally worthless* predictors shows how much room there is to look good on one number while doing nothing.

### Cell 4 (code) — The single-rule baseline (Q1, part 2)

A depth-1 decision tree on `Log_Income` alone, which is the simplest non-trivial rule the data supports.

**The learned rule:** predict "on track" if annual income exceeds **₹121,685**. 30.9% of households sit above that line — almost exactly the 31.9% base rate of the positive class, which is what a well-placed single split should produce.

| Baseline | Accuracy | Macro-F1 | ROC-AUC |
| --- | --- | --- | --- |
| Single rule: income only (depth 1) | 0.7753 | **0.7425** | 0.7438 |
| Income only (depth 3) | 0.7823 | 0.7281 | 0.8279 |

**This is the number Phase 4 has to justify itself against, not the majority baseline.** One threshold on one variable recovers macro-F1 0.7425. The full 28-feature engineered model from Phase 2 reaches 0.8202. Everything the project does beyond "ask how much they earn" is worth **+0.078 macro-F1** — real, but far from the whole story, and it should be reported that way rather than as "the model achieves 0.82."

**Why the depth-3 tree has better ROC-AUC but worse macro-F1:** deeper splits give a finer-grained ranking (helping AUC, which is threshold-free) while shifting the default 0.5 decision boundary toward the majority class (hurting recall, and so macro-F1, at that specific threshold). This is an early, concrete instance of a distinction Phase 4 has to handle explicitly: **ranking quality and decision quality are different things**, and a model can improve on one while losing on the other.

### Cell 6 (code) — Feature-by-feature progression (Q2)

| Feature set | Accuracy | Macro-F1 | ROC-AUC | ΔAUC |
| --- | --- | --- | --- | --- |
| Income only | 0.7528 | 0.7329 | 0.8348 | — |
| \+ `Groceries_Share` | 0.7880 | 0.7705 | 0.8756 | **+0.0408** |
| \+ `Spends_On_Insurance` | 0.7902 | 0.7728 | 0.8771 | +0.0015 |
| All 28 engineered features | 0.8357 | 0.8202 | 0.9212 | +0.0441 |

**Why `Groceries_Share` was chosen as the "1–2 expense shares" to test:** Phase 1 found it was the share most correlated with log income (r = −0.266) and identified that relationship as Engel's law — food's budget share falls as households get richer. A baseline should test the feature with the strongest prior justification, not an arbitrary one.

**The result confirms it carries independent signal.** Note that `Groceries_Share` adds +0.041 AUC *on top of income*, despite being correlated with income. If it were merely a proxy for income it would add nothing here. It is measuring something income does not: two households earning the same amount but spending 35% versus 70% of their budget on food are in materially different positions, and the second is far less likely to be saving.

**Why `Spends_On_Insurance` adds almost nothing (+0.0015) despite a +9.7pp univariate lift in Phase 2:** its Phase 2 lift was largely a proxy for income — richer households buy insurance. Once income and grocery share are already in the model, the independent contribution is nearly exhausted. This is a useful caution for Phase 7: **univariate lifts are not additive, and the Phase 2 indicator table should not be read as a list of independent levers.**

**The remaining +0.044 from the other 25 features** is roughly the same size as the single-feature gain from `Groceries_Share`. Diminishing returns are steep here, which is worth knowing before Phase 4 spends compute on hyperparameter search.

### Cell 7 (code) — Persist and plot

Writes `results/baseline.csv` and a horizontal bar chart. The chart is deliberately drawn with all three metrics side by side and a reference line at 0.5, so the majority baseline's 0.68-accuracy / 0.40-macro-F1 split is visible as a single visual fact — that is the phase's main message to a non-technical reader.

---

## What this changes for later phases

| Phase | Consequence |
| --- | --- |
| **4 — Model comparison** | The bar is **macro-F1 0.7425 from one income threshold**, not 0.4050 from the majority class. Report every model's margin over the single-rule baseline, not just over chance. Diminishing returns are steep, so a large hyperparameter search is unlikely to pay. |
| **5 — Explainability** | `Log_Income` and `Groceries_Share` alone reach AUC 0.8756 of the full model's 0.9212 — any SHAP story that does not put these two at the top is contradicting the baseline. |
| **7 — Business translation** | Two cautions: the "model" is ~90% an income threshold in AUC terms, and univariate lifts from Phase 2 are not independent levers (`Spends_On_Insurance` adds +0.0015 once income is known). |
