# Phase 3 — Baseline

**Source:** [README § Phase 3 — Baseline](../README.md#phase-3--baseline)
**Notebook:** [`notebooks/03_baseline.ipynb`](../notebooks/03_baseline.ipynb)
**Builds on:** [Phase 2](phase2.md)
**Artifacts:** `results/baseline.csv`, `results/baseline.png`

A baseline exists to answer one question before any modelling effort is justified: **how much of this problem is solved by doing almost nothing?** The answer here sets a far higher bar for Phase 4 than the majority class alone would suggest — a single threshold on income recovers most of the achievable performance.

> **In plain terms — what a baseline is and why it comes first.** A score means nothing on its own. "Macro-F1 0.84" is only impressive relative to something, and the something has to be established *before* the real model is built — otherwise the comparison gets chosen after the fact to flatter the result.
>
> A **baseline** is a deliberately stupid predictor used as that yardstick. This phase builds two kinds:
> - **Chance baselines**, which use no information whatsoever — always answer with the most common class, or guess randomly in the right proportions. These set the floor below which a model is worthless.
> - A **simple-rule baseline**, which uses one obvious piece of information in the crudest possible way — here, one cut-off on income. This is the far more demanding comparison, because it represents what a competent person could do in an afternoon with a spreadsheet and no machine learning at all.
>
> The gap between the sophisticated model and *that* rule is the project's honest contribution.

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

> **In plain terms — the three dummies.** A **dummy classifier** ignores the features entirely and answers by a fixed recipe. **Majority class**: always say "not on track", since that is the more common answer. **Stratified random**: roll a weighted die, saying "on track" about 32% of the time to match the real proportion, but with no regard to which household it is. **Prior probability**: always predict using the overall base rate, which here collapses to the same behaviour as the majority rule.
>
> Notice the columns of zeros. The majority classifier's precision and recall on "on track" are exactly **0.0000** — it never once identifies an on-track household, because it never predicts that class at all. Yet its accuracy is 0.6807. That single row is the argument for everything that follows.

**Why three dummies rather than one:** they fail differently, and quoting only one flatters a real model on whichever metric that dummy happens to be weak on. The majority classifier scores a respectable-looking 0.68 accuracy while being completely useless — it never identifies a single on-track household. The stratified dummy has *worse* accuracy (0.56) but much better macro-F1 (0.49), because it at least predicts both classes. A model has to beat the harder of the two on each metric to have learned anything.

**The reason accuracy is retired as the headline metric from here on:** 0.68 accuracy is obtainable with a constant. The gap between 0.4050 and 0.4948 macro-F1 between two *equally worthless* predictors shows how much room there is to look good on one number while doing nothing.

### Cell 4 (code) — The single-rule baseline (Q1, part 2)

A depth-1 decision tree on `Log_Income` alone, which is the simplest non-trivial rule the data supports.

> **In plain terms — a depth-1 decision tree.** A **decision tree** is a flowchart of yes/no questions ending in a prediction. Its **depth** is how many questions deep it is allowed to go. Depth 1 permits exactly *one* question — so the entire "model" is a single line drawn on the income axis, with everyone above it called on track and everyone below called at risk. The only thing that was learned is where to draw the line, and the algorithm places it wherever it separates the two groups best. A **depth-3** tree, shown for comparison, gets to ask three nested questions and so can carve income into eight bands instead of two.

**The learned rule:** predict "on track" if annual income exceeds **₹121,685**. 30.9% of households sit above that line — almost exactly the 31.9% base rate of the positive class, which is what a well-placed single split should produce.

> **In plain terms — base rate.** The **base rate** is simply how common the outcome is overall: 31.9% of households meet the goal. It is the number any prediction has to be judged against — being right 31.9% of the time about who saves is achievable by guessing. Here it doubles as a sanity check: a well-placed single cut should flag roughly as many households as there really are, and 30.9% against 31.9% says the line landed sensibly rather than in some lopsided corner.

| Baseline | Accuracy | Macro-F1 | ROC-AUC |
| --- | --- | --- | --- |
| Single rule: income only (depth 1) | 0.7753 | **0.7425** | 0.7438 |
| Income only (depth 3) | 0.7823 | 0.7281 | 0.8279 |

**This is the number Phase 4 has to justify itself against, not the majority baseline.** One threshold on one variable recovers macro-F1 0.7425. The full 28-feature engineered model from Phase 2 reaches 0.8202. Everything the project does beyond "ask how much they earn" is worth **+0.078 macro-F1** — real, but far from the whole story, and it should be reported that way rather than as "the model achieves 0.82."

**Why the depth-3 tree has better ROC-AUC but worse macro-F1:** deeper splits give a finer-grained ranking (helping AUC, which is threshold-free) while shifting the default 0.5 decision boundary toward the majority class (hurting recall, and so macro-F1, at that specific threshold). This is an early, concrete instance of a distinction Phase 4 has to handle explicitly: **ranking quality and decision quality are different things**, and a model can improve on one while losing on the other.

> **In plain terms — ranking vs deciding.** A classifier really does two jobs, and they are scored by different numbers.
> - **Ranking**: put households in order, most likely to save first. **ROC-AUC** measures only this. It is called **threshold-free** because it never asks where you draw the line — shuffle the cut-off anywhere you like and the ROC-AUC is unchanged.
> - **Deciding**: convert that order into an actual yes/no by drawing a line somewhere (by default, probability 0.5). **Macro-F1 and accuracy** measure this, and they depend entirely on where the line sits.
>
> So a model can order households better while making worse calls, simply because its probabilities cluster on one side of the default cut-off. That is exactly the depth-3 tree's situation, and it is fixable — you move the line, which [Phase 4](phase4.md) does deliberately. The lesson is that a disagreement between these two metrics is usually a statement about the **threshold**, not about which model understands the data better.

### Cell 6 (code) — Feature-by-feature progression (Q2)

| Feature set | Accuracy | Macro-F1 | ROC-AUC | ΔAUC |
| --- | --- | --- | --- | --- |
| Income only | 0.7528 | 0.7329 | 0.8348 | — |
| \+ `Groceries_Share` | 0.7880 | 0.7705 | 0.8756 | **+0.0408** |
| \+ `Spends_On_Insurance` | 0.7902 | 0.7728 | 0.8771 | +0.0015 |
| All 28 engineered features | 0.8357 | 0.8202 | 0.9212 | +0.0441 |

> **In plain terms — reading a progression table.** Each row adds one thing to the row above, so the last column (**ΔAUC**, "delta AUC", meaning *change in* ROC-AUC) shows what each addition bought. The Greek delta always means "the change in" in this project.
>
> This is the incremental test described in [Phase 0](phase0.md), run for real. Grocery share buys +0.041, which is substantial. The insurance indicator buys +0.0015, which is nothing — despite looking strong on its own in [Phase 2](phase2.md). And *twenty-five further features* together buy +0.044, roughly what the first single share bought. That final row is the shape of **diminishing returns**: each addition helps less than the last, and the curve flattens long before the feature list runs out.

**Why `Groceries_Share` was chosen as the "1–2 expense shares" to test:** Phase 1 found it was the share most correlated with log income (r = −0.266) and identified that relationship as Engel's law — food's budget share falls as households get richer. A baseline should test the feature with the strongest prior justification, not an arbitrary one.

**The result confirms it carries independent signal.** Note that `Groceries_Share` adds +0.041 AUC *on top of income*, despite being correlated with income. If it were merely a proxy for income it would add nothing here. It is measuring something income does not: two households earning the same amount but spending 35% versus 70% of their budget on food are in materially different positions, and the second is far less likely to be saving.

**Why `Spends_On_Insurance` adds almost nothing (+0.0015) despite a +9.7pp univariate lift in Phase 2:** its Phase 2 lift was largely a proxy for income — richer households buy insurance. Once income and grocery share are already in the model, the independent contribution is nearly exhausted. This is a useful caution for Phase 7: **univariate lifts are not additive, and the Phase 2 indicator table should not be read as a list of independent levers.**

> **In plain terms — "univariate", and why lifts don't add up.** **Univariate** means looked at one at a time, in isolation. [Phase 2](phase2.md)'s table showed that households paying insurance premiums meet the goal 9.7 points more often — measured on its own, with nothing else accounted for.
>
> The trap is treating a column of such numbers as a menu of separate effects to be summed. They overlap heavily: buying insurance is largely a symptom of having money, so once income is in the model the insurance indicator has almost nothing left to say. It was never contributing 9.7 points of its own; it was reflecting income's contribution back at us. This is the single most common way a business recommendation goes wrong — reading an isolated comparison as though it were an independent lever someone could pull.

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
