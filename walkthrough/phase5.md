# Phase 5 — Explainability

**Source:** [README § Phase 5 — Explainability](../README.md#phase-5--explainability)
**Notebook:** [`notebooks/05_explainability.ipynb`](../notebooks/05_explainability.ipynb)
**Builds on:** [Phase 4 — Model Comparison](phase4.md)

Phase 4 compared six model families and selected the **Neural Net (MLP)** — `hidden_layer_sizes=(64,)`, `alpha=0.001`, trained on an oversampled, scaled feature matrix — as the winning model, on the grounds that it reached perfect recall on the at-risk class (`Goal_Met = 0`) with the best precision and macro-F1 among the models that did. A model chosen partly for a business-facing purpose is only useful if the *reasons* behind its predictions can be explained to the people who have to act on them. Phase 5's job is to answer that directly, using SHAP.

---

## Research questions & answers

| # | Question | Answer |
|---|---|---|
| 1 | Which features matter most globally for the winning model? | `Loan_Repayment_Ratio` dominates (mean \|SHAP\| ≈ 0.0204, ~1.7× the next feature), followed by `City_Tier_Tier_1`, `Education_Ratio`, `Groceries_Ratio`, and `Utilities_Ratio` — expense-ratio features, not raw `Income`, `Age`, or `Dependents`, drive the model. |
| 2 | Are there notable interaction effects (e.g., `City_Tier` × `Dependents`)? | The specific `Dependents` × `City_Tier` example does **not** hold — `Dependents`' SHAP contribution is negligible in every tier. A real interaction exists instead between `City_Tier` and expense-ratio burden: `Education_Ratio`'s effect is ~5× more variable in Tier-1 than Tier-3, and `Loan_Repayment_Ratio`'s is ~2.6× more variable in Tier-1 than Tier-2/3. |
| 3 | Can individual predictions be explained in plain business language? | Yes — SHAP decomposes each prediction into the same handful of globally important, business-tracked features (expense ratios, city tier), producing a concrete, actionable sentence per individual rather than an opaque score. |

The rest of this document walks through *how* the notebook arrives at each answer, cell by cell.

---

## Notebook walkthrough

### Cell 0 (markdown) — Title and scope

States the three Phase 5 questions and Phase 4's handoff: the winning model's identity and hyperparameters, and why explainability is a first-class requirement (Phase 0 framed this as a decision-support tool for a marketing team, not a black box).

### Cell 1 (code) — Rebuilding the feature matrix, cast to `float`

```python
# SHAP's masker requires numeric (non-boolean) data; the one-hot columns are bool by default.
X_full = engineered.drop(columns=["Goal_Met"]).astype(float)
```

**What it does:** Rebuilds the same engineered matrix and stratified split used in every prior phase, with one addition: `X_full` is cast to `float` right after construction.

**Why the cast is necessary:** `pd.get_dummies` produces `bool`-dtype one-hot columns by default in the pandas version this project uses. SHAP's default tabular masker compares feature values with `np.isclose`, which raises a `TypeError` on boolean input. Casting to `float` immediately after building the matrix sidesteps the error without changing any value (`True`/`False` become `1.0`/`0.0`, identical to how they're already treated numerically).

### Cells 3–5 — Retraining Phase 4's winning model

```python
best_mlp = ImbPipeline([
    ("scaler", StandardScaler()),
    ("sampler", RandomOverSampler(random_state=42)),
    ("clf", MLPClassifier(hidden_layer_sizes=(64,), alpha=0.001, max_iter=500, early_stopping=True, random_state=42)),
])
best_mlp.fit(X_train, y_train)
...
print(f"Accuracy: {accuracy:.4f} (Phase 4: 0.9982)")
print(f"Macro-F1: {f1_macro:.4f} (Phase 4: 0.9309)")
```

**What it does:** Hardcodes Phase 4's found hyperparameters and refits the exact pipeline directly, rather than re-running the search — the search's job (choosing the hyperparameters) is already done. Prints the refit model's test metrics next to Phase 4's original numbers for direct comparison.

**Why this sanity check matters before any explainability work:** the printed numbers match Phase 4's exactly (accuracy 0.9982, macro-F1 0.9309, `recall_0` 1.0000, `precision_0` 0.7586). If they hadn't, it would mean this notebook's "identical" rebuild wasn't actually identical, and every SHAP value that follows would be explaining a subtly different model than the one Phase 4 selected.

### Cells 6–8 — Question 1 setup: why SHAP, and the computational trade-off

`MLPClassifier` has no built-in feature-importance attribute — no split gains like a tree, no coefficients like a linear model. Its "reasoning" is distributed across hidden-layer weights that don't map cleanly onto input features. **SHAP** works around this by treating the model as a black box and calling only its `predict_proba`, attributing each prediction's deviation from the average to each input feature in a way that's mathematically guaranteed to add up exactly (Shapley values).

```python
background = shap.sample(X_train, 100, random_state=42)
sample_test = X_test.sample(300, random_state=42)
explainer = shap.Explainer(predict_class1_proba, background, seed=42)
shap_values = explainer(sample_test)
```

**What it does:** Wraps `best_mlp.predict_proba`'s `P(Goal_Met = 1)` column in a plain function, and computes SHAP values with a **permutation-based** explainer (the fallback for an arbitrary black-box function) against a **100-row background sample** and a **300-row test sample** — not the full 4,000-row test set.

**Why sample rather than explain the whole test set, and why `seed=42`:** permutation-based SHAP is computationally expensive (roughly 30 seconds for 300 rows against a 100-row background on this hardware); explaining all 4,000 test rows would scale that up substantially for a notebook meant to run in a reasonable time. This trade-off is disclosed explicitly here, not hidden — the sampled values are representative, not exhaustive. The explicit `seed=42` matters separately: SHAP's permutation explainer has its own internal randomness that isn't controlled by `numpy`'s global random state, so without a seed, re-running this cell would produce slightly different SHAP values each time (verified during development — the same code without a seed gave `Loan_Repayment_Ratio`'s mean \|SHAP\| as 0.0200 on one run and 0.0204 on another). Setting `seed=42` makes every number in this notebook exactly reproducible.

### Cells 9–13 — Global feature importance and Answer to Question 1

```python
mean_abs_shap = pd.Series(np.abs(shap_values.values).mean(axis=0), index=X_full.columns).sort_values(ascending=False)
```

**What it does:** Averages each feature's *absolute* SHAP value across the 300 sampled rows (largest-average-impact features first, whether they typically push predictions up or down), plotted as a bar chart, followed by SHAP's beeswarm plot (adds direction: point color = raw feature value, position = impact).

**Why absolute value:** a feature that pushes some rows up and others down in equal measure would average to ~0 under a plain mean, hiding that it matters a great deal to individual predictions.

**Answer to Question 1** (Cell 13, markdown): `Loan_Repayment_Ratio` is by far the most important feature (mean \|SHAP\| ≈ 0.0204, ~1.7× the next-highest), and the beeswarm plot shows a clean pattern — high `Loan_Repayment_Ratio` (red) pushes predicted `P(Goal_Met=1)` down, low (blue) pushes it up. This matches Phase 3's finding that it was the strongest raw correlate of `Goal_Met`, confirming the model relies on a signal a simple correlation already flagged, not something spurious. Behind it: `City_Tier_Tier_1`, `Education_Ratio`, `Groceries_Ratio`, `Utilities_Ratio` — all expense-ratio or city-tier features, none of them `Income`, `Age`, or `Dependents` directly. Notably, `Rent_Ratio` — the second-strongest raw correlate in Phase 3 — ranks *lower* in SHAP importance than `City_Tier`, plausibly because `City_Tier` already captures much of the same signal (Phase 2 found rent burden rises with city tier), leaving less additional value for the model to extract from the ratio directly.

### Cells 14–18 — Question 2: checking for interaction effects

```python
tier_cols = [c for c in X_full.columns if c.startswith("City_Tier_")]
tier_of_row = sample_test[tier_cols].idxmax(axis=1).str.replace("City_Tier_", "")
dep_shap = shap_values.values[:, dep_idx]
dep_by_tier = pd.DataFrame(...).groupby("tier")["dependents_shap"].agg(["mean", "std", "count"])
```

**What it does:** Recovers each sampled row's `City_Tier` from its one-hot columns, then checks the README's specific example first — does `Dependents`' SHAP contribution vary by tier? — by grouping its SHAP values by tier and comparing spread (`std`) and direction (`mean`).

**Result for the specific example:** `Dependents`' SHAP values are tiny in every tier (`std` ≈ 0.001–0.003, two orders of magnitude below `Loan_Repayment_Ratio`'s typical contribution) and don't differ meaningfully across tiers. **The interaction the question poses as an example does not hold up** — `Dependents` simply isn't important enough, in any tier, for an interaction to be visible. This is reported as-is rather than treated as a reason to keep searching for a "yes" — Phase 5's job is to report what the model actually does.

```python
top_ratio_features = ["Loan_Repayment_Ratio", "Education_Ratio", "Rent_Ratio", "Income"]
interaction_df["max_over_min_ratio"] = interaction_df.max(axis=1) / interaction_df.min(axis=1)
```

**What it does:** Repeats the by-tier grouping for the four highest-ranked non-categorical features from Question 1, comparing the SHAP-value spread across tiers with a single sortable ratio (`max_over_min_ratio`).

**Answer to Question 2** (Cell 18, inline): `Education_Ratio` shows the clearest interaction with `City_Tier` — its SHAP-value spread is ~5× wider in Tier-1 than Tier-3 — and `Loan_Repayment_Ratio` shows a smaller version of the same pattern (~2.6× wider in Tier-1 than Tier-2/3). In plain terms: **a high education- or loan-repayment burden moves the model's prediction much more sharply for a Tier-1 resident than for someone in Tier-2 or Tier-3** — the model has effectively learned that the same expense ratio is a stronger risk signal in a higher cost-of-living area. So the *specific* example the question poses (`Dependents`) doesn't hold, but a related, more consequential interaction — between `City_Tier` and expense-ratio burden generally — does, and it involves feature families Question 1 already flagged as globally important.

### Cells 19–24 — Question 3: individual predictions in plain language

```python
proba_sample = best_mlp.predict_proba(sample_test)[:, 1]
at_risk_pos = np.argsort(proba_sample)[0]    # lowest P(Goal_Met=1) => most confidently at-risk
on_track_pos = np.argsort(proba_sample)[-1]  # highest P(Goal_Met=1) => most confidently on-track
```

**What it does:** Picks the two most confidently-predicted individuals in the 300-row sample — one at-risk, one on-track — prints each one's top-5 SHAP contributors alongside their raw feature values, and renders SHAP's waterfall plot for each (a step-by-step chart from the background-average prediction to that individual's final one).

**Answer to Question 3** (Cell 24, markdown), with the actual computed numbers:

- **At-risk example** (test row 18421, predicted `P(Goal_Met=1)` ≈ 0.088, actual `Goal_Met = 0`): the single largest reason is an unusually high `Loan_Repayment_Ratio` (shap ≈ −0.329), with elevated `Groceries_Ratio`, `Insurance_Ratio`, and `Utilities_Ratio` each adding further downward pressure; a below-typical `Education_Ratio` (shap ≈ +0.076) pulls the other way but not nearly enough to offset the rest. In plain business language: *"This customer is flagged at-risk mainly because loan repayments are consuming an unusually large share of their income, compounded by above-typical grocery, insurance, and utility spending — a savings nudge focused on debt repayment burden is the most relevant intervention."*
- **On-track example** (test row 3290, predicted `P(Goal_Met=1)` ≈ 1.000, actual `Goal_Met = 1`): the two largest drivers are effectively tied — zero loan-repayment burden (shap ≈ +0.0135) and zero education spending (shap ≈ +0.0136) — together outweighing the small negative pull of also living in a Tier-1 city (shap ≈ −0.0119). In plain language: *"This customer isn't at risk primarily because they carry no loan-repayment or education-spending burden — that outweighs the higher cost-of-living pressure of their city, and no outreach is needed here."*

Both examples share the property a stakeholder actually needs: **the explanation is always in terms of features the business already tracks** (expense ratios, city tier) **and can act on**, not an opaque score.

### Cells 25–26 — Persisting results

```python
os.makedirs("../results", exist_ok=True)
fig.savefig("../results/shap_summary.png", dpi=150)
```

**What it does:** Saves the global SHAP importance chart to `results/shap_summary.png`, matching the repository structure the README documents as planned for this phase.

### Cell 27 (markdown) — Summary table and handoff to Phase 6

Closes with the three questions and answers, and points to Phase 6 — Unsupervised Extension, which clusters individuals on the same expense-ratio features this notebook just showed drive the supervised model, from a complementary unsupervised angle.

---

## What Phase 5 sets up for later phases

| Finding | Where it gets used |
|---|---|
| `Loan_Repayment_Ratio` is the dominant global driver, followed by `City_Tier_Tier_1`, `Education_Ratio`, `Groceries_Ratio`, `Utilities_Ratio` | Candidate features to check as clustering inputs in Phase 6, and the headline finding for Phase 7's business translation |
| `City_Tier` amplifies the effect of `Education_Ratio` and `Loan_Repayment_Ratio` specifically (not `Dependents`) | An actionable, non-obvious finding for Phase 7 — expense-ratio-based interventions should be prioritized differently by city tier |
| Individual predictions decompose into a small number of business-tracked features with a concrete plain-language explanation | Demonstrates the model is usable by a non-technical marketing team, directly answering Phase 0's framing requirement |
| SHAP computation requires a `float`-cast feature matrix and a fixed `seed` for reproducibility | A reusable technical note for any future notebook in this project that applies SHAP to a non-tree model |

**Next:** Phase 6 — Unsupervised Extension (`06_clustering_personas.ipynb`, not yet created), which clusters individuals on expense-category proportions to find natural spending personas, and checks whether those personas correlate with `Goal_Met` — using the same expense-ratio features this phase just showed drive the supervised model's predictions.
