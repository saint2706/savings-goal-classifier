# Phase 5 — Explainability

**Source:** [README § Phase 5 — Explainability](../README.md#phase-5--explainability)
**Notebook:** [`notebooks/05_explainability.ipynb`](../notebooks/05_explainability.ipynb)
**Builds on:** [Phase 4 — Model Comparison](phase4.md)

Phase 4 selected **SVM (Linear)** — `LinearSVC`, `C=10`, `class_weight="balanced"`, decision threshold `t≈-0.4` (chosen via cross-validation) — as the winning model, on the grounds that it was one of only two models (alongside Logistic Regression) reaching perfect cross-validated recall on the at-risk class (`Goal_Met = 0`), with the better precision and macro-F1 of the two. A model chosen partly for a business-facing purpose is only useful if the *reasons* behind its predictions can be explained to the people who have to act on them. Phase 5's job is to answer that directly, using SHAP.

Because the winning model is **linear**, this notebook uses SHAP's exact `LinearExplainer` rather than the sampled, permutation-based approximation a non-linear model (e.g. the MLP that an earlier, less careful comparison had briefly pointed to) would require.

---

## Research questions & answers

| # | Question | Answer |
|---|---|---|
| 1 | Which features matter most globally for the winning model? | `Loan_Repayment_Ratio` dominates (mean \|SHAP\| ≈ 6.21, more than 2× the next feature), followed by `Education_Ratio`, `City_Tier_Tier_1`, `Groceries_Ratio`, `City_Tier_Tier_2`, `Utilities_Ratio`, and `Rent_Ratio` — expense-ratio and city-tier features, not raw `Income`, `Age`, or `Dependents`, drive the model. |
| 2 | Are there notable interaction effects (e.g., `City_Tier` × `Dependents`)? | **No — structurally impossible for this model.** Because the winning model is linear and SHAP is computed with `feature_perturbation="interventional"`, every feature's contribution is provably a function of that one feature's own value alone (verified: every feature's SHAP values correlate at \|r\| ≥ 0.9999999999 with its own raw values). No feature's contribution can depend on any other feature's value — including the README's `Dependents` × `City_Tier` example. |
| 3 | Can individual predictions be explained in plain business language? | Yes, and *exactly* — SHAP decomposes each prediction into the same handful of globally important, business-tracked features, producing a concrete, actionable sentence per individual, with no approximation involved. |

The rest of this document walks through *how* the notebook arrives at each answer, cell by cell.

---

## Notebook walkthrough

The notebook carries only section headers and code; the reasoning behind each step lives here. Phase 4's handoff: the winning model's identity, hyperparameters, and threshold, and why it being linear specifically changes the explainability approach available (exact `LinearExplainer` vs. an approximate, sampled explainer).

### Cell 0 (markdown) — Title

### Cell 1 (code) — Rebuilding the feature matrix, cast to `float`

Rebuilds the same engineered matrix and stratified split used in every prior phase, cast to `float` (SHAP's default tabular masker errors on the `bool` dtype `pd.get_dummies` produces for one-hot columns).

### Cell 2 (markdown) — "Retraining Phase 4's winning model"

### Cell 3 (code) — Retraining Phase 4's winning model

```python
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_full.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_full.columns, index=X_test.index)

svm_clf = LinearSVC(class_weight="balanced", C=10, max_iter=5000, random_state=42)
svm_clf.fit(X_train_scaled, y_train)

CHOSEN_THRESHOLD = -0.4
test_scores = svm_clf.decision_function(X_test_scaled)
pred = (test_scores >= CHOSEN_THRESHOLD).astype(int)
```

**What it does:** Hardcodes Phase 4's found hyperparameters and threshold, and refits directly rather than re-running the search. Unlike Phase 4's `ImbPipeline`, the scaler and classifier are fit as two **separate** steps here — `LinearExplainer` needs direct access to the linear estimator's coefficients, which a bundled `Pipeline` object doesn't expose as conveniently.

**Sanity check:** the refit model's metrics (accuracy 0.9992, macro-F1 0.9679, `recall_0` 1.0000, `precision_0` 0.8800) match Phase 4's final test-set numbers exactly — confirming this notebook explains the actual model Phase 4 selected, not an approximation of it.

### Cell 4 (markdown) — "Global feature importance"

`LinearSVC` has coefficients, which are themselves a form of feature importance — but they don't decompose an *individual* prediction the way the per-row explanations below need, and aren't directly comparable to a SHAP value. `shap.LinearExplainer(svm_clf, X_train_scaled, feature_perturbation="interventional")` computes **exact** Shapley values for a linear model — for this setting, the SHAP value of feature *j* on row *i* is provably `coefficient_j × (x_ij − mean_j)`, with no approximation and no cross-feature term. Because it's exact, it runs on **all 4,000 test rows in under a second** — no background-sample-size trade-off to disclose, unlike a permutation-based explainer would need.

### Cell 5 (code) — Building the explainer

### Cell 6 (code) — Global feature importance

```python
mean_abs_shap = pd.Series(np.abs(shap_values.values).mean(axis=0), index=X_full.columns).sort_values(ascending=False)
```

**Result:** `Loan_Repayment_Ratio` is by far the most important feature (mean \|SHAP\| ≈ 6.21, more than double `Education_Ratio`'s ≈ 2.66). Behind those two: `City_Tier_Tier_1`, `Groceries_Ratio`, `City_Tier_Tier_2`, `Utilities_Ratio`, and `Rent_Ratio`, all in a similar (~1.0–1.2) range. This matches Phase 3's finding that `Loan_Repayment_Ratio` was the strongest raw correlate of `Goal_Met` — the model relies on a signal a simple correlation already flagged. `Income`, `Age`, and `Dependents` are absent from the top features: spending discipline, not income level, is what the model leans on.

### Cell 7 (markdown) — "Interaction effects"

### Cell 8 (code) — Proving interaction effects are structurally absent

```python
same_feature_corr = {
    feature: np.corrcoef(shap_values.values[:, j], X_test_scaled[feature].values)[0, 1]
    for j, feature in enumerate(X_full.columns)
}
worst_feature, worst_corr = min(same_feature_corr.items(), key=lambda kv: abs(kv[1]))
```

**What it does:** For every feature, correlates that feature's column of SHAP values against that same feature's own raw values across all 4,000 test rows. If any interaction existed — if `Loan_Repayment_Ratio`'s contribution depended even slightly on `City_Tier` — rows sharing a `Loan_Repayment_Ratio` value but differing in `City_Tier` would get different SHAP values, pulling this correlation measurably below 1.0.

**Result:** every feature's correlation is 1.0 to at least 9 decimal places (worst case: `Occupation_Student` at 0.9999999999). Each feature's SHAP contribution is fully and exactly determined by that feature's own value alone, with zero influence from any other feature.

### Cell 9 (code) — By-tier interaction check

```python
tier_cols = [c for c in X_full.columns if c.startswith("City_Tier_")]
interaction_df["max_over_min_ratio"] = interaction_df.max(axis=1) / interaction_df.min(axis=1)
```

**Conclusion:** a clean, structural **no**. Because the winning model is linear and its SHAP values are provably additive (verified above), `City_Tier` cannot mathematically change how much `Dependents` — or any other feature — contributes to a prediction. A supplementary by-tier grouping confirms this empirically: `Loan_Repayment_Ratio` and `Education_Ratio`'s SHAP spread is nearly identical across tiers (ratios ~1.0–1.1); `Dependents` (the README's specific example) is similarly flat and, more importantly, tiny in absolute size (SHAP std ≈ 0.23 vs. `Loan_Repayment_Ratio`'s ≈ 6.6–6.8) — it simply isn't important in any tier. `Rent_Ratio` shows a large spread ratio (~3×) but with near-zero *within-tier* variance — almost every individual in a given city tier has nearly the same `Rent_Ratio`, a **data characteristic** (plausibly this dataset generating `Rent` as a roughly fixed proportion of income conditional on tier, consistent with the README's Limitations note about possibly-synthetic figures), not a model-learned interaction. Capturing genuine interactions, if ever needed, would require one of Phase 4's non-linear candidates — a trade-off Phase 4 made deliberately in favor of this model's superior cross-validated recall.

### Cell 10 (markdown) — "Individual predictions"

### Cell 11 (code) — Explaining two representative rows

```python
at_risk_pos = np.argsort(test_scores)[0]     # lowest decision score => most confidently at-risk
on_track_pos = np.argsort(test_scores)[-1]   # highest decision score => most confidently on-track
```

With the actual computed numbers:

- **At-risk example** (test row 18909, decision score ≈ −5.46, actual `Goal_Met = 0`): the single largest reason is an unusually high `Loan_Repayment_Ratio` (≈19% of income, shap ≈ −13.64 — more than double the next contributor); below-typical `Education_Ratio`, living in a **Tier-1 city**, an above-typical `Rent_Ratio`, and above-typical `Groceries_Ratio` each add further downward pressure. In plain business language: *"This customer is flagged at-risk mainly because loan repayments are consuming an unusually large share of their income, compounded by above-typical rent and grocery spending while living in a higher cost-of-living city — a savings nudge focused on debt repayment burden is the most relevant intervention."*
- **On-track example** (test row 786, decision score ≈ +41.97, actual `Goal_Met = 1`): the two largest drivers are zero education spending and zero loan-repayment burden, together contributing more than every other feature combined; above-typical `Groceries_Ratio`, `Utilities_Ratio`, and `Insurance_Ratio` each add a smaller positive contribution. In plain language: *"This customer isn't at risk primarily because they carry no loan-repayment or education-spending burden — that comfortably outweighs their other, more typical living expenses, and no outreach is needed here."*

Both examples share the property a stakeholder needs: the explanation is always in terms of features the business already tracks, and — because the model is linear — every explanation is mathematically exact, not an approximation.

### Cell 12 (code) — Waterfall plots

Visualizes the same two breakdowns as SHAP waterfall charts.

### Cell 13 (markdown) — "Persisting results"

### Cell 14 (code) — Persisting results

Saves the global SHAP importance chart to `results/shap_summary.png`.

Phase 4's non-linear candidates (Random Forest, XGBoost, MLP) remain available if a future iteration needs genuine interaction-capturing capability — a trade-off knowingly made in favor of the linear model's superior cross-validated recall.

---

## What Phase 5 sets up for later phases

| Finding | Where it gets used |
|---|---|
| `Loan_Repayment_Ratio` is the dominant global driver, followed by `Education_Ratio`, `City_Tier_Tier_1`, `Groceries_Ratio` | Candidate features to check as clustering inputs in Phase 6, and the headline finding for Phase 7's business translation |
| No interaction effects exist for this model, by construction | An honest scope-limiting finding for Phase 7 — any interaction-based recommendation would need a different (non-linear) model |
| `Rent_Ratio` has near-zero within-tier variance | A data-generation observation worth flagging in Phase 8's reporting, tying back to the README's synthetic-data limitation |
| Individual predictions decompose exactly into a small number of business-tracked features | Demonstrates the model is usable by a non-technical marketing team, directly answering Phase 0's framing requirement |

**Next:** [Phase 6 — Unsupervised Extension](phase6.md) (`06_clustering_personas.ipynb`), which clusters individuals on expense-category proportions to find natural spending personas, and checks whether those personas correlate with `Goal_Met` — using the same expense-ratio features this phase just showed drive the supervised model's predictions.
