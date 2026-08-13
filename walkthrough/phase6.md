# Phase 6 — Unsupervised Extension

**Source:** [README § Phase 6 — Unsupervised Extension](../README.md#phase-6--unsupervised-extension)
**Notebook:** [`notebooks/06_clustering_personas.ipynb`](../notebooks/06_clustering_personas.ipynb)
**Builds on:** [Phase 2 — Feature Engineering](phase2.md), [Phase 5 — Explainability](phase5.md)

Phases 1–5 build a supervised model of `Goal_Met`. Phase 6 asks a different kind of question: **without using the label at all**, does the population naturally separate into distinct spending personas on the same expense-ratio features Phase 2 engineered — and if so, do those personas say anything about `Goal_Met`? This is a check on whether the supervised model's story (Phase 5: `Loan_Repayment_Ratio`, `Education_Ratio`, and `City_Tier` drive risk) is corroborated by an entirely independent, unsupervised method, or whether it's an artifact of how the classifier happens to be built.

---

## Research questions & answers

| # | Question | Answer |
|---|---|---|
| 1 | What spending personas emerge from clustering on expense-category proportions? | Three personas, separated almost entirely by two near-categorical signals already present in the raw data: **Persona 0** — Tier-1 residents with dependents (n=4,758); **Persona 1** — no dependents, any city tier (n=4,061); **Persona 2** — Tier-2/Tier-3 residents with dependents (n=11,181). Not a rich multivariate spending-mix discovery — `Education_Ratio` and `Rent_Ratio` alone account for essentially all the separation. |
| 2 | How many clusters are statistically justified? | `k=3` — silhouette score peaks there (0.096) and declines for every larger `k`; the elbow in inertia is also past by `k=3`. Silhouette scores are uniformly low (< 0.1) across every `k` tested — the dataset supports only weak cluster structure overall. |
| 3 | Do the resulting personas correlate meaningfully with `Goal_Met`? | Yes, sharply: **all 112 at-risk individuals (100%) fall in Persona 0**; Personas 1 and 2 have an at-risk rate of exactly 0% (χ² = 360.8, p ≈ 4.5×10⁻⁷⁹). This independently corroborates Phase 5's SHAP finding that `City_Tier_Tier_1` pushes the model's decision score toward "at-risk." |

The rest of this document walks through *how* the notebook arrives at each answer, cell by cell, and why each analysis step was chosen.

---

## Notebook walkthrough

The notebook carries only section headers and code; the reasoning behind each step lives here.

### Cell 0 (markdown) — Title

### Cell 1 (code) — Imports, data load, and ratio features

Rebuilds `Goal_Met` and the Phase 2 expense-to-income ratio columns (`{category}_Ratio` for each of the 11 expense categories) directly from `dataset/data.csv`, matching every prior phase's self-contained, rebuild-from-raw-CSV pattern. `Goal_Met` is reconstructed here **only to evaluate the clusters against it afterward** — it plays no role in fitting the clustering itself, which is the entire point of an unsupervised method: if the resulting groups still say something about the label, that's discovered structure, not something engineered in.

### Cell 2 (markdown) — "Choosing k"

### Cell 3 (code) — Elbow and silhouette sweep

```python
scaler = StandardScaler()
X_ratios = scaler.fit_transform(df[ratio_cols])

for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_ratios)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_ratios, labels))
```

**What it does:** Standardizes the 11 ratio columns (`StandardScaler`, the same choice Phase 2 recommended for distance-based methods — KMeans is exactly that) and fits `KMeans` for every `k` from 2 to 8, recording both inertia (within-cluster sum of squares — the elbow-method input) and the silhouette score (how well-separated the resulting clusters are — ranges from -1 to 1, higher is better).

**Why both diagnostics, not just one:** inertia always decreases as `k` grows (more clusters can only fit the data at least as well), so it never has a true minimum to pick — only an "elbow" where the rate of decrease slows, which is often subjective to read off a chart. Silhouette score doesn't have that problem: it penalizes over-splitting directly, so it can and does peak at a specific `k` and decline afterward. Using both means the elbow's informal read can be checked against a metric that doesn't share its blind spot.

**Result:** silhouette peaks at `k=3` (0.096) and declines monotonically for every larger `k`; the elbow in the inertia curve is also past by `k=3` (the per-`k` improvement flattens out noticeably from there). Both diagnostics agree, so `k=3` is used for the rest of the notebook.

**Why report the low absolute silhouette value plainly, rather than only the relative ranking:** a silhouette of ~0.1 is weak by the standard rule of thumb (values above ~0.5 are considered strong, reasonable structure is usually >0.25) — this dataset's expense-ratio space does not have well-separated natural clusters. Reporting `k=3` as "the best available `k`" rather than "strong evidence of 3 real personas" is the honest framing, and it's exactly what the persona-composition finding below explains mechanically.

### Cell 4 (markdown) — "Fitting k=3 and profiling the personas"

### Cell 5 (code) — Fitting KMeans and building the persona profile table

```python
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
df["Persona"] = kmeans.fit_predict(X_ratios)
```

**What it does:** Fits the chosen `k=3` model and attaches the resulting cluster label as a new `Persona` column, then tabulates each persona's size and mean value for every one of the 11 ratio features.

### Cell 6 (code) — Between-persona variance share

```python
between_var = pd.DataFrame(X_ratios, columns=ratio_cols).assign(Persona=df["Persona"]).groupby("Persona")[ratio_cols].mean().var()
total_var = pd.DataFrame(X_ratios, columns=ratio_cols).var()
(between_var / total_var).sort_values(ascending=False)
```

**What it does:** For each standardized feature, computes the variance *of the three persona means* relative to that feature's *total* variance across the whole population — a quick, single-number version of a one-way ANOVA's eta-squared, showing how much of each feature's spread is "explained" by which persona a row belongs to.

**Why this is necessary, not just a nice-to-have:** the persona profile table (Cell 5) shows *what* each persona's average ratios are, but with 11 features it's easy to eyeball small, meaningless differences as if they were the defining trait. This ranks features by how much they actually *drove* the clustering, which is a very different (and much shorter) list.

**Result:** `Education_Ratio` (share ≈ 1.74) and `Rent_Ratio` (share ≈ 1.12) dominate; every other feature's share is below 0.001 — three orders of magnitude smaller. The clustering is, in effect, a 2-feature clustering wearing an 11-feature label.

### Cell 7 (markdown) — "Persona composition"

### Cell 8 (code) — What `Dependents`, `City_Tier`, `Income`, and `Age` look like per persona

### Cell 9 (code) — Persona × `City_Tier` crosstab

**What it does:** Cross-references the personas against columns that were never part of the clustering input (`Dependents`, `City_Tier`, `Income`, `Age`, and the raw share of individuals with `Education_Ratio == 0`) to explain *why* `Education_Ratio` and `Rent_Ratio` split the way they do.

**Result, and why it matters:** `Education_Ratio` is exactly 0 for every individual with 0 `Dependents` and positive otherwise — in this dataset it functions as a `Dependents == 0` indicator, not a continuous spending choice. `Rent_Ratio` separates `City_Tier_1` residents from `City_Tier_2`/`City_Tier_3` residents, which lines up exactly with Phase 2's and Phase 5's finding that `Rent_Ratio` has near-zero *within*-tier variance (plausibly because this dataset generates `Rent` as a roughly fixed proportion of income conditional on tier). `Income`, `Age`, and `Occupation` are flat across all three personas (mean age ≈ 41 in every persona; occupation split ≈25%/25%/25%/25% in every persona) — they play no role in the split at all.

**Conclusion for Question 1:** the three personas are **Persona 0 — Tier-1 residents with dependents**, **Persona 1 — no dependents (any tier)**, and **Persona 2 — Tier-2/Tier-3 residents with dependents**. This is a demographic/geographic split that was already fully present in the raw `Dependents` and `City_Tier` columns — clustering on 11 ratio features rediscovered a 2-column segmentation rather than surfacing a new, richer behavioral archetype. That's a legitimate, useful finding in its own right (see Question 3), just not the kind of discovery "spending persona" might suggest at first read.

### Cell 10 (markdown) — "Persona vs. `Goal_Met`"

### Cell 11 (code) — Cross-tabulating personas against the target

```python
persona_goal = pd.crosstab(df["Persona"], df["Goal_Met"])
persona_goal["at_risk_rate"] = persona_goal["Not met (0)"] / persona_goal.sum(axis=1)
```

### Cell 12 (code) — Chi-square test of independence

```python
chi2, p_value, dof, _ = chi2_contingency(pd.crosstab(df["Persona"], df["Goal_Met"]))
```

**What it does:** Tests whether persona membership and `Goal_Met` are statistically independent. A large χ² with a small p-value rejects independence — persona membership and at-risk status move together more than chance would predict.

### Cell 13 (code) — At-risk rate by persona (chart)

**Result:** χ² = 360.8, p ≈ 4.5×10⁻⁷⁹ — decisively not independent. More useful than the p-value itself: **all 112 at-risk individuals in the entire dataset fall in Persona 0**; Personas 1 and 2 have an at-risk rate of exactly 0.00%. This is not a mild skew — it's total concentration in one segment out of three.

**Answer to Question 3:** yes, and the relationship is as strong as a categorical split can be. This independently corroborates Phase 5's SHAP finding that `City_Tier_Tier_1` is among the winning model's top global drivers, pushing predictions toward "at-risk" — an unsupervised method that never saw the label agrees with the supervised model's explanation, from a completely different angle.

### Cell 14 (markdown) — "Persisting results"

### Cell 15 (code) — Persisting results

Saves `results/persona_profiles.csv` (the full per-persona ratio-feature means) and `results/persona_clusters.png` (the at-risk-rate-by-persona chart).

---

## What Phase 6 sets up for later phases

| Finding | Where it gets used |
|---|---|
| All 112 at-risk individuals fall in the Tier-1-with-dependents persona | Headline targeting finding for Phase 7's business translation — corroborates Phase 5's SHAP result via an independent, unsupervised method |
| Clustering on 11 ratio features reduces to a 2-feature (`Education_Ratio`, `Rent_Ratio`) split, both proxies for columns already in the raw data | An honest scope note for Phase 7: a persona-based targeting layer would add complexity without adding signal beyond `City_Tier` and `Dependents`, which the Phase 4 model already uses |
| Silhouette scores are uniformly low (< 0.1) across every `k` tested | A limitation worth stating plainly rather than overstating "3 natural personas" as strong multivariate structure |

**Next:** [Phase 7 — Business Translation](phase7.md) (`07_business_translation.ipynb`), which converts this phase's and Phase 4/5's findings into stakeholder-facing recommendations and quantifies where the most actionable, recoverable savings potential sits.
