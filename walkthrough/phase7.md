# Phase 7 — Business Translation

**Source:** [README § Phase 7 — Business Translation](../README.md#phase-7--business-translation)
**Notebook:** [`notebooks/07_business_translation.ipynb`](../notebooks/07_business_translation.ipynb)
**Builds on:** [Phase 4 — Model Comparison](phase4.md), [Phase 5 — Explainability](phase5.md), [Phase 6 — Unsupervised Extension](phase6.md)

Phases 1–6 answer analytical questions. Phase 7 asks a different one: **given everything found so far, what should a fintech marketing team actually do, and what should they be told before they act on it?** This phase does not introduce new modeling — it reframes existing results (Phase 4's classifier, Phase 5's SHAP explanations, Phase 6's personas, and the dataset's own `Potential_Savings_*` columns, unused until now) into recommendations, a quantified answer on where unrealized savings concentrate, and an explicit statement of the model's reliability limits.

---

## Research questions & answers

| # | Question | Answer |
|---|---|---|
| 1 | What are the 3–5 most actionable findings, stated as recommendations rather than statistics? | Five: (1) target outreach by `City_Tier`, not occupation or age; (2) lead nudge campaigns with grocery-spend reduction; (3) frame the ask as small and achievable, not remedial; (4) deploy the SVM (Linear) classifier at threshold t=-0.4 for flagging; (5) don't build a separate persona-based targeting layer. See the Recommendations table below for the evidence behind each. |
| 2 | Which expense category carries the most unrealized (potential) savings across the population? | `Groceries` — ₹18.2M/month in aggregate (35% of the ₹51.6M addressable across all 8 tracked categories) and ₹912/month per person, roughly double the next-largest category (`Transport`). |
| 3 | Where does the model fail or lose reliability — what should a stakeholder be told before acting on it? | Four caveats: the minority-class sample is tiny (112 of 20,000 rows); precision is ~0.88, not 1.0 (~1 in 8 flagged individuals are false positives); the model is linear and structurally blind to any real interaction effect (Phase 5); and the dataset shows signs of being synthetically generated (Phases 1, 2, 5). |

The rest of this document walks through *how* the notebook arrives at each answer, cell by cell, and why each analysis step was chosen.

---

## Notebook walkthrough

The notebook carries only section headers and code; the reasoning behind each step lives here.

### Cell 0 (markdown) — Title

### Cell 1 (code) — Imports and data load

Loads `dataset/data.csv` directly (not the Phase 2 engineered matrix — this phase works with the raw `Potential_Savings_*` and money columns, which were never part of the feature set). Reconstructs `Goal_Met`, and adds two derived columns used throughout: `Total_Potential_Savings` (sum of the 8 `Potential_Savings_*` columns) and `Shortfall` (`Desired_Savings - Disposable_Income` — how far short of their own goal an individual falls; positive only for the 112 at-risk individuals).

### Cell 2 (markdown) — "Unrealized savings by category"

### Cell 3 (code) — Aggregating potential savings by category

```python
category_totals = df[potential_cols].sum().sort_values(ascending=False)
category_pct_income = (df[potential_cols].div(df["Income"], axis=0)).mean()
```

**What it does:** For each of the 8 `Potential_Savings_*` columns, computes the population-wide total (₹/month, aggregate), the mean per person, and the mean as a percentage of that person's income — three views of the same ranking, so the answer isn't an artifact of one framing (e.g., a category could look large in aggregate purely because it applies to more people, not because it's large per person).

### Cell 4 (code) — Bar chart of unrealized savings by category

**Answer to Question 2:** `Groceries` leads by a wide margin under every framing — ₹18.2M/month aggregate (35% of the ₹51.6M total addressable across all 8 categories), ₹912/month mean per person, and ~2.2% of income — roughly double `Transport`, the next-largest category (₹9.5M / ₹473 / ~1.1%). This is not a small-household artifact: it holds in aggregate and per-person terms simultaneously.

### Cell 5 (markdown) — "Coverage of the at-risk shortfall"

### Cell 6 (code) — Can addressable savings close the gap?

```python
at_risk = df[df["Goal_Met"] == 0]
coverage = {
    "All 8 categories": (at_risk["Total_Potential_Savings"] >= at_risk["Shortfall"]).mean(),
    "Groceries + Transport only": ((at_risk["Potential_Savings_Groceries"] + at_risk["Potential_Savings_Transport"]) >= at_risk["Shortfall"]).mean(),
    "Groceries only": (at_risk["Potential_Savings_Groceries"] >= at_risk["Shortfall"]).mean(),
}
```

**What it does:** For each of the 112 at-risk individuals, compares their addressable savings potential (under three different scopes: all 8 categories, the top-2 categories, or `Groceries` alone) against their actual shortfall, and reports what fraction of at-risk individuals have *enough* addressable potential to fully close their gap.

**Why this matters, beyond "which category is biggest":** Question 2's answer alone doesn't say whether the recoverable savings are *large enough to matter* for the people who actually need them — a category could be the biggest source of unrealized savings in aggregate while still being a drop in the bucket for the specific people this project cares about (the at-risk group). This cell tests that directly.

**Result:** for **99.1%** of at-risk individuals, their total addressable savings potential across all 8 tracked categories already exceeds their shortfall (mean shortfall ≈ ₹784/month vs. mean total potential ≈ ₹2,912/month — potential exceeds the gap by roughly 3.7× on average). A narrower two-category nudge (`Groceries` + `Transport`) still closes the gap for **87.5%** of them. This is the evidence behind recommendation 3 below: the at-risk classification reflects a recoverable gap, not a structural one.

### Cell 7 (markdown) — "At-risk segment profile"

### Cell 8 (code) — Who is at risk?

**What it does:** Breaks the 112 at-risk individuals down by `City_Tier`, `Occupation`, and mean `Dependents`, comparing each against the population baseline.

**Result:** every one of the 112 at-risk individuals lives in a **Tier-1** city (100%, 0 in Tier-2 or Tier-3); `Occupation` is close to evenly split among them (no concentration); `Dependents` is only mildly elevated (2.25 vs. 1.996 population mean) — nowhere near as decisive a signal as `City_Tier`. This matches Phase 5's SHAP finding (`City_Tier_Tier_1` among the top global drivers) and Phase 6's clustering finding (100% of at-risk individuals fall in the Tier-1-with-dependents persona) — three independent analyses (supervised explanation, unsupervised clustering, and this direct cross-tab) converge on the same segment.

### Cell 9 (markdown) — "Model reliability"

### Cell 10 (code) — Pulling Phase 4's final test-set numbers

Reads `results/final_test_evaluation.csv` (written by Phase 4) rather than retraining anything — Phase 7's job is to communicate the existing model's numbers honestly to a stakeholder audience, not to reproduce Phase 4 or Phase 5.

**Answer to Question 3 — four caveats a stakeholder should have before acting on the model's flags:**

1. **Small minority-class sample.** Only 112 of 20,000 individuals (0.56%) are at-risk. Phase 4's `recall_0 = 1.0` — both cross-validated and on the held-out test set — is measured against roughly 90–112 examples; it is strong evidence, not a guarantee that a genuinely new population would also be caught perfectly.
2. **Precision, not certainty.** `precision_0 ≈ 0.88` means about 1 in 8 people the model flags as at-risk are not. Phase 4's cost framing treats this as acceptable (an unneeded nudge is low-cost), but a stakeholder acting on a specific flagged individual should know it is not a certainty.
3. **No interaction effects, by construction.** The winning model is linear, and Phase 5 proved its SHAP explanations cannot represent any interaction between features. A real-world risk factor that only shows up as a *combination* of features (e.g., high rent stacked with high loan repayment specifically) would not be visible to this model even if it existed in the data.
4. **Possibly synthetic data.** Phases 1, 2, and 5 each separately noted signs the dataset may be synthetically generated (e.g., `Rent_Ratio`'s near-zero within-tier variance). Absolute figures — and possibly the relationships themselves — should be validated against real customer data before this model or these recommendations are deployed operationally.

### Cell 11 (markdown) — "Recommendations"

### Cell 12 (code) — Recommendations table

Builds and saves `results/business_recommendations.csv`:

| Recommendation | Evidence |
|---|---|
| Prioritize outreach by `City_Tier`, not occupation or age | 100% of at-risk individuals (112/112) live in Tier-1 cities; `Occupation` and `Age` show no comparable skew |
| Lead nudge campaigns with grocery-spend reduction | `Groceries` is the largest unrealized-savings category: ₹18.2M/month aggregate, ~2× the next category (`Transport`) |
| Frame the ask as small and achievable, not remedial | `Groceries` + `Transport` alone closes the shortfall for 87.5% of at-risk individuals; all 8 tracked categories close it for 99.1% |
| Deploy the SVM (Linear) classifier at threshold t=-0.4 for flagging | `recall_0 = 1.0` (test set), `precision_0 = 0.88` — catches every at-risk case at the cost of ~12% unnecessary low-cost nudges |
| Do not build a separate persona-based targeting layer | Phase 6 clustering found personas driven by `City_Tier` and `Dependents` — signal already used by the Phase 4/5 model, not an independent source of lift |

**Why these five, and in this order:** they're ordered by how directly actionable they are for a marketing team — the first two are "who to target" and "with what message," the middle one is a framing choice for that same message, and the last two are guidance for whoever owns the classifier deployment (deploy this, don't build that). Each recommendation is paired with the specific number behind it rather than left as an unsupported assertion, so a stakeholder can trace every claim back to a phase and a cell.

---

## What Phase 7 sets up for later phases

| Finding | Where it gets used |
|---|---|
| `Groceries` is the dominant unrealized-savings category, both in aggregate and per-person | The headline number for Phase 8's stakeholder-facing report |
| 99.1% of at-risk individuals have addressable savings potential exceeding their shortfall | Reframes "at-risk" as "recoverable" for Phase 8 — an important tone-setting fact for a non-technical audience |
| Four explicit model-reliability caveats (sample size, precision, no interactions, possibly-synthetic data) | Phase 8 should carry these forward rather than presenting the model's numbers without qualification |
| Five ranked, evidence-backed recommendations | The direct input to Phase 8's "3–4 visualizations that communicate findings fastest" — each recommendation is a candidate headline finding |

**Next:** Phase 8 — Reporting, which assembles the final write-up: the reasoning behind each modeling and business decision across all seven prior phases, plus the 3–4 visualizations that communicate the findings fastest to a non-technical reader.
