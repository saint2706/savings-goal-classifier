# Phase 0 — Framing

**Source:** [README § Phase 0 — Framing](../README.md#phase-0--framing)
**Notebook:** none — Phase 0 is a framing exercise, answered directly in prose (in the README and here), not through data analysis. There's nothing to load or compute yet: the point of this phase is to pin down *what problem we're solving and how we'll know if we solved it* before touching the data, so every later phase has a fixed target to build toward instead of drifting.

This document exists mainly so the phase-by-phase walkthrough series is complete from the start (`phase0.md` → `phase2.md` and onward) — Phases 1 and 2 (`phase1.md`, `phase2.md`) additionally walk through a notebook cell by cell, since those phases involve actual code and analysis.

---

## Research questions & answers

### Q1. What real business decision does this project inform?

**Answer:** Whether a fintech / savings-product marketing team should treat an individual as **on-track** or **at-risk** for their own stated savings goal, and therefore which customers to target with which intervention — a nudge campaign, a round-up savings product, or a budgeting tool aimed at a specific overspend category.

**Why this framing matters:** A model is only useful if its output maps onto an action someone can actually take. "Predict whether this person hits their goal" is a diagnostic curiosity on its own; "flag at-risk customers so the marketing team can target them" is a decision. Naming the decision up front constrains everything downstream — it's why Q3 below lands on classification instead of regression, and it's why Phase 4 (Model Comparison) later has to weigh precision against recall against the cost of a *false negative* (missing someone who's actually at risk) instead of just optimizing a generic accuracy number.

### Q2. What is the precise, one-sentence definition of the target variable?

**Answer:**

```
Goal_Met = 1 if Disposable_Income >= Desired_Savings else 0
```

A binary label that is `1` when an individual's actual leftover income (after all recorded expenses) is enough to cover the monthly savings amount they themselves said they want to save.

**Why this matters:** `Disposable_Income` and `Desired_Savings` are the two columns used to *construct* the label — which means they can never be used as model inputs (see [README § 3](../README.md#3-target-variable)). This one-sentence definition is what Phase 1 later verifies mechanically: `phase1.md` § 5 (Leakage check) shows that this formula reconstructs `Goal_Met` with 100% accuracy directly from data, confirming the definition holds exactly, not just approximately.

### Q3. Is the primary task classification or regression, and why?

**Answer:** **Binary classification.** The business decision from Q1 is a yes/no call — is this person on-track or not, so the pipeline can flag them for a marketing/outreach decision — not a request for a precise savings-shortfall forecast.

**Why not regression:** A regression on `Disposable_Income` or on the savings gap (`Desired_Savings - Disposable_Income`) would be a reasonable *follow-up* analysis once a classifier exists, but it isn't what the stated business decision requires, and the target as defined in Q2 is already binary by construction. Framing this as regression would mean predicting a continuous quantity and then re-thresholding it at inference time — that throws away the information about *why* `>=` was chosen as the threshold in the first place, and it complicates evaluation against the actual yes/no decision the business needs to make. Committing to classification now is also what makes the rest of the pipeline concrete: it's why Phase 3's baseline is a majority-class/logistic-regression accuracy-and-F1 check rather than an RMSE check, and it's why Phase 4 compares classifier families (Random Forest, XGBoost, SVM, …) rather than regressors.

---

## What Phase 0 sets up for later phases

| Decision made here | Where it gets used |
|---|---|
| The business decision is a targeting/flagging action | Phase 4's precision-vs-recall framing; Phase 7's business translation |
| `Goal_Met = 1 if Disposable_Income >= Desired_Savings else 0` | Phase 1's leakage check reconstructs this exact formula from raw data |
| The task is binary classification | Every later phase (baseline, model comparison, explainability) is scoped to classification, not regression |

**Next:** [Phase 1 — Data Understanding](phase1.md), which starts from this target definition and verifies it empirically against the actual dataset.
