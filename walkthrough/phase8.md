# Phase 8 — Reporting

**Source:** [README § Phase 8 — Reporting](../README.md#phase-8--reporting)
**Builds on:** Phases 0–7
**No notebook.** Phase 8 runs no new analysis — every figure it cites was computed and persisted by an earlier phase. Its job is to assemble them into something a non-technical stakeholder would read, and to make sure the *reasoning* survives the compression.

---

## Research questions & answers

| # | Question | Answer |
| --- | --- | --- |
| 1 | Does the write-up explain reasoning rather than just reporting numbers? | Yes — see [Why each decision was made](#why-each-decision-was-made) below, which traces every material choice to the phase that made it and the evidence that forced it, including four cases where the evidence overturned the approach we started with. |
| 2 | Which 3–4 visualisations communicate the findings fastest? | Four: `phase1_eda.png` (what the data is really like), `shap_summary.png` (what drives the outcome), `business_translation.png` (what the model is worth), and `personas.png` (who the segments are). Rationale in [Visual appendix](#visual-appendix). |

---

## The stakeholder write-up

*Written for the audience Phase 0 defined: a savings-product or financial-inclusion team deciding which households to prioritise, with no assumed ML background.*

### Executive summary

Across **41,518 Indian households** surveyed in 2011–12, **68% retain less than 20% of their annual income** after recorded consumption. A model can rank households by that risk with good accuracy (ROC-AUC 0.93, macro-F1 0.84) and unusually reliable probabilities.

But three findings should temper any campaign built on it:

1. **Income does most of the work.** A single threshold — earning more than ₹121,685/year — already recovers most of the model's performance. The full model adds real but modest value on top.
2. **The model barely beats "contact the poorest first."** At a 25% contact budget it reaches 36.6% of at-risk households against 35.0% for a simple income rule. Its genuine advantage is precision: 99.6% of those contacted are truly at risk, against 68.3% at random.
3. **For most at-risk households the shortfall is structural, not behavioural.** Only **28.5%** could reach the benchmark even by matching the spending of on-track households at the same income. A budgeting nudge cannot fix the other 71.5%.

**Recommended action:** use the model to prioritise outreach *within* income bands rather than as a discovery tool; size the campaign as prioritisation rather than needle-finding; and pair any savings product with income-side support, because for seven in ten at-risk households more income — not less spending — is what closes the gap.

### The business question

A savings-product or financial-inclusion team cannot review 41,518 households by hand to decide who needs support. This project builds a classifier that triages them, flagging households unlikely to retain an adequate share of income so outreach goes where it is most needed (Phase 0).

**One framing caveat that matters.** The original version of this project asked whether someone was on track for *their own stated savings goal*. No real household survey collects that, so the target here is **normative**: a household is "on track" if it retains at least 20% of annual income. That is a different question — closer to measuring savings *capacity* than goal alignment — and no claim in this report should be read as being about goals people set for themselves.

### What we found

**1. Who is at risk.** At-risk rates fall steadily with income, from **97.8%** in the poorest decile to **16.9%** in the richest. Geography follows the same gradient: least-developed villages 73.3%, metro urban 57.9%. Occupation matters too — households with no regular worker (75.0%) or in agricultural labour (74.9%) versus salaried households (55.9%).

**Geography here is largely income wearing a geographic label.** Phase 5 found income's effect is near-identical across area types, so an area-based campaign is an income-based campaign with extra steps — worth knowing before designing outreach around location.

**2. What drives it.** Income is the dominant factor: 38.4% of the model's total explanation, more than four times the next feature. Spending mix matters too (26.7%), but second.

Within spending, one finding is counter-intuitive and important: **households spending a larger share of their budget on food are *more* likely to be on track**, once income is accounted for. That is the opposite of the obvious reading. A food-dominated budget signals the *absence* of large irregular outlays — health shocks, durables, school fees — and those are what push consumption above income. Consistent with this, at-risk households spend **8.8 percentage points less** of their budget on groceries than on-track households at the same income.

**3. What to do about it.**

| Recommendation | Why |
| --- | --- |
| Prioritise by income first; use the model to re-rank within income bands | The model adds only +1.5 points of at-risk capture over a simple income rule at a 25% budget |
| Size the campaign as prioritisation, not needle-finding | At-risk households are 68% of the population; lift over random is 1.46× |
| Treat the shortfall as structural for most households | Only 28.5% could close the gap by matching peer spending in every category |
| If a behavioural lever exists it is Miscellaneous — not groceries | Groceries runs the wrong way; healthcare and education are the largest excesses but are not optional |
| Report relative priority only, never a prevalence figure | 32.3% of the at-risk group report spending more than twice their income |

### What a stakeholder must be told before acting

- **The model is weakest exactly where the decision is hardest.** Accuracy is 0.98 in the poorest decile and 0.89 in the richest — but **0.78–0.80 in deciles 5–7**, where at-risk rates run 70% down to 49% and targeting is genuinely contested. The headline 0.86 overstates usefulness at the point of decision.
- **About a third of the "at-risk" group is a measurement artifact.** 22.0% of households report spending more than twice their income; all are classified at-risk and they are 32.3% of that group. Indian household surveys under-report income relative to item-by-item consumption. The *ranking* remains useful; the *count* is not a population estimate.
- **The 20% benchmark is a convention we chose**, not a measurement. Moving it to 10% or 30% shifts the at-risk share from 68.1% to 61.7% or 74.7%. Findings here hold across that range; levels do not.
- **The data is from 2011–12.** Sound for methodology, not current for market sizing.
- **Probabilities can be trusted.** Calibration error is at most 0.036 across the range — a predicted 70% really means about 70%, so scores support expected-value arithmetic, not just ranking.

---

## Why each decision was made

Every material decision, the phase that made it, and the evidence behind it.

| Decision | Phase | Reasoning |
| --- | --- | --- |
| Normative 20% target | Phase 0 | No real survey collects a self-declared savings goal. Threshold made configurable, and sensitivity published so no reader takes it on trust. |
| Composition shares, not expense-to-income ratios | Phases 1–2 | Ratios reconstruct the target with **99.75%** agreement — textbook leakage. Shares sum to exactly 1 and cannot recover consumption-vs-income. |
| Macro-F1, not accuracy | Phase 3 | The majority baseline scores 0.68 accuracy while never identifying a single on-track household. |
| Benchmark against a single income rule | Phase 3 | Comparing only to the majority baseline flatters the model by 0.43 macro-F1; against the income rule the honest gain is 0.095. |
| XGBoost over logistic regression | Phase 4 | Macro-F1 0.837 vs 0.819. Phase 5 explains why: **41.9% of the model's behaviour is interactions**, which a linear model cannot represent. |
| Optimise for recall on the at-risk class | Phase 4 | A missed at-risk household is a silent failure; an unnecessary nudge is cheap. Reaching 95% recall costs only ~8 points of precision. |
| SHAP on trees, not coefficients | Phases 2, 5 | The 11 shares sum to 1, so they are exactly singular (VIF = ∞) and no coefficient is identified. |
| Peer benchmarking for "recoverable" | Phase 7 | The survey does not say which spending was avoidable; comparing within income deciles avoids having to assume it. |

### Four times the evidence overturned our approach

Recording these matters more than recording the successes, because each was a plausible choice that measurement rejected.

1. **The CLR transform was proposed, implemented, and rejected** (Phase 2). Compositional data theory says to log-ratio transform simplex data. It scored 0.9117 against 0.9212 for raw shares, and its components are singular by construction (VIF 3×10⁵). The theory was right about the problem and wrong about which transform.
2. **Expense-to-income ratios generalise *worst*** (Phase 2). Dividing spending by income is the intuitive way to make households comparable across income levels. Tested on a neutral target, it came last (0.589 against 0.625 for raw rupees) — income is the under-reported side of this survey, and so the worst available divisor.
3. **A high grocery share predicts being *on track*** (Phase 5). We captioned the dependence plot with the opposite, from the obvious Engel's-law reading. The plot contradicted it. Reading this off Phase 1's unconditional correlation would have produced a backwards recommendation.
4. **The spending personas are not an income split** (Phase 6). Phase 5 found income carries 38.4% of attribution against spending's 26.7%, which suggested the personas would largely re-encode income. The adjusted Rand index against income tertiles is 0.006. Income was *suppressing* the persona effect — controlling for it nearly doubled the spread (0.167 vs 0.092).

A fifth, methodological: Phase 6's notebook initially **hardcoded the worst-scoring representation** instead of selecting from the sweep it had just computed. Caught by reading the output rather than skimming it.

---

## Visual appendix

Four charts, chosen because each answers a question a stakeholder will actually ask.

**1. `results/phase1_eda.png` — "What is this data really like?"**
Four panels: income skew, the savings-rate distribution with the 20% line marked, zero-inflation by category, and goal attainment by area type. It earns first place because it makes the two facts that constrain every later claim visible at once — the median household has a **negative** savings rate, and rent is absent for 90% of households.

**2. `results/shap_summary.png` — "What drives the outcome?"**
The SHAP summary. `Log_Income` dominates by 4.5×, which is the single most important thing for a stakeholder to internalise before believing any spending-behaviour story. Paired with `shap_dependence.png` when the counter-intuitive grocery finding needs explaining.

**3. `results/business_translation.png` — "Is the model worth deploying?"**
Three panels: the capture curve (model vs income rule vs random — the model's line sits just above the income rule's and both are near the diagonal, which is the honest picture), aggregate excess by category, and the calibration curve. This is the chart to show anyone approving budget, because it does not oversell.

**4. `results/personas.png` — "Who are the segments?"**
Spending signature, goal attainment by persona, and income distribution. Included with the caveat that the personas are defined by which categories are *absent*, so they are descriptive rather than targetable.

**Deliberately excluded:** `model_comparison.png` and `cluster_selection.png`. Both are model-selection diagnostics that answer questions a technical reviewer asks and a stakeholder does not. They belong in Phases 4 and 6, not in a summary deck.

---

## Honest limitations

- **Income under-reporting** puts 55.9% of households in an implausible consumption-exceeds-income position and makes ~32% of the at-risk group a measurement artifact. Relative rankings survive; absolute prevalence does not.
- **2011–12 vintage.** IHDS-3 fieldwork is complete but public microdata was not released as of this work.
- **Household grain.** No per-individual claims are supported.
- **Survey weights are carried but not applied.** `WT` is in the dataset; model fitting is unweighted. Any nationally-framed figure must apply it.
- **Debt is a stock, not a flow.** IHDS records outstanding debt and interest rates but no repayment instalment, so `Debt_To_Income_W` stands in for a cash-flow burden it cannot directly measure.
- **The at-risk class is the majority (68%)**, so lift-based business cases are structurally weak here regardless of model quality.
- **Personas are partly a zero-replacement artifact** (adjusted Rand index 0.858 against the raw zero-pattern), not discovered behavioural archetypes.
