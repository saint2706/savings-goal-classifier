# Phase 8 — Reporting

**Source:** [README § Phase 8 — Reporting](../README.md#phase-8--reporting)
**Builds on:** Phases 0–7 (all)
**No notebook.** Phase 8 doesn't run any new analysis — every number and finding it uses was already computed and persisted by an earlier phase (`results/*.csv`, `results/*.png`). Its job is to assemble those findings into the write-up a non-technical stakeholder would actually read, and to make sure the *reasoning* behind each modeling decision — not just its final metric — survives into that write-up. Because there's no code to walk through cell by cell, this document doubles as that final write-up.

---

## Research questions & answers

| #   | Question                                                                                                    | Answer                                                                                                                                                                                                                                                                                                    |
| --- | ------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Does the final write-up explain reasoning (e.g., why leakage columns were excluded, why a given metric was chosen) rather than just reporting numbers? | Yes — see [Why each decision was made](#why-each-decision-was-made-the-reasoning-trail) below, which traces every material modeling decision back to the phase that made it and the specific reason, not just the resulting score.                                                                     |
| 2   | Which 3–4 visualizations communicate the findings fastest to a non-technical reader?                          | Three: `results/persona_clusters.png` (where risk concentrates), `results/shap_summary.png` (what drives it), and `results/model_comparison.png` (why the recommended classifier can be trusted to catch it). See [Visual appendix](#visual-appendix--the-stakeholder-deck) for why these three and not others. |

---

## The final stakeholder write-up

*Written for the audience Phase 0 defined this project around — a fintech/savings-product marketing team deciding which customers to target, with no assumed ML background.*

### Executive summary

Out of 20,000 customers, **112 (0.56%) are not on track to hit their own stated savings goal**. Every one of them lives in a **Tier-1 city**, and the single biggest reason is loan-repayment burden eating an unusually large share of their income — not low income itself. A tuned classifier can flag all 112 with no misses, at the cost of roughly one unnecessary low-cost nudge for every eight real flags. For the vast majority of these customers, the shortfall is recoverable, not structural: 99.1% of them already have more addressable savings potential (mostly in groceries) than they need to close their gap. Recommended action: target Tier-1 customers the model flags with a savings nudge that leads with grocery spending, framed as a small, achievable ask.

### The business question

A savings-product marketing team can't manually review 20,000 customers to decide who needs an outreach nudge. This project builds a classifier that does that triage automatically — flagging customers who look unlikely to hit their own savings goal, so outreach effort goes where it's needed instead of being spread evenly or left to guesswork (Phase 0).

### What we found

**1. Who's at risk.** Every one of the 112 at-risk customers lives in a Tier-1 city — none in Tier-2 or Tier-3. Occupation and age show no comparable pattern; number of dependents is only mildly elevated. Three independent methods agree on this: the classifier's own explanations (Phase 5), an unsupervised clustering that never saw the label (Phase 6), and a direct breakdown of the at-risk group (Phase 7).

**2. What drives it.** The single strongest signal is how much of a customer's income goes to loan repayments — more than twice as influential as the next factor. Education spending (a proxy for having dependents) and living in a Tier-1 city follow. Raw income, age, and number of dependents on their own matter far less than *how* a customer's income is being spent (Phase 5).

**3. What to do about it.**

| Recommendation | Why |
| --- | --- |
| Target outreach by city tier, not occupation or age | 100% of at-risk customers (112/112) live in Tier-1 cities |
| Lead nudge campaigns with grocery-spend reduction | Groceries is the largest source of unrealized savings — ₹18.2M/month across the population, about double the next category |
| Frame the ask as small and achievable, not remedial | For 99.1% of at-risk customers, addressable savings already exceed their shortfall — this is a recoverable gap, not a structural one |
| Deploy the recommended classifier at its tuned threshold for flagging | Catches all known at-risk customers, at the cost of roughly 1-in-8 unnecessary (but low-cost) nudges |
| Don't build a separate persona-based targeting layer | The clustering found personas driven by the same city-tier/dependents signal the classifier already uses — it wouldn't add new targeting power |

(Full evidence for each: [`results/business_recommendations.csv`](../results/business_recommendations.csv), [Phase 7](phase7.md).)

**4. How much to trust this.** Before acting on it, a stakeholder should know: the at-risk group used to validate the model is small (112 customers), so "catches every case" is strong evidence, not a lifetime guarantee; about 1 in 8 flagged customers won't actually be at risk; the model can't detect risk factors that only show up as a *combination* of features, only ones visible on their own; and the dataset may be partly synthetic, so absolute figures should be checked against real customer data before this goes into production (Phase 7).

### Why each decision was made (the reasoning trail)

A number on its own doesn't tell a stakeholder whether to trust it. Each row below is a decision earlier phases made *on purpose*, and why — the same reasoning a reviewer or a skeptical stakeholder would ask for.

| Decision | Reasoning | Phase |
| --- | --- | --- |
| `Disposable_Income`, `Desired_Savings`, `Desired_Savings_Percentage` excluded from model inputs | These columns define the target itself; feeding them to the model would let it reconstruct the label instead of predicting it — perfect-looking accuracy with no real predictive value | 1 |
| Framed as binary classification, not regression | The business decision is a yes/no targeting call, and the target is binary by construction; forecasting a continuous shortfall and re-thresholding it would throw away *why* that threshold matters | 0 |
| Models judged on macro-F1, not accuracy | With 99.4% of customers on-track, a model that never once flags anyone still scores 99.4% accuracy — accuracy can't tell a useful model from a useless one here | 3 |
| Winning model chosen from cross-validated predictions, never from test-set scores until the very end | Picking a "winner" by whoever scores best on the test set lets that one score influence the pick itself, inflating the reported result — cross-validation avoids that by scoring every candidate on data it never trained on | 4 |
| SVM (linear) selected over tree ensembles and XGBoost, despite those having higher raw accuracy | This project's real cost is a missed at-risk customer, not a false alarm; only the linear SVM and logistic regression caught every at-risk customer under cross-validation, and the SVM did so with better precision | 4 |
| Decision threshold tuned using training data only, then tested once | Tuning against the test set would mean the "final" number was itself chosen to look good on that specific data — the threshold was picked blind to the test set, then evaluated on it exactly once | 4 |
| Explanations computed with SHAP's exact method, not an approximation | Because the winning model is linear, an exact per-customer explanation is available at no extra cost — no need to settle for a sampled approximation | 5 |
| Clustering run as a cross-check, not as a second targeting system | An unsupervised method that never sees the label agreeing with the classifier's own explanation is stronger evidence than either alone; it turned out to rely on the same two signals the classifier already uses, so it doesn't add new targeting power | 6 |
| Reliability caveats reported alongside the strong metrics, not left out | A stakeholder acting on a "flagged" list needs to know what a flag does and doesn't guarantee before using it operationally | 7 |

### Visual appendix — the stakeholder deck

Three charts, already produced and saved by earlier phases, in the order they'd be shown to a non-technical audience:

1. **[`results/persona_clusters.png`](../results/persona_clusters.png) — "Where does risk concentrate?"** A bar chart of at-risk rate by customer segment. Shown first because it answers the most concrete question a marketing stakeholder has — *who do I target* — without requiring any explanation of how the model works. (Phase 6)
2. **[`results/shap_summary.png`](../results/shap_summary.png) — "What's driving it?"** A ranked bar chart of which factors matter most to the model's decisions. Shown second, once the audience already trusts the "who," to explain *why* — loan-repayment burden dominates, ahead of city tier and grocery spend. (Phase 5)
3. **[`results/model_comparison.png`](../results/model_comparison.png) — "Can we trust the flags?"** A side-by-side comparison of every model tried, with the recommended one's near-perfect catch rate on at-risk customers highlighted. Shown last, as the reliability case for actually deploying the recommendation. (Phase 4)

Two other candidates were considered and left out: a raw model-accuracy chart (misleading on its own at 99%+ accuracy for every model, including the useless do-nothing baseline — the exact trap Phase 3 identified) and a full SHAP waterfall plot for an individual customer (informative but requires more model literacy than a three-chart, three-minute deck allows). The five recommendations table above stands in for a fourth visual — it's a table rather than a chart because the ranking and evidence pairing are the point, not a distribution.

### Limitations carried into this report

- The dataset's figures may be partly synthetic (Phases 1, 2, 5) — validate absolute numbers against real customer data before deploying operationally.
- `Goal_Met` measures whether a customer meets *their own stated* goal, not an external measure of financial health.
- The spending personas (Phase 6) are a demographic/geographic split already present in `Dependents` and `City_Tier`, not a newly discovered behavioral archetype.
- The recommended model is linear and cannot represent any feature-interaction effect; its perfect recall rests on only 112 at-risk examples.

(Full detail: [README § Limitations](../README.md#9-limitations).)

---

## Where this report fits

This document is the plain-language deliverable Phase 8 asks for; the IEEE-format academic report (rigor, related work, evaluation rubric alignment) is maintained separately at [`project/main.tex`](../project/main.tex), and should cite the same reasoning trail and visual appendix above rather than re-deriving them. There is no Phase 9 — this is the terminal phase of the pipeline described in [README § Methodology](../README.md#5-methodology--pipeline).
