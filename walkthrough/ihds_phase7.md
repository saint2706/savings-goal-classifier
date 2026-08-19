# Phase 7 (IHDS-II) — Business Translation

**Source:** [README § Phase 7 — Business Translation](../README.md#phase-7--business-translation)
**Notebook:** [`notebooks/ihds_07_business_translation.ipynb`](../notebooks/ihds_07_business_translation.ipynb)
**Builds on:** Phases 0–6 (IHDS-II)
**Artifacts:** `results/ihds_business_recommendations.csv`, `results/ihds_business_translation.png`
**Replaces:** [`phase7.md`](phase7.md)

This is where the project has to say what it would actually recommend. On the synthetic data that was easy and dramatic: all 112 at-risk individuals lived in Tier-1 cities, groceries held ₹18.2M/month of unrealised savings, and 99.1% of shortfalls were recoverable. **Every one of those findings reverses or dissolves on real data**, and the recommendations that survive are more modest and more honest.

---

## Research questions & answers

| # | Question | Answer |
| --- | --- | --- |
| 1 | What are the 3–5 most actionable findings, stated as recommendations? | Five, below. The lead finding is uncomfortable: **the model beats "contact the poorest households first" by only 1.5 percentage points** of at-risk capture at a 25% contact budget (36.6% vs 35.0%). |
| 2 | Which expense category carries the most unrealised savings? | **Miscellaneous** (₹36 crore/yr aggregate excess vs same-income peers). But the next two — **Healthcare** and **Education** — are largely non-discretionary, and at-risk households spend **8.8pp *less* of their budget on groceries** than on-track peers. The synthetic project's grocery-led recommendation is reversed. |
| 3 | Where does the model fail, and what must a stakeholder be told? | Calibration is excellent (max error 0.036), but accuracy sags to **0.78–0.80 in income deciles 5–7** — exactly the middle where the decision is real. And **32.3% of the at-risk group** report spending more than twice their income, which is a measurement artifact rather than observed distress. |

---

## Notebook walkthrough

### Cell 1 (code) — Out-of-fold scores for every household

Predictions come from `cross_val_predict`, so every household has a score from a model that never saw it. **Why not fit once and predict in-sample:** the targeting analysis below ranks households by score and measures capture at a contact budget. In-sample scores would be optimistically ordered and would overstate the campaign's reach — the exact number a stakeholder would plan against.

### Cell 3 (code) — Does the model beat the obvious rule? (Q1)

The comparison is against the **Phase 3 income rule**, not against random contact. Phase 3 exists so this phase cannot quietly benchmark against the weakest possible alternative.

| Strategy | Capture @10% | @25% | @50% | Precision @25% |
| --- | --- | --- | --- | --- |
| **Model** | 14.7% | **36.6%** | 70.8% | **99.6%** |
| Income rule | 14.4% | 35.0% | 65.8% | 95.4% |
| Random contact | 10.0% | 25.1% | 49.8% | 68.3% |

**The model's edge over "contact the poorest first" is 1.5 percentage points of capture.** That is the single most important number in this phase, and it is deliberately placed first in the recommendations rather than buried.

**Why the lift over random looks so unimpressive.** At-risk households are **68.1%** of the population, so contacting 25% at random already reaches 25.1% of them. The model reaches 36.6% — a lift of **1.46×**. Lift figures are only impressive when the target class is rare; here it is the majority, and any business case written around "finding a hidden segment" is unsupportable.

**Where the model does add real value: precision.** 99.6% of the top-quartile contacts are genuinely at risk, against 68.3% at random. If the cost is per-contact and the campaign has a fixed budget, that is a meaningful reduction in wasted outreach — it is just a different argument from the one a lift curve usually makes.

### Cell 4 (code) — Where the at-risk households are

| Area type | At-risk rate | Share of all at-risk |
| --- | --- | --- |
| Less developed village | 73.3% | 37.4% |
| Developed village | 69.1% | 30.9% |
| Other urban | 63.0% | 25.4% |
| Metro urban | 57.9% | 6.3% |

| Occupation | At-risk rate |
| --- | --- |
| No regular worker | 75.0% |
| Agricultural labour | 74.9% |
| Farm | 74.3% |
| Business | 71.3% |
| Non-ag labour | 68.3% |
| Salaried | 55.9% |

**Both columns matter and they say different things.** Less-developed villages have the *highest rate* and also the largest *share* of the at-risk population — so they are the priority on either reading. But metro urban has a 57.9% at-risk rate, which is high in absolute terms even though metro contributes only 6.3% of all at-risk households; a campaign sized by rate alone would misallocate.

**This is the direct reversal of the synthetic project's headline.** There, 100% of at-risk individuals lived in Tier-1 cities and the flagship recommendation was to target them. Here metro households are the *least* likely to be at risk. Any slide deck carrying the Tier-1 recommendation is now wrong.

The income-decile table shows why geography is mostly a proxy: at-risk falls from **97.8%** in decile 0 to **16.9%** in decile 9. Phase 5 already found income's SHAP effect is near-identical across area types, so the area gradient is largely an income gradient wearing a geographic label.

### Cells 6–7 (code) — What "recoverable" means without a `Potential_Savings` column (Q2)

The synthetic dataset shipped `Potential_Savings_*` columns that simply asserted how much each household could save. IHDS has nothing equivalent, and inventing a discretionary/non-discretionary split would smuggle in an assumption. Instead, **recoverable is defined by peer benchmarking**: for each at-risk household, how much more of its budget goes to each category than the *median on-track household in the same income decile*.

**Why the same income decile:** comparing a household earning ₹40,000 against the population median would just re-measure poverty. Holding income roughly constant makes the comparison one of allocation between households with comparable means.

| Category | Median excess (pp) | % above peer | Aggregate excess (₹cr/yr) |
| --- | --- | --- | --- |
| Miscellaneous | −0.21 | 48.6% | **35.8** |
| Healthcare | **+2.53** | 63.5% | 34.2 |
| Education | +1.41 | 64.4% | 19.8 |
| Transport | +1.33 | 62.8% | 17.6 |
| Groceries | **−8.79** | 27.1% | 4.5 |
| Utilities | −2.77 | 29.8% | 3.4 |

**Two findings that reverse the synthetic recommendation.**

First, **groceries is not the lever.** At-risk households spend **8.8 percentage points less** of their budget on food than on-track households at the same income, and only 27.1% exceed their peers. The synthetic project's flagship advice — lead nudge campaigns with grocery-spend reduction — is exactly backwards here. This is the same relationship Phase 5 found conditionally: a high food share marks a household *without* large non-food outlays, and those households save more.

Second, **the largest genuine excesses are in categories nobody can be asked to cut.** Healthcare has the biggest median excess (+2.5pp, 63.5% of at-risk households above their peers) and Education is third. Both are close to non-negotiable, and a nudge campaign built on them would be both ineffective and inappropriate. `Miscellaneous` tops the aggregate table, but its median excess is *negative* (−0.21pp) — the aggregate is driven by a minority with very large miscellaneous spend, not by a broad tendency.

### Cell 7 (code) — Is the gap even closable?

| | |
| --- | --- |
| Median annual gap to the 20% benchmark | **₹37,114** |
| Median recoverable (peer excess, all categories) | **₹22,939** |
| At-risk households whose peer-excess would close the gap | **28.5%** |

**Only 28.5% of at-risk households could reach the benchmark even by matching their income peers' spending in every single category.** For the other 71.5%, the shortfall is not a spending problem that a budgeting nudge can fix.

**The synthetic project reported 99.1%.** That figure came from the generated `Potential_Savings_*` columns — the generator had written recoverability into the data. Measured against real peers, the answer nearly inverts. This is the single largest substantive difference between the two tracks, and it changes the product implication completely: for most at-risk households in this data, **the answer is more income, not less spending.**

### Cells 9–10 (code) — Reliability (Q3)

**Calibration is genuinely good.** Across ten probability bins the maximum absolute error is **0.036**, and most bins are within 0.01. A predicted 70% really does mean about 70%, so the scores can be used for ranking *and* for expected-value arithmetic — which is not something most classifiers earn without an explicit calibration step.

**Accuracy by income decile shows where it degrades:**

| Decile | 0 | 3 | 5 | 6 | 7 | 9 |
| --- | --- | --- | --- | --- | --- | --- |
| Accuracy | 0.979 | 0.870 | 0.799 | 0.788 | 0.779 | 0.887 |

**The model is weakest exactly where the decision matters.** At the extremes it is nearly perfect, but those households need no model — the poorest are almost all at risk (97.8%) and the richest mostly are not (16.9%). In deciles 5–7, where at-risk rates run 70% → 49% and a targeting decision is genuinely contested, accuracy falls to 0.78–0.80. Reporting overall accuracy of 0.86 without this breakdown would overstate usefulness at the only point where the model changes an action.

**The measurement caveat, quantified.** 9,137 households (22.0%) report spending more than twice their reported income. **All of them are classified at-risk, and they alone are 32.3% of the entire at-risk group.** Roughly a third of the "at-risk" population is therefore an artifact of income under-reporting rather than observed distress. That does not make the ranking useless — under-reporting correlates with informal, irregular income, which is itself a risk marker — but it forbids treating the at-risk count as a population estimate.

**Threshold sensitivity:** moving the benchmark from 20% to 10% or 30% shifts the at-risk share from 68.1% to 61.7% or 74.7%, with ~93.5% agreement in both directions. Findings quoted here hold across that range; the *level* does not.

---

## The five recommendations

1. **Prioritise by income first; use the model to re-rank within income bands.** The model adds +0.095 macro-F1 over the single income rule, but only **+1.5 points** of at-risk capture at a 25% budget. *Caveat: do not present the model as discovering who is at risk.*

2. **Size the campaign as prioritisation, not needle-finding.** At-risk households are 68.1% of the population; the model's lift over random is **1.46×**. Its real value is precision — 99.6% vs 68.3%. *Caveat: a 1.5× lift will not support a "find the hidden segment" business case.*

3. **Treat the shortfall as structural for most at-risk households.** Only **28.5%** could close the gap by matching peer spending in every category. *Caveat: this reverses the synthetic project's 99.1%; for most households the answer is more income, not less spending.*

4. **Where a behavioural lever exists it is Miscellaneous — and it is NOT groceries.** At-risk households spend 8.8pp *less* on food than their income peers. *Caveat: never build a nudge on healthcare or education, the two largest genuine excesses.*

5. **Report relative priority only.** 32.3% of the at-risk group report spending more than twice their income. *Caveat: "X% of Indian households save inadequately" is unsupported by this work.*

---

## What changed from the synthetic track

| Finding | Synthetic | IHDS-II |
| --- | --- | --- |
| Who to target | 100% of at-risk in Tier-1 cities | Metro households are the **least** at risk (57.9% vs 73.3% in villages) |
| Top savings lever | Groceries, ₹18.2M/month, ~2× next category | Groceries is **negative** (−8.8pp vs peers); largest excesses are non-discretionary |
| Is the gap closable? | **99.1%** of at-risk | **28.5%** |
| Model's targeting value | Perfect minority recall on 112 cases | +1.5pp capture over an income rule |

Every headline reverses. The synthetic findings were artifacts of a generator that set rent by tier, made spending a fixed share of income, and wrote recoverability directly into a column.
