# Phase 7 — Business Translation

**Source:** [README § Phase 7 — Business Translation](../README.md#phase-7--business-translation)
**Notebook:** [`notebooks/07_business_translation.ipynb`](../notebooks/07_business_translation.ipynb)
**Builds on:** Phases 0–6
**Artifacts:** `results/business_recommendations.csv`, `results/business_translation.png`

This is where the project has to say what it would actually recommend. The recommendations that survive contact with the evidence are more modest than a 0.93 ROC-AUC suggests: the model's margin over a plain income rule is small, and for most at-risk households the shortfall is not something a spending nudge can fix.

---

## Research questions & answers

| # | Question | Answer |
| --- | --- | --- |
| 1 | What are the 3–5 most actionable findings, stated as recommendations? | Five, below. The lead finding is uncomfortable: **the model beats "contact the poorest households first" by only 1.5 percentage points** of at-risk capture at a 25% contact budget (36.6% vs 35.0%). |
| 2 | Which expense category carries the most unrealised savings? | **Miscellaneous** (₹36 crore/yr aggregate excess vs same-income peers). But the next two — **Healthcare** and **Education** — are largely non-discretionary, and at-risk households spend **8.8pp *less* of their budget on groceries** than on-track peers, so food is not the lever it is usually assumed to be. |
| 3 | Where does the model fail, and what must a stakeholder be told? | Calibration is excellent (max error 0.036), but accuracy sags to **0.78–0.80 in income deciles 5–7** — exactly the middle where the decision is real. And **32.3% of the at-risk group** report spending more than twice their income, which is a measurement artifact rather than observed distress. |

---

## Notebook walkthrough

### Cell 1 (code) — Out-of-fold scores for every household

Predictions come from `cross_val_predict`, so every household has a score from a model that never saw it (the fold machinery from [Phase 4](phase4.md), used here to score all 41,518 households rather than to compare models). **In-sample** means scoring households the model was trained on — the model grading its own homework, which always flatters. **Why not fit once and predict in-sample:** the targeting analysis below ranks households by score and measures capture at a contact budget. In-sample scores would be optimistically ordered and would overstate the campaign's reach — the exact number a stakeholder would plan against.

### Cell 3 (code) — Does the model beat the obvious rule? (Q1)

The comparison is against the **Phase 3 income rule**, not against random contact. Phase 3 exists so this phase cannot quietly benchmark against the weakest possible alternative.

| Strategy | Capture @10% | @25% | @50% | Precision @25% |
| --- | --- | --- | --- | --- |
| **Model** | 14.7% | **36.6%** | 70.8% | **99.6%** |
| Income rule | 14.4% | 35.0% | 65.8% | 95.4% |
| Random contact | 10.0% | 25.1% | 49.8% | 68.3% |

> **In plain terms — capture at a contact budget.** A real campaign cannot contact everyone, so fix a **contact budget** — say the 25% of households you can afford to reach. Each strategy ranks all households, you take its top 25%, and you count what fraction of all the genuinely at-risk households you caught. That fraction is the **capture**.
>
> Three strategies are compared: the model's ranking, ranking by income alone (poorest first), and contacting at random. Random is the floor and behaves exactly as arithmetic demands — contact 25% at random and you catch about 25% of every group, including the at-risk one.
>
> **Precision @25%** asks the complementary question: of the households you did contact, what fraction were genuinely at risk? Capture is about who you reached; precision is about how little effort you wasted. The model and the income rule are nearly tied on the first and clearly separated on the second, which is why the recommendations lead with precision rather than capture.

**The model's edge over "contact the poorest first" is 1.5 percentage points of capture.** That is the single most important number in this phase, and it is deliberately placed first in the recommendations rather than buried.

**Why the lift over random looks so unimpressive.** At-risk households are **68.1%** of the population, so contacting 25% at random already reaches 25.1% of them. The model reaches 36.6% — a lift of **1.46×**. Lift figures are only impressive when the target class is rare; here it is the majority, and any business case written around "finding a hidden segment" is unsupportable.

> **In plain terms — lift, and the ceiling on it.** **Lift** is how many times better a targeted campaign does than a random one: 36.6% captured versus 25.1%, so 1.46×. Marketing decks routinely quote lifts of 5× or 10×, and against those this looks feeble.
>
> But the comparison is unfair, because **lift has a hard ceiling set by how common the target is**. If at-risk households were 5% of the population, a perfect model contacting the top 25% would catch all of them — a 20× lift. Here they are 68%, so even a flawless model contacting 25% of households could capture at most about 37% of them (there simply are not enough slots), capping lift at roughly 1.5×. **The model is essentially at that ceiling.** It is performing close to perfectly on this measure; the measure just cannot produce an impressive number for a majority target. This is the structural weakness [Phase 0](phase0.md) flagged in advance, arriving exactly as predicted.

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

**Note the direction.** Metro households are the *least* likely to be at risk, not the most — the intuition that urban living strains a budget does not hold once the whole income distribution is in view.

The income-decile table shows why geography is mostly a proxy: at-risk falls from **97.8%** in decile 0 to **16.9%** in decile 9. Phase 5 already found income's SHAP effect is near-identical across area types, so the area gradient is largely an income gradient wearing a geographic label.

### Cells 6–7 (code) — Defining "recoverable" (Q2)

IHDS does not say which spending a household could have avoided, and inventing a discretionary/non-discretionary split would smuggle in an assumption about what people can give up. Instead, **recoverable is defined by peer benchmarking**: for each at-risk household, how much more of its budget goes to each category than the *median on-track household in the same income decile*.

**Why the same income decile:** comparing a household earning ₹40,000 against the population median would just re-measure poverty. Holding income roughly constant makes the comparison one of allocation between households with comparable means.

> **In plain terms — peer benchmarking, and why "recoverable" needs defining at all.** The business question is "which spending could this household cut?" — but the survey never asked that, and there is no column marking spending as essential or optional. Splitting the categories into discretionary and non-discretionary by hand would just be writing our own assumptions into the answer and reading them back out.
>
> **Peer benchmarking** sidesteps the assumption by letting comparable households define the standard. For each at-risk household, find the on-track households in its own income decile, look at how *they* divide their budget, and treat the difference as potentially recoverable. It never claims any particular rupee could have been saved — only that households with similar means manage on a different allocation.
>
> The income-decile restriction is the essential part. Compared against all households, a poor household looks "excessive" on almost nothing and "deficient" on everything, and the exercise would simply rediscover that it is poor. Compared against its own income peers, the question becomes one of allocation — this is the same conditional-versus-unconditional discipline that [Phase 5](phase5.md) showed was the difference between a correct recommendation and a backwards one.

| Category | Median excess (pp) | % above peer | Aggregate excess (₹cr/yr) |
| --- | --- | --- | --- |
| Miscellaneous | −0.21 | 48.6% | **35.8** |
| Healthcare | **+2.53** | 63.5% | 34.2 |
| Education | +1.41 | 64.4% | 19.8 |
| Transport | +1.33 | 62.8% | 17.6 |
| Groceries | **−8.79** | 27.1% | 4.5 |
| Utilities | −2.77 | 29.8% | 3.4 |

> **In plain terms — the three columns, which say different things.**
> - **Median excess (pp)** — for the typical at-risk household, how many *percentage points* more of its budget goes to this category than its income peers'. A negative number means the typical at-risk household spends *less* here, not more.
> - **% above peer** — what fraction of at-risk households exceed their peers at all. This catches breadth: is this a widespread tendency or a minority's behaviour?
> - **Aggregate excess (₹cr/yr)** — the total dilemma across every at-risk household, in **crore** (1 crore = 10 million rupees, standard in Indian financial reporting).
>
> Reading them together is what prevents the wrong conclusion. `Miscellaneous` tops the aggregate column but has a *negative* median — meaning the total is driven by a minority spending heavily, not by a general tendency, so a campaign aimed at everyone would misfire. `Healthcare` is the reverse: a genuine, broad excess (+2.5pp, affecting 63.5%) that nobody can be asked to cut. Any single column on its own supports a recommendation the other two contradict.

**Two findings that overturn the obvious campaign.**

First, **groceries is not the lever.** At-risk households spend **8.8 percentage points less** of their budget on food than on-track households at the same income, and only 27.1% exceed their peers. A nudge campaign led by grocery-spend reduction — the intuitive choice, since food is the largest single category — would be aimed at the wrong households. This is the same relationship Phase 5 found conditionally: a high food share marks a household *without* large non-food outlays, and those households save more.

Second, **the largest genuine excesses are in categories nobody can be asked to cut.** Healthcare has the biggest median excess (+2.5pp, 63.5% of at-risk households above their peers) and Education is third. Both are close to non-negotiable, and a nudge campaign built on them would be both ineffective and inappropriate. `Miscellaneous` tops the aggregate table, but its median excess is *negative* (−0.21pp) — the aggregate is driven by a minority with very large miscellaneous spend, not by a broad tendency.

### Cell 7 (code) — Is the gap even closable?

| | |
| --- | --- |
| Median annual gap to the 20% benchmark | **₹37,114** |
| Median recoverable (peer excess, all categories) | **₹22,939** |
| At-risk households whose peer-excess would close the gap | **28.5%** |

> **In plain terms — what these three numbers are compared against each other.** The **gap** is how many rupees short of the 20% benchmark a household is — ₹37,114 a year for the typical at-risk household. The **recoverable** amount is the total peer-excess from the table above: everything that household spends beyond what its income peers spend, added across all eleven categories.
>
> The third row puts them side by side, household by household, and asks whether the second could cover the first. Note how demanding a test this already is — it assumes the household matches its peers' spending in **every category at once**, including healthcare and education, which is not something anyone would actually recommend. So 28.5% is a generous upper bound on how many households a behaviour-change product could possibly help. **For the other 71.5%, no rearrangement of spending reaches the benchmark**, because the shortfall is larger than the entire amount they spend above their peers.

**Only 28.5% of at-risk households could reach the benchmark even by matching their income peers' spending in every single category.** For the other 71.5%, the shortfall is not a spending problem that a budgeting nudge can fix.

**This is the finding with the largest product implication in the whole project.** A budgeting tool, a round-up savings feature, a spending nudge — all of them assume the shortfall is behavioural. For roughly seven in ten at-risk households here it is not: **the answer is more income, not less spending.** Any product built on the opposite assumption will fail for the majority of the people it targets.

### Cells 9–10 (code) — Reliability (Q3)

**Calibration is genuinely good.** Across ten probability bins the maximum absolute error is **0.036**, and most bins are within 0.01. A predicted 70% really does mean about 70%, so the scores can be used for ranking *and* for expected-value arithmetic — which is not something most classifiers earn without an explicit calibration step.

> **In plain terms — calibration, and why it is a separate virtue.** A model outputs numbers between 0 and 1, but nothing forces those numbers to be *honest probabilities*. A model is **calibrated** when they are: gather every household it scored around 0.70, and about 70% of them should genuinely be at risk.
>
> The test is exactly that. Sort predictions into ten **bins** (0–10%, 10–20%, and so on), and in each bin compare the average predicted probability with the actual proportion at risk. The largest gap anywhere is 0.036, so the promised and observed rates never diverge by more than about 4 points.
>
> This is a different property from being accurate, and models routinely have one without the other. A model can rank households perfectly while every probability it emits is far too high — order preserved, numbers meaningless. Calibration is what allows **expected-value arithmetic**: "we will contact 10,000 households at an average 72% risk, so roughly 7,200 genuinely need this" is only sound if 72% means 72%. Many classifiers need an extra corrective step to earn this; gradient boosting on a large, non-extreme dataset often arrives calibrated, as it has here.

**Accuracy by income decile shows where it degrades:**

| Decile | 0 | 3 | 5 | 6 | 7 | 9 |
| --- | --- | --- | --- | --- | --- | --- |
| Accuracy | 0.979 | 0.870 | 0.799 | 0.788 | 0.779 | 0.887 |

**The model is weakest exactly where the decision matters.** At the extremes it is nearly perfect, but those households need no model — the poorest are almost all at risk (97.8%) and the richest mostly are not (16.9%). In deciles 5–7, where at-risk rates run 70% → 49% and a targeting decision is genuinely contested, accuracy falls to 0.78–0.80. Reporting overall accuracy of 0.86 without this breakdown would overstate usefulness at the only point where the model changes an action.

**The measurement caveat, quantified.** 9,137 households (22.0%) report spending more than twice their reported income. **All of them are classified at-risk, and they alone are 32.3% of the entire at-risk group.** Roughly a third of the "at-risk" population is therefore an artifact of income under-reporting rather than observed distress. That does not make the ranking useless — under-reporting correlates with informal, irregular income, which is itself a risk marker — but it forbids treating the at-risk count as a population estimate.

**Threshold sensitivity:** moving the benchmark from 20% to 10% or 30% shifts the at-risk share from 68.1% to 61.7% or 74.7%, with ~93.5% agreement in both directions. Findings quoted here hold across that range; the *level* does not.

> **In plain terms — what this check is for.** The 20% benchmark was a convention chosen at the outset ([Phase 0](phase0.md)), so every result carries an obvious challenge: would it survive a different, equally defensible choice? Redefining the target at 10% and at 30% and re-running answers it.
>
> **~93.5% agreement** means that moving the benchmark changes which side of the line a household falls on for only about 1 household in 15; the other 14 keep the same label. So the *ordering* of households — who is more at risk than whom — is barely disturbed, and the recommendations built on that ordering hold. What does move is the **level**: the headline "68.1% of households are at risk" becomes 61.7% or 74.7% depending purely on where the line was drawn. This is the concrete evidence for the rule stated in Phase 0 and repeated throughout — **relative priority is defensible, absolute prevalence is not.**

---

## The five recommendations

1. **Prioritise by income first; use the model to re-rank within income bands.** The model adds +0.095 macro-F1 over the single income rule, but only **+1.5 points** of at-risk capture at a 25% budget. *Caveat: do not present the model as discovering who is at risk.*

2. **Size the campaign as prioritisation, not needle-finding.** At-risk households are 68.1% of the population; the model's lift over random is **1.46×**. Its real value is precision — 99.6% vs 68.3%. *Caveat: a 1.5× lift will not support a "find the hidden segment" business case.*

3. **Treat the shortfall as structural for most at-risk households.** Only **28.5%** could close the gap by matching peer spending in every category. *Caveat: a budgeting or nudge product cannot help the other 71.5%; for them the answer is more income, not less spending.*

4. **Where a behavioural lever exists it is Miscellaneous — and it is NOT groceries.** At-risk households spend 8.8pp *less* on food than their income peers. *Caveat: never build a nudge on healthcare or education, the two largest genuine excesses.*

5. **Report relative priority only.** 32.3% of the at-risk group report spending more than twice their income. *Caveat: "X% of Indian households save inadequately" is unsupported by this work.*

---
