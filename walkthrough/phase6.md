# Phase 6 — Unsupervised Extension

**Source:** [README § Phase 6 — Unsupervised Extension](../README.md#phase-6--unsupervised-extension)
**Notebook:** [`notebooks/06_clustering_personas.ipynb`](../notebooks/06_clustering_personas.ipynb)
**Builds on:** [Phase 2](phase2.md), [Phase 5](phase5.md)
**Artifacts:** `results/persona_profiles.csv`, `results/personas.png`, `results/cluster_selection.png`

Phase 2 rejected the CLR transform for classification but left the log-ratio question open for clustering, which is the step that genuinely needs a metric on the simplex. This phase builds an **ILR** basis (full-rank, unlike CLR), uses it, and then discovers that the clean-looking clusters it produces are keyed on something other than what "spending persona" implies.

Two predictions were made going in. **One was confirmed, one was falsified**, and both are reported as such.

> **In plain terms — supervised vs unsupervised.** Everything up to now has been **supervised**: the data came with an answer key (`Goal_Met`), and the model learned to reproduce it. This phase is **unsupervised** — the answer key is hidden and the algorithm is asked simply to group similar households together. Nothing tells it what to look for.
>
> That freedom is the appeal and the danger. Unsupervised methods **always** return groups, whether or not real groups exist, and the groups they return are whatever the chosen notion of "similar" happens to reward. So the work is not in running the algorithm, it is in checking what the resulting groups are actually keyed on — which is the entire drama of this phase.
>
> A **metric** is that notion of similarity: the rule for measuring how far apart two households are. As [Phase 1](phase1.md) established, ordinary straight-line distance is the wrong rule for budget shares, and picking the right one is what the first half of this phase is about.

---

## Research questions & answers

| # | Question | Answer |
| --- | --- | --- |
| 1 | What spending personas emerge from clustering on expense-category proportions? | Three, and they are defined by **which core categories are absent**, not by how present categories are allocated: **Persona 0** (11.7%) records no transport spend; **Persona 2** (18.6%) records no healthcare spend; **Persona 1** (69.7%) spends on everything. The adjusted Rand index between the personas and the raw zero-pattern of the core parts is **0.858**. |
| 2 | How many clusters are statistically justified? | **k = 3 on the 6-part ILR**, silhouette **0.4115** — the only clustering in this project ever to clear the ~0.25 "reasonable structure" threshold. But that score is largely **manufactured by the zero-replacement step**, so it overstates how much genuine structure exists. Raw shares peak at 0.2516 (k=8); ILR plus participation indicators is worst at 0.2054. |
| 3 | Do the personas correlate meaningfully with `Goal_Met`? | **Yes, and more than the headline number suggests.** Raw association is weak (χ² = 238.8, p ≈ 1.4×10⁻⁵², **Cramér's V = 0.076**), but income *suppresses* it: the mean spread in goal attainment across personas **within** an income decile is **0.167**, versus **0.092** unconditionally — **181% of the raw effect survives the control**. At income decile 7, Persona 0 attains 78.9% versus Persona 1's 46.6%. |

---

## Notebook walkthrough

### Cell 1 (code) — Load and drop the undefined rows

The 2 households Phase 2 left with undefined CLR (they spend nothing on any core category) are dropped, leaving 41,516. Reported rather than silently filtered, because a clustering that quietly discards rows is a clustering of a different population than the one described.

### Cell 3 (code) — Building and verifying the ILR basis

```python
def helmert_basis(D):
    V = np.zeros((D, D - 1))
    for i in range(D - 1):
        V[: i + 1, i] = 1.0 / (i + 1)
        V[i + 1, i] = -1.0
        V[:, i] *= np.sqrt((i + 1) / (i + 2))
    return V
```

ILR coordinates are `clr(x) @ V`, where `V` is a D×(D−1) orthonormal contrast matrix. Because it drops one dimension, ILR is **full-rank** — which is precisely the defect that disqualified CLR in Phase 2 (CLR components sum to zero, so their covariance is singular).

> **In plain terms — what that code builds.** Recall from [Phase 2](phase2.md) that the CLR values always sum to zero, so six columns carry only five columns' worth of independent information — which is what made them singular. The ILR fixes this by rewriting the same information in five columns instead of six, losing nothing.
>
> The rewriting is done by multiplying by **V**, a fixed 6×5 table of numbers. **Orthonormal** describes what makes V trustworthy: its columns are mutually at right angles and each has length exactly 1, so the multiplication rotates the data without stretching, squashing or skewing it. Distances are preserved; only the coordinate axes change — the same view, described from a different angle. The **Helmert** construction is one standard recipe for such a table, chosen because each of its columns has a readable meaning: the first contrasts part 1 against part 2, the second contrasts those two against part 3, and so on.
>
> **D** is the number of parts (6 here), so **D−1** is 5 — the "drops one dimension" that makes the result full-rank.

**Three properties verified rather than assumed:**

| Check | Result |
| --- | --- |
| `V'V = I` (orthonormal) | max error 2.22×10⁻¹⁶ |
| Columns sum to zero (orthogonal to the constant) | exactly 0 |
| Aitchison distance = Euclidean distance in ILR space | max error 1.78×10⁻¹⁵ |

> **In plain terms — the three checks.** Each converts a claimed property into a measured one, and the tiny error figures are floating-point dust rather than real disagreement.
> - **`V'V = I`** is the algebraic test for orthonormality described above: multiply the table by its own transpose and you should get the identity matrix (1s on the diagonal, 0s elsewhere), which is the signature of a rotation that distorts nothing.
> - **Columns sum to zero** confirms the new coordinates ignore the overall level and describe only the budget's shape — the property the whole log-ratio apparatus exists to provide.
> - **Aitchison distance = Euclidean distance in ILR space.** The **Aitchison distance** is the correct way to measure how different two budget compositions are, respecting the simplex geometry from [Phase 1](phase1.md). It is also awkward to compute inside a clustering algorithm. This check confirms that once the data is in ILR coordinates, plain everyday straight-line distance gives exactly the same answer — so KMeans, which only knows how to do straight-line distance, is unknowingly doing the right thing.

The third is the one that matters for this phase: it is the formal statement that **KMeans in ILR space is doing legitimate geometry on the simplex**, which it is not doing on raw shares. Checking it costs three lines and converts an assumption into a fact.

### Cell 5 (code) — Choosing the representation and k (Q2)

| k | ILR (6 core) | ILR + indicators | Raw shares (11) |
| --- | --- | --- | --- |
| 2 | 0.3705 | 0.1890 | 0.1988 |
| **3** | **0.4115** | 0.2054 | 0.1756 |
| 4 | 0.2829 | 0.1907 | 0.1999 |
| 6 | 0.2358 | 0.1857 | 0.2321 |
| 8 | 0.2385 | 0.2026 | 0.2516 |

> **In plain terms — k, and the silhouette score.** **KMeans** requires you to state up front how many groups you want — that number is **k**. It will happily produce 2 or 8 whether or not the data contains 2 or 8 real groups, so k has to be chosen by evidence.
>
> The **silhouette score** is the usual evidence. For each household it asks: how close am I to my own group's members, compared with the nearest *other* group's members? Averaged over everyone, it runs from −1 to +1. Around 0 means the groups overlap so heavily they are arbitrary; **above roughly 0.25 is conventionally read as "there is some real structure here"**; above 0.5 is strong separation. The sweep runs every combination of representation and k, and picks the peak.
>
> Note the caution the rest of this phase develops: silhouette rewards groups that are **tight and well-separated**, and it cannot tell whether that separation reflects genuine behaviour or an artefact the preprocessing manufactured. Here it turns out to be the latter.

**The ILR transform is vindicated on the metric it was proposed for.** Silhouette 0.4115 versus 0.2516 for raw shares is a large margin, and it is the only representation tested here that clears the ~0.25 mark conventionally taken to indicate reasonable structure.

**A design error worth recording.** The first version of this notebook **hardcoded** `BEST_REP = "ILR + participation indicators"` — the representation that scored *worst* (0.2054). The sweep existed to make that choice and was then overridden by an assumption. The notebook now selects `argmax(silhouette)` from the sweep itself. This is exactly the failure mode a selection step is supposed to prevent, and it was caught only because the sweep's output was read rather than skimmed.

**Why adding participation indicators hurt.** The five binary indicators contribute variance that is unrelated to the ILR geometry, and after standardisation each binary column carries as much weight as a continuous ILR coordinate. They dilute a well-formed metric with five axes that KMeans cannot use coherently.

> **In plain terms — why adding information made it worse.** More features normally helps a supervised model, which can learn to ignore useless ones. Clustering has no such defence: **every column feeds into the distance calculation with equal weight**, and nothing tells the algorithm which ones matter.
>
> A yes/no column can only ever take two values, 0 and 1. After standardisation it contributes as much to the distance between two households as a full continuous coordinate does — but it can only say "same" or "different", never "a bit further". Adding five of these to five carefully constructed geometric coordinates half-drowns a well-formed measure of similarity in coarse ones. Hence the worst score in the sweep.

### Cell 8 (code) — The personas (Q1)

| Persona | n | % | Median income | Goal_Met | Median savings rate |
| --- | --- | --- | --- | --- | --- |
| 0 | 4,876 | 11.7% | ₹48,820 | 0.3123 | −0.119 |
| 1 | 28,937 | 69.7% | ₹78,700 | 0.3008 | −0.145 |
| 2 | 7,703 | 18.6% | ₹84,340 | **0.3930** | **+0.035** |

Spending signatures (mean share):

| Category | P0 | P1 | P2 |
| --- | --- | --- | --- |
| Groceries | 0.540 | 0.448 | 0.470 |
| **Transport** | **0.000** | 0.073 | 0.083 |
| **Healthcare** | 0.099 | 0.117 | **0.001** |
| Miscellaneous | 0.148 | 0.135 | 0.187 |
| Utilities | 0.112 | 0.097 | 0.107 |

Two cells are exactly zero, and they are the whole story.

### Cell 9 (code) — What the clusters are actually keyed on

**This is the cell that reframes the phase.** The core parts include Transport (11.4% zero) and Healthcare (19.2% zero), and Phase 2 replaced those zeros with a small δ before taking logs — which places them far out in log-ratio space. So the question is whether the personas are allocation patterns or just zero patterns.

**Share of households recording zero spend:**

| Core part | P0 | P1 | P2 |
| --- | --- | --- | --- |
| Transport | **0.956** | 0.003 | 0.000 |
| Healthcare | 0.246 | 0.002 | **0.868** |
| Groceries | 0.001 | 0.000 | 0.001 |
| Utilities | 0.004 | 0.001 | 0.002 |

Zero-pattern signatures: Persona 0 = `001000` (transport absent), Persona 1 = `000000` (all present), Persona 2 = `000100` (healthcare absent).

**Adjusted Rand index between the personas and the raw zero-pattern of the core parts: 0.858.**

> **In plain terms — the adjusted Rand index.** The **ARI** measures how much two different groupings of the same households agree. Take every pair of households and ask whether the two schemes both put them together, both apart, or disagree. **1.0 means the groupings are identical; 0 means they agree no more than two random groupings would; negative means worse than random.** The "adjusted" part is what subtracts off the agreement you would get by luck alone — without it, any two groupings look somewhat similar simply because most pairs of households are apart in both.
>
> **0.858 is very high agreement.** The elaborate ILR-plus-KMeans machinery has almost exactly reproduced a grouping you could have obtained by asking "which of these categories does this household record as zero?" — no transform, no clustering, one glance at the raw data.

**So the silhouette of 0.4115 is largely manufactured, not discovered.** The multiplicative zero replacement assigns every transport-less household the *same* imputed value, and δ sits far from any genuine share. Those households therefore form a tight, well-separated blob in ILR space — and silhouette rewards exactly that. The clustering has mostly rediscovered which categories the survey recorded as zero.

**This does not make the result worthless, but it changes what it is.** "Persona" implies a style of allocating a budget. What the algorithm found is a **participation pattern**: which categories a household spends on at all. That is a real and interpretable distinction — it just is not the question Phase 6 was posed to answer, and calling these "spending personas" without the qualifier would oversell them.

**The honest methodological conclusion:** ILR is the correct transform for clustering compositional data, and it beat the alternatives on the intended metric. But **zero replacement and silhouette interact badly** — any log-ratio clustering of zero-inflated compositions will tend to find the zero pattern first, and a good silhouette is weak evidence against that. The zero-pattern ARI is the check that distinguishes the two, and it should accompany any log-ratio clustering of data like this.

### Cell 11 (code) — Association with `Goal_Met` (Q3, unconditional)

χ² = 238.8, p ≈ 1.4×10⁻⁵², **Cramér's V = 0.0758**. Highly significant and substantively weak — at n = 41,516 significance is close to guaranteed, so the effect size is the number that matters, and 0.076 is below the conventional 0.1 "weak" mark.

> **In plain terms — significance is not size, and this line shows why.**
> - **χ² (chi-squared)** tests whether two categorical things are related at all — here, persona and goal attainment. It compares the counts observed against the counts you would expect if the two were completely unrelated.
> - The **p-value** is the probability of seeing a pattern this strong purely by chance if there were genuinely no relationship. **1.4×10⁻⁵²** is a decimal point followed by 51 zeros — chance is ruled out about as thoroughly as a number can rule it out.
> - **Cramér's V** answers the entirely different question of *how big* the relationship is, on a 0-to-1 scale. **0.076 is very small.**
>
> Both statements are true at once, and reconciling them is the point. With 41,516 households, even a trivial relationship will register as overwhelmingly "significant" — a large enough sample can detect a nudge. **A p-value tells you the effect is real; it says nothing about whether it is big enough to act on.** Quoting the p-value alone here would imply a strong finding where there is a faint one, which is why the effect size is reported alongside — and why the next cell goes looking for what is hiding it.

Persona 2 attains 39.3% against Persona 1's 30.1% — a 9.2-point spread around a 31.9% base rate.

**This is where the phase would have stopped if it took the headline number at face value**, and it would have concluded that spending personas barely relate to the outcome.

### Cell 12 (code) — Conditioning on income, and the falsified prediction

**The prediction going in (from Phase 5) was that the personas would largely be an income split**, since income is 38.4% of model attribution and spending mix only 26.7%. Two measurements say that is wrong:

**Adjusted Rand index between personas and income tertiles: 0.0057** — essentially independent. The personas are not income groups in disguise.

> **In plain terms — tertiles, deciles, and what this test rules out.** **Tertiles** split households into three equal-sized income groups (poorest third, middle third, richest third); **deciles** split them into ten. Sorting by a value and cutting into equal-sized bands is the standard way to compare like with like.
>
> The worry being tested: the personas might just be income brackets under another name — a real risk, since [Phase 5](phase5.md) found income dominates everything. If so, the ARI against income tertiles would be high. **At 0.0057 it is indistinguishable from zero**, so the personas are genuinely capturing something other than how much money a household has. That matters for the next result, because it means the persona effect and the income effect are separate things that can be disentangled.

**And the association with the outcome is *suppressed* by income, not driven by it:**

| Income decile | Persona 0 | Persona 1 | Persona 2 |
| --- | --- | --- | --- |
| 0 | 0.051 | 0.009 | 0.030 |
| 3 | 0.261 | 0.105 | 0.231 |
| 5 | 0.472 | 0.249 | 0.368 |
| **7** | **0.789** | 0.466 | 0.569 |
| 9 | 0.911 | 0.835 | 0.811 |

| | |
| --- | --- |
| Raw spread across personas | 0.0921 |
| Mean spread **within** income decile | **0.1668** |
| Proportion of the effect surviving the control | **181%** |

**The persona effect nearly doubles once income is held constant.** This is classic suppression: Persona 0 has the *lowest* median income (₹48,820) but the *highest* within-decile attainment, so in the unconditional comparison its low income cancels its behavioural advantage. At decile 7 the gap between Persona 0 and Persona 1 is **32 percentage points** — far from the 9-point unconditional spread.

> **In plain terms — suppression, and how an effect can hide.** Persona 0 has two things going on that pull in **opposite** directions: its households are poorer than average (which lowers goal attainment) and they spend on fewer categories (which raises it). Compare the personas without accounting for income and the two cancel, leaving a modest 9-point gap that badly understates both.
>
> Compare households **within the same income decile** — like with like — and the income disadvantage is removed from the comparison, leaving the behavioural effect standing alone: 32 points at decile 7. A third variable that masks a relationship this way is called a **suppressor**, and it is the mirror image of the more familiar confounder, where a third variable manufactures a relationship that is not really there.
>
> "**181% of the effect survives the control**" is the odd-looking way of stating this: rather than shrinking when income was accounted for, as most effects do, the gap grew to nearly twice its original size. That is why the phase reports the falsified prediction so prominently — the expectation was that controlling for income would make the personas *less* interesting, and the opposite happened.

**Why this makes mechanical sense.** Personas 0 and 2 are defined by *not spending* on a whole category. Fewer active expense categories means lower total consumption at a given income, and the target is `1 − COTOTAL/INCOME`. Households that record no transport or no healthcare spending are, other things equal, spending less overall — so they save more. This is the same mechanism Phase 5 found behind the grocery-share result: a budget concentrated in fewer categories signals the absence of the outlays that push consumption above income.

**A necessary caveat.** Some of this is measurement, not behaviour. A household recording zero healthcare spend may have had no health event that year rather than a savings strategy, and zero transport spend plausibly marks a household that does not commute. These are not levers a savings-product team can pull — nobody should be advised to stop spending on healthcare. Phase 7 must treat these personas as **descriptive segments, not as interventions.**

### Cell 14 (code) — Figure and artifacts

Three panels: the spending-signature heatmap (where the two exact zeros are visible at a glance), goal attainment by persona with Cramér's V in the title, and income distribution by persona labelled with the ARI. The third panel's caption originally asserted that "the segmentation is largely an income split" — the ARI of 0.006 contradicted it, and the caption now states the measured result instead.

---

## What this changes for later phases

| Phase | Consequence |
| --- | --- |
| **7 — Business translation** | The personas are **descriptive, not actionable** — they are defined by category absence, some of which is measurement (no health event) rather than choice. Report the within-income spread (0.167), never the unconditional 0.092, and never Cramér's V alone. |
| **8 — Reporting** | `results/personas.png` is the persona figure. The headline is that spending structure matters **more** than it first appears, once income is controlled — the opposite of the naive read of Cramér's V. |
