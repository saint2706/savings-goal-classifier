# Phase 6 (IHDS-II) — Unsupervised Extension

**Source:** [README § Phase 6 — Unsupervised Extension](../README.md#phase-6--unsupervised-extension)
**Notebook:** [`notebooks/ihds_06_clustering_personas.ipynb`](../notebooks/ihds_06_clustering_personas.ipynb)
**Builds on:** [Phase 2 (IHDS-II)](ihds_phase2.md), [Phase 5 (IHDS-II)](ihds_phase5.md)
**Artifacts:** `results/ihds_persona_profiles.csv`, `results/ihds_personas.png`, `results/ihds_cluster_selection.png`
**Replaces:** [`phase6.md`](phase6.md)

Phase 2 rejected the CLR transform for classification but left the log-ratio question open for clustering, which is the step that genuinely needs a metric on the simplex. This phase builds an **ILR** basis (full-rank, unlike CLR), uses it, and then discovers that the clean-looking clusters it produces are keyed on something other than what "spending persona" implies.

Two predictions were made going in. **One was confirmed, one was falsified**, and both are reported as such.

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

**Three properties verified rather than assumed:**

| Check | Result |
| --- | --- |
| `V'V = I` (orthonormal) | max error 2.22×10⁻¹⁶ |
| Columns sum to zero (orthogonal to the constant) | exactly 0 |
| Aitchison distance = Euclidean distance in ILR space | max error 1.78×10⁻¹⁵ |

The third is the one that matters for this phase: it is the formal statement that **KMeans in ILR space is doing legitimate geometry on the simplex**, which it is not doing on raw shares. Checking it costs three lines and converts an assumption into a fact.

### Cell 5 (code) — Choosing the representation and k (Q2)

| k | ILR (6 core) | ILR + indicators | Raw shares (11) |
| --- | --- | --- | --- |
| 2 | 0.3705 | 0.1890 | 0.1988 |
| **3** | **0.4115** | 0.2054 | 0.1756 |
| 4 | 0.2829 | 0.1907 | 0.1999 |
| 6 | 0.2358 | 0.1857 | 0.2321 |
| 8 | 0.2385 | 0.2026 | 0.2516 |

**The ILR transform is vindicated on the metric it was proposed for.** Silhouette 0.4115 versus 0.2516 for raw shares is a large margin, and it is the first clustering in either track of this project to clear ~0.25. The synthetic Phase 6 never got above 0.1 at any k.

**A design error worth recording.** The first version of this notebook **hardcoded** `BEST_REP = "ILR + participation indicators"` — the representation that scored *worst* (0.2054). The sweep existed to make that choice and was then overridden by an assumption. The notebook now selects `argmax(silhouette)` from the sweep itself. This is exactly the failure mode a selection step is supposed to prevent, and it was caught only because the sweep's output was read rather than skimmed.

**Why adding participation indicators hurt.** The five binary indicators contribute variance that is unrelated to the ILR geometry, and after standardisation each binary column carries as much weight as a continuous ILR coordinate. They dilute a well-formed metric with five axes that KMeans cannot use coherently.

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

**So the silhouette of 0.4115 is largely manufactured, not discovered.** The multiplicative zero replacement assigns every transport-less household the *same* imputed value, and δ sits far from any genuine share. Those households therefore form a tight, well-separated blob in ILR space — and silhouette rewards exactly that. The clustering has mostly rediscovered which categories the survey recorded as zero.

**This does not make the result worthless, but it changes what it is.** "Persona" implies a style of allocating a budget. What the algorithm found is a **participation pattern**: which categories a household spends on at all. That is a real and interpretable distinction — it just is not the question Phase 6 was posed to answer, and calling these "spending personas" without the qualifier would oversell them.

**The honest methodological conclusion:** ILR is the correct transform for clustering compositional data, and it beat the alternatives on the intended metric. But **zero replacement and silhouette interact badly** — any log-ratio clustering of zero-inflated compositions will tend to find the zero pattern first, and a good silhouette is weak evidence against that. The zero-pattern ARI is the check that distinguishes the two, and it should accompany any log-ratio clustering of data like this.

### Cell 11 (code) — Association with `Goal_Met` (Q3, unconditional)

χ² = 238.8, p ≈ 1.4×10⁻⁵², **Cramér's V = 0.0758**. Highly significant and substantively weak — at n = 41,516 significance is close to guaranteed, so the effect size is the number that matters, and 0.076 is below the conventional 0.1 "weak" mark.

Persona 2 attains 39.3% against Persona 1's 30.1% — a 9.2-point spread around a 31.9% base rate.

**This is where the phase would have stopped if it took the headline number at face value**, and it would have concluded that spending personas barely relate to the outcome.

### Cell 12 (code) — Conditioning on income, and the falsified prediction

**The prediction going in (from Phase 5) was that the personas would largely be an income split**, since income is 38.4% of model attribution and spending mix only 26.7%. Two measurements say that is wrong:

**Adjusted Rand index between personas and income tertiles: 0.0057** — essentially independent. The personas are not income groups in disguise.

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

**Why this makes mechanical sense.** Personas 0 and 2 are defined by *not spending* on a whole category. Fewer active expense categories means lower total consumption at a given income, and the target is `1 − COTOTAL/INCOME`. Households that record no transport or no healthcare spending are, other things equal, spending less overall — so they save more. This is the same mechanism Phase 5 found behind the grocery-share result: a budget concentrated in fewer categories signals the absence of the outlays that push consumption above income.

**A necessary caveat.** Some of this is measurement, not behaviour. A household recording zero healthcare spend may have had no health event that year rather than a savings strategy, and zero transport spend plausibly marks a household that does not commute. These are not levers a savings-product team can pull — nobody should be advised to stop spending on healthcare. Phase 7 must treat these personas as **descriptive segments, not as interventions.**

### Cell 14 (code) — Figure and artifacts

Three panels: the spending-signature heatmap (where the two exact zeros are visible at a glance), goal attainment by persona with Cramér's V in the title, and income distribution by persona labelled with the ARI. The third panel's caption originally asserted that "the segmentation is largely an income split" — the ARI of 0.006 contradicted it, and the caption now states the measured result instead.

---

## What this changes for later phases

| Phase | Consequence |
| --- | --- |
| **7 — Business translation** | The personas are **descriptive, not actionable** — they are defined by category absence, some of which is measurement (no health event) rather than choice. Report the within-income spread (0.167), never the unconditional 0.092, and never Cramér's V alone. |
| **8 — Reporting** | `results/ihds_personas.png` replaces the synthetic persona figure. The headline is that spending structure matters **more** than it first appears, once income is controlled — the opposite of the naive read. |

## Comparison with the synthetic Phase 6

| | Synthetic | IHDS-II |
| --- | --- | --- |
| Best silhouette | < 0.1 at every k | 0.4115 (k = 3, ILR) |
| What drove the clusters | `Education_Ratio`, `Rent_Ratio` — proxies for `Dependents` and `City_Tier` | Zero-pattern of transport and healthcare spend |
| Relation to `Goal_Met` | **All 112** at-risk cases in one persona (χ² = 360.8) | Cramér's V 0.076 raw; within-income spread 0.167 |
| Honest verdict | A demographic split already present in the raw data | A **participation** pattern, partly a zero-replacement artifact, but carrying real suppressed signal |

Both phases end at the same methodological place from opposite directions: the clusters are real but are not the rich multivariate spending archetypes the phase name implies. In the synthetic data they re-encoded demographics that were baked into the generator; here they re-encode which categories the survey recorded as zero.
