# Phase 6: Spending clusters

**Source:** [README § Phase 6: Spending clusters](../README.md#phase-6-spending-clusters)
**Notebook:** [`notebooks/06_clustering_personas.ipynb`](../notebooks/06_clustering_personas.ipynb)
**Builds on:** [Phase 2](phase2.md) (zero replacement and log-ratio transforms), [Phase 5](phase5.md) (how spending shares relate to the label)
**Artifacts:** `results/personas.json`, `results/persona_profiles.csv`, `results/cluster_selection.png`, `results/personas.png`

This phase groups households by the shape of their budgets, without using `Goal_Met`, and then asks what the groups are keyed on and whether they relate to the target once income is held fixed. The clustering runs on one representation, isometric log-ratio (ILR) coordinates of the six core expense categories, and the number of clusters is chosen within it. The result is three stable spending clusters, defined almost entirely by which categories a household records as zero.

> **In plain terms: supervised and unsupervised.** Phases 3 to 5 were supervised: the data came with an answer (`Goal_Met`) and the model learned to reproduce it. Clustering is unsupervised. The algorithm sees only the spending shares and is asked to put similar households together. It always returns groups, whether or not real groups exist, so most of the work is checking what the groups turned out to be based on.

### Why "spending clusters" and not "personas"

A persona implies a style of managing a budget: a group of households that choose to allocate money in a recognisable way. Two of the three clusters here are defined by a category the household reports no spending on at all (transport in one, healthcare in the other), and the clusters agree with the raw pattern of zeros at an adjusted Rand index of 0.858. A zero for healthcare can mean no one fell ill that year; a zero for transport can mean no one commutes. Those are circumstances and survey recall, not budgeting styles, so this walkthrough calls the groups spending clusters. File and function names in the code (`personas.json`, `persona_income_interaction`) still use the word persona.

---

## Research questions & answers

| # | Question | Answer |
| --- | --- | --- |
| 1 | What spending clusters emerge from clustering on expense-category proportions, and are they behavioural personas? | Three. A no-transport cluster (4,876 households, 11.7%; 95.6% record zero transport), a no-healthcare cluster (7,703, 18.6%; 86.8% record zero healthcare), and a cluster that spends on all six core categories (28,937, 69.7%). Agreement with the raw zero pattern of the core categories is ARI 0.858, and stays between 0.83 and 0.92 across the zero-replacement settings tested. They are participation patterns, not behavioural personas. |
| 2 | How many clusters are statistically justified? | k = 3. Silhouette (0.412), Davies–Bouldin (0.976) and Calinski–Harabasz (17,244) all pick k = 3 out of k = 2 to 8. The partition is stable under bootstrap resampling (median ARI 0.998, 5th to 95th percentile 0.996 to 0.999). The silhouette value itself depends on the zero-replacement constant: it ranges from 0.39 to 0.48 as that constant changes. |
| 3 | Do the clusters correlate meaningfully with `Goal_Met`? | Weakly on their own: on-track rates of 31.2%, 30.1% and 39.3% (spread 9.2 points; χ² 238.8, p 1.4e-52, Cramér's V 0.076). Cluster membership carries almost no information about income decile (NMI 0.012). Within income deciles the association is larger: adding cluster to a logistic model of `Goal_Met` on income decile gives a likelihood-ratio statistic of 774 on 2 df (pseudo-R² 0.265 to 0.280), and at the same income decile the all-categories cluster has 0.34 times the odds of being on track of the no-transport cluster. The likely route is budget arithmetic: an empty category means smaller total spending at a given income. |

---

## Notebook walkthrough

### Cell 1: load, drop two households, build ILR coordinates

The notebook loads the engineered features (`savings_goal.io.load_features`) and takes the core sub-composition from `savings_goal.features.engineer.spec_from_frame`: Groceries, Utilities, Transport, Healthcare, Clothing & footwear, Miscellaneous. These six are the categories that are zero for fewer than 20% of training households; the other five (eating out, rent, education, entertainment, insurance) are zero for 36% to 90% and are left out of the clustering. Two households spend nothing on any core category and have no defined composition, so they are dropped and the count is printed: 41,516 remain.

`savings_goal.models.personas.ilr_coordinates` re-closes the six shares to sum to 1, replaces zeros with `savings_goal.features.transforms.multiplicative_replacement` (each zero becomes 0.65 times the smallest positive value seen in that category, from `zero_replacement_deltas`), and applies `savings_goal.features.transforms.ilr`, which multiplies the centred log-ratios by a 6 × 5 Helmert basis. `savings_goal.models.personas.standardise` then scales the five coordinates to unit variance.

> **In plain terms: why ILR.** Budget shares live on a constrained space: they are non-negative and sum to 1, so ordinary straight-line distance between two households' shares is distorted (a change from 1% to 2% is a doubling; a change from 50% to 51% is not). Log-ratios measure the relative size of categories instead. The centred log-ratio has six columns that always sum to zero, so only five carry information. ILR rewrites the same information in exactly five columns using a fixed rotation, so straight-line distance in ILR space equals the correct compositional (Aitchison) distance. That lets k-means, which only knows straight-line distance, measure similarity in a way that respects the constraint.

> **In plain terms: zero replacement.** A logarithm of zero is undefined, so zeros must be replaced with a small positive number before taking log-ratios. The choice of that number (here 0.65 times the smallest observed positive share) is a convention. Every household with a zero gets the same replacement value in that category, and that value sits far from any real share on the log scale. Cell 5 tests how much the clusters depend on it.

### Cell 3: choose k within the ILR representation

`savings_goal.models.personas.kmeans_sweep` fits k-means for k = 2 to 8 (10 initialisations each) and records three internal validity indices. Silhouette is computed on a 5,000-household sample.

| k | Silhouette (higher better) | Davies–Bouldin (lower better) | Calinski–Harabasz (higher better) | Inertia |
| --- | --- | --- | --- | --- |
| 2 | 0.371 | 1.410 | 16,154 | 149,432 |
| 3 | 0.412 | 0.976 | 17,244 | 113,384 |
| 4 | 0.283 | 1.222 | 15,377 | 98,320 |
| 5 | 0.294 | 1.286 | 13,604 | 89,826 |
| 6 | 0.236 | 1.308 | 12,766 | 81,799 |
| 7 | 0.251 | 1.217 | 12,416 | 74,278 |
| 8 | 0.239 | 1.211 | 11,862 | 69,185 |

All three indices pick k = 3, and the silhouette of 0.412 is above the 0.25 line conventionally read as weak structure. Inertia (total within-cluster distance) falls steadily with k, as it always does, and shows no sharp elbow, so it is not used to choose k.

`savings_goal.models.personas.bootstrap_stability` then refits k = 3 on 30 bootstrap resamples and compares each resulting partition with the reference partition on the full data. Median ARI is 0.998 (5th to 95th percentile 0.996 to 0.999): the same three groups come back every time.

`cluster_selection.png` plots the three indices against k, with the 0.25 silhouette line marked.

> **In plain terms: k and the validity indices.** k-means needs the number of groups, k, in advance and will produce any k it is given. The indices score how well a grouping separates the data. Silhouette asks, for each household, whether it is closer to its own group than to the nearest other group, averaged over households on a scale from −1 to +1. Davies–Bouldin compares how spread out each group is with how far apart the groups are (lower is better). Calinski–Harabasz is the ratio of between-group to within-group spread (higher is better). None of them can tell whether the separation reflects behaviour or an artefact of preprocessing.

> **In plain terms: bootstrap stability.** Draw a new sample of the same size from the households, with replacement, re-run the clustering, and see whether it finds the same groups. Repeating this 30 times shows whether the grouping depends on which particular households happened to be in the data.

### Cell 5: is the partition real structure, or the zero pattern?

The cell fits the final k = 3 model with `savings_goal.models.personas.fit_kmeans` (20 initialisations). `savings_goal.models.personas.zero_pattern` labels each household by which of the six core categories it records as zero (for example "transport zero, everything else positive"), and the ARI between that labelling and the clusters is 0.858.

`savings_goal.models.personas.delta_sensitivity` then reruns the whole clustering with the zero-replacement constant at five multiples of the smallest positive share:

| Replacement factor | Silhouette | ARI vs the 0.65 partition | ARI vs zero pattern |
| --- | --- | --- | --- |
| 0.10 | 0.477 | 0.932 | 0.922 |
| 0.25 | 0.447 | 0.955 | 0.899 |
| 0.50 | 0.422 | 0.985 | 0.872 |
| 0.65 | 0.412 | 1.000 | 0.858 |
| 1.00 | 0.392 | 0.966 | 0.828 |

Three things follow. The partition itself barely moves (ARI 0.93 to 1.00 against the baseline). The silhouette that makes k = 3 look well separated moves from 0.39 to 0.48 with a constant that has no substantive meaning: smaller replacement values push zero households further from everyone else and make their cluster look tighter. And at every setting the clusters stay keyed on the zero pattern (ARI 0.83 to 0.92), more so as the replacement value shrinks.

> **In plain terms: the adjusted Rand index (ARI).** ARI measures how much two groupings of the same households agree. For every pair of households it checks whether both groupings put them together, both put them apart, or disagree, then subtracts the agreement two random groupings would reach by chance. 1 means identical, 0 means no better than chance. An ARI of 0.858 with the zero pattern means the ILR and k-means pipeline has mostly reproduced a grouping you could get by asking each household which categories it reported no spending on.

### Cell 7: what the clusters look like

The cell profiles each cluster: size, median income, median household size, on-track rate (unweighted and survey-weighted with `savings_goal.evaluation.metrics.weighted_mean`), median savings rate, share of the sample and of the weighted population, mean expense shares, and the share of households with zero spend in each core category. A cluster is labelled by the core categories that more than half its households report as zero.

| Cluster | Households | % of sample | % of population (weighted) | Median income (Rs) | Median size | On track | On track (weighted) | Median savings rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| No transport spend | 4,876 | 11.7 | 12.2 | 48,820 | 4 | 31.2% | 29.9% | −0.119 |
| Spends on all core categories | 28,937 | 69.7 | 71.6 | 78,700 | 5 | 30.1% | 28.8% | −0.145 |
| No healthcare spend | 7,703 | 18.6 | 16.2 | 84,340 | 4 | 39.3% | 38.3% | +0.035 |

Mean shares and zero rates in the categories that separate them:

| Category | No transport: mean share | No transport: % zero | All core: mean share | All core: % zero | No healthcare: mean share | No healthcare: % zero |
| --- | --- | --- | --- | --- | --- | --- |
| Groceries | 0.540 | 0.1 | 0.448 | 0.0 | 0.470 | 0.1 |
| Transport | 0.000 | 95.6 | 0.073 | 0.3 | 0.083 | 0.0 |
| Healthcare | 0.099 | 24.6 | 0.117 | 0.2 | 0.001 | 86.8 |
| Miscellaneous | 0.148 | 0.0 | 0.135 | 0.1 | 0.187 | 0.0 |
| Utilities | 0.112 | 0.4 | 0.097 | 0.1 | 0.107 | 0.2 |
| Clothing & footwear | 0.044 | 3.0 | 0.042 | 1.4 | 0.052 | 0.4 |

Outside the zero columns the clusters differ modestly: the no-transport cluster puts more of its budget on groceries (54% against 45% to 47%), and the no-healthcare cluster more on miscellaneous items (19% against 14% to 15%). The full 11-category profile is in `results/persona_profiles.csv`.

### Cell 9: do the clusters relate to `Goal_Met`, and is that just income?

Unconditionally the link is weak. On-track rates span 9.2 points (30.1% to 39.3%); a chi-squared test gives χ² 238.8 and p 1.4e-52, with Cramér's V 0.076.

> **In plain terms: significance and size.** The chi-squared test asks whether cluster and on-track status are related at all; the p-value is the chance of a pattern this strong if they were unrelated, and 1.4e-52 rules chance out. Cramér's V measures how strong the relation is, from 0 to 1, and 0.076 is very small. With 41,516 households even a small difference is detected with near certainty, so the size figure is the one to read.

`savings_goal.models.personas.income_association` checks whether the clusters are income bands in disguise. The normalised mutual information between cluster and income decile is 0.012, so knowing a household's cluster says almost nothing about its income decile. A Kruskal–Wallis test finds that income distributions do differ across clusters (H = 1,391, p 7.5e-303), but the effect size ε² is 0.033: cluster explains about 3% of the variation in income ranks. The no-transport cluster is the poorest (median Rs 48,820).

> **In plain terms: NMI and ε².** Normalised mutual information measures how much knowing one grouping tells you about another, from 0 (nothing) to 1 (one determines the other). The Kruskal–Wallis test compares groups by the ranks of their incomes rather than the incomes themselves, which suits a skewed variable like income; ε² is its effect size, the share of rank variation explained by the grouping.

The notebook then compares clusters within income deciles:

| Income decile | No transport | All core | No healthcare |
| --- | --- | --- | --- |
| 0 (lowest) | 5.1% | 0.9% | 3.0% |
| 1 | 12.9% | 3.1% | 7.8% |
| 2 | 17.5% | 5.1% | 12.3% |
| 3 | 26.1% | 10.5% | 23.1% |
| 4 | 36.2% | 16.9% | 30.9% |
| 5 | 47.2% | 24.9% | 36.8% |
| 6 | 57.2% | 34.5% | 43.1% |
| 7 | 78.9% | 46.6% | 56.9% |
| 8 | 81.6% | 63.6% | 68.3% |
| 9 (highest) | 91.1% | 83.5% | 81.1% |

In every decile the no-transport cluster has the highest on-track rate, and in deciles 0 to 8 the all-core cluster has the lowest. The unconditional comparison hides this because the no-transport cluster is the poorest: its low income pulls its overall rate down to about the level of the all-core cluster.

`savings_goal.models.personas.persona_income_interaction` tests this formally with three nested logistic regressions of `Goal_Met`: on income decile alone, on decile plus cluster, and with a decile × cluster interaction.

| Comparison | LR statistic | df | p |
| --- | --- | --- | --- |
| Adding cluster to income decile | 774.1 | 2 | 8.2e-169 |
| Adding decile × cluster interaction | 133.5 | 18 | 1.1e-19 |

McFadden pseudo-R² rises from 0.265 (decile only) to 0.280 with cluster added. At the same income decile, the odds of being on track are 0.34 times as high in the all-core cluster and 0.54 times as high in the no-healthcare cluster as in the no-transport cluster. The significant interaction means the size of the gaps varies across deciles (at decile 9 the all-core and no-healthcare clusters swap order).

> **In plain terms: the likelihood-ratio test.** Fit a model without the cluster, fit it again with the cluster, and compare how well each explains the observed outcomes. The likelihood-ratio statistic is twice the improvement in log-likelihood; if cluster added nothing, it would follow a chi-squared distribution with as many degrees of freedom as the parameters added (2 here). A statistic of 774 on 2 df is far beyond chance. Pseudo-R² is a rough analogue of R² for logistic models; a rise from 0.265 to 0.280 shows the gain is real but small next to income.

> **In plain terms: odds ratio.** The odds of being on track are the on-track probability divided by the at-risk probability. An odds ratio of 0.34 means that, comparing households in the same income decile, the all-core cluster's odds are about a third of the no-transport cluster's.

The most likely mechanism is arithmetic about the label. The target is a savings rate of at least 20%, that is total spending at most 80% of income. At a given income, a household that reports no transport or no healthcare spending has, other things equal, a smaller total, and so a higher savings rate. This is the same route Phase 5 found behind the food-share attribution: a share pattern that implies a smaller budget relative to income predicts being on track. Part of it is also measurement: a zero in a recall-based category can reflect no health event or no commute in that year. Neither is something a household could be advised to change.

### Cell 10: figure and results

`personas.png` has three panels: a heatmap of mean expense shares by cluster (the near-zero transport and healthcare cells stand out), the on-track rate per cluster against the overall rate with Cramér's V in the title, and the on-track rate by income decile with one line per cluster, titled with the likelihood-ratio p-value and NMI.

The cell writes `results/persona_profiles.csv` (cluster profile joined with mean shares) and `results/personas.json` via `savings_goal.io.write_result`: chosen k, the k sweep, the index votes, the 30 bootstrap ARIs, ARI with the zero pattern, the replacement-constant sweep, cluster labels, chi-squared and Cramér's V, income-association statistics, the logistic-regression tests and odds ratios, the within-decile table, and zero rates by cluster.

---

## What this means for later phases

| Phase | Consequence |
| --- | --- |
| 7: Business translation | Treat the three spending clusters as descriptive segments. They are defined by absent categories, and the gap in on-track rates between them at a given income follows from smaller budgets, not from a habit that could be taught. Report the income-adjusted comparison (odds ratios 0.34 and 0.54, LR 774 on 2 df) rather than Cramér's V alone, and do not recommend cutting healthcare or transport spending. |
| 8: Reporting | `personas.png` is the cluster figure. The main points for a reader: the clusters are stable, they are mostly a zero-spend pattern, and their link to `Goal_Met` appears within income deciles. |
