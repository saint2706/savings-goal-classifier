# Phase 5 — Explainability

**Source:** [README § Phase 5 — Explainability](../README.md#phase-5--explainability)
**Notebook:** [`notebooks/05_explainability.ipynb`](../notebooks/05_explainability.ipynb)
**Builds on:** [Phase 4](phase4.md), [Phase 2](phase2.md)
**Artifacts:** `results/shap_importance.csv`, `results/shap_summary.png`, `results/shap_dependence.png`

Phase 2 left an unexploded charge under this phase: the 11 expense shares sum to 1, so they are **exactly singular** (VIF = ∞) and no coefficient on them is identified. Phase 4 confirmed the logistic fit is only unique because of its L2 penalty. This phase deals with that first, then explains the model.

> **In plain terms — what explainability is, and the problem blocking it.** A model that predicts well but cannot say *why* is hard to trust, hard to debug, and impossible to build a business recommendation on. **Explainability** is the work of extracting reasons: which features drive the outcome overall, and why this particular household got the score it did.
>
> The obvious route is to read the model's **coefficients** — the weights a linear model attaches to each feature. [Phase 2](phase2.md) closed that route: because the eleven shares always add to 1, there are infinitely many sets of coefficients that predict identically, and the one reported was picked by the L2 penalty rather than by the data. Reading it as "the effect of food spending" would be reporting an arbitrary choice as a finding.
>
> So this phase does two things in order: confirms that the route really is closed (and, importantly, shows why the obvious test for it is misleading), then explains the model a different way — with **SHAP**, introduced below.

---

## Research questions & answers

| # | Question | Answer |
| --- | --- | --- |
| 1 | Which features matter most globally? | `Log_Income` dominates at mean \|SHAP\| **2.617** — 4.5× the next feature and **38.4%** of all attribution. Then `Household_Size` (0.576), `Groceries_Share` (0.555), `Utilities_Share` (0.314). By group: income 38.4%, spending mix 26.7%, categoricals 13.3%, household composition 11.2%. |
| 2 | Are there notable interaction effects? | **Yes — interactions are 41.9% of total attribution**, far from negligible. The strongest are `Groceries_Share × Log_Income` (0.166), `Household_Size × Log_Income` (0.102) and `Utilities_Share × Log_Income` (0.089). Income does not merely add to the prediction; it changes what every spending feature means. |
| 3 | Can individual predictions be explained in plain business language? | Yes — SHAP is additive, and the notebook prints per-household explanations verified against the raw margin to 8.6×10⁻⁶. The worked examples also expose a data-quality caveat that a purely aggregate view hides. |

---

## Dealing with the identification problem first

### The stability test — and why it under-detects the problem

The same regularised logistic model was fitted on **5 disjoint subsamples**. If the shares were identified, coefficients would be stable across them.

| Share feature | mean | sd | sd/\|mean\| | flips sign |
| --- | --- | --- | --- | --- |
| `Insurance_Share` | 0.006 | 0.011 | **1.808** | **yes** |
| `Healthcare_Share` | −0.204 | 0.132 | 0.648 | **yes** |
| `Entertainment_Share` | −0.017 | 0.010 | 0.603 | **yes** |
| `Eating_Out_Share` | 0.035 | 0.020 | 0.555 | no |
| `Groceries_Share` | 1.084 | 0.163 | 0.150 | no |

Control group (non-compositional): `Log_Income` sd/|mean| **0.029**, `Household_Size` 0.137, `Debt_To_Income_W` 0.161.

> **In plain terms — the test and its columns.** Split the households into five separate groups, fit the same model on each, and compare the coefficients. If a coefficient is measuring something real, five independent fits should broadly agree on it.
>
> - **sd/|mean|** puts the disagreement on a comparable scale: the wobble divided by the size of the thing wobbling. 0.029 means the five fits agree to within about 3% — solid. **1.808 means the wobble is nearly twice the value itself** — the coefficient is not being measured, it is being invented afresh each time.
> - **Flips sign** is the starkest version: on some subsamples the feature appears to help, on others to hurt. There is no coherent reading of a feature that cannot decide which direction it points.
> - The **control group** is the comparison that makes the result interpretable. `Log_Income` and `Household_Size` are ordinary features, not part of the sum-to-1 constraint, and they behave themselves. So the instability is not "small samples wobble" — it is specific to the compositional features, exactly as the algebra predicted.

**3 of 11 shares flip sign across subsamples; 1 of 5 controls does.** `Log_Income`'s coefficient varies by 2.9% of its magnitude; `Insurance_Share`'s standard deviation is **1.8× its own mean**.

**But this test is weaker evidence than it looks, and that matters.** L2 regularisation resolves a singular system by picking the *minimum-norm* solution — and it picks the same tie-break every time. So coefficients can be perfectly **reproducible** across subsamples while still not being **identified**: the regulariser, not the data, is choosing how credit is split between collinear shares. Reproducibility is not identification, and a stability test alone cannot distinguish them. The algebra (a feature that is an exact linear combination of others has no unique coefficient) is the real argument; this table only shows the symptom leaking through where the penalty binds loosest.

> **In plain terms — reproducible is not the same as identified, and why that is a trap.** The **minimum-norm solution** is the specific tie-break L2 uses: among the infinitely many coefficient sets that predict identically, take the one with the smallest coefficients overall. It is a fixed rule, so it lands on the same answer every single time.
>
> Now consider what that does to a stability test. The five subsamples could have produced *perfectly matching* coefficients — and it would have proved nothing, because the agreement would come from all five obeying the same tie-break rule, not from the data pinning down an answer. **Reproducible** means "you get the same number when you repeat it". **Identified** means "the data determines that number". A consistent bathroom scale that reads 5 kg heavy is reproducible and wrong.
>
> This is why the section says the algebra is the real argument. The proof was already complete in [Phase 2](phase2.md); the table is only a visible symptom in the places where the penalty happens to grip loosest — and had the table come out clean, the conclusion would have been unchanged.

### The reference-part fix, which does not fully work

The standard remedy is to drop one part and read every remaining coefficient *relative* to it — the compositional analogue of a dummy baseline. Dropping `Groceries_Share`:

| Feature | sd (all 11) | sd (reference dropped) | improvement |
| --- | --- | --- | --- |
| `Education_Share` | 0.043 | 0.009 | **4.9×** |
| `Miscellaneous_Share` | 0.052 | 0.016 | 3.3× |
| `Healthcare_Share` | 0.132 | 0.059 | 2.2× |
| `Insurance_Share` | 0.011 | 0.007 | 1.7× |
| `Utilities_Share` | 0.077 | 0.083 | 0.9× |
| **`Rent_Share`** | 0.062 | **1.027** | **0.06× (17× worse)** |

Median improvement: **1.22×** — modest. And `Rent_Share` becomes *dramatically* less stable, because it is zero for 90.4% of households (Phase 1); once the largest part is removed as the reference, the remaining variation in a mostly-zero column is too thin to pin its coefficient down.

**Conclusion: the reference-part fix is not reliable here**, and it is reported as attempted-and-partially-failed rather than presented as the solution. **SHAP on the tree model is used instead** — it requires no matrix inversion, makes no identifiability assumption, and distributes credit by a game-theoretic rule that is well defined even under exact collinearity.

> **In plain terms — what SHAP is.** SHAP answers, for one household: *how much did each feature push this prediction up or down, relative to the average household?*
>
> The idea comes from **cooperative game theory** and is nearly a century old. Imagine the features as a team that jointly produced a result, and you must divide the credit fairly. The Shapley value does it by considering every possible order in which team members could have joined, measuring what each one added when they arrived, and averaging over all those orders. It is the unique way of splitting credit that satisfies a short list of fairness conditions — including that the parts must add up exactly to the whole.
>
> Two properties make it the right tool here. It needs **no matrix inversion**, so the singularity that wrecked the coefficients is irrelevant to it. And it is **additive**: the contributions for one household sum exactly to that household's prediction, which is what makes a per-household explanation readable rather than merely indicative. The catch is cost — considering every ordering is astronomically expensive in general, which is why **TreeSHAP**, an exact shortcut that exploits the structure of decision trees, is what makes this practical on 41,518 households.

---

## Notebook walkthrough

### Cell 1 (code) — Fit Phase 4's winning configuration on transformed data

The `ColumnTransformer` is applied explicitly and XGBoost fitted on the resulting frame, rather than wrapping both in a `Pipeline`. **Why:** SHAP needs to attribute to the *encoded* columns (50 of them, after one-hot expansion), and a pipeline hides those names. Test accuracy 0.8615 confirms this reproduces Phase 4.

> **In plain terms — the two tools, and why one was unwrapped.** A **`ColumnTransformer`** applies the right preparation to each column — scale the numeric ones, one-hot the categorical ones. A **`Pipeline`** then bolts that preparation onto the model so the pair travels as one object, which is normally exactly what you want: the same preparation is guaranteed to be applied at prediction time as at training time, so it cannot be forgotten or applied differently by mistake.
>
> The problem here is that the model does not see the 28 features written in [Phase 2](phase2.md) — one-hot encoding **expands** them to 50, since each category level becomes its own column. SHAP attributes to those 50, and a pipeline keeps them anonymous behind its interface. Unwrapping the two steps makes the encoded names visible so the attributions can be labelled. The accuracy check exists because unwrapping means reassembling by hand: matching [Phase 4](phase4.md)'s 0.8615 confirms the same model was rebuilt, not a subtly different one.

### Cell 6 (code) — Global importance via native TreeSHAP

**An implementation note worth recording.** `shap.TreeExplainer` **fails on this model**: shap 0.49 cannot parse XGBoost 3.1's `base_score` serialisation (`could not convert string to float: '[3.1929308E-1]'`). The notebook uses XGBoost's own `pred_contribs=True` instead — the same TreeSHAP algorithm, computed inside XGBoost. It is verified rather than assumed: **SHAP values plus bias reproduce the raw model margin to a maximum absolute error of 8.58×10⁻⁶**, which is the additivity property TreeSHAP guarantees.

> **In plain terms — what that verification proves.** The **bias** (or base value) is the model's starting point before any feature is considered — roughly, the average household's score. The **raw margin** is the model's output before it is squashed into a 0–1 probability. SHAP's promise is that *bias + all the feature contributions = the actual prediction*, exactly, for every household.
>
> So the check is straightforward: add them up and compare with what the model really said. Agreement to 8.58×10⁻⁶ — eight millionths, i.e. rounding dust — means the contributions genuinely account for the prediction with nothing unexplained. This matters because the library's usual entry point had to be bypassed; rather than trusting the substitute route, the arithmetic was checked directly.

**Top features (mean |SHAP|):**

> **In plain terms — mean |SHAP|.** Each household gets its own SHAP value per feature, positive or negative. To rank features overall, take the **absolute value** (drop the minus signs, so a strong downward push counts as much as a strong upward one) and average across households. It measures how much a feature **moves** predictions, not which direction it moves them — a feature that pushes half the households up and half down still scores high, correctly, because it is doing a lot of work.

| Feature | mean \|SHAP\| |
| --- | --- |
| `Log_Income` | **2.6170** |
| `Household_Size` | 0.5755 |
| `Groceries_Share` | 0.5546 |
| `Utilities_Share` | 0.3136 |
| `Clothing_Footwear_Share` | 0.2204 |
| `Max_Adult_Education` | 0.2151 |
| `Debt_To_Income_W` | 0.2036 |

**By group:**

| Group | Share of attribution |
| --- | --- |
| Income | **38.4%** |
| Spending mix (11 shares) | 26.7% |
| Categoricals (all levels) | 13.3% |
| Household composition | 11.2% |
| Participation indicators | 4.0% |
| Debt | 3.3% |
| Education | 3.2% |

**This confirms Phase 3's prediction.** Income is 38.4% of all attribution on its own, and Phase 3 showed a single income threshold already reaches macro-F1 0.7425. **Spending mix is real and worth 26.7%, but it is the second story, not the first** — any narrative that leads with spending behaviour is contradicting the model's own attribution.

`Household_Size` at second place is new and was not visible in the univariate work — larger households consume more at any income, mechanically depressing the savings rate.

**How to read a share's SHAP value (the compositional caveat).** Because the shares sum to 1, a household cannot raise one without lowering others. So `Groceries_Share`'s attribution is never "the effect of spending more on food"; it is **"the effect of food taking a larger portion of the budget, and everything else a correspondingly smaller one."** Every sentence about a share in Phase 7 must carry that relative framing.

### Cell 9 (code) — Interactions (Q2)

> **In plain terms — an interaction.** Two features **interact** when the effect of one depends on the value of the other. Sugar improves coffee and improves lemonade, but sugar-in-coffee-with-salt is a different story — the effect is not a fixed amount you can add up independently.
>
> Here the headline example is that a high grocery share means something different at ₹40,000 of income than at ₹400,000. An **additive** model — logistic regression, linear SVM — assigns each feature one fixed effect and literally cannot express that; a tree can, because a branch reached only by low-income households can behave differently from its high-income sibling.

Computed with `pred_interactions=True` on a 3,000-household sample (the full tensor is n × 50 × 50).

> **In plain terms — why only 3,000 households.** Attributing to *pairs* of features means one number for every pair, for every household: 50 × 50 for each of 41,518 rows. That grid of numbers (a **tensor**, the general term for an array with more than two dimensions) runs to about 104 million entries, so the calculation is run on a random 3,000-household sample instead. Averages over 3,000 households are plenty precise for ranking which pairs matter.

| Pair | mean \|interaction SHAP\| |
| --- | --- |
| `Groceries_Share` × `Log_Income` | **0.16632** |
| `Household_Size` × `Log_Income` | 0.10154 |
| `Utilities_Share` × `Log_Income` | 0.08910 |
| `Groceries_Share` × `Utilities_Share` | 0.04021 |
| `Max_Adult_Education` × `Log_Income` | 0.03561 |

Main effects total 6.509; interactions total 4.689 — **interactions are 41.9% of all attribution.**

**Why this is a substantive finding, not a technical footnote.** More than two-fifths of the model's behaviour is interaction, and **eight of the ten strongest pairs involve `Log_Income`**. A purely additive model — logistic regression, a linear SVM — is provably incapable of representing any of it. Income is not simply an additive term: it changes what every spending signal means. A high grocery share means something different for a household earning ₹40,000 than for one earning ₹400,000.

This also explains why Phase 4's logistic regression trailed XGBoost on macro-F1 (0.8186 vs 0.8371) while nearly matching it on ROC-AUC (0.9213 vs 0.9306) — a linear model captures the ranking well but misses the interaction structure that sharpens the decision boundary.

**One caveat on the 41.9% figure:** TreeSHAP interaction values are defined relative to the fitted tree ensemble, so this measures how much *this model* relies on interactions, not how much interaction exists in the population. Phase 4 found `max_depth=4` optimal with very low sensitivity, so the model has limited capacity for deep interaction — the true figure could be higher.

> **In plain terms — the caveat, and the apparent contradiction with Phase 4.** SHAP explains *the model*, not the world. It reports how this particular fitted ensemble reaches its answers; a different model trained on the same households would produce different attributions. So 41.9% is a fact about our model's machinery, and it is a floor rather than an estimate — a model capped at four questions deep can only represent so much interaction, so the households may well contain more than it captured.
>
> This also resolves what looks like a clash with [Phase 4](phase4.md), where deeper trees bought nothing and the signal was called "close to additive". Both are true: the interactions that matter here are **shallow** — mostly one feature paired with income, which four levels of depth capture comfortably. Depth beyond that adds nothing because there is no deeper structure to find, not because interactions are unimportant.

### Cell 11 (code) — The dependence plots, and a finding that reverses the naive reading

> **In plain terms — a dependence plot.** One dot per household: its value for a feature along the bottom, that feature's SHAP contribution for that household up the side. It answers "as this feature rises, which way and how hard does the model push?" — and, because the dots are coloured by a second feature, whether that push differs between groups. Dots of different colours lying on top of one another means the relationship is the same for everyone; colours separating into distinct bands means an interaction.

**Left panel — income.** The SHAP curve is monotone increasing and sigmoid-shaped, saturating below log-income ≈ 9.5 and above ≈ 13.5.

> **In plain terms — sigmoid, saturating, and those log numbers.** A **sigmoid** is an S-shape: flat, then a steep middle, then flat again. **Saturating** names the flat ends — past a point, more of the feature stops changing anything. In words: below a certain income essentially every household is at risk and further poverty adds no information; above a certain income essentially every household is fine; **all the model's discrimination happens in the middle**, which is exactly where [Phase 7](phase7.md) finds accuracy at its lowest and the decision hardest.
>
> The axis is in **log** rupees, so the numbers need translating: log-income 9.5 is about **₹13,000** a year, and 13.5 is about **₹730,000**. Each step of 1 on this axis multiplies income by roughly 2.7. Crucially, **the four area types lie on top of one another.** Income's effect does not differ by geography. Combined with Phase 1's monotone area gradient (metro 42.1% → village 26.7%), the implication is that **the area-type gradient is largely an income gradient** — metro households save more because they earn more, not because location independently changes savings behaviour.

**Right panel — grocery share. This is where I got it wrong, and the data corrected me.**

The naive expectation, and the caption I first wrote, was that a high food share pushes toward "not on track" — Engel's law says food share falls as income rises, so a high food share marks a poorer household. **The plot shows the exact opposite.** SHAP for `Groceries_Share` runs from roughly −3 at a 10% food share to about +1 at 80%, crossing zero near 0.45. **Conditional on income, a higher food share predicts being ON TRACK.**

**Why, mechanically.** `Log_Income` is already in the model, so the grocery share is no longer acting as a poverty proxy — that job is done. What remains is budget *shape*: a household whose spending is dominated by food is, by the closure constraint, spending proportionally little on everything else. The categories that fall are the lumpy, non-routine ones — healthcare shocks, durables, education fees, social functions — and those are exactly what push total consumption above income and destroy the savings rate. A low food share at a given income does not signal affluence; it signals **a large non-food outlay.**

This is consistent with the strongest interaction in the model (`Groceries_Share × Log_Income`, 0.166), with the colour gradient in the panel, and with the borderline worked example below, where a 66.3% grocery share contributes **+0.711 toward "on track"**.

**Why this matters beyond one chart.** Phase 1's univariate correlation (`Groceries_Share` vs log income, r = −0.266) is *real* and points the other way. Reading a conditional model's behaviour off an unconditional correlation would have produced a confident, backwards business recommendation — "target households with high food shares" is close to the opposite of the truth. The two views are not in conflict; they answer different questions, and only the conditional one describes what the model does.

> **In plain terms — conditional vs unconditional, the single most useful distinction in this project.** Two different questions, easily confused:
> - **Unconditional** (Phase 1): *across all households as they are*, do high food shares go with low income? Yes — because poorer households spend proportionally more on food. This is Engel's law.
> - **Conditional** (this phase): *among households earning roughly the same*, does a higher food share go with saving more? **Also yes** — because at a given income, a food-dominated budget means the absence of the lumpy healthcare, durables and school-fee outlays that wreck a savings rate.
>
> Both are correct. They differ because the first lets income vary and the second holds it fixed, and food share is doing two different jobs in the two settings — a marker of poverty in one, a marker of a simple budget in the other. Once income is already in the model, the first job is taken and only the second remains.
>
> The practical lesson is blunt: a business recommendation is always a conditional claim ("among comparable households, target these"), so grounding it in an unconditional correlation can invert it. That is precisely what nearly happened here, and it is why [Phase 7](phase7.md) benchmarks every household against peers *within its own income decile*.

### Cell 12 (code) — Individual explanations (Q3)

> **In plain terms — how to read these three examples.** Because SHAP is additive, one household's explanation is a short arithmetic story: start from the average household, then add each feature's push. **Positive numbers push toward "on track", negative numbers toward "at risk"**, and their size is how hard. `Log_Income` +6.737 means income alone moved this household a long way toward on-track; `Household_Size` −1.033 means a large family pulled another household back.
>
> **P** is the model's final probability after all the pushes are totalled and converted to a 0–1 scale. The three cases are deliberately chosen from the two extremes and the middle — a classifier's confident calls and its genuinely uncertain ones fail in different ways, and only the borderline case tests whether an explanation is actually usable.

**Most confident "on track"** — P = 100.0%, correct. Income ₹1,158,000/yr, 2 people, 0 dependents: `Log_Income` +6.737, `Household_Size` +0.929, `Groceries_Share` (0.465) +0.470. Plain language: *"very high income, small household, no dependents, and an ordinary food-centred budget."*

**Most confident "not on track"** — P = 0.0%, correct. Income ₹2,990/yr, 7 people, 4 dependents: `Log_Income` −5.653, `Miscellaneous_Share` (0.716) −1.160, `Debt_To_Income_W` at its winsorised ceiling of 10.234 −0.972.

**This example is also a data-quality exhibit.** A reported annual income of ₹2,990 — about ₹250 a month for seven people — is not credible as a true income. It is Phase 1's income under-reporting (55.9% of households report consumption exceeding income) showing up in a single record. The model is confidently right about the *label*, and the label is itself an artifact of measurement. Worth showing rather than quietly picking a tidier example: it is the clearest possible illustration of why Phase 0 restricted this project to relative prioritisation rather than absolute claims.

**Borderline case** — P = 50.0%, actual on track. Income ₹111,660/yr, 8 people, 2 dependents: `Household_Size` −1.033 against `Groceries_Share` (0.663) +0.711 and `Log_Income` +0.413. Plain language: *"a moderate income stretched across eight people, offset by a lean, food-dominated budget with no large discretionary outlays."* The competing contributions are legible, which is the practical test of whether an explanation is usable by a non-technical reader.

---

## What this changes for later phases

| Phase | Consequence |
| --- | --- |
| **6 — Clustering** | Personas built on spending mix explain at most ~27% of what drives the outcome. Expect them to segment on income more than on behaviour; say so rather than over-reading them. |
| **7 — Business translation** | Three constraints: (a) lead with income, which is 38.4% of attribution; (b) **a high grocery share is a positive signal conditional on income** — do not invert it from the Phase 1 correlation; (c) the area-type gradient is largely an income gradient, so geographic targeting is mostly income targeting by proxy. |
| **8 — Reporting** | `results/shap_summary.png` and `shap_dependence.png` are the explainability figures. The interaction result (41.9%) is the concrete reason a linear model was not selected. |
