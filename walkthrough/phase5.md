# Phase 5 (IHDS-II) — Explainability

**Source:** [README § Phase 5 — Explainability](../README.md#phase-5--explainability)
**Notebook:** [`notebooks/05_explainability.ipynb`](../notebooks/05_explainability.ipynb)
**Builds on:** [Phase 4 (IHDS-II)](phase4.md), [Phase 2 (IHDS-II)](phase2.md)
**Artifacts:** `results/shap_importance.csv`, `results/shap_summary.png`, `results/shap_dependence.png`
**Replaces:** [`phase5.md`](phase5.md)

Phase 2 left an unexploded charge under this phase: the 11 expense shares sum to 1, so they are **exactly singular** (VIF = ∞) and no coefficient on them is identified. Phase 4 confirmed the logistic fit is only unique because of its L2 penalty. This phase deals with that first, then explains the model.

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

**3 of 11 shares flip sign across subsamples; 1 of 5 controls does.** `Log_Income`'s coefficient varies by 2.9% of its magnitude; `Insurance_Share`'s standard deviation is **1.8× its own mean**.

**But this test is weaker evidence than it looks, and that matters.** L2 regularisation resolves a singular system by picking the *minimum-norm* solution — and it picks the same tie-break every time. So coefficients can be perfectly **reproducible** across subsamples while still not being **identified**: the regulariser, not the data, is choosing how credit is split between collinear shares. Reproducibility is not identification, and a stability test alone cannot distinguish them. The algebra (a feature that is an exact linear combination of others has no unique coefficient) is the real argument; this table only shows the symptom leaking through where the penalty binds loosest.

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

---

## Notebook walkthrough

### Cell 1 (code) — Fit Phase 4's winning configuration on transformed data

The `ColumnTransformer` is applied explicitly and XGBoost fitted on the resulting frame, rather than wrapping both in a `Pipeline`. **Why:** SHAP needs to attribute to the *encoded* columns (50 of them, after one-hot expansion), and a pipeline hides those names. Test accuracy 0.8615 confirms this reproduces Phase 4.

### Cell 6 (code) — Global importance via native TreeSHAP

**An implementation note worth recording.** `shap.TreeExplainer` **fails on this model**: shap 0.49 cannot parse XGBoost 3.1's `base_score` serialisation (`could not convert string to float: '[3.1929308E-1]'`). The notebook uses XGBoost's own `pred_contribs=True` instead — the same TreeSHAP algorithm, computed inside XGBoost. It is verified rather than assumed: **SHAP values plus bias reproduce the raw model margin to a maximum absolute error of 8.58×10⁻⁶**, which is the additivity property TreeSHAP guarantees.

**Top features (mean |SHAP|):**

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

**This confirms Phase 3's prediction and completes the reversal of the synthetic project.** There, `Loan_Repayment_Ratio` dominated and raw income mattered far less than *how* income was spent. Here income is 38.4% of all attribution on its own, and Phase 3 showed a single income threshold already reaches macro-F1 0.7425. **Spending mix is real and worth 26.7%, but it is the second story, not the first.**

`Household_Size` at second place is new and was not visible in the univariate work — larger households consume more at any income, mechanically depressing the savings rate.

**How to read a share's SHAP value (the compositional caveat).** Because the shares sum to 1, a household cannot raise one without lowering others. So `Groceries_Share`'s attribution is never "the effect of spending more on food"; it is **"the effect of food taking a larger portion of the budget, and everything else a correspondingly smaller one."** Every sentence about a share in Phase 7 must carry that relative framing.

### Cell 9 (code) — Interactions (Q2)

Computed with `pred_interactions=True` on a 3,000-household sample (the full tensor is n × 50 × 50).

| Pair | mean \|interaction SHAP\| |
| --- | --- |
| `Groceries_Share` × `Log_Income` | **0.16632** |
| `Household_Size` × `Log_Income` | 0.10154 |
| `Utilities_Share` × `Log_Income` | 0.08910 |
| `Groceries_Share` × `Utilities_Share` | 0.04021 |
| `Max_Adult_Education` × `Log_Income` | 0.03561 |

Main effects total 6.509; interactions total 4.689 — **interactions are 41.9% of all attribution.**

**Why this is a substantive finding, not a technical footnote.** The synthetic project's winning model was a *linear SVM*, which is provably incapable of representing any interaction — a limitation its Phase 5 had to acknowledge. Here, more than two-fifths of the model's behaviour is interaction, and **eight of the ten strongest pairs involve `Log_Income`**. Income is not simply an additive term: it changes what every spending signal means. A high grocery share means something different for a household earning ₹40,000 than for one earning ₹400,000.

This also explains why Phase 4's logistic regression trailed XGBoost on macro-F1 (0.8186 vs 0.8371) while nearly matching it on ROC-AUC (0.9213 vs 0.9306) — a linear model captures the ranking well but misses the interaction structure that sharpens the decision boundary.

**One caveat on the 41.9% figure:** TreeSHAP interaction values are defined relative to the fitted tree ensemble, so this measures how much *this model* relies on interactions, not how much interaction exists in the population. Phase 4 found `max_depth=4` optimal with very low sensitivity, so the model has limited capacity for deep interaction — the true figure could be higher.

### Cell 11 (code) — The dependence plots, and a finding that reverses the naive reading

**Left panel — income.** The SHAP curve is monotone increasing and sigmoid-shaped, saturating below log-income ≈ 9.5 and above ≈ 13.5. Crucially, **the four area types lie on top of one another.** Income's effect does not differ by geography. Combined with Phase 1's monotone area gradient (metro 42.1% → village 26.7%), the implication is that **the area-type gradient is largely an income gradient** — metro households save more because they earn more, not because location independently changes savings behaviour.

**Right panel — grocery share. This is where I got it wrong, and the data corrected me.**

The naive expectation, and the caption I first wrote, was that a high food share pushes toward "not on track" — Engel's law says food share falls as income rises, so a high food share marks a poorer household. **The plot shows the exact opposite.** SHAP for `Groceries_Share` runs from roughly −3 at a 10% food share to about +1 at 80%, crossing zero near 0.45. **Conditional on income, a higher food share predicts being ON TRACK.**

**Why, mechanically.** `Log_Income` is already in the model, so the grocery share is no longer acting as a poverty proxy — that job is done. What remains is budget *shape*: a household whose spending is dominated by food is, by the closure constraint, spending proportionally little on everything else. The categories that fall are the lumpy, non-routine ones — healthcare shocks, durables, education fees, social functions — and those are exactly what push total consumption above income and destroy the savings rate. A low food share at a given income does not signal affluence; it signals **a large non-food outlay.**

This is consistent with the strongest interaction in the model (`Groceries_Share × Log_Income`, 0.166), with the colour gradient in the panel, and with the borderline worked example below, where a 66.3% grocery share contributes **+0.711 toward "on track"**.

**Why this matters beyond one chart.** Phase 1's univariate correlation (`Groceries_Share` vs log income, r = −0.266) is *real* and points the other way. Reading a conditional model's behaviour off an unconditional correlation would have produced a confident, backwards business recommendation — "target households with high food shares" is close to the opposite of the truth. The two views are not in conflict; they answer different questions, and only the conditional one describes what the model does.

### Cell 12 (code) — Individual explanations (Q3)

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
| **8 — Reporting** | `results/shap_summary.png` and `shap_dependence.png` replace the synthetic SHAP figure. The interaction result (41.9%) is the concrete reason a linear model was not selected. |
