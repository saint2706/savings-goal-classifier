# Predicting Savings Adequacy & Spending Personas — Indian Household Survey Data

> **Purpose of this document:** This README is written to be parsed by both humans and AI agents/LLMs. Every research question the project answers is listed in plain language under "Research Questions" **and** repeated in a machine-readable YAML manifest at the end of that section. An agent reading this file should be able to enumerate the full scope of the project, the target variable definition, the known data-leakage constraint, and the pipeline phases without needing to read any code.

## 1. Overview

This project uses the **India Human Development Survey-II** (IHDS-II, 2011-12; 41,518 households with income, category-level annual expenditure, and demographics) to answer two linked questions:

1. **Classification:** Can we predict — from income, occupation, area type, and spending composition — whether a household retains an adequate share of its income after consumption?
2. **Segmentation:** What natural spending personas exist in the population, and how do they relate to savings adequacy?

The project is framed as a decision-support tool for a savings-product or financial-inclusion team deciding which households to prioritise for outreach.

**Headline result:** XGBoost reaches cross-validated macro-F1 **0.8371** / ROC-AUC **0.9306**, confirmed on a held-out test set at macro-F1 **0.838**. The honest margin is smaller than that sounds — a single income threshold already reaches macro-F1 0.7425, and the model beats a plain "contact the poorest first" rule by only **1.5 percentage points** of at-risk capture. See [§8 Results](#8-results).

## 2. Dataset

**Source:** [India Human Development Survey-II (IHDS-II), 2011-12](https://www.icpsr.umich.edu/web/DSDR/studies/36151) — ICPSR study 36151, dataset DS0002 (Household)
**Rows:** 41,518 households (from 42,152 raw; 634 dropped for non-positive income or missing consumption)
**Grain:** One row per **household**, annual snapshot
**Money units:** annual ₹

The data is **not committed** — ICPSR's terms prohibit redistribution. [`dataset/README.md`](dataset/README.md) documents the download, the build command, and the notebook execution order. A fresh clone can reproduce every artifact in this repository in roughly 30 minutes.

### Column reference

| Column | Description | Role |
| --- | --- | --- |
| `INCOME`, `Log_Income` | Total annual household income | Feature |
| `Household_Size`, `Dependents`, `Dependency_Ratio` | Household composition | Feature |
| `Head_Age` | Age of household head (male head, else female head) | Feature |
| `Max_Adult_Education` | Highest adult education in the household | Feature |
| `Occupation` | Dominant worker type (salaried / business / farm / ag-labour / non-ag-labour / none) | Feature (categorical) |
| `Area_Type` | Metro urban / other urban / developed village / less-developed village | Feature (categorical) |
| `Caste_Group`, `Religion` | Social group | Feature (categorical) |
| `Debt_To_Income_W`, `Has_Debt` | Outstanding debt as a multiple of income, winsorised at p99 | Feature |
| `{category}_Share` (11) | Each expense category as a share of **total expenditure** | Feature |
| `Spends_On_{category}` (5) | Participation indicators for the majority-zero categories | Feature |
| `Has_*` (6) | Survey-reported holdings: bank savings, fixed deposit, pension/LIC, securities, post office, gold | **External validation only — never a feature** |
| 11 raw rupee expense categories | Groceries, Eating_Out, Utilities, Rent, Transport, Healthcare, Education, Entertainment, Insurance, Clothing_Footwear, Miscellaneous | Used to derive the target — **excluded from feature set** |
| `COTOTAL`, `Savings`, `Savings_Rate` | Total consumption and the savings residual | Used to derive the target — **excluded from feature set** |

## 3. Target Variable

```text
Savings      = INCOME - COTOTAL
Savings_Rate = Savings / INCOME
Goal_Met     = 1 if Savings_Rate >= 0.20 else 0
```

A household is **on track** if it retains at least 20% of its annual income after all recorded consumption. Class balance: **31.9% positive** (2.13 : 1).

The 20% threshold is a **convention, not a measurement**. It is configurable via `--threshold` in `src/build_dataset.py`, and [`walkthrough/phase1.md`](walkthrough/phase1.md) publishes the full sensitivity curve (0% → 44.1% positive, 30% → 25.3%). Findings quoted in this repository hold across 10–30%; absolute levels do not.

**⚠️ Data leakage constraint (read before modifying the feature set):**
`Savings` is an exact accounting identity over the expense columns — verified to 5.8×10⁻¹¹. Consequently:

- The 11 **raw rupee** categories, plus `COTOTAL`, `Savings` and `Savings_Rate`, must **never** be features. With `INCOME` they reconstruct `Goal_Met` with **99.75%** agreement.
- **Expense-to-income ratios are equally unusable** — the same 99.75%, because they are the same quantity divided through by income.
- The legitimate representation is each category's **share of total expenditure**. The 11 shares sum to exactly 1, so they carry no information about consumption relative to income. The strongest correlation between any share and `Savings_Rate` is 0.056.

## 4. Research Questions

Every question maps to one phase of the pipeline (Section 5). Questions are answered in order; later phases depend on decisions made in earlier ones.

### Phase 0 — Framing

- **What real business decision does this project inform?**
  Whether a savings-product or financial-inclusion team should treat a household as **on-track** or **at-risk** against a normative savings-adequacy benchmark, and therefore which households to prioritise for a low-cost intervention. The output drives a triage/prioritisation decision, not a savings forecast.
- **What is the precise, one-sentence definition of the target variable?**
  `Goal_Met = 1` if a household retains at least 20% of its annual income after all recorded consumption expenditure (see Section 3).
- **Is the primary task classification or regression, and why?**
  **Binary classification**, on three grounds: the business decision is binary; the continuous alternative (`Savings_Rate`, median −0.108, minimum −1647) is dominated by a long left tail from the least reliable part of the data; and the binary label is robust to exactly the measurement error this survey suffers from. A robust regression on `Savings_Rate` is the clearest follow-up this project leaves open.

### Phase 1 — Data Understanding

- What does each column mean, and what unit/time period does it represent?
- What is the distribution of income, expenses, and savings — skew, outliers, implausible values?
- Are there missing values or duplicate rows, and how are they handled?
- How correlated are expense categories with income and with each other?
- Is the target mathematically derivable from any candidate feature (leakage check)?
- What is the class balance of `Goal_Met` once leakage columns are excluded?

### Phase 2 — Feature Engineering

- Do expense-to-income ratios generalise better across income levels than raw expense values?
- How should the categorical features be encoded, and how should missing categories be handled?
- Should the majority-zero expense categories get explicit participation indicators?
- Which features require scaling, and does that depend on the downstream model?
- Are any features redundant or highly collinear?

### Phase 3 — Baseline

- What accuracy/F1 does a majority-class or simple single-rule baseline achieve?
- What does plain logistic regression achieve using only income and 1–2 expense shares?

### Phase 4 — Model Comparison

- Which 5–7 model families are appropriate given the data (n=41,518, mixed numeric/categorical, moderate dimensionality)?
- What validation strategy fits the class balance found in Phase 1?
- What hyperparameter search method is used, and what parameters move performance most?
- Given the business framing, is precision or recall more important — is a missed at-risk household more costly than an unnecessary contact?

### Phase 5 — Explainability

- Which features matter most globally for the winning model (SHAP)?
- Are there notable interaction effects (e.g. does income change how much a spending signal matters)?
- Can individual predictions be explained in plain business language?

### Phase 6 — Unsupervised Extension

- What spending personas emerge from clustering on expense-category proportions?
- How many clusters are statistically justified (elbow method, silhouette score)?
- Do the resulting personas correlate meaningfully with `Goal_Met`?

### Phase 7 — Business Translation

- What are the 3–5 most actionable findings, stated as recommendations rather than statistics?
- Which expense category carries the most recoverable spend across the population?
- Where does the model fail or lose reliability — what should a stakeholder be told before acting on it?

### Phase 8 — Reporting

- Does the final write-up explain _reasoning_ rather than just reporting numbers?
- Which 3–4 visualisations communicate the findings fastest to a non-technical reader?

### Machine-readable question manifest

```yaml
project: indian-household-savings-adequacy
dataset:
  name: India Human Development Survey-II (IHDS-II)
  source: ICPSR study 36151, dataset DS0002 (Household)
  rows: 41518
  grain: household
  money_units: annual INR
target_variable:
  name: Goal_Met
  definition: "1 if (INCOME - COTOTAL) / INCOME >= 0.20 else 0"
  threshold: 0.20
  threshold_is_convention: true
  positive_rate: 0.3193
  derived_from: [INCOME, COTOTAL]
  leakage_excluded_features:
    [raw expense categories, COTOTAL, Savings, Savings_Rate, expense-to-income ratios]
  reserved_for_validation:
    [Has_Securities, Has_Fixed_Deposit, Has_Bank_Savings, Has_Post_Office_Account,
     Has_Pension_LIC, Has_Gold_Jewellery]
questions:
  - id: P0-Q1
    phase: framing
    text: "What real business decision does this project inform?"
    answer: "Whether to treat a household as on-track or at-risk against a normative savings-adequacy benchmark, to prioritise outreach."
  - id: P0-Q2
    phase: framing
    text: "What is the precise definition of the target variable?"
    answer: "Goal_Met = 1 if the annual savings rate is at least 20%."
  - id: P0-Q3
    phase: framing
    text: "Is the primary task classification or regression, and why?"
    answer: "Binary classification: the decision is binary, the continuous target is dominated by an unreliable left tail, and the binary label is robust to the survey's income under-reporting."
  - id: P1-Q1
    phase: data_understanding
    text: "What does each column mean, and what unit/period does it represent?"
    answer: "All money is annual INR; the grain is the household. 50 columns across 6 roles: identifiers, survey weight, features, external-validation, leakage-excluded, target."
  - id: P1-Q2
    phase: data_understanding
    text: "What is the distribution of income, expenses, and savings?"
    answer: "Extreme right skew (INCOME 15.8, Clothing_Footwear 112.3). Median savings rate -10.8%; 55.9% of households report consumption exceeding income, a documented survey artifact."
  - id: P1-Q3
    phase: data_understanding
    text: "Are there missing values or duplicate rows, and how are they handled?"
    answer: "Nine columns under 0.5% missing; zero duplicate households. Categorical gaps get an explicit Unknown level."
  - id: P1-Q4
    phase: data_understanding
    text: "How correlated are expense categories with income and each other?"
    answer: "Weakly: raw categories correlate 0.09-0.43 with income, mean |r| between categories 0.143. Groceries_Share correlates -0.266 with log income (Engel's law)."
  - id: P1-Q5
    phase: data_understanding
    type: leakage_check
    text: "Is the target mathematically derivable from any candidate feature?"
    answer: "Yes from raw categories and from expense-to-income ratios (both 99.75% agreement); no from composition shares, which sum to exactly 1."
  - id: P1-Q6
    phase: data_understanding
    text: "What is the class balance of Goal_Met once leakage columns are excluded?"
    answer: "31.93% positive, 2.13:1. Survey-weighted 30.47%."
  - id: P2-Q1
    phase: feature_engineering
    text: "Do expense-to-income ratios generalise better across income levels than raw values?"
    answer: "No - they generalise worst. Cross-income transfer ROC-AUC: raw rupees 0.6245, composition shares 0.6175, expense/income ratios 0.5893."
  - id: P2-Q2
    phase: feature_engineering
    text: "How should categoricals be encoded and missing categories handled?"
    answer: "One-hot with an explicit Unknown level; all four are low-cardinality (4-8 levels). Missingness is not informative."
  - id: P2-Q3
    phase: feature_engineering
    text: "Should majority-zero categories get participation indicators?"
    answer: "Yes. ROC-AUC 0.9183 -> 0.9204. Spends_On_Insurance +9.7pp and Spends_On_Education -8.3pp univariate."
  - id: P2-Q4
    phase: feature_engineering
    text: "Which features require scaling, and does it depend on the model?"
    answer: "Barely matters (0.9212 / 0.9212 / 0.9211 for Robust / Standard / none) despite a 658x spread in standard deviations. RobustScaler retained for coefficient comparability."
  - id: P2-Q5
    phase: feature_engineering
    text: "Are any features redundant or highly collinear?"
    answer: "Yes, structurally: the 11 shares sum to 1 so VIF is infinite. Household_Size x Dependents r=0.714."
  - id: P3-Q1
    phase: baseline
    text: "What does a majority-class or single-rule baseline achieve?"
    answer: "Majority: 0.6807 accuracy but 0.4050 macro-F1. A single income threshold (> Rs 121,685/yr) reaches 0.7753 accuracy / 0.7425 macro-F1."
  - id: P3-Q2
    phase: baseline
    text: "What does logistic regression achieve on income plus 1-2 expense shares?"
    answer: "Income alone ROC-AUC 0.8348; plus Groceries_Share 0.8756; all 28 features 0.9212."
  - id: P4-Q1
    phase: model_comparison
    text: "Which model families are appropriate?"
    answer: "Seven compared. XGBoost wins at CV macro-F1 0.8371 / ROC-AUC 0.9306; HistGradientBoosting is statistically tied."
  - id: P4-Q2
    phase: model_comparison
    text: "What validation strategy fits the class balance?"
    answer: "5-fold stratified CV on an 80% split, with a 20% test set held out and evaluated once. Test macro-F1 0.838 matches the CV estimate exactly."
  - id: P4-Q3
    phase: model_comparison
    text: "What hyperparameter search is used and what moves performance?"
    answer: "15-iteration randomised search; worth +0.0006 macro-F1. learning_rate matters most (sd 0.0040), max_depth least (0.0003)."
  - id: P4-Q4
    phase: model_comparison
    text: "Is precision or recall more important?"
    answer: "Recall on the at-risk class. Reaching 95% recall costs only ~8 points of precision (0.941 -> 0.856)."
  - id: P5-Q1
    phase: explainability
    text: "Which features matter most globally?"
    answer: "Log_Income at 38.4% of attribution (mean |SHAP| 2.617, 4.5x the next feature); spending mix 26.7%."
  - id: P5-Q2
    phase: explainability
    text: "Are there notable interaction effects?"
    answer: "Yes - 41.9% of total attribution. Eight of the ten strongest pairs involve Log_Income."
  - id: P5-Q3
    phase: explainability
    text: "Can individual predictions be explained in plain business language?"
    answer: "Yes; SHAP is additive and was verified against the raw model margin to 8.6e-06."
  - id: P6-Q1
    phase: unsupervised
    text: "What spending personas emerge?"
    answer: "Three, defined by which core categories are ABSENT (adjusted Rand index 0.858 vs the raw zero-pattern), not by how present categories are allocated."
  - id: P6-Q2
    phase: unsupervised
    text: "How many clusters are statistically justified?"
    answer: "k=3 on a 6-part ILR basis, silhouette 0.4115 - but largely manufactured by zero replacement."
  - id: P6-Q3
    phase: unsupervised
    text: "Do the personas correlate meaningfully with Goal_Met?"
    answer: "Cramer's V is a weak 0.076 unconditionally, but the effect nearly doubles within income deciles (spread 0.167 vs 0.092) - income suppresses it."
  - id: P7-Q1
    phase: business_translation
    text: "What are the most actionable findings?"
    answer: "Five recommendations in results/business_recommendations.csv; the lead finding is that the model beats an income rule by only 1.5pp of at-risk capture."
  - id: P7-Q2
    phase: business_translation
    text: "Which expense category carries the most recoverable spend?"
    answer: "Miscellaneous (Rs 36 crore/yr excess vs same-income peers). Healthcare and Education rank next but are non-discretionary. Groceries runs the wrong way: at-risk households spend 8.8pp LESS on food."
  - id: P7-Q3
    phase: business_translation
    text: "Where does the model fail?"
    answer: "Accuracy sags to 0.78-0.80 in income deciles 5-7 where targeting is contested. 32.3% of the at-risk group report spending more than twice their income."
  - id: P8-Q1
    phase: reporting
    text: "Does the write-up explain reasoning, not just results?"
    answer: "Yes - walkthrough/phase8.md carries a reasoning trail mapping each decision to the evidence that forced it."
  - id: P8-Q2
    phase: reporting
    text: "Which visualisations communicate findings fastest?"
    answer: "phase1_eda.png, shap_summary.png, business_translation.png, personas.png."
```

## 5. Methodology / Pipeline

```text
ICPSR DS0002 → src/build_dataset.py → dataset/households.csv
        → Phase 1 (EDA + leakage check) → Phase 2 (feature engineering → features.csv)
        → Phase 3 (baseline) → Phase 4 (model comparison, 7 families, CV + tuning)
        → Phase 5 (SHAP explainability) → Phase 6 (ILR clustering / personas)
        → Phase 7 (business translation) → Phase 8 (final report)
```

## 6. Repository Structure

```text
.
├── dataset/
│   ├── README.md                           (acquisition, build, run order)
│   ├── households.csv                      (built locally; gitignored)
│   └── features.csv                        (built locally; gitignored)
├── notebooks/
│   ├── 01_eda_and_leakage_check.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_model_comparison.ipynb
│   ├── 05_explainability.ipynb
│   ├── 06_clustering_personas.ipynb
│   └── 07_business_translation.ipynb
├── src/
│   └── build_dataset.py
├── results/
│   ├── phase1_eda.png
│   ├── phase1_share_correlations.png
│   ├── baseline.csv / baseline.png
│   ├── model_comparison.csv / model_comparison.png
│   ├── shap_summary.png / shap_dependence.png / shap_importance.csv
│   ├── cluster_selection.png
│   ├── persona_profiles.csv / personas.png
│   └── business_recommendations.csv / business_translation.png
├── walkthrough/
│   ├── dataset_construction.md
│   └── phase0.md … phase8.md
├── project/
│   └── main.tex                            (instructor's report template)
├── README.md
└── requirements.txt
```

Each `walkthrough/phaseN.md` answers that phase's research questions and, for phases with a notebook, walks through it cell by cell — what each cell does and the motivation behind it — so the reasoning behind the code doesn't have to be reconstructed from the code alone. Notebooks themselves carry only section headers and code; all narrative explanation lives in the matching walkthrough document.

## 7. Setup & Usage

```bash
git clone <repo-url>
cd savings-goal-classifier
pip install -r requirements.txt
```

Then follow [`dataset/README.md`](dataset/README.md) to obtain the IHDS-II source data and build `households.csv` and `features.csv`. Once built:

```bash
cd notebooks
for nb in 0*.ipynb; do
    jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3000 "$nb"
done
```

## 8. Results

**Winning model:** XGBoost (`max_depth=4`, `learning_rate=0.15`, `n_estimators=200`, `min_child_weight=5`, `subsample=0.9`) — CV macro-F1 **0.8371**, ROC-AUC **0.9306**. Held-out test (8,304 households): macro-F1 **0.838**, identical to the CV estimate. Hyperparameter search was worth **+0.0006** macro-F1.

| Model | CV Accuracy | CV F1 (macro) | CV ROC-AUC |
| --- | --- | --- | --- |
| **XGBoost** | 0.8614 | **0.8371** | 0.9306 |
| HistGradientBoosting | 0.8606 | 0.8360 | 0.9306 |
| Random Forest | 0.8512 | 0.8249 | 0.9170 |
| Logistic Regression | 0.8343 | 0.8186 | 0.9213 |
| Linear SVM | 0.8332 | 0.8177 | — |
| Decision Tree | 0.8067 | 0.7889 | 0.8847 |
| Single income rule (Phase 3) | 0.7753 | 0.7425 | 0.7438 |
| Majority baseline | 0.6807 | 0.4050 | 0.5000 |

**The honest margin is +0.095 macro-F1 over a single income threshold** (predict "on track" if annual income > ₹121,685), not +0.432 over the majority baseline.

**Explainability (Phase 5):** `Log_Income` is **38.4%** of all attribution (mean |SHAP| 2.617, 4.5× the next feature); spending mix 26.7%. **Interactions are 41.9% of total attribution**, and 8 of the 10 strongest pairs involve income — the reason a linear model was not selected.

**Key findings:**

- **Income dominates.** `Log_Income` alone reaches ROC-AUC 0.835; adding one feature (`Groceries_Share`) reaches 0.876 against the full model's 0.921.
- **A high grocery share predicts being _on track_, conditional on income** — the reverse of the naive Engel reading. Once income is controlled, a food-dominated budget signals the *absence* of large lumpy outlays (health shocks, durables, fees) that push consumption above income.
- **Geography is largely income by proxy.** Goal attainment falls monotonically metro → village (0.4212 / 0.3698 / 0.3091 / 0.2666), but income's SHAP effect is near-identical across area types.
- **Occupation is real signal:** salaried households meet the goal at 1.76× the rate of agricultural labourers (0.4410 vs 0.2512).
- **Rent is absent for 90.5% of households** — most own their homes, so any rent-based analysis describes a ~10% urban subsample.

**Spending personas (Phase 6):** k=3 on a 6-part ILR basis, silhouette **0.4115** — but the clusters are keyed on *which categories are absent* (adjusted Rand index 0.858 against the raw zero-pattern), largely an artifact of zero replacement. Their association with `Goal_Met` looks weak unconditionally (Cramér's V 0.076) yet **nearly doubles once income is held constant** (within-decile spread 0.167 vs 0.092). Personas are nearly independent of income (ARI 0.006).

**Business translation (Phase 7):** the model beats a plain income rule by only **1.5 percentage points** of at-risk capture at a 25% contact budget (36.6% vs 35.0%); its real value is precision (99.6% vs 68.3% random). Only **28.5%** of at-risk households could close their gap even by matching the spending of on-track peers at the same income — for the rest the shortfall is structural. Full stakeholder write-up in [`walkthrough/phase8.md`](walkthrough/phase8.md).

## 9. Limitations

- **Income under-reporting.** 55.9% of households report consumption exceeding income — a documented property of Indian household surveys, where income is recalled poorly and consumption is captured item by item. This biases `Goal_Met` downward at every threshold. **Relative comparisons are sound; absolute prevalence figures are not.** 32.3% of the at-risk group report spending more than twice their income.
- **Vintage.** 2011-12. Sound for methodology, not current for market sizing. IHDS-3 fieldwork is complete but public microdata was not released as of this work.
- **Household grain.** One row is a household, not an individual. No per-person claims are supported.
- **Survey weights carried but not applied.** `WT` is in the dataset; model fitting is unweighted. Any nationally-framed figure must apply it.
- **The 11 shares are exactly singular** (they sum to 1, so VIF = ∞). Regularised and tree models are unaffected, but individual share coefficients are not identified — Phase 5 uses SHAP on the tree model for this reason, and an unregularised linear model must not be fitted on the full share set.
- **Debt is a stock, not a flow.** IHDS records outstanding debt and interest rates but no monthly repayment amount, so `Debt_To_Income_W` stands in for a cash-flow burden it cannot directly measure.
- **The at-risk class is the majority (68%)**, so lift-based business cases are structurally weak here regardless of model quality.
- **Personas are partly a zero-replacement artifact** rather than discovered behavioural archetypes.

## 10. License

Code in this repository: MIT License.
Data: not redistributed; see the [ICPSR terms of use](https://www.icpsr.umich.edu/web/ICPSR/support/terms) for study 36151.

## 11. Citation

Desai, Sonalde, Reeve Vanneman, and National Council of Applied Economic Research, New Delhi. *India Human Development Survey-II (IHDS-II), 2011-12.* Inter-university Consortium for Political and Social Research \[distributor\], 2018-08-08. <https://doi.org/10.3886/ICPSR36151.v6>

## 12. Course Submission Information

This project is submitted for **Advanced Machine Learning for Business Transformation (AMLBT)**, Goa Institute of Management — Big Data Analytics. [`project/main.tex`](project/main.tex) is the instructor-provided report template (submission requirements, rubric, prompts) and is retained unedited for reference.

### Team

| Name | Student ID | Email |
| --- | --- | --- |
| Rishabh Agrawal | B2025100 | rishabh.agrawal25b@gim.ac.in |
| Prisha Kothari | B2026092 | prisha.kothari2026b@gim.ac.in |
| Akshit Kashyap | B2026059 | akshit.kashyap2026b@gim.ac.in |
