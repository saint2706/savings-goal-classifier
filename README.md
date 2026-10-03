# Predicting Savings Adequacy — Indian Household Survey Data

> **Purpose of this document:** This README is written to be parsed by both humans and AI agents/LLMs. Every research question the project answers is listed in plain language under "Research Questions" **and** repeated in a machine-readable YAML manifest at the end of that section. An agent reading this file should be able to enumerate the full scope of the project, the target variable definition, the known data-leakage constraints, and the pipeline phases without needing to read any code.

## 1. Overview

This project uses the **India Human Development Survey-II** (IHDS-II, 2011-12; 41,518 households with income, category-level annual expenditure, and demographics) to answer two linked questions:

1. **Classification:** Can we predict whether a household retains an adequate share of its income after consumption — and from which features can we do so *without* reconstructing the label?
2. **Segmentation:** What spending clusters exist in the population, and how do they relate to savings adequacy?

The project is framed as a decision-support tool for a savings-product or financial-inclusion team deciding which households to prioritise for outreach.

**Headline result (post-audit, October 2026).** Spending composition is *not* leakage-free: household size × food share × income reconstructs total spend, and a model-free oracle built from those three columns ranks households at **ROC-AUC 0.895**. The headline model therefore uses only what is known at onboarding — income, demographics, debt, social group. Under cross-validation **grouped by primary sampling unit (PSU)**, tuned XGBoost reaches **ROC-AUC 0.882 / macro-F1 0.782** (held-out PSUs: **0.886 / 0.781**). Adding spending shares lifts that to 0.93, but mostly through the reconstruction route. At a 25% contact budget the model reaches **98.5%** of the attainable at-risk households against **95.4%** for "contact the poorest first" — **+1.1 pp (95% CI 1.0–1.3)**. See [§8 Results](#8-results) and [`TODO.md`](TODO.md) for the audit.

## 2. Dataset

**Source:** [India Human Development Survey-II (IHDS-II), 2011-12](https://www.icpsr.umich.edu/web/DSDR/studies/36151) — ICPSR study 36151, dataset DS0002 (Household)
**Rows:** 41,518 households in 2,461 PSUs (from 42,152 raw; 634 dropped for non-positive income or missing consumption)
**Grain:** One row per **household**, annual snapshot
**Money units:** annual ₹

The data is **not committed** — ICPSR's terms prohibit redistribution. [`dataset/README.md`](dataset/README.md) documents the download, the `sgc build` command, and the notebook order.

### Column reference

| Column | Description | Role |
| --- | --- | --- |
| `INCOME`, `Log_Income` | Total annual household income | Feature (headline) |
| `Household_Size`, `Age_Dependents`, `Dependency_Ratio` | Household composition; `Age_Dependents` = members aged 0–14 or 60+ | Feature (headline) |
| `Head_Age` | Age of household head (male head, else female head) | Feature (headline) |
| `Max_Adult_Education` | Highest adult education in the household | Feature (headline) |
| `Occupation` | Dominant worker type (salaried / business / farm / ag-labour / non-ag-labour / none); ties go to the more regular income stream | Feature (categorical) |
| `Area_Type` | Metro urban / other urban / developed village / less-developed village | Feature (categorical) |
| `Caste_Group`, `Religion` | Social group | Feature (categorical) |
| `Debt_To_Income`, `Has_Debt`, `Debt_Missing` | Outstanding debt as a multiple of income (winsorised at p99 inside the pipeline, per fold); a blank `DB5` is missing, not zero | Feature (headline) |
| `{category}_Share` (11) | Each expense category as a share of **total expenditure** | **Diagnostic only** — jointly with size and income it reconstructs the label (A1) |
| `Spends_On_{category}` (5) | Participation indicators for the majority-zero categories | Diagnostic only |
| `Has_*` (6) | Survey-reported holdings: bank savings, fixed deposit, pension/LIC, securities, post office, gold | **External validation only — never a feature** |
| 11 raw rupee categories, `Category_Total` | Groceries, Eating_Out, Utilities, Rent, Transport, Healthcare, Education, Entertainment, Insurance, Clothing_Footwear, Miscellaneous; their sum | **Leakage — excluded** |
| `COTOTAL`, `Savings`, `Savings_Rate` | Total consumption and the savings residual | **Leakage — excluded** |
| `IDPSU`, `WT` | Primary sampling unit (CV group key) and household survey weight | Grouping / population figures |

## 3. Target Variable

```text
Savings      = INCOME - COTOTAL
Savings_Rate = Savings / INCOME
Goal_Met     = 1 if Savings_Rate >= 0.20 else 0
```

A household is **on track** if it retains at least 20% of its annual income after all recorded consumption. Class balance: **31.9% positive** (2.13 : 1); **30.5%** survey-weighted.

The 20% threshold is a **convention, not a measurement**. It is configurable via `sgc build --threshold`, and [`walkthrough/phase1.md`](walkthrough/phase1.md) publishes the sensitivity curve (0% → 44.1% positive, 30% → 25.3%).

**⚠️ Data leakage constraints (read before modifying the feature set):**

- `Savings` is an exact identity over `INCOME` and `COTOTAL`. Our 11 categories sum to within 1% of `COTOTAL` for 97.7% of households, so the identity is approximate over the expense columns.
- The 11 **raw rupee** categories, `Category_Total`, `COTOTAL`, `Savings` and `Savings_Rate` must **never** be features: with `INCOME` they reconstruct `Goal_Met` with **99.75%** agreement. **Expense-to-income ratios** are the same quantity divided by income — also 99.75%.
- **Composition shares are not safe either (A1).** No single share predicts the label (largest marginal ROC-AUC distance from 0.5: 0.10), but food rupees track household size, so `COTOTAL ≈ k × Household_Size / Groceries_Share`. That oracle, with `k` = median food spend per person, ranks households at **ROC-AUC 0.895** with no learning. Shares may enter a deployed model only if observed *before* the prediction window — the scoring API enforces this. See [`results/leakage.json`](results/leakage.json).

## 4. Research Questions

Every question maps to one phase of the pipeline (Section 5). Questions are answered in order; later phases depend on decisions made in earlier ones.

### Phase 0 — Framing

- **What real business decision does this project inform?**
  Whether a savings-product or financial-inclusion team should treat a household as **on-track** or **at-risk** against a normative savings-adequacy benchmark, and therefore which households to prioritise for a low-cost intervention. The output drives a triage/prioritisation decision, not a savings forecast.
- **What is the precise, one-sentence definition of the target variable?**
  `Goal_Met = 1` if a household retains at least 20% of its annual income after all recorded consumption expenditure (see Section 3).
- **Is the primary task classification or regression, and why?**
  **Binary classification**, on three grounds: the business decision is binary; the continuous alternative (`Savings_Rate`, median −0.108, minimum −1647) is dominated by a long left tail from the least reliable part of the data; and the binary label is robust to the survey's measurement error. A robust (quantile) regression on `Savings_Rate` is now run as a complement in notebook 08.

### Phase 1 — Data Understanding

- What does each column mean, and what unit/time period does it represent?
- What is the distribution of income, expenses, and savings — skew, outliers, implausible values?
- Are there missing values or duplicate rows, and how are they handled?
- How correlated are expense categories with income and with each other?
- Is the target mathematically derivable from any candidate feature, singly or jointly (leakage check, A1)?
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

- What spending clusters emerge from clustering on expense-category proportions, and are they behavioural personas?
- How many clusters are statistically justified (elbow method, silhouette score)?
- Do the resulting personas correlate meaningfully with `Goal_Met`?

### Phase 7 — Business Translation

- What are the 3–5 most actionable findings, stated as recommendations rather than statistics?
- Which expense category carries the most recoverable spend across the population?
- Where does the model fail or lose reliability — what should a stakeholder be told before acting on it?

### Phase 8 — Reporting, and distance from adequacy

- Does the final write-up explain _reasoning_ rather than just reporting numbers?
- Which 3–4 visualisations communicate the findings fastest to a non-technical reader?
- Does ranking by predicted rupee shortfall (quantile regression on `Savings_Rate`) reach more of the gap than a classifier? (notebook 08)

### Machine-readable question manifest

```yaml
project: indian-household-savings-adequacy
revision: post-audit 2026-10 (see TODO.md)
dataset:
  name: India Human Development Survey-II (IHDS-II)
  source: ICPSR study 36151, dataset DS0002 (Household)
  rows: 41518
  psus: 2461
  grain: household
  money_units: annual INR
target_variable:
  name: Goal_Met
  definition: "1 if (INCOME - COTOTAL) / INCOME >= 0.20 else 0"
  threshold: 0.20
  threshold_is_convention: true
  positive_rate: 0.3193
  positive_rate_weighted: 0.3047
  derived_from: [INCOME, COTOTAL]
  leakage_excluded_features:
    [raw expense categories, Category_Total, COTOTAL, Savings, Savings_Rate, expense-to-income ratios]
  diagnostic_only_features:
    [11 composition shares, 5 participation indicators]
  reserved_for_validation:
    [Has_Securities, Has_Fixed_Deposit, Has_Bank_Savings, Has_Post_Office_Account,
     Has_Pension_LIC, Has_Gold_Jewellery]
validation: "StratifiedGroupKFold(5) grouped by IDPSU on a training split; one PSU-grouped fold (8,299 households, 472 PSUs) held out"
headline_model:
  features: [Log_Income, Household_Size, Age_Dependents, Dependency_Ratio, Head_Age,
             Max_Adult_Education, Debt_To_Income, Has_Debt, Debt_Missing,
             Occupation, Area_Type, Caste_Group, Religion]
  estimator: "XGBoost max_depth=4 learning_rate=0.08 n_estimators=200 subsample=0.7 min_child_weight=1"
  cv: {roc_auc: 0.882, macro_f1: 0.782, pr_auc_at_risk: 0.942, ece: 0.011}
  held_out: {roc_auc: 0.886, macro_f1: 0.781, pr_auc_at_risk: 0.944, brier: 0.127, ece: 0.009}
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
    answer: "Binary classification; a quantile regression on Savings_Rate complements it (notebook 08)."
  - id: P1-Q1
    phase: data_understanding
    text: "What does each column mean, and what unit/period does it represent?"
    answer: "All money is annual INR; the grain is the household. 54 columns: identifiers (incl. IDPSU), survey weight, features, a build diagnostic, external-validation, leakage-excluded, target."
  - id: P1-Q2
    phase: data_understanding
    text: "What is the distribution of income, expenses, and savings?"
    answer: "Extreme right skew (INCOME 15.8). Median savings rate -10.8%; 55.9% of households (57.7% weighted) report consumption exceeding income, a documented survey artifact."
  - id: P1-Q3
    phase: data_understanding
    text: "Are there missing values or duplicate rows, and how are they handled?"
    answer: "Debt is missing for 7.6% (kept missing, with Debt_Missing); other columns under 0.5%; zero duplicate households. Categorical gaps get an Unknown level."
  - id: P1-Q4
    phase: data_understanding
    text: "How correlated are expense categories with income and each other?"
    answer: "Raw categories Spearman 0.09-0.58 with income; Groceries_Share -0.280 with income (Engel's law)."
  - id: P1-Q5
    phase: data_understanding
    type: leakage_check
    text: "Is the target mathematically derivable from any candidate feature, singly or jointly?"
    answer: "Yes from raw categories and expense-to-income ratios (99.75%). No single share carries it, but size x food share x income does: a model-free oracle reaches ROC-AUC 0.895, and a model without shares keeps 94.3% of the full model's AUC. Decision: headline = income + demographics model."
  - id: P1-Q6
    phase: data_understanding
    text: "What is the class balance of Goal_Met once leakage columns are excluded?"
    answer: "31.93% positive, 2.13:1. Survey-weighted 30.47%."
  - id: P2-Q1
    phase: feature_engineering
    text: "Do expense-to-income ratios generalise better across income levels than raw values?"
    answer: "No - worst even after p99 winsorising. Transfer ROC-AUC: raw 0.628, shares 0.618, ratios 0.603."
  - id: P2-Q2
    phase: feature_engineering
    text: "How should categoricals be encoded and missing categories handled?"
    answer: "One-hot with an explicit Unknown level; missing caste is uninformative (0.333 vs 0.319, n=69)."
  - id: P2-Q3
    phase: feature_engineering
    text: "Should majority-zero categories get participation indicators?"
    answer: "Yes for the diagnostic full set: ROC-AUC 0.917 -> 0.919."
  - id: P2-Q4
    phase: feature_engineering
    text: "Which features require scaling, and does it depend on the model?"
    answer: "StandardScaler for linear models only; never RobustScaler on zero-inflated shares (IQR ~0 inflates values above 150). Trees unscaled."
  - id: P2-Q5
    phase: feature_engineering
    text: "Are any features redundant or highly collinear?"
    answer: "The 11 shares sum to 1 (VIF infinite); linear models drop Groceries_Share as reference. Age_Dependents VIF 8.9 with size."
  - id: P3-Q1
    phase: baseline
    text: "What does a majority-class or single-rule baseline achieve?"
    answer: "Majority: 0.681 accuracy, 0.405 macro-F1. Income threshold re-learned per fold (mean Rs 122,249/yr): 0.778 accuracy / 0.742 macro-F1."
  - id: P3-Q2
    phase: baseline
    text: "What does logistic regression achieve on income plus a few features?"
    answer: "Income ROC-AUC 0.835; + Groceries_Share 0.875; + size + Groceries_Share 0.894 (the reconstruction route); income + demographics 0.876."
  - id: P4-Q1
    phase: model_comparison
    text: "Which model families are appropriate?"
    answer: "Seven compared on both feature sets. Tuned XGBoost leads on the headline set (macro-F1 0.782, ROC-AUC 0.882); HistGradientBoosting within a fold sd."
  - id: P4-Q2
    phase: model_comparison
    text: "What validation strategy fits the class balance and the survey design?"
    answer: "Stratified 5-fold CV grouped by PSU on a training split; one grouped fold held out and evaluated once (ROC-AUC 0.886, macro-F1 0.781)."
  - id: P4-Q3
    phase: model_comparison
    text: "What hyperparameter search is used and what moves performance?"
    answer: "15-iteration randomised search per feature set; worth +0.007 macro-F1 on the headline set; max_depth matters most. Written to results/model_final.json."
  - id: P4-Q4
    phase: model_comparison
    text: "Is precision or recall more important?"
    answer: "Recall on the at-risk class. Tuned threshold gives held-out at-risk precision 0.826 / recall 0.913; 80% -> 95% recall costs ~10 points of precision."
  - id: P5-Q1
    phase: explainability
    text: "Which features matter most globally?"
    answer: "Income 58.0% of grouped SHAP (82% of grouped permutation importance); household size family 14.5%."
  - id: P5-Q2
    phase: explainability
    text: "Are there notable interaction effects?"
    answer: "31.6% of attribution over the full held-out set (headline); led by size x income. In the full model, food share x income leads - the reconstruction route."
  - id: P5-Q3
    phase: explainability
    text: "Can individual predictions be explained in plain business language?"
    answer: "Yes; shap.TreeExplainer values reproduce the raw margin to 6.5e-06."
  - id: P6-Q1
    phase: unsupervised
    text: "What spending clusters emerge, and are they personas?"
    answer: "Three, keyed on which categories are absent (ARI 0.858 vs the zero pattern): no transport (11.7%), no healthcare (18.6%), everything (69.7%). Descriptive segments, not personas."
  - id: P6-Q2
    phase: unsupervised
    text: "How many clusters are statistically justified?"
    answer: "k=3 by silhouette, Davies-Bouldin and Calinski-Harabasz within one ILR representation; bootstrap ARI 0.998; silhouette 0.39-0.48 depending on the zero-replacement delta."
  - id: P6-Q3
    phase: unsupervised
    text: "Do the clusters correlate meaningfully with Goal_Met?"
    answer: "Weakly unconditionally (Cramer's V 0.076); significant given income decile (LR 774 on 2 df, pseudo-R2 0.265 -> 0.280); nearly independent of income (NMI 0.012)."
  - id: P7-Q1
    phase: business_translation
    text: "What are the most actionable findings?"
    answer: "Five recommendations in results/business_recommendations.csv; lead: the model adds +1.1 pp (CI 1.0-1.3) of at-risk capture over a poorest-first rule at 25% (98.5% vs 95.4% of ceiling)."
  - id: P7-Q2
    phase: business_translation
    text: "Which expense category carries the most recoverable spend?"
    answer: "None demonstrably. Rupee excess is positive in every category by construction. In budget shares, at-risk households over-weight healthcare (+4.4pp), education (+2.8pp), miscellaneous (+2.1pp) and under-weight food (-9.0pp). The 'structural for 71.5%' claim is withdrawn: closable share is 22% or 93% depending on the benchmark."
  - id: P7-Q3
    phase: business_translation
    text: "Where does the model fail?"
    answer: "Accuracy falls to 0.66 in income decile 7. 32.3% of the at-risk group report spending more than twice their income."
  - id: P8-Q1
    phase: reporting
    text: "Does the write-up explain reasoning, not just results?"
    answer: "Yes - walkthrough/phase8.md and the TODO.md audit trail."
  - id: P8-Q2
    phase: reporting
    text: "Which visualisations communicate findings fastest?"
    answer: "project/figures/fig1-fig4, built from results/*.json by make_figures.py."
  - id: P8-Q3
    phase: reporting
    text: "Does ranking by predicted rupee shortfall beat the classifier?"
    answer: "On the rupee gap, yes: 45.5% of the total gap reached at a 25% budget vs 37.0% (classifier) and 28.7% (income rule)."
```

## 5. Methodology / Pipeline

```text
ICPSR DS0002 TSV → sgc build (Polars + Pandera) → dataset/households.parquet + features.parquet (with PSU-grouped Is_Test)
        → 01 EDA + leakage audit (A1) → 02 feature checks (train split) → 03 baselines
        → 04 model comparison (7 families × 2 feature sets, grouped CV, tuning → model_final.json)
        → 05 grouped SHAP + permutation + interactions → 06 ILR clustering
        → 07 business translation (capture vs ceiling, bootstrap CIs, peer benchmarks)
        → 08 quantile regression on Savings_Rate → make_figures.py → report.tex
```

All logic lives in the `savings_goal` package (`src/savings_goal/`): `data/` (build, schema), `features/` (engineering, transforms), `models/` (pipelines, clusters, regression, uplift), `evaluation/` (`cv.py`, `metrics.py`, `leakage.py`, `explain.py`, `business.py`), `cli.py` and `api.py`. Notebooks are thin callers that print aggregates and write `results/*.json`.

## 6. Repository Structure

```text
.
├── dataset/README.md                       (acquisition and build; data files are gitignored)
├── notebooks/01_… 08_*.ipynb               (thin callers of the package)
├── src/savings_goal/                       (the package: data, features, models, evaluation, cli, api)
├── tests/                                  (unit + integration tests on a synthetic DS0002)
├── results/                                (aggregate CSV / JSON / PNG artifacts only)
├── walkthrough/                            (phase-by-phase reasoning)
├── project/                                (report.tex, figures/make_figures.py)
├── .github/workflows/ci.yml                (ruff, ruff format, mypy --strict, pytest)
├── pyproject.toml / uv.lock
└── TODO.md                                 (the audit and its resolution)
```

## 7. Setup & Usage

```bash
git clone <repo-url> && cd savings-goal-classifier
uv sync                                                    # Python 3.11–3.13
uv run sgc build --tsv path/to/36151-0002-Data.tsv         # --tsv is required
for nb in notebooks/0*.ipynb; do
    uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=6000 "$nb"
done
uv run python project/figures/make_figures.py
```

Checks (what CI runs): `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest`.

**Scoring service.** `uv run sgc train` fits two tiers into `artifacts/` (gitignored). `uv sync --extra api && uv run sgc serve` exposes:

- `POST /score/onboarding` — tier 1, income + demographics + debt only;
- `POST /score/with-spending` — tier 2, adds spending shares, **rejected (422)** unless `shares_observed_through` is strictly before `prediction_window_start`.

Requests are validated by Pydantic and a Pandera schema (ranges, label sets, shares summing to one).

## 8. Results

**Headline model:** XGBoost on the deployable features (`max_depth=4`, `learning_rate=0.08`, `n_estimators=200`, `subsample=0.7`, `min_child_weight=1`). Grouped-CV macro-F1 **0.782**, ROC-AUC **0.882**; held-out PSUs (8,299 households): ROC-AUC **0.886**, macro-F1 **0.781**, at-risk PR-AUC 0.944 (base rate 0.681), Brier 0.127, ECE 0.009.

| Model (grouped CV, training split) | F1 (macro) | ROC-AUC | PR-AUC (at-risk) | ECE | ROC-AUC with shares |
| --- | --- | --- | --- | --- | --- |
| **XGBoost (tuned)** | **0.782** | **0.882** | 0.942 | 0.011 | 0.929 |
| HistGradientBoosting | 0.778 | 0.880 | 0.941 | 0.012 | 0.928 |
| XGBoost (untuned) | 0.774 | 0.875 | 0.938 | 0.023 | 0.929 |
| Logistic Regression | 0.773 | 0.876 | 0.938 | 0.103 | 0.920 |
| Linear SVM | 0.774 | 0.876 | 0.938 | — | 0.920 |
| Random Forest | 0.777 | 0.870 | 0.934 | 0.082 | 0.915 |
| Decision Tree | 0.751 | 0.850 | 0.914 | 0.104 | 0.882 |
| Single income threshold | 0.742 | 0.739 | 0.808 | 0.003 | — |
| Majority class | 0.405 | 0.500 | 0.681 | 0.319 | — |
| *Model-free oracle (size × food share × income)* | — | *0.895* | — | — | — |

**Leakage audit (A1):** the oracle beats every model that does not see spending shares; dropping all shares keeps 94.3% of the full model's AUC. Spending shares are diagnostic.

**Explainability:** income is **58.0%** of grouped SHAP (82% of grouped permutation importance), the household-size family 14.5%; interactions are 31.6% of attribution over the full held-out set. The earlier "a higher food share predicts being on track, conditional on income" is the reconstruction route, not an Engel reversal.

**Clusters:** k=3 on an ILR basis (all three indices agree; bootstrap ARI 0.998), keyed on absent categories (ARI 0.858 with the zero pattern). Significant given income decile (LR 774, 2 df), close to independent of income (NMI 0.012). Not personas.

**Business translation:** at a 25% budget, 98.5% of the attainable at-risk households vs 95.4% for poorest-first (+1.1 pp, CI 1.0–1.3; +2.7 pp at 50%). Precision 98.5% vs 95.4%. Accuracy dips to 0.66 in income decile 7. Peer benchmarks cannot separate structural from behavioural shortfall (22% vs 93% "closable" depending on construction); miscellaneous is the third-largest composition excess, not the lever. Ranking by predicted rupee shortfall reaches 45.5% of the total rupee gap at 25% (classifier 37.0%).

## 9. Limitations

- **Income under-reporting.** 55.9% of households report consumption exceeding income; 32.3% of the at-risk group report spending more than twice their income. Relative comparisons are sound; absolute prevalence figures are not.
- **Vintage.** 2011-12. Sound for methodology, not current for market sizing.
- **Household grain.** No per-person claims are supported.
- **Weights.** `WT` is applied to population figures (prevalence, weighted capture) but not to model fitting. Rupee totals in `results/` are sample totals unless labelled otherwise.
- **Spending shares leak jointly** with household size and income; they are usable only if measured before the outcome period.
- **Debt is a stock, not a flow.**
- **The at-risk class is the majority (68%)**, so capture must be read against its ceiling.
- **No intervention data.** IHDS records no outreach, so nothing here estimates who responds to a nudge; `savings_goal.models.uplift` (T-learner) is ready for campaign data.

## 10. License

Code in this repository: MIT License.
Data: not redistributed; see the [ICPSR terms of use](https://www.icpsr.umich.edu/web/ICPSR/support/terms) for study 36151.

## 11. Citation

Desai, Sonalde, Reeve Vanneman, and National Council of Applied Economic Research, New Delhi. *India Human Development Survey-II (IHDS-II), 2011-12.* Inter-university Consortium for Political and Social Research \[distributor\], 2018-08-08. <https://doi.org/10.3886/ICPSR36151.v6>

## 12. Course Submission Information

This project is submitted for **Advanced Machine Learning for Business Transformation (AMLBT)**, Goa Institute of Management — Big Data Analytics. [`project/main.tex`](project/main.tex) is the instructor-provided report template and is retained unedited for reference.

The report itself is [`project/report.tex`](project/report.tex), written to that template's structure (`pdflatex report.tex`, run twice). Its four figures are built by [`project/figures/make_figures.py`](project/figures/make_figures.py) from the aggregate files in `results/`, so they can be regenerated without the restricted microdata.

### Team

| Name | Student ID | Email |
| --- | --- | --- |
| Rishabh Agrawal | B2025100 | rishabh.agrawal25b@gim.ac.in |
| Prisha Kothari | B2026092 | prisha.kothari2026b@gim.ac.in |
| Akshit Kashyap | B2026059 | akshit.kashyap2026b@gim.ac.in |
