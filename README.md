# Predicting Savings-Goal Attainment & Spending Personas — Indian Personal Finance

> **Purpose of this document:** This README is written to be parsed by both humans and AI agents/LLMs. Every research question the project answers is listed in plain language under "Research Questions" **and** repeated in a machine-readable YAML manifest at the end of that section. An agent reading this file should be able to enumerate the full scope of the project, the target variable definition, the known data-leakage constraint, and the pipeline phases without needing to read any code.

## 1. Overview

This project uses the **Indian Personal Finance and Spending Habits** dataset (20,000 individuals; income, category-level monthly expenses, and savings goals) to answer two linked questions:

1. **Classification:** Can we predict — from income, occupation, city tier, and spending pattern alone — whether an individual is on track to meet their own stated savings goal?
2. **Segmentation:** What natural spending personas exist in the population, and how do they relate to savings-goal attainment?

The project is framed as a decision-support tool for a hypothetical fintech / savings-product marketing team deciding which customer segments to target.

## 2. Dataset

**Source:** [Indian Personal Finance and Spending Habits](https://www.kaggle.com/datasets/shriyashjagtap/indian-personal-finance-and-spending-habits) (Kaggle, uploader: shriyashjagtap)
**Rows:** 20,000 individuals
**Grain:** One row per individual, monthly snapshot

### Column reference

| Column                                                                                                                                                  | Description                                           | Role                                                  |
| ------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------- |
| `Income`                                                                                                                                                | Monthly income, currency units (verify ₹ in raw file) | Feature                                               |
| `Age`                                                                                                                                                   | Age of individual                                     | Feature                                               |
| `Dependents`                                                                                                                                            | Number of dependents                                  | Feature                                               |
| `Occupation`                                                                                                                                            | Employment / job category                             | Feature (categorical)                                 |
| `City_Tier`                                                                                                                                             | Tier 1 / Tier 2 / Tier 3 living area                  | Feature (categorical)                                 |
| `Rent`, `Loan_Repayment`, `Insurance`, `Groceries`, `Transport`, `Eating_Out`, `Entertainment`, `Utilities`, `Healthcare`, `Education`, `Miscellaneous` | Monthly spend per category                            | Features                                              |
| `Desired_Savings_Percentage`                                                                                                                            | Target % of income the person wants to save           | Used to derive target — **excluded from feature set** |
| `Desired_Savings`                                                                                                                                       | Target absolute monthly savings amount                | Used to derive target — **excluded from feature set** |
| `Disposable_Income`                                                                                                                                     | Income minus all expenses                             | Used to derive target — **excluded from feature set** |
| `Potential_Savings_*` (8 columns, one per expense category)                                                                                             | Estimated recoverable savings per category            | Feature / exploratory use                             |

## 3. Target Variable

```text
Goal_Met = 1 if Disposable_Income >= Desired_Savings else 0
Goal_Met = 0 otherwise
```

**⚠️ Data leakage constraint (read before modifying the feature set):**
`Disposable_Income` and `Desired_Savings` are used to _construct_ `Goal_Met`. They must **never** be included as model input features — doing so lets the model trivially reconstruct the label rather than predict it, producing artificially perfect accuracy that has no real-world meaning. The legitimate feature set is limited to columns known independently of the target: `Income`, `Age`, `Dependents`, `Occupation`, `City_Tier`, and the raw expense category columns (optionally converted to income ratios).

## 4. Research Questions

Every question below maps to one phase of the project pipeline (Section 5). Questions are answered in order; later phases depend on decisions made in earlier ones.

### Phase 0 — Framing

- **What real business decision does this project inform?**
  Whether a fintech / savings-product marketing team should treat an individual as **on-track** or **at-risk** for their own stated savings goal, and therefore which customers to target with which intervention (e.g., a nudge campaign, a round-up savings product, a budgeting tool aimed at a specific overspend category). The model output is meant to drive a targeting/prioritization decision, not to produce a precise savings forecast.
- **What is the precise, one-sentence definition of the target variable?**
  `Goal_Met = 1` if `Disposable_Income >= Desired_Savings`, else `0` — i.e., a binary label that is `1` when an individual's actual leftover income (after all recorded expenses) is enough to cover the monthly savings amount they themselves said they want to save (see Section 3 for the full leakage discussion).
- **Is the primary task classification or regression, and why?**
  **Binary classification.** The business decision this project informs is a yes/no call — is this person on-track or not, so the pipeline can flag them for a marketing/outreach decision — not a request for a precise savings-shortfall forecast. A regression on `Disposable_Income` or on the savings gap would be a reasonable _follow-up_ analysis, but it is not what the stated business decision requires, and the target as defined is already binary by construction — framing it as regression would mean predicting a continuous quantity and then re-thresholding it, which throws away information about _why_ that threshold was chosen and complicates evaluation against the yes/no decision the business actually needs to make.

### Phase 1 — Data Understanding

- What does each column mean, and what unit/time period does it represent?
- What is the distribution of income, expenses, and savings — skew, outliers, implausible values (e.g., expenses exceeding income)?
- Are there missing values or duplicate rows, and how are they handled?
- How correlated are expense categories with income and with each other?
- Is the target mathematically derivable from any candidate feature (leakage check)?
- What is the class balance of `Goal_Met` once leakage columns are excluded?

### Phase 2 — Feature Engineering

- Do expense-to-income ratios generalize better across income levels than raw expense values?
- How should `Occupation` and `City_Tier` be encoded?
- Which features require scaling, and does that depend on the downstream model?
- Are any features redundant or highly collinear?

### Phase 3 — Baseline

- What accuracy/F1 does a majority-class or simple single-rule baseline achieve?
- What does plain logistic regression achieve using only income and 1–2 expense ratios?

### Phase 4 — Model Comparison

- Which 5–7 model families are appropriate given the data (n=20,000, mixed numeric/categorical, moderate dimensionality)?
- What validation strategy fits the class balance found in Phase 1 (e.g., stratified k-fold)?
- What hyperparameter search method is used, and what parameters move performance most?
- Given the business framing, is precision or recall more important — i.e., is a false negative (missing someone who would meet their goal) more costly than a false positive?

### Phase 5 — Explainability

- Which features matter most globally for the winning model (SHAP / permutation importance)?
- Are there notable interaction effects (e.g., does `City_Tier` change how much `Dependents` reduces savings potential)?
- Can individual predictions be explained in plain business language?

### Phase 6 — Unsupervised Extension

- What spending personas emerge from clustering on expense-category proportions?
- How many clusters are statistically justified (elbow method, silhouette score)?
- Do the resulting personas correlate meaningfully with `Goal_Met`?

### Phase 7 — Business Translation

- What are the 3–5 most actionable findings, stated as recommendations rather than statistics?
- Which expense category carries the most unrealized (potential) savings across the population?
- Where does the model fail or lose reliability — what should a stakeholder be told before acting on it?

### Phase 8 — Reporting

- Does the final write-up explain _reasoning_ (e.g., why leakage columns were excluded, why a given metric was chosen) rather than just reporting numbers?
- Which 3–4 visualizations communicate the findings fastest to a non-technical reader?

### Bonus / extension questions (not required for core deliverable)

- Can `City_Tier` be predicted from spending mix alone?
- Can `Occupation` be predicted from income + expense pattern?
- Does a lifecycle pattern exist between age and discretionary spending (entertainment, eating out)?
- Does `City_Tier` still affect disposable income after controlling for income level?

### Machine-readable question manifest

```yaml
project: indian-personal-finance-savings-prediction
target_variable:
  name: Goal_Met
  definition: "1 if Disposable_Income >= Desired_Savings else 0"
  derived_from: [Disposable_Income, Desired_Savings]
  leakage_excluded_features:
    [Disposable_Income, Desired_Savings, Desired_Savings_Percentage]
questions:
  - id: P0-Q1
    phase: framing
    text: "What real business decision does this project inform?"
    answer: "Whether to flag an individual as on-track or at-risk for their stated savings goal, to target fintech savings-product marketing/outreach."
  - id: P0-Q2
    phase: framing
    text: "What is the precise definition of the target variable?"
    answer: "Goal_Met = 1 if Disposable_Income >= Desired_Savings else 0."
  - id: P0-Q3
    phase: framing
    text: "Is the primary task classification or regression, and why?"
    answer: "Binary classification, because the business decision needed is a yes/no targeting call and the target is binary by construction."
  - id: P1-Q1
    phase: data_understanding
    text: "What does each column mean, and what unit/period does it represent?"
  - id: P1-Q2
    phase: data_understanding
    text: "What is the distribution of income, expenses, and savings — skew, outliers, implausible values?"
  - id: P1-Q3
    phase: data_understanding
    text: "Are there missing values or duplicate rows, and how are they handled?"
  - id: P1-Q4
    phase: data_understanding
    text: "How correlated are expense categories with income and each other?"
  - id: P1-Q5
    phase: data_understanding
    type: leakage_check
    text: "Is the target mathematically derivable from any candidate feature?"
  - id: P1-Q6
    phase: data_understanding
    text: "What is the class balance of Goal_Met once leakage columns are excluded?"
  - id: P2-Q1
    phase: feature_engineering
    text: "Do expense-to-income ratios generalize better than raw expense values?"
  - id: P2-Q2
    phase: feature_engineering
    text: "How should Occupation and City_Tier be encoded?"
  - id: P2-Q3
    phase: feature_engineering
    text: "Which features require scaling, and does that depend on the model?"
  - id: P2-Q4
    phase: feature_engineering
    text: "Are any features redundant or highly collinear?"
  - id: P3-Q1
    phase: baseline
    text: "What accuracy/F1 does a majority-class or single-rule baseline achieve?"
  - id: P3-Q2
    phase: baseline
    text: "What does plain logistic regression achieve with minimal features?"
  - id: P4-Q1
    phase: model_comparison
    text: "Which model families are appropriate for this data shape?"
  - id: P4-Q2
    phase: model_comparison
    text: "What validation strategy fits the observed class balance?"
  - id: P4-Q3
    phase: model_comparison
    text: "What hyperparameter search method is used, and what moves performance?"
  - id: P4-Q4
    phase: model_comparison
    text: "Is precision or recall more important given the business framing?"
  - id: P5-Q1
    phase: explainability
    text: "Which features matter most globally for the winning model?"
  - id: P5-Q2
    phase: explainability
    text: "Are there notable interaction effects between features?"
  - id: P5-Q3
    phase: explainability
    text: "Can individual predictions be explained in plain business language?"
  - id: P6-Q1
    phase: unsupervised_extension
    text: "What spending personas emerge from clustering on expense proportions?"
  - id: P6-Q2
    phase: unsupervised_extension
    text: "How many clusters are statistically justified?"
  - id: P6-Q3
    phase: unsupervised_extension
    text: "Do personas correlate meaningfully with Goal_Met?"
  - id: P7-Q1
    phase: business_translation
    text: "What are the 3-5 most actionable findings?"
  - id: P7-Q2
    phase: business_translation
    text: "Which expense category carries the most unrealized savings potential?"
  - id: P7-Q3
    phase: business_translation
    text: "Where does the model fail or lose reliability?"
  - id: P8-Q1
    phase: reporting
    text: "Does the write-up explain reasoning, not just results?"
  - id: P8-Q2
    phase: reporting
    text: "Which visualizations communicate findings fastest?"
  - id: BONUS-Q1
    phase: extension
    text: "Can City_Tier be predicted from spending mix alone?"
  - id: BONUS-Q2
    phase: extension
    text: "Can Occupation be predicted from income + expense pattern?"
  - id: BONUS-Q3
    phase: extension
    text: "Does a lifecycle pattern exist between age and discretionary spending?"
  - id: BONUS-Q4
    phase: extension
    text: "Does City_Tier still affect disposable income after controlling for income?"
```

## 5. Methodology / Pipeline

```text
data/raw → Phase 1 (EDA + leakage check) → Phase 2 (feature engineering)
        → Phase 3 (baseline) → Phase 4 (model comparison, 5-7 models, CV + tuning)
        → Phase 5 (SHAP explainability) → Phase 6 (clustering / personas)
        → Phase 7 (business translation) → Phase 8 (final report)
```

## 6. Repository Structure

```text
.
├── dataset/
│   └── data.csv
├── notebooks/
│   ├── 01_eda_and_leakage_check.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_model_comparison.ipynb
│   ├── 05_explainability.ipynb
│   ├── 06_clustering_personas.ipynb
│   └── 07_business_translation.ipynb
├── src/
│   ├── preprocessing.py                    (planned)
│   ├── models.py                           (planned)
│   └── evaluation.py                       (planned)
├── results/
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   ├── final_test_evaluation.csv
│   ├── shap_summary.png
│   ├── persona_profiles.csv
│   ├── persona_clusters.png
│   └── business_recommendations.csv
├── walkthrough/
│   ├── phase0.md
│   ├── phase1.md
│   ├── phase2.md
│   ├── phase3.md
│   ├── phase4.md
│   ├── phase5.md
│   ├── phase6.md
│   └── phase7.md
├── README.md
└── requirements.txt
```

Each `walkthrough/phaseN.md` answers that phase's research questions and, for phases with a notebook, walks through it cell by cell — what each cell does and the motivation behind it — so the reasoning behind the code doesn't have to be reconstructed from the code alone. Notebooks themselves carry only section headers and code; all narrative explanation lives in the matching walkthrough document.

## 7. Setup & Usage

```bash
git clone <repo-url>
cd <repo-name>
pip install -r requirements.txt
jupyter notebook notebooks/01_eda_and_leakage_check.ipynb
```

## 8. Results

> Phase 8's final stakeholder write-up is still to be produced; the modeling, explainability, and business-translation results below are final.

**Winning model:** SVM (Linear) — `LinearSVC`, `C=10`, `class_weight="balanced"`, decision threshold `t≈-0.4` (tuned via cross-validation) — selected in Phase 4 as one of only two models reaching perfect _cross-validated_ recall on the at-risk class (`Goal_Met = 0`), with the better precision and macro-F1 of the two. Final single test-set evaluation: accuracy ≈ 0.999, macro-F1 ≈ 0.968, `recall_0 = 1.0`, `precision_0 ≈ 0.88`.

**Top 3 SHAP features (Phase 5):** `Loan_Repayment_Ratio` (dominant, >2× the next feature), `Education_Ratio`, `City_Tier_Tier_1`.

The table below reports **cross-validated (out-of-fold)** accuracy/macro-F1 for every candidate — not test-set numbers — because Phase 4 selects its winning model from cross-validated predictions specifically to avoid biasing the reported score of whichever model happens to look best on the held-out test set. See [`walkthrough/phase4.md`](walkthrough/phase4.md) for why, and for the winning model's separate, single, final test-set evaluation.

| Model               | CV Accuracy | CV F1 (macro) | Notes                                                                                                                                                                                       |
| ------------------- | ----------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Majority baseline   | 0.9944      | 0.4986        | Predicts `Goal_Met = 1` for everyone; 0 precision/recall/F1 on the minority class                                                                                                           |
| Decision Tree       | 0.9947      | 0.8028        | `class_weight="balanced"`; `recall_0 = 0.73` — class weighting alone under-serves the minority class in a single tree                                                                       |
| XGBoost             | 0.9969      | 0.8255        | Search preferred `scale_pos_weight=1` (no reweighting) over the textbook formula; `recall_0 = 0.51`                                                                                         |
| Random Forest       | 0.9965      | 0.8436        | `class_weight="balanced"`; `recall_0 = 0.69` — same limitation as the decision tree, only partly offset by ensembling                                                                       |
| Neural Net (MLP)    | 0.9959      | 0.8579        | `RandomOverSampler`-based imbalance handling; `recall_0 = 0.93` — close, but not perfect, under cross-validation                                                                            |
| Logistic Regression | 0.9970      | 0.8940        | `class_weight="balanced"`, `C=10` (tuned); `recall_0 = 1.0`                                                                                                                                 |
| SVM (Linear)        | 0.9979      | **0.9220**    | **Winning model** — `class_weight="balanced"`, `C=10` (tuned), linear kernel chosen for scalability at n=20,000; `recall_0 = 1.0`, highest macro-F1 among the two perfect-recall candidates |

**Spending personas (Phase 6):** `k=3` (silhouette-selected) — Persona 0, Tier-1 residents with dependents (n=4,758); Persona 1, no dependents, any tier (n=4,061); Persona 2, Tier-2/Tier-3 residents with dependents (n=11,181). **All 112 at-risk individuals fall in Persona 0** (χ² = 360.8, p ≈ 4.5×10⁻⁷⁹), independently corroborating Phase 5's SHAP finding that `City_Tier_Tier_1` drives the model toward "at-risk." See [`walkthrough/phase6.md`](walkthrough/phase6.md).

**Business translation (Phase 7):** every one of the 112 at-risk individuals lives in a Tier-1 city — the single highest-leverage targeting signal. `Groceries` carries the most unrealized savings potential (₹18.2M/month aggregate, ~2× the next category). For 99.1% of at-risk individuals, addressable savings across all 8 tracked categories already exceeds their shortfall — the gap is recoverable, not structural. See [`walkthrough/phase7.md`](walkthrough/phase7.md) and [`results/business_recommendations.csv`](results/business_recommendations.csv) for the full set of five recommendations and their supporting evidence.

## 9. Limitations

- The dataset's income/expense figures may be synthetically generated rather than survey-collected — treat absolute figures as illustrative rather than nationally representative.
- `Goal_Met` is a self-referential target (defined from the individual's own stated goal), not an external measure of financial health.
- Phase 6's clustering silhouette scores are uniformly low (< 0.1) across every `k` tested, and the chosen 3-cluster solution is driven almost entirely by two features (`Education_Ratio`, `Rent_Ratio`) that are themselves proxies for `Dependents` and `City_Tier` — the personas are a demographic/geographic split already present in the raw data, not a richly discovered multivariate spending archetype.
- The winning model (SVM, linear) is provably incapable of representing any feature-interaction effect (Phase 5); the minority class (`Goal_Met = 0`) contains only 112 rows, so its `recall_0 = 1.0` result — while consistent across cross-validation and the held-out test set — rests on a small sample and is not a guarantee against a genuinely new population.

## 10. License

Code in this repository: MIT License.
Dataset: refer to the original Kaggle listing for license terms.

## 11. Citation

Dataset: shriyashjagtap, _Indian Personal Finance and Spending Habits_, Kaggle. <https://www.kaggle.com/datasets/shriyashjagtap/indian-personal-finance-and-spending-habits>
