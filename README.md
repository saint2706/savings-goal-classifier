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

| Column | Description | Role |
|---|---|---|
| `Income` | Monthly income, currency units (verify ₹ in raw file) | Feature |
| `Age` | Age of individual | Feature |
| `Dependents` | Number of dependents | Feature |
| `Occupation` | Employment / job category | Feature (categorical) |
| `City_Tier` | Tier 1 / Tier 2 / Tier 3 living area | Feature (categorical) |
| `Rent`, `Loan_Repayment`, `Insurance`, `Groceries`, `Transport`, `Eating_Out`, `Entertainment`, `Utilities`, `Healthcare`, `Education`, `Miscellaneous` | Monthly spend per category | Features |
| `Desired_Savings_Percentage` | Target % of income the person wants to save | Used to derive target — **excluded from feature set** |
| `Desired_Savings` | Target absolute monthly savings amount | Used to derive target — **excluded from feature set** |
| `Disposable_Income` | Income minus all expenses | Used to derive target — **excluded from feature set** |
| `Potential_Savings_*` (8 columns, one per expense category) | Estimated recoverable savings per category | Feature / exploratory use |

## 3. Target Variable

```
Goal_Met = 1 if Disposable_Income >= Desired_Savings else 0
Goal_Met = 0 otherwise
```

**⚠️ Data leakage constraint (read before modifying the feature set):**
`Disposable_Income` and `Desired_Savings` are used to *construct* `Goal_Met`. They must **never** be included as model input features — doing so lets the model trivially reconstruct the label rather than predict it, producing artificially perfect accuracy that has no real-world meaning. The legitimate feature set is limited to columns known independently of the target: `Income`, `Age`, `Dependents`, `Occupation`, `City_Tier`, and the raw expense category columns (optionally converted to income ratios).

## 4. Research Questions

Every question below maps to one phase of the project pipeline (Section 5). Questions are answered in order; later phases depend on decisions made in earlier ones.

### Phase 0 — Framing

- **What real business decision does this project inform?**
  Whether a fintech / savings-product marketing team should treat an individual as **on-track** or **at-risk** for their own stated savings goal, and therefore which customers to target with which intervention (e.g., a nudge campaign, a round-up savings product, a budgeting tool aimed at a specific overspend category). The model output is meant to drive a targeting/prioritization decision, not to produce a precise savings forecast.
- **What is the precise, one-sentence definition of the target variable?**
  `Goal_Met = 1` if `Disposable_Income >= Desired_Savings`, else `0` — i.e., a binary label that is `1` when an individual's actual leftover income (after all recorded expenses) is enough to cover the monthly savings amount they themselves said they want to save (see Section 3 for the full leakage discussion).
- **Is the primary task classification or regression, and why?**
  **Binary classification.** The business decision this project informs is a yes/no call — is this person on-track or not, so the pipeline can flag them for a marketing/outreach decision — not a request for a precise savings-shortfall forecast. A regression on `Disposable_Income` or on the savings gap would be a reasonable *follow-up* analysis, but it is not what the stated business decision requires, and the target as defined is already binary by construction — framing it as regression would mean predicting a continuous quantity and then re-thresholding it, which throws away information about *why* that threshold was chosen and complicates evaluation against the yes/no decision the business actually needs to make.

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
- Does the final write-up explain *reasoning* (e.g., why leakage columns were excluded, why a given metric was chosen) rather than just reporting numbers?
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
  leakage_excluded_features: [Disposable_Income, Desired_Savings, Desired_Savings_Percentage]
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

```
data/raw → Phase 1 (EDA + leakage check) → Phase 2 (feature engineering)
        → Phase 3 (baseline) → Phase 4 (model comparison, 5-7 models, CV + tuning)
        → Phase 5 (SHAP explainability) → Phase 6 (clustering / personas)
        → Phase 7 (business translation) → Phase 8 (final report)
```

## 6. Repository Structure

```
.
├── dataset/
│   └── data.csv
├── notebooks/
│   ├── 01_eda_and_leakage_check.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_model_comparison.ipynb
│   ├── 05_explainability.ipynb
│   └── 06_clustering_personas.ipynb        (planned)
├── src/
│   ├── preprocessing.py                    (planned)
│   ├── models.py                           (planned)
│   └── evaluation.py                       (planned)
├── results/
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   ├── shap_summary.png
│   └── persona_clusters.png                (planned)
├── walkthrough/
│   ├── phase0.md
│   ├── phase1.md
│   ├── phase2.md
│   ├── phase3.md
│   ├── phase4.md
│   ├── phase5.md
│   └── phase6.md …                         (planned, added alongside each new notebook)
├── README.md
└── requirements.txt
```

Each `walkthrough/phaseN.md` answers that phase's research questions and, for phases with a notebook, walks through it cell by cell — what each cell does and the motivation behind it — so the reasoning behind the code doesn't have to be reconstructed from the code alone.

## 7. Setup & Usage

```bash
git clone <repo-url>
cd <repo-name>
pip install -r requirements.txt
jupyter notebook notebooks/01_eda_and_leakage_check.ipynb
```

## 8. Results

*(Phase 6–8 results — persona count, business translation — still to be filled in once those notebooks exist.)*

**Winning model:** Neural Net (MLP), `hidden_layer_sizes=(64,)`, `alpha=0.001`, trained on an oversampled/scaled feature matrix — selected in Phase 4 for reaching perfect recall on the at-risk class (`Goal_Met = 0`) with the best precision and macro-F1 among the models that did.

**Top 3 SHAP features (Phase 5):** `Loan_Repayment_Ratio` (dominant, ~1.7× the next feature), `City_Tier_Tier_1`, `Education_Ratio`.

| Model | Accuracy | F1 (macro) | Notes |
|---|---|---|---|
| Majority baseline | 0.9945 | 0.4986 | Predicts `Goal_Met = 1` for everyone; 0 precision/recall/F1 on the minority class (Phase 3) |
| Logistic Regression | 0.9975 | 0.9068 | `class_weight="balanced"`, `C=10` (tuned); `recall_0 = 1.0` (Phase 4) |
| Decision Tree | 0.9932 | 0.7436 | `class_weight="balanced"`; `recall_0 = 0.59` — class weighting alone under-serves the minority class in a single tree (Phase 4) |
| Random Forest | 0.9945 | 0.7486 | `class_weight="balanced"`; `recall_0 = 0.50` — same limitation as the decision tree, only partly offset by ensembling (Phase 4) |
| XGBoost | 0.9972 | 0.8326 | Search preferred `scale_pos_weight=1` (no reweighting) over the textbook formula — reweighting cost more precision than it gained in recall (Phase 4) |
| SVM (Linear) | 0.9980 | 0.9226 | `class_weight="balanced"`, `C=10` (tuned), linear kernel chosen for scalability at n=20,000; `recall_0 = 1.0` (Phase 4) |
| Neural Net (MLP) | 0.9982 | **0.9309** | **Winning model** — `RandomOverSampler`-based imbalance handling; `recall_0 = 1.0`, `precision_0 = 0.759` (Phase 4) |

## 9. Limitations

- The dataset's income/expense figures may be synthetically generated rather than survey-collected — treat absolute figures as illustrative rather than nationally representative.
- `Goal_Met` is a self-referential target (defined from the individual's own stated goal), not an external measure of financial health.
- Clustering results (Phase 6) are sensitive to the chosen expense-ratio features and scaling method; persona labels are interpretive, not ground truth.

## 10. License

Code in this repository: MIT License.
Dataset: refer to the original Kaggle listing for license terms.

## 11. Citation

Dataset: shriyashjagtap, *Indian Personal Finance and Spending Habits*, Kaggle. https://www.kaggle.com/datasets/shriyashjagtap/indian-personal-finance-and-spending-habits
