# Dataset — acquisition and build

The data files this project analyses are **not committed to the repository**. They are derived from the India Human Development Survey-II, whose terms of use prohibit redistribution:

> *"You agree not to redistribute data or other materials without the written agreement of ICPSR"* — ICPSR Terms of Use, study 36151

`.gitignore` therefore excludes `dataset/*.tsv`, `dataset/*.parquet`, the legacy CSVs and `artifacts/` (fitted models). Everything below rebuilds them from your own ICPSR download in under a minute. Never put ICPSR credentials in this repository.

---

## 1. Download the source data

**India Human Development Survey-II (IHDS-II), 2011-12 — ICPSR study 36151**
<https://www.icpsr.umich.edu/web/DSDR/studies/36151>

- Free. Registration is required; ICPSR membership is not.
- Download the **DS0002 (Household)** dataset in the **tab-delimited** format: `36151-0002-Data.tsv` (~80 MB, 42,152 rows × 758 columns). The other 13 datasets are not used.
- Keep `36151-0002-Codebook.pdf` too — it is the authoritative source for every `CO*` consumption item, and it does **not** agree with the ordering on the IHDS website.

## 2. Build the analysis tables

```bash
uv sync
uv run sgc build --tsv path/to/ICPSR_36151/DS0002/36151-0002-Data.tsv
```

`--tsv` is required. This streams the TSV through a lazy Polars pipeline, validates the result against the Pandera contract in `src/savings_goal/data/schema.py` (shares sum to one, label monotone in the savings rate, label sets, ranges), and writes:

- `dataset/households.parquet` — 41,518 households × 54 columns;
- `dataset/features.parquet` — the engineered feature table, including the PSU-grouped train/test split (`Is_Test`).

Expected output:

```text
raw households:        42,152
dropped (income<=0 / missing consumption): 634
final households:      41,518
Goal_Met rate @ 20%:  0.3193
occupation ties broken by priority: 7,491
debt (DB5) missing:    3,150
COTOTAL vs 11-category sum: within 1% for 97.72%; label agreement 0.9975
```

`--threshold` changes the savings-rate benchmark that defines `Goal_Met` (default `0.20`); every downstream result changes with it. `uv run sgc features` rebuilds only the feature table.

## 3. Run the pipeline

| Notebook | Writes |
| --- | --- |
| `01_eda_and_leakage_check.ipynb` | `results/phase1_*`, **`leakage.json`** (A1) |
| `02_feature_engineering.ipynb` | `results/feature_engineering.json` |
| `03_baseline.ipynb` | `results/baseline.*` |
| `04_model_comparison.ipynb` | `results/model_comparison.*`, **`model_final.json`**, `model_test.json` |
| `05_explainability.ipynb` | `results/shap_*`, `explain.json` |
| `06_clustering_personas.ipynb` | `results/persona*`, `cluster_selection.png` |
| `07_business_translation.ipynb` | `results/business_*` |
| `08_savings_rate_regression.ipynb` | `results/savings_rate_regression.*` |

```bash
for nb in notebooks/0*.ipynb; do
    uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=6000 "$nb"
done
uv run python project/figures/make_figures.py
uv run sgc train            # optional: fit the two scoring tiers into artifacts/
```

Runtime is roughly 15 minutes on four cores; Phase 4's two randomised searches and Phase 5's interaction decomposition dominate.

---

## What the build produces

| Group | Columns |
| --- | --- |
| Identifiers | `STATEID`, `DISTID`, `PSUID`, `IDPSU`, `HHID`, `HHSPLITID`, `IDHH` |
| Survey weight | `WT` (households represented; applied to population figures, not to fitting) |
| Features | `INCOME`/`Log_Income`, `Household_Size`, `Age_Dependents`, `Head_Age`, `Max_Adult_Education`, `Occupation`, `Area_Type`, `Caste_Group`, `Religion`, `Debt_To_Income`, `Debt_Missing`, 11 × `*_Share` |
| Build diagnostic | `Occupation_Tie` |
| External validation | 6 × `Has_*` (bank savings, fixed deposit, pension/LIC, securities, post office, gold) |
| **Leakage — never features** | 11 raw rupee categories, `Category_Total`, `COTOTAL`, `Savings`, `Savings_Rate` |
| Target | `Goal_Met` |

**Things to know before modifying the build:**

- IHDS uses **two recall windows** — `CO1X`–`CO33` are 30-day, `CO34`–`CO52` annual. The build annualises with `12 × monthly + annual`; the 11-category sum (`Category_Total`) then matches the survey's `COTOTAL` within 1% for 97.7% of households (median difference 0), and a label computed from it agrees with `Goal_Met` for 99.75%. The target uses `COTOTAL`; the shares use `Category_Total`. The identity is exact only over `INCOME` and `COTOTAL`.
- **Group key.** `PSUID` repeats across districts (39 distinct values); the unique primary sampling unit is `IDPSU` (2,461 in the analysis file). All CV and the train/test split group by `IDPSU`.
- **Household composition.** `NADULT*` counts members aged 21+ (elders included), `NTEEN*` 15–20, `NCHILD*` 0–14, `NELDER*` 60+. `Age_Dependents` = 0–14 plus 60+ (the standard age-dependency count; 15–20-year-olds are working age). It was called `Dependents` before the audit.
- **Debt.** A blank `DB5` is unknown debt, not zero: `Debt_To_Income` stays missing and `Debt_Missing = 1` (7.6% of households). Winsorisation at p99 happens inside the model pipeline, per training fold.
- **Occupation.** The label is the worker type with the most workers; ties (18% of households) go to the first in the order Salaried → Business → Non-ag labour → Farm → Ag labour, i.e. the more regular cash income wins.
- **Religion code 9** is labelled `No_Religion` (the old label `None` was read back as missing by pandas).
- The **expense-to-income ratios reconstruct the target** (99.75% agreement); and A1 (`results/leakage.json`) shows household size × food share × income approximately reconstructs it too.

## Citation

Desai, Sonalde, Reeve Vanneman, and National Council of Applied Economic Research, New Delhi.
*India Human Development Survey-II (IHDS-II), 2011-12.* Inter-university Consortium for Political and Social Research \[distributor\], 2018-08-08. <https://doi.org/10.3886/ICPSR36151.v6>
