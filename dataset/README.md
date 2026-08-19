# Dataset — acquisition and build

The data files this project analyses are **not committed to the repository**. They are derived from the India Human Development Survey-II, whose terms of use prohibit redistribution:

> *"You agree not to redistribute data or other materials without the written agreement of ICPSR"* — ICPSR Terms of Use, study 36151

`households.csv` and `features.csv` are therefore listed in `.gitignore`. Everything below rebuilds them from your own ICPSR download in about a minute.

---

## 1. Download the source data

**India Human Development Survey-II (IHDS-II), 2011-12 — ICPSR study 36151**
<https://www.icpsr.umich.edu/web/DSDR/studies/36151>

- Free. Registration is required; ICPSR membership is not.
- Download the **DS0002 (Household)** dataset. The other 13 datasets in the study are not used.
- Any format works, but the build script expects the **tab-delimited** file: `36151-0002-Data.tsv` (~80 MB unpacked, 42,152 rows × 758 columns).

You will also want `36151-0002-Codebook.pdf` from the same download — it is the authoritative source for what each `CO*` consumption variable means, and it does **not** agree with the variable ordering shown on the IHDS project website.

## 2. Build the analysis dataset

```bash
python src/build_dataset.py \
    --tsv path/to/ICPSR_36151/DS0002/36151-0002-Data.tsv \
    --out dataset/households.csv
```

Expected output:

```text
raw households:        42,152
dropped (income<=0 / missing consumption): 634
final households:      41,518
Goal_Met rate @ 20%:   0.3193
```

The `--threshold` flag changes the savings-rate benchmark that defines `Goal_Met` (default `0.20`). If you change it, every downstream result changes with it — see the sensitivity table in `walkthrough/phase1.md`.

## 3. Build the engineered feature matrix

Run the feature-engineering notebook, which writes `dataset/features.csv`:

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 02_feature_engineering.ipynb
```

## 4. Run the rest of the pipeline

In order — each phase depends on the artifacts of the previous one:

| Notebook | Reads | Writes |
| --- | --- | --- |
| `01_eda_and_leakage_check.ipynb` | `households.csv` | `results/phase1_*.png` |
| `02_feature_engineering.ipynb` | `households.csv` | **`features.csv`** |
| `03_baseline.ipynb` | `features.csv` | `results/baseline.*` |
| `04_model_comparison.ipynb` | `features.csv` | `results/model_comparison.*` |
| `05_explainability.ipynb` | `features.csv` | `results/shap_*` |
| `06_clustering_personas.ipynb` | `features.csv` | `results/persona*`, `cluster_selection.png` |
| `07_business_translation.ipynb` | `features.csv` | `results/business_*` |

```bash
cd notebooks
for nb in 0*.ipynb; do
    jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3000 "$nb"
done
```

Runtime is roughly 20–30 minutes end to end on a laptop; Phase 4's randomised hyperparameter search dominates.

---

## What the build script produces

`households.csv` — 41,518 rows × 50 columns, one row per household, all money in **annual rupees**.

| Group | Columns |
| --- | --- |
| Identifiers | `STATEID`, `DISTID`, `PSUID`, `HHID`, `HHSPLITID`, `IDHH` |
| Survey weight | `WT` (carried, not applied) |
| Features | `INCOME`, `Log_Income`, `Household_Size`, `Dependents`, `Head_Age`, `Max_Adult_Education`, `Occupation`, `Area_Type`, `Caste_Group`, `Religion`, `Debt_To_Income`, 11 × `*_Share` |
| External validation | 6 × `Has_*` (bank savings, fixed deposit, pension/LIC, securities, post office, gold) |
| **Leakage — never features** | 11 raw rupee categories, `COTOTAL`, `Savings`, `Savings_Rate` |
| Target | `Goal_Met` |

**Two things to know before modifying the build:**

- IHDS uses **two recall windows** — `CO1X`–`CO33` are 30-day, `CO34`–`CO52` are annual. The script annualises with `12 × monthly + annual` and this reproduces IHDS's own `COTOTAL` with a median error of 0. Mixing them silently inflates food 12× relative to durables.
- The **expense-to-income ratios reconstruct the target** with 99.75% agreement, which is why the feature set uses each category's share of *total expenditure* instead. See `walkthrough/phase1.md` § Q5.

## Citation

Desai, Sonalde, Reeve Vanneman, and National Council of Applied Economic Research, New Delhi.
*India Human Development Survey-II (IHDS-II), 2011-12.* Inter-university Consortium for Political and Social Research \[distributor\], 2018-08-08. <https://doi.org/10.3886/ICPSR36151.v6>
