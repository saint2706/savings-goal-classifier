# TODO — audit follow-ups and modernization

Source: codebase audit of 2026-10-03. Notebook refs are 0-indexed cell numbers
(NB02 c11 = `02_feature_engineering.ipynb`, cell 11). IDs (A1, C1, ...) match the audit matrix.

> **Data:** `dataset/36151-0002-Data.tsv` (ICPSR 36151, DS0002) is not in the repo and must never be
> committed (ICPSR terms prohibit redistribution; `.gitignore` now covers it). Download it yourself
> from <https://www.icpsr.umich.edu/web/DSDR/studies/36151> (free, login required), then:
> `uv run sgc build --tsv dataset/36151-0002-Data.tsv`  (or `python src/build_dataset.py --tsv ...` until the port lands).
> Never put credentials in this repo.

## 0. Blocked on the dataset

- [ ] Download DS0002 (tab-delimited) into `dataset/` and build `households.csv` / `features.csv`.

## 1. Do first — these can change the conclusions

- [ ] **A1 (P0) Falsify the leakage hypothesis.** Food rupees ≈ f(household size), so
      `Household_Size` + `Groceries_Share` + `Log_Income` can recover total spend and hence the label.
      Run on `households.csv` under grouped CV:
  - [ ] T1 ablation: all / no `Groceries_Share` / no size family (`Household_Size`, `Dependents`, `Dependency_Ratio`) / no shares+indicators (the deployable model).
  - [ ] T2 `R²[log(Groceries) ~ log(Household_Size)]`.
  - [ ] T3 model-free oracle: `implied_sr = 1 - (k*size/share)/INCOME`, `k = median(Groceries/size)`; drop the ~0.06% of rows with `Groceries_Share == 0` (division by zero gives `inf`, which `roc_auc_score` rejects; also drop NaN shares) and report `roc_auc_score(Goal_Met[m], implied_sr[m])` on the remaining mask `m`, noting the excluded count.
  - [ ] Decision: if T3 is high (≳0.85) or the no-shares model keeps most of the AUC, reframe the headline as the income+demographics model and treat shares as post-hoc/diagnostic. Update README, `report.tex` abstract and conclusion.
- [ ] **C1 (P1) Grouped CV.** Carry `PSUID` into `features.csv`; replace `StratifiedKFold` / `train_test_split` with `StratifiedGroupKFold(5, shuffle=True, random_state=42)` grouped by `PSUID` (NB02 c1, NB03 c1, NB04 c1, NB05 c1, NB07 c1). Record the drop in AUC / macro-F1.
- [ ] **D1 (P1) No full-data preprocessing.** Move the p99 winsorization (NB02 c11) into the pipeline (`QuantileClipper`); fit the income-rule threshold per fold (NB03 c4, NB07 c3); take the train/test split *before* any Phase 2/3 design decision.
- [ ] **E1 (P1) Tuned vs untuned XGBoost.** Table 1 / Fig 1 use the untuned config (depth 6, 400 trees, lr 0.08 → 0.8371); text and test/SHAP use the tuned one (depth 4, 200, lr 0.15 → 0.8377). **Add** to NB04 a cell that writes the new file `results/model_final.json` (`best_score_`, `best_params_`; it does not exist today, NB04 only writes `model_comparison.csv`); then fix `make_figures.py:142`, `report.tex:122,148`, Table 1 caption (single-threshold/baseline rows are full-sample CV, not the training split).
- [ ] **F1/F2 (P1) Capture ceiling.** At a 25% budget capture is capped at 25/68.07 = 36.7%; model = 99.6% of ceiling, income rule = 95.4%. Report 10/25/50/65% budgets (`capture_vs_ceiling`), add a grouped-bootstrap CI on model − income, compare precision against the income rule (not random). Rewrite `report.tex:34,198,210,234` and `business_recommendations.csv` row 2.
- [ ] **G1 (P1) "Miscellaneous is the lever".** Replace the clipped one-sided sum and median benchmark (NB07 c6) with mean benchmarks, signed excess, bootstrap CIs, and rupee (not share) peer benchmarks. Drop/hedge `report.tex:206` and recommendation 4 unless the CI excludes zero.

## 2. Then — correctness and honesty of the write-up

- [ ] **B1** Replace NB01 c13 test C (Pearson on a fat-tailed target) with Spearman + marginal AUC + the A1 multivariate tests; drop the circular identity-1 check (`build_dataset.py:183`).
- [ ] **H1** Drop one share (or use ILR) in linear models; compute SHAP on grouped feature families + permutation-importance cross-check; report one interaction decomposition (full set, not the 3,000-row sample); don't hide `Rent_Share` 17× worse behind a median.
- [ ] **I1** Personas: remove "personas" from the abstract; δ-sensitivity sweep (0.1×–1× min positive); bootstrap ARI + Davies–Bouldin + Calinski–Harabasz; choose *k* within one representation; fix the hardcoded NB06 c6 title ("far below weak-structure threshold" vs silhouette 0.4115); replace "181%" with a persona × income-decile interaction; replace ARI-vs-income-tertiles with MI / Kruskal–Wallis.
- [ ] **J1** Add `statsmodels` (NB02 c20 imports it); fix `requirements.txt:16` (`ihds_phase5.md` doesn't exist) and `report.tex:221` (`--tsv` is required).
- [ ] **K1** Add PR-AUC, Brier, log-loss, ECE to NB04; rank on threshold-free metrics too. Name the positive class: NB04's `y = Goal_Met` has on-track as positive (prevalence 0.319, matches the existing ROC-AUC); the at-risk view (positive = `1 - Goal_Met`, scores `1 - p`) has prevalence 0.681. Report PR-AUC for at-risk, with its 0.681 baseline, and label the class in the table.
- [ ] **K2** Fix the NB04 c8 "precision falls to" print string; evaluate the tuned threshold once on the held-out set.
- [ ] **K3** Don't `RobustScaler` zero-inflated shares (IQR collapses: scale 0.0033 → values up to ~150; 90% zeros → scale 1).
- [ ] **K4** Apply `WT` in reported/population figures; label "Rs 36 crore" as a sample total.
- [ ] **K5/K6** Reconcile `COTOTAL` vs sum-of-categories (`build_dataset.py:183` vs `:193`); soften "exact identity over expense columns"; treat missing `DB5` as missing (add `Debt_Missing`), include teens in `Dependents` or rename; document the occupation tie-break.
- [ ] **K7** Make the notebooks write the figure inputs (new `results/*.json` / CSVs; none exist today beyond the current CSVs, so keep the CSV contract until each producer is added), then have `make_figures.py` read them (no transcribed tables; "Other" SHAP slice is 10.4 not 10.5); plot all 10 deciles in Fig 3b; replace hardcoded display strings (NB05 c10, NB06 c6/c14, NB07 c12/c13) with f-strings.
- [ ] **K8** Winsorize the expense/income ratios before the NB02 c3 "ratios transfer worst" comparison.
- [ ] **K9** Name the AI tool/version in `report.tex:229`; substantiate or drop the "<20%" claim.

## 3. Modernization (uv · polars · ruff · pandera · mypy · pytest)

Verified in a scratch project: `uv sync` resolves; Polars port matches pandas `build()` on all 50 columns;
`ruff`, `mypy --strict` and `pytest` (4 tests, 100% coverage) pass.

- [ ] `uv init` / add `pyproject.toml` (see audit write-up): `requires-python = ">=3.11,<3.14"` **and** `[tool.uv] environments = [...]` (without both, uv falls back to an unbuildable `llvmlite` sdist via shap/numba).
- [ ] `uv sync`, commit `uv.lock`, then `git rm requirements.txt`.
- [ ] Port `src/build_dataset.py` → `src/savings_goal/data/build.py` (lazy Polars, `sink_parquet`); add `data/schema.py` (Pandera contract, incl. shares-sum-to-1) and `cli.py` (`sgc build`).
- [ ] Add `tests/conftest.py` (synthetic raw TSV) + `tests/unit/test_build.py` + integration CLI test.
- [ ] Move notebook logic into `features/`, `models/`, `evaluation/` (`cv.py`, `metrics.py`, `leakage.py`, `explain.py`); notebooks become thin callers.
- [ ] Drop the `pred_contribs` SHAP workaround — `shap.TreeExplainer` works with shap 0.51 + xgboost 3.2 (identical values).
- [ ] CI: `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest`.

## 4. Later — product

- [ ] Two-tier deployable scoring (income/demographics at onboarding; spending shares only if observed *before* the prediction window) behind FastAPI + Pandera request validation. **Only after A1 is resolved.**
- [ ] Robust/quantile regression on `Savings_Rate` (or decile of `COTOTAL/INCOME`); rank by predicted rupee shortfall.
- [ ] Uplift modeling (T-learner / causal forest) once any intervention data exists — IHDS cannot supply it.
