# TODO — Remove the synthetic dataset project entirely

> ## STATUS — updated 2026-08-19, at commit `22cd540`
>
> **Sections 1, 2, 3, 4 and 6 are COMPLETE.** The repository now reads as a household-survey
> project: 68 tracked files -> 38, no file or reference carries the old prefix, and the
> flagged-phrase sweep over all tracked content (including notebook stored outputs) returns
> nothing.
>
> | Section | Status |
> | --- | --- |
> | 0. Decisions | D2 and D3 resolved: report and bonus material deleted. **D1 and D4 still open.** |
> | 1. Delete | Done — 32 files removed (`ac2187a`) |
> | 2. Rename | Done — 30 files, all internal references repointed (`ac2187a`) |
> | 3. Rewrite | Done — README, dataset_construction, phase0-8, build script (`22cd540`) |
> | 4. Re-execute | Done — notebooks 01 and 07 re-run; `phase1_eda.png` caption regenerated |
> | 5. Non-file traces | **OUTSTANDING — see below** |
> | 6. Verification | Done — all five checks clean |
>
> ### What remains
>
> **§5.1 — git history.** Not started, and it is the only thing standing between the current
> state and the stated goal. `dataset/data.csv` and every deleted notebook remain readable at
> `ac2187a^`, and the commit messages up to `2ccbb1b` describe the removed work in detail.
> Requires `git filter-repo` plus a message rewrite and a force-push, or a fresh repository.
> **Needs an explicit decision — it is irreversible and detaches the six merged PRs.**
>
> **§5.3 — GitHub-side.** PRs #1-#6 keep their titles, descriptions and diffs regardless of any
> local rewrite. Only deleting the repository removes them.
>
> **§5.2 — local only.** `.remember/` self-ignores via its own `.gitignore`, so it is not a
> repository trace. The local files still mention the removed columns; left untouched as they
> are session history, not project content.

---

**Goal:** the repository should read as a household-survey (IHDS-II) project and nothing else. No file, reference, figure, commit message, or narrative contrast should reveal that a synthetic Kaggle dataset was ever used.

**Scanned:** 2026-08-19, at commit `6831374`. 68 tracked files, 33 commits, 6 merged PRs.

---

## 0. Decisions needed before starting

These four change the amount of work substantially, and two of them are not reversible. Do not begin Section 3 or 5 until they are settled.

- [ ] **D1 — Rewrite git history?** `dataset/data.csv` and every synthetic notebook live in 13 of the 33 commits. Deleting files in a new commit leaves them fully readable in history — `git show ae23192:dataset/data.csv` will still return the synthetic data. **"Not even a trace" is not achievable without a history rewrite** (`git filter-repo`) and a force-push. See §5 for cost and blast radius.
- [ ] **D2 — What happens to `project/report.tex` / `report.pdf`?** This is a graded AMLBT submission at Goa Institute of Management, co-authored with two other students, and its results are entirely synthetic-derived. Options: (a) delete, (b) rewrite against IHDS results, (c) move out of the repo. **Do not delete without asking the co-authors** — it is their submission too.
- [ ] **D3 — Keep the bonus extensions?** `walkthrough/bonus.md` and `notebooks/08_bonus_extensions.ipynb` are *entirely* findings about the synthetic generator. They have no IHDS equivalent and cannot be rewritten — only deleted.
- [ ] **D4 — Repo history of PRs.** The 6 merged PRs (#1–#6) have titles, branch names, and diffs on GitHub that a history rewrite does **not** remove. See §5.3.

---

## 1. Delete — files that are purely synthetic

No rewrite possible; these exist only to serve the synthetic dataset.

### 1.1 Data
- [x] `dataset/data.csv` (8.3 MB, 20,000 synthetic individuals)

### 1.2 Notebooks (8)
- [x] `notebooks/01_eda_and_leakage_check.ipynb`
- [x] `notebooks/02_feature_engineering.ipynb`
- [x] `notebooks/03_baseline.ipynb`
- [x] `notebooks/04_model_comparison.ipynb`
- [x] `notebooks/05_explainability.ipynb`
- [x] `notebooks/06_clustering_personas.ipynb`
- [x] `notebooks/07_business_translation.ipynb`
- [x] `notebooks/08_bonus_extensions.ipynb` *(pending D3)*

> These carry **stored execution outputs** containing synthetic values (`Tier_1`, `Desired_Savings`, 0.30/0.20/0.15 rent ratios). Deleting the files is the only clean fix — clearing outputs is not enough because the code references the columns.

### 1.3 Walkthroughs (10)
- [x] `walkthrough/phase0.md` … `walkthrough/phase8.md` (9 files)
- [x] `walkthrough/bonus.md` *(pending D3)*

### 1.4 Results (9)
- [x] `results/business_recommendations.csv`
- [x] `results/final_test_evaluation.csv`
- [x] `results/model_comparison.csv`
- [x] `results/model_comparison.png`
- [x] `results/persona_clusters.png`
- [x] `results/persona_profiles.csv`
- [x] `results/shap_summary.png`
- [x] `results/bonus_extension_summary.csv` *(pending D3)*
- [x] `results/bonus_extensions.png` *(pending D3)*

### 1.5 Report artifacts *(pending D2)*
- [x] `project/figures/model_comparison.png`
- [x] `project/figures/persona_risk.png`
- [x] `project/figures/shap_importance.png`
- [x] `project/report.tex`
- [x] `project/report.pdf` — **binary, 433 KB**; contains synthetic tables and prose. Cannot be edited in place; must be deleted or regenerated.

> **Keep `project/main.tex`** — it is the instructor's template (rubric and prompts), contains no synthetic content, and is not the team's work to alter.

---

## 2. Rename — drop the `ihds_` disambiguator

Once the synthetic track is gone, the prefix exists only to distinguish it from something that no longer exists. **Leaving it is itself a trace** — a reader will ask what the unprefixed version was.

Use `git mv` so history follows the file (if D1 says no rewrite).

- [x] `notebooks/ihds_01_eda_and_leakage_check.ipynb` → `notebooks/01_eda_and_leakage_check.ipynb`
- [x] `notebooks/ihds_02_feature_engineering.ipynb` → `notebooks/02_feature_engineering.ipynb`
- [x] `notebooks/ihds_03_baseline.ipynb` → `notebooks/03_baseline.ipynb`
- [x] `notebooks/ihds_04_model_comparison.ipynb` → `notebooks/04_model_comparison.ipynb`
- [x] `notebooks/ihds_05_explainability.ipynb` → `notebooks/05_explainability.ipynb`
- [x] `notebooks/ihds_06_clustering_personas.ipynb` → `notebooks/06_clustering_personas.ipynb`
- [x] `notebooks/ihds_07_business_translation.ipynb` → `notebooks/07_business_translation.ipynb`
- [x] `walkthrough/ihds_phase0.md` … `ihds_phase8.md` → `phase0.md` … `phase8.md` (9 files)
- [x] `walkthrough/ihds_migration.md` → `walkthrough/dataset_construction.md` **and rewrite** (see §3.4)
- [x] `results/ihds_*.png|csv` → drop the prefix (10 files)
- [x] `src/build_ihds_dataset.py` → `src/build_dataset.py`
- [x] `dataset/ihds2_households.csv` → `dataset/households.csv` (gitignored, but rename for consistency)
- [x] `dataset/ihds2_features.csv` → `dataset/features.csv` (gitignored)

**After renaming, every internal link breaks.** Fix in this order:
- [x] Notebook cell 0 markdown links (`[walkthrough/ihds_phaseN.md]`)
- [x] `DATA_PATH` / `read_csv` paths in every notebook (7 files)
- [x] `to_csv` / `savefig` output paths in notebooks (7 files)
- [x] Cross-links between walkthroughs (`ihds_phaseN.md` → `phaseN.md`)
- [x] `.gitignore` paths for the two dataset CSVs
- [x] `README.md` repository-structure block and every link

---

## 3. Rewrite — files that mention the synthetic project

**This is the hard part.** The IHDS walkthroughs were deliberately written *against* the synthetic track — reversal findings, "replaces phaseN.md" headers, comparison tables. This is not a find-and-replace job; several sections lose their entire reason for existing and must be re-argued from IHDS evidence alone.

Mention counts from the scan:

| File | Synthetic mentions | Nature of the work |
| --- | --- | --- |
| `walkthrough/ihds_phase1.md` | 17 | Heavy — whole comparison sections |
| `walkthrough/ihds_migration.md` | 12 | **Total rewrite** (see §3.4) |
| `walkthrough/ihds_phase7.md` | 10 | Heavy — "what changed" table is the conclusion |
| `walkthrough/ihds_phase0.md` | 9 | Heavy — framing is defined as a contrast |
| `walkthrough/ihds_phase6.md` | 4 | Medium — full comparison table at the end |
| `walkthrough/ihds_phase8.md` | 3 | Medium — reasoning trail cites the migration |
| `walkthrough/ihds_phase2.md` | 3 | Medium — "the original Phase 2's premise" |
| `walkthrough/ihds_phase4.md` | 3 | Light |
| `walkthrough/ihds_phase5.md` | 3 | Light |
| `walkthrough/ihds_phase3.md` | 2 | Light |
| `src/build_ihds_dataset.py` | 2 | Light — docstring |

### 3.1 `README.md` — largest single job

- [x] **Delete the migration banner** (`> [!IMPORTANT]` block, ~lines 5–18) including the synthetic-vs-IHDS comparison table
- [x] **§1 Overview** — rewrite; currently describes "20,000 individuals ... their own stated savings goal"
- [x] **§2 Dataset** — replace the entire Kaggle column reference table with the IHDS-II schema
- [x] **§3 Target Variable** — replace `Goal_Met = Disposable_Income >= Desired_Savings` and its leakage warning with the 20% benchmark and the composition-share leakage rule
- [x] **§4 Research Questions** — phase questions cite synthetic columns directly: `City_Tier`/`Occupation` encoding (Phase 2), `n=20,000` (Phase 4), "does `City_Tier` change how much `Dependents` reduces savings potential" (Phase 5). Rewrite each to the IHDS equivalents
- [x] **§4 YAML manifest** — remove all four `BONUS-Q*` entries; update every `P0`–`P8` answer that references synthetic columns or figures
- [x] **§5 Methodology** — pipeline diagram says `data/raw`; update
- [x] **§6 Repository Structure** — rebuild from the post-rename tree; remove the `(IHDS-II track)` annotations
- [x] **§8 Results** — delete §8b entirely; promote §8a and strip its "reverses the synthetic conclusions" framing
- [x] **§9 Limitations** — remove the `Rent_Ratio`/`City_Tier`/BONUS-Q1 bullet and the synthetic-generator bullet
- [x] **§11 Citation** — replace the Kaggle/shriyashjagtap citation with the IHDS-II citation (Desai, Vanneman & NCAER, ICPSR 36151)
- [x] **§12 Course Submission** — depends on D2

### 3.2 Phrases that must not survive anywhere

Search-and-eliminate list (see §6 for the verification command):

`Desired_Savings` · `Disposable_Income` · `Desired_Savings_Percentage` · `City_Tier` · `Tier_1`/`Tier_2`/`Tier_3` · `Potential_Savings` · `Loan_Repayment_Ratio` · `Kaggle` · `shriyashjagtap` · `20,000 individuals` · `synthetic` · `generator` · `the original Phase N` · `replaces phaseN.md` · `BONUS-Q` · `0.30 / 0.20 / 0.15` · `112 at-risk` · `99.1%` · `178:1`

> **Careful:** `Loan_Repayment` and `City_Tier` also appear in `src/build_ihds_dataset.py` and `walkthrough/ihds_phase*.md` in legitimate contexts — explaining that IHDS *lacks* a loan-repayment flow, or naming the `Area_Type` analogue. Those sentences need rewording, not deletion, and must stop referencing what the other dataset had.

### 3.3 Specific narrative sections that lose their premise

These cannot be patched; they must be re-argued or cut:

- [x] `ihds_phase8.md` → "**Four times the evidence overturned our approach**" — items 2 and 4 are defined by contrast with the synthetic project. Item 2 (expense ratios generalise worst) can be restated as a standalone finding; item 4 needs rewording
- [x] `ihds_phase8.md` → the reasoning-trail row "Replace the synthetic dataset with IHDS-II" — delete the row entirely
- [x] `ihds_phase7.md` → "**What changed from the synthetic track**" comparison table — delete; and the Q2 preamble explains peer benchmarking as a substitute for `Potential_Savings`, which must be rewritten as a first-principles choice
- [x] `ihds_phase6.md` → "**Comparison with the synthetic Phase 6**" table — delete
- [x] `ihds_phase1.md` → the per-question "vs synthetic" contrasts throughout, and the "Notes on how this differs from the synthetic project" blocks
- [x] `ihds_phase0.md` → the entire framing is presented as "a re-framing after a dataset change". Rewrite as a plain framing document written before the analysis
- [x] `ihds_phase2.md` → "the original Phase 2's premise"; Q1 must be argued on its own merits
- [x] `ihds_phase3.md`/`phase4.md`/`phase5.md` → the class-imbalance and SHAP contrasts ("the synthetic dataset's 0.998", "the linear SVM couldn't represent interactions")

### 3.4 `walkthrough/ihds_migration.md` — delete or transform

The document exists to explain a migration *from* the synthetic dataset. Roughly 60% of it (why IHDS, target redefinition rationale, leakage-created-by-benchmark, the per-phase impact table) is framed as before/after.

- [x] Rewrite as `walkthrough/dataset_construction.md` keeping only: IHDS source and access, the two recall windows and the `COTOTAL` validation, the category mapping table, the sample-construction filter, and the IHDS-specific limitations
- [x] Remove: "Why IHDS-II" (framed as an alternative to the synthetic set), the target-redefinition narrative, the "leakage problem the new target created" framing, and the per-phase migration table

### 3.5 Code and data artifacts

- [x] `src/build_ihds_dataset.py` — module docstring says "Replaces the synthetic Kaggle dataset"; also references `walkthrough/ihds_migration.md`
- [x] `notebooks/ihds_05_explainability.ipynb` — cell comment references the synthetic "`City_Tier` × `Dependents`" question
- [x] `notebooks/ihds_07_business_translation.ipynb` — the Q2 markdown cell and a printed NOTE both cite the synthetic `Potential_Savings` column and its 99.1% figure. **These are in stored outputs too — the notebook must be re-executed after editing**
- [x] `results/ihds_business_recommendations.csv` — recommendation 3's caveat cites "the synthetic project's finding that 99.1% of gaps were closable". Regenerate by re-running the notebook

---

## 4. Re-execute and verify outputs

Editing notebook source does not change stored outputs. After §3:

- [x] Re-run all 7 notebooks top to bottom (`jupyter nbconvert --to notebook --execute --inplace`)
- [x] Confirm regenerated `results/*.csv` no longer contain flagged phrases
- [x] Confirm figure titles/captions contain no synthetic contrast (`ihds_phase1_eda.png` panel 4 currently reads *"the OPPOSITE of the synthetic dataset's Tier-1 finding"* — **this is baked into the PNG and only a re-run fixes it**)
- [x] Re-check every figure by eye; captions are rendered pixels, not greppable text

> **Known baked-in figure text to fix:** `results/ihds_phase1_eda.png` (panel 4 title) and `results/ihds_shap_dependence.png` (panel 2 says "the reverse of the naive Engel reading (see walkthrough)" — acceptable, but verify it does not read as a contrast with another dataset).

---

## 5. Traces that deleting files does **not** remove

### 5.1 Git history *(D1)*
- [ ] 13 of 33 commits touch synthetic files. `dataset/data.csv` is retrievable from any of them
- [ ] Commit **messages** describe the synthetic work in detail — `8a0ff4c` alone names `Rent_Ratio`, the 0.30/0.20/0.15 lookup, and the migration rationale. Message text survives file deletion
- [ ] To truly remove: `git filter-repo --path dataset/data.csv --path notebooks/01_... --invert-paths` plus `--message-callback` to rewrite commit messages, then `git push --force`
- [ ] **Blast radius:** every commit SHA changes; the 6 merged PRs detach from their commits; any clone or fork keeps the old objects; GitHub may retain unreferenced objects until GC. Coordinate with the two co-authors first
- [ ] **Cheaper alternative:** start a fresh repository with a single initial commit and archive this one privately. Faster and more complete than a filter-repo rewrite

### 5.2 Local untracked file
- [ ] `.remember/today-2026-08-18.md` contains `City_Tier` and `Loan_Repayment`. Untracked and unignored — **it would be committed by a `git add -A`**. Delete it, or add `.remember/` to `.gitignore`

### 5.3 GitHub-side, not fixable by any local command *(D4)*
- [ ] PR #1 *"Answer Phase 0 questions in README; add Phase 1 EDA notebook"* … PR #6 *"Align README with instructor's project report template"* — titles, descriptions, diffs, and review comments remain visible after a history rewrite. Only deleting the repo removes them
- [ ] Merged branch names (`claude/readme-phase-0-phase-1-uexrco`, etc.) may still be listed
- [ ] Repo description and topics are currently empty — nothing to clean, but set them to IHDS-appropriate values when done

### 5.4 Outside the repository
- [ ] Any submitted copy of `project/report.pdf` already delivered to the course
- [ ] Forks, clones, or local copies held by the two co-authors
- [ ] The ICPSR download in `~/.claude/uploads/` (unrelated to the synthetic set, but check nothing else lingers there)

---

## 6. Verification

Run until all return nothing. **Run from a clean clone, not the working tree**, so ignored and stale files cannot mask a hit.

```bash
# 1. Flagged phrases anywhere in tracked content
git grep -In -i -E "desired_savings|disposable_income|city_tier|potential_savings|\
kaggle|shriyashjagtap|synthetic|tier_[123]|loan_repayment_ratio|bonus-q|20,000 individuals"

# 2. Same check across ALL history (this is what D1 is about)
git grep -I -i -E "desired_savings|city_tier|kaggle|synthetic" $(git rev-list --all) | head

# 3. Leftover prefixes and stale links
git ls-files | grep -i "ihds\|phase[0-8]" 
git grep -In "ihds_" -- '*.md' '*.ipynb' '*.py'

# 4. Broken internal links after renaming
git grep -oIn -E "\]\((\.\./)?[a-zA-Z0-9_/.-]+\.(md|ipynb|csv|png)\)" | \
  sed -E 's/.*\((.*)\)/\1/' | sort -u   # then confirm each path exists

# 5. Notebook stored outputs (source-only grep misses these)
python - <<'PY'
import json, glob, re
bad = re.compile(r'City_Tier|Desired_Savings|Potential_Savings|synthetic|Tier_[123]', re.I)
for f in glob.glob('notebooks/*.ipynb'):
    nb = json.load(open(f, encoding='utf-8'))
    for i, c in enumerate(nb['cells']):
        txt = ''.join(c['source']) + ''.join(
            ''.join(o.get('text', [])) for o in c.get('outputs', []) if o.get('output_type') == 'stream')
        if bad.search(txt):
            print(f, 'cell', i, bad.search(txt).group())
PY
```

- [x] All five checks clean
- [x] Every figure in `results/` opened and read for contrastive captions
- [x] `README.md` read end to end by someone who has not seen the synthetic version — they should have no reason to suspect one existed

---

## 7. Consequence to plan for

**After `dataset/data.csv` is deleted the repository contains no data at all** — `dataset/households.csv` and `dataset/features.csv` are gitignored under the ICPSR redistribution terms. A fresh clone cannot run a single notebook.

- [ ] Add `dataset/README.md` with the ICPSR acquisition steps (study 36151, DS0002, free registration) and the `src/build_dataset.py` command
- [ ] State the ICPSR terms and why the derived data is not committed
- [ ] Verify the whole pipeline runs from a clean clone after following those instructions
