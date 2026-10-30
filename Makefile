# Savings Goal Classifier: developer interface.
#
# `make` shows the help. Everything here is a thin wrapper: pyproject.toml, uv.lock, ruff, mypy,
# pytest and the `sgc` CLI stay the source of truth, and the logic that would differ between sh and
# cmd.exe lives in scripts/mk*.py (standard library only, unit-tested). Recipes are single commands
# with double-quoted arguments so they run under sh, cmd.exe and PowerShell alike (GNU Make 4+).
#
# Reading guide: configuration -> environment -> quality -> tests -> health -> data -> ML ->
# notebooks -> reproducibility -> docs -> reports -> cleanup -> CI -> git -> release.
# Lines of the form `target: deps ## [tag] text` and `VAR ?= default ## text` feed `make help`.

.DEFAULT_GOAL := help
MAKEFLAGS += --no-builtin-rules --no-print-directory
.SUFFIXES:

# --- Repository root --------------------------------------------------------------------------
# `make -f /path/to/Makefile target` from another directory re-runs itself from the root.
ROOT_DIR := $(patsubst %/,%,$(dir $(abspath $(firstword $(MAKEFILE_LIST)))))
ROOT_OK  := $(if $(MK_REROOTED),1,$(filter $(abspath $(CURDIR)),$(ROOT_DIR)))

ifeq ($(ROOT_OK),)
.PHONY: $(or $(MAKECMDGOALS),help)
$(or $(MAKECMDGOALS),help):
	@$(MAKE) -C "$(ROOT_DIR)" MK_REROOTED=1 $@
else

# --- Configuration ----------------------------------------------------------------------------
# Override on the command line (make test COV_MIN=90) or in the environment. No secrets belong here.
## auto = uv if installed, else the pip fallback | uv | pip
MODE ?= auto
## Interpreter for the helper and the pip fallback (default python3, python on Windows)
PYTHON ?=
## uv executable
UV ?= uv
## Pin the project interpreter, e.g. 3.13 (must satisfy requires-python)
PYTHON_VERSION ?=
## Project virtual environment
VENV ?= .venv
## Optional dependency extras for install targets, e.g. api
EXTRAS ?=
## Verbosity: 0 quiet, 1 normal, 2 echo every command
V ?= 1
## 1 = preview what would run or change; nothing is executed
DRY_RUN ?= 0
## 1 = pre-approve [confirm] and [publish] targets (never honoured in CI)
CONFIRM ?= 0
## N or auto: sets make -j and caps native threads for ML runs
JOBS ?=
## Random seed, exported to the package as SGC_SEED
SEED ?= 42
## Minimum coverage percent for coverage and test-ci
COV_MIN ?= 95
## Paths for `make test`
TEST_PATHS ?= tests
## Paths for test-unit and test-fast
UNIT_PATHS ?= tests/unit
## Paths for test-integration
INTEGRATION_PATHS ?= tests/integration
## Extra pytest arguments, e.g. -k name -x
PYTEST_ARGS ?=
## Seconds allowed per notebook cell
NB_TIMEOUT ?= 6000
## Scratch and output root (git-ignored)
BUILD_DIR ?= build
## Data tables, exported as SGC_DATA_DIR
DATA_DIR ?= dataset
## Raw ICPSR DS0002 file (never fetched or modified)
RAW_TSV ?= $(DATA_DIR)/ICPSR_36151/DS0002/36151-0002-Data.tsv
## Result artefacts figures read, exported as SGC_RESULTS_DIR
RESULTS_DIR ?= results
## Fitted models (earlier runs are archived, not overwritten)
MODEL_DIR ?= artifacts
## Built figures
FIGURES_DIR ?= $(BUILD_DIR)/figures
## Built report PDF
REPORT_DIR ?= $(BUILD_DIR)/report
## Sandbox for executed notebooks and their results
NB_RUN_DIR ?= $(BUILD_DIR)/notebooks
## Notebook exports
NB_EXPORT_DIR ?= $(NB_RUN_DIR)/export
## Output of the synthetic demo
DEMO_DIR ?= $(BUILD_DIR)/demo
## Git remote used by the git and release targets
REMOTE ?= origin

# Per-target arguments (set on the command line):
#   NB=03 | NB=03_baseline | NB=notebooks/03_baseline.ipynb   WITH_DEPS=1  FAST=1  KEEP_GOING=1  NB_FORMAT=html
#   MSG="feat: ..."  FILES="a.py b.py"  BUMP=patch|minor|major  OFFLINE=1  STRICT=1  N=3000  HOST= PORT=
NB         ?=
WITH_DEPS  ?=
FAST       ?=
KEEP_GOING ?=
NB_FORMAT  ?= html
MSG        ?=
FILES      ?=
BUMP       ?=
OFFLINE    ?=
STRICT     ?=
FORCE      ?=
N          ?= 3000
HOST       ?= 127.0.0.1
PORT       ?= 8000

# --- Plumbing ---------------------------------------------------------------------------------
ifeq ($(OS),Windows_NT)
PYTHON_CMD := $(or $(PYTHON),python)
else
PYTHON_CMD := $(or $(PYTHON),python3)
endif

PY_PROBE := $(shell "$(PYTHON_CMD)" -c "import sys; print('ok' if sys.version_info >= (3, 11) else 'old')" 2>&1)
ifneq ($(PY_PROBE),ok)
$(error Python 3.11+ is required to run this Makefile (tried "$(PYTHON_CMD)", got "$(PY_PROBE)"). Install uv, which can fetch Python: https://docs.astral.sh/uv/getting-started/installation/ - or install Python from https://www.python.org/downloads/ - or point PYTHON at one: make PYTHON=/path/to/python3.12 <target>)
endif

ifneq ($(JOBS),)
JOBS_N := $(if $(filter auto,$(JOBS)),$(shell "$(PYTHON_CMD)" -c "import os; print(os.cpu_count() or 1)"),$(JOBS))
MAKEFLAGS += -j$(JOBS_N) --output-sync=target
endif
ifeq ($(V),0)
MAKEFLAGS += -s
endif
ifeq ($(V),2)
Q :=
else
Q := @
endif

MK  := "$(PYTHON_CMD)" scripts/mk.py
RUN := $(MK) run --

export MK_ROOT := $(ROOT_DIR)
export MK_MODE := $(MODE)
export MK_PYTHON := $(PYTHON)
export MK_UV := $(UV)
export MK_PYTHON_VERSION := $(PYTHON_VERSION)
export MK_VENV := $(VENV)
export MK_EXTRAS := $(EXTRAS)
export MK_V := $(V)
export MK_DRY_RUN := $(DRY_RUN)
export MK_CONFIRM := $(CONFIRM)
export MK_JOBS := $(JOBS_N)
export MK_SEED := $(SEED)
export MK_COV_MIN := $(COV_MIN)
export MK_NB_TIMEOUT := $(NB_TIMEOUT)
export MK_BUILD_DIR := $(BUILD_DIR)
export MK_DATA_DIR := $(DATA_DIR)
export MK_RAW_TSV := $(RAW_TSV)
export MK_RESULTS_DIR := $(RESULTS_DIR)
export MK_MODELS_DIR := $(MODEL_DIR)
export MK_FIGURES_DIR := $(FIGURES_DIR)
export MK_REPORT_DIR := $(REPORT_DIR)
export MK_NB_RUN_DIR := $(NB_RUN_DIR)
export MK_NB_EXPORT_DIR := $(NB_EXPORT_DIR)
export MK_DEMO_DIR := $(DEMO_DIR)
export MK_REMOTE := $(REMOTE)
export MK_OFFLINE := $(OFFLINE)
export MK_ARG_NB := $(NB)
export MK_ARG_WITH_DEPS := $(WITH_DEPS)
export MK_ARG_FAST := $(FAST)
export MK_ARG_KEEP_GOING := $(KEEP_GOING)
export MK_ARG_FORMAT := $(NB_FORMAT)
export MK_ARG_MSG := $(MSG)
export MK_ARG_FILES := $(FILES)
export MK_ARG_BUMP := $(BUMP)
export MK_ARG_STRICT := $(STRICT)
export MK_ARG_FORCE := $(FORCE)
export MK_ARG_N := $(N)
export MK_ARG_HOST := $(HOST)
export MK_ARG_PORT := $(PORT)

HOUSEHOLDS := $(DATA_DIR)/households.parquet
FEATURES   := $(DATA_DIR)/features.parquet
TEST_DIR   := $(BUILD_DIR)/test

##@ Help
help: ## Show this help (the default target)
	$(Q)$(MK) help

##@ Setup and environment
setup: install-dev check-env ## [env] First-time setup: build the dev environment, then verify it
	$(Q)$(MK) say "environment ready. Next: make doctor | make check | make hooks-install"

install: ## [env] Runtime dependencies only (uv sync --locked --no-dev --inexact)
	$(Q)$(MK) install

install-dev: ## [env] Runtime + dev tools from uv.lock; a no-op when already in sync (pip fallback without uv)
	$(Q)$(MK) install-dev

sync: ## [env] Force-sync the environment to uv.lock exactly
	$(Q)$(MK) sync

ensure-env: ## [safe] Fail with instructions if the environment is missing or stale (used by tool targets)
	$(Q)$(MK) ensure-env

check-env: ## [safe] Python version, environment, dev tools, lock consistency
	$(Q)$(MK) check-env

check-tools: ## [safe] Detect uv, git, python, pre-commit, latexmk with install hints
	$(Q)$(MK) check-tools

python-info: ## [safe] requires-python, supported minors, selected interpreters, venv
	$(Q)$(MK) python-info

doctor: ## [safe] Diagnose tools, environment, lock and git state (PASS/WARN/FAIL)
	$(Q)$(MK) doctor

lock-check: ## [safe] Fail if uv.lock is out of date with pyproject.toml
	$(Q)$(MK) lock-check

##@ Development workflow
dev: install-dev check ## [env] Prepare the environment, then run the fast checks
all: install-dev validate ## [env] Environment + the full validation gate (never commits, pushes, deletes or releases)
check: lint format-check typecheck test-fast ## [safe] Fast gate: lint, format, types, unit tests without coverage
validate: quality test lock-check docs-check notebook-check ## [safe] Full gate: quality, tests with coverage, lock, docs, notebooks

##@ Code quality
lint: ensure-env ## [safe] ruff check
	$(Q)$(RUN) ruff check .

lint-fix: ensure-env ## [writes] ruff check --fix (edits files in place)
	$(Q)$(RUN) ruff check --fix .

format: ensure-env ## [writes] ruff format (edits files in place)
	$(Q)$(RUN) ruff format .

format-check: ensure-env ## [safe] ruff format --check
	$(Q)$(RUN) ruff format --check .

typecheck: ensure-env ## [safe] mypy (strict, configured in pyproject.toml)
	$(Q)$(RUN) mypy

import-check: ensure-env ## [safe] Import every savings_goal module
	$(Q)$(MK) import-check

quality: lint format-check typecheck import-check ## [safe] lint + format-check + typecheck + import-check

precommit: ## [safe] Run every pre-commit hook on all files (needs pre-commit)
	$(Q)$(MK) precommit

hooks-install: ## [git] Install the pre-commit git hook (writes .git/hooks/pre-commit)
	$(Q)$(MK) hooks-install

##@ Testing
test: ensure-env ## [safe] Whole test suite with coverage (pyproject.toml addopts)
	$(Q)$(RUN) pytest $(TEST_PATHS) $(PYTEST_ARGS)

test-unit: export COVERAGE_FILE := .coverage.unit
test-unit: ensure-env ## [safe] Unit tests (tests/unit)
	$(Q)$(RUN) pytest $(UNIT_PATHS) $(PYTEST_ARGS)

test-integration: export COVERAGE_FILE := .coverage.integration
test-integration: ensure-env ## [safe] Integration tests (tests/integration: CLI, Makefile)
	$(Q)$(RUN) pytest $(INTEGRATION_PATHS) $(PYTEST_ARGS)

test-fast: ensure-env ## [safe] Unit tests only, no coverage, stop at the first failure
	$(Q)$(RUN) pytest $(UNIT_PATHS) --no-cov -x -p no:cacheprovider $(PYTEST_ARGS)

coverage: export COVERAGE_FILE := .coverage.report
coverage: ensure-env ## [writes] Coverage report (terminal + HTML in build/test/htmlcov) enforcing COV_MIN
	$(Q)$(RUN) pytest $(TEST_PATHS) --cov-report=term-missing --cov-report=html:$(TEST_DIR)/htmlcov --cov-fail-under=$(COV_MIN) $(PYTEST_ARGS)

test-ci: export COVERAGE_FILE := .coverage.ci
test-ci: ensure-env ## [writes] CI run: COV_MIN enforced, coverage.xml and junit.xml in build/test
	$(Q)$(RUN) pytest $(TEST_PATHS) -p no:cacheprovider --cov-report=term-missing --cov-report=xml:$(TEST_DIR)/coverage.xml --cov-fail-under=$(COV_MIN) --junitxml=$(TEST_DIR)/junit.xml $(PYTEST_ARGS)

##@ Project health
health: ## [safe] Structure, imports, test and quality config, notebooks, docs, output dirs, data expectations
	$(Q)$(MK) health

audit: ## [safe] No secrets or restricted data tracked, .gitignore sound, dependency vulnerabilities (uv audit)
	$(Q)$(MK) audit

verify: ## [safe] lock + version + audit + health in one read-only report (STRICT=1 fails on warnings)
	$(Q)$(MK) verify

##@ Data pipeline (raw ICPSR file required; it is never downloaded or modified)
data: data-pipeline ## [writes] Same as data-pipeline
data-check: ## [safe] Which raw, derived and model files exist, and how to create the missing ones
	$(Q)$(MK) data-check

data-prepare: $(HOUSEHOLDS) ## [writes] RAW_TSV -> households.parquet (validated build; skipped when up to date, -B forces)
	$(Q)$(MK) say "households table is up to date: $(HOUSEHOLDS)"

$(HOUSEHOLDS): $(wildcard $(RAW_TSV)) | ensure-env
	$(Q)$(MK) data-prepare

data-features: $(FEATURES) ## [writes] households.parquet -> features.parquet (PSU-grouped split; skipped when up to date)
	$(Q)$(MK) say "feature table is up to date: $(FEATURES)"

$(FEATURES): $(HOUSEHOLDS) | ensure-env
	$(Q)$(MK) data-features

data-validate: ensure-env ## [safe] Validate households.parquet against the Pandera contract
	$(Q)$(MK) data-validate

data-pipeline: data-prepare data-features data-validate ## [writes] prepare -> features -> validate
data-clean: export MK_ARG_WHAT := data
data-clean: ## [confirm] Delete derived tables (households, features, oof); never the raw files
	$(Q)$(MK) clean

demo: ensure-env ## [writes] End-to-end run on a SYNTHETIC survey (no ICPSR data); output in DEMO_DIR
	$(Q)$(MK) demo

##@ ML workflows (notebooks run in a sandbox; tracked results change only via promote-results)
train: $(FEATURES) | ensure-env ## [writes] Fit both scoring tiers into MODEL_DIR; archives any previous run first
	$(Q)$(MK) train

eda: export MK_ARG_NB := 01_eda_and_leakage_check
eda: ensure-env ## [writes] Notebook 01: EDA and the leakage check
	$(Q)$(MK) notebook-run

baseline: export MK_ARG_NB := 03_baseline
baseline: ensure-env ## [writes] Notebook 03: baselines
	$(Q)$(MK) notebook-run

compare: export MK_ARG_NB := 04_model_comparison
compare: ensure-env ## [writes] Notebook 04: model comparison and tuning (slow)
	$(Q)$(MK) notebook-run

explain: export MK_ARG_NB := 05_explainability
explain: ensure-env ## [writes] Notebook 05: grouped SHAP and permutation importance (slow)
	$(Q)$(MK) notebook-run

cluster: export MK_ARG_NB := 06_clustering_personas
cluster: ensure-env ## [writes] Notebook 06: ILR clustering / personas
	$(Q)$(MK) notebook-run

business-analysis: export MK_ARG_NB := 07_business_translation
business-analysis: ensure-env ## [writes] Notebook 07: business translation (also refreshes dataset/oof.parquet)
	$(Q)$(MK) notebook-run

regression: export MK_ARG_NB := 08_savings_rate_regression
regression: ensure-env ## [writes] Notebook 08: quantile regression on the savings rate
	$(Q)$(MK) notebook-run

outreach: export MK_ARG_NB := 09_outreach_simulation
outreach: ensure-env ## [writes] Notebook 09: outreach what-if (needs oof.parquet from 07)
	$(Q)$(MK) notebook-run

external-validity: export MK_ARG_NB := 10_external_validity
external-validity: ensure-env ## [writes] Notebook 10: HCES comparison, leakage curve (HCES files optional)
	$(Q)$(MK) notebook-run

ml: train notebooks-run ## [writes] Full ML workflow: train, then every notebook in dependency order (FAST=1 skips 04, 05)

##@ Notebooks (originals are never modified or cleared)
notebooks: notebook-list notebook-check ## [safe] List the notebooks with their derived order, then validate them
notebook-list: ## [safe] Notebooks in dependency order, what each needs and writes
	$(Q)$(MK) notebook-list

notebook-check: ## [safe] Parse, syntax-check, kernelspec, machine-path check, dependency graph, inputs present
	$(Q)$(MK) notebook-check

notebook-run: ensure-env ## [writes] Run one notebook: NB=03 (WITH_DEPS=1 adds upstream notebooks); executed copy in NB_RUN_DIR
	$(Q)$(if $(NB),$(MK) notebook-run,$(error NB is required, e.g. make notebook-run NB=03_baseline))

notebooks-run: export MK_ARG_NB :=
notebooks-run: ensure-env ## [writes] Run every notebook in dependency order (FAST=1, KEEP_GOING=1, NB_TIMEOUT=)
	$(Q)$(MK) notebook-run

notebook-status: ## [safe] Result of the last sandboxed run
	$(Q)$(MK) notebook-status

notebook-export: ensure-env ## [writes] Export saved notebook outputs to NB_EXPORT_DIR (NB_FORMAT=html|markdown|script|rst)
	$(Q)$(MK) notebook-export

##@ Reproducibility
reproducibility: lock-check seed-check environment-report experiment-info ## [writes] Seed, lock, environment and experiment records together
seed-check: ensure-env ## [safe] SEED reaches the package; lists hard-coded seeds that ignore it
	$(Q)$(MK) seed-check

environment-report: ensure-env ## [writes] Versions, lock hash, commit, seed -> build/repro/environment.json
	$(Q)$(MK) environment-report

experiment-info: ## [safe] Recorded training/notebook/reference runs and drift from the reference environment
	$(Q)$(MK) experiment-info

##@ Documentation (Markdown only: there is no Sphinx/MkDocs to build)
docs: docs-check ## [safe] Same as docs-check
docs-check: ## [safe] docs-lint + docs-links
	$(Q)$(MK) docs-check

docs-lint: ## [safe] Markdown structure (balanced code fences)
	$(Q)$(MK) docs-lint

docs-links: ## [safe] Relative links and #anchors resolve (external URLs are not fetched)
	$(Q)$(MK) docs-links

##@ Reports and figures (built into build/; tracked files change only via promote-*)
figures-check: ## [safe] Are the result files make_figures.py reads present?
	$(Q)$(MK) figures-check

figures: ensure-env ## [writes] Build the report figures into FIGURES_DIR from RESULTS_DIR
	$(Q)$(MK) figures

report: ## [writes] Build project/report.tex into REPORT_DIR with latexmk (needs a TeX distribution)
	$(Q)$(MK) report

reports: figures report ## [writes] figures, then report
report-all: ## [writes] Slow: notebooks-run, figures from those results, then the report
	$(Q)$(MAKE) notebooks-run
	$(Q)$(MAKE) figures RESULTS_DIR="$(NB_RUN_DIR)/results"
	$(Q)$(MAKE) report

promote-results: export MK_ARG_WHAT := results
promote-results: ## [confirm] Copy sandbox notebook results over results/ (backs up replaced files)
	$(Q)$(MK) promote

promote-figures: export MK_ARG_WHAT := figures
promote-figures: ## [confirm] Copy built figures over project/figures/ (backs up replaced files)
	$(Q)$(MK) promote

promote-report: export MK_ARG_WHAT := report
promote-report: ## [confirm] Copy the built PDF over project/report.pdf (backs up the old one)
	$(Q)$(MK) promote

##@ Cleanup (allowlisted paths only; DRY_RUN=1 lists exactly what would go)
clean: export MK_ARG_WHAT := clean
clean: ## [writes] Caches, coverage data, demo output and repro scratch files
	$(Q)$(MK) clean

clean-cache: export MK_ARG_WHAT := cache
clean-cache: ## [writes] __pycache__, .pytest_cache, .mypy_cache, .ruff_cache, notebook checkpoints
	$(Q)$(MK) clean

clean-test: export MK_ARG_WHAT := test
clean-test: ## [writes] Coverage data, HTML coverage, junit/coverage XML
	$(Q)$(MK) clean

clean-notebooks: export MK_ARG_WHAT := notebooks
clean-notebooks: ## [confirm] Sandboxed notebook runs and exports (never the notebooks themselves)
	$(Q)$(MK) clean

figures-clean: export MK_ARG_WHAT := figures
figures-clean: ## [confirm] Built figures in FIGURES_DIR (never project/figures)
	$(Q)$(MK) clean

clean-reports: export MK_ARG_WHAT := reports
clean-reports: ## [confirm] Built figures and report in build/ (never tracked outputs)
	$(Q)$(MK) clean

clean-models: export MK_ARG_WHAT := models
clean-models: ## [confirm] Fitted models in MODEL_DIR (archived runs are kept)
	$(Q)$(MK) clean

clean-all: export MK_ARG_WHAT := all
clean-all: ## [confirm] clean + notebooks + reports + models. Never data, results, sources or backups
	$(Q)$(MK) clean

##@ CI (non-interactive; .github/workflows/ci.yml calls these)
ci-quality: lint format-check typecheck lock-check ## [safe] ruff, ruff format, mypy, lock consistency
ci-test: test-ci ## [writes] pytest with coverage threshold and XML reports
ci-demo: demo ## [writes] Synthetic end-to-end run (the old "End-to-end demo" step)
ci-docs: docs-check ## [safe] Markdown lint and links
ci-notebooks: notebook-check ## [safe] Notebook structure, order and dependency checks (they are not executed: no data in CI)
ci: install-dev ci-quality ci-test ci-demo ci-docs ci-notebooks ## [env] Everything CI runs, in order

##@ Git workflow (only when named; never force, amend, rewrite history or bypass hooks)
git-status: ## [safe] git status --short --branch
	$(Q)$(MK) git-status

git-branch: ## [safe] Branch, detected default branch, upstream, ahead/behind, remote
	$(Q)$(MK) git-branch

git-diff: ## [safe] Unstaged, staged and untracked summary
	$(Q)$(MK) git-diff

git-check: ## [safe] Repo, detached HEAD, in-progress operations, conflicts, staged secrets
	$(Q)$(MK) git-check

git-stage: ## [git] Stage tracked changes, or exactly FILES="a b"; refuses secrets, ignored and restricted data
	$(Q)$(MK) git-stage

git-commit: ## [git] Commit what is staged: MSG="type(scope): summary" (runs hooks; refuses detached HEAD)
	$(Q)$(MK) git-commit

git-pull: ## [git] Fetch and fast-forward only; refuses a dirty tracked tree or divergence
	$(Q)$(MK) git-pull

git-sync: ## [git] git-pull, then report unpushed commits (does not push)
	$(Q)$(MK) git-sync

git-push: ## [publish] Plain push of the current branch (sets upstream on first push); never force
	$(Q)$(MK) git-push

git-ready: git-check check ## [safe] Commit readiness: git-check and the fast gate

##@ Versions and releases (SemVer; version source: [project].version in pyproject.toml)
version: ## [safe] Current version, tag name, latest tag, next patch/minor/major
	$(Q)$(MK) version

version-check: ## [safe] SemVer valid; pyproject.toml, CITATION.cff and uv.lock agree
	$(Q)$(MK) version-check

version-patch: export MK_ARG_PART := patch
version-patch: ## [writes] Bump the patch version in pyproject.toml, CITATION.cff, uv.lock (clean tree; no commit)
	$(Q)$(MK) version-bump

version-minor: export MK_ARG_PART := minor
version-minor: ## [writes] Bump the minor version (no commit)
	$(Q)$(MK) version-bump

version-major: export MK_ARG_PART := major
version-major: ## [writes] Bump the major version (no commit)
	$(Q)$(MK) version-bump

release-notes: ## [writes] Release notes from commits since the last tag -> build/release/ (BUMP= for the next version)
	$(Q)$(MK) release-notes

release-check: release-state version-check lock-check quality test-ci docs-check notebook-check health ## [safe] Strict gate: tree, branch, tag, version, quality, tests, docs, notebooks, health
release-state: ## [safe] Branch, clean tree, upstream, tag availability, monotonic version (BUMP= for the next one)
	$(Q)$(MK) release-state

release-need-bump:
	$(Q)$(if $(BUMP),,$(error BUMP=patch|minor|major is required; to tag the version already in pyproject.toml use: make release-tag))

release-tag: release-check ## [git] Tag the current version (annotated, local only) after the strict gate
	$(Q)$(MK) release-tag

release: release-need-bump release-check ## [git] BUMP=patch|minor|major: bump, commit the version files, tag. Local only
	$(Q)$(MK) release

release-push: ## [publish] Push the release branch and its tag (CONFIRM=1); never force, never --tags
	$(Q)$(MK) release-push

release-verify: ## [safe] Tag exists, is annotated, matches the version and HEAD, and is (or is not yet) on the remote
	$(Q)$(MK) release-verify

##@ Serving
serve: ensure-env ## [safe] Run the FastAPI scoring service (needs: make install-dev EXTRAS=api)
	$(Q)$(MK) serve

# --- Declarations -----------------------------------------------------------------------------
.PHONY: help setup install install-dev sync ensure-env check-env check-tools python-info doctor lock-check \
	dev all check validate lint lint-fix format format-check typecheck import-check quality precommit hooks-install \
	test test-unit test-integration test-fast coverage test-ci health audit verify \
	data data-check data-prepare data-features data-validate data-pipeline data-clean demo \
	train eda baseline compare explain cluster business-analysis regression outreach external-validity ml \
	notebooks notebook-list notebook-check notebook-run notebooks-run notebook-status notebook-export \
	reproducibility seed-check environment-report experiment-info docs docs-check docs-lint docs-links \
	figures-check figures report reports report-all promote-results promote-figures promote-report \
	clean clean-cache clean-test clean-notebooks figures-clean clean-reports clean-models clean-all \
	ci-quality ci-test ci-demo ci-docs ci-notebooks ci \
	git-status git-branch git-diff git-check git-stage git-commit git-pull git-sync git-push git-ready \
	version version-check version-patch version-minor version-major release-notes release-check release-state \
	release-need-bump release-tag release release-push release-verify serve

# Order matters here (prerequisites run left to right), and these targets share outputs or state.
# Independent read-only checks (lint, typecheck, tests, ...) stay parallel-safe under `make -j`.
.NOTPARALLEL: setup dev all ci ml data-pipeline report-all reports reproducibility release-check release release-tag git-ready

endif
