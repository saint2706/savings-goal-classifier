# The Makefile: developer guide

One `Makefile` at the repository root is the single command interface for setup, quality checks, tests, the data and ML workflows, notebooks, reports, cleanup, Git and releases. It wraps what the repository already defines (`pyproject.toml`, `uv.lock`, ruff, mypy, pytest, the `sgc` CLI) and adds no second copy of any setting. GitHub Actions calls the same targets, so `make ci` reproduces the CI job locally.

```bash
make            # the help screen (the default target)
make setup      # first-time environment
make doctor     # what is installed, what is missing, what is wrong
make check      # fast gate: lint, format, types, unit tests
make validate   # full gate: adds coverage threshold, lock, docs and notebook checks
```

Every target carries a tag in `make help`:

| Tag | Meaning |
| --- | --- |
| `[safe]` | Read-only. |
| `[writes]` | Creates generated files in `build/`, `artifacts/` or the tool caches; safe to repeat. |
| `[env]` | Changes the Python environment (`.venv`). |
| `[confirm]` | Destructive: needs `CONFIRM=1` or an interactive `yes`; preview with `DRY_RUN=1`. |
| `[git]` | Changes local Git state, only when you name that target. |
| `[publish]` | Pushes to the remote; needs `CONFIRM=1` for releases. |

Nothing commits, pushes, tags, deletes or publishes as a side effect of `make`, `make all`, `make check`, `make validate` or `make ci`. A test (`tests/integration/test_makefile.py`) checks that.

## 1. Prerequisites

| Need | Why | Install |
| --- | --- | --- |
| GNU Make 4.0+ (tested with 4.4.1) | runs the Makefile | Windows: `scoop install make`, `choco install make` or `winget install ezwinports.make`. macOS ships Make 3.81: `brew install make` and run `gmake`. Linux: `sudo apt install make`. |
| Python 3.11+ on `PATH` | runs the helper scripts in `scripts/` (standard library only) | python.org, or `uv python install 3.13`. |
| [uv](https://docs.astral.sh/uv/) (recommended) | creates the environment from `uv.lock` | `winget install --id=astral-sh.uv -e`, `brew install uv`, or `curl -LsSf https://astral.sh/uv/install.sh | sh`. |
| Git | the git and release targets | any recent Git. |
| pre-commit (optional) | `make precommit`, `make hooks-install` | `uv tool install pre-commit`. |
| latexmk (optional) | `make report` | MiKTeX (Windows), `brew install --cask mactex-no-gui`, or `sudo apt install latexmk texlive-latex-extra texlive-publishers`. |

`make check-tools` reports which of these it can find, with the install command for each one that is missing.

The Python the helper runs under only needs to be 3.11+. The Python the *project* uses is a separate question: it comes from `requires-python` in `pyproject.toml` (currently `>=3.11,<3.15`, so 3.11 to 3.14), read on every run, never hard-coded. `make python-info` shows the range, the interpreters it found and the one in `.venv`.

## 2. Platforms

| Platform | Status |
| --- | --- |
| Linux | CI runs `make install-dev`, `make ci-quality`, `make ci-test`, `make ci-demo`, `make ci-docs` and `make ci-notebooks` on Ubuntu 24.04 for Python 3.11, 3.13 and 3.14. |
| Windows | Developed and run on Windows 11 with GNU Make 4.4.1 under Git Bash (`sh`) and under PowerShell with no `sh` on `PATH` (Make then uses `cmd.exe`). |
| macOS | Not run by the author. Use Homebrew's GNU Make (`gmake`) rather than the system 3.81. |

Recipes are single commands with double-quoted arguments, so they work under `sh` and `cmd.exe` alike. Anything that would differ (path handling, deleting trees, Git plumbing, notebook order) is in `scripts/mk*.py`, which is plain Python. Bash is never assumed. `.gitattributes` pins the Makefile and the scripts to LF line endings, because a stray `\r` breaks Make on every platform.

The path of the repository may contain spaces. The settings that name directories (`BUILD_DIR`, `DATA_DIR` and so on) should not, because Make cannot use such names as file targets.

## 3. Installation and the environment

```bash
make setup           # = make install-dev, then make check-env
make install-dev     # runtime + dev tools from uv.lock; does nothing when already in sync
make install         # runtime dependencies only
make sync            # force-sync to uv.lock exactly (removes extraneous packages)
make install-dev EXTRAS=api              # add the optional `api` extra (needed by `make serve`)
make install-dev PYTHON_VERSION=3.13     # pick the interpreter (must satisfy requires-python)
```

**uv first.** With uv installed, installation is `uv sync --locked`: the environment matches `uv.lock` exactly, and a stale lock is an error rather than a silent re-resolve. `make install-dev` first asks uv whether the environment is already in sync and, if so, changes nothing.

**pip fallback.** Without uv (or with `MODE=pip`) the Makefile creates `.venv` with a compatible interpreter and runs `pip install -e .` plus the `dev` dependency group from `pyproject.toml`. It says every time that this resolves from the bounds in `pyproject.toml` and **does not honour `uv.lock`**, so the environment can differ from CI. `make lock-check` then reports that lock consistency is not verified, and `make release-check` cannot pass without uv. The fallback never substitutes an unsupported interpreter: if none satisfies `requires-python` (and `PYTHON_VERSION`, if set) it stops and prints how to get one. It never deletes or recreates an existing `.venv`.

**Tool targets do not install anything.** `make lint`, `make test` and the rest first run `make ensure-env`, a read-only check. If the environment is missing or out of date with `pyproject.toml` / `uv.lock` (editing `pyproject.toml` is enough), they stop with `Run: make install-dev`. The environment changes only when you run an install target.

## 4. Target reference

### Setup and environment

| Target | What it does |
| --- | --- |
| `help` | The help screen (default target). |
| `setup` | `install-dev`, then `check-env`. |
| `install` | Runtime dependencies only. Does not uninstall dev tools already present. |
| `install-dev` | Runtime plus dev tools; idempotent. |
| `sync` | Force-sync the environment to `uv.lock`. |
| `ensure-env` | Fail with instructions if the environment is missing or stale (used by every tool target). |
| `check-env` | Python version, environment, dev tools, lock consistency. |
| `check-tools` | Detect uv, git, python, pre-commit, latexmk; print install hints. |
| `python-info` | `requires-python`, supported minors, interpreters, venv. |
| `doctor` | Tools, environment, lock and Git state as PASS / WARN / FAIL. |
| `lock-check` | Fail if `uv.lock` is out of date with `pyproject.toml`. |

### Development workflow

| Target | What it does |
| --- | --- |
| `dev` | `install-dev`, then `check`. |
| `check` | `lint`, `format-check`, `typecheck`, `test-fast`. |
| `validate` | `quality`, `test` (with coverage), `lock-check`, `docs-check`, `notebook-check`. |
| `all` | `install-dev`, then `validate`. Never commits, pushes, deletes or releases. |

### Code quality

| Target | What it does |
| --- | --- |
| `lint` | `ruff check .` |
| `lint-fix` | `ruff check --fix .` (edits files). |
| `format` | `ruff format .` (edits files). |
| `format-check` | `ruff format --check .` |
| `typecheck` | `mypy` (strict; covers `src/savings_goal` and `scripts`). |
| `import-check` | Imports every `savings_goal` module. |
| `quality` | `lint`, `format-check`, `typecheck`, `import-check`. |
| `precommit` | Runs every hook in `.pre-commit-config.yaml` on all files. |
| `hooks-install` | Installs the git hook (`pre-commit install`). |

The pre-commit hooks run `make lint`, `make format-check` and `make typecheck`, so they use the ruff and mypy pinned in `uv.lock` and the settings in `pyproject.toml`, and also work in the pip fallback. The remaining hooks (merge-conflict markers, YAML/TOML syntax, private keys, files over 5 MB) come from `pre-commit-hooks`.

### Testing

| Target | What it does |
| --- | --- |
| `test` | The whole suite (`TEST_PATHS`) with coverage, as `pyproject.toml` configures it. |
| `test-unit` | `tests/unit`. |
| `test-integration` | `tests/integration` (the CLI and this Makefile). |
| `test-fast` | Unit tests only, no coverage, stop at the first failure. |
| `coverage` | Terminal and HTML report (`build/test/htmlcov`), failing under `COV_MIN`. |
| `test-ci` | What CI runs: `COV_MIN` enforced, `build/test/coverage.xml` and `junit.xml` written. |

The suite has two directories and no custom markers; there is no separate "slow" marker, so `test-fast` is "unit tests without coverage". Each target writes its own coverage data file, so `make -j test-unit test-integration` cannot corrupt one.

### Project health

| Target | What it does |
| --- | --- |
| `health` | Structure, importability, test and quality configuration, notebooks, docs, output directories (writable?), data and model expectations. |
| `audit` | No credentials or restricted data tracked, `.gitignore` covers them, no file over 5 MB, `uv audit` for known vulnerabilities. |
| `verify` | `lock-check`, `version-check`, `audit` and `health` in one read-only report. `STRICT=1` makes warnings fail. |

Missing restricted inputs (the ICPSR file, derived tables, trained models) are **warnings**, not failures, because a fresh clone cannot have them. Broken structure, failing imports, a stale lock or a leaked secret are **failures**.

### Data pipeline

The raw IHDS-II file is licensed (ICPSR forbids redistribution). Nothing here downloads, replaces or modifies it. Put `36151-0002-Data.tsv` at `RAW_TSV` (default `dataset/ICPSR_36151/DS0002/36151-0002-Data.tsv`) or pass `RAW_TSV=...`. Layout: `dataset/` holds the raw download plus `households.parquet`, `features.parquet` and `oof.parquet`, all git-ignored; see [`dataset/README.md`](../dataset/README.md).

| Target | What it does |
| --- | --- |
| `data` | Same as `data-pipeline`. |
| `data-check` | Which raw, derived and model files exist, and the command that creates each missing one. |
| `data-prepare` | `RAW_TSV` to `households.parquet` through `sgc build` (Polars, validated against the Pandera schema). A file rule: skipped when the output is newer than the input; `make -B data-prepare` forces it. |
| `data-features` | `households.parquet` to `features.parquet` through `sgc features`; skipped when up to date. |
| `data-validate` | Validates `households.parquet` against the Pandera contract. |
| `data-pipeline` | `data-prepare`, `data-features`, `data-validate`, in order. |
| `data-clean` | Deletes the derived tables (`[confirm]`); never the raw files. |
| `demo` | `sgc demo` on a **synthetic** survey: no ICPSR data, output in `DEMO_DIR`. `N=` sets the household count. |

There is no separate "clean" or "preprocess" stage in the code (`sgc build` does cleaning, preprocessing and validation in one validated pass), so the Makefile does not pretend there is.

### ML workflows

The analysis lives in the `savings_goal` package and the notebooks; the Makefile only runs it. Training is one CLI command (`sgc train`). Evaluation, comparison, explanation and the business analysis are notebook logic, so those targets run the corresponding notebook, **on a copy, into a sandbox** (see section 6).

| Target | What it runs |
| --- | --- |
| `train` | `sgc train`: fits both scoring tiers into `MODEL_DIR` (default `artifacts/`). An earlier run is moved to `artifacts/previous/<timestamp>/` first, never overwritten, and `train_manifest.json` records seed, versions, commit and the SHA-256 of the feature table. |
| `eda` | Notebook 01, EDA and leakage check. |
| `baseline` | Notebook 03, baselines. |
| `compare` | Notebook 04, model comparison and tuning (slow). |
| `explain` | Notebook 05, grouped SHAP and permutation importance (slow). |
| `cluster` | Notebook 06, ILR clustering and personas. |
| `business-analysis` | Notebook 07, business translation. Also refreshes `dataset/oof.parquet`. |
| `regression` | Notebook 08, quantile regression on the savings rate. |
| `outreach` | Notebook 09, outreach what-if (needs `oof.parquet` from 07). |
| `external-validity` | Notebook 10, HCES comparison and leakage curve (the HCES files are optional). |
| `ml` | `train`, then every notebook in dependency order. `FAST=1` skips 04 and 05 as `sgc reproduce --fast` does. |

There is no `evaluate` or `regression-train` target: held-out evaluation is part of notebooks 04 and 07, and a Make target for it would duplicate that logic. The full run takes about 25 minutes on four cores (the README's figure); `DRY_RUN=1` prints the plan first.

### Notebooks

| Target | What it does |
| --- | --- |
| `notebooks` | `notebook-list`, then `notebook-check`. |
| `notebook-list` | The notebooks in dependency order, what each needs and writes. |
| `notebook-check` | Valid JSON and nbformat, a kernelspec, every code cell parses, no machine-specific absolute path in a source line or saved output, an acyclic dependency graph, inputs present (warnings when the restricted tables are absent). |
| `notebook-run` | One notebook: `NB=03`, `NB=03_baseline` or `NB=notebooks/03_baseline.ipynb`. `WITH_DEPS=1` adds the notebooks it depends on. |
| `notebooks-run` | Every notebook in dependency order. |
| `notebook-status` | Per-notebook result of the last run. |
| `notebook-export` | Exports the saved outputs (no re-execution) as `NB_FORMAT` (`html`, `markdown`, `script`, `rst`) into `NB_EXPORT_DIR`. |

The execution order is **derived**, not assumed from the numbers: the helper scans each notebook for `write_result` / `read_result` calls and for the parquet files it writes and reads, then sorts topologically. Today that gives 01 to 10 in numeric order with real edges 07 after 03 and 04 (it reads their results) and 09 after 07 (it reads `oof.parquet`); a new notebook is placed by what it reads, not by its filename.

### Reports and figures

| Target | What it does |
| --- | --- |
| `figures-check` | Are the result files `project/figures/make_figures.py` reads present? (Read from the script itself.) |
| `figures` | Builds the report figures into `FIGURES_DIR` with the project's own script and `savings_goal.viz` styling. |
| `report` | Builds `project/report.tex` with latexmk into `REPORT_DIR`, using the committed figures overlaid by any in `FIGURES_DIR`. |
| `reports` | `figures`, then `report`. |
| `report-all` | `notebooks-run`, then `figures` from those results, then `report`. Slow. |
| `promote-results` | Copies sandbox notebook results over `results/` (`[confirm]`, backs up each replaced file). |
| `promote-figures` | Copies built figures over `project/figures/` (`[confirm]`, backs up). |
| `promote-report` | Copies the built PDF over `project/report.pdf` (`[confirm]`, backs up). |

Builds never write into `results/`, `project/` or any other tracked path: `FIGURES_DIR=project/figures` is refused. Updating the committed artefacts is a separate, explicit, backed-up step: preview with `make promote-results DRY_RUN=1`, apply with `make promote-results CONFIRM=1`. Backups go to `build/backups/<timestamp>/` and are never cleaned.

### Reproducibility

| Target | What it does |
| --- | --- |
| `reproducibility` | `lock-check`, `seed-check`, `environment-report`, `experiment-info`. |
| `seed-check` | Confirms `SEED` reaches the package, and lists hard-coded seeds that ignore it. |
| `environment-report` | Writes `build/repro/environment.json`: Python, package versions, `uv.lock` SHA-256, commit, dirty flag, seed. |
| `experiment-info` | Shows the recorded training, notebook and reference runs, and warns where this environment differs from the committed reference run. |

Central control: `SEED` (default 42) is exported as `SGC_SEED`, which `savings_goal.config.SEED` reads; every seeded function in the package defaults to that value, so one setting changes the CV splits, clustering, bootstraps, XGBoost and the synthetic demo together. `make seed-check` reports the exceptions (an intentional tie-breaking generator in `evaluation/transport.py`, and a documentation string in notebook 04). `DATA_DIR` and `RESULTS_DIR` are exported the same way (`SGC_DATA_DIR`, `SGC_RESULTS_DIR`).

What this does **not** guarantee: the IHDS-II data cannot be redistributed, so a reader needs their own download (the manifest records its SHA-256 for comparison); results move in the third or fourth decimal between XGBoost and Python versions (see "Numerical sensitivity" in the README); and only an environment synced from `uv.lock` is pinned.

### Documentation

There is no Sphinx or MkDocs: the documentation is Markdown rendered by GitHub, so there is no build step and no `docs-build` target.

| Target | What it does |
| --- | --- |
| `docs` | Same as `docs-check`. |
| `docs-check` | `docs-lint` and `docs-links`. |
| `docs-lint` | Markdown structure (balanced code fences) for `README.md`, `dataset/README.md`, `walkthrough/*.md`, `docs/*.md`. |
| `docs-links` | Every relative link and `#anchor` resolves (GitHub heading slugs). External URLs are not fetched. |

### Cleanup

| Target | Removes | Confirmation |
| --- | --- | --- |
| `clean` | caches, coverage data, demo output, repro/release scratch | no |
| `clean-cache` | `__pycache__`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`, notebook checkpoints | no |
| `clean-test` | `.coverage*`, `htmlcov`, `build/test` | no |
| `clean-notebooks` | the notebook sandbox and exports (`NB_RUN_DIR`, `NB_EXPORT_DIR`) | yes |
| `figures-clean` | built figures (`FIGURES_DIR`) | yes |
| `clean-reports` | built figures and report | yes |
| `clean-models` | the current models in `MODEL_DIR` | yes |
| `clean-all` | `clean` + notebooks + reports + models | yes |
| `data-clean` | derived tables in `DATA_DIR` | yes |

Never removed by any target: source, tests, notebooks, `results/`, `project/`, `walkthrough/`, the raw ICPSR files, `README.md` and the other config files, `build/backups/`, `artifacts/previous/`.

### CI

| Target | What it does |
| --- | --- |
| `ci-quality` | `lint`, `format-check`, `typecheck`, `lock-check`. |
| `ci-test` | `test-ci`. |
| `ci-demo` | `demo`: the old "End-to-end demo" step. |
| `ci-docs` | `docs-check`. |
| `ci-notebooks` | `notebook-check`: structure, order and dependencies. Notebooks are validated, not executed, because CI has no ICPSR data. |
| `ci` | `install-dev`, then all of the above in order. |

### Git workflow

| Target | What it does |
| --- | --- |
| `git-status` | `git status --short --branch`. |
| `git-branch` | Branch (or detached HEAD), detected default branch, remote, upstream, ahead/behind. |
| `git-diff` | Unstaged, staged and untracked summary. |
| `git-check` | Repository, detached HEAD, merge/rebase in progress, conflicts, secrets in staged content. |
| `git-stage` | Stages tracked changes, or exactly `FILES="a b"`. |
| `git-commit` | Commits what is staged with `MSG="..."`. |
| `git-pull` | Fetch and fast-forward only. |
| `git-sync` | `git-pull`, then reports unpushed commits. |
| `git-push` | Plain push of the current branch. |
| `git-ready` | `git-check` and `check`: commit readiness. |

### Versions and releases

| Target | What it does |
| --- | --- |
| `version` | Current version, tag name, latest tag, next patch/minor/major. |
| `version-check` | SemVer valid; `pyproject.toml`, `CITATION.cff` and `uv.lock` agree. |
| `version-patch`, `version-minor`, `version-major` | Edit the version files and refresh `uv.lock`; no commit. |
| `release-notes` | Release notes from the commits since the last tag, written to `build/release/`. |
| `release-state` | Branch, clean tree, upstream, tag availability, monotonic version. |
| `release-check` | `release-state`, `version-check`, `lock-check`, `quality`, `test-ci`, `docs-check`, `notebook-check`, `health`. |
| `release-tag` | `release-check`, then an annotated tag for the current version. Local only. |
| `release` | `BUMP=patch|minor|major`: `release-check`, bump, commit, tag. Local only. |
| `release-push` | Pushes the release branch and its tag. |
| `release-verify` | Tag exists, is annotated, matches the version and HEAD, and is or is not yet on the remote. |

### Serving

| Target | What it does |
| --- | --- |
| `serve` | `sgc serve` (the FastAPI scoring service). Needs `make install-dev EXTRAS=api`. |

## 5. Common workflows

```bash
# a fresh clone
make setup
make doctor
make check
make test

# before opening a pull request
make validate

# the CI job, locally
make ci

# the analysis, with your own ICPSR download
make data-pipeline RAW_TSV=/path/to/36151-0002-Data.tsv
make train
make notebooks-run
make promote-results DRY_RUN=1      # what would change in results/
```

Parallel runs are safe for the independent read-only checks: `make -j4 quality` runs `lint`, `format-check`, `typecheck` and `import-check` concurrently (the environment check runs once). `JOBS=auto` (or `JOBS=4`) sets `-j` for you and also caps the native thread pools (`OMP_NUM_THREADS` and friends) of the tools it launches. Targets that share outputs or must run in order (`ci`, `release`, `data-pipeline`, `ml`, `reports`, `report-all`, ...) are declared `.NOTPARALLEL`. Notebook, training, figure and report runs also take a lock file in `build/.locks/`, so two `make` invocations never write the same outputs at once; the second waits.

## 6. Notebook execution and the sandbox

`notebook-run` and `notebooks-run` never modify a notebook. They execute the original with `jupyter nbconvert --execute --output-dir`, so the executed copy lands in `build/notebooks/executed/` and the source file is untouched. The package honours `SGC_RESULTS_DIR`, which the Makefile points at `build/notebooks/results/`, so the notebooks' `results/*.json`, CSV and PNG outputs also go there, not over the committed `results/`. The first run seeds that sandbox from the committed `results/` so a partial run (say notebook 07 alone) can read what 03 and 04 produced.

One output is shared with the rest of the project: notebook 07 writes `dataset/oof.parquet` (derived and git-ignored), which notebook 09 reads. The plan says so, and a previous copy is backed up to `build/backups/` first.

A failing notebook stops the run (`KEEP_GOING=1` continues), is named in the error, and its log is kept in `build/notebooks/executed/<name>.log`. `build/notebooks/status.json` and `make notebook-status` hold the per-notebook result. `NB_TIMEOUT` is the per-cell timeout in seconds (default 6000, as in `sgc reproduce`). No target clears notebook outputs.

## 7. Cleanup safety

Cleanup is an allowlist, not a pattern. Each group is a fixed list of paths, and every path is validated before it is touched:

- it must resolve strictly inside the repository (never the root, the home directory or a filesystem root);
- it must not be, or contain, a protected path (`src`, `tests`, `notebooks`, `results`, `project`, `scripts`, `dataset` itself, `.git`, `pyproject.toml`, ...);
- it must not contain `build/backups` or `artifacts/previous`;
- a symlink is removed as a link and never followed.

A misconfigured setting cannot aim a deletion at source: `NB_RUN_DIR=.`, `NB_RUN_DIR=src` or `FIGURES_DIR=results` are refused with `refusing to remove ...`, and the tests assert it. The plan (every path, file count, size) is printed before anything is removed; `DRY_RUN=1` stops there. Groups marked `[confirm]` then need `CONFIRM=1` or an interactive `yes`, and **never run when `CI` is set**, even with `CONFIRM=1`. Cleaning is idempotent: nothing left to remove is a success.

## 8. Git automation

Git targets run only when you name them. They check the repository and the remote first and stop with an actionable message otherwise.

- The **default branch is detected**, in this order: the remote's `HEAD` ref, the remote itself (`git ls-remote --symref`), `init.defaultBranch`, then a local `main` or `master`. It is never assumed.
- A **detached HEAD** blocks `git-stage`, `git-commit`, `git-pull`, `git-push` and the release targets. A merge, rebase or cherry-pick in progress and unresolved conflicts fail `git-check`.
- A **missing upstream** is reported; the first `git-push` of a branch pushes it and sets the upstream, refusing if a remote branch of that name already exists.
- `git-stage` never uses `add -A`. With no `FILES` it stages modifications to tracked files; new files must be named. It refuses ignored files, credentials (`.env`, `*.pem`, `*.key`, ...), generated or restricted data (`dataset/*.parquet`, `artifacts/`, `build/`, ICPSR files, `*.joblib`), and content that looks like an AWS key, a private key block, a GitHub or Slack token or a hard-coded credential. Matches are reported by name, never echoed.
- `git-commit` needs `MSG` and staged changes, re-scans the staged diff, and runs your hooks (it never passes `--no-verify` and cannot amend).
- `git-pull` and `git-sync` fetch and fast-forward only. They refuse a tree with uncommitted changes to tracked files and a diverged branch; they never stash, reset or discard anything.
- `git-push` is a plain `git push`. It refuses when the branch is behind its upstream, does not retry on rejection, and there is no force option (`FORCE=1` is an error). If the branch is protected, push a feature branch and open a pull request.

## 9. Release management

The version has one source: `[project].version` in `pyproject.toml`. `CITATION.cff` and the `savings-goal` entry in `uv.lock` mirror it, and `version-check` fails if any of the three disagrees or the version is not valid [Semantic Versioning](https://semver.org). A bump edits `pyproject.toml` and `CITATION.cff` (including `date-released`), runs `uv lock`, and restores all three files if any step fails. The repository keeps no `CHANGELOG.md`, so none is created; `release-notes` generates notes from the Git history instead, grouping conventional-commit prefixes (`feat`, `fix`, ...) and listing the rest under "Other".

```bash
make version                  # where we are, what the next versions would be
make release-check            # the strict gate (read-only)
make release BUMP=minor       # gate, bump, commit "chore(release): vX.Y.Z", annotated tag. Local only.
make release-verify           # the tag, its content and whether it is on the remote
make release-push DRY_RUN=1   # shows the two pushes
make release-push CONFIRM=1   # publishes the branch and the tag
```

To tag the version already in `pyproject.toml` without bumping (the first release, for example), use `make release-tag`.

`release-check` requires, in order: a valid SemVer; a clean working tree; a branch that is the detected default branch (or `release/*`) and is not behind its upstream (fetched first; `OFFLINE=1` skips the network checks and downgrades them to warnings); a tag that exists neither locally nor on the remote and a version newer than every existing `vX.Y.Z` tag; agreeing version files; `lock-check`; `quality`; `test-ci` (tests with the coverage threshold); `docs-check`; `notebook-check`; and `health`. It needs uv.

`release-push` is the only publishing step: two plain pushes, `HEAD:refs/heads/<branch>` and `refs/tags/<tag>`. It never uses `--tags`, never forces, never moves or overwrites a tag, and refuses when the tag is missing locally, does not point at `HEAD`, or already exists on the remote. Creating a GitHub release from the tag is manual: this repository has no release workflow or release credential, and the Makefile does not invent one.

## 10. Configuration variables

Set on the command line (`make test COV_MIN=90`) or in the environment (`export COV_MIN=90`). The configuration section of the `Makefile` is the single place they are defined; `make help` lists them with defaults.

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODE` | `auto` | `auto` (uv if installed, else pip), `uv`, `pip`. |
| `PYTHON` | `python3` (`python` on Windows) | Interpreter for the helper and the pip fallback. |
| `UV` | `uv` | uv executable. |
| `PYTHON_VERSION` | unset | Pin the project interpreter, e.g. `3.13`. Must satisfy `requires-python`. |
| `VENV` | `.venv` | Project virtual environment. |
| `EXTRAS` | unset | Optional dependency extras, e.g. `api`. |
| `V` | `1` | `0` quiet, `1` normal, `2` echo every command. |
| `DRY_RUN` | `0` | `1` previews: tools are not run, nothing is changed. |
| `CONFIRM` | `0` | `1` pre-approves `[confirm]` and `[publish]` targets (ignored in CI). |
| `JOBS` | unset | `N` or `auto`: `make -j` and native thread caps. |
| `SEED` | `42` | Random seed (`SGC_SEED`). |
| `COV_MIN` | `95` | Minimum coverage percent for `coverage` and `test-ci`. |
| `TEST_PATHS`, `UNIT_PATHS`, `INTEGRATION_PATHS` | `tests`, `tests/unit`, `tests/integration` | Test locations. |
| `PYTEST_ARGS` | unset | Extra pytest arguments. |
| `NB_TIMEOUT` | `6000` | Seconds per notebook cell. |
| `BUILD_DIR` | `build` | Scratch and output root. |
| `DATA_DIR` | `dataset` | Data tables (`SGC_DATA_DIR`). |
| `RAW_TSV` | `dataset/ICPSR_36151/DS0002/36151-0002-Data.tsv` | The raw ICPSR file. |
| `RESULTS_DIR` | `results` | Results that figures read (`SGC_RESULTS_DIR`). |
| `MODEL_DIR` | `artifacts` | Fitted models. |
| `FIGURES_DIR`, `REPORT_DIR` | `build/figures`, `build/report` | Built figures and report. |
| `NB_RUN_DIR`, `NB_EXPORT_DIR` | `build/notebooks`, `build/notebooks/export` | Notebook sandbox and exports. |
| `DEMO_DIR` | `build/demo` | Synthetic demo output. |
| `REMOTE` | `origin` | Git remote. |

Per-target arguments: `NB`, `WITH_DEPS`, `FAST`, `KEEP_GOING`, `NB_FORMAT`, `MSG`, `FILES`, `BUMP`, `OFFLINE`, `STRICT`, `N`, `HOST`, `PORT`. No secret belongs in the Makefile or a `.env` file; the Makefile reads none.

## 11. Dry-run and confirmation

- `DRY_RUN=1` is a semantic preview: it prints what would be removed, copied, staged, committed, tagged, pushed or run, and does none of it. For tool targets it prints the command. Combine it with any `[writes]`, `[confirm]`, `[git]` or `[publish]` target.
- GNU Make's own `-n` also works (`make -n clean-all CONFIRM=1` prints the recipe and deletes nothing).
- `[confirm]` and `[publish]` targets proceed only with `CONFIRM=1`, or after you type `yes` at a terminal prompt. Without a terminal and without `CONFIRM=1` they stop with the exact command to run. With `CI` set they refuse outright.

```bash
make clean-all DRY_RUN=1      # list exactly what would go
make clean-all CONFIRM=1      # then do it
```

## 12. Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `Python 3.11+ is required to run this Makefile` | No suitable `python3` / `python` on `PATH`. Install uv or Python, or `make PYTHON=/path/to/python3.12 ...`. |
| `the project environment is not ready` | No `.venv`, or `pyproject.toml` / `uv.lock` changed since the last install. Run `make install-dev`. |
| `no interpreter satisfies requires-python` | The installed Pythons are outside the supported range. `uv python install 3.13`, or set `PYTHON`. |
| `MODE=uv but uv was not found` | Install uv, or use `MODE=pip` (no lockfile guarantees). |
| `make: *** No rule to make target` for a data file | The raw ICPSR file is absent: `make data-check`, then see `dataset/README.md`. |
| `confirmation required` | The target is destructive. Preview with `DRY_RUN=1`, then add `CONFIRM=1`. |
| `refusing to ... CI is set` | Destructive and publishing targets never run in CI. Run them locally. |
| `HEAD is detached` | `git switch <branch>`. Write targets need a branch. |
| `another run of ... is still in progress` | A lock in `build/.locks/` is held (or left by a crash). Wait, or delete the lock file. |
| `uv.lock` reported out of date | `uv lock`, review, commit. `make lock-check` confirms. |
| Windows: `make` not found | Install GNU Make (see section 1) and reopen the terminal. |
| macOS: odd syntax errors | You are using the system Make 3.81. Use Homebrew's `gmake`. |
| Notebook `PosixPath('/home/...')` in `notebook-check` | A cell ended on a bare `write_result(...)`, so Jupyter echoed the returned path into the saved output. End the cell with a `print` instead. |
