"""The root Makefile, exercised through ``make`` itself.

Tests that delete, tag or commit run against a throwaway copy of the Makefile and helper
scripts in ``tmp_path``; the real checkout is only ever read (help, ``-n`` previews, the
read-only notebook and docs checks).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(shutil.which("make") is None, reason="GNU make is not installed")

ROOT = Path(__file__).resolve().parents[2]
# Every variable the Makefile reads, plus the ones make passes down: a developer's shell or CI
# must not change what these tests see.
CLEAN = [
    "MODE",
    "PYTHON",
    "UV",
    "PYTHON_VERSION",
    "VENV",
    "EXTRAS",
    "V",
    "DRY_RUN",
    "CONFIRM",
    "JOBS",
    "SEED",
    "COV_MIN",
    "TEST_PATHS",
    "UNIT_PATHS",
    "INTEGRATION_PATHS",
    "PYTEST_ARGS",
    "NB_TIMEOUT",
    "BUILD_DIR",
    "DATA_DIR",
    "RAW_TSV",
    "RESULTS_DIR",
    "MODEL_DIR",
    "FIGURES_DIR",
    "REPORT_DIR",
    "NB_RUN_DIR",
    "NB_EXPORT_DIR",
    "DEMO_DIR",
    "REMOTE",
    "NB",
    "WITH_DEPS",
    "FAST",
    "KEEP_GOING",
    "NB_FORMAT",
    "MSG",
    "FILES",
    "BUMP",
    "OFFLINE",
    "STRICT",
    "FORCE",
    "N",
    "HOST",
    "PORT",
    "CI",
    "MAKEFLAGS",
    "MAKELEVEL",
    "MFLAGS",
    "GNUMAKEFLAGS",
    "MK_REROOTED",
]
PYTHON = Path(sys.executable).as_posix()
VENV = Path(sys.prefix).as_posix()


def env(**extra: str) -> dict[str, str]:
    e = {
        k: v for k, v in os.environ.items() if k not in CLEAN and not k.startswith(("MK_", "SGC_"))
    }
    e["PYTHON"] = PYTHON
    e.update(
        GIT_AUTHOR_NAME="Test",
        GIT_AUTHOR_EMAIL="t@example.com",
        GIT_COMMITTER_NAME="Test",
        GIT_COMMITTER_EMAIL="t@example.com",
    )
    e.update(extra)
    return e


def make(*args: str, cwd: Path = ROOT, **extra_env: str) -> subprocess.CompletedProcess[str]:
    # stdin is detached so a developer's terminal can never turn into an interactive "yes".
    return subprocess.run(
        ["make", *args],
        cwd=cwd,
        env=env(**extra_env),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )


def database() -> str:
    return subprocess.run(
        ["make", "-pn", "help"], cwd=ROOT, env=env(), capture_output=True, text=True, check=False
    ).stdout


def documented_targets() -> list[str]:
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts/mk.py"), "targets"],
        env=env(MK_ROOT=str(ROOT)),
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.split()


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    """A disposable project with the real Makefile and helper scripts, never the real checkout."""
    root = tmp_path / "sandbox"
    (root / "scripts").mkdir(parents=True)
    shutil.copy2(ROOT / "Makefile", root / "Makefile")
    for script in (ROOT / "scripts").glob("mk*.py"):
        shutil.copy2(script, root / "scripts" / script.name)
    for d in ("src/pkg", "tests", "notebooks", "results", "dataset", "build/test", "build/demo"):
        (root / d).mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "sandbox"\nversion = "1.2.3"\nrequires-python = ">=3.11"\n'
        '[dependency-groups]\ndev = ["pytest>=8"]\n',
        encoding="utf-8",
    )
    (root / "CITATION.cff").write_text(
        'version: "1.2.3"\ndate-released: "2026-01-01"\n', encoding="utf-8"
    )
    (root / "src/pkg/mod.py").write_text("X = 1\n", encoding="utf-8")
    (root / "results/a.json").write_text("{}\n", encoding="utf-8")
    (root / "dataset/README.md").write_text("data\n", encoding="utf-8")
    (root / "build/test/junit.xml").write_text("<x/>", encoding="utf-8")
    (root / "build/demo/households.parquet").write_text("x", encoding="utf-8")
    (root / ".pytest_cache").mkdir()
    (root / ".pytest_cache/v").write_text("x", encoding="utf-8")
    return root


def listing(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))


# --- the interface ---------------------------------------------------------------------


def test_default_goal_is_a_useful_help_screen() -> None:
    plain, explicit = make(), make("help")
    assert plain.returncode == 0 and plain.stdout == explicit.stdout
    for needle in (
        "Usage: make <target>",
        "Configuration",
        "[confirm]",
        "Windows:",
        "release-push",
    ):
        assert needle in plain.stdout


def test_every_annotated_target_is_a_real_phony_rule() -> None:
    db = database()
    rules = set(re.findall(r"(?m)^([A-Za-z0-9_.-]+):", db))
    phony = set(re.search(r"(?m)^\.PHONY: (.*)$", db).group(1).split())  # type: ignore[union-attr]
    targets = documented_targets()
    assert len(targets) > 80 and len(set(targets)) == len(targets), "duplicate documented targets"
    assert not [t for t in targets if t not in rules]
    assert not [t for t in targets if t not in phony], "documented target missing from .PHONY"


def test_documentation_mentions_only_real_targets() -> None:
    targets = set(documented_targets())
    problems = []
    for doc in (ROOT / "docs/makefile.md", ROOT / "README.md"):
        text = doc.read_text(encoding="utf-8")
        found = set(re.findall(r"`make ([a-z][a-z0-9-]*)", text))
        found |= set(re.findall(r"(?m)^\s*(?:\$ )?make ([a-z][a-z0-9-]*)", text))
        problems += [f"{doc.name}: make {t}" for t in sorted(found - targets)]
    assert not problems


def test_every_documented_target_is_in_the_guide() -> None:
    guide = (ROOT / "docs/makefile.md").read_text(encoding="utf-8")
    missing = [t for t in documented_targets() if not re.search(rf"`{re.escape(t)}`", guide)]
    assert not missing, f"targets absent from docs/makefile.md: {missing}"


def test_order_sensitive_targets_are_not_parallelised() -> None:
    notparallel = re.search(r"(?m)^\.NOTPARALLEL: (.*)$", database())
    assert notparallel
    assert {"ci", "release-check", "release", "data-pipeline", "ml"} <= set(
        notparallel.group(1).split()
    )


def test_shared_prerequisites_run_once_under_parallel_make() -> None:
    plan = make("-n", "-j4", "quality")
    assert plan.returncode == 0
    assert plan.stdout.count("mk.py ensure-env") == 1
    assert {"ruff check", "ruff format --check", "mypy"} <= {
        m for m in ("ruff check", "ruff format --check", "mypy") if m in plan.stdout
    }


def test_jobs_variable_sets_make_parallelism() -> None:
    flags = subprocess.run(
        ["make", "-pn", "JOBS=3", "help"], cwd=ROOT, env=env(), capture_output=True, text=True
    ).stdout
    assert re.search(r"(?m)^MAKEFLAGS = .*-j3", flags) and "MK_JOBS := 3" in flags


def test_configuration_overrides_reach_the_recipes() -> None:
    plan = make(
        "-n", "coverage", "COV_MIN=81", "PYTEST_ARGS=-x -k fast", "TEST_PATHS=tests/unit"
    ).stdout
    assert "--cov-fail-under=81" in plan and "-x -k fast" in plan and "tests/unit" in plan
    from_env = make("-n", "test-ci", COV_MIN="77").stdout
    assert "--cov-fail-under=77" in from_env and "junit.xml" in from_env
    assert "--cov-fail-under=" in make("-n", "ci-test").stdout


def test_verbosity_levels() -> None:
    assert "scripts/mk.py version" not in make("version").stdout
    assert "scripts/mk.py version" in make("V=2", "version").stdout


# --- environment detection and diagnostics ---------------------------------------------


def test_missing_python_is_diagnosed_with_install_instructions() -> None:
    proc = make("help", "PYTHON=definitely-not-a-python")
    assert proc.returncode != 0
    assert "Python 3.11+ is required" in proc.stderr and "docs.astral.sh/uv" in proc.stderr


def test_missing_uv_is_diagnosed_with_install_instructions(sandbox: Path) -> None:
    proc = make("install-dev", "MODE=uv", "UV=no-such-uv", cwd=sandbox)
    assert proc.returncode != 0
    assert "`no-such-uv` was not found" in proc.stderr and "Install uv" in proc.stderr


def test_python_version_outside_requires_python_is_rejected(sandbox: Path) -> None:
    proc = make("check-env", "PYTHON_VERSION=3.9", cwd=sandbox)
    assert proc.returncode != 0
    assert "3.9 is outside requires-python >=3.11" in proc.stdout


def test_python_info_reads_requires_python_dynamically(sandbox: Path) -> None:
    (sandbox / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "1.0.0"\nrequires-python = ">=3.12,<3.14"\n',
        encoding="utf-8",
    )
    out = make("python-info", cwd=sandbox).stdout
    assert "requires-python:   >=3.12,<3.14" in out and "supported minors:  3.12, 3.13" in out


def test_tool_targets_refuse_to_run_in_a_missing_environment(sandbox: Path) -> None:
    proc = make("lint", cwd=sandbox)
    assert proc.returncode == 2
    assert "environment is not ready" in proc.stderr and "make install-dev" in proc.stderr
    assert not (sandbox / ".venv").exists(), "a tool target must not create an environment"


def test_pip_fallback_preview_is_honest_about_the_lockfile(sandbox: Path) -> None:
    proc = make("install-dev", "MODE=pip", "DRY_RUN=1", cwd=sandbox)
    assert proc.returncode == 0
    assert "would create" in proc.stdout and "pip install -e ." in proc.stdout
    assert not (sandbox / ".venv").exists()


def test_optional_targets_fail_gracefully_not_with_tracebacks(sandbox: Path) -> None:
    for target, message in (
        ("precommit", "no .pre-commit-config.yaml"),
        ("figures-check", "make_figures.py not found"),
        ("report", "project/report.tex not found"),
        ("data-prepare", "raw ICPSR file was not found"),
        ("notebook-run", "NB is required"),
    ):
        proc = make(target, cwd=sandbox, MODE="pip", VENV=VENV)
        assert proc.returncode != 0, target
        assert message in proc.stdout + proc.stderr, target
        assert "Traceback" not in proc.stderr, target


def test_invocation_from_another_directory(tmp_path: Path) -> None:
    proc = make("-f", str(ROOT / "Makefile"), "version", cwd=tmp_path)
    assert proc.returncode == 0 and re.search(r"version:\s+\d+\.\d+\.\d+", proc.stdout)


# --- cleanup safety --------------------------------------------------------------------


def test_clean_is_idempotent_and_leaves_sources_alone(sandbox: Path) -> None:
    first = make("clean", cwd=sandbox)
    assert first.returncode == 0 and "removed" in first.stdout
    assert not (sandbox / ".pytest_cache").exists() and not (sandbox / "build/test").exists()
    second = make("clean", cwd=sandbox)
    assert second.returncode == 0 and "already clean" in second.stdout
    assert (sandbox / "src/pkg/mod.py").exists() and (sandbox / "results/a.json").exists()


def test_clean_quiet_mode_prints_nothing_when_there_is_nothing_to_do(sandbox: Path) -> None:
    make("clean", cwd=sandbox)
    assert make("clean", "V=0", cwd=sandbox).stdout == ""


def test_destructive_targets_preview_confirm_and_refuse_in_ci(sandbox: Path) -> None:
    generated = sandbox / "build/notebooks/results/x.json"
    generated.parent.mkdir(parents=True)
    generated.write_text("{}", encoding="utf-8")
    before = listing(sandbox)

    dry = make("clean-all", "DRY_RUN=1", cwd=sandbox)
    assert dry.returncode == 0 and "build/notebooks" in dry.stdout and "dry-run" in dry.stdout
    assert listing(sandbox) == before

    unconfirmed = make("clean-all", cwd=sandbox)
    assert unconfirmed.returncode != 0 and "confirmation required" in unconfirmed.stderr
    assert listing(sandbox) == before

    in_ci = make("clean-all", "CONFIRM=1", cwd=sandbox, CI="true")
    assert in_ci.returncode != 0 and "CI is set" in in_ci.stderr
    assert listing(sandbox) == before

    confirmed = make("clean-all", "CONFIRM=1", cwd=sandbox)
    assert confirmed.returncode == 0 and not generated.exists()
    for kept in ("src/pkg/mod.py", "results/a.json", "dataset/README.md", "pyproject.toml"):
        assert (sandbox / kept).exists(), kept


def test_make_dash_n_never_deletes(sandbox: Path) -> None:
    before = listing(sandbox)
    assert make("-n", "clean-all", "CONFIRM=1", cwd=sandbox).returncode == 0
    assert listing(sandbox) == before


def test_dangerous_directory_settings_are_refused(sandbox: Path) -> None:
    before = listing(sandbox)
    for setting in ("NB_RUN_DIR=.", "NB_RUN_DIR=src", "NB_RUN_DIR=results", "NB_RUN_DIR=.."):
        proc = make("clean-notebooks", setting, "CONFIRM=1", cwd=sandbox)
        assert proc.returncode != 0 and "refusing to remove" in proc.stderr, setting
    assert listing(sandbox) == before


def test_builds_cannot_be_pointed_at_tracked_files(sandbox: Path) -> None:
    for target, setting in (
        ("figures", "FIGURES_DIR=project/figures"),
        ("report", "REPORT_DIR=results"),
    ):
        proc = make(target, setting, cwd=sandbox, MODE="pip", VENV=VENV)
        assert proc.returncode != 0 and "tracked sources or results" in proc.stderr, target
    inside_root = make("figures", "FIGURES_DIR=.", cwd=sandbox, MODE="pip", VENV=VENV)
    assert inside_root.returncode != 0 and "the repository itself" in inside_root.stderr


def test_promote_is_dry_run_first_and_needs_confirmation(sandbox: Path) -> None:
    built = sandbox / "build/notebooks/results"
    built.mkdir(parents=True)
    (built / "a.json").write_text('{"new": true}\n', encoding="utf-8")
    assert make("promote-results", cwd=sandbox).returncode != 0
    assert (sandbox / "results/a.json").read_text(encoding="utf-8") == "{}\n"
    assert make("promote-results", "DRY_RUN=1", cwd=sandbox).returncode == 0
    assert (sandbox / "results/a.json").read_text(encoding="utf-8") == "{}\n"
    assert make("promote-results", "CONFIRM=1", cwd=sandbox).returncode == 0
    assert (sandbox / "results/a.json").read_text(encoding="utf-8") == '{"new": true}\n'
    backup = next((sandbox / "build/backups").glob("*/results/a.json"))
    assert backup.read_text(encoding="utf-8") == "{}\n"


# --- git and releases ------------------------------------------------------------------


@pytest.fixture
def git_sandbox(sandbox: Path) -> Path:
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=sandbox, env=env(), check=True, capture_output=True)

    (sandbox / ".gitignore").write_text("build/\n.pytest_cache/\n", encoding="utf-8")
    git("init", "-b", "main")
    git("add", "-A")
    git("commit", "-m", "initial")
    return sandbox


def test_git_targets_detect_the_branch_and_refuse_unsafe_input(git_sandbox: Path) -> None:
    branch = make("git-branch", cwd=git_sandbox)
    assert "default branch: main" in branch.stdout and "upstream:       (none)" in branch.stdout

    no_msg = make("git-commit", cwd=git_sandbox)
    assert no_msg.returncode != 0 and "MSG is required" in no_msg.stderr

    (git_sandbox / ".env").write_text("TOKEN=abc\n", encoding="utf-8")
    secret = make("git-stage", "FILES=.env", cwd=git_sandbox)
    assert secret.returncode != 0 and "credentials or environment file" in secret.stderr

    no_remote = make("git-push", cwd=git_sandbox)
    assert no_remote.returncode != 0 and "`origin` is not configured" in no_remote.stderr

    forced = make("git-push", "FORCE=1", cwd=git_sandbox)
    assert forced.returncode != 0 and "force-pushing is not supported" in forced.stderr


def test_git_stage_and_commit_workflow(git_sandbox: Path) -> None:
    (git_sandbox / "src/pkg/mod.py").write_text("X = 2\n", encoding="utf-8")
    assert make("git-stage", "DRY_RUN=1", cwd=git_sandbox).returncode == 0
    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only"], cwd=git_sandbox, capture_output=True, text=True
    )
    assert staged.stdout == ""
    assert make("git-stage", cwd=git_sandbox).returncode == 0
    assert make("git-commit", "MSG=feat(pkg): change X", cwd=git_sandbox).returncode == 0
    log = subprocess.run(
        ["git", "log", "-1", "--format=%s"], cwd=git_sandbox, capture_output=True, text=True
    )
    assert log.stdout.strip() == "feat(pkg): change X"


def test_release_check_blocks_a_dirty_or_unsynced_tree_before_running_any_gate(
    git_sandbox: Path,
) -> None:
    (git_sandbox / "src/pkg/mod.py").write_text("X = 9\n", encoding="utf-8")
    proc = make("release-check", cwd=git_sandbox)
    assert proc.returncode != 0
    assert "working tree: not clean" in proc.stdout and "no upstream configured" in proc.stdout
    assert "PASS  tests" not in proc.stdout and "pytest" not in proc.stdout  # failed fast


def test_release_requires_bump_and_is_never_part_of_all_or_ci() -> None:
    no_bump = make("release")
    assert no_bump.returncode != 0 and "BUMP=patch|minor|major is required" in no_bump.stderr
    for goal in ("all", "ci", "validate", "check", "dev", "setup"):
        plan = make("-n", goal).stdout
        for forbidden in ("release", "git-push", "git-commit", "git-stage", "clean-all", "promote"):
            assert f"mk.py {forbidden}" not in plan, f"`make {goal}` would run {forbidden}"


def test_version_targets(sandbox: Path) -> None:
    out = make("version", cwd=sandbox).stdout
    assert "1.2.3" in out and "next patch:     1.2.4" in out and "next major:     2.0.0" in out
    assert make("version-check", cwd=sandbox).returncode == 0
    (sandbox / "CITATION.cff").write_text('version: "9.9.9"\n', encoding="utf-8")
    bad = make("version-check", cwd=sandbox)
    assert bad.returncode != 0 and "9.9.9 != pyproject.toml 1.2.3" in bad.stdout


def test_version_bump_needs_a_clean_tree_and_previews(git_sandbox: Path) -> None:
    before = (git_sandbox / "pyproject.toml").read_bytes()
    preview = make("version-minor", "DRY_RUN=1", cwd=git_sandbox)
    assert preview.returncode == 0 and "1.2.3 -> 1.3.0" in preview.stdout
    assert (git_sandbox / "pyproject.toml").read_bytes() == before
    (git_sandbox / "src/pkg/mod.py").write_text("X = 3\n", encoding="utf-8")
    dirty = make("version-minor", cwd=git_sandbox)
    assert dirty.returncode != 0 and "working tree is not clean" in dirty.stderr
    assert (git_sandbox / "pyproject.toml").read_bytes() == before


# --- notebooks, docs, reproducibility on the real repository (read-only) ---------------


def test_notebook_discovery_on_the_real_repository() -> None:
    listing_out = make("notebook-list").stdout
    rows = [ln.split() for ln in listing_out.splitlines() if re.match(r"\d+\s+\d\d_", ln)]
    order = [r[1] for r in rows]
    assert len(order) == len(list((ROOT / "notebooks").glob("*.ipynb")))
    assert order.index("03_baseline") < order.index("07_business_translation")
    assert order.index("04_model_comparison") < order.index("07_business_translation")
    assert order.index("07_business_translation") < order.index("09_outreach_simulation")
    assert make("notebook-check").returncode == 0


def test_notebook_dry_run_lists_the_plan_without_executing() -> None:
    proc = make("business-analysis", "DRY_RUN=1", MODE="pip", VENV=VENV)
    out = proc.stdout + proc.stderr
    if "prerequisites are missing" in out:  # CI has no restricted data: that is the correct answer
        assert "features.parquet" in out
    else:
        assert proc.returncode == 0 and "dry-run: nothing executed" in out


def test_docs_check_passes_on_the_repository_docs() -> None:
    proc = make("docs-check")
    assert proc.returncode == 0 and "every relative link and anchor resolves" in proc.stdout


def test_seed_variable_is_exported_to_the_package() -> None:
    proc = make("seed-check", "SEED=11", MODE="pip", VENV=VENV)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "SEED=11" in proc.stdout and "SGC_SEED=11" in proc.stdout


def test_environment_report_records_seed_lock_and_versions(tmp_path: Path) -> None:
    proc = make(
        "environment-report",
        "SEED=5",
        f"BUILD_DIR={(tmp_path / 'b').as_posix()}",
        MODE="pip",
        VENV=VENV,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads((tmp_path / "b/repro/environment.json").read_text(encoding="utf-8"))
    assert data["seed"] == 5 and data["python"] == sys.version.split()[0]
    assert data["packages"]["numpy"] and data["uv_lock_sha256"] and data["install_mode"] == "pip"


# --- diagnostics on the real repository (read-only) ------------------------------------

READ_ONLY = {"MODE": "pip", "VENV": VENV}  # pip mode: no uv needed, no network, no lock checks


@pytest.mark.parametrize(
    "target, expected",
    [
        ("check-tools", "PASS  python"),
        ("python-info", "supported minors:"),
        ("check-env", "PASS  dev dependencies"),
        ("doctor", "PASS  required paths"),
        ("health", "PASS  import savings_goal"),
        ("audit", "PASS  tracked files"),
        ("verify", "PASS  pyproject.toml"),
        ("data-check", "households table"),
        ("figures-check", "PASS  leakage.json"),
        ("version-check", "PASS  CITATION.cff"),
        ("experiment-info", "recorded experiment metadata"),
        ("git-check", "PASS  repository"),
    ],
)
def test_read_only_diagnostics_run_cleanly_on_the_repository(target: str, expected: str) -> None:
    proc = make(target, **READ_ONLY)
    if target == "git-check" and "HEAD is detached" in proc.stdout:
        pytest.skip("CI checks out a detached HEAD; git-check fails there by design")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert expected in proc.stdout and "Traceback" not in proc.stderr


def test_audit_finds_no_secrets_or_restricted_data_in_the_tracked_files() -> None:
    proc = make("audit", **READ_ONLY)
    assert "FAIL" not in proc.stdout, proc.stdout
    assert "no secrets, credentials or restricted data" in proc.stdout


def test_notebook_export_writes_saved_outputs_without_executing(tmp_path: Path) -> None:
    out = tmp_path / "export"
    proc = make(
        "notebook-export",
        "NB=03",
        "NB_FORMAT=markdown",
        f"NB_EXPORT_DIR={out.as_posix()}",
        **READ_ONLY,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (out / "03_baseline.md").stat().st_size > 0


def test_data_features_train_pipeline_on_synthetic_data(tmp_path: Path) -> None:
    """build -> validate -> features -> train on the synthetic survey, in temp dirs, twice."""
    scratch = {"BUILD_DIR": (tmp_path / "b").as_posix(), **READ_ONLY}
    demo, models = tmp_path / "demo", tmp_path / "models"
    where = [f"DATA_DIR={demo.as_posix()}", f"MODEL_DIR={models.as_posix()}"]

    made = make("demo", f"DEMO_DIR={demo.as_posix()}", "N=1500", **scratch)
    assert made.returncode == 0, made.stdout + made.stderr
    assert "SYNTHETIC" in made.stdout and (demo / "households.parquet").exists()

    valid = make("data-validate", *where, **scratch)
    assert valid.returncode == 0 and "satisfy the Pandera contract" in valid.stdout

    feats = make("data-features", *where, **scratch)
    assert feats.returncode == 0 and (demo / "features.parquet").exists()
    again = make("data-features", *where, **scratch)  # a file rule: nothing to do the second time
    assert (
        again.returncode == 0 and "up to date" in again.stdout and "features:" not in again.stdout
    )

    first = make("train", *where, "SEED=42", **scratch)
    assert first.returncode == 0, first.stdout + first.stderr
    assert (models / "tier1.joblib").exists() and (models / "train_manifest.json").exists()
    manifest = json.loads((models / "train_manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 42 and manifest["input"]["file"] == "features.parquet"
    assert not (models / "previous").exists()

    second = make("train", *where, "SEED=7", **scratch)
    assert second.returncode == 0 and "archiving" in second.stdout
    (archived,) = (models / "previous").iterdir()
    assert {p.name for p in archived.iterdir()} == {
        "tier1.joblib",
        "tier2.joblib",
        "train_manifest.json",
    }
    assert json.loads((archived / "train_manifest.json").read_text(encoding="utf-8"))["seed"] == 42
    assert json.loads((models / "train_manifest.json").read_text(encoding="utf-8"))["seed"] == 7


def test_missing_raw_data_is_explained_not_downloaded(tmp_path: Path) -> None:
    empty = tmp_path / "data"
    empty.mkdir()
    proc = make(
        "data-pipeline",
        f"DATA_DIR={empty.as_posix()}",
        f"BUILD_DIR={(tmp_path / 'b').as_posix()}",
        **READ_ONLY,
    )
    assert proc.returncode != 0
    assert "raw ICPSR file was not found" in proc.stderr and "icpsr.umich.edu" in proc.stderr
    assert list(empty.iterdir()) == [], "nothing may be created or downloaded"
