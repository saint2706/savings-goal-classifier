"""Configuration, Python discovery, diagnostics, locks and confirmation in ``scripts/mk_core.py``."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import mk
import mk_core
import mk_env
import pytest
from mk_core import Config, MkError, Report


def test_config_defaults_and_overrides(tmp_path: Path) -> None:
    cfg = Config.from_env({"MK_ROOT": str(tmp_path)})
    assert (cfg.mode, cfg.seed, cfg.cov_min, cfg.verbosity, cfg.dry_run) == (
        "auto",
        42,
        95,
        1,
        False,
    )
    cfg = Config.from_env(
        {
            "MK_ROOT": str(tmp_path),
            "MK_SEED": "7",
            "MK_DRY_RUN": "1",
            "MK_CONFIRM": "yes",
            "MK_MODE": "PIP",
            "MK_BUILD_DIR": "out",
            "MK_ARG_MSG": "feat: x",
            "CI": "true",
        }
    )
    assert cfg.seed == 7 and cfg.dry_run and cfg.confirm and cfg.ci
    assert cfg.mode == "pip" and cfg.arg("msg") == "feat: x"
    assert cfg.dir("build") == (tmp_path / "out").resolve()


@pytest.mark.parametrize(
    "env, message",
    [({"MK_MODE": "conda"}, "MODE must be"), ({"MK_SEED": "forty-two"}, "not a number")],
)
def test_config_rejects_bad_values(tmp_path: Path, env: dict[str, str], message: str) -> None:
    with pytest.raises(MkError, match=message):
        Config.from_env({"MK_ROOT": str(tmp_path), **env})


@pytest.mark.parametrize(
    "version, spec, expected",
    [
        ((3, 11, 0), ">=3.11,<3.15", True),
        ((3, 14, 7), ">=3.11,<3.15", True),
        ((3, 15, 0), ">=3.11,<3.15", False),
        ((3, 10, 9), ">=3.11,<3.15", False),
        ((3, 12, 1), "==3.12.*", True),
        ((3, 13, 0), "==3.12.*", False),
        ((3, 12, 4), "~=3.12.2", True),
        ((3, 13, 0), "~=3.12.2", False),
        ((3, 12, 0), "!=3.12.*,>=3.9", False),
    ],
)
def test_specifier_matching(version: tuple[int, ...], spec: str, expected: bool) -> None:
    assert mk_core.satisfies(version, spec) is expected


def test_supported_minors_come_from_the_specifier() -> None:
    assert mk_core.supported_minors(">=3.11,<3.15") == ["3.11", "3.12", "3.13", "3.14"]
    assert mk_core.supported_minors(">=3.12")[0] == "3.12"
    with pytest.raises(MkError, match="unsupported version specifier"):
        mk_core.parse_specifier("^3.11")


def test_requires_python_is_read_from_pyproject(project: Path) -> None:
    assert mk_core.requires_python(project) == ">=3.11"
    (project / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "1"\n', encoding="utf-8"
    )
    with pytest.raises(MkError, match=r"no \[project\]\.requires-python"):
        mk_core.requires_python(project)


def test_dependency_group_expands_includes(project: Path) -> None:
    (project / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "1"\nrequires-python = ">=3.11"\n'
        '[dependency-groups]\ntest = ["pytest>=8"]\ndev = ["ruff", {include-group = "test"}]\n',
        encoding="utf-8",
    )
    assert mk_core.dependency_group(project, "dev") == ["ruff", "pytest>=8"]
    with pytest.raises(MkError, match=r"no \[dependency-groups\]\.nope"):
        mk_core.dependency_group(project, "nope")


def test_select_python_never_falls_back_to_an_unsupported_interpreter(project: Path) -> None:
    (project / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "1"\nrequires-python = ">=3.99"\n', encoding="utf-8"
    )
    cfg = Config(root=project, python=sys.executable)
    with pytest.raises(MkError, match=r"no interpreter satisfies requires-python >=3\.99") as exc:
        mk_core.select_python(cfg)
    assert "uv python install" in exc.value.hint


def test_select_python_honours_python_version_request(project: Path) -> None:
    cfg = Config(root=project, python=sys.executable)
    _, version = mk_core.select_python(cfg)
    assert version[:2] == sys.version_info[:2]
    wrong = f"3.{sys.version_info.minor - 1}"
    with pytest.raises(MkError, match=f"PYTHON_VERSION={re.escape(wrong)}"):
        mk_core.select_python(Config(root=project, python=sys.executable, python_version=wrong))


def test_env_ready_and_run_tool_diagnose_a_missing_environment(project: Path) -> None:
    cfg = Config(root=project, venv=".venv")
    ready, reason = mk_core.env_ready(cfg)
    assert not ready and "does not exist" in reason
    with pytest.raises(MkError, match="environment is not ready") as exc:
        mk_core.run_tool(cfg, ["ruff", "check"])
    assert exc.value.code == 2 and "make install-dev" in exc.value.hint


def test_run_tool_reports_a_tool_missing_from_the_environment(project: Path) -> None:
    cfg = Config(root=project, venv=sys.prefix)
    with pytest.raises(MkError, match="`definitely-not-a-tool` is not installed") as exc:
        mk_core.run_tool(cfg, ["definitely-not-a-tool"])
    assert exc.value.code == 2


def test_run_tool_dry_run_executes_nothing(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = Config(root=project, venv=".venv", dry_run=True)
    assert mk_core.run_tool(cfg, ["ruff", "check", "."]).returncode == 0
    assert "dry-run: would run: ruff check ." in capsys.readouterr().out


def test_run_tool_exports_the_seed_and_directories(project: Path) -> None:
    cfg = Config(root=project, venv=sys.prefix, seed=7, jobs="3")
    proc = mk_core.run_tool(
        cfg,
        ["python", "-c", "import os; print(os.environ['SGC_SEED'], os.environ['OMP_NUM_THREADS'])"],
        capture=True,
    )
    assert proc.stdout.split() == ["7", "3"]


def test_run_host_missing_tool_gives_an_install_hint(project: Path) -> None:
    with pytest.raises(MkError, match="was not found on PATH") as exc:
        mk_core.run_host(Config(root=project), ["definitely-not-a-tool"], tool="uv")
    assert "uv" in exc.value.hint and exc.value.code == 2


def test_report_exit_codes_and_strictness(capsys: pytest.CaptureFixture[str]) -> None:
    r = Report("t")
    r.passed("a")
    r.warned("b", "careful", "do x")
    assert r.finish() == 0
    assert r.finish(strict=True) == 1
    r.failed("c", "broken")
    assert r.finish() == 1
    out = capsys.readouterr().out
    assert "WARN  b: careful" in out and "do x" in out and "1 failure(s)" in out


def test_confirmation_policy(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdin", None)
    with pytest.raises(MkError, match="confirmation required") as exc:
        mk_core.require_confirmation(Config(root=project), "delete things")
    assert "DRY_RUN=1" in exc.value.hint and "CONFIRM=1" in exc.value.hint
    mk_core.require_confirmation(Config(root=project, confirm=True), "delete things")
    with pytest.raises(MkError, match="CI is set"):
        mk_core.require_confirmation(Config(root=project, confirm=True, ci=True), "delete things")


def test_interactive_confirmation_needs_a_literal_yes(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Tty:
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(sys, "stdin", Tty())
    cfg = Config(root=project)
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")
    with pytest.raises(MkError, match="aborted"):
        mk_core.require_confirmation(cfg, "delete things")

    def closed(_prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", closed)  # a prompt nobody can answer is not consent
    with pytest.raises(MkError, match="confirmation required"):
        mk_core.require_confirmation(cfg, "delete things")
    monkeypatch.setattr("builtins.input", lambda _prompt: "yes")
    mk_core.require_confirmation(cfg, "delete things")


def test_file_lock_allows_one_writer(project: Path) -> None:
    cfg = Config(root=project)
    lock = project / "build/.locks/data.lock"
    with mk_core.file_lock(cfg, "data"):
        assert lock.exists()
        with (
            pytest.raises(MkError, match="still in progress"),
            mk_core.file_lock(cfg, "data", wait=0),
        ):
            pass
    assert not lock.exists()
    with mk_core.file_lock(cfg, "data", wait=0):
        pass


def test_file_lock_clears_a_stale_lock(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = Config(root=project)
    lock = project / "build/.locks/old.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("123", encoding="utf-8")
    with mk_core.file_lock(cfg, "old", stale_after=-1, wait=0):
        pass
    assert "stale lock" in capsys.readouterr().err


def test_install_hints_cover_every_platform() -> None:
    for tool, by_platform in mk_core.INSTALL_HINTS.items():
        assert set(by_platform) == {"windows", "macos", "linux"}, tool
        assert mk_core.install_hint(tool)


def test_pip_fallback_dry_run_names_the_lock_gap(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (project / "pyproject.toml").write_text(PYPROJECT_WITH_GROUP, encoding="utf-8")
    cfg = Config(root=project, mode="pip", python=sys.executable, dry_run=True, venv=".venv")
    assert mk_env.cmd_install_dev(cfg) == 0
    out = capsys.readouterr().out
    assert "would create" in out and "pip install -e . pytest>=8" in out


def test_pip_mode_lock_check_is_an_honest_skip(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = Config(root=project, mode="pip")
    assert mk_env.cmd_lock_check(cfg) == 0
    assert "NOT verified" in capsys.readouterr().err


def test_mode_uv_without_uv_explains_how_to_install(project: Path) -> None:
    with pytest.raises(MkError, match="MODE=uv but") as exc:
        mk_env.resolve_mode(Config(root=project, mode="uv", uv="no-such-uv-binary"))
    assert "MODE=pip" in exc.value.hint and exc.value.code == 2


def test_auto_mode_falls_back_to_pip_and_warns(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert mk_env.resolve_mode(Config(root=project, uv="no-such-uv-binary")) == "pip"
    assert "does NOT honour uv.lock" in capsys.readouterr().err


def test_help_is_generated_from_the_makefile(tmp_path: Path) -> None:
    makefile = tmp_path / "Makefile"
    makefile.write_text(
        "##@ Group\n"
        "build: dep ## [safe] Build it\n"
        "\t@echo\n"
        "## Where to build\n"
        "OUT ?= out\n"
        "x: export V := 1\n"
        "hidden:\n",
        encoding="utf-8",
    )
    sections, variables = mk.parse_makefile(makefile)
    assert sections == [("Group", [("build", "[safe] Build it")])]
    assert variables == [("OUT", "out", "Where to build")]


PYPROJECT_WITH_GROUP = (
    '[project]\nname = "x"\nversion = "1"\nrequires-python = ">=3.11"\n'
    '[dependency-groups]\ndev = ["pytest>=8"]\n'
)


def test_ensure_env_passes_uv_s_own_explanation_through(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = Config(root=project, venv=sys.prefix, mode="uv", uv=sys.executable)
    outcome = {"code": 0, "err": ""}

    def fake(_cfg: Config, argv: list[str], **_kw: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, outcome["code"], "", outcome["err"])

    monkeypatch.setattr(mk_env, "run_host", fake)
    assert mk_env.cmd_ensure_env(cfg) == 0  # in sync

    outcome.update(
        code=2,
        err="Resolved 1 package\nerror: No interpreter found for Python 3.9\nhint: install one\n",
    )
    with pytest.raises(MkError, match="not in sync") as exc:
        mk_env.cmd_ensure_env(cfg)
    assert exc.value.code == 2 and "make install-dev" in exc.value.hint
    assert "uv: error: No interpreter found for Python 3.9" in exc.value.hint
    assert "uv: hint: install one" in exc.value.hint and "Resolved" not in exc.value.hint
