"""Fixtures for the Makefile helper tests (``scripts/mk*.py``).

Every destructive or git-touching test runs against a throwaway project under ``tmp_path``
(and a local bare repository standing in for the remote), never against the real checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from mk_support import git  # noqa: E402

PYPROJECT = '[project]\nname = "sandbox"\nversion = "1.2.3"\nrequires-python = ">=3.11"\n'
CITATION = 'cff-version: 1.2.0\ntitle: "t"\nversion: "1.2.3"\ndate-released: "2026-01-01"\n'


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A miniature repository layout: sources, results, a raw data file and generated junk."""
    root = (tmp_path / "proj").resolve()
    for d in ("src/pkg", "tests", "notebooks", "results", "dataset/ICPSR_36151/DS0002", "build"):
        (root / d).mkdir(parents=True)
    (root / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    (root / "CITATION.cff").write_text(CITATION, encoding="utf-8")
    (root / "README.md").write_text("# sandbox\n", encoding="utf-8")
    (root / "src/pkg/mod.py").write_text("X = 1\n", encoding="utf-8")
    (root / "results/baseline.json").write_text('{"a": 1}\n', encoding="utf-8")
    (root / "dataset/README.md").write_text("data\n", encoding="utf-8")
    (root / "dataset/ICPSR_36151/DS0002/36151-0002-Data.tsv").write_text("raw\n", encoding="utf-8")
    return root


@pytest.fixture
def git_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    empty = tmp_path / "empty.gitconfig"
    empty.write_text("", encoding="utf-8")
    for key, value in {
        "GIT_CONFIG_GLOBAL": str(empty),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }.items():
        monkeypatch.setenv(key, value)


@pytest.fixture
def repo(project: Path, tmp_path: Path, git_env: None) -> tuple[Path, Path]:
    """(working tree, bare remote): ``project`` committed on ``main``, pushed, origin/HEAD set."""
    remote = tmp_path / "remote.git"
    git(tmp_path, "init", "--bare", "-b", "main", str(remote))
    git(project, "init", "-b", "main")
    (project / ".gitignore").write_text("build/\n.env\n", encoding="utf-8")
    git(project, "add", "-A")
    git(project, "commit", "-m", "initial")
    git(project, "remote", "add", "origin", str(remote))
    git(project, "push", "-u", "origin", "main")
    git(project, "remote", "set-head", "origin", "main")
    return project, remote
