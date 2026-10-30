"""Git safety, version handling and the release flow in ``scripts/mk_git.py``.

Everything runs in a temporary repository with a local bare repository as ``origin``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import mk_core
import mk_git
import pytest
from mk_core import Config, MkError
from mk_support import git


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def cfg_for(root: Path, **args: str) -> Config:
    # Only the presence of `uv` is checked here; its one real call is faked where it would run.
    return Config(root=root, args=args, uv=sys.executable)


# --- semantic versions -----------------------------------------------------------------


@pytest.mark.parametrize("good", ["0.0.1", "1.2.3", "10.20.30", "1.0.0-rc.1", "1.0.0+build.5"])
def test_valid_semver(good: str) -> None:
    assert mk_git.parse_semver(good)[:3] == tuple(
        int(x) for x in good.split("-")[0].split("+")[0].split(".")
    )


@pytest.mark.parametrize("bad", ["1.2", "01.2.3", "1.2.3.4", "v1.2.3", "1.2.x", "", "1.2.3-"])
def test_invalid_semver(bad: str) -> None:
    with pytest.raises(MkError, match="not a Semantic Version"):
        mk_git.parse_semver(bad)


def test_bump_rules() -> None:
    assert mk_git.bump("1.2.3", "patch") == "1.2.4"
    assert mk_git.bump("1.2.3", "minor") == "1.3.0"
    assert mk_git.bump("1.2.3", "major") == "2.0.0"
    with pytest.raises(MkError, match="pre-release"):
        mk_git.bump("1.2.3-rc.1", "patch")
    with pytest.raises(MkError, match="patch, minor or major"):
        mk_git.bump("1.2.3", "huge")


def test_version_edit_touches_only_the_project_table() -> None:
    text = (
        '[tool.other]\nversion = "9.9.9"\n\n[project]\nname = "x"\nversion = "1.2.3"  # keep me\n'
        'requires-python = ">=3.11"\n\n[tool.more]\nversion = "7"\n'
    )
    new = mk_git.set_pyproject_version(text, "1.3.0")
    assert new == text.replace('version = "1.2.3"', 'version = "1.3.0"')
    with pytest.raises(MkError, match="could not find"):
        mk_git.set_pyproject_version("[project]\nname = 'x'\n", "1.0.0")


def test_citation_edit_updates_version_and_date() -> None:
    text = 'title: "t"\nversion: "1.2.3"\ndate-released: "2026-01-01"\n'
    assert mk_git.set_citation(text, "1.3.0", "2026-02-02") == (
        'title: "t"\nversion: "1.3.0"\ndate-released: "2026-02-02"\n'
    )


def test_version_report_catches_drift(project: Path) -> None:
    assert mk_git.version_report(cfg_for(project)).count("FAIL") == 0
    write(project / "CITATION.cff", 'version: "1.0.0"\n')
    report = mk_git.version_report(cfg_for(project))
    assert report.count("FAIL") == 1 and "1.0.0 != pyproject.toml 1.2.3" in report.render()
    write(
        project / "pyproject.toml",
        '[project]\nname = "x"\nversion = "1.2"\nrequires-python = ">=3.11"\n',
    )
    assert "not a Semantic Version" in mk_git.version_report(cfg_for(project)).render()


def test_lock_version_is_compared(project: Path) -> None:
    write(project / "uv.lock", '[[package]]\nname = "sandbox"\nversion = "0.9.0"\n')
    assert mk_git.lock_version(project) == "0.9.0"
    assert (
        "uv.lock: 0.9.0 != pyproject.toml 1.2.3" in mk_git.version_report(cfg_for(project)).render()
    )


def test_failed_bump_restores_every_file(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write(project / "uv.lock", '[[package]]\nname = "sandbox"\nversion = "1.2.3"\n')
    before = {p: p.read_bytes() for p in (project / n for n in mk_git.VERSION_FILES)}

    def fail(cfg: Config, argv: list[str], **kw: Any) -> None:
        raise MkError("uv lock failed")

    monkeypatch.setattr(mk_git, "run_host", fail)
    with pytest.raises(MkError, match="uv lock failed"):
        mk_git.apply_bump(cfg_for(project), "1.2.4")
    assert {p: p.read_bytes() for p in before} == before


# --- secret and restricted-data screening ----------------------------------------------


def test_name_screening() -> None:
    assert mk_git.name_problems(".env")
    assert mk_git.name_problems("config/.env.production")
    assert not mk_git.name_problems(".env.example")
    assert mk_git.name_problems("keys/server.pem")
    assert mk_git.name_problems("dataset/households.parquet")
    assert mk_git.name_problems("dataset/ICPSR_36151/DS0002/36151-0002-Data.tsv")
    assert mk_git.name_problems("artifacts/tier1.joblib")
    assert not mk_git.name_problems("dataset/README.md")
    assert not mk_git.name_problems("results/baseline.csv")


def test_content_screening_never_echoes_the_value() -> None:
    fake_aws = "AKIA" + "A" * 16  # assembled at runtime so no literal key sits in the source
    fake_gh = "ghp_" + "b" * 36
    assert mk_git.content_problems(f"key = {fake_aws}") == ["AWS access key"]
    assert mk_git.content_problems(f"t = {fake_gh}") == ["GitHub token"]
    pem_header = "-----BEGIN " + "RSA PRIVATE KEY-----"
    fake_password = 'password = "' + "Zx9fQ2mK7vLp4Rt8Yw3N" + '"'
    assert mk_git.content_problems(pem_header) == ["private key block"]
    assert mk_git.content_problems(fake_password) == ["hard-coded credential"]
    assert mk_git.content_problems('password = os.environ["PASSWORD"]') == []
    assert mk_git.content_problems("def token(): pass") == []


# --- branch detection ------------------------------------------------------------------


def test_default_branch_is_detected_not_assumed(
    tmp_path: Path, git_env: None, project: Path
) -> None:
    remote = tmp_path / "trunk-remote.git"
    git(tmp_path, "init", "--bare", "-b", "trunk", str(remote))
    git(project, "init", "-b", "trunk")
    git(project, "add", "-A")
    git(project, "commit", "-m", "init")
    git(project, "remote", "add", "origin", str(remote))
    git(project, "push", "-u", "origin", "trunk")
    git(project, "remote", "set-head", "origin", "trunk")
    assert mk_git.default_branch(cfg_for(project)) == "trunk"
    assert mk_git.current_branch(cfg_for(project)) == "trunk"


def test_default_branch_falls_back_to_the_remote_then_to_local_names(
    repo: tuple[Path, Path],
) -> None:
    root, _ = repo
    git(root, "remote", "set-head", "origin", "--delete")
    assert mk_git.default_branch(cfg_for(root)) == "main"  # asked of the remote itself
    offline = Config(root=root, offline=True)
    assert mk_git.default_branch(offline) == "main"  # local branch heuristics


def test_detached_head_is_detected_and_blocks_writes(repo: tuple[Path, Path]) -> None:
    root, _ = repo
    git(root, "checkout", "--detach")
    cfg = cfg_for(root, msg="x")
    assert mk_git.current_branch(cfg) is None
    for command in (mk_git.cmd_commit, mk_git.cmd_stage, mk_git.cmd_push, mk_git.cmd_pull):
        with pytest.raises(MkError, match="HEAD is detached"):
            command(cfg)
    report = mk_git.git_report(cfg)
    assert report.count("FAIL") == 1
    assert mk_git.git_report(cfg, detached_is_failure=False).count("FAIL") == 0


def test_missing_upstream_is_reported_and_push_sets_it(repo: tuple[Path, Path]) -> None:
    root, remote = repo
    git(root, "switch", "-c", "feature")
    write(root / "src/pkg/new.py", "Y = 2\n")
    git(root, "add", "src/pkg/new.py")
    git(root, "commit", "-m", "add new")
    cfg = cfg_for(root)
    assert mk_git.upstream(cfg) is None
    report = mk_git.git_report(cfg)
    assert "no upstream configured" in report.render()
    mk_git.cmd_push(cfg)
    assert mk_git.upstream(cfg) == "origin/feature"
    assert "refs/heads/feature" in git(root, "ls-remote", "--heads", str(remote))


def test_push_never_forces(repo: tuple[Path, Path]) -> None:
    root, _ = repo
    with pytest.raises(MkError, match="force-pushing is not supported"):
        mk_git.cmd_push(cfg_for(root, force="1"))
    with pytest.raises(MkError, match="force-pushing is not supported"):
        mk_git.cmd_release_push(cfg_for(root, force="1"))


def test_push_refuses_when_behind_and_does_not_retry(
    repo: tuple[Path, Path], tmp_path: Path
) -> None:
    root, remote = repo
    other = tmp_path / "other"
    git(tmp_path, "clone", str(remote), str(other))
    write(other / "other.txt", "o\n")
    git(other, "add", "other.txt")
    git(other, "commit", "-m", "from elsewhere")
    git(other, "push")
    write(root / "mine.txt", "m\n")
    git(root, "add", "mine.txt")
    git(root, "commit", "-m", "mine")
    with pytest.raises(MkError, match="behind origin/main"):
        mk_git.cmd_push(cfg_for(root))
    with pytest.raises(MkError, match="has diverged"):
        mk_git.cmd_pull(cfg_for(root))


def test_pull_fast_forwards_and_refuses_a_dirty_tree(
    repo: tuple[Path, Path], tmp_path: Path
) -> None:
    root, remote = repo
    other = tmp_path / "other"
    git(tmp_path, "clone", str(remote), str(other))
    write(other / "other.txt", "o\n")
    git(other, "add", "other.txt")
    git(other, "commit", "-m", "from elsewhere")
    git(other, "push")
    write(root / "src/pkg/mod.py", "X = 99\n")
    with pytest.raises(MkError, match="uncommitted changes to tracked files"):
        mk_git.cmd_pull(cfg_for(root))
    assert (root / "src/pkg/mod.py").read_text(encoding="utf-8") == "X = 99\n"  # never touched
    git(root, "checkout", "--", "src/pkg/mod.py")
    mk_git.cmd_pull(cfg_for(root))
    assert (root / "other.txt").exists()


# --- staging and committing ------------------------------------------------------------


def test_stage_refuses_secrets_ignored_and_restricted_files(repo: tuple[Path, Path]) -> None:
    root, _ = repo
    write(root / ".env", "TOKEN=1\n")
    write(root / "dataset/households.parquet", "bytes")
    write(root / "build/out.txt", "generated")
    write(root / "keys/server.pem", "x")
    write(root / "src/pkg/leaky.py", f"KEY = '{'AKIA' + 'A' * 16}'\n")
    for named in (".env", "dataset/households.parquet", "build/out.txt"):
        with pytest.raises(MkError, match=r"refusing to stage|git-ignored"):
            mk_git.cmd_stage(cfg_for(root, files=named))
    with pytest.raises(MkError, match="credentials or environment file"):
        mk_git.cmd_stage(cfg_for(root, files="keys/server.pem"))
    with pytest.raises(MkError, match="AWS access key"):
        mk_git.cmd_stage(cfg_for(root, files="src/pkg/leaky.py"))
    assert git(root, "diff", "--cached", "--name-only") == ""


def test_stage_without_files_stages_tracked_changes_only(repo: tuple[Path, Path]) -> None:
    root, _ = repo
    write(root / "src/pkg/mod.py", "X = 2\n")
    write(root / "untracked.txt", "new\n")
    mk_git.cmd_stage(cfg_for(root))
    assert git(root, "diff", "--cached", "--name-only") == "src/pkg/mod.py"


def test_stage_dry_run_changes_nothing(repo: tuple[Path, Path]) -> None:
    root, _ = repo
    write(root / "src/pkg/mod.py", "X = 2\n")
    mk_git.cmd_stage(Config(root=root, dry_run=True))
    assert git(root, "diff", "--cached", "--name-only") == ""


def test_commit_needs_a_message_staged_changes_and_clean_content(repo: tuple[Path, Path]) -> None:
    root, _ = repo
    with pytest.raises(MkError, match="MSG is required"):
        mk_git.cmd_commit(cfg_for(root))
    with pytest.raises(MkError, match="nothing is staged"):
        mk_git.cmd_commit(cfg_for(root, msg="feat: x"))
    write(root / "src/pkg/mod.py", "X = 2\n")
    mk_git.cmd_stage(cfg_for(root))
    mk_git.cmd_commit(cfg_for(root, msg="feat(pkg): bump X"))
    assert git(root, "log", "-1", "--format=%s") == "feat(pkg): bump X"
    write(root / "src/pkg/mod.py", f"X = '{'ghp_' + 'c' * 36}'\n")
    git(root, "add", "src/pkg/mod.py")
    with pytest.raises(MkError, match="GitHub token"):
        mk_git.cmd_commit(cfg_for(root, msg="feat: leak"))
    assert git(root, "log", "-1", "--format=%s") == "feat(pkg): bump X"


# --- releases --------------------------------------------------------------------------


def test_release_state_passes_on_a_clean_synced_default_branch(repo: tuple[Path, Path]) -> None:
    root, _ = repo
    report = mk_git.release_report(cfg_for(root), "1.2.3")
    assert report.count("FAIL") == 0, report.render()


def test_release_state_blocks_each_unsafe_condition(repo: tuple[Path, Path]) -> None:
    root, _ = repo
    write(root / "src/pkg/mod.py", "X = 2\n")
    assert "working tree: not clean" in mk_git.release_report(cfg_for(root), "1.2.3").render()
    git(root, "checkout", "--", "src/pkg/mod.py")

    git(root, "switch", "-c", "topic")
    git(root, "push", "-u", "origin", "topic")
    assert "neither the default branch" in mk_git.release_report(cfg_for(root), "1.2.3").render()
    git(root, "switch", "main")

    git(root, "tag", "-a", "v1.2.3", "-m", "x")
    assert "v1.2.3 already exists locally" in mk_git.release_report(cfg_for(root), "1.2.3").render()
    assert "not newer than v1.2.3" in mk_git.release_report(cfg_for(root), "1.2.2").render()
    assert "not a Semantic Version" in mk_git.release_report(cfg_for(root), "1.2").render()


def test_release_state_detects_a_tag_already_on_the_remote(repo: tuple[Path, Path]) -> None:
    root, _ = repo
    git(root, "tag", "-a", "v1.2.3", "-m", "x")
    git(root, "push", "origin", "v1.2.3")
    git(root, "tag", "-d", "v1.2.3")
    assert "already exists on origin" in mk_git.release_report(cfg_for(root), "1.2.3").render()


def test_release_tag_creates_one_annotated_local_tag_and_never_overwrites(
    repo: tuple[Path, Path],
) -> None:
    root, remote = repo
    cfg = cfg_for(root)
    mk_git.cmd_release_tag(cfg)
    assert git(root, "cat-file", "-t", "refs/tags/v1.2.3") == "tag"  # annotated
    assert git(root, "ls-remote", "--tags", str(remote)) == ""  # nothing published
    with pytest.raises(MkError, match="release state is not valid"):
        mk_git.cmd_release_tag(cfg)
    assert mk_git.cmd_release_verify(cfg) == 0


def test_release_bumps_commits_tags_and_pushes_nothing(
    repo: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, remote = repo
    real = mk_core.run_host

    def fake(cfg: Config, argv: list[str], **kw: Any) -> Any:
        return real(cfg, ["git", "--version"], **kw) if argv[0] == cfg.uv else real(cfg, argv, **kw)

    monkeypatch.setattr(mk_git, "run_host", fake)  # `uv lock` is out of scope here
    mk_git.cmd_release(cfg_for(root, bump="minor"))
    assert mk_core.project_version(root) == "1.3.0"
    assert 'version: "1.3.0"' in (root / "CITATION.cff").read_text(encoding="utf-8")
    assert git(root, "log", "-1", "--format=%s") == "chore(release): v1.3.0"
    assert git(root, "status", "--porcelain") == ""
    assert git(root, "cat-file", "-t", "refs/tags/v1.3.0") == "tag"
    assert git(root, "ls-remote", "--tags", str(remote)) == ""
    assert "refs/heads/main" in git(root, "ls-remote", "--heads", str(remote))
    assert git(root, "rev-list", "--count", "origin/main..HEAD") == "1"  # committed, not pushed


def test_release_requires_a_bump_and_a_valid_state(repo: tuple[Path, Path]) -> None:
    root, _ = repo
    with pytest.raises(MkError, match="BUMP is required"):
        mk_git.cmd_release(cfg_for(root))
    write(root / "src/pkg/mod.py", "X = 2\n")
    with pytest.raises(MkError, match="release state is not valid"):
        mk_git.cmd_release(cfg_for(root, bump="patch"))
    assert mk_core.project_version(root) == "1.2.3"  # nothing changed


def test_release_dry_run_changes_nothing(repo: tuple[Path, Path]) -> None:
    root, _ = repo
    before = (root / "pyproject.toml").read_bytes()
    assert (
        mk_git.cmd_release(
            Config(root=root, dry_run=True, args={"bump": "major"}, uv=sys.executable)
        )
        == 0
    )
    assert (root / "pyproject.toml").read_bytes() == before
    assert git(root, "tag", "--list") == ""


def test_release_push_publishes_branch_and_one_tag_only_when_confirmed(
    repo: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, remote = repo
    mk_git.cmd_release_tag(cfg_for(root))
    git(root, "tag", "-a", "v0.0.1-local", "-m", "unrelated local tag")
    monkeypatch.setattr("sys.stdin", None)

    with pytest.raises(MkError, match="confirmation required"):
        mk_git.cmd_release_push(cfg_for(root))
    assert git(root, "ls-remote", "--tags", str(remote)) == ""

    mk_git.cmd_release_push(Config(root=root, dry_run=True))
    assert git(root, "ls-remote", "--tags", str(remote)) == ""

    with pytest.raises(MkError, match="CI is set"):
        mk_git.cmd_release_push(Config(root=root, confirm=True, ci=True))

    mk_git.cmd_release_push(Config(root=root, confirm=True))
    remote_tags = git(root, "ls-remote", "--tags", str(remote))
    assert "refs/tags/v1.2.3" in remote_tags and "v0.0.1-local" not in remote_tags  # no --tags

    with pytest.raises(MkError, match="already exists on origin"):
        mk_git.cmd_release_push(Config(root=root, confirm=True))


def test_release_push_needs_a_tag_at_head(repo: tuple[Path, Path]) -> None:
    root, _ = repo
    with pytest.raises(MkError, match="does not exist locally"):
        mk_git.cmd_release_push(Config(root=root, confirm=True))
    mk_git.cmd_release_tag(cfg_for(root))
    write(root / "src/pkg/mod.py", "X = 5\n")
    git(root, "add", "-A")
    git(root, "commit", "-m", "after the tag")
    with pytest.raises(MkError, match="does not point at HEAD"):
        mk_git.cmd_release_push(Config(root=root, confirm=True))


def test_release_notes_group_commits_since_the_last_tag(
    repo: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    root, _ = repo
    git(root, "tag", "-a", "v1.0.0", "-m", "first")
    for message in ("feat(api): add endpoint", "fix: off by one", "update the readme"):
        write(root / "f.txt", message)
        git(root, "add", "f.txt")
        git(root, "commit", "-m", message)
    mk_git.cmd_release_notes(cfg_for(root, bump="minor"))
    text = (root / "build/release/RELEASE_NOTES-v1.3.0.md").read_text(encoding="utf-8")
    assert "# v1.3.0" in text and "since v1.0.0 (3 commits)" in text
    assert "## Features\n- add endpoint" in text and "## Fixes\n- off by one" in text
    assert "## Other\n- update the readme" in text
    assert "initial" not in text  # made before the last tag
    capsys.readouterr()


def test_release_needs_uv_because_the_lock_cannot_be_verified_without_it(
    repo: tuple[Path, Path],
) -> None:
    root, _ = repo
    for cfg in (
        Config(root=root, uv="no-such-uv"),
        Config(root=root, mode="pip", uv=sys.executable),
    ):
        report = mk_git.release_report(cfg, "1.2.3")
        assert "releases need uv" in report.render() and report.count("FAIL") >= 1


def test_version_bump_keeps_crlf_line_endings(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    crlf, lf = b"\r\n", b"\n"
    for name in ("pyproject.toml", "CITATION.cff"):
        raw = (project / name).read_bytes().replace(crlf, lf).replace(lf, crlf)
        (project / name).write_bytes(raw)
    monkeypatch.setattr(mk_git, "run_host", lambda *a, **k: None)  # `uv lock` is out of scope
    mk_git.apply_bump(cfg_for(project), "1.2.4")
    for name in ("pyproject.toml", "CITATION.cff"):
        data = (project / name).read_bytes()
        assert crlf in data and lf not in data.replace(crlf, b"")
    assert mk_core.project_version(project) == "1.2.4"
    assert b'version: "1.2.4"' in (project / "CITATION.cff").read_bytes()
