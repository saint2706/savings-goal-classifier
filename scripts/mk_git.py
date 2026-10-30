"""Git workflow, version and release commands for the Makefile helper.

Safety rules enforced here, not left to the caller: nothing force-pushes, rewrites
history, amends, deletes a tag, or stages with ``add -A``; every write command refuses a
detached HEAD; pushes and tags are only ever created by the target that names them.
"""

from __future__ import annotations

import fnmatch
import re
import shlex
import shutil
import tomllib
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from mk_core import (
    Config,
    MkError,
    Report,
    info,
    load_pyproject,
    ok,
    project_version,
    require_confirmation,
    run_host,
    warn,
)

SEMVER_RE = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)
TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
VERSION_FILES = ("pyproject.toml", "CITATION.cff", "uv.lock")

# Names that must never be committed, and generated or licensed-data paths that are
# only committed on purpose (and so never by the staging helpers).
SECRET_NAMES = (
    ".env", ".env.*", "*.pem", "*.key", "*.p12", "*.pfx", "id_rsa*", "id_ed25519*",
    "*.kdbx", "credentials*.json", ".netrc", "*.keystore",
)  # fmt: skip
SECRET_NAME_OK = (".env.example", ".env.sample", ".env.template")
RESTRICTED_PATHS = (
    "dataset/*.parquet", "dataset/*.csv", "dataset/*.tsv", "dataset/ICPSR_36151/*",
    "dataset/hces/*", "36151-*", "*/36151-*", "artifacts/*", "*.joblib", "build/*",
    "demo_out/*", ".venv/*", "feedback/*",
)  # fmt: skip
SECRET_CONTENT = {
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private key block": re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "Slack token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    "hard-coded credential": re.compile(
        r"""(?i)\b(?:api[_-]?key|secret|passwd|password|token)\b["']?\s*[:=]\s*["'][^"'\s]{16,}["']"""
    ),
}


# --- git plumbing ----------------------------------------------------------------------


def git(cfg: Config, *args: str, check: bool = False, timeout: float = 60) -> tuple[int, str, str]:
    proc = run_host(cfg, ["git", *args], capture=True, check=check, timeout=timeout)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def git_out(cfg: Config, *args: str) -> str:
    rc, out, _ = git(cfg, *args)
    return out if rc == 0 else ""


def git_passthrough(cfg: Config, *args: str) -> None:
    run_host(cfg, ["git", *args], check=True)


def in_repo(cfg: Config) -> bool:
    return git_out(cfg, "rev-parse", "--is-inside-work-tree") == "true"


def require_repo(cfg: Config) -> None:
    if not in_repo(cfg):
        raise MkError(
            f"{cfg.root} is not inside a git work tree", "Run `git init` or clone the repo."
        )


def current_branch(cfg: Config) -> str | None:
    rc, out, _ = git(cfg, "symbolic-ref", "--quiet", "--short", "HEAD")
    return out if rc == 0 and out else None


def require_branch(cfg: Config) -> str:
    branch = current_branch(cfg)
    if branch is None:
        raise MkError(
            "HEAD is detached",
            "Check out a branch first:  git switch <branch>   (or: git switch -c <new-branch>)",
        )
    return branch


def default_branch(cfg: Config) -> str | None:
    """From the remote's HEAD ref, then the remote itself, then local conventions: never assumed."""
    remote = cfg.remote
    rc, out, _ = git(cfg, "symbolic-ref", "--quiet", "--short", f"refs/remotes/{remote}/HEAD")
    if rc == 0 and out.startswith(f"{remote}/"):
        return out[len(remote) + 1 :]
    if not cfg.offline:
        rc, out, _ = git(cfg, "ls-remote", "--symref", remote, "HEAD", timeout=20)
        m = re.search(r"ref: refs/heads/(\S+)\s+HEAD", out) if rc == 0 else None
        if m:
            return m.group(1)
    configured = git_out(cfg, "config", "--get", "init.defaultBranch")
    for name in filter(None, (configured, "main", "master")):
        if git(cfg, "show-ref", "--verify", "--quiet", f"refs/heads/{name}")[0] == 0:
            return name
    return None


def upstream(cfg: Config) -> str | None:
    rc, out, _ = git(cfg, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
    return out if rc == 0 and out else None


def ahead_behind(cfg: Config) -> tuple[int, int] | None:
    rc, out, _ = git(cfg, "rev-list", "--left-right", "--count", "HEAD...@{upstream}")
    if rc != 0:
        return None
    ahead, behind = out.split()
    return int(ahead), int(behind)


def status_lines(cfg: Config) -> list[str]:
    return git_out(cfg, "status", "--porcelain=v1").splitlines()


def tracked_changes(cfg: Config) -> list[str]:
    return [line for line in status_lines(cfg) if not line.startswith("??")]


def operation_in_progress(cfg: Config) -> str | None:
    for marker, name in (
        ("MERGE_HEAD", "merge"),
        ("CHERRY_PICK_HEAD", "cherry-pick"),
        ("REVERT_HEAD", "revert"),
    ):
        if git(cfg, "rev-parse", "-q", "--verify", marker)[0] == 0:
            return name
    for d in ("rebase-merge", "rebase-apply"):
        if Path(git_out(cfg, "rev-parse", "--git-path", d)).exists():
            return "rebase"
    return None


def remote_url(cfg: Config) -> str | None:
    return git_out(cfg, "remote", "get-url", cfg.remote) or None


def remote_reachable(cfg: Config) -> bool:
    return git(cfg, "ls-remote", "--exit-code", "--quiet", cfg.remote, "HEAD", timeout=30)[0] == 0


def fetch(cfg: Config) -> bool:
    if cfg.offline:
        return False
    rc, _, err = git(cfg, "fetch", "--quiet", "--prune", cfg.remote, timeout=120)
    if rc != 0:
        warn(f"fetch from {cfg.remote} failed: {err.splitlines()[-1] if err else 'unknown error'}")
    return rc == 0


# --- secret and restricted-data screening ----------------------------------------------


def name_problems(path: str) -> list[str]:
    posix = path.replace("\\", "/")
    base = posix.rsplit("/", 1)[-1]
    problems: list[str] = []
    if base not in SECRET_NAME_OK and any(fnmatch.fnmatch(base, p) for p in SECRET_NAMES):
        problems.append("looks like a credentials or environment file")
    if any(fnmatch.fnmatch(posix, p) for p in RESTRICTED_PATHS):
        problems.append("is generated or restricted data (ICPSR terms forbid redistribution)")
    return problems


def content_problems(text: str) -> list[str]:
    return [label for label, rx in SECRET_CONTENT.items() if rx.search(text)]


def scan_paths(cfg: Config, paths: Sequence[str]) -> list[str]:
    """Problems with the named files' names and (if text) contents; values are never echoed."""
    found: list[str] = []
    for p in paths:
        found += [f"{p}: {why}" for why in name_problems(p)]
        fp = cfg.path(p)
        if fp.is_file() and fp.stat().st_size < 2_000_000:
            try:
                text = fp.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            found += [f"{p}: contains a possible {why}" for why in content_problems(text)]
    return found


def scan_staged(cfg: Config) -> list[str]:
    names = git_out(cfg, "diff", "--cached", "--name-only", "--diff-filter=ACMR").splitlines()
    found = [f"{n}: {why}" for n in names for why in name_problems(n)]
    current = ""
    for line in git_out(cfg, "diff", "--cached", "-U0", "--no-color").splitlines():
        if line.startswith("+++ b/"):
            current = line[6:]
        elif line.startswith("+") and not line.startswith("+++"):
            found += [f"{current}: contains a possible {why}" for why in content_problems(line[1:])]
    return sorted(set(found))


# --- targets: inspection ---------------------------------------------------------------


def cmd_status(cfg: Config) -> int:
    require_repo(cfg)
    git_passthrough(cfg, "status", "--short", "--branch")
    return 0


def cmd_branch(cfg: Config) -> int:
    require_repo(cfg)
    branch = current_branch(cfg)
    head = git_out(cfg, "rev-parse", "--short", "HEAD")
    print(f"branch:         {branch or f'(detached HEAD at {head})'}")
    print(f"default branch: {default_branch(cfg) or '(not detected)'}")
    print(f"remote:         {cfg.remote} -> {remote_url(cfg) or '(not configured)'}")
    up = upstream(cfg)
    print(f"upstream:       {up or '(none)'}")
    ab = ahead_behind(cfg)
    if ab:
        print(f"ahead/behind:   {ab[0]} ahead, {ab[1]} behind (as of the last fetch)")
    changes = status_lines(cfg)
    print(f"working tree:   {'clean' if not changes else f'{len(changes)} changed path(s)'}")
    return 0


def cmd_diff(cfg: Config) -> int:
    require_repo(cfg)
    print("-- unstaged --")
    git_passthrough(cfg, "diff", "--stat")
    print("-- staged --")
    git_passthrough(cfg, "diff", "--cached", "--stat")
    untracked = [ln[3:] for ln in status_lines(cfg) if ln.startswith("??")]
    print("-- untracked (not staged by `make git-stage` unless named in FILES) --")
    print("\n".join(untracked) or "(none)")
    return 0


def git_report(cfg: Config, detached_is_failure: bool = True) -> Report:
    r = Report("git safety check")
    if not in_repo(cfg):
        r.failed(
            "repository", f"{cfg.root} is not a git work tree", "Run `git init` or clone the repo."
        )
        return r
    r.passed("repository", str(cfg.root))
    branch = current_branch(cfg)
    if branch is None:
        add = r.failed if detached_is_failure else r.warned
        add("branch", "HEAD is detached", "git switch <branch>   (or: git switch -c <new-branch>)")
    else:
        r.passed("branch", branch)
    op = operation_in_progress(cfg)
    if op:
        r.failed("operation", f"a {op} is in progress", f"Finish or abort it (git {op} --abort).")
    unmerged = [
        ln for ln in status_lines(cfg) if ln[:2] in {"UU", "AA", "DD", "AU", "UA", "DU", "UD"}
    ]
    if unmerged:
        r.failed("conflicts", f"{len(unmerged)} unmerged path(s)", "Resolve them, then `git add`.")
    default = default_branch(cfg)
    if default:
        r.passed("default branch", default)
    else:
        r.warned("default branch", "not detected", f"Run: git remote set-head {cfg.remote} -a")
    if remote_url(cfg):
        r.passed("remote", f"{cfg.remote} -> {remote_url(cfg)}")
    else:
        r.warned(
            "remote", f"`{cfg.remote}` is not configured", f"git remote add {cfg.remote} <url>"
        )
    if branch and upstream(cfg):
        ab = ahead_behind(cfg)
        r.passed("upstream", upstream(cfg) or "")
        if ab and ab[1]:
            r.warned("sync", f"{ab[1]} commit(s) behind upstream", "make git-sync")
    elif branch:
        r.warned("upstream", "no upstream configured", "make git-push sets it on the first push")
    changes = status_lines(cfg)
    if changes:
        r.warned("working tree", f"{len(changes)} changed path(s)", "make git-diff")
    else:
        r.passed("working tree", "clean")
    staged = scan_staged(cfg)
    if staged:
        r.failed("staged content", "; ".join(staged), "Unstage with: git restore --staged <path>")
    return r


def cmd_check(cfg: Config) -> int:
    return git_report(cfg).finish()


# --- targets: writing ------------------------------------------------------------------


def cmd_stage(cfg: Config) -> int:
    """Stage tracked modifications, or exactly the paths in FILES; never ``add -A``."""
    require_repo(cfg)
    require_branch(cfg)
    named = shlex.split(cfg.arg("files"))
    if named:
        paths = named
        for p in paths:
            if not cfg.path(p).exists() and git(cfg, "ls-files", "--", p)[1] == "":
                raise MkError(f"FILES entry does not exist: {p}")
            if git(cfg, "check-ignore", "-q", "--", p)[0] == 0:
                raise MkError(f"{p} is git-ignored (generated or local-only); not staging it")
    else:
        paths = git_out(cfg, "diff", "--name-only").splitlines()
    if not paths:
        info('nothing to stage (no tracked file is modified; name new files with FILES="a b")')
        return 0
    problems = scan_paths(cfg, paths)
    if problems:
        raise MkError(
            "refusing to stage:\n  " + "\n  ".join(problems), "Remove or ignore these files first."
        )
    info(
        f"staging {len(paths)} path(s): "
        + ", ".join(paths[:8])
        + (" ..." if len(paths) > 8 else "")
    )
    if cfg.dry_run:
        info("dry-run: nothing staged")
        return 0
    if named:
        git_passthrough(cfg, "add", "--", *paths)
    else:
        git_passthrough(cfg, "add", "-u")
    return 0


def cmd_commit(cfg: Config) -> int:
    require_repo(cfg)
    branch = require_branch(cfg)
    message = cfg.arg("msg")
    if not message:
        raise MkError("MSG is required", 'make git-commit MSG="feat(scope): what changed"')
    if "\n" not in message and len(message) > 72:
        warn(f"subject is {len(message)} characters; 72 or fewer reads better in `git log`")
    if git(cfg, "diff", "--cached", "--quiet")[0] == 0:
        raise MkError("nothing is staged", "make git-stage   (or: git add <files>)")
    problems = scan_staged(cfg)
    if problems:
        raise MkError(
            "refusing to commit:\n  " + "\n  ".join(problems), "git restore --staged <path>"
        )
    info(f"committing on {branch}: {message.splitlines()[0]}")
    if cfg.dry_run:
        info("dry-run: no commit created")
        return 0
    git_passthrough(cfg, "commit", "-m", message)  # hooks run; never --no-verify
    return 0


def cmd_pull(cfg: Config) -> int:
    require_repo(cfg)
    branch = require_branch(cfg)
    if not upstream(cfg):
        raise MkError(
            f"{branch} has no upstream", f"git branch --set-upstream-to={cfg.remote}/{branch}"
        )
    if tracked_changes(cfg):
        raise MkError(
            "the working tree has uncommitted changes to tracked files",
            "Commit them (make git-stage git-commit MSG=...) or set them aside first;\n"
            "this target never stashes or discards your work.",
        )
    if cfg.dry_run:
        info(f"dry-run: would fetch {cfg.remote} and fast-forward {branch}")
        return 0
    if not fetch(cfg):
        raise MkError(
            f"could not fetch from {cfg.remote}", "Check the network, or run with OFFLINE=1."
        )
    ab = ahead_behind(cfg)
    if ab and ab[0] and ab[1]:
        raise MkError(
            f"{branch} has diverged from {upstream(cfg)} ({ab[0]} ahead, {ab[1]} behind)",
            "Reconcile it yourself (git rebase / git merge); this target only fast-forwards.",
        )
    if ab and ab[1]:
        git_passthrough(cfg, "merge", "--ff-only", "@{upstream}")
    else:
        ok(f"{branch} is up to date with {upstream(cfg)}")
    return 0


def cmd_sync(cfg: Config) -> int:
    cmd_pull(cfg)
    ab = ahead_behind(cfg)
    if ab and ab[0]:
        info(f"{ab[0]} local commit(s) not pushed; review them, then `make git-push`")
    return 0


def cmd_push(cfg: Config) -> int:
    """Plain ``git push`` of the current branch: no force, no tags, no history rewriting."""
    if cfg.arg("force"):
        raise MkError("force-pushing is not supported by this Makefile")
    require_repo(cfg)
    branch = require_branch(cfg)
    if not remote_url(cfg):
        raise MkError(
            f"remote `{cfg.remote}` is not configured", f"git remote add {cfg.remote} <url>"
        )
    if tracked_changes(cfg):
        warn("uncommitted changes are not part of this push")
    if cfg.dry_run:
        info(f"dry-run: would push {branch} to {cfg.remote}")
        return 0
    if not remote_reachable(cfg):
        raise MkError(
            f"cannot reach {cfg.remote} ({remote_url(cfg)})", "Check the network and credentials."
        )
    fetch(cfg)
    up = upstream(cfg)
    if up:
        ab = ahead_behind(cfg)
        if ab and ab[1]:
            raise MkError(
                f"{branch} is {ab[1]} commit(s) behind {up}",
                "make git-sync first; a rejected push is never retried with force.",
            )
        if ab and not ab[0]:
            ok("nothing to push")
            return 0
        args = ["push", cfg.remote, f"HEAD:refs/heads/{branch}"]
    else:
        if git(cfg, "ls-remote", "--exit-code", "--heads", cfg.remote, branch, timeout=30)[0] == 0:
            raise MkError(
                f"{cfg.remote}/{branch} already exists but {branch} does not track it",
                f"git branch --set-upstream-to={cfg.remote}/{branch}, then make git-sync",
            )
        info(f"{branch} has no upstream: pushing and setting {cfg.remote}/{branch} as upstream")
        args = ["push", "--set-upstream", cfg.remote, f"HEAD:refs/heads/{branch}"]
    if branch == default_branch(cfg):
        warn(f"pushing the default branch; branch protection on {cfg.remote} may reject it")
    rc, _, err = git(cfg, *args, timeout=300)
    if rc != 0:
        raise MkError(
            f"push rejected or failed: {err.splitlines()[-1] if err else 'unknown error'}",
            "Not retried. If the branch is protected, push a feature branch and open a pull request.",
        )
    ok(f"pushed {branch} to {cfg.remote}")
    return 0


# --- versions --------------------------------------------------------------------------


def parse_semver(version: str) -> tuple[int, int, int, str | None]:
    m = SEMVER_RE.match(version)
    if not m:
        raise MkError(
            f"{version!r} is not a Semantic Version (MAJOR.MINOR.PATCH)", "See https://semver.org"
        )
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)


def bump(version: str, part: str) -> str:
    major, minor, patch, pre = parse_semver(version)
    if pre:
        raise MkError(f"{version} is a pre-release; set the next version by hand")
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    if part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise MkError(f"BUMP must be patch, minor or major (got {part!r})")


def citation_version(root: Path) -> str | None:
    path = root / "CITATION.cff"
    if not path.exists():
        return None
    m = re.search(r'(?m)^version:\s*"?([^"\s#]+)"?', path.read_text(encoding="utf-8"))
    return m.group(1) if m else None


def lock_version(root: Path) -> str | None:
    path = root / "uv.lock"
    if not path.exists():
        return None
    name = str(load_pyproject(root).get("project", {}).get("name", ""))
    for pkg in tomllib.loads(path.read_text(encoding="utf-8")).get("package", []):
        if pkg.get("name") == name:
            return str(pkg.get("version"))
    return None


def set_pyproject_version(text: str, new: str) -> str:
    """Rewrite ``version`` inside the ``[project]`` table only, leaving every other byte alone."""
    out, in_project, done = [], False, False
    for line in text.splitlines(keepends=True):
        header = re.match(r"^\s*\[+([^\]]+)\]+\s*$", line)
        if header:
            in_project = header.group(1).strip() == "project"
        elif in_project and not done and re.match(r"^version\s*=", line):
            line = re.sub(r'"[^"]*"', f'"{new}"', line, count=1)
            done = True
        out.append(line)
    if not done:
        raise MkError("could not find `version = ...` under [project] in pyproject.toml")
    return "".join(out)


def set_citation(text: str, new: str, today: str) -> str:
    text, n = re.subn(r'(?m)^(version:\s*)"?[^"\s#]+"?', rf'\g<1>"{new}"', text, count=1)
    if n != 1:
        raise MkError("could not find `version:` in CITATION.cff")
    return re.sub(r'(?m)^(date-released:\s*)"?[^"\s#]+"?', rf'\g<1>"{today}"', text, count=1)


def version_report(cfg: Config) -> Report:
    r = Report("version metadata")
    version = project_version(cfg.root)
    try:
        parse_semver(version)
        r.passed("pyproject.toml", f"{version} is valid SemVer")
    except MkError as exc:
        r.failed("pyproject.toml", str(exc))
    for label, found in (
        ("CITATION.cff", citation_version(cfg.root)),
        ("uv.lock", lock_version(cfg.root)),
    ):
        if found is None:
            r.warned(label, "no version recorded")
        elif found != version:
            fix = (
                "make version-check after `uv lock`"
                if label == "uv.lock"
                else "edit it to match pyproject.toml"
            )
            r.failed(label, f"{found} != pyproject.toml {version}", fix)
        else:
            r.passed(label, found)
    return r


def cmd_version(cfg: Config) -> int:
    version = project_version(cfg.root)
    parse_semver(version)
    print(f"version:        {version}   (source of truth: [project].version in pyproject.toml)")
    print(f"tag:            v{version}")
    print(f"latest tag:     {latest_tag(cfg) or '(none)'}")
    for part in ("patch", "minor", "major"):
        print(f"next {part}:".ljust(16) + bump(version, part))
    return 0


def cmd_version_check(cfg: Config) -> int:
    return version_report(cfg).finish()


def cmd_version_bump(cfg: Config) -> int:
    part = cfg.arg("part") or cfg.arg("bump")
    old = project_version(cfg.root)
    new = bump(old, part)
    require_repo(cfg)
    if tracked_changes(cfg) or status_lines(cfg):
        raise MkError(
            "the working tree is not clean",
            "Commit or set aside your changes before bumping the version.",
        )
    info(f"{old} -> {new}: editing {', '.join(VERSION_FILES)}")
    if cfg.dry_run:
        info("dry-run: no file changed")
        return 0
    apply_bump(cfg, new)
    ok(
        f"version is now {new}; review with `git diff`, then commit it or use `make release BUMP={part}`"
    )
    return 0


def apply_bump(cfg: Config, new: str) -> None:
    """Edit pyproject.toml and CITATION.cff, refresh uv.lock; restore everything if any step fails."""
    pyproject, citation = cfg.root / "pyproject.toml", cfg.root / "CITATION.cff"
    saved = {p: p.read_bytes() for p in (pyproject, cfg.root / "uv.lock", citation) if p.exists()}
    try:
        # Bytes in, bytes out: a CRLF checkout (core.autocrlf on Windows) keeps its line endings.
        pyproject.write_bytes(
            set_pyproject_version(pyproject.read_bytes().decode("utf-8"), new).encode("utf-8")
        )
        if citation.exists():
            today = datetime.now(UTC).strftime("%Y-%m-%d")
            citation.write_bytes(
                set_citation(citation.read_bytes().decode("utf-8"), new, today).encode("utf-8")
            )
        run_host(cfg, [cfg.uv, "lock"], tool="uv")
        report = version_report(cfg)
        if report.count("FAIL"):
            raise MkError(report.render())
    except (MkError, OSError):
        for p, data in saved.items():
            p.write_bytes(data)
        warn("version bump failed; restored pyproject.toml, CITATION.cff and uv.lock")
        raise


# --- tags and releases -----------------------------------------------------------------


def semver_tags(cfg: Config) -> list[tuple[tuple[int, int, int], str]]:
    tags = git_out(cfg, "tag", "--list", "v*").splitlines()
    parsed = [((int(m[1]), int(m[2]), int(m[3])), t) for t in tags if (m := TAG_RE.match(t))]
    return sorted(parsed)


def latest_tag(cfg: Config) -> str | None:
    tags = semver_tags(cfg)
    return tags[-1][1] if tags else None


def remote_has_tag(cfg: Config, tag: str) -> bool | None:
    if cfg.offline:
        return None
    rc, out, _ = git(cfg, "ls-remote", "--tags", cfg.remote, f"refs/tags/{tag}", timeout=30)
    return bool(out) if rc == 0 else None


def release_report(cfg: Config, target: str) -> Report:
    """Branch, tree and tag state a release needs: every FAIL here blocks tagging."""
    r = Report(f"release state for v{target}")
    try:
        parse_semver(target)
        r.passed("version", f"{target} is valid SemVer")
    except MkError as exc:
        r.failed("version", str(exc))
        return r
    require_repo(cfg)
    if cfg.mode == "pip" or shutil.which(cfg.uv) is None:
        r.failed(
            "uv",
            "releases need uv: without it uv.lock cannot be verified or refreshed",
            "Install uv and drop MODE=pip.",
        )
    else:
        r.passed("uv", "available")
    branch = current_branch(cfg)
    default = default_branch(cfg)
    if branch is None:
        r.failed("branch", "HEAD is detached", "Check out the release branch first.")
    elif default is None:
        r.failed(
            "branch",
            f"{branch}: default branch could not be detected",
            f"git remote set-head {cfg.remote} -a",
        )
    elif branch == default or fnmatch.fnmatch(branch, "release/*"):
        r.passed("branch", f"{branch} (default branch: {default})")
    else:
        r.failed(
            "branch",
            f"{branch} is neither the default branch ({default}) nor release/*",
            f"git switch {default}",
        )
    if tracked_changes(cfg) or status_lines(cfg):
        r.failed(
            "working tree", "not clean", "Commit or set aside every change first (make git-diff)."
        )
    else:
        r.passed("working tree", "clean")
    op = operation_in_progress(cfg)
    if op:
        r.failed("operation", f"a {op} is in progress")
    fetched = fetch(cfg)
    if not upstream(cfg):
        r.failed("upstream", "no upstream configured", "make git-push sets one; then re-run.")
    else:
        ab = ahead_behind(cfg)
        detail = f"{ab[0]} ahead, {ab[1]} behind {upstream(cfg)}" if ab else str(upstream(cfg))
        if ab and ab[1]:
            r.failed("upstream", detail, "make git-sync")
        elif fetched or cfg.offline:
            r.passed("upstream", detail)
        else:
            r.failed(
                "upstream",
                f"{detail}; could not fetch to verify",
                "Fix the connection, or OFFLINE=1 to skip.",
            )
    tag = f"v{target}"
    if tag in git_out(cfg, "tag", "--list", tag).splitlines():
        r.failed(
            "tag",
            f"{tag} already exists locally; tags are never overwritten",
            "Choose a new version.",
        )
    else:
        r.passed("tag", f"{tag} is unused locally")
    remote = remote_has_tag(cfg, tag)
    if remote:
        r.failed("remote tag", f"{tag} already exists on {cfg.remote}")
    elif remote is None:
        (r.warned if cfg.offline else r.failed)(
            "remote tag", f"could not check {cfg.remote}", "Retry online, or OFFLINE=1."
        )
    else:
        r.passed("remote tag", f"{tag} is unused on {cfg.remote}")
    tags = semver_tags(cfg)
    if tags:
        newest = tags[-1]
        if tuple(parse_semver(target)[:3]) <= newest[0]:
            r.failed("monotonic", f"{target} is not newer than {newest[1]}", "Bump the version.")
        else:
            r.passed("monotonic", f"{target} is newer than {newest[1]}")
    return r


def release_target(cfg: Config) -> str:
    current = project_version(cfg.root)
    part = cfg.arg("bump")
    return bump(current, part) if part else current


def cmd_release_state(cfg: Config) -> int:
    report = release_report(cfg, release_target(cfg))
    if not cfg.arg("bump"):
        report.extend(version_report(cfg))
    return report.finish()


def cmd_release_notes(cfg: Config) -> int:
    require_repo(cfg)
    version = release_target(cfg)
    last = latest_tag(cfg)
    rng = f"{last}..HEAD" if last else "HEAD"
    subjects = git_out(cfg, "log", "--no-merges", "--format=%s", rng).splitlines()
    groups = {
        "feat": "Features", "fix": "Fixes", "perf": "Performance", "refactor": "Refactoring",
        "docs": "Documentation", "test": "Tests", "build": "Build", "ci": "CI", "style": "Style",
        "chore": "Maintenance",
    }  # fmt: skip
    buckets: dict[str, list[str]] = {}
    for s in subjects:
        m = re.match(r"^(\w+)(?:\([^)]*\))?!?:\s*(.+)$", s)
        key = groups.get(m.group(1), "Other") if m else "Other"
        buckets.setdefault(key, []).append(m.group(2) if m and key != "Other" else s)
    lines = [
        f"# v{version}",
        "",
        f"Changes since {last or 'the start of the history'} ({len(subjects)} commits).",
        "",
    ]
    for title in [*groups.values(), "Other"]:
        if title in buckets:
            lines += [f"## {title}", *[f"- {s}" for s in buckets[title]], ""]
    text = "\n".join(lines)
    out = cfg.dir("build") / "release" / f"RELEASE_NOTES-v{version}.md"
    print(text)
    if cfg.dry_run:
        info(f"dry-run: would write {out}")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    ok(f"wrote {out}")
    return 0


def create_tag(cfg: Config, version: str) -> None:
    tag = f"v{version}"
    git(cfg, "tag", "-a", tag, "-m", f"Release {tag}", check=True)
    ok(
        f"created annotated tag {tag} at {git_out(cfg, 'rev-parse', '--short', 'HEAD')} (local only)"
    )


def cmd_release_tag(cfg: Config) -> int:
    """Tag the current, already-committed version. Local only; nothing is pushed."""
    version = project_version(cfg.root)
    report = release_report(cfg, version)
    report.extend(version_report(cfg))
    if report.count("FAIL"):
        print(report.render())
        raise MkError("release state is not valid; no tag created")
    info(f"tagging HEAD as v{version}")
    if cfg.dry_run:
        info("dry-run: no tag created")
        return 0
    create_tag(cfg, version)
    info("publish it explicitly with: make release-push CONFIRM=1")
    return 0


def cmd_release(cfg: Config) -> int:
    """Bump, refresh the lock, commit the version files, tag. Local only; nothing is pushed."""
    part = cfg.arg("bump")
    old = project_version(cfg.root)
    if part not in {"patch", "minor", "major"}:
        raise MkError(
            "BUMP is required: patch, minor or major",
            "make release BUMP=patch   (to tag the version already in pyproject.toml use: make release-tag)",
        )
    new = bump(old, part)
    report = release_report(cfg, new)
    if report.count("FAIL"):
        print(report.render())
        raise MkError("release state is not valid; nothing changed")
    info(
        f"release {old} -> {new}: edit {', '.join(VERSION_FILES)}, commit, tag v{new} (nothing is pushed)"
    )
    if cfg.dry_run:
        info("dry-run: nothing changed")
        return 0
    apply_bump(cfg, new)
    files = [f for f in VERSION_FILES if (cfg.root / f).exists()]
    try:
        git(cfg, "add", "--", *files, check=True)
        git(cfg, "commit", "-m", f"chore(release): v{new}", check=True, timeout=300)
    except MkError:
        git(cfg, "reset", "-q", "HEAD", "--", *files)
        git(cfg, "restore", "--", *files)
        warn("commit failed; the version bump was reverted (the tree was clean before it)")
        raise
    create_tag(cfg, new)
    info("review it, then publish explicitly with: make release-push CONFIRM=1")
    return 0


def cmd_release_push(cfg: Config) -> int:
    """Push the release commit's branch and its one tag: two plain pushes, never force, never --tags."""
    if cfg.arg("force"):
        raise MkError("force-pushing is not supported by this Makefile")
    require_repo(cfg)
    branch = require_branch(cfg)
    version = project_version(cfg.root)
    tag = f"v{version}"
    if git(cfg, "rev-parse", "-q", "--verify", f"refs/tags/{tag}^{{commit}}")[0] != 0:
        raise MkError(
            f"tag {tag} does not exist locally", "make release BUMP=... (or make release-tag) first"
        )
    if git_out(cfg, "rev-parse", f"{tag}^{{commit}}") != git_out(cfg, "rev-parse", "HEAD"):
        raise MkError(
            f"{tag} does not point at HEAD",
            "Check out the tagged commit's branch; tags are never moved.",
        )
    if tracked_changes(cfg):
        raise MkError("the working tree has uncommitted changes")
    if not remote_url(cfg):
        raise MkError(f"remote `{cfg.remote}` is not configured")
    commands = [
        f"git push {cfg.remote} HEAD:refs/heads/{branch}",
        f"git push {cfg.remote} refs/tags/{tag}",
    ]
    info(f"publishing {tag}: " + "; ".join(commands))
    if cfg.dry_run:
        info("dry-run: nothing pushed")
        return 0
    require_confirmation(
        cfg, f"push {branch} and tag {tag} to {cfg.remote} (publishes the release)"
    )
    if not remote_reachable(cfg):
        raise MkError(f"cannot reach {cfg.remote}")
    if remote_has_tag(cfg, tag):
        raise MkError(f"{tag} already exists on {cfg.remote}; tags are never overwritten")
    fetch(cfg)
    if upstream(cfg):
        ab = ahead_behind(cfg)
        if ab and ab[1]:
            raise MkError(
                f"{branch} is {ab[1]} commit(s) behind {upstream(cfg)}", "make git-sync first."
            )
    for args in (
        ["push", cfg.remote, f"HEAD:refs/heads/{branch}"],
        ["push", cfg.remote, f"refs/tags/{tag}"],
    ):
        rc, _, err = git(cfg, *args, timeout=300)
        if rc != 0:
            raise MkError(
                f"`git {' '.join(args)}` failed: {err.splitlines()[-1] if err else 'unknown error'}",
                "Nothing was forced. If the branch is protected, open a pull request instead.",
            )
    ok(
        f"pushed {branch} and {tag}; create the GitHub release from the tag yourself if you want one"
    )
    return 0


def cmd_release_verify(cfg: Config) -> int:
    require_repo(cfg)
    r = version_report(cfg)
    version = project_version(cfg.root)
    tag = f"v{version}"
    if git(cfg, "rev-parse", "-q", "--verify", f"refs/tags/{tag}^{{commit}}")[0] != 0:
        r.failed("tag", f"{tag} does not exist locally", "make release-tag")
        return r.finish()
    kind = git_out(cfg, "cat-file", "-t", f"refs/tags/{tag}")
    (r.passed if kind == "tag" else r.failed)(
        "tag type", f"{tag} is {'annotated' if kind == 'tag' else 'lightweight'}"
    )
    tagged = git_out(cfg, "show", f"{tag}:pyproject.toml")
    m = re.search(r'(?m)^version\s*=\s*"([^"]+)"', tagged)
    if m and m.group(1) == version:
        r.passed("tagged content", f"{tag} contains pyproject.toml version {version}")
    else:
        r.failed(
            "tagged content",
            f"{tag} contains version {m.group(1) if m else '?'}, expected {version}",
        )
    head = git_out(cfg, "rev-parse", "HEAD")
    if git_out(cfg, "rev-parse", f"{tag}^{{commit}}") == head:
        r.passed("HEAD", f"{tag} points at HEAD")
    else:
        r.warned("HEAD", f"{tag} is not at HEAD (commits since the release)")
    remote = remote_has_tag(cfg, tag)
    if remote is None:
        r.warned("remote", f"could not check {cfg.remote}")
    elif remote:
        r.passed("remote", f"{tag} is on {cfg.remote}")
    else:
        r.warned("remote", f"{tag} has not been pushed", "make release-push CONFIRM=1")
    return r.finish()
