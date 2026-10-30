"""Environment, diagnostics and reproducibility commands for the Makefile helper.

Installation has two paths. ``uv sync --locked`` is the supported one and honours
``uv.lock``. The pip fallback resolves from the bounds in ``pyproject.toml`` only, says so
every time it runs, and never substitutes an interpreter the project does not support.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mk_git
import mk_work
from mk_core import (
    Config,
    MkError,
    Report,
    dependency_group,
    env_ready,
    fmt_version,
    info,
    install_hint,
    load_pyproject,
    ok,
    probe_python,
    project_version,
    python_snippet,
    request_matches,
    requires_python,
    run_host,
    run_tool,
    satisfies,
    select_python,
    supported_minors,
    venv_dir,
    venv_python,
    warn,
)

REQUIRED_PATHS = (
    "pyproject.toml", "README.md", "LICENSE", "Makefile", "scripts/mk.py",
    "src/savings_goal/__init__.py", "tests/unit", "tests/integration", "notebooks",
    "results", ".github/workflows/ci.yml", "project/figures/make_figures.py",
)  # fmt: skip
OPTIONAL_PATHS = ("uv.lock", "CITATION.cff", "dataset/README.md")
# Restricted or large inputs: absent in a fresh clone by design, so absence is a warning.
DATA_EXPECTATIONS = (
    ("raw ICPSR DS0002 TSV", "raw_tsv", "download it (see dataset/README.md), then `make data-pipeline`"),
    ("households table", "data/households.parquet", "make data-prepare"),
    ("feature table", "data/features.parquet", "make data-features"),
    ("out-of-fold scores (notebook 07)", "data/oof.parquet", "make business-analysis"),
    ("HCES level files (notebook 10, optional)", "data/hces", "see dataset/README.md"),
    ("fitted tier models", "models/tier1.joblib", "make train"),
)  # fmt: skip
DEV_TOOLS = ("ruff", "mypy", "pytest", "coverage")
IMPORT_MODULES = ("pytest", "mypy", "nbconvert", "nbformat", "ipykernel", "jupyter_core")

IMPORT_CHECK_CODE = """
import importlib, pkgutil, sys
import savings_goal
failed = []
for m in pkgutil.walk_packages(savings_goal.__path__, "savings_goal."):
    try:
        importlib.import_module(m.name)
    except Exception as exc:
        failed.append(f"{m.name}: {type(exc).__name__}: {exc}")
for line in failed:
    print(line)
sys.exit(1 if failed else 0)
"""

ENV_SNIPPET = """
import json, platform, sys
from importlib import metadata
from savings_goal.manifest import PACKAGES
out = {"python": platform.python_version(), "packages": {}}
for name in (*PACKAGES, "pytest", "ruff", "mypy", "nbconvert"):
    try:
        out["packages"][name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        pass
print(json.dumps(out))
"""


def resolve_mode(cfg: Config, quiet: bool = False) -> str:
    uv_found = shutil.which(cfg.uv) is not None
    if cfg.mode == "uv":
        if not uv_found:
            raise MkError(
                f"MODE=uv but `{cfg.uv}` was not found",
                f"Install uv:  {install_hint('uv')}\nor use MODE=pip (no lockfile guarantees).",
                code=2,
            )
        return "uv"
    if cfg.mode == "pip":
        return "pip"
    if uv_found:
        return "uv"
    if not quiet:
        warn(
            "uv was not found: using the pip fallback, which does NOT honour uv.lock\n"
            f"           Install uv for reproducible environments:  {install_hint('uv')}"
        )
    return "pip"


def extras_list(cfg: Config) -> list[str]:
    return [e for e in re.split(r"[,\s]+", cfg.extras) if e]


# --- install ---------------------------------------------------------------------------


def uv_sync_args(cfg: Config, dev: bool) -> list[str]:
    args = [cfg.uv, "sync", "--locked"]
    if cfg.python_version or cfg.python:
        args += ["--python", cfg.python_version or cfg.python]
    for extra in extras_list(cfg):
        args += ["--extra", extra]
    if not dev:
        args += [
            "--no-dev",
            "--inexact",
        ]  # runtime only, but never uninstall dev tools already there
    return args


def install(cfg: Config, *, dev: bool, force: bool) -> int:
    mode = resolve_mode(cfg)
    if mode == "uv":
        argv = uv_sync_args(cfg, dev)
        env = {"UV_PROJECT_ENVIRONMENT": str(venv_dir(cfg))}
        if cfg.dry_run:
            info(f"dry-run: would run: {' '.join(argv)}")
            return 0
        info(f"{'dev' if dev else 'runtime'} environment via uv (locked): {venv_dir(cfg)}")
        run_host(cfg, argv, tool="uv", env=env)
    else:
        install_pip(cfg, dev)
    ready, reason = env_ready(cfg)
    if not ready and not cfg.dry_run:
        raise MkError(f"install finished but the environment is not usable: {reason}")
    ok(reason)
    return 0


def install_pip(cfg: Config, dev: bool) -> None:
    exe, version = select_python(cfg)
    target = f".[{','.join(extras_list(cfg))}]" if extras_list(cfg) else "."
    reqs = dependency_group(cfg.root, "dev") if dev else []
    py = venv_python(cfg)
    argv = [str(py), "-m", "pip", "install", "-e", target, *reqs]
    if cfg.dry_run:
        info(f"dry-run: would create {venv_dir(cfg)} with {' '.join(exe)} ({fmt_version(version)})")
        info(f"dry-run: would run: {' '.join(argv)}")
        return
    if py.exists():
        ready, reason = env_ready(cfg)
        if not ready:
            raise MkError(
                f"the existing {venv_dir(cfg)} is not usable: {reason}",
                "Delete it yourself (it is never removed automatically) or pick another with VENV=...",
            )
    else:
        info(f"creating {venv_dir(cfg)} with {' '.join(exe)} ({fmt_version(version)})")
        run_host(cfg, [*exe, "-m", "venv", str(venv_dir(cfg))])
    warn(
        "pip fallback: versions resolve from pyproject.toml bounds, NOT uv.lock, so this\n"
        "           environment can differ from CI and from the committed results."
    )
    run_host(cfg, argv, tool="python")


def cmd_install(cfg: Config) -> int:
    return install(cfg, dev=False, force=False)


def cmd_install_dev(cfg: Config) -> int:
    """Idempotent: a no-op (one read-only check) when the environment already matches the lock."""
    if resolve_mode(cfg, quiet=True) == "uv" and not cfg.dry_run:
        ready, _ = env_ready(cfg)
        if ready:
            argv = [*uv_sync_args(cfg, True), "--check"]
            proc = run_host(
                cfg,
                argv,
                tool="uv",
                capture=True,
                check=False,
                env={"UV_PROJECT_ENVIRONMENT": str(venv_dir(cfg))},
            )
            if proc.returncode == 0:
                ok(f"environment already matches uv.lock ({venv_dir(cfg)})")
                return 0
    return install(cfg, dev=True, force=False)


def cmd_sync(cfg: Config) -> int:
    return install(cfg, dev=True, force=True)


def cmd_ensure_env(cfg: Config) -> int:
    """Read-only gate before every tool target: refuse to run in a missing or stale environment."""
    ready, reason = env_ready(cfg)
    if not ready:
        raise MkError(
            f"the project environment is not ready: {reason}", "Run:  make install-dev", code=2
        )
    if resolve_mode(cfg, quiet=True) == "uv":
        argv = [*uv_sync_args(cfg, True), "--check"]
        proc = run_host(
            cfg,
            argv,
            tool="uv",
            capture=True,
            check=False,
            env={"UV_PROJECT_ENVIRONMENT": str(venv_dir(cfg))},
        )
        if proc.returncode != 0:
            # Usually "outdated", but uv can fail for other reasons (no matching interpreter, ...):
            # pass its own explanation through rather than guess.
            said = [
                ln.strip()
                for ln in (proc.stderr or "").splitlines()
                if ln.strip().startswith(("error", "hint"))
            ][:3]
            raise MkError(
                "the environment is not in sync with pyproject.toml / uv.lock",
                "Run:  make install-dev" + "".join(f"\nuv: {ln}" for ln in said),
                code=2,
            )
    return 0


def cmd_lock_check(cfg: Config) -> int:
    if resolve_mode(cfg, quiet=True) != "uv":
        warn("lock consistency NOT verified: uv is not in use (the pip fallback has no lockfile)")
        return 0
    run_host(cfg, [cfg.uv, "lock", "--check"], tool="uv")
    ok("uv.lock is consistent with pyproject.toml")
    return 0


def cmd_import_check(cfg: Config) -> int:
    proc = run_tool(cfg, ["python", "-c", IMPORT_CHECK_CODE], capture=True, check=False)
    if proc.returncode != 0:
        raise MkError(
            "some modules fail to import:\n  " + proc.stdout.strip().replace("\n", "\n  ")
        )
    ok("every savings_goal module imports")
    return 0


# --- reports ---------------------------------------------------------------------------


def tools_report(cfg: Config) -> Report:
    r = Report("tools")
    r.passed("make", os.environ.get("MAKE_COMMAND", "GNU Make (this is running under it)"))
    uv_path = shutil.which(cfg.uv)
    if uv_path:
        out = run_host(
            cfg, [cfg.uv, "--version"], tool="uv", capture=True, check=False
        ).stdout.strip()
        r.passed("uv", f"{out} ({uv_path})")
    else:
        r.warned(
            "uv",
            "not found; the pip fallback has no lockfile guarantees",
            f"Install: {install_hint('uv')}",
        )
    if shutil.which("git"):
        r.passed(
            "git", run_host(cfg, ["git", "--version"], capture=True, check=False).stdout.strip()
        )
    else:
        r.failed("git", "not found", f"Install: {install_hint('git')}")
    try:
        exe, version = select_python(cfg)
        r.passed(
            "python",
            f"{' '.join(exe)} {fmt_version(version)} satisfies requires-python {requires_python(cfg.root)}",
        )
    except MkError as exc:
        r.failed("python", str(exc), exc.hint)
    for tool, why in (("pre-commit", "make precommit"), ("latexmk", "make report")):
        path = shutil.which(tool)
        if path:
            r.passed(tool, path)
        else:
            r.warned(tool, f"not found (needed only for `{why}`)", f"Install: {install_hint(tool)}")
    return r


def python_report(cfg: Config) -> Report:
    r = Report("python and environment")
    spec = requires_python(cfg.root)
    r.passed(
        "requires-python", f"{spec}  ->  supported minors: {', '.join(supported_minors(spec))}"
    )
    if cfg.python_version:
        want = tuple(int(x) for x in cfg.python_version.split(".") if x.isdigit())
        minors = [
            m
            for m in supported_minors(spec)
            if request_matches(tuple(map(int, m.split("."))), cfg.python_version)
        ]
        if want and minors:
            r.passed("PYTHON_VERSION", f"{cfg.python_version} is allowed")
        else:
            r.failed("PYTHON_VERSION", f"{cfg.python_version} is outside requires-python {spec}")
    ready, reason = env_ready(cfg)
    if ready:
        r.passed("environment", reason)
    else:
        r.failed("environment", reason, "Run:  make install-dev")
        return r
    code = (
        "import importlib.util, json, shutil\n"
        f"mods = {list(IMPORT_MODULES)!r}\n"
        f"tools = {list(DEV_TOOLS)!r}\n"
        "print(json.dumps({'mods': {m: importlib.util.find_spec(m) is not None for m in mods},"
        " 'tools': {t: shutil.which(t) is not None for t in tools}}))"
    )
    try:
        found = json.loads(python_snippet(cfg, code))
    except (MkError, json.JSONDecodeError) as exc:
        r.failed("dev dependencies", str(exc), "Run:  make install-dev")
        return r
    missing = [m for m, there in found["mods"].items() if not there] + [
        t for t, there in found["tools"].items() if not there
    ]
    if missing:
        r.failed("dev dependencies", "missing: " + ", ".join(missing), "Run:  make install-dev")
    else:
        r.passed(
            "dev dependencies",
            "ruff, mypy, pytest, coverage, nbconvert, nbformat, ipykernel present",
        )
    return r


def lock_report(cfg: Config) -> Report:
    r = Report("lock")
    if not (cfg.root / "uv.lock").exists():
        r.warned("uv.lock", "missing", "Run: uv lock")
    elif resolve_mode(cfg, quiet=True) != "uv":
        r.warned("uv.lock", "present but unverified: uv is not in use")
    else:
        proc = run_host(cfg, [cfg.uv, "lock", "--check"], tool="uv", capture=True, check=False)
        if proc.returncode == 0:
            r.passed("uv.lock", "consistent with pyproject.toml")
        else:
            r.failed("uv.lock", "out of date with pyproject.toml", "Run: uv lock")
    return r


def structure_report(cfg: Config) -> Report:
    r = Report("project structure")
    missing = [p for p in REQUIRED_PATHS if not cfg.path(p).exists()]
    if missing:
        r.failed("required paths", "missing: " + ", ".join(missing))
    else:
        r.passed("required paths", f"{len(REQUIRED_PATHS)} present")
    for p in OPTIONAL_PATHS:
        if not cfg.path(p).exists():
            r.warned(p, "not present")
    return r


def expectation_path(cfg: Config, key: str) -> Path:
    if key == "raw_tsv":
        return cfg.path(cfg.raw_tsv)
    head, _, tail = key.partition("/")
    return cfg.dir(head) / tail


def data_report(cfg: Config) -> Report:
    r = Report("data and model artefacts")
    for label, key, hint in DATA_EXPECTATIONS:
        path = expectation_path(cfg, key)
        if path.exists():
            r.passed(
                label,
                str(path.relative_to(cfg.root)) if path.is_relative_to(cfg.root) else str(path),
            )
        else:
            r.warned(label, f"not found at {path.name}", hint)
    return r


def tests_report(cfg: Config) -> Report:
    r = Report("tests and quality tooling")
    tool = load_pyproject(cfg.root).get("tool", {})
    for section in ("pytest", "ruff", "mypy"):
        if section in tool:
            r.passed(f"[tool.{section}]", "configured in pyproject.toml")
        else:
            r.failed(f"[tool.{section}]", "missing from pyproject.toml")
    for d in ("tests/unit", "tests/integration"):
        n = len(list(cfg.path(d).glob("test_*.py"))) if cfg.path(d).exists() else 0
        (r.passed if n else r.failed)(d, f"{n} test module(s)")
    if cfg.path(".pre-commit-config.yaml").exists():
        r.passed("pre-commit config", ".pre-commit-config.yaml")
    else:
        r.warned("pre-commit config", "no .pre-commit-config.yaml")
    return r


def outputs_report(cfg: Config) -> Report:
    r = Report("output directories")
    for key in ("build", "results", "models", "figures", "report", "nb_run"):
        d = cfg.dir(key)
        probe = d
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        if os.access(probe, os.W_OK):
            r.passed(
                f"{key} dir",
                f"{d.relative_to(cfg.root) if d.is_relative_to(cfg.root) else d} writable",
            )
        else:
            r.failed(f"{key} dir", f"{d} is not writable (nearest existing parent: {probe})")
    return r


def imports_report(cfg: Config) -> Report:
    r = Report("importability")
    ready, reason = env_ready(cfg)
    if not ready:
        r.failed("import savings_goal", reason, "Run:  make install-dev")
        return r
    proc = run_tool(cfg, ["python", "-c", IMPORT_CHECK_CODE], capture=True, check=False)
    if proc.returncode == 0:
        r.passed("import savings_goal", "every submodule imports")
    else:
        r.failed("import savings_goal", proc.stdout.strip().replace("\n", "; "))
    return r


def cmd_check_tools(cfg: Config) -> int:
    return tools_report(cfg).finish()


def cmd_python_info(cfg: Config) -> int:
    spec = requires_python(cfg.root)
    print(f"requires-python:   {spec}  (pyproject.toml)")
    print(f"supported minors:  {', '.join(supported_minors(spec))}")
    print(
        f"PYTHON_VERSION:    {cfg.python_version or '(unset: uv / the first compatible interpreter decides)'}"
    )
    print(f"mode:              {resolve_mode(cfg, quiet=True)}  (MODE={cfg.mode})")
    print(f"helper runs under: {sys.executable} {platform.python_version()}")
    try:
        exe, version = select_python(cfg)
        print(f"selected:          {' '.join(exe)} -> {fmt_version(version)}")
    except MkError as exc:
        print(f"selected:          none ({exc})")
    py = venv_python(cfg)
    venv_version = probe_python([str(py)]) if py.exists() else None
    print(
        f"project venv:      {venv_dir(cfg)} -> "
        + (fmt_version(venv_version) if venv_version else "(not created)")
    )
    if venv_version and not satisfies(venv_version, spec):
        print("                   WARNING: this interpreter does not satisfy requires-python")
    return 0


def cmd_check_env(cfg: Config) -> int:
    r = python_report(cfg)
    r.extend(lock_report(cfg))
    return r.finish()


def cmd_doctor(cfg: Config) -> int:
    r = Report("doctor")
    for part in (
        tools_report(cfg),
        python_report(cfg),
        lock_report(cfg),
        mk_git.git_report(cfg, detached_is_failure=False),
        structure_report(cfg),
    ):
        r.extend(part)
    return r.finish(strict=bool(cfg.arg("strict")))


def health_report(cfg: Config) -> Report:
    r = Report("health")
    for part in (
        structure_report(cfg),
        imports_report(cfg),
        tests_report(cfg),
        mk_work.notebook_report(cfg),
        mk_work.docs_report(cfg),
        outputs_report(cfg),
        data_report(cfg),
    ):
        r.extend(part)
    return r


def cmd_health(cfg: Config) -> int:
    return health_report(cfg).finish(strict=bool(cfg.arg("strict")))


# --- audit -----------------------------------------------------------------------------


def tracked_files(cfg: Config) -> list[str]:
    return mk_git.git_out(cfg, "ls-files", "-z").split("\0") if mk_git.in_repo(cfg) else []


def audit_report(cfg: Config) -> Report:
    r = Report("audit")
    files = [f for f in tracked_files(cfg) if f]
    problems = [p for f in files for p in mk_git.scan_paths(cfg, [f])]
    if problems:
        r.failed("tracked files", f"{len(problems)} problem(s)", "\n".join(problems[:10]))
    else:
        r.passed("tracked files", f"{len(files)} files: no secrets, credentials or restricted data")
    big = [f for f in files if cfg.path(f).exists() and cfg.path(f).stat().st_size > 5_000_000]
    if big:
        r.warned(
            "large files", ", ".join(big), "Over 5 MB: confirm they belong in version control."
        )
    else:
        r.passed("large files", "no tracked file over 5 MB")
    leaks = [
        p for p in ("dataset/households.parquet", "dataset/ICPSR_36151/x.tsv", "artifacts/tier1.joblib", ".env")
        if mk_git.git(cfg, "check-ignore", "-q", "--", p)[0] != 0
    ]  # fmt: skip
    if leaks:
        r.failed(".gitignore", "does not ignore: " + ", ".join(leaks))
    else:
        r.passed(".gitignore", "restricted data, models and .env are ignored")
    if resolve_mode(cfg, quiet=True) != "uv":
        r.warned(
            "dependency vulnerabilities",
            "skipped: needs uv (not installed, or MODE=pip)",
            f"Install: {install_hint('uv')}",
        )
        return r
    if cfg.offline:
        r.warned("dependency vulnerabilities", "skipped (OFFLINE=1)")
        return r
    proc = run_host(
        cfg, [cfg.uv, "audit", "--locked", "--preview-features", "audit-command"], tool="uv",
        capture=True, check=False, timeout=180,
    )  # fmt: skip
    text = (proc.stdout + proc.stderr).strip()
    last = text.splitlines()[-1] if text else ""
    if proc.returncode == 0:
        r.passed("dependency vulnerabilities", last)
    elif re.search(r"vulnerab|advisor|GHSA|CVE", text, re.I) and "Found no" not in text:
        r.failed("dependency vulnerabilities", last, text[-800:])
    else:
        r.warned("dependency vulnerabilities", f"could not run `uv audit`: {last}")
    return r


def cmd_audit(cfg: Config) -> int:
    return audit_report(cfg).finish()


def cmd_verify(cfg: Config) -> int:
    """Everything read-only and non-test: health, audit, lock and version consistency."""
    r = Report("verify")
    for part in (lock_report(cfg), mk_git.version_report(cfg), audit_report(cfg)):
        r.extend(part)
    r.extend(health_report(cfg))
    return r.finish(strict=bool(cfg.arg("strict")))


# --- reproducibility -------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def current_environment(cfg: Config) -> dict[str, Any]:
    env_info: dict[str, Any] = json.loads(python_snippet(cfg, ENV_SNIPPET))
    lock = cfg.root / "uv.lock"
    commit = mk_git.git_out(cfg, "rev-parse", "HEAD") if mk_git.in_repo(cfg) else ""
    uv_version = ""
    if shutil.which(cfg.uv):
        uv_version = run_host(
            cfg, [cfg.uv, "--version"], tool="uv", capture=True, check=False
        ).stdout.strip()
    return {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "project_version": project_version(cfg.root),
        "git_commit": commit or None,
        "git_dirty": bool(mk_git.status_lines(cfg)) if commit else None,
        "python": env_info["python"],
        "platform": platform.platform(),
        "packages": env_info["packages"],
        "uv": uv_version or None,
        "uv_lock_sha256": sha256_file(lock) if lock.exists() else None,
        "seed": cfg.seed,
        "jobs": cfg.jobs or None,
        "ci": cfg.ci,
        "install_mode": resolve_mode(cfg, quiet=True),
    }


def cmd_environment_report(cfg: Config) -> int:
    report = current_environment(cfg)
    text = json.dumps(report, indent=2)
    print(text)
    out = cfg.dir("build") / "repro" / "environment.json"
    if cfg.dry_run:
        info(f"dry-run: would write {out}")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8", newline="\n")
    info(f"wrote {out}")
    if report["install_mode"] == "pip":
        warn("installed via the pip fallback: package versions are not pinned by uv.lock")
    return 0


LITERAL_SEED = re.compile(
    r"(?:random_state|seed)\s*[=:]\s*\d+|default_rng\(\s*\d+\s*\)|\.seed\(\s*\d+\s*\)"
)


def seed_report(cfg: Config) -> Report:
    r = Report(f"seed check (SEED={cfg.seed})")
    ready, reason = env_ready(cfg)
    if ready:
        got = python_snippet(cfg, "from savings_goal.config import SEED; print(SEED)").strip()
        if got == str(cfg.seed):
            r.passed(
                "effective seed",
                f"savings_goal.config.SEED == {got} when SGC_SEED={cfg.seed} is exported",
            )
        else:
            r.failed(
                "effective seed",
                f"package reports {got}, expected {cfg.seed}",
                "Is SGC_SEED honoured in config.py?",
            )
    else:
        r.warned("effective seed", f"not checked: {reason}", "Run: make install-dev")
    literals: list[str] = []
    for base in ("src", "notebooks"):
        for path in sorted(cfg.path(base).rglob("*.py" if base == "src" else "*.ipynb")):
            if path.name == "config.py":
                continue
            text = path.read_text(encoding="utf-8")
            if path.suffix == ".ipynb":
                text = "\n".join(
                    "".join(c["source"])
                    for c in json.loads(text)["cells"]
                    if c["cell_type"] == "code"
                )
            for n, line in enumerate(text.splitlines(), 1):
                if LITERAL_SEED.search(line):
                    literals.append(
                        f"{path.relative_to(cfg.root).as_posix()}:{n}: {line.strip()[:80]}"
                    )
    if literals:
        r.warned(
            "literal seeds",
            f"{len(literals)} hard-coded seed(s) ignore SEED",
            "\n".join(literals[:8]),
        )
    else:
        r.passed("literal seeds", "none outside config.py")
    return r


def cmd_seed_check(cfg: Config) -> int:
    return seed_report(cfg).finish()


def cmd_experiment_info(cfg: Config) -> int:
    print("== recorded experiment metadata ==")
    shown = False
    for label, path in (
        ("last training run", cfg.dir("models") / "train_manifest.json"),
        ("last full reproduction (committed)", cfg.dir("results") / "run_manifest.json"),
        ("last notebook run", cfg.dir("nb_run") / "status.json"),
    ):
        if not path.exists():
            print(f"{label}: (none at {path.name})")
            continue
        shown = True
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"{label}: {path}")
        if "packages" in data:
            print(
                f"  commit {data.get('git_commit')}, python {data.get('python')}, seed {data.get('seed')}"
            )
            print("  " + ", ".join(f"{k} {v}" for k, v in data["packages"].items()))
            if data.get("input"):
                print(f"  input {data['input']['file']}  sha256 {data['input']['sha256']}")
        else:
            print("  " + json.dumps(data)[:400])
    ready, _ = env_ready(cfg)
    manifest = cfg.dir("results") / "run_manifest.json"
    if ready and manifest.exists():
        now = current_environment(cfg)
        ref = json.loads(manifest.read_text(encoding="utf-8"))
        drift = (
            [f"python {ref.get('python')} -> {now['python']}"]
            if ref.get("python") != now["python"]
            else []
        )
        drift += [
            f"{k} {v} -> {now['packages'].get(k)}"
            for k, v in ref.get("packages", {}).items()
            if now["packages"].get(k) != v
        ]
        if drift:
            warn(
                "this environment differs from the committed reference run; results can move in the 3rd-4th decimal:\n           "
                + "\n           ".join(drift)
            )
        else:
            ok("this environment matches the committed reference run")
    if not shown:
        info("nothing recorded yet: run `make train` or `make notebooks-run`")
    return 0


def cmd_reproducibility(cfg: Config) -> int:
    code = cmd_seed_check(cfg)
    cmd_environment_report(cfg)
    cmd_experiment_info(cfg)
    info(
        "Reproducibility is bounded: the IHDS-II data cannot be redistributed, results can shift in the\n"
        "           3rd-4th decimal across xgboost / Python versions (README, 'Numerical sensitivity'), and\n"
        "           only an environment synced from uv.lock is pinned."
    )
    return code
