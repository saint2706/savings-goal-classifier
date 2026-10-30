"""Shared plumbing for the Makefile helper (``scripts/mk.py``).

Standard library only: the helper runs before the project environment exists, so it
cannot import anything the project installs. Configuration arrives from the Makefile as
``MK_*`` environment variables (see ``Config.from_env``), which sidesteps every
shell-quoting difference between sh and cmd.exe.
"""

from __future__ import annotations

import contextlib
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TRUTHY = frozenset({"1", "true", "yes", "y", "on"})

# Defaults mirror the Makefile's configuration section; the Makefile always passes its own.
DEFAULT_DIRS: dict[str, str] = {
    "build": "build",
    "data": "dataset",
    "results": "results",
    "models": "artifacts",
    "figures": "build/figures",
    "report": "build/report",
    "nb_run": "build/notebooks",
    "nb_export": "build/notebooks/export",
    "demo": "build/demo",
}
DEFAULT_RAW_TSV = "dataset/ICPSR_36151/DS0002/36151-0002-Data.tsv"


class MkError(Exception):
    """An expected failure: reported as ``[mk:error]`` plus a hint, never as a traceback."""

    def __init__(self, message: str, hint: str = "", code: int = 1) -> None:
        super().__init__(message)
        self.hint = hint
        self.code = code


def truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in TRUTHY


@dataclass(frozen=True)
class Config:
    root: Path
    mode: str = "auto"
    python: str = ""
    uv: str = "uv"
    python_version: str = ""
    venv: str = ".venv"
    extras: str = ""
    verbosity: int = 1
    dry_run: bool = False
    confirm: bool = False
    ci: bool = False
    seed: int = 42
    cov_min: int = 95
    nb_timeout: int = 6000
    jobs: str = ""
    remote: str = "origin"
    offline: bool = False
    raw_tsv: str = DEFAULT_RAW_TSV
    dirs: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_DIRS))
    args: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Config:
        e = os.environ if env is None else env

        def get(name: str, default: str = "") -> str:
            return (e.get(f"MK_{name}") or default).strip()

        root = Path(get("ROOT") or Path(__file__).resolve().parents[1]).resolve()
        try:
            seed = int(get("SEED", "42"))
            cov_min = int(get("COV_MIN", "95"))
            timeout = int(get("NB_TIMEOUT", "6000"))
            verbosity = int(get("V", "1"))
        except ValueError as exc:
            raise MkError(f"a numeric setting is not a number: {exc}") from exc
        mode = get("MODE", "auto").lower()
        if mode not in {"auto", "uv", "pip"}:
            raise MkError(f"MODE must be auto, uv or pip (got {mode!r})")
        dirs = {k: get(k.upper() + "_DIR", v) for k, v in DEFAULT_DIRS.items()}
        args = {k[len("MK_ARG_") :].lower(): v for k, v in e.items() if k.startswith("MK_ARG_")}
        return cls(
            root=root,
            mode=mode,
            python=get("PYTHON"),
            uv=get("UV", "uv"),
            python_version=get("PYTHON_VERSION"),
            venv=get("VENV", ".venv"),
            extras=get("EXTRAS"),
            verbosity=verbosity,
            dry_run=truthy(get("DRY_RUN")),
            confirm=truthy(get("CONFIRM")),
            ci=truthy(e.get("CI")),
            seed=seed,
            cov_min=cov_min,
            nb_timeout=timeout,
            jobs=get("JOBS"),
            remote=get("REMOTE", "origin"),
            offline=truthy(get("OFFLINE")),
            raw_tsv=get("RAW_TSV", DEFAULT_RAW_TSV),
            dirs=dirs,
            args=args,
        )

    def path(self, rel: str | Path) -> Path:
        """Absolute path for a repo-relative (or absolute) setting."""
        return (self.root / rel).resolve()

    def dir(self, key: str) -> Path:
        return self.path(self.dirs[key])

    def arg(self, name: str, default: str = "") -> str:
        return self.args.get(name, default).strip()


# --- logging ---------------------------------------------------------------------------

_verbosity = 1


def set_verbosity(level: int) -> None:
    global _verbosity
    _verbosity = level


def info(msg: str) -> None:
    if _verbosity >= 1:
        print(f"[mk] {msg}", flush=True)


def ok(msg: str) -> None:
    if _verbosity >= 1:
        print(f"[mk:ok] {msg}", flush=True)


def debug(msg: str) -> None:
    if _verbosity >= 2:
        print(f"[mk:debug] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[mk:warn] {msg}", file=sys.stderr, flush=True)


def error(msg: str, hint: str = "") -> None:
    print(f"[mk:error] {msg}", file=sys.stderr, flush=True)
    if hint:
        for line in hint.splitlines():
            print(f"           {line}", file=sys.stderr, flush=True)


# --- diagnostics report ----------------------------------------------------------------


@dataclass
class Report:
    """PASS / WARN / FAIL lines for the diagnostic targets; FAIL (or WARN, if strict) exits 1."""

    title: str
    rows: list[tuple[str, str, str, str]] = field(default_factory=list)

    def add(self, status: str, name: str, detail: str = "", hint: str = "") -> None:
        self.rows.append((status, name, detail, hint))

    def passed(self, name: str, detail: str = "") -> None:
        self.add("PASS", name, detail)

    def warned(self, name: str, detail: str = "", hint: str = "") -> None:
        self.add("WARN", name, detail, hint)

    def failed(self, name: str, detail: str = "", hint: str = "") -> None:
        self.add("FAIL", name, detail, hint)

    def extend(self, other: Report) -> None:
        self.rows.extend(other.rows)

    def count(self, status: str) -> int:
        return sum(1 for r in self.rows if r[0] == status)

    def render(self) -> str:
        lines = [f"== {self.title} =="]
        for status, name, detail, hint in self.rows:
            lines.append(f"{status:<5} {name}" + (f": {detail}" if detail else ""))
            if hint and status != "PASS":
                lines.extend(f"        {h}" for h in hint.splitlines())
        lines.append(
            f"-- {self.count('PASS')} passed, {self.count('WARN')} warning(s), "
            f"{self.count('FAIL')} failure(s)"
        )
        return "\n".join(lines)

    def finish(self, strict: bool = False) -> int:
        print(self.render(), flush=True)
        bad = self.count("FAIL") + (self.count("WARN") if strict else 0)
        return 1 if bad else 0


# --- platform and install hints --------------------------------------------------------


def platform_key() -> str:
    return {"Windows": "windows", "Darwin": "macos"}.get(platform.system(), "linux")


INSTALL_HINTS: dict[str, dict[str, str]] = {
    "uv": {
        "windows": 'winget install --id=astral-sh.uv -e   (or: powershell -c "irm https://astral.sh/uv/install.ps1 | iex")',
        "macos": "brew install uv   (or: curl -LsSf https://astral.sh/uv/install.sh | sh)",
        "linux": "curl -LsSf https://astral.sh/uv/install.sh | sh   (or: pipx install uv)",
    },
    "python": {
        "windows": "uv python install 3.13   (or: winget install Python.Python.3.13)",
        "macos": "uv python install 3.13   (or: brew install python@3.13)",
        "linux": "uv python install 3.13   (or: sudo apt install python3.13 python3.13-venv)",
    },
    "git": {
        "windows": "winget install --id Git.Git -e",
        "macos": "xcode-select --install   (or: brew install git)",
        "linux": "sudo apt install git",
    },
    "make": {
        "windows": "scoop install make   (or: choco install make, or: winget install ezwinports.make)",
        "macos": "xcode-select --install",
        "linux": "sudo apt install make",
    },
    "pre-commit": {
        "windows": "uv tool install pre-commit",
        "macos": "uv tool install pre-commit   (or: brew install pre-commit)",
        "linux": "uv tool install pre-commit",
    },
    "latexmk": {
        "windows": "Install MiKTeX (https://miktex.org/download); it ships latexmk and needs Perl",
        "macos": "brew install --cask mactex-no-gui",
        "linux": "sudo apt install latexmk texlive-latex-extra texlive-publishers",
    },
}


def install_hint(tool: str) -> str:
    hints = INSTALL_HINTS.get(tool, {})
    return hints.get(platform_key(), "")


# --- pyproject.toml --------------------------------------------------------------------


def load_pyproject(root: Path) -> dict[str, Any]:
    path = root / "pyproject.toml"
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise MkError(f"{path} not found", "Run make from the repository root.") from exc
    except tomllib.TOMLDecodeError as exc:
        raise MkError(f"{path} is not valid TOML: {exc}") from exc


def requires_python(root: Path) -> str:
    spec = load_pyproject(root).get("project", {}).get("requires-python")
    if not spec:
        raise MkError("pyproject.toml has no [project].requires-python")
    return str(spec)


def project_version(root: Path) -> str:
    version = load_pyproject(root).get("project", {}).get("version")
    if not version:
        raise MkError("pyproject.toml has no [project].version")
    return str(version)


def dependency_group(root: Path, group: str = "dev") -> list[str]:
    """Requirement strings of a PEP 735 group, with ``include-group`` entries expanded."""
    groups: dict[str, list[Any]] = load_pyproject(root).get("dependency-groups", {})

    def expand(name: str, seen: tuple[str, ...]) -> list[str]:
        if name in seen:
            raise MkError(f"dependency group {name!r} includes itself")
        if name not in groups:
            raise MkError(f"pyproject.toml has no [dependency-groups].{name}")
        out: list[str] = []
        for item in groups[name]:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict) and "include-group" in item:
                out.extend(expand(str(item["include-group"]), (*seen, name)))
        return out

    return expand(group, ())


# --- Python version specifiers (PEP 440 subset, enough for requires-python) -------------

Clause = tuple[str, tuple[int, ...], bool]
_SPEC_RE = re.compile(r"^(~=|==|!=|<=|>=|<|>)\s*(\d+(?:\.\d+)*)(\.\*)?$")


def parse_specifier(spec: str) -> list[Clause]:
    clauses: list[Clause] = []
    for part in spec.split(","):
        m = _SPEC_RE.match(part.strip())
        if not m:
            raise MkError(f"unsupported version specifier {part.strip()!r} in {spec!r}")
        clauses.append((m.group(1), tuple(int(x) for x in m.group(2).split(".")), bool(m.group(3))))
    return clauses


def _cmp(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    n = max(len(a), len(b))
    a, b = a + (0,) * (n - len(a)), b + (0,) * (n - len(b))
    return (a > b) - (a < b)


def satisfies(version: tuple[int, ...], spec: str) -> bool:
    for op, ref, wild in parse_specifier(spec):
        c = _cmp(version, ref)
        if op in {"==", "!="}:
            hit = version[: len(ref)] == ref if wild else c == 0
            if hit == (op == "!="):
                return False
        elif op == "~=":
            prefix = ref[:-1] or ref
            if c < 0 or version[: len(prefix)] != prefix:
                return False
        elif not {">=": c >= 0, "<=": c <= 0, ">": c > 0, "<": c < 0}[op]:
            return False
    return True


def supported_minors(spec: str, major: int = 3, lo: int = 6, hi: int = 30) -> list[str]:
    """Minor versions (e.g. ``3.11``) that some patch release of satisfies ``spec``."""
    return [
        f"{major}.{m}"
        for m in range(lo, hi)
        if satisfies((major, m, 0), spec) or satisfies((major, m, 999), spec)
    ]


def fmt_version(v: Sequence[int]) -> str:
    return ".".join(str(x) for x in v)


# --- interpreters ----------------------------------------------------------------------


def probe_python(exe: Sequence[str]) -> tuple[int, int, int] | None:
    try:
        out = subprocess.run(
            [*exe, "-c", "import sys; print(*sys.version_info[:3])"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    parts = out.stdout.split()
    if out.returncode != 0 or len(parts) != 3 or not all(p.isdigit() for p in parts):
        return None
    return int(parts[0]), int(parts[1]), int(parts[2])


def request_matches(version: tuple[int, ...], request: str) -> bool:
    """True when ``version`` starts with the requested ``3.13``-style prefix (or no request)."""
    if not request:
        return True
    want = tuple(int(x) for x in request.split(".") if x.isdigit())
    return version[: len(want)] == want


def candidate_pythons(cfg: Config, spec: str) -> list[list[str]]:
    if cfg.python:
        return [[cfg.python]]
    minors = [
        m
        for m in supported_minors(spec)
        if request_matches(tuple(map(int, m.split("."))), cfg.python_version)
    ]
    names = [[f"python{m}"] for m in reversed(minors)]
    if os.name == "nt":
        names += [["py", f"-{m}"] for m in reversed(minors)]
    return [*names, ["python3"], ["python"]]


def select_python(cfg: Config) -> tuple[list[str], tuple[int, int, int]]:
    """The first interpreter satisfying requires-python (and PYTHON_VERSION); never a lesser one."""
    spec = requires_python(cfg.root)
    tried: list[str] = []
    for exe in candidate_pythons(cfg, spec):
        if shutil.which(exe[0]) is None and not Path(exe[0]).exists():
            continue
        version = probe_python(exe)
        label = " ".join(exe) + (f" ({fmt_version(version)})" if version else " (unusable)")
        tried.append(label)
        if version and satisfies(version, spec) and request_matches(version, cfg.python_version):
            return exe, version
    wanted = f"requires-python {spec}" + (
        f" and PYTHON_VERSION={cfg.python_version}" if cfg.python_version else ""
    )
    raise MkError(
        f"no interpreter satisfies {wanted}"
        + (f"; found: {', '.join(tried)}" if tried else "; none found on PATH"),
        "Install a supported Python, e.g.:\n"
        f"  {install_hint('python')}\n"
        "or point PYTHON at one:  make <target> PYTHON=/path/to/python",
    )


# --- the project environment -----------------------------------------------------------


def venv_dir(cfg: Config) -> Path:
    return cfg.path(cfg.venv)


def venv_bin(cfg: Config) -> Path:
    return venv_dir(cfg) / ("Scripts" if os.name == "nt" else "bin")


def venv_python(cfg: Config) -> Path:
    return venv_bin(cfg) / ("python.exe" if os.name == "nt" else "python")


def tool_env(cfg: Config, extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment of an activated venv plus the SGC_* overrides the package honours."""
    env = dict(os.environ)
    env.pop("PYTHONHOME", None)
    env["PATH"] = str(venv_bin(cfg)) + os.pathsep + env.get("PATH", "")
    env["VIRTUAL_ENV"] = str(venv_dir(cfg))
    env["SGC_SEED"] = str(cfg.seed)
    env["SGC_DATA_DIR"] = str(cfg.dir("data"))
    env["SGC_RESULTS_DIR"] = str(cfg.dir("results"))
    env["PYTHONUTF8"] = "1"
    if cfg.jobs.isdigit() and int(cfg.jobs) > 0:
        for var in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "LOKY_MAX_CPU_COUNT",
        ):
            env[var] = cfg.jobs
    env.update(extra or {})
    return env


def env_ready(cfg: Config) -> tuple[bool, str]:
    """(ready, reason): the venv exists, runs, and holds a Python the project supports."""
    py = venv_python(cfg)
    if not py.exists():
        return False, f"{venv_dir(cfg)} does not exist"
    version = probe_python([str(py)])
    if version is None:
        return False, f"{py} does not run"
    spec = requires_python(cfg.root)
    if not satisfies(version, spec):
        return False, f"venv Python {fmt_version(version)} does not satisfy requires-python {spec}"
    if not request_matches(version, cfg.python_version):
        return (
            False,
            f"venv Python {fmt_version(version)} is not PYTHON_VERSION={cfg.python_version}",
        )
    return True, f"Python {fmt_version(version)} in {venv_dir(cfg)}"


def require_env(cfg: Config) -> None:
    ready, reason = env_ready(cfg)
    if not ready:
        raise MkError(
            f"the project environment is not ready: {reason}", "Run:  make install-dev", code=2
        )


def run_tool(
    cfg: Config,
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run ``argv`` with the project venv first on PATH; ``DRY_RUN=1`` only prints it."""
    shown = " ".join(argv)
    if cfg.dry_run:
        info(f"dry-run: would run: {shown}")
        return subprocess.CompletedProcess(list(argv), 0, "", "")
    require_env(cfg)
    exe = shutil.which(argv[0], path=str(venv_bin(cfg)))
    if exe is None:
        raise MkError(
            f"`{argv[0]}` is not installed in {venv_dir(cfg)}",
            "Run:  make install-dev   (the project's dev dependency group provides it)",
            code=2,
        )
    debug(f"run: {shown}")
    proc = subprocess.run(
        [exe, *argv[1:]],
        cwd=cwd or cfg.root,
        env=tool_env(cfg, env),
        capture_output=capture,
        text=True,
        check=False,
    )
    if check and proc.returncode != 0:
        raise MkError(f"`{argv[0]}` failed (exit {proc.returncode})", code=proc.returncode or 1)
    return proc


def run_host(
    cfg: Config,
    argv: Sequence[str],
    *,
    tool: str | None = None,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    capture: bool = False,
    check: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a program from the host PATH (git, uv, latexmk, ...) with a useful missing-tool error."""
    name = tool or argv[0]
    exe = shutil.which(argv[0])
    if exe is None:
        raise MkError(
            f"`{argv[0]}` was not found on PATH", f"Install it:  {install_hint(name)}", code=2
        )
    debug(f"run: {' '.join(argv)}")
    try:
        proc = subprocess.run(
            [exe, *argv[1:]],
            cwd=cwd or cfg.root,
            env={**os.environ, **(env or {})},
            capture_output=capture,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise MkError(f"`{' '.join(argv)}` timed out after {timeout}s") from exc
    if check and proc.returncode != 0:
        raise MkError(
            f"`{' '.join(argv)}` failed (exit {proc.returncode})", code=proc.returncode or 1
        )
    return proc


def python_snippet(cfg: Config, code: str, *args: str, env: Mapping[str, str] | None = None) -> str:
    """Run ``python -c code`` in the project venv and return stdout (raises on failure)."""
    proc = run_tool(cfg, ["python", "-c", code, *args], env=env, capture=True, check=False)
    if proc.returncode != 0:
        raise MkError(f"project Python snippet failed: {(proc.stderr or '').strip()[-400:]}")
    return proc.stdout


# --- safety: confirmation, locks, path checks ------------------------------------------


def require_confirmation(cfg: Config, what: str) -> None:
    """Gate a destructive or publishing action: CONFIRM=1, or an interactive ``yes``; never in CI."""
    if cfg.ci:
        raise MkError(
            f"refusing to {what}: CI is set and destructive targets never run unattended",
            "Run this locally, not from CI.",
        )
    if cfg.confirm:
        return
    if sys.stdin is not None and sys.stdin.isatty():
        try:
            answer: str | None = input(f"[mk] About to {what}. Type 'yes' to continue: ")
        except EOFError:
            # Python on Windows reports NUL as a terminal: a prompt nobody can answer is not consent.
            answer = None
        if answer is not None:
            if answer.strip().lower() == "yes":
                return
            raise MkError("aborted: confirmation not given")
    raise MkError(
        f"confirmation required to {what}",
        "Preview first:   make <target> DRY_RUN=1\nThen approve:    make <target> CONFIRM=1",
    )


@contextlib.contextmanager
def file_lock(
    cfg: Config, name: str, stale_after: float = 12 * 3600, wait: float = 3600
) -> Iterator[None]:
    """One writer per ``name`` across make invocations: atomic create, waits up to ``wait`` seconds.

    A lock older than ``stale_after`` is assumed to be left by a crash. No PID probing:
    ``os.kill(pid, 0)`` terminates the process on Windows.
    """
    if cfg.dry_run:
        yield
        return
    lock = cfg.dir("build") / ".locks" / f"{name}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + wait
    announced = False
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                age = time.time() - lock.stat().st_mtime
            except FileNotFoundError:
                continue  # released between the two calls
            if age > stale_after:
                warn(f"removing stale lock {lock} ({age / 3600:.0f}h old)")
                lock.unlink(missing_ok=True)
                continue
            if time.monotonic() >= deadline:
                raise MkError(
                    f"another run of `{name}` is still in progress ({lock})",
                    "If it crashed, delete the lock file and retry.",
                ) from None
            if not announced:
                info(f"waiting for another `{name}` run to finish ({lock})")
                announced = True
            time.sleep(2)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def is_inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True
