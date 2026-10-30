"""Data, notebook, documentation, figure/report, cleanup and promotion commands.

The rule that organises this module: builds only ever write to disposable locations
(``build/``, ``artifacts/``); the tracked artefacts under ``results/`` and ``project/``
change only through the ``promote`` command, which backs them up and asks first. Cleanup
removes paths from a fixed allowlist, after validating every one of them.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import stat
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from mk_core import (
    Config,
    MkError,
    Report,
    file_lock,
    info,
    install_hint,
    is_inside,
    ok,
    python_snippet,
    require_confirmation,
    require_env,
    run_host,
    run_tool,
    warn,
)

# --- shared path safety ----------------------------------------------------------------

# Never a deletion or build-output target, whatever a variable says.
PROTECTED = (
    ".git", ".github", "src", "tests", "notebooks", "scripts", "walkthrough", "project",
    "results", "dataset", "docs", "pyproject.toml", "uv.lock", "Makefile", "README.md",
    "LICENSE", "CITATION.cff", ".gitignore", ".gitattributes", ".pre-commit-config.yaml",
)  # fmt: skip
# Backups made by this Makefile: no removal target may contain them.
KEEP = ("build/backups", "artifacts/previous")


def rel(cfg: Config, path: Path) -> str:
    try:
        return path.relative_to(cfg.root).as_posix() or "."
    except ValueError:
        return str(path)


def is_tracked_location(cfg: Config, path: Path, *, follow: bool = True) -> bool:
    return any(
        path == cfg.root / p or is_inside(path, cfg.root / p, follow=follow) for p in PROTECTED
    )


def refuse_tracked(cfg: Config, path: Path, setting: str) -> Path:
    """Builds never write into tracked locations; ``make promote-*`` is the only way in."""
    resolved = path.resolve()
    if is_inside(cfg.root, resolved):  # the repository root, or a directory that contains it
        raise MkError(
            f"{setting}={rel(cfg, resolved)} is the repository itself or one of its parents",
            "Point it at a scratch directory: the default is under build/, and anywhere outside "
            "the repository works too.",
        )
    if is_inside(resolved, cfg.root) and is_tracked_location(cfg, resolved):
        raise MkError(
            f"{setting}={rel(cfg, resolved)} is inside the repository's tracked sources or results",
            "Build into a scratch directory (the default is under build/) and publish it with\n"
            "make promote-results / promote-figures / promote-report, which back up and ask first.",
        )
    return resolved


def check_removable(cfg: Config, path: Path, *, allow_inside_protected: bool = False) -> Path:
    """Raise unless ``path`` may be deleted. Every deletion in this file goes through here."""
    home = Path.home().resolve()
    # A symlink is removed as a link, never followed: judge the link's own location.
    resolved = (path.parent.resolve() / path.name) if path.is_symlink() else path.resolve()
    root = cfg.root
    if resolved in {root, home, Path(resolved.anchor)} or not is_inside(
        resolved, root, follow=False
    ):
        raise MkError(f"refusing to remove {resolved}: not strictly inside the repository")
    if any(root / p == resolved or is_inside(root / p, resolved) for p in PROTECTED):
        raise MkError(
            f"refusing to remove {rel(cfg, resolved)}: it is or contains protected source/config"
        )
    if is_tracked_location(cfg, resolved, follow=False) and not allow_inside_protected:
        raise MkError(f"refusing to remove {rel(cfg, resolved)}: inside a protected directory")
    for keep in KEEP:
        kp = (root / keep).resolve()
        if kp == resolved or is_inside(kp, resolved):
            raise MkError(
                f"refusing to remove {rel(cfg, resolved)}: it contains {keep}, which is never deleted"
            )
    return resolved


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n} B"


def tree_stats(path: Path) -> tuple[int, int]:
    if path.is_symlink() or path.is_file():
        return 1, path.lstat().st_size
    files = size = 0
    for p in path.rglob("*"):
        if p.is_file() and not p.is_symlink():
            files += 1
            size += p.stat().st_size
    return files, size


def _writable(func: Callable[[str], Any], p: str, _exc: object) -> None:
    os.chmod(p, stat.S_IWRITE)
    func(p)


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        if sys.version_info >= (3, 12):
            shutil.rmtree(path, onexc=_writable)
        else:
            shutil.rmtree(path, onerror=_writable)  # onexc needs 3.12


# --- clean -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Item:
    path: Path
    allow_inside_protected: bool = False


@dataclass(frozen=True)
class Group:
    description: str
    confirm: bool
    build: Callable[[Config], list[Item]]


def _walk_named(cfg: Config, bases: tuple[str, ...], names: frozenset[str]) -> Iterator[Path]:
    skip = {".venv", ".git", "node_modules", "build"}
    for base in bases:
        top = cfg.root / base
        if not top.is_dir():
            continue
        for dirpath, dirnames, _ in os.walk(top, followlinks=False):
            hits = [d for d in dirnames if d in names]
            for d in hits:
                yield Path(dirpath) / d
            dirnames[:] = [d for d in dirnames if d not in skip and d not in names]


def _cache(cfg: Config) -> list[Item]:
    items = [Item(cfg.root / d) for d in (".pytest_cache", ".mypy_cache", ".ruff_cache")]
    names = frozenset({"__pycache__", ".ipynb_checkpoints"})
    items += [
        Item(p, True)
        for p in _walk_named(cfg, ("src", "tests", "scripts", "project", "notebooks"), names)
    ]
    return items


def _test(cfg: Config) -> list[Item]:
    items = [
        Item(cfg.root / "htmlcov"),
        Item(cfg.root / ".coverage"),
        Item(cfg.dir("build") / "test"),
    ]
    items += [Item(p) for p in cfg.root.glob(".coverage.*")]
    return items


def _scratch(cfg: Config) -> list[Item]:
    build = cfg.dir("build")
    return [Item(cfg.dir("demo")), Item(build / "repro"), Item(build / "release")]


def _notebooks(cfg: Config) -> list[Item]:
    return [Item(cfg.dir("nb_run")), Item(cfg.dir("nb_export"))]


def _figures(cfg: Config) -> list[Item]:
    return [Item(cfg.dir("figures"))]


def _report(cfg: Config) -> list[Item]:
    return [Item(cfg.dir("report"))]


def _models(cfg: Config) -> list[Item]:
    m = cfg.dir("models")
    return [Item(m / n) for n in ("tier1.joblib", "tier2.joblib", "train_manifest.json")]


def _data(cfg: Config) -> list[Item]:
    d = cfg.dir("data")
    return [Item(d / n, True) for n in ("households.parquet", "features.parquet", "oof.parquet")]


GROUPS: dict[str, Group] = {
    "cache": Group("Python/tool caches and notebook checkpoints", False, _cache),
    "test": Group("coverage data, HTML coverage, test reports", False, _test),
    "scratch": Group("synthetic demo output and repro/release scratch files", False, _scratch),
    "notebooks": Group("sandboxed notebook runs and exports", True, _notebooks),
    "figures": Group("built figures (project/figures is never touched)", True, _figures),
    "report": Group("the built report (project/report.pdf is never touched)", True, _report),
    "models": Group("fitted models (archived runs in artifacts/previous are kept)", True, _models),
    "data": Group("derived dataset tables (the raw ICPSR files are never touched)", True, _data),
}
COMPOSITES = {
    "clean": ("cache", "test", "scratch"),
    "reports": ("figures", "report"),
    "all": ("cache", "test", "scratch", "notebooks", "figures", "report", "models"),
}


@dataclass(frozen=True)
class Entry:
    path: Path
    files: int
    size: int
    allow_inside_protected: bool


def plan_clean(cfg: Config, names: tuple[str, ...]) -> tuple[list[Entry], bool]:
    """Validated, existing removal targets, plus whether any group needs consent."""
    plan: list[Entry] = []
    seen: set[Path] = set()
    needs_confirm = False
    for name in names:
        group = GROUPS[name]
        needs_confirm = needs_confirm or group.confirm
        for item in group.build(cfg):
            if not (item.path.exists() or item.path.is_symlink()):
                continue
            where = item.path.resolve()
            if not is_inside(where, cfg.root) and not is_inside(cfg.root, where):
                warn(f"leaving {where} alone: it is outside the repository, so you manage it")
                continue
            resolved = check_removable(
                cfg, item.path, allow_inside_protected=item.allow_inside_protected
            )
            if resolved in seen:
                continue
            seen.add(resolved)
            plan.append(Entry(resolved, *tree_stats(item.path), item.allow_inside_protected))
    # Drop entries nested in another entry: the parent's removal covers them.
    plan = [
        e for e in plan if not any(o.path != e.path and is_inside(e.path, o.path) for o in plan)
    ]
    return plan, needs_confirm


def cmd_clean(cfg: Config) -> int:
    which = cfg.arg("what", "clean")
    names = COMPOSITES.get(which) or ((which,) if which in GROUPS else None)
    if names is None:
        raise MkError(
            f"unknown clean group {which!r}", "Known: " + ", ".join([*COMPOSITES, *GROUPS])
        )
    plan, needs_confirm = plan_clean(cfg, names)
    if not plan:
        ok(f"nothing to remove for `{which}` (already clean)")
        return 0
    total_files = sum(e.files for e in plan)
    total_bytes = sum(e.size for e in plan)
    info(f"`{which}` ({', '.join(names)}) would remove:")
    for e in plan:
        kind = "dir " if e.path.is_dir() and not e.path.is_symlink() else "file"
        print(f"  {kind} {rel(cfg, e.path)}  ({e.files} file(s), {human_size(e.size)})", flush=True)
    print(
        f"  total: {len(plan)} path(s), {total_files} file(s), {human_size(total_bytes)}",
        flush=True,
    )
    if cfg.dry_run:
        info("dry-run: nothing removed")
        return 0
    if needs_confirm:
        require_confirmation(cfg, f"delete the {len(plan)} path(s) listed above")
    for e in plan:
        check_removable(cfg, e.path, allow_inside_protected=e.allow_inside_protected)
        remove_path(e.path)
    ok(f"removed {len(plan)} path(s)")
    return 0


# --- backups and promotion -------------------------------------------------------------


def backup_dir(cfg: Config, stamp: str | None = None) -> Path:
    return cfg.dir("build") / "backups" / (stamp or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))


def backup_file(cfg: Config, src: Path, bdir: Path) -> Path:
    dest = bdir / rel(cfg, src)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest


def promote_pairs(cfg: Config) -> dict[str, tuple[Path, Path]]:
    return {
        "results": (cfg.dir("nb_run") / "results", cfg.root / "results"),
        "figures": (cfg.dir("figures"), cfg.root / "project" / "figures"),
        "report": (cfg.dir("report") / "report.pdf", cfg.root / "project" / "report.pdf"),
    }


def _files(src: Path) -> list[Path]:
    if src.is_file():
        return [src]
    return sorted(p for p in src.rglob("*") if p.is_file()) if src.is_dir() else []


def cmd_promote(cfg: Config) -> int:
    """Copy built outputs over their tracked counterparts, backing up each replaced file first."""
    pairs = promote_pairs(cfg)
    which = cfg.arg("what", "all")
    chosen = list(pairs) if which == "all" else [which]
    if any(c not in pairs for c in chosen):
        raise MkError(f"unknown promote target {which!r}", "Known: all, " + ", ".join(pairs))
    changes: list[tuple[Path, Path, bool]] = []  # (source, destination, destination existed)
    for key in chosen:
        src, dest = pairs[key]
        files = _files(src)
        if not files and which != "all":
            raise MkError(
                f"nothing to promote: {rel(cfg, src)} does not exist or is empty", hint_for(key)
            )
        for f in files:
            target = dest if src.is_file() else dest / f.relative_to(src)
            if target.exists() and target.read_bytes() == f.read_bytes():
                continue
            if not is_inside(target, cfg.root):
                raise MkError(f"refusing to write outside the repository: {target}")
            changes.append((f, target, target.exists()))
    if not changes:
        ok("nothing to promote: built outputs are identical to the tracked files (or absent)")
        return 0
    replaced = sum(1 for *_, existed in changes if existed)
    info(
        f"promote `{which}`: {replaced} tracked file(s) would be replaced, {len(changes) - replaced} added"
    )
    for _, target, existed in changes[:25]:
        print(f"  {'replace' if existed else 'add    '} {rel(cfg, target)}", flush=True)
    if len(changes) > 25:
        print(f"  ... and {len(changes) - 25} more", flush=True)
    if cfg.dry_run:
        info("dry-run: nothing copied")
        return 0
    require_confirmation(cfg, f"overwrite {replaced} tracked file(s) with freshly built ones")
    bdir = backup_dir(cfg)
    for src, target, existed in changes:
        if existed:
            backup_file(cfg, target, bdir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
    ok(f"promoted {len(changes)} file(s); replaced files are backed up in {rel(cfg, bdir)}")
    info("review with `git diff --stat`; undo with `git restore <path>` or the backup copies")
    return 0


def hint_for(key: str) -> str:
    return {
        "results": "make notebooks-run (or a single notebook target) first",
        "figures": "make figures first",
        "report": "make report first",
    }[key]


# --- data ------------------------------------------------------------------------------


def require_raw(cfg: Config) -> Path:
    tsv = cfg.path(cfg.raw_tsv)
    if not tsv.exists():
        raise MkError(
            f"the raw ICPSR file was not found at {rel(cfg, tsv)}",
            "Download DS0002 (free, registration required) from\n"
            "  https://www.icpsr.umich.edu/web/DSDR/studies/36151\n"
            "and either place 36151-0002-Data.tsv there or pass RAW_TSV=/path/to/36151-0002-Data.tsv.\n"
            "Details: dataset/README.md. The data cannot be redistributed, so this Makefile never fetches it.",
            code=2,
        )
    return tsv


def sgc(cfg: Config, *args: str) -> None:
    run_tool(cfg, ["python", "-m", "savings_goal.cli", *args])


def cmd_data_prepare(cfg: Config) -> int:
    tsv = require_raw(cfg)
    out = cfg.dir("data") / "households.parquet"
    manifest = cfg.dir("build") / "repro" / "data_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    info(f"building {rel(cfg, out)} from {rel(cfg, tsv)} (validated against the Pandera schema)")
    with file_lock(cfg, "data"):
        sgc(
            cfg,
            "build",
            "--tsv",
            str(tsv),
            "--out",
            str(out),
            "--no-features",
            "--manifest",
            str(manifest),
        )
    return 0


def cmd_data_features(cfg: Config) -> int:
    hh = cfg.dir("data") / "households.parquet"
    if not hh.exists() and not cfg.dry_run:
        raise MkError(f"{rel(cfg, hh)} does not exist", "make data-prepare")
    out = cfg.dir("data") / "features.parquet"
    with file_lock(cfg, "data"):
        sgc(cfg, "features", "--households", str(hh), "--out", str(out))
    return 0


def cmd_data_validate(cfg: Config) -> int:
    hh = cfg.dir("data") / "households.parquet"
    if not hh.exists():
        raise MkError(f"{rel(cfg, hh)} does not exist", "make data-prepare")
    code = (
        "import sys, polars as pl\n"
        "from savings_goal.data.schema import validate_households\n"
        "df = pl.read_parquet(sys.argv[1]); validate_households(df)\n"
        "print(f'{df.height:,} households x {df.width} columns satisfy the Pandera contract')"
    )
    if cfg.dry_run:
        info(f"dry-run: would validate {rel(cfg, hh)}")
        return 0
    print(python_snippet(cfg, code, str(hh)).strip())
    return 0


def cmd_demo(cfg: Config) -> int:
    out = refuse_tracked(cfg, cfg.dir("demo"), "DEMO_DIR")
    n = cfg.arg("n", "3000")
    info(f"end-to-end demo on a SYNTHETIC survey ({n} households) -> {rel(cfg, out)}")
    sgc(cfg, "demo", "--out", rel(cfg, out), "--n", n)  # relative, like `sgc demo --out`
    return 0


def cmd_train(cfg: Config) -> int:
    feats = cfg.dir("data") / "features.parquet"
    if not feats.exists() and not cfg.dry_run:
        raise MkError(f"{rel(cfg, feats)} does not exist", "make data-pipeline")
    models = cfg.dir("models")
    if is_tracked_location(cfg, models) or models == cfg.root:
        raise MkError(
            f"MODEL_DIR={rel(cfg, models)} would overwrite tracked files",
            "Use artifacts/ (the default).",
        )
    names = ("tier1.joblib", "tier2.joblib", "train_manifest.json")
    existing = [models / n for n in names if (models / n).exists()]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    archive = models / "previous" / stamp
    if existing:
        info(
            f"previous run found: archiving {', '.join(p.name for p in existing)} -> {rel(cfg, archive)}"
        )
    info(
        f"training both scoring tiers on {rel(cfg, feats)} (seed {cfg.seed}) -> {rel(cfg, models)}"
    )
    if cfg.dry_run:
        return 0
    with file_lock(cfg, "train"):
        if existing:
            archive.mkdir(parents=True, exist_ok=True)
            for p in existing:
                shutil.move(str(p), archive / p.name)
        sgc(cfg, "train", "--features", str(feats), "--artifacts", str(models))
        code = (
            "import sys; from pathlib import Path\n"
            "from savings_goal.manifest import write_manifest\n"
            "write_manifest(Path(sys.argv[1]), Path(sys.argv[2]))"
        )
        python_snippet(cfg, code, str(models / "train_manifest.json"), str(feats))
    ok(f"wrote {rel(cfg, models / 'train_manifest.json')} (seed, versions, commit, input SHA-256)")
    return 0


def cmd_serve(cfg: Config) -> int:
    probe = python_snippet(
        cfg, "import importlib.util as u; print(u.find_spec('uvicorn') is not None)"
    )
    if probe.strip() != "True":
        raise MkError(
            "uvicorn is not installed (it is in the optional `api` extra)",
            "make install-dev EXTRAS=api",
        )
    sgc(cfg, "serve", "--host", cfg.arg("host", "127.0.0.1"), "--port", cfg.arg("port", "8000"))
    return 0


# --- notebooks -------------------------------------------------------------------------

RE_WRITE_RESULT = re.compile(r"""write_result\(\s*["']([\w-]+)["']""")
RE_READ_RESULT = re.compile(r"""read_result\(\s*["']([\w-]+)["']""")
RE_WRITE_DATA = re.compile(r"""to_parquet\(\s*(?:C\.)?DATASET\s*/\s*["']([^"']+)["']""")
RE_READ_DATA = re.compile(r"""read_parquet\(\s*(?:C\.)?DATASET\s*/\s*["']([^"']+)["']""")
RE_HCES = re.compile(r"""DATASET\s*/\s*["']hces["']""")
LOCAL_PATH = re.compile(
    r"[A-Za-z]:[\\/]+Users[\\/]|/Users/[^/\s\"]+/|/home/[^/\s\"]+/|\\\\Users\\\\"
)


@dataclass
class Notebook:
    path: Path
    stem: str
    reads: set[str] = field(default_factory=set)
    writes: set[str] = field(default_factory=set)
    needs: set[str] = field(default_factory=set)
    optional: set[str] = field(default_factory=set)
    has_outputs: bool = False
    problems: list[str] = field(default_factory=list)

    @property
    def number(self) -> str:
        return self.stem.split("_", 1)[0]


def load_notebook(path: Path) -> Notebook:
    nb = Notebook(path=path, stem=path.stem)
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        nb.problems.append(f"not valid JSON: {exc}")
        return nb
    cells = data.get("cells")
    if data.get("nbformat", 0) < 4 or not isinstance(cells, list):
        nb.problems.append("not an nbformat>=4 notebook")
        return nb
    if not data.get("metadata", {}).get("kernelspec"):
        nb.problems.append("no kernelspec in metadata")
    sources: list[str] = []
    for i, cell in enumerate(cells):
        if cell.get("cell_type") not in {"markdown", "code", "raw"} or "source" not in cell:
            nb.problems.append(f"cell {i} is malformed")
            continue
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else str(cell["source"])
        sources.append(src)
        nb.has_outputs = nb.has_outputs or bool(cell.get("outputs"))
        plain = "\n".join(
            "" if ln.lstrip().startswith(("%", "!")) else ln for ln in src.splitlines()
        )
        try:
            ast.parse(plain)
        except SyntaxError as exc:
            nb.problems.append(f"code cell {i} does not parse: {exc.msg} (line {exc.lineno})")
    if LOCAL_PATH.search(raw):
        nb.problems.append(
            "contains a machine-specific absolute path (source or saved output); a cell that ends "
            "in a bare write_result(...) echoes the returned Path, so end it with a print instead"
        )
    code = "\n".join(sources)
    nb.writes |= {f"results:{n}.json" for n in RE_WRITE_RESULT.findall(code)}
    nb.reads |= {f"results:{n}.json" for n in RE_READ_RESULT.findall(code)}
    nb.writes |= {f"data:{n}" for n in RE_WRITE_DATA.findall(code)}
    nb.reads |= {f"data:{n}" for n in RE_READ_DATA.findall(code)}
    if "load_features(" in code:
        nb.needs.add("data:features.parquet")
    if "load_households(" in code:
        nb.needs.add("data:households.parquet")
    if RE_HCES.search(code):
        nb.optional.add("data:hces")
    return nb


def discover_notebooks(cfg: Config) -> list[Notebook]:
    paths = sorted((cfg.root / "notebooks").glob("*.ipynb"))
    if not paths:
        raise MkError("no notebooks found in notebooks/")
    return [load_notebook(p) for p in paths]


def dependencies(nbs: list[Notebook]) -> dict[str, set[str]]:
    """Notebook -> notebooks that produce something it reads (static scan, not file numbering)."""
    producer = {art: nb.stem for nb in nbs for art in nb.writes}
    return {
        nb.stem: {p for art in nb.reads if (p := producer.get(art)) and p != nb.stem} for nb in nbs
    }


def execution_order(nbs: list[Notebook]) -> list[Notebook]:
    """Topological order of the dependency graph; file name breaks ties. A cycle is an error."""
    deps = dependencies(nbs)
    by_stem = {nb.stem: nb for nb in nbs}
    done: list[str] = []
    remaining = sorted(by_stem)
    while remaining:
        ready = [s for s in remaining if deps[s] <= set(done)]
        if not ready:
            raise MkError("notebook dependency cycle among: " + ", ".join(remaining))
        done.append(ready[0])
        remaining.remove(ready[0])
    return [by_stem[s] for s in done]


def closure(nbs: list[Notebook], stems: set[str]) -> set[str]:
    deps = dependencies(nbs)
    out: set[str] = set()
    todo = list(stems)
    while todo:
        s = todo.pop()
        if s not in out:
            out.add(s)
            todo.extend(deps[s])
    return out


def select_notebook(nbs: list[Notebook], query: str) -> Notebook:
    q = query.replace("\\", "/").rsplit("/", 1)[-1].removesuffix(".ipynb")
    hits = [nb for nb in nbs if nb.stem == q or nb.number == q or nb.stem.startswith(q)]
    if len(hits) != 1:
        names = ", ".join(nb.stem for nb in nbs)
        raise MkError(
            f"NB={query!r} matches {len(hits)} notebooks",
            f"Use a number, a name or a path. Available: {names}",
        )
    return hits[0]


def data_file(cfg: Config, token: str) -> Path:
    return cfg.dir("data") / token.removeprefix("data:")


def notebook_report(cfg: Config) -> Report:
    r = Report("notebooks")
    try:
        nbs = discover_notebooks(cfg)
    except MkError as exc:
        r.failed("notebooks", str(exc))
        return r
    bad = [(nb.stem, p) for nb in nbs for p in nb.problems]
    if bad:
        for stem, problem in bad:
            r.failed(stem, problem)
    else:
        r.passed(
            "structure",
            f"{len(nbs)} notebooks parse; code cells are valid Python; kernelspec set; "
            "no machine-specific paths",
        )
    try:
        order = execution_order(nbs)
        r.passed("dependency order", " -> ".join(nb.number for nb in order))
    except MkError as exc:
        r.failed("dependency order", str(exc))
        return r
    external = sorted(
        {a for nb in nbs for a in nb.reads if a not in {w for o in nbs for w in o.writes}}
    )
    for token in sorted({t for nb in nbs for t in nb.needs}):
        if data_file(cfg, token).exists():
            r.passed(f"input {token}", "present")
        else:
            r.warned(
                f"input {token}",
                "missing: notebooks cannot be executed here",
                "make data-pipeline (needs the ICPSR download)",
            )
    if external:
        r.warned(
            "committed inputs",
            "read from the committed results, not produced by a notebook: " + ", ".join(external),
        )
    stale = [nb.stem for nb in nbs if not nb.has_outputs]
    if stale:
        r.warned("saved outputs", "no saved outputs in: " + ", ".join(stale))
    return r


def cmd_notebook_list(cfg: Config) -> int:
    nbs = discover_notebooks(cfg)
    deps = dependencies(nbs)
    rows = [("step", "notebook", "after", "needs (inputs)", "writes (shared state)", "saved")]
    for i, nb in enumerate(execution_order(nbs), 1):
        rows.append(
            (
                str(i),
                nb.stem,
                ",".join(sorted(d.split("_", 1)[0] for d in deps[nb.stem])) or "-",
                ",".join(sorted(t.split(":", 1)[1] for t in nb.needs | nb.optional)) or "-",
                ",".join(sorted(w.split(":", 1)[1] for w in nb.writes)) or "-",
                "yes" if nb.has_outputs else "no",
            )
        )
    widths = [max(len(row[c]) for row in rows) for c in range(6)]
    print("Order is derived from what each notebook reads and writes (not from its number):\n")
    for row in rows:
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)).rstrip())
    return 0


def cmd_notebook_check(cfg: Config) -> int:
    return notebook_report(cfg).finish()


def seed_sandbox(cfg: Config, sandbox: Path) -> None:
    """First run only: copy the committed results in, so a partial run can read its upstreams."""
    results = sandbox / "results"
    if results.exists():
        return
    shutil.copytree(cfg.root / "results", results)
    info(f"seeded {rel(cfg, results)} from the committed results/ (the originals are not touched)")


def jupyter_ready(cfg: Config) -> None:
    proc = run_tool(
        cfg, ["python", "-m", "jupyter", "nbconvert", "--version"], capture=True, check=False
    )
    if proc.returncode != 0:
        raise MkError(
            "jupyter nbconvert is not available in the project environment",
            "make install-dev (it is in the dev group)",
            code=2,
        )


def plan_notebooks(cfg: Config) -> tuple[list[Notebook], list[Notebook]]:
    """(ordered notebooks to execute, all notebooks), with prerequisites verified."""
    nbs = discover_notebooks(cfg)
    broken = [f"{nb.stem}: {p}" for nb in nbs for p in nb.problems]
    if broken:
        raise MkError(
            "notebook validation failed:\n  " + "\n  ".join(broken), "make notebook-check"
        )
    order = execution_order(nbs)
    query = cfg.arg("nb")
    if query:
        stems = {select_notebook(nbs, query).stem}
        if cfg.arg("with_deps"):
            stems = closure(nbs, stems)
        chosen = [nb for nb in order if nb.stem in stems]
    else:
        chosen = order
    if cfg.arg("fast") and not query:
        slow = cfg.arg("slow", "04_model_comparison 05_explainability").split()
        skipped = [nb.stem for nb in chosen if nb.stem.startswith(tuple(slow))]
        chosen = [nb for nb in chosen if nb.stem not in skipped]
        if skipped:
            warn(
                f"FAST=1: skipping {', '.join(skipped)}; their committed results are used by later notebooks"
            )
    produced = {a for nb in chosen for a in nb.writes}
    deps = dependencies(nbs)
    sandbox_results = cfg.dir("nb_run") / "results"
    missing: list[str] = []
    for nb in chosen:
        missing += [
            f"{nb.stem}: needs {t.split(':', 1)[1]}"
            for t in sorted(nb.needs)
            if not data_file(cfg, t).exists()
        ]
        for art in sorted(nb.reads - produced):
            if art.startswith("data:") and not data_file(cfg, art).exists():
                owner = ",".join(sorted(deps[nb.stem])) or "an upstream notebook"
                missing.append(
                    f"{nb.stem}: needs {art.split(':', 1)[1]} (written by {owner}; run it first or add WITH_DEPS=1)"
                )
            elif art.startswith("results:") and not (
                (sandbox_results / art.split(":", 1)[1]).exists()
                or (cfg.root / "results" / art.split(":", 1)[1]).exists()
            ):
                missing.append(f"{nb.stem}: needs results/{art.split(':', 1)[1]}")
    if missing:
        raise MkError(
            "prerequisites are missing:\n  " + "\n  ".join(missing),
            "make data-pipeline for the tables; WITH_DEPS=1 to include upstream notebooks",
            code=2,
        )
    return chosen, nbs


def cmd_notebook_run(cfg: Config) -> int:
    chosen, _ = plan_notebooks(cfg)
    sandbox = refuse_tracked(cfg, cfg.dir("nb_run"), "NB_RUN_DIR")
    executed = sandbox / "executed"
    sandbox_results = sandbox / "results"
    shared = sorted(
        {w.split(":", 1)[1] for nb in chosen for w in nb.writes if w.startswith("data:")}
    )
    info(f"executing {len(chosen)} notebook(s) in order: {' -> '.join(nb.number for nb in chosen)}")
    info(
        f"originals stay untouched; executed copies -> {rel(cfg, executed)}; results -> {rel(cfg, sandbox_results)}"
    )
    if shared:
        info(
            f"note: also refreshes the derived, git-ignored {', '.join(shared)} in {rel(cfg, cfg.dir('data'))} (previous copy is backed up)"
        )
    if {"04_model_comparison", "05_explainability"} & {nb.stem for nb in chosen}:
        info(
            "04 (tuning) and 05 (SHAP) are the slow ones: the full run takes roughly 25 minutes on 4 cores"
        )
    if cfg.dry_run:
        info(f"dry-run: nothing executed (timeout {cfg.nb_timeout}s per cell, seed {cfg.seed})")
        return 0
    require_env(cfg)
    jupyter_ready(cfg)
    keep_going = bool(cfg.arg("keep_going"))
    rows: list[dict[str, Any]] = []
    with file_lock(cfg, "notebooks"):
        executed.mkdir(parents=True, exist_ok=True)
        seed_sandbox(cfg, sandbox)
        bdir = backup_dir(cfg)
        for rel_name in shared:
            f = cfg.dir("data") / rel_name
            if f.exists():
                backup_file(cfg, f, bdir)
        failed = False
        for nb in chosen:
            if failed and not keep_going:
                rows.append(
                    {
                        "notebook": nb.stem,
                        "status": "skipped",
                        "seconds": 0,
                        "error": "an earlier notebook failed",
                    }
                )
                continue
            info(f"running {nb.stem} ...")
            start = time.monotonic()
            proc = run_tool(
                cfg,
                ["python", "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                 "--output-dir", str(executed), f"--ExecutePreprocessor.timeout={cfg.nb_timeout}", str(nb.path)],
                env={"SGC_RESULTS_DIR": str(sandbox_results), "MPLBACKEND": "Agg"},
                capture=True, check=False,
            )  # fmt: skip
            secs = round(time.monotonic() - start, 1)
            (executed / f"{nb.stem}.log").write_text(
                (proc.stdout or "") + (proc.stderr or ""), encoding="utf-8"
            )
            if proc.returncode == 0:
                rows.append({"notebook": nb.stem, "status": "ok", "seconds": secs, "error": ""})
                ok(f"{nb.stem} finished in {secs}s")
            else:
                tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
                rows.append(
                    {
                        "notebook": nb.stem,
                        "status": "failed",
                        "seconds": secs,
                        "error": "\n".join(tail),
                    }
                )
                warn(
                    f"{nb.stem} FAILED after {secs}s (log: {rel(cfg, executed / (nb.stem + '.log'))})\n           "
                    + "\n           ".join(tail)
                )
                failed = True
        status = {
            "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "seed": cfg.seed,
            "notebooks": rows,
        }
        (sandbox / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print_status(rows)
    bad = [r["notebook"] for r in rows if r["status"] == "failed"]
    if bad:
        raise MkError(
            f"notebook execution failed: {', '.join(bad)}",
            "See the logs above; results of the failed notebook are incomplete.",
        )
    info(
        "to publish the new results over the tracked ones: make promote-results DRY_RUN=1  (preview), then CONFIRM=1"
    )
    return 0


def print_status(rows: list[dict[str, Any]]) -> None:
    print(f"\n{'notebook':<34}{'status':<9}seconds", flush=True)
    for r in rows:
        print(f"{r['notebook']:<34}{r['status']:<9}{r['seconds']}", flush=True)


def cmd_notebook_status(cfg: Config) -> int:
    path = cfg.dir("nb_run") / "status.json"
    if not path.exists():
        info("no notebook run recorded yet (make notebooks-run)")
        return 0
    data = json.loads(path.read_text(encoding="utf-8"))
    info(f"last run: {data['generated_utc']} (seed {data.get('seed')})")
    print_status(data["notebooks"])
    return 1 if any(r["status"] == "failed" for r in data["notebooks"]) else 0


def cmd_notebook_export(cfg: Config) -> int:
    fmt = cfg.arg("format", "html")
    if fmt not in {"html", "markdown", "script", "rst"}:
        raise MkError(f"NB_FORMAT must be html, markdown, script or rst (got {fmt!r})")
    out = refuse_tracked(cfg, cfg.dir("nb_export"), "NB_EXPORT_DIR")
    nbs = discover_notebooks(cfg)
    chosen = [select_notebook(nbs, cfg.arg("nb"))] if cfg.arg("nb") else nbs
    info(
        f"exporting {len(chosen)} notebook(s) as {fmt} (saved outputs, nothing is re-executed) -> {rel(cfg, out)}"
    )
    if cfg.dry_run:
        return 0
    require_env(cfg)
    jupyter_ready(cfg)
    out.mkdir(parents=True, exist_ok=True)
    for nb in chosen:
        run_tool(
            cfg,
            [
                "python",
                "-m",
                "jupyter",
                "nbconvert",
                "--to",
                fmt,
                "--output-dir",
                str(out),
                str(nb.path),
            ],
            capture=True,
        )
    ok(f"exported to {rel(cfg, out)}")
    return 0


# --- documentation ---------------------------------------------------------------------

RE_LINK = re.compile(r"!?\[(?:[^\]\\]|\\.)*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
RE_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
RE_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")


def doc_files(cfg: Config) -> list[Path]:
    files = [cfg.root / "README.md", cfg.root / "dataset" / "README.md"]
    for sub in ("walkthrough", "docs"):
        files += sorted((cfg.root / sub).glob("*.md"))
    return [f for f in files if f.exists()]


def github_slug(heading: str) -> str:
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", heading)
    text = re.sub(r"[`*_~]", "", text).strip().lower()
    text = re.sub(r"[^\w\- ]", "", text, flags=re.UNICODE)
    return text.replace(" ", "-")


def markdown_lines(text: str) -> Iterator[tuple[int, str, bool]]:
    """(line number, line, inside_fenced_code) for every line."""
    fence: str | None = None
    for n, line in enumerate(text.splitlines(), 1):
        m = RE_FENCE.match(line)
        if fence is None and m:
            fence = m.group(1)[0] * 3
            yield n, line, True
        elif (
            fence is not None and m and m.group(1).startswith(fence) and line.strip() == m.group(1)
        ):
            fence = None
            yield n, line, True
        else:
            yield n, line, fence is not None


def anchors_of(text: str) -> set[str]:
    seen: dict[str, int] = {}
    out: set[str] = set()
    for _, line, in_code in markdown_lines(text):
        m = None if in_code else RE_HEADING.match(line)
        if m:
            slug = github_slug(m.group(2))
            count = seen.get(slug, 0)
            seen[slug] = count + 1
            out.add(slug if count == 0 else f"{slug}-{count}")
    return out


def lint_markdown(path: Path, text: str) -> list[str]:
    issues: list[str] = []
    state_open: int | None = None
    fence: str | None = None
    for n, line in enumerate(text.splitlines(), 1):
        m = RE_FENCE.match(line)
        if fence is None and m:
            fence, state_open = m.group(1)[0] * 3, n
        elif (
            fence is not None and m and m.group(1).startswith(fence) and line.strip() == m.group(1)
        ):
            fence = None
    if fence is not None:
        issues.append(f"{path.name}:{state_open}: code fence is never closed")
    return issues


def check_links(cfg: Config, path: Path, text: str, cache: dict[Path, set[str]]) -> list[str]:
    issues: list[str] = []
    for n, line, in_code in markdown_lines(text):
        if in_code:
            continue
        scrubbed = re.sub(r"`[^`]*`", "", line)
        for m in RE_LINK.finditer(scrubbed):
            target = m.group(1)
            if re.match(r"^(?:[a-z][a-z0-9+.-]*:|//)", target, re.I):
                continue  # external: not checked (no network in this target)
            ref, _, frag = unquote(target).partition("#")
            dest = path if not ref else (path.parent / ref).resolve()
            where = f"{rel(cfg, path)}:{n}"
            if not dest.exists():
                issues.append(f"{where}: broken link to {ref or path.name}")
            elif frag and dest.suffix == ".md":
                if dest not in cache:
                    cache[dest] = anchors_of(dest.read_text(encoding="utf-8"))
                if frag.lower() not in cache[dest]:
                    issues.append(f"{where}: no heading for anchor #{frag} in {rel(cfg, dest)}")
    return issues


def docs_report(cfg: Config) -> Report:
    r = Report("documentation")
    files = doc_files(cfg)
    if not files:
        r.failed("markdown files", "none found")
        return r
    cache: dict[Path, set[str]] = {}
    lint: list[str] = []
    links: list[str] = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        lint += lint_markdown(f, text)
        links += check_links(cfg, f, text, cache)
    (r.failed if lint else r.passed)(
        "markdown lint", "\n".join(lint) if lint else f"{len(files)} files: fences balanced"
    )
    (r.failed if links else r.passed)(
        "local links", "\n".join(links) if links else "every relative link and anchor resolves"
    )
    return r


def cmd_docs_lint(cfg: Config) -> int:
    r = Report("docs lint")
    problems = [p for f in doc_files(cfg) for p in lint_markdown(f, f.read_text(encoding="utf-8"))]
    (r.failed if problems else r.passed)("markdown lint", "\n".join(problems) if problems else "ok")
    return r.finish()


def cmd_docs_links(cfg: Config) -> int:
    r = Report("docs links")
    cache: dict[Path, set[str]] = {}
    problems = [
        p for f in doc_files(cfg) for p in check_links(cfg, f, f.read_text(encoding="utf-8"), cache)
    ]
    (r.failed if problems else r.passed)("local links", "\n".join(problems) if problems else "ok")
    return r.finish()


def cmd_docs_check(cfg: Config) -> int:
    return docs_report(cfg).finish()


# --- figures and report ----------------------------------------------------------------


def figure_inputs(cfg: Config) -> list[str]:
    """Result files make_figures.py reads, taken from the script itself."""
    script = cfg.root / "project" / "figures" / "make_figures.py"
    if not script.exists():
        raise MkError(
            f"{rel(cfg, script)} not found", "This repository has no figure script.", code=2
        )
    text = script.read_text(encoding="utf-8")
    names = {f"{n}.json" for n in re.findall(r"""result\(\s*["'](\w+)["']""", text)}
    names |= set(re.findall(r"""RESULTS\s*/\s*["']([\w.-]+\.(?:csv|json))["']""", text))
    return sorted(names)


def cmd_figures_check(cfg: Config) -> int:
    r = Report("figure inputs")
    results = cfg.dir("results")
    for name in figure_inputs(cfg):
        (r.passed if (results / name).exists() else r.failed)(name, rel(cfg, results / name))
    return r.finish()


def cmd_figures(cfg: Config) -> int:
    out = refuse_tracked(cfg, cfg.dir("figures"), "FIGURES_DIR")
    results = cfg.dir("results")
    missing = [n for n in figure_inputs(cfg) if not (results / n).exists()]
    if missing and not cfg.dry_run:
        raise MkError(
            f"missing inputs in {rel(cfg, results)}: {', '.join(missing)}",
            "make notebooks-run, or point RESULTS_DIR at a directory that has them",
        )
    info(
        f"building the report figures from {rel(cfg, results)} -> {rel(cfg, out)} (the tracked PNGs are not touched)"
    )
    with file_lock(cfg, "figures"):
        run_tool(cfg, ["python", str(cfg.root / "project" / "figures" / "make_figures.py")],
                 env={"SGC_FIGURES_DIR": str(out), "SGC_RESULTS_DIR": str(results), "MPLBACKEND": "Agg"})  # fmt: skip
    info("to publish: make promote-figures DRY_RUN=1  (preview), then CONFIRM=1")
    return 0


def cmd_report(cfg: Config) -> int:
    out = refuse_tracked(cfg, cfg.dir("report"), "REPORT_DIR")
    project = cfg.root / "project"
    if not (project / "report.tex").exists():
        raise MkError(
            "project/report.tex not found", "This repository has no report source.", code=2
        )
    built = cfg.dir("figures")
    info(f"building project/report.tex with latexmk -> {rel(cfg, out / 'report.pdf')}")
    if cfg.dry_run:
        info("dry-run: nothing built")
        return 0
    if shutil.which("latexmk") is None:
        raise MkError(
            "latexmk was not found",
            f"Install a TeX distribution:  {install_hint('latexmk')}",
            code=2,
        )
    with file_lock(cfg, "report"):
        (out / "figures").mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / "report.tex", out / "report.tex")
        for f in (project / "figures").glob("*.png"):
            shutil.copy2(f, out / "figures" / f.name)
        overlay = sorted(built.glob("*.png")) if built.is_dir() else []
        for f in overlay:
            shutil.copy2(f, out / "figures" / f.name)
        info(
            f"figures: committed ones{f', overlaid by {len(overlay)} from {rel(cfg, built)}' if overlay else ''}"
        )
        run_host(
            cfg,
            [
                "latexmk",
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-file-line-error",
                "report.tex",
            ],
            tool="latexmk",
            cwd=out,
        )
    ok(f"built {rel(cfg, out / 'report.pdf')}")
    info("to publish: make promote-report DRY_RUN=1  (preview), then CONFIRM=1")
    return 0
