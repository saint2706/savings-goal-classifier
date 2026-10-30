"""Command-line helper behind the root ``Makefile``.

    python scripts/mk.py <command> [args...]

The Makefile calls this for everything beyond a one-line tool invocation, because the
work is easier to test and to keep portable (sh / cmd.exe / PowerShell) in Python than in
shell. Settings arrive as ``MK_*`` environment variables exported by the Makefile
(``MK_ARG_*`` for per-target arguments such as ``MK_ARG_MSG``); see
``mk_core.Config.from_env``. Standard library only; Python 3.11+.

Commands (each is one Make target, see ``make help``):

    help, targets                 list the Makefile's documented targets
    say <text...>                 print one [mk] progress line
    run -- <tool> [args...]       run a tool from the project venv (DRY_RUN-aware)
    ensure-env, install, install-dev, sync, lock-check, import-check
    check-tools, python-info, check-env, doctor, health, audit, verify
    seed-check, environment-report, experiment-info, reproducibility
    data-check, data-prepare, data-features, data-validate, demo, train, serve
    notebook-{list,check,run,status,export}, docs-{lint,links,check}
    figures-check, figures, report, promote, clean
    precommit, hooks-install
    git-{status,branch,diff,check,stage,commit,pull,sync,push}
    version, version-check, version-bump, release-{state,notes,tag,push,verify}, release

Exit status: 0 success, 1 a check or action failed, 2 a prerequisite (tool, environment,
input file) is missing.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mk_env
import mk_git
import mk_work
from mk_core import (
    Config,
    MkError,
    error,
    info,
    ok,
    project_version,
    run_host,
    run_tool,
    set_verbosity,
)

HELP_TAGS = """\
Tags:  [safe] read-only   [writes] creates generated files (idempotent)   [env] changes the Python environment
       [confirm] destructive: needs CONFIRM=1 or an interactive 'yes'; preview with DRY_RUN=1
       [git] changes local git state, only when you name it   [publish] pushes to the remote, needs CONFIRM=1"""


# --- help ------------------------------------------------------------------------------

RE_SECTION = re.compile(r"^##@\s*(.+?)\s*$")
RE_TARGET = re.compile(r"^([A-Za-z0-9_.-]+(?:\s+[A-Za-z0-9_.-]+)*)\s*:(?!=)[^#=]*##\s*(.+?)\s*$")
RE_VAR = re.compile(r"^([A-Z][A-Z0-9_]*)\s*\?=\s*(.*?)\s*$")
RE_VAR_DOC = re.compile(r"^##\s+([^@].*?)\s*$")


def parse_makefile(
    path: Path,
) -> tuple[list[tuple[str, list[tuple[str, str]]]], list[tuple[str, str, str]]]:
    """(sections of (target, description), config variables of (name, default, description))."""
    sections: list[tuple[str, list[tuple[str, str]]]] = []
    variables: list[tuple[str, str, str]] = []
    doc = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if m := RE_SECTION.match(line):
            sections.append((m.group(1), []))
        elif m := RE_VAR_DOC.match(line):
            doc = m.group(1)
        elif (m := RE_VAR.match(line)) and doc:
            variables.append((m.group(1), m.group(2), doc))
            doc = ""
        elif (m := RE_TARGET.match(line)) and sections:
            for name in m.group(1).split():
                sections[-1][1].append((name, m.group(2)))
    return sections, variables


def cmd_help(cfg: Config) -> int:
    sections, variables = parse_makefile(cfg.root / "Makefile")
    version = project_version(cfg.root)
    print(f"Savings Goal Classifier {version}: developer interface")
    print("Usage: make <target> [VAR=value ...]      `make` alone shows this help\n")
    print(HELP_TAGS)
    for title, targets in sections:
        if not targets:
            continue
        print(f"\n{title}")
        for name, desc in targets:
            print(f"  {name:<20} {desc}")
    print(
        "\nConfiguration (override on the command line or in the environment: make test COV_MIN=90)"
    )
    name_w = max(len(v[0]) for v in variables) + 2
    default_w = 26  # a long default (RAW_TSV) simply overflows its column
    for name, default, desc in variables:
        print(f"  {name:<{name_w}}{default or '(unset)':<{default_w}} {desc}")
    print(
        "\nNotes\n"
        "  Windows: needs GNU Make (scoop install make | choco install make | winget install ezwinports.make)\n"
        "           and Python 3.11+; recipes avoid sh-only syntax and also run under cmd.exe / PowerShell.\n"
        "  Safety:  nothing here commits, pushes, tags or deletes unless you name that target; destructive\n"
        "           targets honour DRY_RUN=1 and need CONFIRM=1; force-push and history rewrites do not exist.\n"
        "  Docs:    docs/makefile.md has the full guide (workflows, releases, troubleshooting)."
    )
    return 0


def cmd_targets(cfg: Config) -> int:
    for _, targets in parse_makefile(cfg.root / "Makefile")[0]:
        for name, _ in targets:
            print(name)
    return 0


# --- small commands --------------------------------------------------------------------


def cmd_precommit(cfg: Config) -> int:
    if not (cfg.root / ".pre-commit-config.yaml").exists():
        raise MkError("there is no .pre-commit-config.yaml", code=2)
    if cfg.dry_run:
        info("dry-run: would run: pre-commit run --all-files")
        return 0
    run_host(cfg, ["pre-commit", "run", "--all-files", "--show-diff-on-failure"], tool="pre-commit")
    ok("pre-commit passed on all files")
    return 0


def cmd_hooks_install(cfg: Config) -> int:
    if not (cfg.root / ".pre-commit-config.yaml").exists():
        raise MkError("there is no .pre-commit-config.yaml", code=2)
    if cfg.dry_run:
        info("dry-run: would run: pre-commit install")
        return 0
    run_host(cfg, ["pre-commit", "install"], tool="pre-commit")
    info("git hooks installed: they run `make lint format-check typecheck` on each commit")
    return 0


COMMANDS: dict[str, Callable[[Config], int]] = {
    "help": cmd_help,
    "targets": cmd_targets,
    "install": mk_env.cmd_install,
    "install-dev": mk_env.cmd_install_dev,
    "sync": mk_env.cmd_sync,
    "ensure-env": mk_env.cmd_ensure_env,
    "lock-check": mk_env.cmd_lock_check,
    "import-check": mk_env.cmd_import_check,
    "check-tools": mk_env.cmd_check_tools,
    "python-info": mk_env.cmd_python_info,
    "check-env": mk_env.cmd_check_env,
    "doctor": mk_env.cmd_doctor,
    "health": mk_env.cmd_health,
    "audit": mk_env.cmd_audit,
    "verify": mk_env.cmd_verify,
    "seed-check": mk_env.cmd_seed_check,
    "environment-report": mk_env.cmd_environment_report,
    "experiment-info": mk_env.cmd_experiment_info,
    "reproducibility": mk_env.cmd_reproducibility,
    "data-check": lambda cfg: mk_env.data_report(cfg).finish(),
    "data-prepare": mk_work.cmd_data_prepare,
    "data-features": mk_work.cmd_data_features,
    "data-validate": mk_work.cmd_data_validate,
    "demo": mk_work.cmd_demo,
    "train": mk_work.cmd_train,
    "serve": mk_work.cmd_serve,
    "notebook-list": mk_work.cmd_notebook_list,
    "notebook-check": mk_work.cmd_notebook_check,
    "notebook-run": mk_work.cmd_notebook_run,
    "notebook-status": mk_work.cmd_notebook_status,
    "notebook-export": mk_work.cmd_notebook_export,
    "docs-lint": mk_work.cmd_docs_lint,
    "docs-links": mk_work.cmd_docs_links,
    "docs-check": mk_work.cmd_docs_check,
    "figures-check": mk_work.cmd_figures_check,
    "figures": mk_work.cmd_figures,
    "report": mk_work.cmd_report,
    "promote": mk_work.cmd_promote,
    "clean": mk_work.cmd_clean,
    "precommit": cmd_precommit,
    "hooks-install": cmd_hooks_install,
    "git-status": mk_git.cmd_status,
    "git-branch": mk_git.cmd_branch,
    "git-diff": mk_git.cmd_diff,
    "git-check": mk_git.cmd_check,
    "git-stage": mk_git.cmd_stage,
    "git-commit": mk_git.cmd_commit,
    "git-pull": mk_git.cmd_pull,
    "git-sync": mk_git.cmd_sync,
    "git-push": mk_git.cmd_push,
    "version": mk_git.cmd_version,
    "version-check": mk_git.cmd_version_check,
    "version-bump": mk_git.cmd_version_bump,
    "release-state": mk_git.cmd_release_state,
    "release-notes": mk_git.cmd_release_notes,
    "release-tag": mk_git.cmd_release_tag,
    "release": mk_git.cmd_release,
    "release-push": mk_git.cmd_release_push,
    "release-verify": mk_git.cmd_release_verify,
}


def cmd_run(cfg: Config, argv: Sequence[str]) -> int:
    if not argv:
        raise MkError("usage: mk.py run -- <tool> [args...]")
    run_tool(cfg, list(argv))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(__doc__)
        return 0 if args else 1
    cmd, rest = args[0], args[1:]
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    try:
        cfg = Config.from_env()
        set_verbosity(cfg.verbosity)
        os.chdir(cfg.root)
        if cmd == "run":
            return cmd_run(cfg, rest[1:] if rest[:1] == ["--"] else rest)
        if cmd == "say":
            info(" ".join(rest))
            return 0
        if cmd not in COMMANDS:
            raise MkError(f"unknown command {cmd!r}", "See `make help`.")
        return COMMANDS[cmd](cfg)
    except MkError as exc:
        if str(exc):
            error(str(exc), exc.hint)
        return exc.code
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
