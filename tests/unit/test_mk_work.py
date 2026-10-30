"""Cleanup safety, promotion, notebook discovery/execution and docs checks in ``scripts/mk_work.py``."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import mk_work
import pytest
from mk_core import Config, MkError


def touch(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --- cleanup: path validation ----------------------------------------------------------


def test_check_removable_refuses_dangerous_paths(project: Path) -> None:
    cfg = Config(root=project)
    (project / "build/backups").mkdir()
    refused = [
        project,  # the repository root
        project.parent,  # outside the repository
        Path.home(),  # the home directory
        Path(project.anchor),  # a filesystem root
        project / "src",  # protected source
        project / "src/pkg/mod.py",  # inside protected source
        project / "results",
        project / "dataset/ICPSR_36151/DS0002/36151-0002-Data.tsv",
        project / "pyproject.toml",
        project / "build/..",  # resolves to the root
        project / "build",  # contains build/backups, which is never deleted
    ]
    for path in refused:
        with pytest.raises(MkError, match="refusing to remove"):
            mk_work.check_removable(cfg, path)


def test_check_removable_allows_generated_locations(project: Path) -> None:
    cfg = Config(root=project)
    assert mk_work.check_removable(cfg, project / "build/test") == project / "build/test"
    cache = project / "src/pkg/__pycache__"
    assert mk_work.check_removable(cfg, cache, allow_inside_protected=True) == cache


def test_symlinks_are_removed_as_links_never_followed(project: Path, tmp_path: Path) -> None:
    outside = tmp_path / "precious"
    touch(outside / "keep.txt")
    link = project / "build/link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks need privileges on this platform")
    cfg = Config(root=project, args={"what": "scratch"})
    mk_work.check_removable(cfg, link)
    mk_work.remove_path(link)
    assert not link.exists() and (outside / "keep.txt").exists()


# --- cleanup: groups, dry-run, confirmation --------------------------------------------


@pytest.fixture
def messy(project: Path) -> Path:
    touch(project / "src/pkg/__pycache__/m.pyc")
    touch(project / "tests/__pycache__/t.pyc")
    touch(project / ".pytest_cache/v/cache")
    touch(project / ".coverage")
    touch(project / ".coverage.unit")
    touch(project / "build/test/junit.xml")
    touch(project / "build/demo/households.parquet")
    touch(project / "build/notebooks/results/a.json")
    touch(project / "build/notebooks/executed/01.ipynb")
    touch(project / "build/figures/fig1.png")
    touch(project / "build/report/report.pdf")
    touch(project / "build/backups/20260101/results/a.json", "old")
    touch(project / "artifacts/tier1.joblib")
    touch(project / "artifacts/previous/20260101/tier1.joblib", "archived")
    touch(project / "dataset/households.parquet")
    touch(project / "dataset/features.parquet")
    return project


def clean_cfg(root: Path, what: str, **kw: Any) -> Config:
    return Config(root=root, args={"what": what}, **kw)


def test_clean_removes_only_disposable_files_and_is_idempotent(messy: Path) -> None:
    assert mk_work.cmd_clean(clean_cfg(messy, "clean")) == 0
    gone = [
        "src/pkg/__pycache__",
        "tests/__pycache__",
        ".pytest_cache",
        ".coverage",
        ".coverage.unit",
    ]
    gone += ["build/test", "build/demo"]
    assert not any((messy / p).exists() for p in gone)
    kept = ["src/pkg/mod.py", "build/notebooks/results/a.json", "build/figures/fig1.png"]
    kept += ["artifacts/tier1.joblib", "dataset/households.parquet", "results/baseline.json"]
    kept += ["build/backups/20260101/results/a.json"]
    assert all((messy / p).exists() for p in kept)
    assert mk_work.cmd_clean(clean_cfg(messy, "clean")) == 0  # nothing left: still success


def test_clean_all_requires_confirmation(messy: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdin", None)
    with pytest.raises(MkError, match="confirmation required"):
        mk_work.cmd_clean(clean_cfg(messy, "all"))
    assert (messy / "build/notebooks").exists()


def test_clean_dry_run_lists_and_removes_nothing(
    messy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = sorted(p.relative_to(messy).as_posix() for p in messy.rglob("*"))
    assert mk_work.cmd_clean(clean_cfg(messy, "all", dry_run=True)) == 0
    out = capsys.readouterr().out
    assert "build/notebooks" in out and "artifacts/tier1.joblib" in out and "dry-run" in out
    assert sorted(p.relative_to(messy).as_posix() for p in messy.rglob("*")) == before


def test_clean_all_with_confirmation_spares_everything_valuable(messy: Path) -> None:
    mk_work.cmd_clean(clean_cfg(messy, "all", confirm=True))
    gone = ["build/notebooks", "build/figures", "build/report", "artifacts/tier1.joblib"]
    assert not any((messy / p).exists() for p in gone)
    kept = [
        "src/pkg/mod.py",
        "results/baseline.json",
        "dataset/households.parquet",
        "dataset/README.md",
    ]
    kept += ["dataset/ICPSR_36151/DS0002/36151-0002-Data.tsv", "pyproject.toml", "CITATION.cff"]
    kept += ["build/backups/20260101/results/a.json", "artifacts/previous/20260101/tier1.joblib"]
    assert all((messy / p).exists() for p in kept), [p for p in kept if not (messy / p).exists()]


def test_clean_refuses_in_ci_even_with_confirm(messy: Path) -> None:
    with pytest.raises(MkError, match="CI is set"):
        mk_work.cmd_clean(clean_cfg(messy, "all", confirm=True, ci=True))
    assert (messy / "build/notebooks").exists()


def test_data_clean_removes_derived_tables_only(messy: Path) -> None:
    mk_work.cmd_clean(clean_cfg(messy, "data", confirm=True))
    assert not (messy / "dataset/households.parquet").exists()
    assert not (messy / "dataset/features.parquet").exists()
    assert (messy / "dataset/ICPSR_36151/DS0002/36151-0002-Data.tsv").exists()
    assert (messy / "dataset/README.md").exists()


def test_misconfigured_directories_cannot_aim_deletion_at_source(messy: Path) -> None:
    for key, value in (
        ("nb_run", "."),
        ("nb_run", "src"),
        ("figures", "results"),
        ("report", ".."),
    ):
        cfg = Config(
            root=messy, args={"what": "notebooks"}, dirs={**Config(root=messy).dirs, key: value}
        )
        cfg_all = Config(root=messy, args={"what": "all"}, dry_run=True, dirs=cfg.dirs)
        with pytest.raises(MkError, match="refusing"):
            mk_work.cmd_clean(cfg_all)
    assert (messy / "src/pkg/mod.py").exists() and (messy / "results/baseline.json").exists()


def test_unknown_clean_group(messy: Path) -> None:
    with pytest.raises(MkError, match="unknown clean group"):
        mk_work.cmd_clean(clean_cfg(messy, "everything"))


def test_builds_refuse_tracked_output_locations(project: Path) -> None:
    cfg = Config(root=project)
    for target in ("project/figures", "results", "src"):
        with pytest.raises(MkError, match="tracked sources or results"):
            mk_work.refuse_tracked(cfg, project / target, "FIGURES_DIR")
    for target in (".", ".."):
        with pytest.raises(MkError, match="the repository itself or one of its parents"):
            mk_work.refuse_tracked(cfg, project / target, "FIGURES_DIR")
    assert mk_work.refuse_tracked(cfg, project / "build/figures", "X") == project / "build/figures"
    elsewhere = project.parent / "elsewhere"  # generated output may live outside the repository
    assert mk_work.refuse_tracked(cfg, elsewhere, "X") == elsewhere.resolve()


def test_clean_leaves_outside_directories_to_their_owner(
    project: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    outside = tmp_path / "my-figures"
    touch(outside / "fig1.png")
    dirs = {**Config(root=project).dirs, "figures": str(outside)}
    cfg = Config(root=project, args={"what": "figures"}, dirs=dirs, confirm=True)
    assert mk_work.cmd_clean(cfg) == 0
    assert (outside / "fig1.png").exists()
    assert "leaving" in capsys.readouterr().err


# --- promotion -------------------------------------------------------------------------


def test_promote_backs_up_and_asks_first(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    touch(project / "results/baseline.json", "old")
    touch(project / "results/same.json", "same")
    touch(project / "build/notebooks/results/baseline.json", "new")
    touch(project / "build/notebooks/results/same.json", "same")
    touch(project / "build/notebooks/results/added.json", "added")
    cfg = Config(root=project, args={"what": "results"})
    monkeypatch.setattr(sys, "stdin", None)

    with pytest.raises(MkError, match="confirmation required"):
        mk_work.cmd_promote(cfg)
    assert (project / "results/baseline.json").read_text(encoding="utf-8") == "old"

    assert mk_work.cmd_promote(Config(root=project, args={"what": "results"}, dry_run=True)) == 0
    assert (project / "results/baseline.json").read_text(encoding="utf-8") == "old"

    assert mk_work.cmd_promote(Config(root=project, args={"what": "results"}, confirm=True)) == 0
    assert (project / "results/baseline.json").read_text(encoding="utf-8") == "new"
    assert (project / "results/added.json").read_text(encoding="utf-8") == "added"
    backups = list((project / "build/backups").glob("*/results/baseline.json"))
    assert len(backups) == 1 and backups[0].read_text(encoding="utf-8") == "old"
    assert not list((project / "build/backups").glob("*/results/same.json"))  # identical: skipped


def test_promote_with_nothing_built_is_an_error_with_a_next_step(project: Path) -> None:
    with pytest.raises(MkError, match="nothing to promote") as exc:
        mk_work.cmd_promote(Config(root=project, args={"what": "figures"}, confirm=True))
    assert "make figures" in exc.value.hint


# --- notebooks -------------------------------------------------------------------------


def write_nb(path: Path, *cells: str, outputs: list[dict[str, Any]] | None = None) -> Path:
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "id": f"c{i}",
                "metadata": {},
                "outputs": outputs or [],
                "source": src.splitlines(keepends=True),
            }
            for i, src in enumerate(cells)
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    return path


def test_order_comes_from_data_flow_not_from_file_numbers(project: Path) -> None:
    nbs = project / "notebooks"
    write_nb(nbs / "01_consumer.ipynb", 'x = read_result("shared")\n')
    write_nb(nbs / "09_producer.ipynb", 'write_result("shared", {})\n')
    write_nb(nbs / "05_independent.ipynb", "y = 1\n")
    found = mk_work.discover_notebooks(Config(root=project))
    assert [nb.stem for nb in mk_work.execution_order(found)] == [
        "05_independent",
        "09_producer",
        "01_consumer",
    ]
    assert mk_work.dependencies(found)["01_consumer"] == {"09_producer"}


def test_data_files_create_dependencies_too(project: Path) -> None:
    nbs = project / "notebooks"
    write_nb(nbs / "07_a.ipynb", 'df.to_parquet(C.DATASET / "oof.parquet", index=False)\n')
    write_nb(nbs / "09_b.ipynb", 'df = pd.read_parquet(C.DATASET / "oof.parquet")\n')
    found = mk_work.discover_notebooks(Config(root=project))
    assert mk_work.dependencies(found)["09_b"] == {"07_a"}
    assert mk_work.closure(found, {"09_b"}) == {"09_b", "07_a"}


def test_dependency_cycle_is_reported(project: Path) -> None:
    nbs = project / "notebooks"
    write_nb(nbs / "01_a.ipynb", 'read_result("y")\nwrite_result("x", {})\n')
    write_nb(nbs / "02_b.ipynb", 'read_result("x")\nwrite_result("y", {})\n')
    with pytest.raises(MkError, match="dependency cycle"):
        mk_work.execution_order(mk_work.discover_notebooks(Config(root=project)))


def test_select_notebook_by_number_name_or_path(project: Path) -> None:
    write_nb(project / "notebooks/03_baseline.ipynb", "pass\n")
    write_nb(project / "notebooks/04_compare.ipynb", "pass\n")
    found = mk_work.discover_notebooks(Config(root=project))
    for query in (
        "03",
        "03_baseline",
        "notebooks/03_baseline.ipynb",
        "notebooks\\03_baseline.ipynb",
    ):
        assert mk_work.select_notebook(found, query).stem == "03_baseline"
    with pytest.raises(MkError, match="matches 0 notebooks"):
        mk_work.select_notebook(found, "99")


def test_notebook_check_flags_structure_and_machine_paths(project: Path) -> None:
    nbs = project / "notebooks"
    write_nb(nbs / "01_ok.ipynb", "x = 1\n")
    write_nb(nbs / "02_syntax.ipynb", "def broken(:\n")
    leak = [
        {"output_type": "execute_result", "data": {"text/plain": ["WindowsPath('C:/Users/me/x')"]}}
    ]
    write_nb(nbs / "03_leak.ipynb", "x = 1\n", outputs=leak)
    (nbs / "04_bad.ipynb").write_text("{not json", encoding="utf-8")
    problems = {nb.stem: nb.problems for nb in mk_work.discover_notebooks(Config(root=project))}
    assert problems["01_ok"] == []
    assert "does not parse" in problems["02_syntax"][0]
    assert "machine-specific" in problems["03_leak"][0]
    assert "not valid JSON" in problems["04_bad"][0]
    assert mk_work.notebook_report(Config(root=project)).count("FAIL") == 3


def test_run_plan_names_missing_prerequisites(project: Path) -> None:
    nbs = project / "notebooks"
    write_nb(nbs / "07_a.ipynb", 'load_features()\ndf.to_parquet(C.DATASET / "oof.parquet")\n')
    write_nb(
        nbs / "09_b.ipynb", 'load_features()\ndf = pd.read_parquet(C.DATASET / "oof.parquet")\n'
    )
    cfg = Config(root=project, args={"nb": "09"})
    with pytest.raises(MkError, match="prerequisites are missing") as exc:
        mk_work.plan_notebooks(cfg)
    assert "features.parquet" in str(exc.value) and "oof.parquet (written by 07_a" in str(exc.value)
    assert exc.value.code == 2


def test_fast_skips_the_slow_notebooks(project: Path) -> None:
    for name in ("03_baseline", "04_model_comparison", "05_explainability"):
        write_nb(project / f"notebooks/{name}.ipynb", "pass\n")
    chosen, _ = mk_work.plan_notebooks(Config(root=project, args={"fast": "1"}))
    assert [nb.stem for nb in chosen] == ["03_baseline"]


@pytest.mark.skipif(importlib.util.find_spec("nbconvert") is None, reason="needs nbconvert")
def test_notebook_run_is_sandboxed_and_reports_the_failing_notebook(project: Path) -> None:
    nbs = project / "notebooks"
    ok_nb = write_nb(
        nbs / "01_ok.ipynb",
        "import os, pathlib\n"
        "out = pathlib.Path(os.environ['SGC_RESULTS_DIR']) / 'made.txt'\n"
        "out.write_text('hello')\n",
    )
    bad_nb = write_nb(nbs / "02_boom.ipynb", "raise RuntimeError('kaboom')\n")
    originals = {p: p.read_bytes() for p in (ok_nb, bad_nb)}
    committed = {p: p.read_bytes() for p in (project / "results").iterdir()}
    cfg = Config(root=project, venv=sys.prefix, nb_timeout=120)

    with pytest.raises(MkError, match="notebook execution failed: 02_boom"):
        mk_work.cmd_notebook_run(cfg)

    status = json.loads((project / "build/notebooks/status.json").read_text(encoding="utf-8"))
    assert {r["notebook"]: r["status"] for r in status["notebooks"]} == {
        "01_ok": "ok",
        "02_boom": "failed",
    }
    assert "kaboom" in status["notebooks"][1]["error"]
    assert (project / "build/notebooks/results/made.txt").read_text(encoding="utf-8") == "hello"
    assert (project / "build/notebooks/executed/01_ok.ipynb").exists()
    # The originals and the committed results are untouched.
    assert {p: p.read_bytes() for p in originals} == originals
    assert {p: p.read_bytes() for p in (project / "results").iterdir()} == committed
    assert not (project / "results/made.txt").exists()
    assert not (project / "build/.locks/notebooks.lock").exists()  # the lock was released


def test_notebook_run_dry_run_executes_nothing(project: Path) -> None:
    write_nb(project / "notebooks/01_ok.ipynb", "raise RuntimeError('must not run')\n")
    assert mk_work.cmd_notebook_run(Config(root=project, dry_run=True)) == 0
    assert not (project / "build/notebooks").exists()


# --- documentation ---------------------------------------------------------------------


def test_docs_links_and_anchors(project: Path) -> None:
    (project / "walkthrough").mkdir()
    touch(project / "walkthrough/a.md", "# A\n\n## Setup and usage\n\n[home](../README.md)\n")
    touch(
        project / "README.md",
        "# Title\n\n## Setup and usage\n\n"
        "[ok](walkthrough/a.md#setup-and-usage) [self](#setup-and-usage) [ext](https://example.com/x)\n"
        "[gone](walkthrough/missing.md) [badanchor](walkthrough/a.md#nope)\n"
        "```\n[inside code](does/not/exist.md)\n```\n"
        "and `[inline](nor/this.md)`\n",
    )
    cfg = Config(root=project)
    issues = mk_work.check_links(
        cfg, project / "README.md", (project / "README.md").read_text(encoding="utf-8"), {}
    )
    assert len(issues) == 2
    assert any("walkthrough/missing.md" in i for i in issues)
    assert any("#nope" in i for i in issues)


def test_docs_lint_flags_an_unclosed_fence(project: Path) -> None:
    path = touch(project / "README.md", "# T\n\n```bash\nmake help\n")
    issues = mk_work.lint_markdown(path, path.read_text(encoding="utf-8"))
    assert issues == ["README.md:3: code fence is never closed"]
    assert mk_work.lint_markdown(path, "```\nx\n```\n") == []


def test_github_slugs() -> None:
    assert mk_work.github_slug("1. Overview") == "1-overview"
    assert mk_work.github_slug("Phase 0: Framing") == "phase-0-framing"
    assert mk_work.github_slug("`sgc` & the *CLI*") == "sgc--the-cli"


def test_environment_variable_paths_are_honoured_by_the_package(tmp_path: Path) -> None:
    import subprocess

    env = {
        **os.environ,
        "SGC_RESULTS_DIR": str(tmp_path / "res"),
        "SGC_DATA_DIR": str(tmp_path / "dat"),
        "SGC_SEED": "7",
    }
    code = (
        "import json; from savings_goal import config as c; "
        "print(json.dumps([str(c.RESULTS), str(c.DATASET), c.SEED]))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    ).stdout
    assert json.loads(out) == [
        str((tmp_path / "res").resolve()),
        str((tmp_path / "dat").resolve()),
        7,
    ]
