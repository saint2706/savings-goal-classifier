from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from savings_goal.cli import main


def test_sgc_build_and_features(
    raw_tsv: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "households.parquet"
    feats = tmp_path / "features.parquet"
    assert main(["build", "--tsv", str(raw_tsv), "--out", str(out), "--features", str(feats)]) == 0
    printed = capsys.readouterr().out
    assert "final households" in printed and "COTOTAL vs 11-category sum" in printed
    hh, f = pd.read_parquet(out), pd.read_parquet(feats)
    assert len(hh) == len(f)
    assert f["Is_Test"].any()

    again = tmp_path / "features2.parquet"
    assert main(["features", "--households", str(out), "--out", str(again)]) == 0
    pd.testing.assert_frame_equal(pd.read_parquet(again), f)


def test_sgc_build_csv_without_features(raw_tsv: Path, tmp_path: Path) -> None:
    out = tmp_path / "households.csv"
    assert main(["build", "--tsv", str(raw_tsv), "--out", str(out), "--no-features"]) == 0
    assert out.exists() and not (tmp_path / "features.parquet").exists()


def test_sgc_train(raw_tsv: Path, tmp_path: Path) -> None:
    out, feats = tmp_path / "hh.parquet", tmp_path / "f.parquet"
    main(["build", "--tsv", str(raw_tsv), "--out", str(out), "--features", str(feats)])
    arts = tmp_path / "artifacts"
    assert main(["train", "--features", str(feats), "--artifacts", str(arts)]) == 0
    assert (arts / "tier1.joblib").exists() and (arts / "tier2.joblib").exists()


def test_console_script_entry_point() -> None:
    r = subprocess.run(
        [sys.executable, "-m", "savings_goal.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0 and "build" in r.stdout
