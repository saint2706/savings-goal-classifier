"""A run manifest: what produced a set of results, so a reader can check they match theirs."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from savings_goal.config import ROOT, SEED, THRESHOLD

PACKAGES = (
    "numpy",
    "pandas",
    "polars",
    "scikit-learn",
    "xgboost",
    "shap",
    "scipy",
    "statsmodels",
    "pandera",
)


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def write_manifest(path: Path, tsv: Path | None = None) -> dict[str, Any]:
    """Record interpreter, pinned package versions, seed, commit and the input file's SHA-256."""
    info: dict[str, Any] = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {p: metadata.version(p) for p in PACKAGES},
        "seed": SEED,
        "threshold": THRESHOLD,
        "input": None
        if tsv is None
        else {"file": tsv.name, "bytes": tsv.stat().st_size, "sha256": sha256(tsv)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2) + "\n")
    return info
