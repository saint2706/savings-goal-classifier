"""Tiny helpers shared by the Makefile-helper tests (a real module, so no conftest import games)."""

from __future__ import annotations

import subprocess
from pathlib import Path


def git(cwd: Path, *args: str) -> str:
    out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    return out.stdout.strip()
