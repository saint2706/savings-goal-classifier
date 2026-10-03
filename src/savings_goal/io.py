"""Reading the (git-ignored) data tables and writing the (committed) result artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from savings_goal.config import FEATURES_PATH, HOUSEHOLDS_PATH, RESULTS


def load_households(path: Path = HOUSEHOLDS_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_features(path: Path = FEATURES_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def to_jsonable(o: Any) -> Any:
    """Recursively convert numpy / pandas objects to JSON types; NaN and inf become null."""
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [to_jsonable(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return to_jsonable(o.to_dict(orient="records"))
    if isinstance(o, pd.Series):
        return to_jsonable(o.to_dict())
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    if isinstance(o, bool | np.bool_):
        return bool(o)
    if isinstance(o, int | np.integer):
        return int(o)
    if isinstance(o, float | np.floating):
        return float(o) if np.isfinite(o) else None
    if o is None or isinstance(o, str):
        return o
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_result(name: str, payload: dict[str, Any], results: Path = RESULTS) -> Path:
    """Write ``results/<name>.json``; aggregates only, never household rows."""
    results.mkdir(parents=True, exist_ok=True)
    path = results / f"{name}.json"
    path.write_text(json.dumps(to_jsonable(payload), indent=2, allow_nan=False) + "\n")
    return path


def read_result(name: str, results: Path = RESULTS) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((results / f"{name}.json").read_text())
    return data
