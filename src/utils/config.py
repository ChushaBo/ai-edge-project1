"""Load config.yaml and resolve paths.

Usage in a pipeline script:

    from utils.config import load_config, get_path, require

    cfg = load_config()
    raw_dir = get_path(cfg, "raw")
    start = require(cfg, "years.claims_start")
"""

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"


class DecisionMissingError(RuntimeError):
    """A config value a script needs has not been decided yet (it is null)."""


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read config.yaml (or another YAML file) and return it as a dict."""
    path = Path(path) if path is not None else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return cfg


def require(cfg: dict[str, Any], dotted_key: str) -> Any:
    """Return cfg value at a dotted key such as "splits.train_end".

    Raises KeyError if the key is absent and DecisionMissingError if it is null,
    so scripts stop instead of guessing an undecided value.
    """
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"config.yaml has no key '{dotted_key}'")
        node = node[part]
    if node is None:
        raise DecisionMissingError(
            f"config.yaml '{dotted_key}' is not decided yet (null). "
            "A person must set it and log it in docs/decisions.md."
        )
    return node


def require_date(cfg: dict[str, Any], dotted_key: str) -> pd.Timestamp:
    """Return a config date as a UTC pd.Timestamp, e.g. require_date(cfg, "splits.train_end").

    YAML loads unquoted 2019-12-31 as datetime.date, while loss dates in the data are
    UTC-aware timestamps; comparing the two directly raises a TypeError.
    """
    ts = pd.Timestamp(require(cfg, dotted_key))
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def get_path(cfg: dict[str, Any], name: str, create: bool = True) -> Path:
    """Return the absolute path for cfg["paths"][name], creating the folder if asked."""
    path = REPO_ROOT / require(cfg, f"paths.{name}")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
