"""YAML config loader for commodity-curve-factors."""

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def load_config(name: str) -> dict[str, Any]:
    """Load a YAML config file by name (without extension)."""
    path = PROJECT_ROOT / "configs" / f"{name}.yaml"
    with open(path) as f:
        result: dict[str, Any] = yaml.safe_load(f)
        return result


def get_path(relative: str) -> Path:
    """Resolve a path relative to project root."""
    return PROJECT_ROOT / relative
