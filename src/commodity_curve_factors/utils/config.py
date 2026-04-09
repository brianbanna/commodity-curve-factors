"""YAML config loader for commodity-curve-factors."""

import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def load_config(name: str) -> dict:
    """Load a YAML config file by name (without extension)."""
    path = PROJECT_ROOT / "configs" / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def get_path(relative: str) -> Path:
    """Resolve a path relative to project root."""
    return PROJECT_ROOT / relative
