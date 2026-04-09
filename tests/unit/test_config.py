"""Tests for config loader and project paths."""

from pathlib import Path

from commodity_curve_factors.utils.config import load_config
from commodity_curve_factors.utils.paths import DATA_RAW, PROJECT_ROOT


def test_load_universe_config() -> None:
    """Universe config loads and contains expected commodities."""
    cfg = load_config("universe")
    assert "commodities" in cfg
    assert "CL" in cfg["commodities"]
    assert "GC" in cfg["commodities"]
    assert len(cfg["commodities"]) == 13


def test_load_universe_has_yfinance_symbols() -> None:
    """Every commodity in universe.yaml has a yfinance_symbol."""
    cfg = load_config("universe")
    for symbol, spec in cfg["commodities"].items():
        assert "yfinance_symbol" in spec, f"{symbol} missing yfinance_symbol"


def test_load_universe_has_nasdaq_prefix() -> None:
    """Every commodity in universe.yaml has a nasdaq_prefix."""
    cfg = load_config("universe")
    for symbol, spec in cfg["commodities"].items():
        assert "nasdaq_prefix" in spec, f"{symbol} missing nasdaq_prefix"


def test_load_universe_date_range() -> None:
    """Universe config has valid date range."""
    cfg = load_config("universe")
    assert "date_range" in cfg
    assert cfg["date_range"]["start"] == "2005-01-01"
    assert cfg["date_range"]["end"] == "2024-12-31"


def test_all_configs_load() -> None:
    """All YAML config files load without error."""
    for name in ["universe", "curve", "factors", "inventory", "strategy", "backtest", "evaluation"]:
        cfg = load_config(name)
        assert isinstance(cfg, dict), f"{name}.yaml did not load as dict"


def test_project_root_exists() -> None:
    """PROJECT_ROOT points to a real directory containing pyproject.toml."""
    assert PROJECT_ROOT.exists()
    assert (PROJECT_ROOT / "pyproject.toml").exists()


def test_data_raw_path() -> None:
    """DATA_RAW resolves to expected location."""
    assert isinstance(DATA_RAW, Path)
    assert DATA_RAW.name == "raw"
