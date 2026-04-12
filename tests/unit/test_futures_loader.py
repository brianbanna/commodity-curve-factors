"""Tests for futures_loader module."""

from commodity_curve_factors.data.futures_loader import _get_yfinance_symbol


def test_get_yfinance_symbol_valid() -> None:
    """Known commodity returns its yfinance symbol."""
    assert _get_yfinance_symbol("CL") == "CL=F"
    assert _get_yfinance_symbol("GC") == "GC=F"
    assert _get_yfinance_symbol("NG") == "NG=F"


def test_get_yfinance_symbol_invalid() -> None:
    """Unknown commodity returns None."""
    assert _get_yfinance_symbol("INVALID") is None
    assert _get_yfinance_symbol("") is None


def test_all_commodities_have_yfinance_symbols() -> None:
    """Every commodity in the universe has a yfinance symbol."""
    from commodity_curve_factors.utils.constants import ALL_COMMODITIES

    for sym in ALL_COMMODITIES:
        result = _get_yfinance_symbol(sym)
        assert result is not None, f"{sym} has no yfinance_symbol in config"
        assert "=F" in result, f"{sym} yfinance symbol '{result}' missing =F suffix"
