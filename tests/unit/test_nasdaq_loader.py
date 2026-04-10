"""Tests for Nasdaq Data Link back-month contract loader."""

from commodity_curve_factors.data.nasdaq_loader import build_chris_symbol


def test_build_chris_symbol_cme() -> None:
    assert build_chris_symbol("CHRIS/CME_CL", 1) == "CHRIS/CME_CL1"
    assert build_chris_symbol("CHRIS/CME_CL", 3) == "CHRIS/CME_CL3"
    assert build_chris_symbol("CHRIS/CME_CL", 12) == "CHRIS/CME_CL12"


def test_build_chris_symbol_ice() -> None:
    assert build_chris_symbol("CHRIS/ICE_KC", 1) == "CHRIS/ICE_KC1"
    assert build_chris_symbol("CHRIS/ICE_CC", 6) == "CHRIS/ICE_CC6"


def test_build_chris_symbol_different_commodities() -> None:
    assert build_chris_symbol("CHRIS/CME_GC", 2) == "CHRIS/CME_GC2"
    assert build_chris_symbol("CHRIS/CME_C", 5) == "CHRIS/CME_C5"
