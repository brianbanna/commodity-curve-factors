"""Contract specifications, sector mappings, and shared constants."""

TRADING_DAYS_PER_YEAR = 252

SECTORS = {
    "energy": ["CL", "NG", "HO", "RB"],
    "metals": ["GC", "SI", "HG", "PL", "PA"],
    "agriculture": ["ZC", "ZS", "ZW", "LB"],
    "softs": ["KC", "SB", "CC"],
    "livestock": ["LC", "LH", "FC"],
}

ALL_COMMODITIES = [sym for syms in SECTORS.values() for sym in syms]

SECTOR_COLORS = {
    "energy": "#ef5350",
    "metals": "#ffa726",
    "agriculture": "#66bb6a",
    "softs": "#4fc3f7",
    "livestock": "#ce93d8",
}

FACTOR_COLORS = {
    "carry": "#4fc3f7",
    "slope": "#66bb6a",
    "curvature": "#ffa726",
    "momentum": "#ef5350",
    "inventory": "#ab47bc",
}

FIGURE_DPI = 300
