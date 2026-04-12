"""Download all market data from public sources.

Orchestrates the four network-dependent data loaders (front-month futures,
macro, EIA inventories, CFTC COT) and prints a summary catalog. WRDS
back-month contract data is NOT downloaded here — it requires manual
authentication via ``scripts/probes/wrds_download_all.py``.

Usage:
    python -m commodity_curve_factors.data
    make data
"""

import logging

from commodity_curve_factors.data.catalog import print_catalog
from commodity_curve_factors.data.cftc_loader import download_cot_history, save_cot_data
from commodity_curve_factors.data.futures_loader import (
    download_all_front_month,
    save_front_month_data,
)
from commodity_curve_factors.data.inventory_loader import (
    download_all_eia,
    save_inventory_data,
)
from commodity_curve_factors.data.macro_loader import download_all_macro, save_macro_data
from commodity_curve_factors.data.validate import main as validate_main

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    logger.info("=== Downloading all market data ===")

    # 1. Front-month continuous futures (yfinance, 13 commodities)
    logger.info("--- Front-month futures ---")
    front = download_all_front_month(use_cache=True)
    save_front_month_data(front)
    logger.info("Front-month: %d commodities saved", len(front))

    # 2. Macro data (FRED + VIX + SPY + AGG)
    logger.info("--- Macro data ---")
    macro = download_all_macro(use_cache=True)
    save_macro_data(macro)
    logger.info("Macro: %d series saved", len(macro))

    # 3. EIA weekly inventories (crude, NG, gasoline, distillate)
    logger.info("--- EIA inventories ---")
    eia = download_all_eia(use_cache=True)
    save_inventory_data(eia)
    logger.info("EIA: %d series saved", len(eia))

    # 4. CFTC COT (disaggregated managed-money positions, 2006-2024)
    logger.info("--- CFTC COT ---")
    cot = download_cot_history(2006, 2024, use_cache=True)
    save_cot_data(cot)
    logger.info("CFTC: %d rows saved", len(cot))

    # Note: WRDS back-month contracts are NOT downloaded here.
    # They require manual 2FA authentication via:
    #   scripts/probes/wrds_download_all.py
    logger.info(
        "NOTE: WRDS back-month contracts must be downloaded separately via "
        "scripts/probes/wrds_download_all.py (requires manual 2FA)"
    )

    # Summary
    logger.info("=== Data download complete ===")
    print_catalog()

    # Spot-check validation
    logger.info("=== Running validation spot-checks ===")
    validate_main()

    logger.info("=== All done ===")


if __name__ == "__main__":
    main()
