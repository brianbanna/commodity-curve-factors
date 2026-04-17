"""Run full evaluation report: python -m commodity_curve_factors.evaluation."""

import logging

from commodity_curve_factors.evaluation.report import main

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    main()
