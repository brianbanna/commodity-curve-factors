"""Historical stress testing and drawdown analysis."""

import logging

import numpy as np
import pandas as pd

from commodity_curve_factors.evaluation.metrics import compute_all_metrics
from commodity_curve_factors.utils.config import load_config

logger = logging.getLogger(__name__)


def historical_stress_test(
    returns: pd.Series,
    periods: dict[str, dict[str, str]] | None = None,
) -> pd.DataFrame:
    """Evaluate strategy performance during historical stress periods.

    Parameters
    ----------
    returns : pd.Series
        Daily log returns with DatetimeIndex.
    periods : dict or None
        Stress periods as ``{name: {"start": str, "end": str}}``.
        If None, loads from ``configs/evaluation.yaml``.

    Returns
    -------
    pd.DataFrame
        One row per period with performance metrics.
    """
    if periods is None:
        cfg = load_config("evaluation")
        periods = cfg.get("stress_periods", {})

    rows = []
    for name, window in periods.items():
        start, end = window["start"], window["end"]
        subset = returns.loc[start:end]
        if len(subset) < 5:
            logger.warning("Stress period %s has only %d observations", name, len(subset))
            continue

        metrics = compute_all_metrics(subset)
        worst_day = float(subset.min())
        worst_date = subset.idxmin()

        rows.append({
            "period": name,
            "start": start,
            "end": end,
            "n_days": len(subset),
            "cumulative_return": float(np.exp(subset.sum()) - 1),
            "max_drawdown": metrics["max_drawdown"],
            "worst_day": worst_day,
            "worst_date": str(worst_date.date()) if hasattr(worst_date, "date") else str(worst_date),
            "volatility": metrics["volatility"],
            "sharpe": metrics["sharpe"],
        })

    result = pd.DataFrame(rows)
    logger.info("historical_stress_test: %d periods evaluated", len(result))
    return result


def drawdown_anatomy(
    returns: pd.Series,
    top_n: int = 5,
) -> list[dict]:
    """Identify and characterise the worst drawdowns.

    Parameters
    ----------
    returns : pd.Series
        Daily log returns with DatetimeIndex.
    top_n : int
        Number of worst drawdowns to return.

    Returns
    -------
    list[dict]
        Each dict has: peak_date, trough_date, recovery_date,
        depth, duration_days, recovery_days.
    """
    cum = np.exp(returns.cumsum())
    running_max = cum.cummax()
    dd = cum / running_max - 1

    drawdowns = []
    in_dd = False
    peak_date = None
    trough_val = 0.0
    trough_date = None

    for dt, val in dd.items():
        if val < 0 and not in_dd:
            in_dd = True
            peak_idx = running_max.loc[:dt].idxmax()
            peak_date = peak_idx
            trough_val = val
            trough_date = dt
        elif val < 0 and in_dd:
            if val < trough_val:
                trough_val = val
                trough_date = dt
        elif val >= 0 and in_dd:
            in_dd = False
            duration = (trough_date - peak_date).days if peak_date else 0
            recovery = (dt - trough_date).days if trough_date else 0
            drawdowns.append({
                "peak_date": str(peak_date.date()) if hasattr(peak_date, "date") else str(peak_date),
                "trough_date": str(trough_date.date()) if hasattr(trough_date, "date") else str(trough_date),
                "recovery_date": str(dt.date()) if hasattr(dt, "date") else str(dt),
                "depth": trough_val,
                "duration_days": duration,
                "recovery_days": recovery,
            })

    if in_dd and peak_date is not None:
        duration = (trough_date - peak_date).days if trough_date else 0
        drawdowns.append({
            "peak_date": str(peak_date.date()) if hasattr(peak_date, "date") else str(peak_date),
            "trough_date": str(trough_date.date()) if hasattr(trough_date, "date") else str(trough_date),
            "recovery_date": None,
            "depth": trough_val,
            "duration_days": duration,
            "recovery_days": None,
        })

    drawdowns.sort(key=lambda x: x["depth"])
    result = drawdowns[:top_n]
    logger.info("drawdown_anatomy: found %d drawdowns, returning top %d", len(drawdowns), len(result))
    return result
