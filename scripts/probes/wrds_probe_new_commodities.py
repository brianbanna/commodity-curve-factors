"""Probe WRDS Datastream for additional commodity contracts.

Searches dsfutcontr.contrname for each candidate commodity and reports
the contrcode, contract count, and date coverage.

Usage:
    conda activate curve-factors
    python scripts/probes/wrds_probe_new_commodities.py

Requires: WRDS credentials in .env (WRDS_USERNAME / WRDS_PASSWORD).
The script will trigger Duo 2FA on first connection.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("wrds_probe_new_commodities")

# ---------------------------------------------------------------------------
# Candidates: symbol -> list of UPPER-CASE search terms (tried in order;
# stops at first term that returns at least one row).
# ---------------------------------------------------------------------------
CANDIDATES: dict[str, list[str]] = {
    "PL": ["PLATINUM"],
    "PA": ["PALLADIUM"],
    "CT": ["COTTON"],
    "LC": ["LIVE CATTLE", "CATTLE"],
    "LH": ["LEAN HOGS", "HOGS"],
    "FC": ["FEEDER CATTLE"],
    "ZL": ["SOYBEAN OIL"],
    "ZM": ["SOYBEAN MEAL"],
    "LB": ["LUMBER", "RANDOM LENGTH"],
}

START_DATE = "2003-01-01"
OUT_PATH = Path(__file__).parent / "new_commodities_probe.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def q(engine, sql: str) -> pd.DataFrame:
    with engine.connect() as c:
        return pd.read_sql_query(text(sql), c)


def search_contrname(engine, term: str) -> pd.DataFrame:
    """Return top-10 (contrcode, contrname, n_contracts) matching LIKE term."""
    sql = f"""
        SELECT contrcode, contrname, COUNT(*) AS n_contracts
        FROM tr_ds_fut.dsfutcontr
        WHERE UPPER(contrname) LIKE '%{term}%'
        GROUP BY contrcode, contrname
        ORDER BY n_contracts DESC
        LIMIT 10
    """
    return q(engine, sql)


def get_coverage(engine, contrcode: int) -> dict:
    """Return date-range and row-count stats for a contrcode since START_DATE."""
    sql = f"""
        SELECT
            MIN(v.date_)                  AS start_date,
            MAX(v.date_)                  AS end_date,
            COUNT(DISTINCT v.date_)       AS n_dates,
            COUNT(*)                      AS n_rows
        FROM tr_ds_fut.dsfutcontrval v
        JOIN tr_ds_fut.dsfutcontrinfo i ON v.futcode = i.futcode
        WHERE i.contrcode = {contrcode}
          AND v.date_ >= '{START_DATE}'
    """
    df = q(engine, sql)
    if df.empty:
        return {}
    row = df.iloc[0]
    return {
        "start_date": str(row["start_date"]),
        "end_date": str(row["end_date"]),
        "n_dates": int(row["n_dates"]) if row["n_dates"] is not None else 0,
        "n_rows": int(row["n_rows"]) if row["n_rows"] is not None else 0,
    }


def get_dscontrid_prefix(engine, contrcode: int) -> str:
    """Derive the Datastream mnemonic prefix for a contrcode (strips last 4 chars)."""
    sql = f"""
        SELECT DISTINCT LEFT(dsmnem, LENGTH(dsmnem) - 4) AS prefix
        FROM tr_ds_fut.dsfutcontrinfo
        WHERE contrcode = {contrcode}
        LIMIT 1
    """
    df = q(engine, sql)
    if df.empty or df.iloc[0]["prefix"] is None:
        return "UNKNOWN"
    return str(df.iloc[0]["prefix"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    username = os.environ.get("WRDS_USERNAME")
    if not username:
        logger.error("WRDS_USERNAME not set in .env")
        return 1

    import wrds

    logger.info("Connecting to WRDS (Duo 2FA may be required)...")
    conn = wrds.Connection(wrds_username=username)
    engine = conn.engine
    logger.info("Connected.")

    results: dict[str, dict] = {}
    not_found: list[str] = []

    try:
        for symbol, search_terms in CANDIDATES.items():
            logger.info("--- %s ---", symbol)
            matched = False

            for term in search_terms:
                logger.info("  Searching contrname LIKE '%%%s%%'...", term)
                df = search_contrname(engine, term)

                if df.empty:
                    logger.info("  No results for '%s'", term)
                    continue

                logger.info("  Found %d candidate row(s) for '%s':", len(df), term)
                for _, row in df.iterrows():
                    logger.info(
                        "    contrcode=%-6s  contracts=%-4s  name=%s",
                        int(row["contrcode"]),
                        int(row["n_contracts"]),
                        row["contrname"],
                    )

                # Pick the best match: most n_contracts (already sorted DESC)
                best = df.iloc[0]
                contrcode = int(best["contrcode"])
                contrname = str(best["contrname"])

                logger.info("  Selected contrcode=%d (%s)", contrcode, contrname)

                # Date coverage
                cov = get_coverage(engine, contrcode)
                if not cov:
                    logger.warning("  No coverage data for contrcode=%d — skipping", contrcode)
                    continue

                logger.info(
                    "  Coverage: %s → %s  (%d dates, %d rows)",
                    cov["start_date"],
                    cov["end_date"],
                    cov["n_dates"],
                    cov["n_rows"],
                )

                # Datastream mnemonic prefix
                dscontrid = get_dscontrid_prefix(engine, contrcode)
                logger.info("  dscontrid prefix: %s", dscontrid)

                results[symbol] = {
                    "contrcode": contrcode,
                    "contrname": contrname,
                    "dscontrid": dscontrid,
                    "search_term_matched": term,
                    "n_contracts": int(best["n_contracts"]),
                    **cov,
                }
                matched = True
                break  # Stop at first search term with results

            if not matched:
                logger.warning("  %s: NOT FOUND in WRDS Datastream", symbol)
                not_found.append(symbol)

        # ------------------------------------------------------------------
        # Save JSON
        # ------------------------------------------------------------------
        OUT_PATH.write_text(json.dumps(results, indent=2))
        logger.info("")
        logger.info("Saved probe results → %s", OUT_PATH)

        # ------------------------------------------------------------------
        # Summary table
        # ------------------------------------------------------------------
        logger.info("")
        logger.info("=" * 70)
        logger.info("SUMMARY  (%d/%d found)", len(results), len(CANDIDATES))
        logger.info("=" * 70)
        logger.info("%-4s  %-8s  %-12s  %-12s  %-6s  %s", "SYM", "CONTRCODE", "START", "END", "DATES", "NAME")
        logger.info("-" * 70)
        for sym, info in results.items():
            logger.info(
                "%-4s  %-8d  %-12s  %-12s  %-6d  %s",
                sym,
                info["contrcode"],
                info["start_date"],
                info["end_date"],
                info["n_dates"],
                info["contrname"],
            )
        if not_found:
            logger.info("")
            logger.info("NOT FOUND: %s", ", ".join(not_found))

        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Paste the probe output to Claude for review.")
        logger.info("  2. Claude will update configs/universe.yaml with the confirmed contrcodes.")
        logger.info("  3. Run: python scripts/probes/wrds_download_new.py")

        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
