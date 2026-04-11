"""WRDS probe v10 — writes findings to wrds_findings.json (no stdout spam).

Run:
    conda run -n curve-factors python scripts/probes/wrds_bulk_download.py

When it finishes, inspect the file:
    scripts/probes/wrds_findings.json

This avoids terminal output truncation. Claude reads the JSON directly.

What it does in one connection:
    1. Search dsfutcontr for every commodity-relevant name keyword
       (saves ALL matching rows to findings.json)
    2. For each strong candidate contrcode, validate data coverage
       in dsfutcontrinfo + dsfutcontrval (2005-2024)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = REPO_ROOT / "scripts" / "probes" / "wrds_findings.json"

# Confirmed from v8 output — high-confidence candidates to validate
CONFIRMED: dict[str, int] = {
    "CL": 1986,  # NCL CRUDE OIL (LIGHT SWEET)
    "HO": 2029,  # NHO HEATING OIL (NEW YORK)
    "RB": 2091,  # NRB GASOLINE RBOB
    "SI": 2108,  # NSL SILVER (5000 OZ)
    "HG": 2026,  # NHG COPPER (HIGH GRADE)
    "ZC": 288,  # CC. CORN
    "ZS": 462,  # CZS CBT Soybean Electronic
    "ZW": 464,  # CZW CBT Wheat Electronic
    "KC": 2038,  # NKC COFFEE 'C'
    "CC": 1980,  # NCC COCOA
}

# Commodities where the correct contrcode is still ambiguous
# v9 will search for these and save all matches
AMBIGUOUS_SEARCHES: dict[str, list[str]] = {
    "GC": ["GOLD 100", "GOLD (100", "COMEX GOLD"],
    "NG": ["NATURAL GAS", "HENRY HUB"],
    "SB": ["SUGAR"],
}

# Also worth sanity-checking alternative ZC/ZS/ZW codes
ADDITIONAL_CANDIDATES: dict[str, list[int]] = {
    "ZC": [288, 3247, 449, 2695],  # CC. / CCF / CZC / DC.
    "ZS": [462, 3376],  # CZS / CSY
    "ZW": [464],  # CZW
}


def _default(o):
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    if isinstance(o, Decimal):
        return float(o)
    if pd.isna(o):
        return None
    return str(o)


def q(engine, sql: str) -> pd.DataFrame:
    with engine.connect() as c:
        return pd.read_sql_query(text(sql), c)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S"
    )
    log = logging.getLogger("wrds_probe_v10")

    load_dotenv(REPO_ROOT / ".env")
    username = os.environ["WRDS_USERNAME"]

    import wrds

    log.info("Connecting...")
    conn = wrds.Connection(wrds_username=username)
    engine = conn.engine
    log.info("Connected.")

    findings: dict = {
        "generated_at": datetime.utcnow().isoformat(),
        "confirmed_candidates": CONFIRMED,
        "ambiguous_searches": {},
        "additional_candidate_validation": {},
        "confirmed_validation": {},
    }

    try:
        # --- Ambiguous: fetch ALL matching rows for GC / NG / SB ---
        for sym, terms in AMBIGUOUS_SEARCHES.items():
            or_clauses = " OR ".join(f"UPPER(contrname) LIKE UPPER('%{t}%')" for t in terms)
            df = q(
                engine,
                f"""
                SELECT contrcode, dscontrid, contrtypecode, contrname, exchtickersymb
                FROM tr_ds_fut.dsfutcontr
                WHERE ({or_clauses})
                ORDER BY contrname, contrcode
                LIMIT 100
            """,
            )
            findings["ambiguous_searches"][sym] = df.to_dict(orient="records")
            log.info("  %s search: %d rows", sym, len(df))

        # --- Validate data coverage for each confirmed candidate ---
        all_to_validate = dict(CONFIRMED)
        for sym, codes in ADDITIONAL_CANDIDATES.items():
            for code in codes:
                all_to_validate[f"{sym}_{code}"] = code

        for label, code in all_to_validate.items():
            try:
                info = q(
                    engine,
                    f"""
                    SELECT
                        COUNT(*) AS n_contracts,
                        MIN(startdate)::text AS first_start,
                        MAX(lasttrddate)::text AS last_expiry,
                        MIN(dsmnem) AS sample_mnem,
                        STRING_AGG(DISTINCT ldb, ',') AS ldbs
                    FROM tr_ds_fut.dsfutcontrinfo
                    WHERE contrcode = {code}
                """,
                )
                vals = q(
                    engine,
                    f"""
                    SELECT
                        COUNT(*) AS n_rows,
                        MIN(date_)::text AS first_date,
                        MAX(date_)::text AS last_date
                    FROM tr_ds_fut.dsfutcontrval v
                    JOIN tr_ds_fut.dsfutcontrinfo i USING (futcode)
                    WHERE i.contrcode = {code}
                """,
                )
                findings["confirmed_validation"][label] = {
                    "contrcode": code,
                    "n_contracts": int(info.iloc[0]["n_contracts"]),
                    "first_start": info.iloc[0]["first_start"],
                    "last_expiry": info.iloc[0]["last_expiry"],
                    "sample_mnem": info.iloc[0]["sample_mnem"],
                    "ldbs": info.iloc[0]["ldbs"],
                    "n_price_rows": int(vals.iloc[0]["n_rows"]),
                    "first_price_date": vals.iloc[0]["first_date"],
                    "last_price_date": vals.iloc[0]["last_date"],
                }
                log.info(
                    "  %s (code=%d): %d contracts, %d rows",
                    label,
                    code,
                    findings["confirmed_validation"][label]["n_contracts"],
                    findings["confirmed_validation"][label]["n_price_rows"],
                )
            except Exception as exc:  # noqa: BLE001
                findings["confirmed_validation"][label] = {"error": str(exc)[:300]}
                log.warning("  %s (code=%d): failed — %s", label, code, str(exc)[:100])

        OUTPUT.write_text(json.dumps(findings, indent=2, default=_default))
        log.info("")
        log.info("=== WROTE %s ===", OUTPUT.relative_to(REPO_ROOT))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
