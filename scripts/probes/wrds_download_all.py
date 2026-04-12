"""Final WRDS bulk download for Task 2.5.

Reads wrds_contrcode per commodity from configs/universe.yaml and pulls every
individual contract's daily time series from tr_ds_fut.dsfutcontrval joined
with dsfutcontrinfo metadata. Writes one Parquet per commodity to
data/raw/futures/contracts/{SYM}/all_contracts.parquet plus a summary JSON.

One Duo Push expected (unless the trusted session from the schema probe is
still warm, in which case zero Duo prompts).

Run in your terminal:
    conda run -n curve-factors python scripts/probes/wrds_download_all.py

Idempotent: skips any commodity whose Parquet already exists. Safe to re-run.
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

from commodity_curve_factors.utils.config import load_config
from commodity_curve_factors.utils.paths import DATA_RAW, PROJECT_ROOT

CONTRACTS_DIR = DATA_RAW / "futures" / "contracts"
SUMMARY_PATH = CONTRACTS_DIR / "_download_summary.json"

# Pull window: earlier than our 2005-2024 backtest window so we have data for
# expanding-window z-score warmup (need 1 year lookback before 2005-01-01).
START_DATE = "2003-01-01"
END_DATE = "2024-12-31"


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


def download_commodity(engine, symbol: str, contrcode: int, log) -> pd.DataFrame:
    """Pull every contract's daily time series for one commodity.

    Joins dsfutcontrinfo (metadata: contract mnemonic, expiry dates) with
    dsfutcontrval (time series: OHLC, settlement, volume, open interest).
    Returns a long-format DataFrame.
    """
    sql = f"""
        SELECT
            i.futcode,
            i.dsmnem,
            i.contrdate,
            i.startdate,
            i.lasttrddate,
            i.sttlmntdate,
            i.isocurrcode,
            i.ldb,
            v.date_        AS trade_date,
            v.open_        AS open_price,
            v.high         AS high_price,
            v.low          AS low_price,
            v.settlement,
            v.volume,
            v.openinterest
        FROM tr_ds_fut.dsfutcontrinfo i
        JOIN tr_ds_fut.dsfutcontrval v USING (futcode)
        WHERE i.contrcode = {contrcode}
          AND v.date_ >= '{START_DATE}'
          AND v.date_ <= '{END_DATE}'
        ORDER BY i.futcode, v.date_
    """
    log.info("  %s: querying contrcode=%d...", symbol, contrcode)
    df = q(engine, sql)
    log.info(
        "  %s: got %d rows across %d contracts (%s → %s)",
        symbol,
        len(df),
        df["futcode"].nunique() if not df.empty else 0,
        df["trade_date"].min() if not df.empty else "—",
        df["trade_date"].max() if not df.empty else "—",
    )
    return df


def save_commodity(symbol: str, df: pd.DataFrame, log) -> Path:
    out_dir = CONTRACTS_DIR / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "all_contracts.parquet"
    df.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
    size_mb = out_path.stat().st_size / 1024 / 1024
    log.info("  %s: saved %s (%.1f MB)", symbol, out_path.relative_to(PROJECT_ROOT), size_mb)
    return out_path


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("wrds_download_all")

    load_dotenv(PROJECT_ROOT / ".env")
    username = os.environ.get("WRDS_USERNAME")
    if not username:
        log.error("WRDS_USERNAME not set in .env")
        return 1

    universe = load_config("universe")
    commodities = universe["commodities"]
    log.info("Universe has %d commodities", len(commodities))

    import wrds

    log.info("Connecting via wrds.Connection (one Duo Push if session not warm)...")
    conn = wrds.Connection(wrds_username=username)
    engine = conn.engine
    log.info("Connected.")

    summary: dict = {
        "generated_at": datetime.utcnow().isoformat(),
        "start_date": START_DATE,
        "end_date": END_DATE,
        "commodities": {},
    }

    try:
        for symbol, spec in commodities.items():
            contrcode = spec["wrds_contrcode"]
            out_path = CONTRACTS_DIR / symbol / "all_contracts.parquet"
            if out_path.exists():
                log.info("  %s: cached at %s, skipping", symbol, out_path.relative_to(PROJECT_ROOT))
                df = pd.read_parquet(out_path)
                summary["commodities"][symbol] = {
                    "contrcode": int(contrcode),
                    "status": "cached",
                    "n_rows": len(df),
                    "n_contracts": int(df["futcode"].nunique()) if not df.empty else 0,
                    "first_date": str(df["trade_date"].min()) if not df.empty else None,
                    "last_date": str(df["trade_date"].max()) if not df.empty else None,
                }
                continue

            try:
                df = download_commodity(engine, symbol, int(contrcode), log)
                if df.empty:
                    log.warning("  %s: empty result, not saving", symbol)
                    summary["commodities"][symbol] = {
                        "contrcode": int(contrcode),
                        "status": "empty",
                    }
                    continue
                save_commodity(symbol, df, log)
                summary["commodities"][symbol] = {
                    "contrcode": int(contrcode),
                    "status": "downloaded",
                    "n_rows": int(len(df)),
                    "n_contracts": int(df["futcode"].nunique()),
                    "first_date": str(df["trade_date"].min()),
                    "last_date": str(df["trade_date"].max()),
                }
            except Exception as exc:  # noqa: BLE001
                log.error("  %s: download failed — %s", symbol, str(exc)[:200])
                summary["commodities"][symbol] = {
                    "contrcode": int(contrcode),
                    "status": "failed",
                    "error": str(exc)[:300],
                }

        CONTRACTS_DIR.mkdir(parents=True, exist_ok=True)
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=_default))
        log.info("")
        log.info("=" * 60)
        log.info("Summary saved to %s", SUMMARY_PATH.relative_to(PROJECT_ROOT))

        n_ok = sum(
            1
            for v in summary["commodities"].values()
            if v.get("status") in ("downloaded", "cached")
        )
        log.info("%d/%d commodities downloaded or cached", n_ok, len(commodities))
        return 0 if n_ok == len(commodities) else 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
