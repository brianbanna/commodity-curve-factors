"""Download WRDS Datastream futures data for the 9 new candidate commodities.

Reads contrcodes from scripts/probes/new_commodities_probe.json (produced by
wrds_probe_new_commodities.py) and pulls every individual contract's daily time
series from tr_ds_fut.dsfutcontrinfo + dsfutcontrval.  Writes one Parquet per
commodity to data/raw/futures/contracts/{SYM}/all_contracts.parquet.

Idempotent: skips any commodity whose Parquet already exists.  Safe to re-run.

Usage:
    conda activate curve-factors
    python scripts/probes/wrds_download_new.py

Requires: WRDS credentials in .env (WRDS_USERNAME).
          Run wrds_probe_new_commodities.py first so the probe JSON exists.
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

CONTRACTS_DIR = PROJECT_ROOT / "data" / "raw" / "futures" / "contracts"
PROBE_PATH = Path(__file__).parent / "new_commodities_probe.json"
SUMMARY_PATH = CONTRACTS_DIR / "_download_summary_new.json"

# Same date window as wrds_download_all.py
START_DATE = "2003-01-01"
END_DATE = "2024-12-31"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("wrds_download_new")


# ---------------------------------------------------------------------------
# Helpers (mirrored from wrds_download_all.py)
# ---------------------------------------------------------------------------


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


def download_commodity(engine, symbol: str, contrcode: int) -> pd.DataFrame:
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


def save_commodity(symbol: str, df: pd.DataFrame) -> Path:
    out_dir = CONTRACTS_DIR / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "all_contracts.parquet"
    df.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
    size_mb = out_path.stat().st_size / 1024 / 1024
    log.info("  %s: saved %s (%.1f MB)", symbol, out_path.relative_to(PROJECT_ROOT), size_mb)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    username = os.environ.get("WRDS_USERNAME")
    if not username:
        log.error("WRDS_USERNAME not set in .env")
        return 1

    if not PROBE_PATH.exists():
        log.error(
            "Probe file not found: %s\nRun wrds_probe_new_commodities.py first.",
            PROBE_PATH,
        )
        return 1

    with open(PROBE_PATH) as f:
        probe: dict[str, dict] = json.load(f)

    if not probe:
        log.error("Probe file is empty — no commodities to download.")
        return 1

    log.info("Probe file has %d commodities: %s", len(probe), ", ".join(probe))

    import wrds

    log.info("Connecting to WRDS (Duo 2FA may be required)...")
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
        for symbol, info in probe.items():
            contrcode = int(info["contrcode"])
            out_path = CONTRACTS_DIR / symbol / "all_contracts.parquet"

            if out_path.exists():
                log.info("  %s: cached at %s, skipping", symbol, out_path.relative_to(PROJECT_ROOT))
                df = pd.read_parquet(out_path)
                summary["commodities"][symbol] = {
                    "contrcode": contrcode,
                    "contrname": info.get("contrname", ""),
                    "status": "cached",
                    "n_rows": len(df),
                    "n_contracts": int(df["futcode"].nunique()) if not df.empty else 0,
                    "first_date": str(df["trade_date"].min()) if not df.empty else None,
                    "last_date": str(df["trade_date"].max()) if not df.empty else None,
                }
                continue

            try:
                df = download_commodity(engine, symbol, contrcode)
                if df.empty:
                    log.warning("  %s: empty result, not saving", symbol)
                    summary["commodities"][symbol] = {
                        "contrcode": contrcode,
                        "contrname": info.get("contrname", ""),
                        "status": "empty",
                    }
                    continue
                save_commodity(symbol, df)
                summary["commodities"][symbol] = {
                    "contrcode": contrcode,
                    "contrname": info.get("contrname", ""),
                    "status": "downloaded",
                    "n_rows": int(len(df)),
                    "n_contracts": int(df["futcode"].nunique()),
                    "first_date": str(df["trade_date"].min()),
                    "last_date": str(df["trade_date"].max()),
                }
            except Exception as exc:  # noqa: BLE001
                log.error("  %s: download failed — %s", symbol, str(exc)[:200])
                summary["commodities"][symbol] = {
                    "contrcode": contrcode,
                    "contrname": info.get("contrname", ""),
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
        log.info("%d/%d commodities downloaded or cached", n_ok, len(probe))

        if n_ok < len(probe):
            failed = [
                sym
                for sym, v in summary["commodities"].items()
                if v.get("status") not in ("downloaded", "cached")
            ]
            log.warning("Not fully successful — check these: %s", ", ".join(failed))

        log.info("")
        log.info("Next: paste the summary to Claude so it can update configs/universe.yaml")

        return 0 if n_ok == len(probe) else 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
