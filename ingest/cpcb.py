"""National CPCB real-time AQI ingest from data.gov.in -> Parquet.

Dataset: "Real time Air Quality Index from various locations"
https://data.gov.in/resources/real-time-air-quality-index-various-locations

The API returns one record per station x pollutant. Values are CPCB
sub-index figures (min / max / avg over the last 24h).

Every API page is cached to data/raw/cpcb/<snapshot>/ before it is parsed.
Re-runs replay the latest complete snapshot with zero network calls unless
--refresh is passed. An interrupted pull resumes from the pages already on
disk.

Usage:
    python -m ingest.cpcb                    # replay latest cached snapshot, or pull if none
    python -m ingest.cpcb --refresh          # force a fresh national pull
    python -m ingest.cpcb --snapshot 20260913T1400Z
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import dotenv_values

RESOURCE_ID = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
API_URL = f"https://api.data.gov.in/resource/{RESOURCE_ID}"

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "cpcb"
OUT_DIR = ROOT / "data" / "cpcb"
COMPLETE_MARKER = "_COMPLETE"

PAGE_SIZE = 1000
MAX_ATTEMPTS = 8
TIMEOUT = (20, 180)  # (connect, read) - data.gov.in is routinely slow
RETRY_STATUS = {429, 500, 502, 503, 504}

# The dataset has shipped both naming schemes over time.
FIELD_RENAMES = {
    "pollutant_min": "min_value",
    "pollutant_max": "max_value",
    "pollutant_avg": "avg_value",
}

POLLUTANT_COLUMNS = {
    "PM2.5": "pm25",
    "PM10": "pm10",
    "NO2": "no2",
    "SO2": "so2",
    "CO": "co",
    "OZONE": "o3",
    "NH3": "nh3",
}

# CPCB National AQI bands.
AQI_BANDS = [
    (50, "Good"),
    (100, "Satisfactory"),
    (200, "Moderate"),
    (300, "Poor"),
    (400, "Very Poor"),
    (np.inf, "Severe"),
]

log = logging.getLogger("ingest.cpcb")


def load_api_key() -> str:
    key = os.environ.get("DATA_GOV_IN_API_KEY") or dotenv_values(ROOT / ".env").get("DATA_GOV_IN_API_KEY")
    if not key:
        sys.exit("DATA_GOV_IN_API_KEY is not set. Copy .env.example to .env and fill it in.")
    return key


# --------------------------------------------------------------------------- fetch


def _redact(text: str, key: str) -> str:
    return text.replace(key, "<REDACTED>")


def fetch_page(session: requests.Session, key: str, offset: int, limit: int) -> dict:
    params = {"api-key": key, "format": "json", "offset": offset, "limit": limit}
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = session.get(API_URL, params=params, timeout=TIMEOUT)
            if resp.status_code in RETRY_STATUS:
                raise requests.HTTPError(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            payload = resp.json()
            if "records" not in payload:
                raise ValueError(f"unexpected payload: {_redact(resp.text[:300], key)}")
            return payload
        except (requests.RequestException, ValueError) as exc:
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(
                    f"offset={offset}: giving up after {MAX_ATTEMPTS} attempts: {_redact(str(exc), key)}"
                ) from None
            wait = min(2**attempt, 120)
            log.warning("offset=%d attempt %d/%d failed (%s); retrying in %ds",
                        offset, attempt, MAX_ATTEMPTS, _redact(str(exc), key), wait)
            time.sleep(wait)
    raise AssertionError("unreachable")


def pull_snapshot(snapshot_dir: Path, key: str, page_size: int = PAGE_SIZE) -> None:
    """Download every page into snapshot_dir, skipping pages already cached."""
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    offset, total = 0, None

    while total is None or offset < total:
        page_path = snapshot_dir / f"page_{offset:06d}.json"
        if page_path.exists():
            payload = json.loads(page_path.read_text(encoding="utf-8"))
            log.info("offset=%d cached (%d records)", offset, len(payload["records"]))
        else:
            payload = fetch_page(session, key, offset, page_size)
            page_path.write_text(_redact(json.dumps(payload, ensure_ascii=False), key), encoding="utf-8")
            log.info("offset=%d fetched (%d records, total=%s)", offset, len(payload["records"]), payload.get("total"))

        total = int(payload.get("total") or 0)
        if not payload["records"]:
            break
        offset += page_size

    (snapshot_dir / COMPLETE_MARKER).write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
    log.info("snapshot %s complete: %d records reported", snapshot_dir.name, total)


def latest_snapshot(complete_only: bool) -> Path | None:
    if not RAW_DIR.exists():
        return None
    dirs = sorted(d for d in RAW_DIR.iterdir() if d.is_dir())
    if complete_only:
        dirs = [d for d in dirs if (d / COMPLETE_MARKER).exists()]
    return dirs[-1] if dirs else None


# --------------------------------------------------------------------------- transform


def _station_id(state: str, city: str, station: str) -> str:
    slug = "|".join([state, city, station]).lower()
    return re.sub(r"[^a-z0-9|]+", "-", slug).strip("-")


def load_readings(snapshot_dir: Path) -> pd.DataFrame:
    """Long format: one row per station x pollutant."""
    records = []
    for page in sorted(snapshot_dir.glob("page_*.json")):
        records.extend(json.loads(page.read_text(encoding="utf-8"))["records"])
    if not records:
        raise RuntimeError(f"no records in {snapshot_dir}")

    df = pd.DataFrame.from_records(records)
    for old, new in FIELD_RENAMES.items():
        if old in df.columns:
            df[new] = df[new].fillna(df[old]) if new in df.columns else df[old]
            df = df.drop(columns=old)
    for col in ("state", "city", "station", "pollutant_id"):
        df[col] = df[col].astype(str).str.strip()
    for col in ("latitude", "longitude", "min_value", "max_value", "avg_value"):
        df[col] = pd.to_numeric(df[col].replace({"NA": None, "": None}), errors="coerce")

    # last_update is IST, e.g. "13-09-2026 14:00:00"
    df["last_update"] = (
        pd.to_datetime(df["last_update"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
        .dt.tz_localize("Asia/Kolkata")
        .dt.tz_convert("UTC")
    )
    df["station_id"] = [_station_id(s, c, n) for s, c, n in zip(df["state"], df["city"], df["station"])]
    df["snapshot"] = snapshot_dir.name

    df = df.drop_duplicates(subset=["station_id", "pollutant_id", "last_update"], keep="last")
    cols = ["station_id", "state", "city", "station", "latitude", "longitude", "last_update",
            "pollutant_id", "min_value", "max_value", "avg_value", "snapshot"]
    return df[cols].reset_index(drop=True)


def aqi_category(aqi: float) -> str | None:
    if pd.isna(aqi):
        return None
    return next(label for upper, label in AQI_BANDS if aqi <= upper)


def build_stations(readings: pd.DataFrame) -> pd.DataFrame:
    """Wide format: one row per station with per-pollutant sub-indices and overall AQI.

    Overall AQI follows the CPCB rule: max sub-index, valid only when at least
    3 pollutants report and one of them is PM2.5 or PM10.
    """
    meta = (
        readings.sort_values("last_update")
        .groupby("station_id")
        .agg(state=("state", "last"), city=("city", "last"), station=("station", "last"),
             latitude=("latitude", "last"), longitude=("longitude", "last"),
             last_update=("last_update", "max"), snapshot=("snapshot", "last"))
    )

    subidx = readings.pivot_table(index="station_id", columns="pollutant_id",
                                  values="avg_value", aggfunc="last")
    subidx = subidx.rename(columns=POLLUTANT_COLUMNS)
    pollutants = [c for c in POLLUTANT_COLUMNS.values() if c in subidx.columns]
    subidx = subidx.reindex(columns=list(POLLUTANT_COLUMNS.values()))

    stations = meta.join(subidx)
    values = stations[pollutants]
    stations["n_pollutants"] = values.notna().sum(axis=1)
    stations["aqi"] = values.max(axis=1)
    stations["dominant_pollutant"] = values.idxmax(axis=1, skipna=True).where(stations["aqi"].notna())
    has_pm = stations[["pm25", "pm10"]].notna().any(axis=1)
    stations["aqi_valid"] = (stations["n_pollutants"] >= 3) & has_pm
    stations["aqi_category"] = stations["aqi"].map(aqi_category)

    return stations.reset_index()


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow", compression="zstd")
    log.info("wrote %s (%d rows)", path.relative_to(ROOT), len(df))


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh", action="store_true", help="ignore cache and pull a new snapshot")
    parser.add_argument("--snapshot", help="replay a specific cached snapshot directory name")
    parser.add_argument("--page-size", type=int, default=PAGE_SIZE)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.snapshot:
        snapshot_dir = RAW_DIR / args.snapshot
        if not (snapshot_dir / COMPLETE_MARKER).exists():
            sys.exit(f"snapshot {args.snapshot} is missing or incomplete")
    elif not args.refresh and (snapshot_dir := latest_snapshot(complete_only=True)):
        log.info("replaying cached snapshot %s (use --refresh to pull live)", snapshot_dir.name)
    else:
        partial = latest_snapshot(complete_only=False)
        if partial and not (partial / COMPLETE_MARKER).exists():
            snapshot_dir = partial
            log.info("resuming partial snapshot %s", snapshot_dir.name)
        else:
            snapshot_dir = RAW_DIR / datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
        pull_snapshot(snapshot_dir, load_api_key(), args.page_size)

    readings = load_readings(snapshot_dir)
    stations = build_stations(readings)

    write_parquet(readings, OUT_DIR / f"readings_{snapshot_dir.name}.parquet")
    write_parquet(stations, OUT_DIR / f"stations_{snapshot_dir.name}.parquet")
    write_parquet(stations, OUT_DIR / "stations_latest.parquet")

    log.info("%d stations across %d states, %d with valid AQI",
             len(stations), stations["state"].nunique(), int(stations["aqi_valid"].sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
