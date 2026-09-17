"""Features for the dark-zone interpolation model: one row per (hex, day).

Two kinds, deliberately kept apart:

STATION-INDEPENDENT - precomputed and cached here, safe to reuse anywhere:
    satellite  no2_trop, aai, co, so2                (data/satellite/features_r7)
    wind       wind_speed, wind_dir_sin/cos          (ERA5-Land u10/v10, same source)
    fire       fire_count, frp_sum                   (this hex, data/fires/hex_daily_r7)
               fires_150km, frp_150km,
               upwind_fires_150km, upwind_frp_150km  (res-5 parent, wind-weighted)
    calendar   doy_sin, doy_cos, month, weekday
    geo        lat, lon

STATION-DERIVED - never precomputed, always built from the REMAINING stations at
fit/predict time (model/train.py), because the held-out station must not appear in
its own features:
    neighbour_idw, neighbour_nearest, dist_nearest_km, n_reporting, neighbour_spread

The same rule kills data/grid's dist_km as a feature: it was computed with every
station, including the one being hidden.

Usage:
    python -m model.features                      # build/refresh the station-hex table, report coverage
    python -m model.features --rebuild
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[1]
SATELLITE_DIR = ROOT / "data" / "satellite" / "features_r7"
FIRE_HEX = ROOT / "data" / "fires" / "hex_daily_r7.parquet"
FIRE_REGION = ROOT / "data" / "fires" / "upwind_daily_r5.parquet"
GRID_FILE = ROOT / "data" / "grid" / "grid_r7_api.parquet"
STATIONS_FILE = ROOT / "data" / "grid" / "stations.parquet"
OUT_DIR = ROOT / "data" / "features"
STATION_HEX_TABLE = OUT_DIR / "station_hex_days.parquet"

SATELLITE_BANDS = ["no2_trop", "aai", "co", "so2"]
FIRE_HEX_COLS = ["fire_count", "frp_sum"]
FIRE_REGION_COLS = ["fires_150km", "frp_150km", "upwind_fires_150km", "upwind_frp_150km"]
STATION_FEATURES = ["neighbour_idw", "neighbour_nearest", "dist_nearest_km", "n_reporting", "neighbour_spread"]
EARTH_RADIUS_KM = 6371.0088

log = logging.getLogger("model.features")


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# --------------------------------------------------------------------------- station-independent


def satellite_frame(hexes: set[str] | None, start: date | None, end: date | None) -> pd.DataFrame:
    """Daily satellite + wind values per hex. Filtering to station hexes keeps this small."""
    dataset = ds.dataset(SATELLITE_DIR, format="parquet", partitioning="hive")
    columns = ["h3", "date"] + SATELLITE_BANDS + ["u10", "v10"]
    filt = None
    if hexes:
        filt = ds.field("h3").isin(list(hexes))
    if start:
        f = ds.field("date") >= pd.Timestamp(start).to_pydatetime().date()
        filt = f if filt is None else filt & f
    if end:
        f = ds.field("date") <= pd.Timestamp(end).to_pydatetime().date()
        filt = f if filt is None else filt & f
    df = dataset.to_table(columns=columns, filter=filt).to_pandas()
    df["date"] = pd.to_datetime(df["date"])

    # Wind as speed + direction. Direction is circular, so sin/cos, never degrees.
    speed = np.hypot(df["u10"], df["v10"])
    df["wind_speed"] = speed.astype("float32")
    with np.errstate(invalid="ignore", divide="ignore"):
        df["wind_dir_sin"] = (df["v10"] / speed).astype("float32")
        df["wind_dir_cos"] = (df["u10"] / speed).astype("float32")
    return df.drop(columns=["u10", "v10"])


def fire_frame(hexes: set[str] | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    own = pd.read_parquet(FIRE_HEX)
    region = pd.read_parquet(FIRE_REGION)
    if hexes:
        own = own[own["h3"].isin(hexes)]
        parents = {h3.cell_to_parent(c, 5) for c in hexes}
        region = region[region["h3_r5"].isin(parents)]
    own["date"] = pd.to_datetime(own["date"])
    region["date"] = pd.to_datetime(region["date"])
    return own, region


def hex_day_features(hexes: set[str], start: date | None = None, end: date | None = None) -> pd.DataFrame:
    """Every station-independent feature, for the given hexes and date range."""
    df = satellite_frame(hexes, start, end)
    own, region = fire_frame(hexes)
    df = df.merge(own, on=["h3", "date"], how="left")
    df[FIRE_HEX_COLS] = df[FIRE_HEX_COLS].fillna(0)      # no detection means no fire, not missing
    df["h3_r5"] = [h3.cell_to_parent(c, 5) for c in df["h3"]]
    df = df.merge(region, on=["h3_r5", "date"], how="left").drop(columns=["h3_r5"])
    df[["fires_150km", "frp_150km"]] = df[["fires_150km", "frp_150km"]].fillna(0)
    # No fire within 150 km means no upwind fire either - zero, not missing. Only a genuine
    # wind gap (ERA5 has no value over sea/coast) leaves upwind_* NaN.
    no_fire = df["fires_150km"] == 0
    for col in ("upwind_fires_150km", "upwind_frp_150km"):
        df.loc[no_fire & df[col].isna(), col] = 0.0

    doy = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25).astype("float32")
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25).astype("float32")
    df["month"] = df["date"].dt.month.astype("int8")
    df["weekday"] = df["date"].dt.weekday.astype("int8")

    latlng = np.array([h3.cell_to_latlng(c) for c in df["h3"]])
    df["lat"], df["lon"] = latlng[:, 0].astype("float32"), latlng[:, 1].astype("float32")
    return df


def build_station_hex_table(rebuild: bool = False) -> pd.DataFrame:
    """Station-independent features for every hex that contains a station (the training set)."""
    if STATION_HEX_TABLE.exists() and not rebuild:
        return pd.read_parquet(STATION_HEX_TABLE)
    stations = pd.read_parquet(STATIONS_FILE, columns=["station_id", "h3"])
    hexes = set(stations["h3"])
    log.info("building station-hex features for %d hexes (this scans the satellite dataset once)", len(hexes))
    df = hex_day_features(hexes)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(STATION_HEX_TABLE, index=False, compression="zstd")
    log.info("wrote %s (%d rows)", STATION_HEX_TABLE.relative_to(ROOT), len(df))
    return df


# --------------------------------------------------------------------------- station-derived


def neighbour_features(values: pd.DataFrame, stations: pd.DataFrame, lat: float, lon: float,
                       periods: pd.Index, k: int = 8, power: float = 2.0) -> pd.DataFrame:
    """Features from the REMAINING stations only: what the neighbours said, and how far they are.

    values:   period x station_id matrix of observations (already excluding the hidden station)
    stations: index station_id, columns hex_lat / hex_lon (already excluding the hidden station)
    """
    dist = haversine_km(lat, lon, stations["hex_lat"], stations["hex_lon"])
    order = np.argsort(dist)
    cols = stations.index[order]
    arr = values.reindex(index=periods, columns=cols).to_numpy(dtype=float)
    d = dist[order]

    valid = ~np.isnan(arr)
    rank = np.cumsum(valid, axis=1)
    use = valid & (rank <= k)                      # k nearest that actually reported that day
    weights = np.where(use, 1.0 / np.maximum(d, 0.5) ** power, 0.0)
    denom = weights.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        idw = (np.where(use, arr, 0.0) * weights).sum(axis=1) / denom
    idw[denom == 0] = np.nan

    first = valid.argmax(axis=1)
    any_valid = valid.any(axis=1)
    nearest = np.where(any_valid, arr[np.arange(len(arr)), first], np.nan)
    nearest_km = np.where(any_valid, d[first], np.nan)

    with np.errstate(invalid="ignore"):
        spread = np.nanstd(np.where(use, arr, np.nan), axis=1)

    return pd.DataFrame({"neighbour_idw": idw, "neighbour_nearest": nearest,
                         "dist_nearest_km": nearest_km, "n_reporting": valid.sum(axis=1).astype("int16"),
                         "neighbour_spread": spread}, index=periods)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuild", action="store_true", help="rescan the satellite dataset")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df = build_station_hex_table(args.rebuild)
    cols = SATELLITE_BANDS + ["wind_speed"] + FIRE_HEX_COLS + FIRE_REGION_COLS
    print(f"\nstation-hex feature table: {len(df):,} rows, {df['h3'].nunique()} hexes, "
          f"{df['date'].min():%Y-%m-%d} -> {df['date'].max():%Y-%m-%d}")
    print(f"\n{'feature':<22}{'non-null %':>12}{'median':>14}{'p95':>14}")
    for c in cols:
        print(f"  {c:<20}{100 * df[c].notna().mean():>11.1f}{df[c].median():>14.4g}{df[c].quantile(0.95):>14.4g}")
    print("\nstation-derived features are NOT in this table by design: "
          f"{', '.join(STATION_FEATURES)} are built per fold in model/train.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
