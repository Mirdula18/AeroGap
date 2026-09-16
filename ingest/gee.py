"""Sentinel-5P columns (+ ERA5-Land wind) per H3 hex per day, via Google Earth Engine.

Export once, work locally (CLAUDE.md rule 4). Each day is pulled ONCE from Earth
Engine with ee.data.computePixels as a small float32 raster over India and cached
to data/satellite/raw/YYYY-MM-DD.npz. A cached day is never queried again.
Hex features are then sampled locally at H3 res-7 centroids.

Why rasters, not CSV: 623k hexes x 362 days x 6 bands is ~40 GB as CSV. At
0.05 deg (~5.5 km, about Sentinel-5P's native footprint) a day over India is a
600 x 640 grid; the whole year is ~1 GB compressed and holds the same values.

Bands (daily mean of all overpasses; clouds and gaps stay NaN):
    no2_trop  COPERNICUS/S5P/OFFL/L3_NO2     tropospheric_NO2_column_number_density  mol/m2
    aai       COPERNICUS/S5P/OFFL/L3_AER_AI  absorbing_aerosol_index                 unitless
    co        COPERNICUS/S5P/OFFL/L3_CO      CO_column_number_density                mol/m2
    so2       COPERNICUS/S5P/OFFL/L3_SO2     SO2_column_number_density               mol/m2
    u10, v10  ECMWF/ERA5_LAND/DAILY_AGGR     10 m wind components                    m/s (upwind fire weighting)

One-off setup (free, no billing account):
    pip install earthengine-api
    earthengine authenticate
    register a Google Cloud project for noncommercial Earth Engine use
    put its id in .env as EE_PROJECT=...

Usage:
    python -m ingest.gee                                   # corpus window, all bands, all hexes
    python -m ingest.gee --start 2025-11-01 --end 2025-11-07
    python -m ingest.gee --skip-download --hexes stations  # rebuild features from cached rasters only
Outputs: data/satellite/raw/*.npz, data/satellite/features_r7/month=YYYY-MM/part-0.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "satellite" / "raw"
OUT_DIR = ROOT / "data" / "satellite" / "features_r7"
GRID_FILE = ROOT / "data" / "grid" / "grid_r7_api.parquet"
STATIONS_FILE = ROOT / "data" / "grid" / "stations.parquet"

BBOX = (68.0, 6.0, 98.0, 38.0)  # west, south, east, north: the grid's extent with a margin
PIXEL_DEG = 0.05
NODATA = -9999.0
WORKERS = 6
MAX_ATTEMPTS = 6

BANDS = {
    "no2_trop": ("COPERNICUS/S5P/OFFL/L3_NO2", "tropospheric_NO2_column_number_density", "mol/m2"),
    "aai": ("COPERNICUS/S5P/OFFL/L3_AER_AI", "absorbing_aerosol_index", "unitless"),
    "co": ("COPERNICUS/S5P/OFFL/L3_CO", "CO_column_number_density", "mol/m2"),
    "so2": ("COPERNICUS/S5P/OFFL/L3_SO2", "SO2_column_number_density", "mol/m2"),
    "u10": ("ECMWF/ERA5_LAND/DAILY_AGGR", "u_component_of_wind_10m", "m/s"),
    "v10": ("ECMWF/ERA5_LAND/DAILY_AGGR", "v_component_of_wind_10m", "m/s"),
}

# The offline (OFFL) S5P products are the quality-controlled ones, but their GEE
# ingestion is patchy: on 2025-11-03 OFFL NO2 had 2 images, fully masked over India,
# while NRTI had 181 covering 66% of it. So: OFFL first, NRTI only where OFFL is masked.
NRTI_FALLBACK = {name: (cid.replace("/OFFL/", "/NRTI/"), band)
                 for name, (cid, band, _) in BANDS.items() if "/OFFL/" in cid}

SETUP = """Earth Engine is not set up. One-off, free, no billing:
  1. earthengine authenticate
  2. register a Cloud project for noncommercial use: https://code.earthengine.google.com/register
  3. add EE_PROJECT=<project-id> to .env"""

log = logging.getLogger("ingest.gee")


def corpus_window() -> tuple[date, date]:
    """Date range of the OpenAQ history corpus, so satellite features line up with it."""
    path = ROOT / "data" / "history" / "_summary.json"
    if path.exists():
        query = json.loads(path.read_text(encoding="utf-8")).get("query", {})
        try:
            return date.fromisoformat(query["start"]), date.fromisoformat(query["end"])
        except (KeyError, ValueError):
            pass
    today = date.today()
    return today - timedelta(days=365), today


# --------------------------------------------------------------------------- download


def init_ee():
    import ee

    project = os.environ.get("EE_PROJECT") or dotenv_values(ROOT / ".env").get("EE_PROJECT")
    if not project:
        sys.exit(SETUP)
    try:
        ee.Initialize(project=project)
    except Exception as exc:  # not authenticated, project not registered, ...
        sys.exit(f"Earth Engine initialisation failed: {exc}\n\n{SETUP}")
    return ee


def grid_spec(pixel_deg: float) -> dict:
    west, south, east, north = BBOX
    return {"west": west, "north": north, "pixel_deg": pixel_deg,
            "width": round((east - west) / pixel_deg), "height": round((north - south) / pixel_deg)}


def day_image(ee, day: date, bands: list[str]):
    start = ee.Date(day.isoformat())
    end = start.advance(1, "day")
    def daily_mean(collection_id: str, band: str):
        col = ee.ImageCollection(collection_id).filterDate(start, end).select(band)
        # A day with no overpass (or ERA5 not yet published) becomes an all-masked layer, not an error.
        return ee.Image(ee.Algorithms.If(col.size().gt(0), col.mean(),
                                         ee.Image.constant(NODATA).rename(band).selfMask()))

    layers = []
    for name in bands:
        collection_id, band, _ = BANDS[name]
        img = daily_mean(collection_id, band)
        if name in NRTI_FALLBACK:  # fill OFFL's masked pixels from the near-real-time product
            img = img.unmask(daily_mean(*NRTI_FALLBACK[name]))
        layers.append(img.rename(name).unmask(NODATA).toFloat())
    return ee.Image.cat(layers)


def fetch_day(ee, day: date, bands: list[str], spec: dict) -> Path:
    path = RAW_DIR / f"{day.isoformat()}.npz"
    grid = {
        "dimensions": {"width": spec["width"], "height": spec["height"]},
        "affineTransform": {"scaleX": spec["pixel_deg"], "shearX": 0, "translateX": spec["west"],
                            "shearY": 0, "scaleY": -spec["pixel_deg"], "translateY": spec["north"]},
        "crsCode": "EPSG:4326",
    }
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            arr = ee.data.computePixels({"expression": day_image(ee, day, bands),
                                         "fileFormat": "NUMPY_NDARRAY", "grid": grid})
            break
        except Exception as exc:  # quota and concurrency errors are transient
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(f"{day}: {exc}") from None
            time.sleep(min(2 ** attempt, 60))

    layers = {}
    for name in bands:
        a = np.asarray(arr[name], dtype=np.float32)
        a[a <= NODATA + 1] = np.nan
        layers[name] = a
    meta = dict(spec, date=day.isoformat(), bands=bands,
                sources={b: BANDS[b][0] + ":" + BANDS[b][1] for b in bands})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, meta=json.dumps(meta), **layers)
    tmp.replace(path)
    return path


def download(start: date, end: date, bands: list[str], pixel_deg: float, workers: int) -> None:
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    todo = [d for d in days if not (RAW_DIR / f"{d.isoformat()}.npz").exists()]
    log.info("%d days in range, %d cached, %d to fetch", len(days), len(days) - len(todo), len(todo))
    if not todo:
        return
    ee = init_ee()
    spec = grid_spec(pixel_deg)
    failed = 0
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(fetch_day, ee, d, bands, spec): d for d in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                fut.result()
            except Exception as exc:
                failed += 1
                log.warning("%s", exc)
            if n % 20 == 0 or n == len(todo):
                log.info("fetched %d/%d days (%d failed; re-run to retry)", n, len(todo), failed)


# --------------------------------------------------------------------------- local sampling


def load_day(day: date, bands: list[str] | None = None) -> tuple[dict, dict[str, np.ndarray]] | None:
    path = RAW_DIR / f"{day.isoformat()}.npz"
    if not path.exists():
        return None
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
        names = bands or meta["bands"]
        return meta, {b: z[b] for b in names if b in z.files}


def sample(meta: dict, layers: dict[str, np.ndarray], lats: np.ndarray, lons: np.ndarray) -> dict[str, np.ndarray]:
    """Nearest-pixel values at the given points; NaN outside the raster."""
    row = np.floor((meta["north"] - lats) / meta["pixel_deg"]).astype(int)
    col = np.floor((lons - meta["west"]) / meta["pixel_deg"]).astype(int)
    inside = (row >= 0) & (row < meta["height"]) & (col >= 0) & (col < meta["width"])
    out = {}
    for name, layer in layers.items():
        values = np.full(len(lats), np.nan, dtype=np.float32)
        values[inside] = layer[row[inside], col[inside]]
        out[name] = values
    return out


def sample_day(day: date, lats, lons, bands: list[str]) -> dict[str, np.ndarray] | None:
    """Values for one day at arbitrary points, or None if that day (or a band) isn't cached."""
    loaded = load_day(day, bands)
    if loaded is None or any(b not in loaded[1] for b in bands):
        return None
    return sample(loaded[0], loaded[1], np.asarray(lats, dtype=float), np.asarray(lons, dtype=float))


def target_hexes(which: str) -> pd.DataFrame:
    if which == "stations":
        cells = pd.read_parquet(STATIONS_FILE, columns=["h3"])["h3"].drop_duplicates()
        latlng = np.array([h3.cell_to_latlng(c) for c in cells])
        return pd.DataFrame({"h3": cells.to_numpy(), "lat": latlng[:, 0], "lon": latlng[:, 1]})
    return pd.read_parquet(GRID_FILE, columns=["h3", "lat", "lon"])


def build_features(start: date, end: date, bands: list[str], hexes: str) -> None:
    """Stream one row group per day into a Parquet file per month; memory stays at one day."""
    targets = target_hexes(hexes)
    lats, lons = targets["lat"].to_numpy(), targets["lon"].to_numpy()
    h3_col = pa.array(targets["h3"]).dictionary_encode()
    n = len(targets)
    log.info("sampling %d bands at %s hexes", len(bands), f"{n:,}")

    month = start.replace(day=1)
    while month <= end:
        next_month = (month.replace(day=28) + timedelta(days=4)).replace(day=1)
        path = OUT_DIR / f"month={month:%Y-%m}" / "part-0.parquet"
        tmp = path.with_suffix(".tmp")
        writer, written, missing = None, 0, 0
        day = max(month, start)
        while day < next_month and day <= end:
            loaded = load_day(day, bands)
            if loaded is None:
                missing += 1
            else:
                values = sample(loaded[0], loaded[1], lats, lons)
                table = pa.table({"h3": h3_col, "date": pa.array(np.full(n, np.datetime64(day, "D"))),
                                  **{b: pa.array(values.get(b, np.full(n, np.nan, dtype=np.float32)))
                                     for b in bands}})
                if writer is None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
                writer.write_table(table)
                written += 1
            day += timedelta(days=1)
        if writer is not None:
            writer.close()
            tmp.replace(path)
            log.info("%s: %d days written, %d days not cached", f"{month:%Y-%m}", written, missing)
        month = next_month


def main(argv: list[str] | None = None) -> int:
    default_start, default_end = corpus_window()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=date.fromisoformat, default=default_start)
    parser.add_argument("--end", type=date.fromisoformat, default=default_end)
    parser.add_argument("--bands", default=",".join(BANDS), help=f"comma list of {list(BANDS)}")
    parser.add_argument("--pixel-deg", type=float, default=PIXEL_DEG)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--hexes", choices=["all", "stations"], default="all",
                        help="all grid hexes, or only hexes containing a station (training)")
    parser.add_argument("--skip-download", action="store_true", help="only rebuild features from cached rasters")
    args = parser.parse_args(argv)
    bands = [b.strip() for b in args.bands.split(",")]
    unknown = [b for b in bands if b not in BANDS]
    if unknown:
        parser.error(f"unknown bands {unknown}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.skip_download:
        download(args.start, args.end, bands, args.pixel_deg, args.workers)
    build_features(args.start, args.end, bands, args.hexes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
