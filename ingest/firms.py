"""NASA FIRMS active fires -> fire features per H3 hex per day.

Source: VIIRS S-NPP 375 m. Standard Processing (VIIRS_SNPP_SP) wherever FIRMS
has published it, Near-Real-Time (VIIRS_SNPP_NRT) after that (the split date comes
from FIRMS' data_availability endpoint). NOAA-20 is never added on top: its swaths
overlap S-NPP's and would double-count the same fires.

S-NPP has outage days (e.g. 2026-04-28 -> 06-02: FIRMS returns header-only files).
India always has active fires, so any requested day with zero S-NPP detections is
treated as a gap and filled from NOAA-20 (same VIIRS 375 m instrument, ~50 min
later overpass). day_sources.parquet records which satellite supplied each day.

Cleaning (logged):
  - low-confidence detections dropped
  - SP detections kept only for type 0 (vegetation fire); static industrial
    sources (type 2) are recorded as a hex mask, which is then applied to NRT
    detections, since NRT files have no type column
Days are IST calendar days, matching the air quality corpus.

Features:
  hex_daily_r7.parquet     h3, date, fire_count, frp_sum                 (hexes with fires only)
  upwind_daily_r5.parquet  h3_r5, date, fires_<R>km, frp_<R>km,
                           upwind_fires_<R>km, upwind_frp_<R>km          (cells with a fire within R only)
Fires are aggregated to H3 res 5 (~250 km2) for the regional features: smoke
influence is regional, and res-7 hexes inherit their res-5 parent's values.
Upwind weight = max(0, cos(angle between the fire->hex bearing and the wind vector))
using ERA5-Land 10 m wind from ingest/gee.py rasters; calm wind (< 0.5 m/s)
weighs 0.5. Days without cached wind get NaN upwind columns.

Raw API responses are cached in data/raw/firms/<source>/ and never re-fetched.

Usage:
    python -m ingest.firms                                 # corpus window
    python -m ingest.firms --start 2025-11-01 --end 2025-11-07
    python -m ingest.firms --skip-download
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import time
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import requests
from dotenv import dotenv_values

from ingest.gee import corpus_window, sample_day

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "firms"
OUT_DIR = ROOT / "data" / "fires"
GRID_FILE = ROOT / "data" / "grid" / "grid_r7_api.parquet"
API = "https://firms.modaps.eosdis.nasa.gov/api"

SP_SOURCE, NRT_SOURCE = "VIIRS_SNPP_SP", "VIIRS_SNPP_NRT"
FALLBACK_SP, FALLBACK_NRT = "VIIRS_NOAA20_SP", "VIIRS_NOAA20_NRT"
BBOX = (66.0, 5.0, 100.0, 39.0)  # wider than India: Pakistan Punjab and Nepal smoke reaches the plains
CHUNK_DAYS = 5                   # FIRMS area API allows 1-10 days per request
HEX_RES, REGION_RES = 7, 5
RADIUS_KM = 150.0
CALM_WIND_MS = 0.5
IST = pd.Timedelta(hours=5, minutes=30)
MAX_ATTEMPTS = 6

log = logging.getLogger("ingest.firms")


# --------------------------------------------------------------------------- download


def load_key() -> str:
    key = dotenv_values(ROOT / ".env").get("FIRMS_MAP_KEY")
    if not key:
        sys.exit("FIRMS_MAP_KEY is not set in .env")
    return key


def fetch(session: requests.Session, path: str, cache: Path, key: str) -> str:
    """GET {API}/{path} (with {key} substituted) unless cached. The key never reaches disk or logs."""
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    url = f"{API}/{path.format(key=key)}"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = session.get(url, timeout=(20, 300))
            if resp.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            text = resp.text
            if text.lstrip().lower().startswith(("invalid", "<!doctype", "<html", "error")):
                raise ValueError(text[:200])
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(".part")
            tmp.write_text(text.replace(key, "<KEY>"), encoding="utf-8")
            tmp.replace(cache)
            return text
        except (requests.RequestException, ValueError) as exc:
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(f"{cache.name}: {str(exc).replace(key, '<KEY>')}") from None
            time.sleep(min(2 ** attempt, 60))
    raise AssertionError("unreachable")


def availability(session: requests.Session, key: str) -> dict[str, tuple[date, date]]:
    cache = RAW_DIR / f"data_availability_{date.today():%Y%m%d}.csv"
    df = pd.read_csv(io.StringIO(fetch(session, "data_availability/csv/{key}/ALL", cache, key)))
    return {r.data_id: (date.fromisoformat(r.min_date), date.fromisoformat(r.max_date)) for r in df.itertuples()}


def plan_chunks(days: list[date], avail: dict[str, tuple[date, date]],
                sources: tuple[str, str] = (SP_SOURCE, NRT_SOURCE)) -> list[tuple[str, date, int]]:
    """(source, first day, n days) for consecutive runs: SP where published, NRT after, <= CHUNK_DAYS each."""
    chunks: list[tuple[str, date, int]] = []
    for day in sorted(days):
        source = next((s for s in sources if s in avail and avail[s][0] <= day <= avail[s][1]), None)
        if source is None:
            log.warning("%s: no %s data published", day, " / ".join(sources))
            continue
        if chunks:
            last_source, first, n = chunks[-1]
            if last_source == source and first + timedelta(days=n) == day and n < CHUNK_DAYS:
                chunks[-1] = (source, first, n + 1)
                continue
        chunks.append((source, day, 1))
    return chunks


def days_with_detections(path: Path) -> set[date]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return set()
    return {date.fromisoformat(d) for d in pd.read_csv(io.StringIO(text), usecols=["acq_date"])["acq_date"].unique()}


def download(start: date, end: date) -> list[tuple[str, Path]]:
    key = load_key()
    session = requests.Session()
    avail = availability(session, key)
    west, south, east, north = BBOX
    area = f"{west:g},{south:g},{east:g},{north:g}"

    def fetch_chunks(chunks: list[tuple[str, date, int]]) -> list[tuple[str, Path]]:
        files = []
        for i, (source, first, n) in enumerate(chunks, 1):
            cache = RAW_DIR / source / f"{first:%Y%m%d}_{n}d.csv"
            fetched = not cache.exists()
            fetch(session, f"area/csv/{{key}}/{source}/{area}/{n}/{first.isoformat()}", cache, key)
            files.append((source, cache))
            if fetched and (i % 10 == 0 or i == len(chunks)):
                log.info("fetched %d/%d chunks", i, len(chunks))
        log.info("%d chunks (%s)", len(chunks), dict(Counter(s for s, _, _ in chunks)))
        return files

    requested = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    files = fetch_chunks(plan_chunks(requested, avail))

    present = set().union(*(days_with_detections(p) for _, p in files))
    gaps = sorted(set(requested) - present)
    if gaps:
        log.warning("S-NPP has no detections on %d requested days (outage); filling from NOAA-20: %s",
                    len(gaps), ", ".join(f"{d:%Y-%m-%d}" for d in gaps))
        files += fetch_chunks(plan_chunks(gaps, avail, sources=(FALLBACK_SP, FALLBACK_NRT)))
    return files


# --------------------------------------------------------------------------- detections


def load_detections(files: list[tuple[str, Path]], start: date, end: date) -> tuple[pd.DataFrame, Counter]:
    frames = []
    for source, path in files:
        text = path.read_text(encoding="utf-8")
        if text.strip():
            df = pd.read_csv(io.StringIO(text))
            if len(df):
                frames.append(df.assign(source=source))
    if not frames:
        sys.exit("no fire detections in range")
    df = pd.concat(frames, ignore_index=True)
    drops: Counter = Counter()

    acq = df["acq_time"].astype(int)
    df["utc"] = pd.to_datetime(df["acq_date"]) + pd.to_timedelta(acq // 100, unit="h") + pd.to_timedelta(acq % 100, unit="m")
    df["date"] = (df["utc"] + IST).dt.normalize()
    before = len(df)
    df = df.drop_duplicates(subset=["latitude", "longitude", "utc", "source"])
    drops["duplicate"] = before - len(df)

    df["h3"] = [h3.latlng_to_cell(la, lo, HEX_RES) for la, lo in zip(df["latitude"], df["longitude"])]

    low = df["confidence"].astype(str).str.lower().eq("l")
    drops["low_confidence"] = int(low.sum())
    df = df[~low]

    if "type" in df.columns:
        # SP files (S-NPP or NOAA-20) carry a type column; NRT files don't, so they get the SP static-source mask.
        sp = df["source"].str.endswith("_SP")
        static_hexes = set(df.loc[sp & (df["type"] == 2), "h3"])
        not_fire = sp & (df["type"].fillna(0) != 0)
        drops["sp_not_vegetation_fire"] = int(not_fire.sum())
        df = df[~not_fire]
        masked = df["source"].str.endswith("_NRT") & df["h3"].isin(static_hexes)
        drops["nrt_static_source_hex"] = int(masked.sum())
        df = df[~masked]
        log.info("static industrial-source mask: %d hexes (from SP type 2)", len(static_hexes))

    in_range = df["date"].between(pd.Timestamp(start), pd.Timestamp(end))
    drops["outside_ist_date_range"] = int((~in_range).sum())
    df = df[in_range]
    return df[["latitude", "longitude", "frp", "utc", "date", "h3", "source", "daynight"]].reset_index(drop=True), drops


# --------------------------------------------------------------------------- features


def hex_daily(fires: pd.DataFrame) -> pd.DataFrame:
    return (fires.groupby(["h3", "date"]).agg(fire_count=("frp", "size"), frp_sum=("frp", "sum"))
            .reset_index().astype({"fire_count": "int32", "frp_sum": "float32"}))


def regional_daily(fires: pd.DataFrame, radius_km: float, chunk: int = 2048) -> tuple[pd.DataFrame, int]:
    """Fire counts within radius of every res-5 cell, isotropic and upwind-weighted."""
    targets = pd.read_parquet(GRID_FILE, columns=["h3_r5"])["h3_r5"].drop_duplicates().to_numpy()
    tll = np.array([h3.cell_to_latlng(c) for c in targets])
    tlat, tlon = tll[:, 0], tll[:, 1]

    fires = fires.assign(h3_r5=[h3.cell_to_parent(c, REGION_RES) for c in fires["h3"]])
    src = fires.groupby(["date", "h3_r5"]).agg(n=("frp", "size"), frp=("frp", "sum"),
                                              lat=("latitude", "mean"), lon=("longitude", "mean"))
    r = int(radius_km)
    out, days_with_wind = [], 0
    for day, f in src.groupby(level="date"):
        flat, flon = f["lat"].to_numpy(), f["lon"].to_numpy()
        n, frp = f["n"].to_numpy(dtype=float), f["frp"].to_numpy(dtype=float)
        wind = sample_day(day.date(), tlat, tlon, ["u10", "v10"])
        days_with_wind += wind is not None
        cols = {f"fires_{r}km": [], f"frp_{r}km": [], f"upwind_fires_{r}km": [], f"upwind_frp_{r}km": []}
        for i in range(0, len(targets), chunk):
            la, lo = tlat[i:i + chunk, None], tlon[i:i + chunk, None]
            dx = (lo - flon[None, :]) * 111.32 * np.cos(np.radians(la))   # km east, fire -> target
            dy = (la - flat[None, :]) * 110.57                              # km north
            dist = np.hypot(dx, dy)
            near = dist <= radius_km
            cols[f"fires_{r}km"].append((near * n).sum(axis=1))
            cols[f"frp_{r}km"].append((near * frp).sum(axis=1))
            if wind is None:
                nan = np.full(len(la), np.nan)
                cols[f"upwind_fires_{r}km"].append(nan)
                cols[f"upwind_frp_{r}km"].append(nan)
                continue
            u, v = wind["u10"][i:i + chunk, None], wind["v10"][i:i + chunk, None]
            speed = np.hypot(u, v)
            with np.errstate(invalid="ignore", divide="ignore"):
                align = (dx * u + dy * v) / (np.maximum(dist, 1e-6) * np.maximum(speed, 1e-6))
            weight = np.where(speed < CALM_WIND_MS, 0.5, np.clip(align, 0.0, None))
            weight = np.where(np.isnan(speed), np.nan, weight)  # wind raster gap (e.g. sea pixel)
            cols[f"upwind_fires_{r}km"].append((near * n * weight).sum(axis=1))
            cols[f"upwind_frp_{r}km"].append((near * frp * weight).sum(axis=1))
        frame = pd.DataFrame({"h3_r5": targets, "date": day, **{k: np.concatenate(v) for k, v in cols.items()}})
        out.append(frame[frame[f"fires_{r}km"] > 0])
    result = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    return result.astype({c: "float32" for c in result.columns if c not in ("h3_r5", "date")}), days_with_wind


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    default_start, default_end = corpus_window()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=date.fromisoformat, default=default_start)
    parser.add_argument("--end", type=date.fromisoformat, default=default_end)
    parser.add_argument("--radius-km", type=float, default=RADIUS_KM)
    parser.add_argument("--skip-download", action="store_true", help="use cached API responses only")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # UTC request days one either side, so IST days at the range edges are complete.
    fetch_start, fetch_end = args.start - timedelta(days=1), args.end + timedelta(days=1)
    if args.skip_download:
        # S-NPP chunks plus NOAA-20 gap fills; overlapping cache files are removed by the duplicate filter.
        files = [(p.parent.name, p) for p in sorted(RAW_DIR.glob("VIIRS_*/*.csv"))]
    else:
        files = download(fetch_start, fetch_end)
    fires, drops = load_detections(files, args.start, args.end)

    day_sources = (fires.groupby("date")["source"].agg(lambda s: ",".join(sorted(s.unique())))
                   .rename("sources").reset_index())
    hexes = hex_daily(fires)
    regional, days_with_wind = regional_daily(fires, args.radius_km)
    day_sources.to_parquet(OUT_DIR / "day_sources.parquet", index=False)
    fires.to_parquet(OUT_DIR / "detections.parquet", index=False, compression="zstd")
    hexes.to_parquet(OUT_DIR / "hex_daily_r7.parquet", index=False, compression="zstd")
    regional.to_parquet(OUT_DIR / "upwind_daily_r5.parquet", index=False, compression="zstd")

    grid = pd.read_parquet(GRID_FILE, columns=["h3", "state"])
    by_state = hexes.merge(grid, on="h3", how="left").groupby("state")["fire_count"].sum()
    outside = int(hexes.loc[~hexes["h3"].isin(grid["h3"]), "fire_count"].sum())
    n_days = (args.end - args.start).days + 1

    print("\n" + "=" * 72)
    print(f"FIRMS VIIRS S-NPP fires  {args.start} -> {args.end}  (IST days)")
    print("=" * 72)
    print(f"detections kept     {len(fires):,}  by source {fires['source'].value_counts().to_dict()}")
    print(f"days by satellite   {day_sources['sources'].value_counts().to_dict()}"
          f"  ({n_days - len(day_sources)} IST days with no detections at all)")
    for reason, count in drops.most_common():
        print(f"  dropped {reason:<26} {count:>10,}")
    print(f"hex-days with fire  {len(hexes):,} across {hexes['h3'].nunique():,} hexes"
          f"  ({outside:,} detections outside India's grid, kept for upwind features)")
    print("\nfire detections by month:")
    print(fires.groupby(fires["date"].dt.strftime("%Y-%m")).size().to_string())
    print("\ntop states by detections:")
    print(by_state.sort_values(ascending=False).head(10).astype(int).to_string())
    print(f"\nregional features: {len(regional):,} res-5 cell-days with a fire within {args.radius_km:g} km")
    print(f"wind available for {days_with_wind} of {regional['date'].nunique() if len(regional) else 0} fire days"
          + ("" if days_with_wind else "  -> upwind columns are NaN until `python -m ingest.gee` has run"))
    print(f"\nwrote {OUT_DIR.relative_to(ROOT)}/detections.parquet, hex_daily_r7.parquet, upwind_daily_r5.parquet"
          f"  ({n_days} days)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
