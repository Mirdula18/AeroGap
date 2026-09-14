"""Historical air quality corpus from the OpenAQ S3 open data archive -> Parquet.

This is the model TRAINING corpus. ingest/cpcb.py only returns the current
live snapshot, which cannot train a spatial interpolation model.

Archive: s3://openaq-data-archive/records/csv.gz/locationid=<id>/year=<y>/month=<m>/location-<id>-<yyyymmdd>.csv.gz
It is a public bucket: we read it over plain HTTPS with unsigned requests.
No account, no API key, no AWS credentials, no boto3.

The record files carry no country/city/state fields. The archive's
provider=<p>/country=<cc>/ tree looks like a shortcut but is a frozen legacy
copy (India's caaqm/cpcb providers stop in 2022/2018), so the pipeline is:

  1. discover  list every location id, probe each one once for the years it
               has and its coordinates. ~55k locations, roughly an hour the
               first time, then cached in data/raw/openaq/<snapshot>/.
  2. geolocate assign state (ADM1) and district (ADM2) by point-in-polygon
               against geoBoundaries (CC BY / ODbL). Points outside the
               country's ADM1 polygons are dropped - that is the country filter.
  3. download  mirror the daily files for matching locations and dates.
               Files already on disk with the right size are never re-fetched.
  4. build     clean + normalise month by month into
               data/history/state=<s>/parameter=<p>/month=<YYYY-MM>/.
               A partition always holds every cached day of that month, so a
               narrower re-run can never clobber data from a wider one.

The `city` column is the ADM2 district containing the station: the archive
has no city field. OpenAQ timestamps mark the END of the averaging period.

Usage:
    python -m ingest.openaq                                   # India, last 12 months
    python -m ingest.openaq --start 2026-08-01 --end 2026-08-31 --states "Tamil Nadu"
    python -m ingest.openaq --country IND --start 2024-01-01
    python -m ingest.openaq --skip-download                   # rebuild Parquet from cache only
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import re
import sys
import threading
import time
import unicodedata
import xml.etree.ElementTree as ET
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BUCKET_URL = "https://openaq-data-archive.s3.amazonaws.com/"
RECORDS_PREFIX = "records/csv.gz/"
S3_NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
GEOBOUNDARIES_API = "https://www.geoboundaries.org/api/current/gbOpen/{iso3}/{level}/"

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "openaq"
OUT_DIR = ROOT / "data" / "history"

TIMEOUT = (15, 120)
WORKERS = 32

# Coastal / border stations can fall just outside simplified polygons.
SNAP_TOLERANCE_DEG = 0.05  # ~5 km

SENTINELS = {999.0, 999.9, 999.99, 9999.0, 99999.0}

# Generous physical ceilings in the archive's native units. Above these a
# reading is an instrument fault, not pollution. Keys: (parameter, unit).
IMPLAUSIBLE_ABOVE = {
    ("pm25", "ug/m3"): 2000.0,
    ("pm10", "ug/m3"): 3000.0,
    ("pm1", "ug/m3"): 2000.0,
    ("no2", "ug/m3"): 2000.0,
    ("no2", "ppb"): 1000.0,
    ("no2", "ppm"): 1.0,
    ("so2", "ug/m3"): 2600.0,
    ("so2", "ppb"): 1000.0,
    ("so2", "ppm"): 1.0,
    ("o3", "ug/m3"): 1000.0,
    ("o3", "ppb"): 500.0,
    ("o3", "ppm"): 0.5,
    ("co", "ug/m3"): 100000.0,
    ("co", "ppb"): 100000.0,
    ("co", "ppm"): 100.0,
    ("co", "mg/m3"): 100.0,
    ("nh3", "ug/m3"): 2000.0,
}

UNIT_ALIASES = {
    "µg/m³": "ug/m3",
    "μg/m³": "ug/m3",
    "ug/m³": "ug/m3",
    "µg/m3": "ug/m3",
    "mg/m³": "mg/m3",
    "particles/cm³": "particles/cm3",
    "°c": "degC",
    "°C": "degC",
}

OUTPUT_COLUMNS = ["station_id", "station_name", "lat", "lon", "city", "state",
                  "parameter", "value", "unit", "timestamp_utc"]

log = logging.getLogger("ingest.openaq")


# --------------------------------------------------------------------------- http


def make_session(workers: int) -> requests.Session:
    retry = Retry(total=6, backoff_factor=1.5, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET",), respect_retry_after_header=True)
    adapter = HTTPAdapter(pool_connections=workers, pool_maxsize=workers, max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    return session


def s3_list(session: requests.Session, prefix: str, delimiter: str | None = "/",
            max_keys: int = 1000, first_page_only: bool = False):
    """Yield ('prefix', str) and ('object', key, size) entries for an anonymous ListObjectsV2."""
    params = {"list-type": "2", "prefix": prefix, "max-keys": str(max_keys)}
    if delimiter:
        params["delimiter"] = delimiter
    while True:
        resp = session.get(BUCKET_URL, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        for p in root.findall("s:CommonPrefixes/s:Prefix", S3_NS):
            yield ("prefix", p.text)
        for c in root.findall("s:Contents", S3_NS):
            yield ("object", c.find("s:Key", S3_NS).text, int(c.find("s:Size", S3_NS).text))
        token = root.find("s:NextContinuationToken", S3_NS)
        if token is None or first_page_only:
            return
        params["continuation-token"] = token.text


# --------------------------------------------------------------------------- geometry


def _ascii(name: str) -> str:
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().strip()


def _norm_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _ascii(name).lower())


@dataclass
class Region:
    name: str
    bbox: tuple[float, float, float, float]  # minx, miny, maxx, maxy
    rings: list[np.ndarray] = field(repr=False)  # each (n, 2) lon/lat, closed


def load_regions(iso3: str, level: str, cache_dir: Path, session: requests.Session) -> list[Region]:
    path = cache_dir / f"geoBoundaries-{iso3}-{level}_simplified.geojson"
    if not path.exists():
        meta = session.get(GEOBOUNDARIES_API.format(iso3=iso3, level=level), timeout=TIMEOUT)
        meta.raise_for_status()
        url = meta.json()["simplifiedGeometryGeoJSON"]
        log.info("downloading %s boundaries for %s", level, iso3)
        resp = session.get(url, timeout=(15, 300))
        resp.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resp.content)

    regions = []
    for feat in json.loads(path.read_text(encoding="utf-8"))["features"]:
        geom = feat["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        # Every ring (outer + holes) goes into the even-odd crossing test,
        # which handles holes correctly.
        rings = [np.asarray(ring, dtype=float)[:, :2] for poly in polys for ring in poly]
        allpts = np.vstack(rings)
        bbox = (*allpts.min(axis=0), *allpts.max(axis=0))
        regions.append(Region(_ascii(feat["properties"]["shapeName"]), bbox, rings))
    return regions


def _crossings(rings: list[np.ndarray], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    inside = np.zeros(len(x), dtype=bool)
    for ring in rings:
        x1, y1 = ring[:-1, 0], ring[:-1, 1]
        x2, y2 = ring[1:, 0], ring[1:, 1]
        # (points, edges) broadcast
        cond = (y1[None, :] > y[:, None]) != (y2[None, :] > y[:, None])
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = x1[None, :] + (y[:, None] - y1[None, :]) * (x2 - x1)[None, :] / (y2 - y1)[None, :]
        inside ^= (cond & (x[:, None] < xint)).sum(axis=1) % 2 == 1
    return inside


def _edge_distance(rings: list[np.ndarray], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    best = np.full(len(x), np.inf)
    for ring in rings:
        a, b = ring[:-1], ring[1:]
        ab = b - a
        denom = np.where((ab**2).sum(axis=1) == 0, 1, (ab**2).sum(axis=1))
        px = np.stack([x, y], axis=1)[:, None, :] - a[None, :, :]
        t = np.clip((px * ab[None]).sum(axis=2) / denom[None], 0, 1)
        d = np.linalg.norm(px - t[..., None] * ab[None], axis=2).min(axis=1)
        best = np.minimum(best, d)
    return best


def assign_regions(lon: np.ndarray, lat: np.ndarray, regions: list[Region]) -> list[str | None]:
    """Name of the region containing each point, snapping to the nearest edge within tolerance."""
    out: list[str | None] = [None] * len(lon)
    nearest = np.full(len(lon), np.inf)
    nearest_name: list[str | None] = [None] * len(lon)
    for region in regions:
        minx, miny, maxx, maxy = region.bbox
        t = SNAP_TOLERANCE_DEG
        cand = np.flatnonzero((lon >= minx - t) & (lon <= maxx + t) & (lat >= miny - t) & (lat <= maxy + t))
        cand = np.array([i for i in cand if out[i] is None], dtype=int)
        if cand.size == 0:
            continue
        inside = _crossings(region.rings, lon[cand], lat[cand])
        for i in cand[inside]:
            out[i] = region.name
        rest = cand[~inside]
        if rest.size:
            d = _edge_distance(region.rings, lon[rest], lat[rest])
            for i, di in zip(rest, d):
                if di < nearest[i]:
                    nearest[i], nearest_name[i] = di, region.name
    for i in range(len(out)):
        if out[i] is None and nearest[i] <= SNAP_TOLERANCE_DEG:
            out[i] = nearest_name[i]
    return out


# --------------------------------------------------------------------------- discover


class ProbeCache:
    """Append-only JSONL of per-location probe results, safe across threads and restarts."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.entries: dict[int, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self.entries[entry["location_id"]] = entry  # later lines win

    def put(self, entry: dict) -> None:
        with self.lock:
            self.entries[entry["location_id"]] = entry
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def list_location_ids(session: requests.Session, snapshot_dir: Path, refresh: bool) -> list[int]:
    path = snapshot_dir / "location_ids.txt"
    if path.exists() and not refresh:
        return [int(x) for x in path.read_text().split()]
    log.info("listing all location ids in the archive (one-off, cached)")
    ids = []
    for kind, value, *_ in s3_list(session, RECORDS_PREFIX):
        m = re.search(r"locationid=(\d+)/$", value) if kind == "prefix" else None
        if m:
            ids.append(int(m.group(1)))
    path.write_text("\n".join(map(str, sorted(ids))))
    log.info("%d location ids", len(ids))
    return sorted(ids)


def _first_row(session: requests.Session, key: str) -> dict | None:
    """Read just the first CSV row of a gzip object via a byte-range request."""
    resp = session.get(BUCKET_URL + key, headers={"Range": "bytes=0-8191"}, timeout=TIMEOUT)
    resp.raise_for_status()
    text = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(resp.content).decode("utf-8", "replace")
    lines = text.splitlines()
    if len(lines) < 2:
        return None
    header = pd.read_csv(io.StringIO(lines[0] + "\n" + lines[1]))
    row = header.iloc[0]
    return {"name": str(row["location"]), "lat": float(row["lat"]), "lon": float(row["lon"])}


def probe_location(session: requests.Session, loc_id: int, years_wanted: set[int]) -> dict:
    prefix = f"{RECORDS_PREFIX}locationid={loc_id}/"
    years = sorted(int(m.group(1)) for kind, v, *_ in s3_list(session, prefix)
                   if kind == "prefix" and (m := re.search(r"year=(\d{4})/$", v)))
    entry = {"location_id": loc_id, "years": years, "name": None, "lat": None, "lon": None,
             "coords_year": None}
    overlap = [y for y in years if y in years_wanted]
    if overlap:
        entry["coords_year"] = max(overlap)
        entry.update(_coordinates(session, prefix, max(overlap)) or {})
    return entry


def _coordinates(session: requests.Session, prefix: str, year: int) -> dict | None:
    for kind, key, *_ in s3_list(session, f"{prefix}year={year}/", delimiter=None, max_keys=1, first_page_only=True):
        if kind == "object":
            return _first_row(session, key)
    return None


def discover(session: requests.Session, snapshot_dir: Path, years_wanted: set[int],
             workers: int, refresh: bool, only_ids: list[int] | None = None) -> pd.DataFrame:
    ids = only_ids or list_location_ids(session, snapshot_dir, refresh)
    cache = ProbeCache(snapshot_dir / "locations_probe.jsonl")
    if refresh:
        cache.entries.clear()

    def needs_probe(i: int) -> bool:
        e = cache.entries.get(i)
        if e is None:
            return True
        # Probed before for a narrower date range: fetch coordinates if newly relevant.
        return (e["lat"] is None and e.get("coords_year") is None
                and any(y in years_wanted for y in e["years"]))

    todo = [i for i in ids if needs_probe(i)]
    log.info("%d locations known, %d already probed, %d to probe", len(ids), len(ids) - len(todo), len(todo))

    failures = 0
    started = time.time()
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(probe_location, session, i, years_wanted): i for i in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                cache.put(fut.result())
            except Exception as exc:  # leave unprobed; the next run retries it
                failures += 1
                log.debug("probe %d failed: %s", futures[fut], exc)
            if n % 2000 == 0 or n == len(todo):
                rate = n / max(time.time() - started, 1e-9)
                log.info("probed %d/%d (%.0f/s, ~%.0f min left, %d failed)",
                         n, len(todo), rate, (len(todo) - n) / max(rate, 1e-9) / 60, failures)
    if failures:
        log.warning("%d probes failed and will be retried on the next run", failures)

    wanted_ids = set(ids)
    locs = pd.DataFrame([e for i, e in cache.entries.items() if i in wanted_ids])
    if locs.empty:
        sys.exit("no locations could be probed - check network access to the archive")
    locs = locs[locs["lat"].notna() & locs["years"].map(lambda ys: any(y in years_wanted for y in ys))]
    return locs.reset_index(drop=True)


def geolocate(locs: pd.DataFrame, iso3: str, session: requests.Session) -> pd.DataFrame:
    cache_dir = RAW_DIR / "boundaries"
    adm1 = load_regions(iso3, "ADM1", cache_dir, session)
    minx = min(r.bbox[0] for r in adm1) - SNAP_TOLERANCE_DEG
    miny = min(r.bbox[1] for r in adm1) - SNAP_TOLERANCE_DEG
    maxx = max(r.bbox[2] for r in adm1) + SNAP_TOLERANCE_DEG
    maxy = max(r.bbox[3] for r in adm1) + SNAP_TOLERANCE_DEG
    locs = locs[locs["lon"].between(minx, maxx) & locs["lat"].between(miny, maxy)].reset_index(drop=True)

    lon, lat = locs["lon"].to_numpy(), locs["lat"].to_numpy()
    locs["state"] = assign_regions(lon, lat, adm1)
    locs = locs[locs["state"].notna()].reset_index(drop=True)
    adm2 = load_regions(iso3, "ADM2", cache_dir, session)
    locs["city"] = assign_regions(locs["lon"].to_numpy(), locs["lat"].to_numpy(), adm2)
    return locs


# --------------------------------------------------------------------------- download


def month_starts(start: date, end: date):
    d = start.replace(day=1)
    while d <= end:
        yield d
        d = (d.replace(day=28) + timedelta(days=4)).replace(day=1)


def _file_date(key: str) -> date | None:
    m = re.search(r"-(\d{8})\.csv\.gz$", key)
    return date(int(m[1][:4]), int(m[1][4:6]), int(m[1][6:])) if m else None


def download(session: requests.Session, snapshot_dir: Path, location_ids: list[int],
             start: date, end: date, workers: int) -> None:
    mirror = snapshot_dir / "records"

    def sync_month(loc_id: int, month: date) -> tuple[int, int, int]:
        prefix = f"{RECORDS_PREFIX}locationid={loc_id}/year={month.year}/month={month.month:02d}/"
        fetched = skipped = 0
        for kind, key, *rest in s3_list(session, prefix, delimiter=None):
            if kind != "object":
                continue
            day = _file_date(key)
            if day is None or not (start <= day <= end):
                continue
            local = mirror / key[len(RECORDS_PREFIX):]
            if local.exists() and local.stat().st_size == rest[0]:
                skipped += 1
                continue
            resp = session.get(BUCKET_URL + key, timeout=TIMEOUT)
            resp.raise_for_status()
            local.parent.mkdir(parents=True, exist_ok=True)
            tmp = local.with_suffix(".part")
            tmp.write_bytes(resp.content)
            tmp.replace(local)
            fetched += 1
        return loc_id, fetched, skipped

    jobs = [(i, m) for i in location_ids for m in month_starts(start, end)]
    log.info("syncing %d location-months (%d locations)", len(jobs), len(location_ids))
    totals = Counter()
    started = time.time()
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(sync_month, i, m) for i, m in jobs]
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                _, fetched, skipped = fut.result()
                totals["fetched"] += fetched
                totals["cached"] += skipped
            except Exception as exc:
                totals["failed_months"] += 1
                log.debug("month sync failed: %s", exc)
            if n % 500 == 0 or n == len(jobs):
                log.info("location-months %d/%d | files fetched %d, already cached %d, failed months %d (%.0fs)",
                         n, len(jobs), totals["fetched"], totals["cached"], totals["failed_months"],
                         time.time() - started)
    if totals["failed_months"]:
        log.warning("%d location-months failed; re-run to retry (cached files are kept)", totals["failed_months"])


# --------------------------------------------------------------------------- build


def normalise_unit(unit: str) -> str:
    unit = str(unit).strip()
    return UNIT_ALIASES.get(unit, UNIT_ALIASES.get(unit.lower(), unit.lower()))


def read_day_file(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    return pd.read_csv(io.StringIO(text), dtype={"value": "string", "units": "string",
                                                 "parameter": "string", "location": "string"})


class DropLog:
    def __init__(self):
        self.counts: Counter = Counter()
        self.by_param: dict[str, Counter] = defaultdict(Counter)
        self.raw_rows = 0
        self.kept_rows = 0

    def drop(self, df: pd.DataFrame, mask: pd.Series, reason: str) -> pd.DataFrame:
        n = int(mask.sum())
        if n:
            self.counts[reason] += n
            for param, c in df.loc[mask, "parameter"].value_counts().items():
                self.by_param[param][reason] += int(c)
        return df[~mask]


def clean_month(frames: list[pd.DataFrame], locs: pd.DataFrame, drops: DropLog) -> pd.DataFrame:
    df = pd.concat(frames, ignore_index=True)
    drops.raw_rows += len(df)

    df["parameter"] = df["parameter"].str.strip().str.lower()
    df["unit"] = df["units"].map(normalise_unit)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["timestamp_utc"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce", format="ISO8601")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    df = drops.drop(df, df["value"].isna(), "null_value")
    df = drops.drop(df, df["timestamp_utc"].isna(), "bad_timestamp")
    df = drops.drop(df, df["lat"].isna() | df["lon"].isna(), "missing_coordinates")
    df = drops.drop(df, df["value"] < 0, "negative")
    df = drops.drop(df, df["value"].isin(SENTINELS), "sentinel_999")
    df = drops.drop(df, df["value"] > _caps(df), "implausible_high")
    dup = df.duplicated(subset=["sensors_id", "timestamp_utc"], keep="last")
    df = drops.drop(df, dup, "duplicate_sensor_timestamp")

    meta = locs.set_index("location_id")[["state", "city"]]
    df = df.join(meta, on="location_id")
    df["station_id"] = "openaq-" + df["location_id"].astype(str)
    df = df.rename(columns={"location": "station_name"})
    drops.kept_rows += len(df)
    return df[OUTPUT_COLUMNS]


def _caps(df: pd.DataFrame) -> pd.Series:
    keys = pd.MultiIndex.from_frame(df[["parameter", "unit"]])
    lookup = pd.Series(IMPLAUSIBLE_ABOVE)
    return pd.Series(lookup.reindex(keys).fillna(np.inf).to_numpy(), index=df.index)


class Stats:
    """Running summary without holding the whole corpus in memory."""

    SAMPLE_PER_MONTH = 20_000

    def __init__(self):
        self.stations: set[str] = set()
        self.unreadable_files = 0
        self.rows_per_param: Counter = Counter()
        self.units: dict[str, Counter] = defaultdict(Counter)
        self.samples: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
        self.tmin = self.tmax = None

    def update(self, df: pd.DataFrame) -> None:
        self.stations.update(df["station_id"].unique())
        self.rows_per_param.update(df["parameter"].value_counts().to_dict())
        for (param, unit), group in df.groupby(["parameter", "unit"])["value"]:
            self.units[param][unit] += len(group)
            vals = group.to_numpy()
            if len(vals) > self.SAMPLE_PER_MONTH:
                vals = np.random.default_rng(0).choice(vals, self.SAMPLE_PER_MONTH, replace=False)
            self.samples[(param, unit)].append(vals)
        lo, hi = df["timestamp_utc"].min(), df["timestamp_utc"].max()
        self.tmin = lo if self.tmin is None else min(self.tmin, lo)
        self.tmax = hi if self.tmax is None else max(self.tmax, hi)


def build(snapshot_dir: Path, locs: pd.DataFrame, start: date, end: date) -> tuple[DropLog, Stats, pd.DataFrame]:
    mirror = snapshot_dir / "records"
    drops, stats = DropLog(), Stats()
    by_state = locs.groupby("state")["location_id"].apply(sorted)
    station_rows = []

    for month in month_starts(start, end):
        label = month.strftime("%Y-%m")
        month_files = month_rows = 0
        # One (month, state) chunk at a time keeps memory bounded: Delhi alone is
        # ~1.1M rows/month, a whole national month would be ~15M.
        for state, loc_ids in by_state.items():
            # Whole month, not just [start, end]: partitions are rewritten wholesale,
            # so they must contain every cached day or a narrow run would lose data.
            files = []
            for loc_id in loc_ids:
                month_dir = mirror / f"locationid={loc_id}" / f"year={month.year}" / f"month={month.month:02d}"
                if month_dir.exists():
                    files += sorted(month_dir.glob("*.csv.gz"))
            if not files:
                continue

            frames = []
            for p in files:
                try:
                    frames.append(read_day_file(p))
                except Exception as exc:
                    stats.unreadable_files += 1
                    log.warning("unreadable %s: %s", p.relative_to(ROOT), exc)
            if not frames:
                continue
            df = clean_month(frames, locs, drops)
            if df.empty:
                continue

            df["month"] = label
            table = pa.Table.from_pandas(df, preserve_index=False)
            ds.write_dataset(
                table, OUT_DIR, format="parquet",
                partitioning=ds.partitioning(
                    pa.schema([("state", pa.string()), ("parameter", pa.string()), ("month", pa.string())]),
                    flavor="hive"),
                existing_data_behavior="delete_matching",
                basename_template="part-{i}.parquet",
            )
            stats.update(df)
            station_rows.append(df.groupby("station_id").agg(
                station_name=("station_name", "last"), lat=("lat", "median"), lon=("lon", "median"),
                city=("city", "last"), state=("state", "last"),
                first_seen=("timestamp_utc", "min"), last_seen=("timestamp_utc", "max"),
                rows=("value", "size")).reset_index())
            month_files += len(files)
            month_rows += len(df)
        log.info("%s: %d files -> %d rows kept", label, month_files, month_rows)

    stations = pd.DataFrame()
    if station_rows:
        stations = (pd.concat(station_rows).groupby("station_id").agg(
            station_name=("station_name", "last"), lat=("lat", "median"), lon=("lon", "median"),
            city=("city", "last"), state=("state", "last"), first_seen=("first_seen", "min"),
            last_seen=("last_seen", "max"), rows=("rows", "sum")).reset_index())
        stations.to_parquet(OUT_DIR / "_stations.parquet", index=False)
    return drops, stats, stations


# --------------------------------------------------------------------------- summary


def summarise(drops: DropLog, stats: Stats, stations: pd.DataFrame, args) -> dict:
    total_dropped = sum(drops.counts.values())
    pct = lambda n: 100 * n / drops.raw_rows if drops.raw_rows else 0.0  # noqa: E731

    unit_table = []
    for (param, unit), chunks in sorted(stats.samples.items()):
        vals = np.concatenate(chunks)
        q = np.percentile(vals, [5, 50, 95, 99])
        unit_table.append({"parameter": param, "unit": unit, "rows": stats.units[param][unit],
                           "p5": q[0], "median": q[1], "p95": q[2], "p99": q[3], "max_sampled": vals.max()})

    states = stations["state"].value_counts() if not stations.empty else pd.Series(dtype=int)
    summary = {
        "query": {"country": args.country, "start": str(args.start), "end": str(args.end),
                  "states_filter": args.states},
        "stations": int(len(stats.stations)),
        "timestamp_utc_min": str(stats.tmin), "timestamp_utc_max": str(stats.tmax),
        "raw_rows": drops.raw_rows, "kept_rows": drops.kept_rows, "dropped_rows": total_dropped,
        "dropped_by_reason": dict(drops.counts),
        "dropped_by_parameter": {k: dict(v) for k, v in drops.by_param.items()},
        "rows_per_parameter": dict(stats.rows_per_param.most_common()),
        "states_covered": int(states.size),
        "stations_per_state": states.to_dict(),
        "stations_without_district": int(stations["city"].isna().sum()) if not stations.empty else 0,
        "units": unit_table,
    }
    (OUT_DIR / "_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    p = print
    p("\n" + "=" * 72)
    p(f"OpenAQ historical corpus  {args.country}  {args.start} -> {args.end}"
      + (f"  states={args.states}" if args.states else ""))
    p("=" * 72)
    p(f"stations            {summary['stations']}")
    p(f"timestamp_utc range {stats.tmin}  ->  {stats.tmax}")
    p(f"rows                raw {drops.raw_rows:,}  kept {drops.kept_rows:,}  dropped {total_dropped:,} ({pct(total_dropped):.1f}%)")
    if stats.unreadable_files:
        p(f"  unreadable files skipped: {stats.unreadable_files}")
    for reason, n in drops.counts.most_common():
        p(f"  dropped {reason:<28} {n:>12,}  ({pct(n):.2f}%)")
    p("\nrows per parameter")
    for param, n in stats.rows_per_param.most_common():
        p(f"  {param:<16} {n:>12,}")
    p(f"\ngeographic spread    {summary['states_covered']} states/UTs"
      f"  ({summary['stations_without_district']} stations without a district match)")
    for state, n in states.items():
        p(f"  {state:<44} {n:>4} stations")
    p("\nunits and value ranges (sampled percentiles, native units)")
    p(f"  {'parameter':<18}{'unit':<15}{'rows':>12}{'p5':>10}{'median':>10}{'p95':>10}{'p99':>10}")
    for u in unit_table:
        p(f"  {u['parameter']:<18}{u['unit']:<15}{u['rows']:>12,}{u['p5']:>10.2f}{u['median']:>10.2f}"
          f"{u['p95']:>10.2f}{u['p99']:>10.2f}")
    p(f"\nwrote {OUT_DIR.relative_to(ROOT)}/state=*/parameter=*/month=*/  +  _stations.parquet  +  _summary.json")
    return summary


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    today = date.today()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--country", default="IND", help="ISO3 code, used for geoBoundaries (default IND)")
    parser.add_argument("--start", type=date.fromisoformat, default=today - timedelta(days=365),
                        help="first day, YYYY-MM-DD (default: 365 days ago)")
    parser.add_argument("--end", type=date.fromisoformat, default=today, help="last day, YYYY-MM-DD (default: today)")
    parser.add_argument("--states", help='comma-separated ADM1 names to keep, e.g. "Delhi,Tamil Nadu"')
    parser.add_argument("--snapshot", help="raw cache directory name (default: country code)")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--refresh-index", action="store_true",
                        help="re-list locations and re-probe them (picks up new stations)")
    parser.add_argument("--skip-download", action="store_true", help="only rebuild Parquet from cached files")
    parser.add_argument("--location-ids", help="comma-separated OpenAQ location ids; skips the full archive scan")
    args = parser.parse_args(argv)
    only_ids = [int(x) for x in args.location_ids.split(",")] if args.location_ids else None
    args.country = args.country.upper()
    if args.start > args.end:
        parser.error("--start is after --end")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("urllib3").setLevel(logging.ERROR)

    snapshot_dir = RAW_DIR / (args.snapshot or args.country)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    session = make_session(args.workers)
    years_wanted = set(range(args.start.year, args.end.year + 1))

    locs = discover(session, snapshot_dir, years_wanted, args.workers, args.refresh_index, only_ids)
    locs = geolocate(locs, args.country, session)
    log.info("%d locations with %s-%s data inside %s across %d states",
             len(locs), min(years_wanted), max(years_wanted), args.country, locs["state"].nunique())

    if args.states:
        wanted = {_norm_key(s) for s in args.states.split(",")}
        locs = locs[locs["state"].map(_norm_key).isin(wanted)].reset_index(drop=True)
        log.info("state filter -> %d locations", len(locs))
        if locs.empty:
            sys.exit(f"no locations matched --states {args.states!r}")
    locs.drop(columns=["years"]).to_parquet(snapshot_dir / "locations.parquet", index=False)

    if not args.skip_download:
        download(session, snapshot_dir, locs["location_id"].tolist(), args.start, args.end, args.workers)

    drops, stats, stations = build(snapshot_dir, locs, args.start, args.end)
    if drops.raw_rows == 0:
        log.warning("no rows found for this query")
        return 1
    summarise(drops, stats, stations, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
