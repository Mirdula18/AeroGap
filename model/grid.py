"""National H3 grid with state/district and distance to the nearest monitoring station.

For every hex covering the country:
    h3, lat, lon (centroid), boundary (WKT), state, district,
    station_count (monitors inside the hex), dist_km + nearest_station_id,
    h3_r4 / h3_r5 / h3_r6 parents (so the API can aggregate without h3).

dist_km is the core feature for the interpolation model and what defines a
"dark zone" on the map: high predicted pollution, nobody measuring nearby.

Hex membership uses H3's own polygon fill (a cell belongs to a polygon when
its centroid is inside), so no point-in-polygon pass is needed:
    ADM1 polygons  -> state     (authoritative national extent)
    ADM2 polygons  -> district  (district cells outside ADM1 slivers still
                                 get the state their district mostly lies in)

Outputs (data/grid/):
    grid_r<res>.parquet                  full grid incl. boundary WKT
    grid_r<res>_api.parquet              slim copy the API serves (no boundaries)
    grid_r<res>_display_r<d>.geojson     coarse simplified GeoJSON for the frontend
    stations.parquet                     stations with their hex

Usage:
    python -m model.grid                         # resolution 7 (~5 km)
    python -m model.grid --res 6 --display-res 4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import h3
import numpy as np
import pandas as pd

from ingest.openaq import RAW_DIR, _ascii, load_regions, make_session

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "grid"
STATIONS = ROOT / "data" / "history" / "_stations.parquet"
EARTH_RADIUS_KM = 6371.0088
PARENT_RES = (4, 5, 6)

log = logging.getLogger("model.grid")


def boundary_features(iso3: str, level: str) -> list[dict]:
    load_regions(iso3, level, RAW_DIR / "boundaries", make_session(4))  # downloads if not cached
    path = RAW_DIR / "boundaries" / f"geoBoundaries-{iso3}-{level}_simplified.geojson"
    return json.loads(path.read_text(encoding="utf-8"))["features"]


def fill(features: list[dict], res: int) -> dict[str, str]:
    """cell -> region name. First polygon wins where simplified boundaries overlap."""
    out: dict[str, str] = {}
    for feat in features:
        name = _ascii(feat["properties"]["shapeName"])
        for cell in h3.geo_to_cells(feat["geometry"], res):
            out.setdefault(cell, name)
    return out


def build_grid(iso3: str, res: int) -> pd.DataFrame:
    log.info("filling ADM1 polygons at res %d", res)
    state_of = fill(boundary_features(iso3, "ADM1"), res)
    log.info("filling ADM2 polygons at res %d", res)
    district_of = fill(boundary_features(iso3, "ADM2"), res)

    # Each district's state = the state most of its cells fall in.
    votes: dict[str, Counter] = {}
    for cell, district in district_of.items():
        if cell in state_of:
            votes.setdefault(district, Counter())[state_of[cell]] += 1
    district_state = {d: c.most_common(1)[0][0] for d, c in votes.items()}

    cells = sorted(set(state_of) | {c for c, d in district_of.items() if d in district_state})
    log.info("%d cells (%d from ADM1, %d added from ADM2 slivers)",
             len(cells), len(state_of), len(cells) - len(state_of))

    latlng = np.array([h3.cell_to_latlng(c) for c in cells])
    df = pd.DataFrame({"h3": cells, "lat": latlng[:, 0], "lon": latlng[:, 1]})
    df["district"] = [district_of.get(c) for c in cells]
    df["state"] = [state_of.get(c) or district_state.get(district_of.get(c)) for c in cells]
    df["boundary"] = [_wkt(h3.cell_to_boundary(c)) for c in cells]
    for pres in PARENT_RES:
        if pres < res:
            df[f"h3_r{pres}"] = [h3.cell_to_parent(c, pres) for c in cells]
    return df


def _wkt(boundary: tuple[tuple[float, float], ...]) -> str:
    ring = list(boundary) + [boundary[0]]
    return "POLYGON((" + ", ".join(f"{lng:.5f} {lat:.5f}" for lat, lng in ring) + "))"


def load_stations(path: Path) -> pd.DataFrame:
    """History-corpus stations, or the discovery index as a fallback before the build finishes.

    data/history/_stations.parquet            -> stations with readings (preferred)
    data/raw/openaq/<snapshot>/locations.parquet -> every discovered location, no reading counts
    """
    df = pd.read_parquet(path)
    if "location_id" in df.columns:
        log.warning("using discovery index %s: stations have no reading counts yet", path.name)
        df = df.rename(columns={"name": "station_name"}).assign(
            station_id="openaq-" + df["location_id"].astype(str), rows=pd.NA, last_seen=pd.NaT)
    return df[["station_id", "station_name", "lat", "lon", "city", "state", "rows", "last_seen"]]


def nearest_station(grid: pd.DataFrame, stations: pd.DataFrame, chunk: int = 20_000) -> tuple[np.ndarray, np.ndarray]:
    """Great-circle distance from every hex centroid to its nearest station (chunked brute force)."""
    glat, glon = np.radians(grid["lat"].to_numpy()), np.radians(grid["lon"].to_numpy())
    slat, slon = np.radians(stations["lat"].to_numpy()), np.radians(stations["lon"].to_numpy())
    dist = np.empty(len(grid))
    idx = np.empty(len(grid), dtype=int)
    for i in range(0, len(grid), chunk):
        la, lo = glat[i:i + chunk, None], glon[i:i + chunk, None]
        a = np.sin((slat - la) / 2) ** 2 + np.cos(la) * np.cos(slat) * np.sin((slon - lo) / 2) ** 2
        d = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        idx[i:i + chunk] = d.argmin(axis=1)
        dist[i:i + chunk] = d[np.arange(len(d)), idx[i:i + chunk]]
    return dist, stations["station_id"].to_numpy()[idx]


def display_geojson(grid: pd.DataFrame, res: int, display_res: int, path: Path) -> int:
    col = f"h3_r{display_res}"
    parents = grid[col] if col in grid else grid["h3"].map(lambda c: h3.cell_to_parent(c, display_res))
    agg = grid.assign(parent=parents).groupby("parent").agg(
        dist_km=("dist_km", "mean"), dist_km_min=("dist_km", "min"),
        station_count=("station_count", "sum"),
        state=("state", lambda s: s.mode().iat[0] if s.notna().any() else None),
    ).reset_index()
    features = []
    for row in agg.itertuples(index=False):
        ring = [[round(lng, 4), round(lat, 4)] for lat, lng in h3.cell_to_boundary(row.parent)]
        ring.append(ring[0])
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {"h3": row.parent, "dist_km": round(float(row.dist_km), 1),
                           "dist_km_min": round(float(row.dist_km_min), 1),
                           "station_count": int(row.station_count), "state": row.state},
        })
    fc = {"type": "FeatureCollection",
          "properties": {"source_res": res, "display_res": display_res,
                         "dist_km": "mean distance from member hex centroids to nearest station"},
          "features": features}
    path.write_text(json.dumps(fc, separators=(",", ":")), encoding="utf-8")
    return len(features)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--res", type=int, default=7, help="H3 resolution (default 7, ~5 km)")
    parser.add_argument("--display-res", type=int, default=5, help="coarser resolution for the GeoJSON (default 5)")
    parser.add_argument("--country", default="IND", help="ISO3 code of the cached geoBoundaries files")
    parser.add_argument("--stations", type=Path, default=STATIONS)
    parser.add_argument("--min-station-rows", type=int, default=0,
                        help="ignore stations with fewer readings than this in the history corpus")
    args = parser.parse_args(argv)
    if not 0 <= args.display_res <= args.res <= 15:
        parser.error("need 0 <= --display-res <= --res <= 15")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    grid = build_grid(args.country.upper(), args.res)

    stations = load_stations(args.stations)
    if args.min_station_rows:
        stations = stations[stations["rows"].fillna(0) >= args.min_station_rows].reset_index(drop=True)
    stations["h3"] = [h3.latlng_to_cell(la, lo, args.res) for la, lo in zip(stations["lat"], stations["lon"])]
    log.info("%d stations (min rows %d)", len(stations), args.min_station_rows)

    counts = stations["h3"].value_counts()
    grid["station_count"] = grid["h3"].map(counts).fillna(0).astype("int16")
    grid["dist_km"], grid["nearest_station_id"] = nearest_station(grid, stations)
    grid["dist_km"] = grid["dist_km"].astype("float32")

    full = OUT_DIR / f"grid_r{args.res}.parquet"
    slim = OUT_DIR / f"grid_r{args.res}_api.parquet"
    geo = OUT_DIR / f"grid_r{args.res}_display_r{args.display_res}.geojson"
    grid.to_parquet(full, index=False, compression="zstd")
    grid.drop(columns=["boundary"]).to_parquet(slim, index=False, compression="zstd")
    stations.to_parquet(OUT_DIR / "stations.parquet", index=False, compression="zstd")
    n_display = display_geojson(grid, args.res, args.display_res, geo)

    area = h3.average_hexagon_area(args.res, "km^2")
    outside = stations[~stations["h3"].isin(grid["h3"])]
    print("\n" + "=" * 72)
    print(f"H3 grid  {args.country}  res {args.res} (~{area:.1f} km2/hex)")
    print("=" * 72)
    print(f"hexes              {len(grid):,}  (~{len(grid) * area:,.0f} km2)")
    print(f"hexes w/o state    {grid['state'].isna().sum():,}   w/o district {grid['district'].isna().sum():,}")
    print(f"stations           {len(stations)} in {stations['h3'].nunique()} hexes"
          f"  ({len(outside)} fall outside the grid, still used for distance)")
    print(f"hexes with station {int((grid['station_count'] > 0).sum()):,}  "
          f"({100 * (grid['station_count'] > 0).mean():.3f}% of the country)")
    print("\ndistance to nearest station")
    for q, v in zip([50, 75, 90, 99], np.percentile(grid["dist_km"], [50, 75, 90, 99])):
        print(f"  p{q:<3} {v:8.1f} km")
    for km in (10, 25, 50, 100):
        print(f"  share of hexes > {km:>3} km from any station: {100 * (grid['dist_km'] > km).mean():5.1f}%")
    by_state = grid.groupby("state").agg(hexes=("h3", "size"), median_km=("dist_km", "median"),
                                         pct_over_50km=("dist_km", lambda d: 100 * (d > 50).mean()))
    by_state["stations"] = stations.groupby("state").size()
    by_state = by_state.fillna({"stations": 0}).astype({"stations": int}).sort_values("median_km")
    print("\nby state (sorted by median distance)")
    print(by_state.round(1).to_string())
    print(f"\nwrote {full.relative_to(ROOT)}, {slim.name}, {geo.name} ({n_display} display hexes), stations.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
