"""Static snapshot of the API's responses, so the map can run on a free static host.

Hugging Face now bills Docker Spaces (402 Payment Required on cpu-basic), and the
live URL is a mandatory submission item, so the frontend falls back to these files
when VITE_API_URL is unset. Same payload shape as GET /grid and GET /stations.

Written to web/public/data/:
    meta.json          what exists, plus each city's bbox, so the map can choose a file
    grid_r4/5/6.json   national, aggregated (mean distances, summed station counts)
    city_<id>.json     full H3 res-7 detail around each demo city
    stations.json      every station with its health

Usage:
    python -m api.export_static
    python -m api.export_static --city-halfwidth 2.0 --city-halfheight 1.5
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GRID_DIR = ROOT / "data" / "grid"
OUT_DIR = ROOT / "web" / "public" / "data"
CITIES = ROOT / "web" / "src" / "cities.json"
NATIONAL_RES = (4, 5, 6)
DIST_COLS = ("dist_working_km", "dist_km")


def write_json(path: Path, payload: dict | list) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return path.stat().st_size


def grid_payload(df: pd.DataFrame, res: int, parent_col: str | None) -> dict:
    if parent_col is None:
        out = df[["h3", *DIST_COLS, "station_count"]]
    else:
        out = (df.groupby(parent_col)
               .agg(**{c: (c, "mean") for c in DIST_COLS}, station_count=("station_count", "sum"))
               .reset_index().rename(columns={parent_col: "h3"}))
    payload = {"res": res, "count": len(out), "h3": out["h3"].tolist(),
               "station_count": out["station_count"].astype(int).tolist()}
    for c in DIST_COLS:
        payload[c] = out[c].round(1).tolist()
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--res", type=int, default=7, help="resolution of the source grid")
    parser.add_argument("--city-halfwidth", type=float, default=1.5, help="degrees of longitude each side of a city")
    parser.add_argument("--city-halfheight", type=float, default=1.2, help="degrees of latitude each side of a city")
    args = parser.parse_args(argv)

    grid = pd.read_parquet(GRID_DIR / f"grid_r{args.res}_api.parquet")
    if "dist_working_km" not in grid.columns:  # grid built before station health existed
        grid["dist_working_km"] = grid["dist_km"]
    stations = pd.read_parquet(GRID_DIR / "stations.parquet")

    files = {}
    for res in NATIONAL_RES:
        name = f"grid_r{res}.json"
        size = write_json(OUT_DIR / name, grid_payload(grid, res, f"h3_r{res}"))
        files[str(res)] = name
        print(f"  {name:<16} {size/1e6:6.2f} MB")

    cities = []
    for city in json.loads(CITIES.read_text(encoding="utf-8")):
        if city["id"] == "india":
            continue
        west, east = city["longitude"] - args.city_halfwidth, city["longitude"] + args.city_halfwidth
        south, north = city["latitude"] - args.city_halfheight, city["latitude"] + args.city_halfheight
        sub = grid[grid["lon"].between(west, east) & grid["lat"].between(south, north)]
        name = f"city_{city['id']}.json"
        size = write_json(OUT_DIR / name, grid_payload(sub, args.res, None))
        cities.append({"id": city["id"], "label": city["label"], "bbox": [west, south, east, north],
                       "res": args.res, "file": name})
        print(f"  {name:<16} {size/1e6:6.2f} MB  ({len(sub):,} hexes)")

    cols = [c for c in ("station_id", "station_name", "lat", "lon", "city", "state", "rows",
                        "health", "health_reason", "usable_coverage") if c in stations.columns]
    st = stations[cols].round({"lat": 5, "lon": 5, "usable_coverage": 3})
    size = write_json(OUT_DIR / "stations.json", st.astype(object).where(st.notna(), None).to_dict(orient="records"))
    print(f"  {'stations.json':<16} {size/1e6:6.2f} MB  ({len(st):,} stations)")

    write_json(OUT_DIR / "meta.json", {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "grid_res": args.res, "hexes": len(grid), "stations": len(stations),
        "station_health": stations["health"].value_counts().to_dict() if "health" in stations else None,
        "national": files, "cities": cities, "stations_file": "stations.json",
    })
    print(f"wrote {OUT_DIR.relative_to(ROOT)}/ (national res {list(NATIONAL_RES)}, {len(cities)} city files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
