"""AeroGap API: serves the H3 grid and monitoring stations straight from Parquet.

    GET /grid?bbox=minLon,minLat,maxLon,maxLat[&res=4..7]
        Hexes whose centroid is inside the bbox, as columnar JSON.
        Without res, picks the finest resolution that stays under MAX_CELLS;
        coarser levels aggregate member hexes (mean distance, summed stations).
    GET /stations[?bbox=...]
    GET /health

No database: both files are loaded into memory at startup (~640k hexes).
Run locally:  uvicorn api.main:app --reload
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("AEROGAP_DATA", ROOT / "data" / "grid"))
GRID_RES = int(os.environ.get("AEROGAP_GRID_RES", "7"))
MAX_CELLS = int(os.environ.get("AEROGAP_MAX_CELLS", "60000"))
MIN_RES = 4

grid = pd.read_parquet(DATA_DIR / f"grid_r{GRID_RES}_api.parquet")
stations = pd.read_parquet(DATA_DIR / "stations.parquet")
_lat, _lon = grid["lat"].to_numpy(), grid["lon"].to_numpy()

app = FastAPI(
    title="AeroGap API",
    version="0.1.0",
    description="National H3 grid with distance to the nearest air quality monitor. "
                "Predicting the pollution your monitors miss.",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])


def parse_bbox(bbox: str | None) -> tuple[float, float, float, float] | None:
    if bbox is None:
        return None
    try:
        min_lon, min_lat, max_lon, max_lat = (float(v) for v in bbox.split(","))
    except ValueError:
        raise HTTPException(422, "bbox must be minLon,minLat,maxLon,maxLat") from None
    if min_lon > max_lon or min_lat > max_lat:
        raise HTTPException(422, "bbox min must be <= max")
    return min_lon, min_lat, max_lon, max_lat


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "hexes": len(grid), "stations": len(stations), "grid_res": GRID_RES}


@app.get("/grid")
def get_grid(
    bbox: str = Query(..., description="minLon,minLat,maxLon,maxLat", examples=["76.6,10.7,77.6,11.3"]),
    res: int | None = Query(None, ge=MIN_RES, le=GRID_RES, description="H3 resolution; omit to auto-select"),
) -> dict:
    min_lon, min_lat, max_lon, max_lat = parse_bbox(bbox)
    mask = (_lon >= min_lon) & (_lon <= max_lon) & (_lat >= min_lat) & (_lat <= max_lat)
    sub = grid[mask]

    if res is None:
        # Each coarser H3 level holds ~7x fewer cells.
        steps = math.ceil(math.log(len(sub) / MAX_CELLS, 7)) if len(sub) > MAX_CELLS else 0
        res = max(MIN_RES, GRID_RES - steps)

    if res == GRID_RES:
        out = sub[["h3", "dist_km", "station_count"]]
    else:
        out = (sub.groupby(f"h3_r{res}")
               .agg(dist_km=("dist_km", "mean"), station_count=("station_count", "sum"))
               .reset_index().rename(columns={f"h3_r{res}": "h3"}))
    if len(out) > MAX_CELLS * 7:
        raise HTTPException(413, f"{len(out)} cells; request a coarser res or smaller bbox")

    return {
        "res": res,
        "count": len(out),
        "h3": out["h3"].tolist(),
        "dist_km": np.round(out["dist_km"].to_numpy(dtype=float), 1).tolist(),
        "station_count": out["station_count"].astype(int).tolist(),
    }


@app.get("/stations")
def get_stations(bbox: str | None = Query(None, description="optional minLon,minLat,maxLon,maxLat")) -> dict:
    sub = stations
    box = parse_bbox(bbox)
    if box:
        min_lon, min_lat, max_lon, max_lat = box
        sub = sub[sub["lon"].between(min_lon, max_lon) & sub["lat"].between(min_lat, max_lat)]
    cols = ["station_id", "station_name", "lat", "lon", "city", "state", "rows", "last_seen"]
    records = sub[cols].assign(last_seen=sub["last_seen"].astype(str)).to_dict(orient="records")
    return {"count": len(records), "stations": records}
