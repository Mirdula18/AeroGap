"""Daily PM2.5 for every hex from the promoted model, with empirical confidence, for the map.

This is deployment, not evaluation: every station that reported that day is a
neighbour. Uncertainty is not invented here. Each hex takes the validated MAE of the
distance band it falls in (distance to the nearest reporting station), taken from
the pre-registered evaluation stored in model/model.pkl.

Station-independent features follow exactly the rules in model/features.py (fire
counts filled with 0, upwind counts 0 when no fire is within 150 km). A check
compares them against the training table for station hexes, to catch train/serve skew.

Usage:
    python -m model.predict                                   # default demo dates
    python -m model.predict --dates 2025-11-12,2026-01-15
Writes web/public/data/pred_<date>_r4.json, _r5.json, _city_<id>.json and a "predictions" block in meta.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from model.features import FIRE_HEX_COLS, SATELLITE_BANDS, STATION_HEX_TABLE, haversine_km
from model.validate import BANDS, load_corpus

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = ROOT / "model" / "model.pkl"
GRID_FILE = ROOT / "data" / "grid" / "grid_r7_api.parquet"
SAT_DIR = ROOT / "data" / "satellite" / "features_r7"
FIRE_HEX = ROOT / "data" / "fires" / "hex_daily_r7.parquet"
FIRE_REGION = ROOT / "data" / "fires" / "upwind_daily_r5.parquet"
WEB_DATA = ROOT / "web" / "public" / "data"
DEFAULT_DATES = "2025-11-12,2026-01-15,2026-09-10"
# Three confidence levels, in RELATIVE error. Validated MAE in µg/m³ is not monotonic across bands
# (50-100 km scores 15.0 vs 18.6 at 20-50 km only because those stations are less polluted), while
# relative error is: ~31% within 20 km, ~38% from 20 to 100 km (the two bands tie), ~59% beyond.
CONFIDENCE = ["high", "medium", "low"]
BAND_LEVEL = {"0-20 km": 0, "20-50 km": 1, "50-100 km": 1, "100 km+": 2}

log = logging.getLogger("model.predict")


def day_features(day: date, grid: pd.DataFrame) -> pd.DataFrame:
    month_file = SAT_DIR / f"month={day:%Y-%m}" / "part-0.parquet"
    sat = ds.dataset(month_file, format="parquet").to_table(
        filter=ds.field("date") == day, columns=["h3"] + SATELLITE_BANDS + ["u10", "v10"]).to_pandas()
    df = grid[["h3", "h3_r4", "h3_r5", "lat", "lon", "dist_working_km"]].merge(sat, on="h3", how="left")
    speed = np.hypot(df["u10"], df["v10"])
    df["wind_speed"] = speed
    with np.errstate(invalid="ignore", divide="ignore"):
        df["wind_dir_sin"], df["wind_dir_cos"] = df["v10"] / speed, df["u10"] / speed

    ts = pd.Timestamp(day)
    own = ds.dataset(FIRE_HEX, format="parquet").to_table(filter=ds.field("date") == ts).to_pandas()
    region = ds.dataset(FIRE_REGION, format="parquet").to_table(filter=ds.field("date") == ts).to_pandas()
    df = df.merge(own.drop(columns=["date"]), on="h3", how="left")
    df[FIRE_HEX_COLS] = df[FIRE_HEX_COLS].fillna(0)
    df = df.merge(region.drop(columns=["date"]), on="h3_r5", how="left")
    df[["fires_150km", "frp_150km"]] = df[["fires_150km", "frp_150km"]].fillna(0)
    no_fire = df["fires_150km"] == 0
    for col in ("upwind_fires_150km", "upwind_frp_150km"):
        df.loc[no_fire & df[col].isna(), col] = 0.0

    doy = ts.dayofyear
    df["doy_sin"], df["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    df["month"], df["weekday"] = ts.month, ts.weekday()
    return df


def check_skew(day: date, df: pd.DataFrame) -> str:
    """Station-independent features must match the training table for station hexes on the same day."""
    table = pd.read_parquet(STATION_HEX_TABLE)
    ref = table[table["date"] == pd.Timestamp(day)].set_index("h3")
    common = ref.index.intersection(df["h3"])[:200]
    if len(common) == 0:
        return "no station hexes to compare"
    ours = df.set_index("h3").loc[common]
    cols = SATELLITE_BANDS + ["wind_speed", "fire_count", "fires_150km", "upwind_fires_150km"]
    worst = 0.0
    for c in cols:
        a, b = ours[c].to_numpy(dtype=float), ref.loc[common, c].to_numpy(dtype=float)
        both_nan = np.isnan(a) & np.isnan(b)
        diff = np.where(both_nan, 0.0, np.abs(a - b))
        worst = max(worst, np.nanmax(np.where(np.isnan(diff), np.inf, diff / np.maximum(np.abs(b), 1e-9))))
    return f"{len(common)} station hexes, worst relative feature difference {worst:.2e}"


def neighbour_features_many(tlat, tlon, slat, slon, svals, k: int = 8, power: float = 2.0,
                            chunk: int = 20000) -> pd.DataFrame:
    """Vectorised twin of model.features.neighbour_features for many targets and one day."""
    n = len(tlat)
    out = {c: np.full(n, np.nan) for c in ("neighbour_idw", "neighbour_nearest", "dist_nearest_km", "neighbour_spread")}
    kk = min(k, len(svals))
    for i in range(0, n, chunk):
        d = haversine_km(tlat[i:i + chunk, None], tlon[i:i + chunk, None], slat[None, :], slon[None, :])
        part = np.argpartition(d, kk - 1, axis=1)[:, :kk]
        dk = np.take_along_axis(d, part, axis=1)
        vk = svals[part]
        w = 1.0 / np.maximum(dk, 0.5) ** power
        out["neighbour_idw"][i:i + chunk] = (w * vk).sum(axis=1) / w.sum(axis=1)
        nearest = dk.argmin(axis=1)
        out["neighbour_nearest"][i:i + chunk] = vk[np.arange(len(vk)), nearest]
        out["dist_nearest_km"][i:i + chunk] = dk.min(axis=1)
        out["neighbour_spread"][i:i + chunk] = vk.std(axis=1)
    frame = pd.DataFrame(out)
    frame["n_reporting"] = len(svals)
    return frame


def predict_bundle(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    """Mean of the fold models (each saw ~80% of hexes); residual models add back the IDW anchor."""
    if len(X) == 0:
        return np.array([])
    preds = np.mean([m.predict(X[bundle["features"]]) for m in bundle["models"].values()], axis=0)
    if bundle["target_mode"] == "residual":
        preds = X["neighbour_idw"].to_numpy(dtype=float) + preds
    return preds


def predict_day(bundle: dict, day: date, grid: pd.DataFrame, corpus) -> pd.DataFrame:
    df = day_features(day, grid)
    ts = pd.Timestamp(day)
    if ts not in corpus.values.index:
        raise SystemExit(f"{day}: no station data that day")
    reporting = corpus.values.loc[ts].dropna()
    st = corpus.stations.loc[reporting.index]
    neigh = neighbour_features_many(df["lat"].to_numpy(), df["lon"].to_numpy(), st["hex_lat"].to_numpy(),
                                    st["hex_lon"].to_numpy(), reporting.to_numpy(dtype=float))
    X = pd.concat([df.reset_index(drop=True), neigh], axis=1)
    d = X["dist_nearest_km"].to_numpy()

    if bundle["type"] == "combined":
        near = d <= bundle["threshold_km"]
        pm = np.empty(len(X))
        pm[near] = predict_bundle(bundle["near"], X[near])
        pm[~near] = predict_bundle(bundle["far"], X[~near])
    else:
        pm = predict_bundle(bundle, X)
    X["pm25"] = np.clip(pm, 0, None)

    band_idx = np.searchsorted([b[1] for b in BANDS[:-1]], d, side="right")  # 0..3
    levels = confidence_levels(bundle)
    level = np.array([BAND_LEVEL[BANDS[i][2]] for i in band_idx])
    rel_error = np.array([levels[l]["rel_error_pct"] for l in range(len(levels))])[level] / 100
    X["confidence"] = level
    X["uncertainty"] = X["pm25"] * rel_error  # typical error in µg/m³, scaled to this hex's predicted level
    X["reporting_stations"] = len(reporting)
    return X


def confidence_levels(bundle: dict) -> list[dict]:
    """Relative validated error per confidence level, pooling bands by station-days."""
    unc = bundle["uncertainty_by_band"]
    out = []
    for level, label in enumerate(CONFIDENCE):
        bands = [b for b, lvl in BAND_LEVEL.items() if lvl == level]
        days = sum(unc[b]["station_days"] for b in bands)
        rel = sum(unc[b]["nmae_pct"] * unc[b]["station_days"] for b in bands) / days
        out.append({"level": level, "label": label, "bands": bands, "rel_error_pct": round(rel, 1)})
    return out


def payload(df: pd.DataFrame, day: date, res: int, key: str | None) -> dict:
    if key is None:
        out = df[["h3", "pm25", "uncertainty", "dist_nearest_km", "confidence"]]
    else:
        out = (df.groupby(key).agg(pm25=("pm25", "mean"), uncertainty=("uncertainty", "mean"),
                                   dist_nearest_km=("dist_nearest_km", "mean"), confidence=("confidence", "max"))
               .reset_index().rename(columns={key: "h3"}))
    return {"date": day.isoformat(), "res": res, "count": len(out), "h3": out["h3"].tolist(),
            "pm25": out["pm25"].round(1).tolist(), "uncertainty": out["uncertainty"].round(1).tolist(),
            "dist_km": out["dist_nearest_km"].round(1).tolist(), "confidence": out["confidence"].astype(int).tolist()}


def write_json(name: str, data: dict) -> float:
    path = WEB_DATA / name
    path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    return path.stat().st_size / 1e6


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dates", default=DEFAULT_DATES)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    bundle = joblib.load(MODEL_FILE)
    if "decision" not in bundle:
        raise SystemExit("model/model.pkl has not been through model/decide.py")
    grid = pd.read_parquet(GRID_FILE, columns=["h3", "h3_r4", "h3_r5", "lat", "lon", "dist_working_km"])
    grid["h3_r4"] = grid["h3_r4"].astype(str)
    corpus = load_corpus("pm25", "D", 7)
    meta_path = WEB_DATA / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    entries = []
    for text in args.dates.split(","):
        day = date.fromisoformat(text.strip())
        df = predict_day(bundle, day, grid, corpus)
        log.info("%s: %s", day, check_skew(day, df))
        stem = f"pred_{day.isoformat()}"
        files = {"national": {"4": f"{stem}_r4.json", "5": f"{stem}_r5.json"}, "cities": {}}
        size = write_json(files["national"]["4"], payload(df, day, 4, "h3_r4"))
        size += write_json(files["national"]["5"], payload(df, day, 5, "h3_r5"))
        for city in meta["cities"]:
            w, s, e, n = city["bbox"]
            sub = df[df["lon"].between(w, e) & df["lat"].between(s, n)]
            files["cities"][city["id"]] = f"{stem}_city_{city['id']}.json"
            size += write_json(files["cities"][city["id"]], payload(sub, day, 7, None))
        entries.append({"date": day.isoformat(), "reporting_stations": int(df["reporting_stations"].iloc[0]), **files})

        poor = (df["pm25"] > 90).mean() * 100
        low_conf = (df["confidence"] >= 2).mean() * 100  # beyond 100 km of any reporting monitor
        dark = ((df["pm25"] > 90) & (df["dist_working_km"] > 50)).mean() * 100
        print(f"{day}: {int(df['reporting_stations'].iloc[0])} stations reporting | national median "
              f"{df['pm25'].median():.0f} µg/m³ | {poor:.1f}% of hexes above 90 (Poor+) | "
              f"{low_conf:.1f}% low confidence | {dark:.1f}% dark zone (Poor+ and >50 km from a working "
              f"monitor) | {size:.1f} MB")

    meta["predictions"] = {
        "model": bundle["decision"]["promoted"], "promoted_utc": bundle.get("promoted_utc"),
        "confidence_levels": confidence_levels(bundle),
        "dates": entries}
    meta_path.write_text(json.dumps(meta, separators=(",", ":")), encoding="utf-8")
    print(f"wrote predictions for {len(entries)} dates to {WEB_DATA.relative_to(ROOT)}/ and updated meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
