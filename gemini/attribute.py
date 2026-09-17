"""Source attribution: a hex's signals -> likely dominant source, action, urgency and uncertainty.

This turns a predicted number into something an official can act on, but only as far as the
signals allow. Gemini gets a JSON of this hex's signals for one day (model prediction and its
validated confidence, satellite columns with national percentiles, upwind fires, wind, distance
to working monitors, nearby monitor health) and must:
  - use only those signals, cite the signal and value behind every claim, name no facilities;
  - answer "insufficient_signal" rather than guess;
  - say explicitly how uncertain the estimate is, and recommend verification, not enforcement,
    when confidence is low.
After the call, every cited signal is checked against the signals supplied; anything else is
recorded as a grounding warning. Every call goes through utils/llm_cache (gemini.common).

Demo hexes per date (fixed rule, not hand-picked): Delhi centre, Coimbatore centre, the two
highest predicted-PM2.5 dark zones (>50 km from a working monitor, different regions), and the
highest-PM2.5 hex >100 km from any reporting monitor (the low-confidence case).

Usage:
    python -m gemini.attribute                         # the three demo dates
    python -m gemini.attribute --offline               # replay only
Outputs: data/explanations/<date>.json, web/public/data/explanations_<date>.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path

import h3
import joblib
import numpy as np
import pandas as pd

from gemini.common import MODEL, generate_json
from model.features import haversine_km
from model.predict import CONFIDENCE, GRID_FILE, MODEL_FILE, confidence_levels, predict_day
from model.validate import load_corpus
from utils.llm_cache import text_part, usage

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "explanations"
WEB_DATA = ROOT / "web" / "public" / "data"
STATIONS = ROOT / "data" / "grid" / "stations.parquet"
DEFAULT_DATES = "2025-11-12,2025-11-22,2026-09-10"
CITY_CENTRES = {"delhi_centre": (28.6315, 77.2167), "coimbatore_centre": (11.0168, 76.9558)}
CATEGORIES = [(30, "Good"), (60, "Satisfactory"), (90, "Moderate"), (120, "Poor"), (250, "Very poor"), (math.inf, "Severe")]

# Batched: one call per date for all demo hexes. The free tier allows 20 requests/day per model
# (docs/gemini-notes.md), so one call per hex does not fit a day's budget.
PROMPT_VERSION = "attribute-batch-v1"
SYSTEM = """You are an air-quality analyst writing for district officials in India. You receive a JSON array of hexagons. Each item has a hex_id and the measured and modelled signals for that single ~5 km hexagon on one day.

Analyse every hexagon INDEPENDENTLY: use only that hexagon's own signals, never compare it with or borrow from another item. Return exactly one result per hex_id, with the same hex_id.

Grounding rules (strict):
- Use ONLY the signals in the JSON. Do not add facts about the place: no named industries, roads, events, crops, weather or history that are not in the JSON.
- Every claim in reasoning and evidence must name the signal it comes from and quote its value.
- A null value means the signal is unavailable (for satellite columns: cloud-covered, no retrieval). Never treat null as low.
- If the signals do not clearly point to a source, set dominant_source to "insufficient_signal". Saying so is better than guessing.
- Do not name specific facilities, companies or people.

How to read the signals (guidance, not facts about this place):
- NO2 column in the top national percentiles points to fuel combustion nearby (traffic, industry, power).
- Many upwind fires or high upwind fire radiative power, with an elevated aerosol index, points to biomass burning transported by wind.
- A high aerosol index with unremarkable NO2 and no upwind fires can indicate dust.
- SO2 in the top national percentiles points towards industrial or power-plant combustion.
- Low wind speed (under ~2 m/s) lets local emissions accumulate.

Confidence and action:
- model_confidence and distance_to_nearest_reporting_monitor_km describe how well constrained the PM2.5 estimate is. If model_confidence is "low", or the nearest reporting monitor is more than 100 km away, uncertainty_statement must say the estimate is weakly constrained, and recommended_action must be verification (for example a portable monitor or a site check), not enforcement.
- If nearby monitors are stuck or dead, say that ground truth nearby is unreliable.
- urgency: "urgent" only when the predicted category is Very poor or Severe AND model_confidence is not low; "elevated" for Poor, or for Very poor/Severe with low confidence; otherwise "routine".
- recommended_action: one or two proportionate steps an official could take this week.
- uncertainty_statement is always required: state what is uncertain and why, in one or two sentences."""
USER_PROMPT = "Hexagons:\n"
HEX_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "hex_id": {"type": "STRING"},
        "dominant_source": {"type": "STRING", "enum": ["vehicular", "industrial", "biomass_burning", "dust",
                                                       "mixed_combustion", "insufficient_signal"]},
        "source_confidence": {"type": "STRING", "enum": ["high", "medium", "low"]},
        "evidence": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "signal": {"type": "STRING", "description": "exact key name from the input JSON"},
            "value": {"type": "STRING"}, "interpretation": {"type": "STRING"}},
            "required": ["signal", "value", "interpretation"]}},
        "reasoning": {"type": "STRING", "description": "2-3 sentences, each citing named signals and values"},
        "recommended_action": {"type": "STRING"},
        "urgency": {"type": "STRING", "enum": ["routine", "elevated", "urgent"]},
        "uncertainty_statement": {"type": "STRING"},
    },
    "required": ["hex_id", "dominant_source", "source_confidence", "evidence", "reasoning", "recommended_action",
                 "urgency", "uncertainty_statement"],
}
SCHEMA = {"type": "OBJECT", "properties": {"results": {"type": "ARRAY", "items": HEX_SCHEMA}}, "required": ["results"]}
REQUIRED = ["results"]
HEX_REQUIRED = HEX_SCHEMA["required"]


def category(pm: float) -> str:
    return next(label for upper, label in CATEGORIES if pm <= upper)


def compass(u: float, v: float) -> str | None:
    if not (np.isfinite(u) and np.isfinite(v)):
        return None
    deg = (math.degrees(math.atan2(-u, -v)) + 360) % 360  # meteorological: direction the wind blows FROM
    points = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return f"{points[int((deg + 22.5) // 45) % 8]} ({deg:.0f}°)"


def percentile_of(series: pd.Series, value: float) -> int | None:
    if value is None or not np.isfinite(value):
        return None
    valid = series.dropna().to_numpy()
    return int(round(100 * (valid < value).mean())) if valid.size else None


def clean(v, digits: int = 3):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return None if not np.isfinite(f) else round(f, digits)


def select_hexes(df: pd.DataFrame) -> list[tuple[str, int]]:
    """(role, row index) per the fixed selection rule."""
    picks, used_regions = [], set()
    for role, (lat, lon) in CITY_CENTRES.items():
        idx = df.index[df["h3"] == h3.latlng_to_cell(lat, lon, 7)]
        if len(idx):
            picks.append((role, int(idx[0])))
            used_regions.add(df.at[idx[0], "h3_r4"])

    def top(mask: pd.Series, n: int, role: str):
        for i in df[mask].sort_values("pm25", ascending=False).index:
            if len([p for p in picks if p[0] == role]) == n:
                return
            if df.at[i, "h3_r4"] not in used_regions:
                picks.append((role, int(i)))
                used_regions.add(df.at[i, "h3_r4"])

    top(df["dist_working_km"] > 50, 2, "dark_zone")
    top(df["confidence"] == 2, 1, "low_confidence")
    return picks


def build_signals(df: pd.DataFrame, i: int, day: date, stations: pd.DataFrame, grid_meta: pd.DataFrame,
                  levels: list[dict]) -> dict:
    r = df.loc[i]
    lat, lon = h3.cell_to_latlng(r["h3"])
    d = haversine_km(lat, lon, stations["lat"].to_numpy(), stations["lon"].to_numpy())
    near = stations.assign(distance_km=d)
    within = near[near["distance_km"] <= 50]
    nearest = near.nsmallest(3, "distance_km")
    meta = grid_meta.loc[r["h3"]] if r["h3"] in grid_meta.index else None
    level = int(r["confidence"])
    return {
        "date": day.isoformat(),
        "location": {"h3": r["h3"], "district": None if meta is None else meta["district"],
                     "state": None if meta is None else meta["state"]},
        "predicted_pm25_ug_m3": clean(r["pm25"], 1),
        "predicted_category": category(float(r["pm25"])),
        "model_confidence": CONFIDENCE[level],
        "prediction_typical_error_pct": levels[level]["rel_error_pct"],
        "distance_to_nearest_reporting_monitor_km": clean(r["dist_nearest_km"], 1),
        "distance_to_nearest_working_monitor_km": clean(r["dist_working_km"], 1),
        "no2_tropospheric_column_mol_m2": clean(r["no2_trop"], 7),
        "no2_national_percentile_today": percentile_of(df["no2_trop"], r["no2_trop"]),
        "aerosol_index": clean(r["aai"], 2),
        "aerosol_index_national_percentile_today": percentile_of(df["aai"], r["aai"]),
        "co_column_mol_m2": clean(r["co"], 5),
        "co_national_percentile_today": percentile_of(df["co"], r["co"]),
        "so2_column_mol_m2": clean(r["so2"], 6),
        "so2_national_percentile_today": percentile_of(df["so2"], r["so2"]),
        "fires_in_this_hex": int(r["fire_count"]),
        "fires_within_150km": clean(r["fires_150km"], 0),
        "upwind_fires_within_150km": clean(r["upwind_fires_150km"], 1),
        "upwind_fire_radiative_power_mw": clean(r["upwind_frp_150km"], 1),
        "upwind_fires_national_percentile_today": percentile_of(df["upwind_fires_150km"], r["upwind_fires_150km"]),
        "wind_speed_m_s": clean(r["wind_speed"], 1),
        "wind_from": compass(r["u10"], r["v10"]),
        "monitors_within_50km_by_health": within["health"].value_counts().to_dict() if len(within) else {},
        "nearest_monitors": [{"name": s.station_name, "distance_km": round(float(s.distance_km), 1),
                              "health": s.health} for s in nearest.itertuples()],
    }


def grounding_warnings(signals: dict, attribution: dict) -> list[str]:
    allowed = set(signals) | {f"location.{k}" for k in signals["location"]} | {"location"}
    return [f"evidence cites unknown signal '{e.get('signal')}'" for e in attribution.get("evidence", [])
            if e.get("signal") not in allowed]


def attribute_batch(signals_by_hex: dict[str, dict], use_cache: bool = True, offline: bool | None = None) -> dict[str, dict]:
    """One Gemini call for all hexes of a date. Returns hex_id -> attribution (or a per-hex failure status)."""
    items = [{"hex_id": cell, "signals": s} for cell, s in signals_by_hex.items()]
    result = generate_json([text_part(USER_PROMPT + json.dumps(items, ensure_ascii=False, indent=1))],
                           schema=SCHEMA, prompt_version=PROMPT_VERSION, system_instruction=SYSTEM,
                           required=REQUIRED, use_cache=use_cache, offline=offline)
    if result["status"] != "ok":
        failure = {"status": result["status"], "detail": result.get("error") or result.get("failures")}
        return {cell: failure for cell in signals_by_hex}

    by_id = {r.get("hex_id"): r for r in result["data"]["results"] if isinstance(r, dict)}
    out = {}
    for cell, signals in signals_by_hex.items():
        a = by_id.get(cell)
        if a is None or not all(k in a for k in HEX_REQUIRED):
            out[cell] = {"status": "missing_from_batch"}
            continue
        out[cell] = {"status": "ok", "cached": result["cached"], "model": MODEL, "prompt_version": PROMPT_VERSION,
                     **{k: v for k, v in a.items() if k != "hex_id"}, "grounding_warnings": grounding_warnings(signals, a)}
    extra = set(by_id) - set(signals_by_hex)
    if extra:
        for cell in out:
            if out[cell]["status"] == "ok":
                out[cell]["grounding_warnings"].append(f"batch reply also contained unknown hex ids: {sorted(extra)}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dates", default=DEFAULT_DATES)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="build signals and select hexes, make no Gemini call (attribution marked pending)")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    offline = True if args.offline else None

    bundle = joblib.load(MODEL_FILE)
    levels = confidence_levels(bundle)
    grid = pd.read_parquet(GRID_FILE, columns=["h3", "h3_r4", "h3_r5", "lat", "lon", "dist_working_km"])
    grid_meta = pd.read_parquet(GRID_FILE, columns=["h3", "district", "state"]).set_index("h3")
    stations = pd.read_parquet(STATIONS, columns=["station_id", "station_name", "lat", "lon", "health"])
    corpus = load_corpus("pm25", "D", 7)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for text in args.dates.split(","):
        day = date.fromisoformat(text.strip())
        df = predict_day(bundle, day, grid, corpus).reset_index(drop=True)
        hexes = {}
        print(f"\n=== {day}")
        for role, i in select_hexes(df):
            cell = df.at[i, "h3"]
            lat, lon = h3.cell_to_latlng(cell)
            hexes[cell] = {"role": role, "h3_r5": df.at[i, "h3_r5"], "h3_r4": df.at[i, "h3_r4"],
                           "lat": round(lat, 4), "lon": round(lon, 4),
                           "signals": build_signals(df, i, day, stations, grid_meta, levels)}
        if args.dry_run:
            attributions = {cell: {"status": "pending"} for cell in hexes}
        else:
            attributions = attribute_batch({cell: e["signals"] for cell, e in hexes.items()},
                                           use_cache=not args.no_cache, offline=offline)
        for cell, entry in hexes.items():
            entry["attribution"] = attribution = attributions[cell]
            role, signals = entry["role"], entry["signals"]
            loc = signals["location"]
            print(f"  {role:<18} {loc['district']}, {loc['state']}: PM2.5 {signals['predicted_pm25_ug_m3']} "
                  f"({signals['model_confidence']} confidence, {signals['distance_to_nearest_reporting_monitor_km']} km) "
                  f"-> {attribution.get('dominant_source', attribution['status'])} / {attribution.get('urgency', '-')} "
                  f"| cached={attribution.get('cached')} | grounding warnings: {len(attribution.get('grounding_warnings', []))}")
        payload = {"date": day.isoformat(), "model": MODEL, "prompt_version": PROMPT_VERSION, "hexes": hexes}
        (OUT_DIR / f"{day.isoformat()}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        (WEB_DATA / f"explanations_{day.isoformat()}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print("\ngemini usage:", usage())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
