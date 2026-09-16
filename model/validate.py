"""Leave-station-out validation harness.

AeroGap's claim is spatial: it predicts air quality where NO station exists.
So the evaluation hides stations, never time periods:

    for each station S with >= --min-coverage hourly coverage:
        hide S, plus any other station in S's H3 hex (so the target hex has no monitor)
        give the predictor ONLY the remaining stations
        predict S's hex centroid for every period S reported
        compare against S's actual readings
    report error grouped by distance from S's hex to the nearest remaining station

There is deliberately no time split anywhere in this file. A random or
chronological split leaves the target station in the training data and grades
interpolation as if it were forecasting at a known site - it measures nothing
about dark zones.

Plugging in a model
-------------------
A predictor is any object with a `name` and
    predict(train: TrainingData, target: Target) -> pd.Series   # index = target.periods
`train` holds only the remaining stations. Satellite / fire / wind features may
be looked up by `target.h3` and period. Station-derived features (nearest-station
value, distance to nearest station, ...) MUST be recomputed from
`train.stations` / `train.values`: the precomputed data/grid dist_km includes
the held-out station and would leak it.

Built-in baselines (so the harness runs before the model exists, and so the
model has a bar to beat): `nearest_station` and inverse-distance weighting.

Usage:
    python -m model.validate                              # all eligible stations, daily PM2.5
    python -m model.validate --only openaq-8914           # SIDCO Kurichi, Coimbatore
    python -m model.validate --freq h --models idw
Outputs: data/validation/<parameter>_<freq>/{per_station.csv, per_period.parquet, bands.csv, bands.md}
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import h3
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from ingest.openaq import USABLE_QUALITY

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "data" / "history"
OUT_DIR = ROOT / "data" / "validation"
EARTH_RADIUS_KM = 6371.0088

BANDS = [(0.0, 20.0, "0-20 km"), (20.0, 50.0, "20-50 km"), (50.0, 100.0, "50-100 km"), (100.0, math.inf, "100 km+")]
MIN_HOURS_PER_DAY = 18  # CPCB: a 24-hour average needs >= 75% of hours
# CPCB NAQI 24-hour PM2.5 category edges (ug/m3): Good | Satisfactory | Moderate | Poor | Very Poor | Severe
PM25_CATEGORY_EDGES = [30, 60, 90, 120, 250]

log = logging.getLogger("model.validate")


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def band_of(km: float) -> str:
    return next(label for lo, hi, label in BANDS if lo <= km < hi)


# --------------------------------------------------------------------------- data


@dataclass
class Corpus:
    values: pd.DataFrame     # index: period, columns: station_id
    coverage: pd.Series      # station_id -> share of hours in the corpus window with a reading
    stations: pd.DataFrame   # index: station_id; station_name, city, state, lat, lon, h3, hex_lat, hex_lon


def load_corpus(parameter: str, freq: str, res: int) -> Corpus:
    dataset = ds.dataset(HISTORY, format="parquet", partitioning="hive")
    # Usable rows only: stuck placeholders and day-long flat runs are flagged in the corpus, filtered here.
    t = dataset.to_table(filter=(ds.field("parameter") == parameter)
                         & ds.field("quality_flag").isin(list(USABLE_QUALITY)),
                         columns=["station_id", "timestamp_utc", "value"]).to_pandas()
    if t.empty:
        sys.exit(f"no {parameter} readings in {HISTORY}")

    t["hour"] = t["timestamp_utc"].dt.floor("h")
    hourly = t.groupby(["station_id", "hour"])["value"].mean()
    hours = hourly.index.get_level_values("hour")
    window_hours = (hours.max() - hours.min()).total_seconds() / 3600 + 1
    coverage = hourly.groupby(level="station_id").size() / window_hours

    if freq == "h":
        values = hourly.unstack("station_id")
    else:  # IST calendar days, CPCB-style 24-hour means
        h = hourly.reset_index()
        h["period"] = h["hour"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None).dt.floor("D")
        daily = h.groupby(["station_id", "period"])["value"].agg(["mean", "size"])
        values = daily.loc[daily["size"] >= MIN_HOURS_PER_DAY, "mean"].unstack("station_id")

    meta = pd.read_parquet(HISTORY / "_stations.parquet").set_index("station_id")
    meta = meta.loc[meta.index.intersection(values.columns),
                    ["station_name", "city", "state", "lat", "lon"]].copy()
    meta["h3"] = [h3.latlng_to_cell(la, lo, res) for la, lo in zip(meta["lat"], meta["lon"])]
    centroids = np.array([h3.cell_to_latlng(c) for c in meta["h3"]])
    meta["hex_lat"], meta["hex_lon"] = centroids[:, 0], centroids[:, 1]
    values = values[meta.index]
    log.info("%s %s: %d stations, %d periods (%s -> %s)", parameter, freq, len(meta), len(values),
             values.index.min(), values.index.max())
    return Corpus(values=values, coverage=coverage.reindex(meta.index), stations=meta)


# --------------------------------------------------------------------------- predictors


@dataclass(frozen=True)
class TrainingData:
    values: pd.DataFrame     # remaining stations only
    stations: pd.DataFrame   # remaining stations only


@dataclass(frozen=True)
class Target:
    h3: str
    lat: float               # hex centroid, not the hidden station's coordinates
    lon: float
    periods: pd.Index


class Predictor(Protocol):
    name: str

    def predict(self, train: TrainingData, target: Target) -> pd.Series: ...


def _sorted_by_distance(train: TrainingData, target: Target) -> tuple[np.ndarray, np.ndarray]:
    dist = haversine_km(target.lat, target.lon, train.stations["hex_lat"], train.stations["hex_lon"])
    order = np.argsort(dist)
    arr = train.values.reindex(index=target.periods, columns=train.stations.index[order]).to_numpy(dtype=float)
    return arr, dist[order]


class NearestStation:
    """Value of the closest remaining station that reported in that period."""

    name = "nearest_station"

    def predict(self, train: TrainingData, target: Target) -> pd.Series:
        arr, _ = _sorted_by_distance(train, target)
        valid = ~np.isnan(arr)
        out = arr[np.arange(len(arr)), valid.argmax(axis=1)]
        out[~valid.any(axis=1)] = np.nan
        return pd.Series(out, index=target.periods)


class InverseDistance:
    """Inverse-distance-weighted mean of the k closest remaining stations reporting in that period."""

    def __init__(self, k: int = 8, power: float = 2.0):
        self.k, self.power = k, power
        self.name = f"idw_k{k}"

    def predict(self, train: TrainingData, target: Target) -> pd.Series:
        arr, dist = _sorted_by_distance(train, target)
        valid = ~np.isnan(arr)
        use = valid & (np.cumsum(valid, axis=1) <= self.k)
        w = np.where(use, 1.0 / np.maximum(dist, 0.5) ** self.power, 0.0)
        den = w.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = (np.where(use, arr, 0.0) * w).sum(axis=1) / den
        out[den == 0] = np.nan
        return pd.Series(out, index=target.periods)


PREDICTORS = {"nearest": NearestStation, "idw": InverseDistance}


# --------------------------------------------------------------------------- harness


def _category(values: np.ndarray) -> np.ndarray:
    return np.searchsorted(PM25_CATEGORY_EDGES, values, side="right")


def leave_station_out(corpus: Corpus, predictors: list, targets: list[str],
                      categories: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    values, meta = corpus.values, corpus.stations
    station_rows, period_frames = [], []

    for sid in targets:
        hex_id = meta.at[sid, "h3"]
        hidden = meta.index[meta["h3"] == hex_id]           # the target hex must have no monitor
        remaining = meta.drop(index=hidden)
        train = TrainingData(values=values.drop(columns=hidden), stations=remaining)
        # The held-out station's readings must never reach a predictor.
        assert sid not in train.values.columns and sid not in train.stations.index

        actual = values[sid].dropna()
        target = Target(h3=hex_id, lat=meta.at[sid, "hex_lat"], lon=meta.at[sid, "hex_lon"], periods=actual.index)
        dist = haversine_km(target.lat, target.lon, remaining["hex_lat"], remaining["hex_lon"])
        nearest_km = float(dist.min())
        # Band by the nearest remaining station that actually REPORTED in each period:
        # a neighbour with 5% coverage 13 km away must not make a 40 km gap look like 13 km.
        order = np.argsort(dist)
        rem_vals = values[remaining.index[order]].reindex(actual.index).to_numpy(dtype=float)
        reporting = ~np.isnan(rem_vals)
        reporting_km = np.where(reporting.any(axis=1), dist[order][reporting.argmax(axis=1)], np.nan)
        period_bands = [band_of(k) if not np.isnan(k) else "no remaining data" for k in reporting_km]
        median_reporting_km = float(np.nanmedian(reporting_km)) if reporting.any() else np.nan

        for predictor in predictors:
            pred = predictor.predict(train, target).reindex(actual.index)
            ok = pred.notna().to_numpy()
            a, p = actual.to_numpy()[ok], pred.to_numpy()[ok]
            err = p - a
            row = {
                "station_id": sid, "station_name": meta.at[sid, "station_name"],
                "city": meta.at[sid, "city"], "state": meta.at[sid, "state"],
                "model": predictor.name, "coverage": round(float(corpus.coverage[sid]), 3),
                "hidden_in_same_hex": len(hidden) - 1,
                "nearest_remaining_km": round(nearest_km, 1),
                "median_nearest_reporting_km": round(median_reporting_km, 1),
                "band": band_of(median_reporting_km) if not np.isnan(median_reporting_km) else "no remaining data",
                "periods": len(actual), "predicted": int(ok.sum()),
                "mae": float(np.abs(err).mean()) if ok.any() else np.nan,
                "bias": float(err.mean()) if ok.any() else np.nan,
                "rmse": float(np.sqrt((err ** 2).mean())) if ok.any() else np.nan,
                "r": float(np.corrcoef(a, p)[0, 1]) if ok.sum() > 2 and a.std() > 0 and p.std() > 0 else np.nan,
                "actual_mean": float(a.mean()) if ok.any() else np.nan,
            }
            if categories and ok.any():
                row["category_agreement"] = float((_category(a) == _category(p)).mean())
            station_rows.append(row)
            period_frames.append(pd.DataFrame({
                "station_id": sid, "model": predictor.name, "period": actual.index,
                "nearest_reporting_km": reporting_km, "band": period_bands,
                "actual": actual.to_numpy(), "predicted": pred.to_numpy()}))

    return pd.DataFrame(station_rows), pd.concat(period_frames, ignore_index=True)


def band_table(per_period: pd.DataFrame, categories: bool) -> pd.DataFrame:
    """Error per band, where each station-period is banded by the nearest remaining station reporting then."""
    pp = per_period.dropna(subset=["predicted"]).assign(abs_err=lambda d: (d.predicted - d.actual).abs(),
                                                        err=lambda d: d.predicted - d.actual)
    if categories:
        pp["cat_ok"] = _category(pp["actual"].to_numpy()) == _category(pp["predicted"].to_numpy())
    rows = []
    # No blended all-band row: a near-band tie and a far-band win must never average into one number.
    for model in per_period["model"].unique():
        m = pp[pp.model == model]
        for label in [b[2] for b in BANDS]:
            pr = m[m.band == label]
            row = {"model": model, "band": label, "stations": pr["station_id"].nunique(),
                   "station_periods": len(pr), "mae": pr["abs_err"].mean(),
                   # MAE relative to the band's mean level: dense bands are also the most polluted,
                   # so raw MAE alone understates how much harder the far bands are.
                   "nmae_pct": 100 * pr["abs_err"].mean() / pr["actual"].mean() if len(pr) else np.nan,
                   "median_station_mae": pr.groupby("station_id")["abs_err"].mean().median(),
                   "bias": pr["err"].mean(), "actual_mean": pr["actual"].mean()}
            if categories:
                row["category_agreement_pct"] = 100 * pr["cat_ok"].mean() if len(pr) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


BASELINE = "idw_k8"


def improvement_vs_baseline(bands: pd.DataFrame, baseline: str = BASELINE) -> pd.DataFrame:
    """Per-band change against inverse-distance weighting.

    The target is the far bands (20-50 km, 100 km+), where satellite, fire and wind
    features carry information distance-weighting cannot see; a tie in the 0-20 km
    band is an acceptable, honest result. Both models are scored on the same
    station-periods, so the comparison is paired.
    """
    base = bands[bands["model"] == baseline].set_index("band")
    rows = []
    for model in bands["model"].unique():
        if model == baseline:
            continue
        m = bands[bands["model"] == model].set_index("band")
        for _, _, label in BANDS:
            if label not in m.index or label not in base.index or pd.isna(base.at[label, "mae"]):
                continue
            b_mae, m_mae = base.at[label, "mae"], m.at[label, "mae"]
            rows.append({"model": model, "band": label, "station_periods": int(m.at[label, "station_periods"]),
                         "baseline_mae": b_mae, "model_mae": m_mae,
                         "mae_improvement": b_mae - m_mae, "mae_improvement_pct": 100 * (b_mae - m_mae) / b_mae,
                         "baseline_nmae_pct": base.at[label, "nmae_pct"], "model_nmae_pct": m.at[label, "nmae_pct"]})
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    fmt = lambda v: "" if pd.isna(v) else (f"{v:,.1f}" if isinstance(v, float) else f"{v:,}" if isinstance(v, int) else str(v))  # noqa: E731
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parameter", default="pm25")
    parser.add_argument("--freq", choices=["D", "h"], default="D", help="D = IST daily means (CPCB AQI basis), h = hourly")
    parser.add_argument("--min-coverage", type=float, default=0.5, help="hourly coverage needed to be a target")
    parser.add_argument("--res", type=int, default=7, help="H3 resolution of the product grid")
    parser.add_argument("--models", default="nearest,idw", help=f"comma list of {sorted(PREDICTORS)}")
    parser.add_argument("--only", help="comma-separated station_ids to hold out (detail mode)")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pd.set_option("display.width", 200)

    corpus = load_corpus(args.parameter, args.freq, args.res)
    eligible = corpus.coverage.index[corpus.coverage >= args.min_coverage].tolist()
    log.info("%d of %d stations have >= %.0f%% hourly coverage", len(eligible), len(corpus.stations),
             100 * args.min_coverage)

    if args.only:
        targets = [s.strip() for s in args.only.split(",")]
        for sid in targets:
            if sid not in corpus.stations.index:
                sys.exit(f"{sid} has no {args.parameter} readings")
            if sid not in eligible:
                log.warning("%s coverage %.0f%% is below --min-coverage", sid, 100 * corpus.coverage[sid])
    else:
        targets = eligible

    predictors = [PREDICTORS[m.strip()]() for m in args.models.split(",")]
    categories = args.parameter == "pm25" and args.freq == "D"
    per_station, per_period = leave_station_out(corpus, predictors, targets, categories)
    bands = band_table(per_period, categories)
    versus = improvement_vs_baseline(bands)

    out = OUT_DIR / (f"{args.parameter}_{args.freq}" + ("_only" if args.only else ""))
    out.mkdir(parents=True, exist_ok=True)
    per_station.to_csv(out / "per_station.csv", index=False)
    per_period.to_parquet(out / "per_period.parquet", index=False)
    bands.to_csv(out / "bands.csv", index=False)
    unit = "ug/m3" if args.parameter in ("pm25", "pm10") else "native units"
    (out / "bands.md").write_text(
        f"Leave-station-out validation, {args.parameter} ({unit}), "
        f"{'daily IST means' if args.freq == 'D' else 'hourly'}; {len(targets)} held-out stations\n\n"
        + to_markdown(bands.round(2)) + "\n", encoding="utf-8")

    print("\n" + "=" * 96)
    print(f"LEAVE-STATION-OUT  {args.parameter} ({unit})  freq={args.freq}  targets={len(targets)}  "
          f"hex res {args.res}  (stations sharing the target hex are hidden too)")
    print("=" * 96)
    print(to_markdown(bands.round(2)))
    if len(versus):
        versus.to_csv(out / f"bands_vs_{BASELINE}.csv", index=False)
        print(f"\nper-band improvement over {BASELINE} (positive = better; never blended across bands)")
        print(to_markdown(versus.round(2)))

    if args.only:
        meta = corpus.stations
        for sid in targets:
            print(f"\n--- {sid}  {meta.at[sid, 'station_name']}  ({meta.at[sid, 'city']}, {meta.at[sid, 'state']})")
            print(per_station[per_station.station_id == sid].drop(columns=["station_id", "station_name", "city", "state"])
                  .round(2).to_string(index=False))
            hidden_hex = meta.index[meta["h3"] == meta.at[sid, "h3"]]
            rem = meta.drop(index=hidden_hex)
            d = haversine_km(meta.at[sid, "hex_lat"], meta.at[sid, "hex_lon"], rem["hex_lat"], rem["hex_lon"])
            near = rem.assign(km=d.round(1), coverage=corpus.coverage.reindex(rem.index).round(2)).nsmallest(5, "km")
            print("nearest remaining stations:")
            print(near[["station_name", "city", "km", "coverage"]].to_string())
            pp = per_period[per_period.station_id == sid].pivot(index="period", columns="model", values="predicted")
            pp.insert(0, "actual", per_period[per_period.station_id == sid].drop_duplicates("period")
                      .set_index("period")["actual"])
            pp.to_csv(out / f"{sid}_series.csv")
            print(f"series written to {(out / f'{sid}_series.csv').relative_to(ROOT)} ({len(pp)} periods)")
    print(f"\nwrote {out.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
