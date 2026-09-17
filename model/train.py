"""LightGBM dark-zone model, scored by leave-station-out cross validation.

The claim is spatial ("we predict where no station exists"), so the score must be
too. Two rules make this honest:

  1. Station folds. The model that predicts a held-out station was trained without
     that station and without any station sharing its H3 hex.
  2. Station-derived features (what neighbours said, how far they are) are rebuilt
     from the REMAINING stations for every row, training rows included. The grid's
     precomputed dist_km is never used: it knows about the hidden station.

Where the value is: interpolation is already near-optimal with a monitor 15 km
away, so the target is the far bands (20-50 km, 100 km+), where satellite NO2,
aerosol index and upwind fire counts see things distance-weighting cannot. A tie
with IDW in the 0-20 km band is an acceptable, honest result - which is why the
report is per band and never blended.

Usage:
    python -m model.train                        # train + leave-station-out evaluation
    python -m model.train --folds 5 --k 8
    python -m model.train --skip-eval            # just fit and save model/model.pkl
Outputs: model/model.pkl, data/validation/gbm/{bands,bands_vs_idw,per_station}.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from model.features import STATION_FEATURES, build_station_hex_table, neighbour_features
from model.validate import (BASELINE, InverseDistance, NearestStation, Target, TrainingData,
                            band_table, improvement_vs_baseline, leave_station_out, load_corpus, to_markdown)

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = ROOT / "model" / "model.pkl"
OUT_DIR = ROOT / "data" / "validation" / "gbm"

SATELLITE_FEATURES = ["no2_trop", "aai", "co", "so2"]
CONTEXT_FEATURES = ["wind_speed", "wind_dir_sin", "wind_dir_cos", "fire_count", "frp_sum",
                    "fires_150km", "frp_150km", "upwind_fires_150km", "upwind_frp_150km",
                    "doy_sin", "doy_cos", "month", "weekday", "lat", "lon"]
FEATURES = SATELLITE_FEATURES + CONTEXT_FEATURES + STATION_FEATURES

DEFAULT_PARAMS = {
    "objective": "regression_l1",   # scored on MAE, so train on MAE
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "n_estimators": 600,
    "verbose": -1,
}

log = logging.getLogger("model.train")


def assign_folds(hexes: list[str], n_folds: int, seed: int) -> dict[str, int]:
    """Folds are assigned per hex, so stations sharing a hex never straddle a split."""
    rng = np.random.default_rng(seed)
    shuffled = list(hexes)
    rng.shuffle(shuffled)
    return {h: i % n_folds for i, h in enumerate(shuffled)}


def training_rows(corpus, hex_features: pd.DataFrame, targets: list[str], k: int) -> pd.DataFrame:
    """One row per (station, day) with station-derived features built without that station."""
    values, meta = corpus.values, corpus.stations
    hex_features = hex_features.set_index(["h3", "date"])
    frames = []
    for sid in targets:
        hex_id = meta.at[sid, "h3"]
        hidden = meta.index[meta["h3"] == hex_id]
        remaining = meta.drop(index=hidden)
        actual = values[sid].dropna()
        if actual.empty:
            continue
        neigh = neighbour_features(values.drop(columns=hidden), remaining,
                                   meta.at[sid, "hex_lat"], meta.at[sid, "hex_lon"], actual.index, k=k)
        rows = neigh.copy()
        rows["target"] = actual
        rows["station_id"] = sid
        rows["h3"] = hex_id
        frames.append(rows.reset_index(names="date"))
    df = pd.concat(frames, ignore_index=True)
    return df.join(hex_features, on=["h3", "date"])


class GBMPredictor:
    """Fold-aware LightGBM. predict() uses the model whose training set excluded this hex.

    target_mode "level" predicts PM2.5 directly; "residual" predicts the correction to the
    neighbour IDW estimate, so the model starts from the baseline and only moves where the
    satellite / fire / wind features justify it.
    """

    def __init__(self, models: dict[int, lgb.LGBMRegressor], hex_fold: dict[str, int],
                 hex_features: pd.DataFrame, k: int, features: list[str] = FEATURES,
                 target_mode: str = "level", name: str = "gbm"):
        self.models = models
        self.hex_fold = hex_fold
        self.hex_set = set(hex_features["h3"])
        self.hex_features = hex_features.set_index(["h3", "date"]).sort_index()
        self.k = k
        self.features = features
        self.target_mode = target_mode
        self.name = name

    def predict(self, train: TrainingData, target: Target) -> pd.Series:
        neigh = neighbour_features(train.values, train.stations, target.lat, target.lon, target.periods, k=self.k)
        # Hexes with no satellite row (11 stations sit outside the grid) still predict,
        # on neighbour features alone; LightGBM handles the NaNs.
        rows = neigh.join(self.hex_features.xs(target.h3, level="h3"), how="left") \
            if target.h3 in self.hex_set else neigh
        rows = rows.reindex(columns=FEATURES)
        model = self.models[self.hex_fold.get(target.h3, 0)]
        out = model.predict(rows[self.features])
        if self.target_mode == "residual":
            out = rows["neighbour_idw"].to_numpy(dtype=float) + out  # NaN where no neighbour reported
        return pd.Series(out, index=target.periods)


def target_values(rows: pd.DataFrame, target_mode: str) -> pd.Series:
    return rows["target"] - rows["neighbour_idw"] if target_mode == "residual" else rows["target"]


def fit_folds(rows: pd.DataFrame, hex_fold: dict[str, int], n_folds: int, params: dict,
              features: list[str] = FEATURES, target_mode: str = "level") -> dict[int, lgb.LGBMRegressor]:
    rows = rows.assign(fold=rows["h3"].map(hex_fold))
    models = {}
    for fold in range(n_folds):
        train = rows[rows["fold"] != fold]
        y = target_values(train, target_mode)
        train, y = train[y.notna()], y[y.notna()]
        started = time.time()
        model = lgb.LGBMRegressor(**params)
        model.fit(train[features], y)
        models[fold] = model
        log.info("fold %d: trained on %d rows from %d hexes (%.1fs)", fold, len(train),
                 train["h3"].nunique(), time.time() - started)
    return models


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parameter", default="pm25")
    parser.add_argument("--freq", choices=["D", "h"], default="D")
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--res", type=int, default=7)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--k", type=int, default=8, help="neighbour stations used for the station-derived features")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_PARAMS["n_estimators"])
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_PARAMS["learning_rate"])
    parser.add_argument("--num-leaves", type=int, default=DEFAULT_PARAMS["num_leaves"])
    parser.add_argument("--target", choices=["level", "residual"], default="level",
                        help="predict PM2.5 directly, or the correction to the neighbour IDW estimate")
    parser.add_argument("--objective", default=DEFAULT_PARAMS["objective"],
                        help="LightGBM objective, e.g. regression_l1 (median) or regression (mean)")
    parser.add_argument("--drop", default="", help="comma-separated features to exclude (ablation)")
    parser.add_argument("--tag", help="name for this run's output folder (default: built from the options)")
    parser.add_argument("--promote", action="store_true", help="also write model/model.pkl (the model the map uses)")
    parser.add_argument("--skip-eval", action="store_true", help="fit and save without the leave-station-out run")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    params = DEFAULT_PARAMS | {"n_estimators": args.n_estimators, "learning_rate": args.learning_rate,
                               "num_leaves": args.num_leaves, "random_state": args.seed,
                               "objective": args.objective}
    dropped = [f.strip() for f in args.drop.split(",") if f.strip()]
    unknown = [f for f in dropped if f not in FEATURES]
    if unknown:
        parser.error(f"unknown features to drop: {unknown}")
    features = [f for f in FEATURES if f not in dropped]
    tag = args.tag or "_".join([args.target, args.objective.replace("regression", "reg")]
                               + ([f"no-{'-'.join(dropped)}"] if dropped else []))
    out_dir = OUT_DIR.parent / f"gbm_{tag}"

    corpus = load_corpus(args.parameter, args.freq, args.res)
    eligible = corpus.coverage.index[corpus.coverage >= args.min_coverage].tolist()
    log.info("%d of %d stations have >= %.0f%% usable coverage", len(eligible), len(corpus.stations),
             100 * args.min_coverage)

    hex_features = build_station_hex_table()
    rows = training_rows(corpus, hex_features, eligible, args.k)
    missing = rows[FEATURES].isna().mean().sort_values(ascending=False)
    log.info("%d training rows; most-missing features: %s", len(rows),
             {k: f"{100*v:.0f}%" for k, v in missing.head(4).items()})

    hex_fold = assign_folds(sorted(corpus.stations.loc[eligible, "h3"].unique()), args.folds, args.seed)
    models = fit_folds(rows, hex_fold, args.folds, params, features, args.target)

    bundle = {"models": models, "hex_fold": hex_fold, "features": features, "params": params,
              "target_mode": args.target, "k": args.k, "parameter": args.parameter, "freq": args.freq,
              "tag": tag, "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_dir / "model.pkl")
    if args.promote:
        joblib.dump(bundle, MODEL_FILE)
        log.info("promoted to %s", MODEL_FILE.relative_to(ROOT))

    importance = pd.Series(models[0].feature_importances_, index=features).sort_values(ascending=False)
    print(f"\n[{tag}] top features (fold 0, split count):")
    print(importance.head(12).to_string())
    if args.skip_eval:
        return 0

    predictor = GBMPredictor(models, hex_fold, hex_features, args.k, features, args.target, name=f"gbm_{tag}")
    per_station, per_period = leave_station_out(corpus, [NearestStation(), InverseDistance(), predictor],
                                                eligible, categories=args.parameter == "pm25" and args.freq == "D")
    bands = band_table(per_period, categories=args.parameter == "pm25" and args.freq == "D")
    versus = improvement_vs_baseline(bands)

    per_station.to_csv(out_dir / "per_station.csv", index=False)
    bands.to_csv(out_dir / "bands.csv", index=False)
    versus.to_csv(out_dir / "bands_vs_idw.csv", index=False)
    (out_dir / "bands.md").write_text(to_markdown(bands.round(2)) + "\n", encoding="utf-8")

    print("\n" + "=" * 96)
    print(f"LEAVE-STATION-OUT [{tag}]  {args.parameter}  freq={args.freq}  {len(eligible)} held-out stations  "
          f"{args.folds} station folds")
    print("=" * 96)
    print(to_markdown(bands[bands["model"] == predictor.name].round(2)))
    print(f"\nper-band change vs {BASELINE} (positive = better):")
    print(to_markdown(versus[versus["model"] == predictor.name].round(2)))

    sidco = per_station[per_station["station_id"] == "openaq-8914"]
    if len(sidco):
        print("\nSIDCO Kurichi, Coimbatore (hidden; nearest reliable station 72 km away):")
        print(sidco[["model", "band", "periods", "mae", "bias", "rmse", "r", "category_agreement"]]
              .round(2).to_string(index=False))
    (out_dir / "summary.json").write_text(json.dumps(
        {"trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "params": params,
         "stations": len(eligible), "rows": len(rows), "folds": args.folds,
         "bands": bands.to_dict(orient="records")}, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
