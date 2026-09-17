"""Train the three recorded variants once and save every day-by-day prediction.

Modelling is frozen after the pre-registered decision in docs/model-decision.md.
This script exists so intervals, the combined model and the deck figures can be
computed from saved predictions, never from another full rerun.

Variants (fixed):
    m1_level_l1           level target, median (L1) loss, all features
    m2_residual_l2        correction to neighbour IDW, squared loss, all features
    m3_residual_l2_nogeo  correction to neighbour IDW, squared loss, without lat/lon
Baselines scored in the same run: nearest_station, idw_k8.

Outputs (data/validation/experiments/):
    per_period.parquet    station_id, model, period, band, nearest_reporting_km, actual, predicted
    per_station.csv
    importance.csv        gain importance per variant, averaged over folds, % of total
    <variant>/model.pkl

Usage:
    python -m model.experiments
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from model.features import build_station_hex_table
from model.train import DEFAULT_PARAMS, FEATURES, GBMPredictor, assign_folds, fit_folds, training_rows
from model.validate import (InverseDistance, NearestStation, band_table, improvement_vs_baseline,
                            leave_station_out, load_corpus, to_markdown)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "validation" / "experiments"

VARIANTS = {
    "m1_level_l1": {"target": "level", "objective": "regression_l1", "drop": []},
    "m2_residual_l2": {"target": "residual", "objective": "regression", "drop": []},
    "m3_residual_l2_nogeo": {"target": "residual", "objective": "regression", "drop": ["lat", "lon"]},
}

log = logging.getLogger("model.experiments")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    corpus = load_corpus("pm25", "D", 7)
    eligible = corpus.coverage.index[corpus.coverage >= args.min_coverage].tolist()
    hex_features = build_station_hex_table()
    rows = training_rows(corpus, hex_features, eligible, args.k)
    hex_fold = assign_folds(sorted(corpus.stations.loc[eligible, "h3"].unique()), args.folds, args.seed)
    log.info("%d stations, %d training rows, %d folds", len(eligible), len(rows), args.folds)

    predictors = [NearestStation(), InverseDistance()]
    importance = []
    for name, spec in VARIANTS.items():
        features = [f for f in FEATURES if f not in spec["drop"]]
        params = DEFAULT_PARAMS | {"objective": spec["objective"], "random_state": args.seed}
        models = fit_folds(rows, hex_fold, args.folds, params, features, spec["target"])
        bundle = {"type": "single", "name": name, "models": models, "hex_fold": hex_fold, "features": features,
                  "params": params, "target_mode": spec["target"], "k": args.k, "parameter": "pm25", "freq": "D",
                  "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        (OUT_DIR / name).mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, OUT_DIR / name / "model.pkl")

        gain = np.mean([m.booster_.feature_importance(importance_type="gain") for m in models.values()], axis=0)
        importance.append(pd.DataFrame({"variant": name, "feature": features, "gain_pct": 100 * gain / gain.sum()}))
        predictors.append(GBMPredictor(models, hex_fold, hex_features, args.k, features, spec["target"], name=name))

    per_station, per_period = leave_station_out(corpus, predictors, eligible, categories=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_period.to_parquet(OUT_DIR / "per_period.parquet", index=False)
    per_station.to_csv(OUT_DIR / "per_station.csv", index=False)
    pd.concat(importance, ignore_index=True).to_csv(OUT_DIR / "importance.csv", index=False)

    bands = band_table(per_period, categories=True)
    print(to_markdown(bands.round(2)))
    print("\nper-band change vs idw_k8 (point estimates; intervals come from model/decide.py):")
    print(to_markdown(improvement_vs_baseline(bands).round(2)))
    print(f"\nwrote {OUT_DIR.relative_to(ROOT)}/ ({len(per_period):,} prediction rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
