"""The single pre-registered evaluation of the combined model. Run once.

Implements docs/model-decision.md exactly: builds the combined predictions from the
saved experiment run (no retraining), computes station-bootstrap intervals for every
model, applies the rule declared before running, promotes the winner to
model/model.pkl and appends the result below the specification. It refuses to run a
second time, so the evaluation cannot be quietly repeated.

Usage:
    python -m model.experiments     # once, first: saves day-level predictions
    python -m model.decide          # once: the pre-registered evaluation
Outputs: model/model.pkl, docs/model-bands-ci.{csv,md}, docs/sidco-kurichi-ci.csv,
         data/validation/experiments/per_period_combined.parquet, Result section of the decision doc
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from model.bootstrap import BAND_LABELS, block_bootstrap_station, paired_frame, station_bootstrap

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "data" / "validation" / "experiments"
DOCS = ROOT / "docs"
DECISION_DOC = DOCS / "model-decision.md"
MODEL_FILE = ROOT / "model" / "model.pkl"

# ---- fixed by the pre-registration; do not edit ----
THRESHOLD_KM = 100.0
NEAR, FAR = "m3_residual_l2_nogeo", "m1_level_l1"
FALLBACK = "m3_residual_l2_nogeo"
TARGET_BANDS = ["20-50 km", "100 km+"]
N_BOOT, SEED = 2000, 0
# ----------------------------------------------------

MODELS = ["nearest_station", "idw_k8", "m1_level_l1", "m2_residual_l2", "m3_residual_l2_nogeo", "combined"]
DISPLAY = {"nearest_station": "Nearest station (baseline)", "idw_k8": "IDW, 8 nearest (baseline)",
           "m1_level_l1": "1. Level, median loss", "m2_residual_l2": "2. Residual on IDW",
           "m3_residual_l2_nogeo": "3. Residual on IDW, no lat/lon", "combined": "Combined (pre-registered)"}
SIDCO = "openaq-8914"
RESULT_MARKER = "### Evaluated"


def combine(per_period: pd.DataFrame) -> pd.DataFrame:
    """Combined predictions: NEAR where the nearest reporting station is within the threshold, else FAR."""
    near = per_period[per_period["model"] == NEAR].set_index(["station_id", "period"])
    far = per_period[per_period["model"] == FAR].set_index(["station_id", "period"]).reindex(near.index)
    use_near = (near["nearest_reporting_km"] <= THRESHOLD_KM).to_numpy()  # NaN distance -> FAR
    out = near.copy()
    out["model"] = "combined"
    out["predicted"] = np.where(use_near, near["predicted"], far["predicted"])
    out["component"] = np.where(use_near, NEAR, FAR)
    return out.reset_index()


def interval(value: float, lo: float, hi: float, pct: bool = False) -> str:
    if pd.isna(value):
        return ""
    if pct:
        return f"{value:+.1f}% [{lo:+.1f}, {hi:+.1f}]"
    return f"{value:.1f} [{lo:.1f}, {hi:.1f}]"


def bands_markdown(ci: pd.DataFrame) -> str:
    lines = ["| Model | Band | Stations | Station-days | MAE µg/m³ [95% CI] | Error % of mean | "
             "vs IDW [95% CI] | vs nearest station [95% CI] |", "|---|---|---|---|---|---|---|---|"]
    for band in BAND_LABELS:
        for model in MODELS:
            r = ci[(ci["model"] == model) & (ci["band"] == band)]
            if r.empty:
                continue
            r = r.iloc[0]
            lines.append(f"| {DISPLAY[model]} | {band} | {r.stations} | {r.station_days:,} | "
                         f"{interval(r.mae, r.mae_lo, r.mae_hi)} | {r.nmae_pct:.1f}% | "
                         f"{interval(r.get('vs_idw_pct'), r.get('vs_idw_lo'), r.get('vs_idw_hi'), pct=True)} | "
                         f"{interval(r.get('vs_nearest_pct'), r.get('vs_nearest_lo'), r.get('vs_nearest_hi'), pct=True)} |")
    return "\n".join(lines)


def spec_commit() -> str:
    try:
        out = subprocess.run(["git", "log", "--format=%h %ad", "--date=iso", "--diff-filter=A", "--",
                              str(DECISION_DOC.relative_to(ROOT))], cwd=ROOT, capture_output=True, text=True)
        return out.stdout.strip().splitlines()[-1] if out.stdout.strip() else "uncommitted"
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-run despite an existing result (don't)")
    parser.add_argument("--note", default="", help="disclosure recorded in the Result section")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    doc = DECISION_DOC.read_text(encoding="utf-8")
    if RESULT_MARKER in doc and not args.force:
        sys.exit("The pre-registered evaluation has already run; its result is in docs/model-decision.md.")

    per_period = pd.read_parquet(EXP / "per_period.parquet")
    combined = combine(per_period)
    combined.to_parquet(EXP / "per_period_combined.parquet", index=False)
    all_periods = pd.concat([per_period, combined.drop(columns=["component"])], ignore_index=True)

    wide = paired_frame(all_periods, MODELS)
    ci = station_bootstrap(wide, MODELS, n_boot=N_BOOT, seed=SEED)
    sidco = block_bootstrap_station(wide[wide["station_id"] == SIDCO], [m for m in MODELS if m != "nearest_station"],
                                    baseline="idw_k8", n_boot=N_BOOT, seed=SEED)

    target = ci[(ci["model"] == "combined") & ci["band"].isin(TARGET_BANDS)].set_index("band")
    checks = {b: bool(b in target.index and target.at[b, "vs_idw_lo"] > 0) for b in TARGET_BANDS}
    wins = all(checks.values())
    promoted = "combined" if wins else FALLBACK

    uncertainty = (ci[ci["model"] == promoted].set_index("band")
                   [["mae", "mae_lo", "mae_hi", "nmae_pct", "stations", "station_days"]].to_dict(orient="index"))
    if wins:
        bundle = {"type": "combined", "name": "combined", "threshold_km": THRESHOLD_KM,
                  "near": joblib.load(EXP / NEAR / "model.pkl"), "far": joblib.load(EXP / FAR / "model.pkl")}
    else:
        bundle = joblib.load(EXP / FALLBACK / "model.pkl")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    bundle |= {"uncertainty_by_band": uncertainty, "promoted_utc": now,
               "decision": {"rule": "combined wins iff lower 95% bound of % MAE improvement over IDW > 0 in "
                                    + " and ".join(TARGET_BANDS), "checks": checks, "wins": wins, "promoted": promoted}}
    joblib.dump(bundle, MODEL_FILE)

    DOCS.mkdir(exist_ok=True)
    ci.round(3).to_csv(DOCS / "model-bands-ci.csv", index=False)
    sidco.round(4).to_csv(DOCS / "sidco-kurichi-ci.csv", index=False)
    (DOCS / "model-bands-ci.md").write_text(
        "# Leave-station-out results with 95% confidence intervals\n\n"
        "Daily PM2.5, 518 hidden stations, bands by distance to the nearest station that reported that day. "
        f"Station-level bootstrap, {N_BOOT:,} draws (stations resampled, not station-days); every model is "
        "scored on the same draws and the same station-days. Positive improvement = lower MAE than the baseline.\n\n"
        + bands_markdown(ci) + "\n", encoding="utf-8")

    rows = []
    for b in BAND_LABELS:
        r = ci[(ci["model"] == "combined") & (ci["band"] == b)].iloc[0]
        tag = " (target)" if b in TARGET_BANDS else ""
        rows.append(f"| {b}{tag} | {interval(r.mae, r.mae_lo, r.mae_hi)} | "
                    f"{interval(r.vs_idw_pct, r.vs_idw_lo, r.vs_idw_hi, pct=True)} |")
    sid_rows = [f"| {DISPLAY[r.model]} | {interval(r.mae, r.mae_lo, r.mae_hi)} | "
                f"{r.r:.2f} [{r.r_lo:.2f}, {r.r_hi:.2f}] | {100*r.category_agreement:.0f}% "
                f"[{100*r.category_lo:.0f}, {100*r.category_hi:.0f}] | "
                f"{interval(getattr(r, 'vs_idw_pct', np.nan), getattr(r, 'vs_idw_lo', np.nan), getattr(r, 'vs_idw_hi', np.nan), pct=True)} |"
                for r in sidco.itertuples()]
    verdict = ("**The combined model wins.** Both target-band lower bounds are above zero. "
               "It is promoted to `model/model.pkl` and drives the map."
               if wins else
               "**The combined model does not win.** " + "; ".join(
                   f"{b}: lower bound {target.at[b, 'vs_idw_lo']:+.1f}%" for b in TARGET_BANDS if b in target.index)
               + ". As declared, variant 3 (residual on IDW, no lat/lon) is promoted to `model/model.pkl` instead.")
    result = f"""
{RESULT_MARKER} {now}

Specification committed first: `{spec_commit()}`. Single run of `python -m model.decide`.
{f"{chr(10)}Note: {args.note}{chr(10)}" if args.note else ""}

Combined model vs IDW (station bootstrap, {N_BOOT:,} draws, 95% CI):

| Band | MAE µg/m³ [95% CI] | Improvement over IDW [95% CI] |
|---|---|---|
{chr(10).join(rows)}

Rule checks (lower bound of improvement over IDW > 0):
{chr(10).join(f"- {b}: {'PASS' if ok else 'FAIL'} (lower bound {target.at[b, 'vs_idw_lo']:+.1f}%)" for b, ok in checks.items())}

{verdict}

SIDCO Kurichi, hidden (7-day block bootstrap over its own {int(sidco['days'].iloc[0])} days; not part of the rule):

| Model | MAE [95% CI] | r [95% CI] | Right AQI category [95% CI] | vs IDW [95% CI] |
|---|---|---|---|---|
{chr(10).join(sid_rows)}

Full tables: `docs/model-bands-ci.md`, `docs/sidco-kurichi-ci.csv`. Modelling is now frozen.
"""
    DECISION_DOC.write_text(doc.rstrip() + "\n" + result, encoding="utf-8")

    print(bands_markdown(ci))
    print("\n" + result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
