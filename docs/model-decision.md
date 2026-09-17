# Model decision: pre-registered combined model

**Written and committed 2026-09-17, before the combined model is evaluated.** This is a
single pre-registered evaluation. The result will be accepted either way, and modelling
is frozen after it: no further variants, no threshold changes, no re-runs to chase a
better number.

## Why a combined model

Three variants were each evaluated once on the same 518 stations (2026-09-16), MAE change
against inverse-distance weighting (IDW), by distance to the nearest reporting station:

| Variant | 0–20 km | 20–50 km | 50–100 km | 100 km+ |
|---|---|---|---|---|
| 1. Level target, median (L1) loss, all features | −2.4% | +11.9% | +6.6% | **+6.9%** |
| 3. Correction to IDW, squared loss, **no lat/lon** | −3.8% | **+12.1%** | +7.3% | −7.5% |

The mechanism behind the split: when a reporting station is within reach, the neighbour
IDW estimate is a good anchor and the satellite features correct it (variant 3). Beyond
~100 km that anchor is unreliable, so correcting it does not help, and a level model with
a regional prior does better (variant 1).

**Disclosure:** this combination was proposed *after* seeing those single-run results on
the same stations. The threshold follows from the mechanism above; it was not tuned, and
no other threshold has been or will be evaluated.

## Specification (fixed)

For every hex-day, let **d** = distance from the hex centroid to the nearest *remaining*
station that reported PM2.5 that day (the same `nearest_reporting_km` the validation
harness uses).

| Condition | Model used |
|---|---|
| d ≤ **100 km** | Variant 3: residual on IDW, squared loss, all features except `lat`, `lon` |
| d > 100 km, or no station reported | Variant 1: level target, median (L1) loss, all features |

- Threshold: **100 km**, fixed in advance.
- Data: daily IST-mean PM2.5, stations with ≥ 50% usable hourly coverage (flagged rows
  excluded), leave-station-out with 5 folds assigned per H3 hex, seed 0, k = 8 neighbours.
- Predictions for variants 1 and 3 come from one saved run (`python -m model.experiments`),
  retrained with the same seed so day-level predictions are on disk. LightGBM threading can
  shift those numbers slightly from the 2026-09-16 runs; the saved run is the one used.

## Uncertainty

- **Per band:** station-level bootstrap, 2,000 draws, seed 0, 95% percentile intervals.
  Stations are resampled, not station-days, because days within a station are not
  independent. Every model is scored on the same draws, so comparisons are paired.
- **SIDCO Kurichi** (one station): moving-block bootstrap over its own days, 7-day blocks,
  2,000 draws. Reported alongside the decision, **not part of the rule**.

## Decision rule (declared before running)

The combined model **wins** if and only if the **lower bound of the 95% interval on its % MAE
improvement over IDW is above zero in both target bands: 20–50 km and 100 km+.**

- **Wins:** promote the combined model to `model/model.pkl`. It becomes the model behind the map.
- **Does not win:** promote variant 3 (residual, no lat/lon) instead, and say so.

The 0–20 km band is not part of the rule: a tie with IDW there is an acceptable, honest result.

## Result

*(Appended by `python -m model.decide` after the single evaluation. Nothing above this line
changes after it runs.)*
