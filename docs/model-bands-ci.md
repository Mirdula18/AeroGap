# Leave-station-out results with 95% confidence intervals

Daily PM2.5, 518 hidden stations, bands by distance to the nearest station that reported that day. Station-level bootstrap, 2,000 draws (stations resampled, not station-days); every model is scored on the same draws and the same station-days. Positive improvement = lower MAE than the baseline.

| Model | Band | Stations | Station-days | MAE µg/m³ [95% CI] | Error % of mean | vs IDW [95% CI] | vs nearest station [95% CI] |
|---|---|---|---|---|---|---|---|
| Nearest station (baseline) | 0-20 km | 357 | 95,684 | 19.1 [17.9, 20.2] | 35.1% | -18.0% [-21.0, -15.0] |  |
| IDW, 8 nearest (baseline) | 0-20 km | 357 | 95,684 | 16.2 [15.2, 17.1] | 29.7% |  | +15.2% [+13.1, +17.4] |
| 1. Level, median loss | 0-20 km | 357 | 95,684 | 16.5 [15.4, 17.7] | 30.4% | -2.3% [-5.1, +0.5] | +13.2% [+10.1, +16.5] |
| 2. Residual on IDW | 0-20 km | 357 | 95,684 | 17.2 [16.1, 18.3] | 31.6% | -6.3% [-9.1, -3.4] | +9.9% [+6.6, +13.1] |
| 3. Residual on IDW, no lat/lon | 0-20 km | 357 | 95,684 | 16.8 [15.8, 17.8] | 30.9% | -3.8% [-5.8, -1.7] | +12.0% [+9.2, +14.7] |
| Combined (pre-registered) | 0-20 km | 357 | 95,684 | 16.8 [15.8, 17.8] | 30.9% | -3.8% [-5.8, -1.7] | +12.0% [+9.2, +14.7] |
| Nearest station (baseline) | 20-50 km | 142 | 18,474 | 25.3 [21.9, 29.0] | 51.5% | -19.4% [-28.4, -11.0] |  |
| IDW, 8 nearest (baseline) | 20-50 km | 142 | 18,474 | 21.2 [18.4, 24.1] | 43.1% |  | +16.2% [+9.9, +22.1] |
| 1. Level, median loss | 20-50 km | 142 | 18,474 | 18.6 [16.3, 21.3] | 38.0% | +11.9% [+5.4, +18.1] | +26.2% [+17.8, +33.9] |
| 2. Residual on IDW | 20-50 km | 142 | 18,474 | 19.1 [16.9, 21.7] | 39.0% | +9.6% [+3.7, +15.1] | +24.3% [+16.1, +32.1] |
| 3. Residual on IDW, no lat/lon | 20-50 km | 142 | 18,474 | 18.6 [16.4, 21.0] | 37.9% | +12.1% [+6.8, +17.0] | +26.3% [+18.7, +33.5] |
| Combined (pre-registered) | 20-50 km | 142 | 18,474 | 18.6 [16.4, 21.0] | 37.9% | +12.1% [+6.8, +17.0] | +26.3% [+18.7, +33.5] |
| Nearest station (baseline) | 50-100 km | 200 | 23,396 | 20.0 [17.5, 23.1] | 50.3% | -23.5% [-31.6, -16.4] |  |
| IDW, 8 nearest (baseline) | 50-100 km | 200 | 23,396 | 16.2 [14.3, 18.4] | 40.8% |  | +19.0% [+14.1, +24.0] |
| 1. Level, median loss | 50-100 km | 200 | 23,396 | 15.1 [13.3, 17.2] | 38.1% | +6.6% [+0.1, +12.6] | +24.4% [+17.5, +31.4] |
| 2. Residual on IDW | 50-100 km | 200 | 23,396 | 15.5 [13.8, 17.4] | 39.0% | +4.4% [-1.9, +10.3] | +22.6% [+15.5, +29.9] |
| 3. Residual on IDW, no lat/lon | 50-100 km | 200 | 23,396 | 15.0 [13.4, 17.0] | 37.8% | +7.3% [+1.5, +12.6] | +24.9% [+18.4, +31.9] |
| Combined (pre-registered) | 50-100 km | 200 | 23,396 | 15.0 [13.4, 17.0] | 37.8% | +7.3% [+1.5, +12.6] | +24.9% [+18.4, +31.9] |
| Nearest station (baseline) | 100 km+ | 130 | 5,033 | 27.2 [20.8, 36.2] | 65.1% | -18.2% [-28.2, -8.5] |  |
| IDW, 8 nearest (baseline) | 100 km+ | 130 | 5,033 | 23.0 [17.8, 29.8] | 55.1% |  | +15.4% [+7.8, +22.0] |
| 1. Level, median loss | 100 km+ | 130 | 5,033 | 21.4 [16.5, 28.3] | 51.3% | +6.9% [-7.7, +21.4] | +21.2% [+9.6, +33.5] |
| 2. Residual on IDW | 100 km+ | 130 | 5,033 | 23.1 [18.6, 29.2] | 55.3% | -0.3% [-13.1, +10.8] | +15.1% [+3.8, +25.0] |
| 3. Residual on IDW, no lat/lon | 100 km+ | 130 | 5,033 | 24.7 [19.6, 31.9] | 59.2% | -7.5% [-19.6, +1.3] | +9.1% [-2.5, +17.9] |
| Combined (pre-registered) | 100 km+ | 130 | 5,033 | 21.4 [16.5, 28.3] | 51.3% | +6.9% [-7.7, +21.4] | +21.2% [+9.6, +33.5] |
