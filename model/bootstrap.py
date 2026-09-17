"""Bootstrap confidence intervals for leave-station-out results.

Unit of independence: the STATION. Days within a station are strongly
autocorrelated, so resampling station-days would produce intervals far too narrow.
Each draw resamples stations with replacement and recomputes every model's pooled
MAE on that same draw, so model-vs-baseline comparisons are paired.

A single station (e.g. SIDCO Kurichi) cannot be station-bootstrapped. For that case
block_bootstrap_station() resamples 7-day blocks of its own days, which respects
short-range autocorrelation. Its interval answers a different question ("how stable
is this one station's result across its own days"), and is labelled as such.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from model.validate import BANDS, PM25_CATEGORY_EDGES

BAND_LABELS = [b[2] for b in BANDS]
BASELINE_KEYS = {"idw_k8": "idw", "nearest_station": "nearest"}


def paired_frame(per_period: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Wide frame: one row per station-day that EVERY listed model predicted, so comparisons are fair."""
    pp = per_period[per_period["model"].isin(models) & per_period["band"].isin(BAND_LABELS)]
    wide = pp.pivot_table(index=["station_id", "period", "band"], columns="model", values="predicted",
                          aggfunc="first").reindex(columns=models).dropna().reset_index()
    actual = pp.drop_duplicates(["station_id", "period"]).set_index(["station_id", "period"])["actual"]
    wide["actual"] = actual.reindex(pd.MultiIndex.from_frame(wide[["station_id", "period"]])).to_numpy()
    return wide


def _pct(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, q)) if finite.size else math.nan


def station_bootstrap(wide: pd.DataFrame, models: list[str], n_boot: int = 2000, seed: int = 0,
                      ci: float = 95.0) -> pd.DataFrame:
    stations = pd.Index(sorted(wide["station_id"].unique()))
    n_stations = len(stations)
    sid = stations.get_indexer(wide["station_id"])
    counts = np.random.default_rng(seed).multinomial(n_stations, np.full(n_stations, 1 / n_stations),
                                                     size=n_boot).astype(np.float64)  # (draws, stations)
    lo_q, hi_q = (100 - ci) / 2, 100 - (100 - ci) / 2
    actual = wide["actual"].to_numpy(dtype=float)

    rows = []
    for band in BAND_LABELS:
        mask = (wide["band"] == band).to_numpy()
        n = np.bincount(sid[mask], minlength=n_stations).astype(float)
        if n.sum() == 0:
            continue
        act = np.bincount(sid[mask], weights=actual[mask], minlength=n_stations)
        sae = {m: np.bincount(sid[mask], weights=np.abs(wide[m].to_numpy(dtype=float)[mask] - actual[mask]),
                              minlength=n_stations) for m in models}
        with np.errstate(invalid="ignore", divide="ignore"):
            boot_n = counts @ n
            boot = {m: (counts @ sae[m]) / boot_n for m in models}
        point = {m: sae[m].sum() / n.sum() for m in models}

        for m in models:
            row = {"model": m, "band": band, "stations": int((n > 0).sum()), "station_days": int(n.sum()),
                   "mae": point[m], "mae_lo": _pct(boot[m], lo_q), "mae_hi": _pct(boot[m], hi_q),
                   "nmae_pct": 100 * point[m] / (act.sum() / n.sum())}
            for base, key in BASELINE_KEYS.items():
                if base in models and base != m:
                    with np.errstate(invalid="ignore", divide="ignore"):
                        improvement = 100 * (boot[base] - boot[m]) / boot[base]
                    row[f"vs_{key}_pct"] = 100 * (point[base] - point[m]) / point[base]
                    row[f"vs_{key}_lo"] = _pct(improvement, lo_q)
                    row[f"vs_{key}_hi"] = _pct(improvement, hi_q)
            rows.append(row)
    return pd.DataFrame(rows)


def paired_improvement(wide: pd.DataFrame, model: str, reference: str, n_boot: int = 2000, seed: int = 0,
                       ci: float = 95.0) -> pd.DataFrame:
    """% MAE improvement of `model` over `reference` per band, with a paired station-bootstrap interval.

    Use this, not two overlapping per-model intervals, to decide whether one model beats another.
    """
    res = station_bootstrap(wide.rename(columns={reference: "idw_k8"}) if reference != "idw_k8" else wide,
                            ["idw_k8", model], n_boot=n_boot, seed=seed, ci=ci)
    out = res[res["model"] == model][["band", "stations", "station_days", "vs_idw_pct", "vs_idw_lo", "vs_idw_hi"]]
    return out.rename(columns={"vs_idw_pct": "improvement_pct", "vs_idw_lo": "lo", "vs_idw_hi": "hi"}).assign(
        model=model, reference=reference)


def block_bootstrap_station(wide_station: pd.DataFrame, models: list[str], baseline: str = "idw_k8",
                            n_boot: int = 2000, block_days: int = 7, seed: int = 0, ci: float = 95.0) -> pd.DataFrame:
    """Moving-block bootstrap over ONE station's days: MAE, correlation, category agreement, gain vs baseline."""
    df = wide_station.sort_values("period")
    actual = df["actual"].to_numpy(dtype=float)
    n = len(actual)
    rng = np.random.default_rng(seed)
    blocks = math.ceil(n / block_days)
    starts = rng.integers(0, max(n - block_days, 0) + 1, size=(n_boot, blocks))
    idx = (starts[:, :, None] + np.arange(block_days)).reshape(n_boot, -1)[:, :n]
    lo_q, hi_q = (100 - ci) / 2, 100 - (100 - ci) / 2
    cat = lambda v: np.searchsorted(PM25_CATEGORY_EDGES, v, side="right")  # noqa: E731

    def corr(a: np.ndarray, p: np.ndarray) -> np.ndarray:
        a = a - a.mean(axis=-1, keepdims=True)
        p = p - p.mean(axis=-1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (a * p).sum(axis=-1) / np.sqrt((a ** 2).sum(axis=-1) * (p ** 2).sum(axis=-1))

    a_boot = actual[idx]
    base_pred = df[baseline].to_numpy(dtype=float)
    base_mae_boot = np.abs(base_pred[idx] - a_boot).mean(axis=1)
    base_mae = np.abs(base_pred - actual).mean()
    rows = []
    for m in models:
        pred = df[m].to_numpy(dtype=float)
        p_boot = pred[idx]
        mae_boot = np.abs(p_boot - a_boot).mean(axis=1)
        r_boot = corr(a_boot, p_boot)
        cat_boot = (cat(a_boot) == cat(p_boot)).mean(axis=1)
        mae = np.abs(pred - actual).mean()
        row = {"model": m, "days": n, "mae": mae, "mae_lo": _pct(mae_boot, lo_q), "mae_hi": _pct(mae_boot, hi_q),
               "bias": float((pred - actual).mean()),
               "r": float(corr(actual, pred)), "r_lo": _pct(r_boot, lo_q), "r_hi": _pct(r_boot, hi_q),
               "category_agreement": float((cat(actual) == cat(pred)).mean()),
               "category_lo": _pct(cat_boot, lo_q), "category_hi": _pct(cat_boot, hi_q)}
        if m != baseline:
            gain = 100 * (base_mae_boot - mae_boot) / base_mae_boot
            row["vs_idw_pct"] = 100 * (base_mae - mae) / base_mae
            row["vs_idw_lo"], row["vs_idw_hi"] = _pct(gain, lo_q), _pct(gain, hi_q)
        rows.append(row)
    return pd.DataFrame(rows)
