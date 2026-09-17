"""Deck artefacts for the frozen modelling result. Run after model/decide.py.

Writes to docs/:
    latlon-ablation.md                     two-row before/after: same model with and without lat/lon
    feature-importance-ablation.csv
    figures/feature-importance-ablation.png
    sidco-kurichi-timeseries.csv
    figures/sidco-kurichi-timeseries.png   actual vs promoted model vs IDW, every day SIDCO reported

Colours follow the validated default palette (categorical slots 1-3, light surface).
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from model.bootstrap import paired_frame, paired_improvement  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "data" / "validation" / "experiments"
DOCS = ROOT / "docs"
FIG = DOCS / "figures"
SIDCO = "openaq-8914"

INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
       "axis": "#c3c2b7", "surface": "#fcfcfb"}
GROUP_COLOUR = {"Satellite columns": "#2a78d6", "Location (lat/lon)": "#eb6834",
                "Wind, fire, season": "#1baf7a", "Neighbour stations": "#898781"}
LABELS = {
    "no2_trop": "NO₂ column", "aai": "Aerosol index", "co": "CO column", "so2": "SO₂ column",
    "wind_speed": "Wind speed", "wind_dir_sin": "Wind direction (N–S)", "wind_dir_cos": "Wind direction (E–W)",
    "fire_count": "Fires in hex", "frp_sum": "Fire power in hex", "fires_150km": "Fires within 150 km",
    "frp_150km": "Fire power within 150 km", "upwind_fires_150km": "Upwind fires, 150 km",
    "upwind_frp_150km": "Upwind fire power, 150 km", "doy_sin": "Season (sin)", "doy_cos": "Season (cos)",
    "month": "Month", "weekday": "Weekday", "lat": "Latitude", "lon": "Longitude",
    "neighbour_idw": "Neighbour IDW estimate", "neighbour_nearest": "Nearest neighbour value",
    "dist_nearest_km": "Distance to nearest monitor", "n_reporting": "Monitors reporting",
    "neighbour_spread": "Spread across neighbours"}
SATELLITE = {"no2_trop", "aai", "co", "so2"}
LOCATION = {"lat", "lon"}
NEIGHBOUR = {"neighbour_idw", "neighbour_nearest", "dist_nearest_km", "n_reporting", "neighbour_spread"}
WITH, WITHOUT = "m2_residual_l2", "m3_residual_l2_nogeo"


def group(feature: str) -> str:
    if feature in SATELLITE:
        return "Satellite columns"
    if feature in LOCATION:
        return "Location (lat/lon)"
    if feature in NEIGHBOUR:
        return "Neighbour stations"
    return "Wind, fire, season"


def style(ax) -> None:
    ax.set_facecolor(INK["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK["axis"])
    ax.tick_params(colors=INK["secondary"], labelsize=9)


def importance_figure(imp: pd.DataFrame, top: int = 12) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), facecolor=INK["surface"])
    xmax = imp.groupby("variant")["gain_pct"].max().max() * 1.18
    titles = {WITH: "With latitude/longitude", WITHOUT: "Without latitude/longitude"}
    for ax, variant in zip(axes, (WITH, WITHOUT)):
        sub = imp[imp["variant"] == variant].sort_values("gain_pct", ascending=False).head(top).iloc[::-1]
        colours = [GROUP_COLOUR[group(f)] for f in sub["feature"]]
        ax.barh([LABELS.get(f, f) for f in sub["feature"]], sub["gain_pct"], color=colours, height=0.72)
        for y, v in enumerate(sub["gain_pct"]):
            ax.text(v + xmax * 0.01, y, f"{v:.1f}%", va="center", fontsize=8.5, color=INK["secondary"])
        full = imp[imp["variant"] == variant]
        sat = full.loc[full["feature"].isin(SATELLITE), "gain_pct"].sum()
        loc = full.loc[full["feature"].isin(LOCATION), "gain_pct"].sum()
        ax.set_title(titles[variant], loc="left", fontsize=12, color=INK["primary"], pad=18, fontweight="semibold")
        ax.text(0, 1.015, f"Satellite columns {sat:.0f}% of gain  ·  location {loc:.0f}%",
                transform=ax.transAxes, fontsize=9, color=INK["secondary"])
        ax.set_xlim(0, xmax)
        ax.set_xlabel("Share of total split gain (%), mean of 5 folds", fontsize=9, color=INK["secondary"])
        ax.xaxis.grid(True, color=INK["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        style(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLOUR.values()]
    fig.legend(handles, GROUP_COLOUR.keys(), loc="lower center", ncol=4, frameon=False, fontsize=9,
               labelcolor=INK["secondary"])
    fig.suptitle("Remove latitude/longitude: neighbour and satellite signals replace location", x=0.01, ha="left",
                 fontsize=14, color=INK["primary"], fontweight="semibold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    path = FIG / "feature-importance-ablation.png"
    fig.savefig(path, dpi=160, facecolor=INK["surface"])
    plt.close(fig)
    return path


def ablation_markdown(imp: pd.DataFrame, ci: pd.DataFrame, sidco: pd.DataFrame) -> str:
    bands = ["0-20 km", "20-50 km", "50-100 km", "100 km+"]
    header = ("| Same model (residual on IDW) | Top 3 features by gain | Satellite share of gain | "
              + " | ".join(f"{b} vs IDW [95% CI]" for b in bands) + " | SIDCO Kurichi MAE [95% CI] |")
    lines = [header, "|" + "---|" * (5 + len(bands))]
    for variant, label in ((WITH, "With lat/lon"), (WITHOUT, "Without lat/lon")):
        full = imp[imp["variant"] == variant].sort_values("gain_pct", ascending=False)
        top3 = ", ".join(f"{LABELS.get(f, f)} {g:.0f}%" for f, g in full.head(3)[["feature", "gain_pct"]].itertuples(index=False))
        sat = full.loc[full["feature"].isin(SATELLITE), "gain_pct"].sum()
        cells = []
        for b in bands:
            r = ci[(ci["model"] == variant) & (ci["band"] == b)].iloc[0]
            cells.append(f"{r.vs_idw_pct:+.1f}% [{r.vs_idw_lo:+.1f}, {r.vs_idw_hi:+.1f}]")
        s = sidco[sidco["model"] == variant].iloc[0]
        lines.append(f"| **{label}** | {top3} | {sat:.0f}% | " + " | ".join(cells)
                     + f" | {s.mae:.1f} [{s.mae_lo:.1f}, {s.mae_hi:.1f}] |")
    return "\n".join(lines)


def paired_ablation_markdown() -> str:
    """The test that actually answers 'does removing lat/lon help': a paired interval, not two overlapping ones."""
    pp = pd.read_parquet(EXP / "per_period.parquet")
    wide = paired_frame(pp, [WITH, WITHOUT])
    delta = paired_improvement(wide, WITHOUT, WITH)
    rows = [f"| {r.band} | {r.station_days:,} | {r.improvement_pct:+.1f}% [{r.lo:+.1f}, {r.hi:+.1f}] | "
            f"{'yes' if r.lo > 0 else 'worse' if r.hi < 0 else 'not established'} |"
            for r in delta.itertuples()]
    delta.round(3).to_csv(DOCS / "latlon-ablation-paired.csv", index=False)
    return ("| Band | Station-days | Without vs with lat/lon: MAE improvement [95% CI, paired] | "
            "Removing lat/lon clearly helps? |\n|---|---|---|---|\n" + "\n".join(rows))


def sidco_figure(promoted: str) -> tuple[Path, Path]:
    pp = pd.concat([pd.read_parquet(EXP / "per_period.parquet"),
                    pd.read_parquet(EXP / "per_period_combined.parquet")], ignore_index=True)
    s = pp[pp["station_id"] == SIDCO]
    wide = s.pivot_table(index="period", columns="model", values="predicted", aggfunc="first")
    wide["actual"] = s.drop_duplicates("period").set_index("period")["actual"]
    wide = wide[["actual", promoted, "idw_k8"]].dropna().sort_index()
    wide.to_csv(DOCS / "sidco-kurichi-timeseries.csv", index_label="date_ist")

    full = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq="D"))  # gaps break the lines
    fig, ax = plt.subplots(figsize=(13, 4.8), facecolor=INK["surface"])
    for edge, name in ((30, "Good ≤30"), (60, "Satisfactory ≤60"), (90, "Moderate ≤90")):
        ax.axhline(edge, color=INK["grid"], linewidth=1, zorder=0)
        ax.text(full.index.max() + pd.Timedelta(days=3), edge, name, fontsize=8, va="center", color=INK["muted"])
    series = [("actual", "Measured (hidden from the model)", INK["secondary"], "-"),
              (promoted, "AeroGap prediction", "#2a78d6", "-"),
              ("idw_k8", "Inverse-distance baseline", "#eb6834", (0, (4, 2)))]
    for col, label, colour, dash in series:
        ax.plot(full.index, full[col], color=colour, linewidth=2 if col != "idw_k8" else 1.6, linestyle=dash,
                label=label, zorder=3 if col == promoted else 2)
    ax.set_ylabel("Daily PM2.5 (µg/m³)", fontsize=9, color=INK["secondary"])
    ax.yaxis.grid(False)
    ax.set_xlim(full.index.min(), full.index.max() + pd.Timedelta(days=28))
    ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK["secondary"], ncol=3)
    style(ax)
    ci = pd.read_csv(DOCS / "sidco-kurichi-ci.csv").set_index("model")
    m, b = ci.loc[promoted], ci.loc["idw_k8"]
    ax.set_title("SIDCO Kurichi, Coimbatore: a real monitor the model was never shown", loc="left",
                 fontsize=13, color=INK["primary"], fontweight="semibold", pad=26)
    ax.text(0, 1.03, f"{len(wide)} days · nearest reliable monitor ~72 km away · MAE {m.mae:.1f} µg/m³ "
                     f"[{m.mae_lo:.1f}, {m.mae_hi:.1f}] vs {b.mae:.1f} [{b.mae_lo:.1f}, {b.mae_hi:.1f}] for IDW · "
                     f"r {m.r:.2f} vs {b.r:.2f}", transform=ax.transAxes, fontsize=9, color=INK["secondary"])
    fig.tight_layout()
    path = FIG / "sidco-kurichi-timeseries.png"
    fig.savefig(path, dpi=160, facecolor=INK["surface"])
    plt.close(fig)
    return path, DOCS / "sidco-kurichi-timeseries.csv"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    FIG.mkdir(parents=True, exist_ok=True)
    imp = pd.read_csv(EXP / "importance.csv")
    ci = pd.read_csv(DOCS / "model-bands-ci.csv")
    sidco = pd.read_csv(DOCS / "sidco-kurichi-ci.csv")
    promoted = joblib.load(ROOT / "model" / "model.pkl")["decision"]["promoted"]

    pivot = imp.pivot_table(index="feature", columns="variant", values="gain_pct").reindex(columns=[WITH, WITHOUT])
    pivot.insert(0, "group", [group(f) for f in pivot.index])
    pivot.sort_values(WITHOUT, ascending=False).round(2).to_csv(DOCS / "feature-importance-ablation.csv")
    print("figure:", importance_figure(imp).relative_to(ROOT))

    sat = {v: imp.loc[(imp["variant"] == v) & imp["feature"].isin(SATELLITE), "gain_pct"].sum() for v in (WITH, WITHOUT)}
    (DOCS / "latlon-ablation.md").write_text(
        "# Lat/lon ablation: is the model more than a map of where pollution usually is?\n\n"
        "Same model (residual on the neighbour IDW estimate, squared loss, same folds and seed), trained twice: "
        "once with latitude/longitude, once without. If it were only memorising that some places are polluted, "
        "removing location would hurt everywhere.\n\n"
        "**What the evidence supports:** removing location does not hurt the 20-100 km bands (the paired table "
        "below says whether it measurably helps). Location's share of the model's gain is replaced by "
        f"neighbour-station features first and satellite columns second: satellite share rises from {sat[WITH]:.0f}% "
        f"to {sat[WITHOUT]:.0f}% of total gain. The 100 km+ band, where there are no neighbours to lean on, "
        "gets worse.\n\n"
        "**What it does not support:** that the satellite columns dominate. By gain they are a minority signal; "
        "the neighbour stations carry most of it. Ranking by split count (how often a feature is used) "
        "overstates the satellite columns and should not go on a slide.\n\n"
        "Per-row intervals: station bootstrap, 2,000 draws; SIDCO: 7-day block bootstrap.\n\n"
        + ablation_markdown(imp, ci, sidco)
        + "\n\n## The paired test\n\nTwo overlapping per-model intervals cannot say whether one model beats the "
        "other. This compares them on the same station bootstrap draws:\n\n"
        + paired_ablation_markdown()
        + "\n\n![Feature importance with and without lat/lon](figures/feature-importance-ablation.png)\n",
        encoding="utf-8")
    print("table:  docs/latlon-ablation.md")
    fig_path, csv_path = sidco_figure(promoted)
    print("figure:", fig_path.relative_to(ROOT), "| data:", csv_path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
