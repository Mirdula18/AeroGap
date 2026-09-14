# AeroGap

> Predicting the pollution your monitors miss.

India's CPCB air quality network is sparse, so most of the country has no monitoring station. AeroGap predicts AQI in the gaps. It fuses government sensors, Sentinel-5P gas columns, NASA FIRMS fire detections and wind data into a national H3 hex grid. It then flags **dark zones**: areas where predicted pollution is dangerously high and there is no monitor.

Submission for **Code for Communities 2** (Google / GDG India): Problem Statement 02, *Clean Air & Climate Resilience*.

## Status

Early development. Available so far:

- `ingest/cpcb.py`: national CPCB real-time AQI ingest (data.gov.in) → Parquet

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                # then fill in your keys
```

## Ingest

```bash
python -m ingest.cpcb                 # uses cached responses when present
python -m ingest.cpcb --refresh       # force a live pull from data.gov.in
```

Outputs (gitignored, under `data/`):

| File | Contents |
|---|---|
| `data/raw/cpcb/<timestamp>/page_*.json` | Raw API pages, the on-disk cache |
| `data/cpcb/readings_<timestamp>.parquet` | Long format: one row per station × pollutant |
| `data/cpcb/stations_<timestamp>.parquet` | Wide format: one row per station, with pollutant columns and the computed NAQI |
| `data/cpcb/stations_latest.parquet` | Copy of the most recent stations snapshot |

## Data sources

| Source | Use | Licence |
|---|---|---|
| [CPCB real-time AQI via data.gov.in](https://data.gov.in/resources/real-time-air-quality-index-various-locations) | Ground truth station readings | GODL-India |
| [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) | Active fire detections | Open |
| Sentinel-5P TROPOMI via Google Earth Engine | NO2 column, aerosol index | Copernicus open |
| open-meteo / IMD | Wind vectors | CC-BY 4.0 |

## Licence

Code: [MIT](LICENSE). Data and docs: CC-BY 4.0.
