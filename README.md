# AeroGap

> Predicting the pollution your monitors miss.

India's CPCB air quality network is sparse, so most of the country has no monitoring station. AeroGap predicts AQI in the gaps. It fuses government sensors, Sentinel-5P gas columns, NASA FIRMS fire detections and wind data into a national H3 hex grid. It then flags **dark zones**: areas where predicted pollution is dangerously high and there is no monitor.

Submission for **Code for Communities 2** (Google / GDG India): Problem Statement 02, *Clean Air & Climate Resilience*.

## Status

Early development. Available so far:

- `ingest/openaq.py`: historical training corpus from the OpenAQ S3 archive → partitioned Parquet
- `ingest/cpcb.py`: live CPCB snapshot from data.gov.in → Parquet (written, **not yet validated against live data**, because data.gov.in has been returning 502s)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                # then fill in your keys
```

## Ingestion: two sources, two jobs

| | `ingest/openaq.py` | `ingest/cpcb.py` |
|---|---|---|
| **Role** | Model **training** corpus | **Live** layer on the map |
| **Source** | [OpenAQ open data archive](https://docs.openaq.org/aws/about) on S3 | [CPCB real-time AQI](https://data.gov.in/resources/real-time-air-quality-index-various-locations) via data.gov.in |
| **Time span** | Months to years of hourly history | The current reading only |
| **Values** | Raw concentrations, with explicit units | CPCB figures (sub-index vs concentration still to be confirmed, see below) |
| **Access** | Anonymous, unsigned HTTPS. No account, key or AWS credentials | Needs `DATA_GOV_IN_API_KEY` |
| **Reliability** | Static files, about a 3-day lag | Frequent 502 outages. Run it opportunistically |

**Why the split:** the data.gov.in resource returns only a *current snapshot*. You can't train a spatial interpolation model on a single timestamp, because it needs readings across many stations over many months. OpenAQ archives that history. CPCB stays as the freshest real-time signal for the map.

### Historical corpus (OpenAQ)

```bash
# small test slice first: one month, one state
python -m ingest.openaq --start 2026-08-01 --end 2026-08-31 --states Delhi

# full default: India, last 12 months
python -m ingest.openaq

# widen or narrow later
python -m ingest.openaq --start 2024-01-01 --states "Delhi,Haryana,Uttar Pradesh"
python -m ingest.openaq --skip-download          # rebuild Parquet from the local cache only
```

The pipeline has four steps:

1. **Discover.** The archive's record files have no country or state fields, and its `provider=/country=` tree is a frozen legacy copy (India's CPCB providers stop in 2022). So every location ID is probed once for its years and coordinates. The first run covers about 55k locations and takes roughly an hour. Results are cached in `data/raw/openaq/<snapshot>/locations_probe.jsonl`, and interrupted runs resume where they stopped.
2. **Geolocate.** Stations get a state (ADM1) and district (ADM2) by point-in-polygon against [geoBoundaries](https://www.geoboundaries.org/). Anything outside the country's boundaries is dropped, which is how the country filter works.
3. **Download.** The script mirrors the daily `.csv.gz` files to `data/raw/openaq/<snapshot>/records/`. A file already on disk with the size S3 reports is never downloaded again.
4. **Build.** Rows are cleaned and written month by month.

Output is `data/history/state=<state>/parameter=<p>/month=<YYYY-MM>/part-0.parquet`, in a long schema:

| column | notes |
|---|---|
| `station_id` | `openaq-<location_id>` |
| `station_name`, `lat`, `lon` | as reported by OpenAQ |
| `city` | ADM2 **district** containing the station (the archive has no city field) |
| `state` | ADM1 state/UT, ASCII-folded (`Tamil Nadu`) |
| `parameter`, `value`, `unit` | native units, normalised spelling (`ug/m3`, `ppm`, `ppb`) |
| `timestamp_utc` | UTC; marks the **end** of the averaging period |

**Cleaning:** the script drops rows with null values, bad timestamps, missing coordinates, negative values, sentinels (`999`, `9999`, …), physically implausible readings, and duplicate sensor-timestamp pairs. Each run logs how many rows it dropped for each reason. It also writes `data/history/_summary.json` (stations, date range, rows per parameter, states covered, unit and value ranges) and `_stations.parquet`.

**Partitions:** each partition always contains every cached day of its month, so a narrow re-run (one state, part of a month) can't overwrite data from a wider one.

### Live snapshot (CPCB)

```bash
python -m ingest.cpcb                 # replay the latest cached snapshot, or pull if none
python -m ingest.cpcb --refresh       # force a live pull from data.gov.in
```

| File | Contents |
|---|---|
| `data/raw/cpcb/<timestamp>/page_*.json` | Raw API pages, the on-disk cache |
| `data/cpcb/readings_<timestamp>.parquet` | Long format: one row per station × pollutant |
| `data/cpcb/stations_<timestamp>.parquet` | Wide format: one row per station, with pollutant columns and the computed NAQI |
| `data/cpcb/stations_latest.parquet` | Copy of the most recent stations snapshot |

### Units: findings, and the reconciliation still to build

These findings come from a test run on Delhi for August 2026: 51 stations and 1.13M rows. OpenAQ's Indian government-station series come from the same CPCB network as data.gov.in, and several of their **unit labels are wrong**:

| parameter | OpenAQ label | observed p5 to p95 | what the values actually are |
|---|---|---|---|
| `co` | ppb | 0.24 to 2.15 | **mg/m³**, CPCB's native CO unit. Outdoor CO around 1 ppb is physically impossible |
| `no`, `no2` | ppb | 1.5 to 39, and 4.1 to 59 | **µg/m³** (proven by the NOx check below) |
| `nox` | ppb | 0.01 to 0.05 | **ppm** (proven by the same check) |
| `so2` | ppb | 1.1 to 28.5 | probably µg/m³, since it comes through the same pipeline, but not proven |
| `o3`, `pm25`, `pm10` | µg/m³ | as expected | µg/m³, correctly labelled |

**NOx check:** across 101,438 hourly rows, NOx×1000 equals NO + NO2 converted from µg/m³ to ppb. The median ratio is 1.000, and the middle half of rows falls between 0.998 and 1.002. Reading NO and NO2 as ppb instead gives a median ratio of 0.60.

Two reconciliation layers are needed before any feature engineering. **Neither is built yet:**

1. **OpenAQ unit correction:** relabel or convert the government-station series above. Low-cost sensors in the corpus, such as PM2.5-only community monitors, need a separate sensor-class flag.
2. **CPCB data.gov.in reconciliation:** the live feed is believed to publish per-pollutant **sub-index** values rather than concentrations, but that hasn't been confirmed, because data.gov.in has been returning 502s. Both sources use identical station names (for example, `Anand Vihar, Delhi - DPCC`), so the test is to join them on station and hour and compare. If the values turn out to be sub-indices, convert them using the CPCB NAQI breakpoints.

`ingest/openaq.py` stores values **exactly as published** (native units, original labels). Corrections belong in the feature layer, where they can be tested and reversed.

## Data sources

| Source | Use | Licence |
|---|---|---|
| [OpenAQ archive](https://openaq.org/) | Historical station readings (training) | CC BY 4.0 (per-provider terms apply) |
| [CPCB real-time AQI via data.gov.in](https://data.gov.in/resources/real-time-air-quality-index-various-locations) | Live station readings | GODL-India |
| [geoBoundaries](https://www.geoboundaries.org/) IND ADM1 / ADM2 | State and district assignment | CC BY 2.5 IN / ODbL |
| [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) | Active fire detections | Open |
| Sentinel-5P TROPOMI via Google Earth Engine | NO2 column, aerosol index | Copernicus open |
| open-meteo / IMD | Wind vectors | CC-BY 4.0 |

## Licence

Code: [MIT](LICENSE). Data and docs: CC-BY 4.0.
