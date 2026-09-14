# AeroGap

> Predicting the pollution your monitors miss.

## What this is

A submission for **Code for Communities 2** (Google / GDG India, hosted on Hack2skill), Problem Statement 02 — *Clean Air & Climate Resilience*. Deadline: **30 September 2026** (submit 29th).

**Solo developer. Zero budget — no Google Cloud billing account.** All infrastructure must run on free tiers with no credit card.

## The core idea

India's CPCB air quality network is sparse. Most of the country has no monitoring station, so hyper-local pollution events — industrial plumes, crop burning, seasonal smog — go undetected entirely.

AeroGap predicts AQI in the gaps. It fuses government sensors, satellite gas columns, satellite fire detections and wind data into a national H3 hex grid, then flags **"dark zones"**: areas with dangerously high *predicted* pollution and **no monitor at all**. Citizen photos analysed by Gemini act as virtual sensors where no hardware exists.

The differentiator versus every other team: they will build an AQI dashboard on existing stations. AeroGap predicts where stations aren't. The problem brief's exact complaint is that cities "consistently miss" hyper-local events — this answers the word *miss*.

## Judging rubric (drives every decision)

| Criterion | Weight |
|---|---|
| AI/Technical Execution — *is Google AI doing meaningful work, does it run end-to-end* | 25% |
| Problem-Solution Fit | 20% |
| Depth & Reach Across India | 20% |
| Deployability & Scalability | 20% |
| Impact Potential | 15% |

**40% of the score is scale.** Therefore: the grid is national from day one, never single-city. A district is a config entry, never a code change. The demo switches between two cities live to prove it.

**Gemini must be load-bearing**, not decorative. It is the virtual sensor and the source-attribution reasoner — the product cannot function without it.

## Tech stack (all free tier, no billing)

- **Data:** CPCB AQI API (data.gov.in), NASA FIRMS, Sentinel-5P NO2/aerosol via Google Earth Engine, IMD/open-meteo wind
- **Storage:** DuckDB + Parquet locally. *Not* BigQuery — no billing account.
- **Grid:** H3 hexagons, resolution 7 (~5km) nationally, resolution 9 (~200m) in active zones
- **Models:** LightGBM trained locally, shipped as `.pkl`. *Not* Vertex AI.
- **AI:** Gemini via AI Studio free-tier key (single key — see caching rule below)
- **Backend:** FastAPI on Hugging Face Spaces or Render free tier
- **Frontend:** React + MapLibre GL + deck.gl hex layer, on Vercel or Netlify. *Not* Google Maps Platform (needs billing).
- **Language/voice:** Gemini for translation, browser Web Speech API for TTS

Google Maps and Vertex are "recommended" by the organisers, not mandatory. Gemini is what satisfies the mandatory Google AI integration.

## Hard rules

1. **Never commit API keys.** Keys live in `.env`, which is gitignored. Commit `.env.example` with empty placeholders. The CPCB and FIRMS keys took days to obtain.
2. **Cache every Gemini response to disk**, keyed by a hash of the input. The demo replays from cache — zero calls, instant, never fails on stage. Build the cache layer *before* any Gemini logic.
3. **Cache every external API response.** Never live-call CPCB, FIRMS or Earth Engine during a recorded demo or live pitch.
4. **Export Earth Engine data once** to CSV, then work locally. Do not re-query per run.
5. **Deploy early and keep it deployed.** A live URL is a mandatory submission item. Ship ugly on day 3, improve after.
6. **Push to GitHub every evening.** Solo on one laptop — the remote is the backup, and commit history spread over two weeks is evidence of real work.
7. **Single Gemini account.** Multi-account key rotation violates Google's terms and this is a Google-sponsored hackathon.

## Scope — what is IN

- National CPCB ingest → Parquet
- H3 grid generation
- LightGBM spatial interpolation model (dark-zone prediction with confidence bands)
- Map UI with confidence-shaded hex layer
- Gemini: citizen photo → structured JSON (haze severity + likely source)
- Gemini: co-located signals → source attribution + drafted multilingual alert
- Two-city switch (Delhi-NCR and Coimbatore–Tiruppur) driven by config
- Deployed live URL

## Scope — what is OUT (roadmap slide only)

- 24–72h forecast model (if time permits, a persistence-plus-wind heuristic demos identically)
- Live district-onboarding API endpoint (replaced by a JSON config file, shown in the video)
- Federated weight-exchange demo (one slide describes it)
- Real WhatsApp dispatch (show the payload, send to own number)

## Model features

Interpolation regressor predicts AQI per hex from:
- nearest-station AQI + distance to nearest station
- TROPOMI NO2 column, aerosol index
- upwind FIRMS fire count within radius
- wind vector (speed, direction)
- hour of day, day of year
- land-use class, elevation

Must degrade gracefully: a district with zero stations still yields a usable prediction from satellite + fire + wind alone, at lower confidence. This is the deployability story — it works in the places that need it most.

## Repo layout

```
aerogap/
├── README.md
├── LICENSE          # MIT
├── .gitignore       # .env, data/, *.parquet
├── .env.example
├── data/            # gitignored
├── ingest/          # cpcb.py, firms.py, gee_export.py
├── model/           # features.py, train.py, model.pkl
├── api/             # FastAPI service
├── web/             # React + MapLibre
└── docs/            # architecture diagram, screenshots
```

## Digital Public Good compliance

The organisers use DPG language throughout, so the repo must demonstrate it:
- MIT licence (code), CC-BY 4.0 (data/docs)
- README with architecture, setup, data sources
- No PII: strip EXIF from citizen photos, store coarse geohash only
- DPDP Act 2023: consent notice before upload, purpose limitation, right to withdraw
- Open standards: GeoJSON, H3, OpenAPI
- Do-no-harm: confidence bands on every prediction, no individual targeting

## Timeline

- **Sep 14–16** — CPCB national ingest, H3 grid, map rendering real stations, deployed to Vercel
- **Sep 17–20** — FIRMS + Sentinel-5P features, train LightGBM, dark-zone predictions rendering
- **Sep 21–24** — Gemini cache layer, photo analysis, attribution, multilingual alerts
- **Sep 25–26** — Coimbatore config, polish, README, freeze features
- **Sep 27–29** — deck (10–12 slides), demo video (3–5 min), submit on the 29th

If slipping: cut Gemini attribution before cutting the deployed map. A map with predictions still demos; Gemini with no map does not.

## Submission checklist

1. Public GitHub repo
2. Demo video, 3–5 min, end-to-end walkthrough
3. Pitch deck, 10–12 slides
4. Brief description, 2–3 lines
5. Live deployed link
