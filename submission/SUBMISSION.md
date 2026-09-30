# AeroGap — submission pack

Everything needed to fill in the Hack2skill form, plus the deck content in plain text as a
backup. Deck file: `AeroGap-pitch-deck.pptx` (12 slides, speaker notes on every slide).

## Links

| Item | Value |
|---|---|
| Live demo | https://mirdula18-aerogap.static.hf.space |
| Public repo | https://github.com/Mirdula18/AeroGap |
| Problem statement | 02 — Clean Air & Climate Resilience |
| Licence | MIT (code), CC-BY 4.0 (data and docs) |

## Brief description (2–3 lines)

> India's 668 public air-quality monitors leave most of the country unmeasured, so
> hyper-local pollution goes undetected. AeroGap predicts daily PM2.5 for all 623,207
> five-kilometre hexagons of India by fusing monitors with Sentinel-5P chemistry, NASA fire
> detections and wind, and flags "dark zones" where pollution is high and no monitor exists.
> Gemini turns each prediction into an attributed cause and a ready-to-send alert in English,
> Tamil and Hindi — and a citizen's sky photo becomes a virtual sensor where there is no
> hardware at all.

Shorter variant, if the field is tight:

> AeroGap predicts PM2.5 for every 5 km hexagon in India — including the places with no
> monitor — and uses Gemini to name the likely source and draft a multilingual alert for the
> authority who can act on it.

## Deck outline (what each slide argues)

| # | Slide | The one thing it proves |
|---|---|---|
| 1 | AeroGap | It exists and it is live |
| 2 | Where there is no monitor, there is no number | The problem is coverage, not dashboards |
| 3 | Everyone else maps the stations | The differentiator, stated bluntly |
| 4 | One click: estimate, cause, alert | The product, in one screenshot |
| 5 | Five open sources, one national grid | Technical architecture, no billing account |
| 6 | Gemini is load-bearing | Google AI does real work (25% of the rubric) |
| 7 | How we stop the model inventing things | Grounding is enforced and checked |
| 8 | Validated by hiding entire monitors | Honest, pre-registered results |
| 9 | The map admits what it does not know | Do-no-harm, confidence-proportionate action |
| 10 | National from the first commit | Depth and reach (20%) |
| 11 | Zero budget, live, demo-proof | Deployability (20%) |
| 12 | The alert is the product | Impact and roadmap |

## The numbers (all verified — use these exactly)

**Coverage**
- 668 public monitors nationwide; 508 healthy, 71 intermittent, 13 stuck on a placeholder,
  **76 dead**
- 623,207 hexagons, H3 resolution 7, ~5 km across, covering all 36 states and union
  territories
- Training corpus: 146.2M readings, 13 Sept 2025 – 10 Sept 2026

**Model** — leave-station-out cross-validation, 2,000-draw station bootstrap

| Distance to nearest monitor | MAE µg/m³ | Improvement over IDW | Verdict |
|---|---|---|---|
| 0–20 km | 16.8 [15.8, 17.8] | −3.8% [−5.8, −1.7] | interpolation wins |
| 20–50 km | 18.6 [16.4, 21.0] | +12.1% [+6.8, +17.0] | real gain |
| 50–100 km | 15.0 [13.4, 17.0] | +7.3% [+1.5, +12.6] | real gain |
| 100 km+ | 21.4 [16.5, 28.3] | +6.9% [−7.7, +21.4] | not demonstrable |

- Confidence shown to users: ±31% within 20 km, ±38% at 20–100 km, ±59% beyond 100 km
- The decision rule was committed to git **before** the evaluation ran. The pre-registered
  model failed its own test at 100 km+, so the simpler variant was promoted and the failure
  published in `docs/model-decision.md`.

**Gemini**
- `gemini-3.6-flash`, AI Studio free tier, one key, three jobs: virtual sensor, source
  attribution, multilingual alert
- Every call cached to disk by a hash of model + prompt version + request body
- Offline replay verified: **0 live calls, 16 cache hits** — the demo cannot fail on stage
- Free-tier limits learned the hard way: 5 requests/minute, 20/day per model. Attribution and
  alerts are therefore batched to one call per date.

## The claim to make, and the claims to avoid

**Say:** within 100 km of a working monitor we beat interpolation by 7–12% with intervals
that hold; beyond 100 km we cannot demonstrate an improvement, and the map says so.

**Never say:** "95% accurate", "beats existing systems everywhere", or any figure without its
interval. The honesty is the differentiator — don't trade it for a rounder number.

## Demo video script (3–5 minutes)

1. **0:00–0:30 — The gap.** Open the live map on "Monitoring gaps", All India. Point at the
   dark blue: 100–200 km from any working monitor. "76 of India's 668 monitors published
   nothing usable this year. This is what nobody is measuring."
2. **0:30–1:15 — The prediction.** Switch to "Predicted PM2.5", 12 Nov 2025. Zoom to Delhi.
   Explain the colour, and that grey means low confidence. "Every hexagon has a number and an
   honest error bar."
3. **1:15–2:15 — Gemini attribution.** Click the ringed Delhi hexagon. Read the source, the
   cited evidence and the uncertainty line. Then click the Leh hexagon: low confidence, and
   the action changes to "send a portable monitor". "The model is allowed to say it doesn't
   know."
4. **2:15–3:00 — The alert.** Same panel, scroll to the drafted alert. Switch Authority →
   Citizens, English → தமிழ் → हिंदी. "This is the last step before somebody acts."
5. **3:00–3:40 — Virtual sensor.** Click a citizen photo. Show haze 85/100 and the fused
   estimate shifting 173 → 214 µg/m³ at 27% weight. Mention EXIF stripping and the consent
   notice.
6. **3:40–4:20 — Scale and deployment.** Switch Delhi-NCR → Coimbatore–Tiruppur. "Same grid,
   same weights, one config entry. National from the first commit." Note ₹0 infrastructure,
   free tier throughout, and that every Gemini response is cached, so this demo never calls an
   API.
7. **4:20–4:40 — Close.** Back to the national view. "AeroGap predicts the pollution your
   monitors miss — and tells someone who can do something about it."

Deep links for recording (append to the live URL):
- `?city=all-india&view=gap&date=2025-11-12`
- `?city=delhi-ncr&date=2025-11-12&explain=873da1140ffffff`
- `?city=all-india&date=2025-11-22&explain=873d06b2bffffff` (Leh, low confidence)
- `?city=delhi-ncr&date=2025-11-22&photo=delhi-road-smog`
- `?city=coimbatore-tiruppur&date=2026-09-10`

## Rubric mapping (for the written submission fields)

| Criterion | Weight | Evidence to cite |
|---|---|---|
| AI / technical execution | 25% | Three load-bearing Gemini features, schema-constrained and citation-checked; LightGBM with leave-station-out validation; runs end-to-end, live |
| Problem–solution fit | 20% | The brief says cities "consistently miss" hyper-local events; we predict where no station exists |
| Depth & reach across India | 20% | National grid from day one, 623,207 hexagons, 36 states, two cities live from one config |
| Deployability & scalability | 20% | Zero-budget free-tier stack, live URL, degrades gracefully to satellite-only where there are no monitors |
| Impact potential | 15% | Role-addressed multilingual alerts, confidence-proportionate actions, DPG-compliant and openly licensed |

## Images used in the deck

In `docs/screenshots/`, all captured from the live deployment:

| File | Shows |
|---|---|
| `national-gaps.png`, `deck-national-map.png` | National monitoring-gap map |
| `attribution-panel.png` | Delhi: prediction, Gemini attribution, drafted alert |
| `low-confidence.png`, `deck-lowconf-panel.png` | Leh: low confidence, verification-first action |
| `photo-sensor.png`, `deck-photo-panel.png` | Citizen photo fused into the grid |
| `dark-zone-alert.png` | Dark zone, 56 km from the nearest monitor |
| `coimbatore.png` | The second city, same model |
