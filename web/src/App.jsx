import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import { MapboxOverlay } from "@deck.gl/mapbox";
import { H3HexagonLayer } from "@deck.gl/geo-layers";
import { IconLayer, ScatterplotLayer } from "@deck.gl/layers";
import { cellToLatLng, cellToParent, getResolution } from "h3-js";
import cities from "./cities.json";
import { ExplanationPanel, PhotoPanel } from "./Detail.jsx";

// With an API URL the map queries it per viewport; without one it reads the static
// snapshot in public/data (api/export_static.py, model/predict.py, gemini/*), which the free static host serves.
const API = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");
const STATIC_BASE = `${import.meta.env.BASE_URL}data`.replace(/\/{2,}/g, "/");

// Free, keyless basemap (CARTO Positron, OSM data).
const BASEMAP = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json";

// Monitoring-gap view: sequential single-hue ramp (blue 100 -> 700), binned because distance is skewed.
const BINS = [
  { max: 5, label: "< 5", color: "#cde2fb" },
  { max: 10, label: "5–10", color: "#9ec5f4" },
  { max: 25, label: "10–25", color: "#6da7ec" },
  { max: 50, label: "25–50", color: "#3987e5" },
  { max: 100, label: "50–100", color: "#256abf" },
  { max: 200, label: "100–200", color: "#184f95" },
  { max: Infinity, label: "200+", color: "#0d366b" },
];

// Prediction view: CPCB 24-hour PM2.5 categories (µg/m³), the scale Indian readers already know.
const CATEGORIES = [
  { max: 30, label: "Good", range: "0–30", color: "#3a9e5c" },
  { max: 60, label: "Satisfactory", range: "31–60", color: "#9cc94a" },
  { max: 90, label: "Moderate", range: "61–90", color: "#f2c230" },
  { max: 120, label: "Poor", range: "91–120", color: "#ef8a2e" },
  { max: 250, label: "Very poor", range: "121–250", color: "#d9432b" },
  { max: Infinity, label: "Severe", range: "250+", color: "#8c1d2c" },
];
// Three levels, from validation on hidden monitors, by distance to the nearest reporting monitor.
// Low confidence blends toward a neutral grey, never toward white: a faded "Severe" must not read as clean air.
const CONFIDENCE_MIX = [0, 0.4, 0.75];
const CONFIDENCE_LABEL = ["High", "Medium", "Low"];
const CONFIDENCE_RANGE = ["within 20 km", "20–100 km", "beyond 100 km"];
const NEUTRAL = [163, 162, 156];

// Monitor health uses the reserved status palette, always paired with a label (legend + tooltip)
// and a shape cue: filled = delivers usable data, hollow ring = reports garbage or nothing.
const HEALTH = {
  healthy: { label: "Healthy", fill: [12, 163, 12, 255], line: [252, 252, 251, 255], width: 2, radius: 5 },
  intermittent: { label: "Intermittent (<50% usable)", fill: [250, 178, 25, 255], line: [252, 252, 251, 255], width: 2, radius: 4 },
  stuck: { label: "Stuck sensor", fill: [252, 252, 251, 0], line: [236, 131, 90, 255], width: 3, radius: 5.5 },
  dead: { label: "Dead (no usable data)", fill: [137, 135, 129, 255], line: [208, 59, 59, 255], width: 3, radius: 5.5 },
};
const HEALTH_ORDER = ["healthy", "intermittent", "stuck", "dead"];
const healthOf = (d) => (HEALTH[d.health] ? d.health : "healthy");

// A camera pin for virtual-sensor photos (inline SVG, no external asset).
const CAMERA_ICON = {
  url: `data:image/svg+xml;charset=utf-8,${encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 48 48">' +
      '<circle cx="24" cy="24" r="21" fill="#0b0b0b" stroke="#fcfcfb" stroke-width="3"/>' +
      '<rect x="12" y="17" width="24" height="16" rx="3" fill="#fcfcfb"/><rect x="19" y="13" width="10" height="5" rx="1.5" fill="#fcfcfb"/>' +
      '<circle cx="24" cy="25" r="5" fill="#0b0b0b"/></svg>'
  )}`,
  width: 48,
  height: 48,
  anchorY: 24,
};

const hexToRgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
const BIN_RGB = BINS.map((b) => hexToRgb(b.color));
const CAT_RGB = CATEGORIES.map((c) => hexToRgb(c.color));
const gapColor = (km) => BIN_RGB[BINS.findIndex((b) => km < b.max)];
const categoryOf = (pm) => CATEGORIES.findIndex((c) => pm <= c.max);
const mix = (rgb, t) => rgb.map((v, i) => Math.round(v * (1 - t) + NEUTRAL[i] * t));
const predColor = (pm, confidence) => mix(CAT_RGB[categoryOf(pm)], CONFIDENCE_MIX[confidence] ?? 0.75);
const rgbCss = (rgb) => `rgb(${rgb.join(",")})`;

// Same inverse-variance blend as gemini/photo_sensor.fuse: one photo nudges, it does not override.
function fusePhoto(prediction, obs) {
  if (!prediction || obs.status !== "ok" || !(obs.effective_confidence > 0)) return null;
  const sigmaPred = Math.max(prediction.u, 1);
  const sigmaObs = (obs.pm25_high - obs.pm25_low) / 2 / Math.max(obs.effective_confidence, 0.05);
  const wObs = 1 / sigmaObs ** 2;
  const wPred = 1 / sigmaPred ** 2;
  const weight = wObs / (wObs + wPred);
  return { fused: weight * obs.pm25_mid + (1 - weight) * prediction.pm, weight };
}

async function sha256Hex(file) {
  const digest = await crypto.subtle.digest("SHA-256", await file.arrayBuffer());
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

export default function App() {
  const params = useMemo(() => new URLSearchParams(window.location.search), []);
  const mapContainer = useRef(null);
  const mapRef = useRef(null);
  const overlayRef = useRef(null);
  const loadRef = useRef(null);
  const [grid, setGrid] = useState(null);
  const [meta, setMeta] = useState(null);
  const [stations, setStations] = useState([]);
  const [hover, setHover] = useState(null);
  const [status, setStatus] = useState("Loading…");
  // ?city=delhi-ncr&view=gap&date=2025-11-22 open straight on a state (for the demo video and screenshots).
  const initialCity = useMemo(() => cities.find((c) => c.id === params.get("city")) || cities[0], [params]);
  const [active, setActive] = useState(initialCity.id);
  const [view, setView] = useState(params.get("view") === "gap" || API ? "gap" : "predicted");
  const [date, setDate] = useState(params.get("date"));
  const [explanations, setExplanations] = useState(null);
  const [sensors, setSensors] = useState(null);
  const [photos, setPhotos] = useState([]);
  const [selected, setSelected] = useState(null);
  const [uploadMsg, setUploadMsg] = useState("");
  const viewRef = useRef(view);
  const dateRef = useRef(date);
  viewRef.current = view;
  dateRef.current = date;

  const predictions = meta?.predictions;
  const hasPredictions = Boolean(predictions?.dates?.length) && !API;
  const activeDate = hasPredictions ? (predictions.dates.find((d) => d.date === date) || predictions.dates[0]) : null;
  const showPredicted = view === "predicted" && hasPredictions;

  useEffect(() => {
    const map = new maplibregl.Map({
      container: mapContainer.current,
      style: BASEMAP,
      center: [initialCity.longitude, initialCity.latitude],
      zoom: initialCity.zoom,
      minZoom: 3,
      maxZoom: 12,
      attributionControl: { compact: true },
    });
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "bottom-right");
    const overlay = new MapboxOverlay({ interleaved: false, layers: [] });
    map.addControl(overlay);
    mapRef.current = map;
    overlayRef.current = overlay;

    let timer;
    let controller;
    const cache = new Map();
    const staticFile = async (name) => {
      if (!cache.has(name)) cache.set(name, fetch(`${STATIC_BASE}/${name}`).then((r) => {
        if (!r.ok) throw new Error(`${name} ${r.status}`);
        return r.json();
      }));
      return cache.get(name);
    };
    const cityAt = (m, center, zoom) => zoom >= 7.5 && m.cities.find(({ bbox: [w, s, e, n] }) =>
      center.lng >= w && center.lng <= e && center.lat >= s && center.lat <= n);

    const load = () => {
      clearTimeout(timer);
      timer = setTimeout(async () => {
        controller?.abort();
        controller = new AbortController();
        try {
          setStatus("Loading hexes…");
          let data;
          let kind = "gap";
          if (API) {
            const b = map.getBounds();
            const bbox = [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()].map((v) => v.toFixed(3)).join(",");
            const res = await fetch(`${API}/grid?bbox=${bbox}`, { signal: controller.signal });
            if (!res.ok) throw new Error(`API ${res.status}`);
            data = await res.json();
          } else {
            const m = await staticFile("meta.json");
            setMeta(m);
            const city = cityAt(m, map.getCenter(), map.getZoom());
            const zoom = map.getZoom();
            if (viewRef.current === "predicted" && m.predictions?.dates?.length) {
              const entry = m.predictions.dates.find((d) => d.date === dateRef.current) || m.predictions.dates[0];
              data = await staticFile(city ? entry.cities[city.id] : entry.national[zoom < 3.5 ? "4" : "5"]);
              kind = "predicted";
              staticFile(`explanations_${entry.date}.json`).then(setExplanations).catch(() => setExplanations(null));
            } else {
              data = await staticFile(city ? city.file : m.national[zoom < 3.5 ? "4" : zoom < 6 ? "5" : "6"]);
            }
          }
          setGrid({ ...data, kind });
          setStatus(`${data.count.toLocaleString()} hexes · H3 res ${data.res}` +
            (kind === "predicted" ? ` · ${data.date}` : API ? "" : " · static snapshot"));
        } catch (err) {
          if (err.name !== "AbortError") {
            setStatus(API
              ? `Could not reach the API (${err.message}). It may be waking up — retrying on the next move.`
              : `Could not load the map data (${err.message}).`);
          }
        }
      }, 250);
    };
    loadRef.current = load;
    // Don't wait for the basemap's "load" event: if the tile/style host is slow
    // or blocked, the hex layer (the actual product) must still render.
    load();
    map.on("moveend", load);

    fetch(API ? `${API}/stations` : `${STATIC_BASE}/stations.json`)
      .then((r) => r.json())
      .then((d) => setStations(Array.isArray(d) ? d : d.stations))
      .catch(() => {});
    if (!API) {
      fetch(`${STATIC_BASE}/virtual_sensors.json`).then((r) => r.json()).then(setSensors).catch(() => {});
    }

    return () => {
      clearTimeout(timer);
      map.remove();
    };
  }, []);

  useEffect(() => {
    loadRef.current?.();
  }, [view, date]);

  // Unhealthy monitors drawn last so they sit on top of healthy neighbours.
  const stationsByHealth = useMemo(
    () => [...stations].sort((a, b) => HEALTH_ORDER.indexOf(healthOf(a)) - HEALTH_ORDER.indexOf(healthOf(b))),
    [stations]
  );
  const healthCounts = useMemo(
    () => stations.reduce((acc, s) => ({ ...acc, [healthOf(s)]: (acc[healthOf(s)] || 0) + 1 }), {}),
    [stations]
  );

  const hexRows = useMemo(() => {
    if (!grid) return [];
    if (grid.kind === "predicted") {
      return grid.h3.map((h, i) => ({ kind: "predicted", h3: h, pm: grid.pm25[i], u: grid.uncertainty[i],
        dist: grid.dist_km[i], c: grid.confidence[i] }));
    }
    // d = distance to nearest WORKING monitor (drives the shading); dn = nearest listed monitor.
    const working = grid.dist_working_km || grid.dist_km;
    return grid.h3.map((h, i) => ({ kind: "gap", h3: h, d: working[i], dn: grid.dist_km[i], s: grid.station_count[i] }));
  }, [grid]);

  const explained = useMemo(
    () => (showPredicted && explanations && activeDate && explanations.date === activeDate.date
      ? Object.entries(explanations.hexes).map(([h3, e]) => ({ h3, ...e }))
      : []),
    [explanations, showPredicted, activeDate]
  );

  // A click at any loaded resolution (7 in a city, 5 or 4 nationally) resolves to the explanation inside it.
  const explanationFor = useCallback(
    (cell) => explained.find((e) => e.h3 === cell || e.h3_r5 === cell || e.h3_r4 === cell) || null,
    [explained]
  );

  // The prediction for a photo's hex at whatever resolution is currently loaded.
  const predictionAt = useCallback((cell) => {
    if (!grid || grid.kind !== "predicted") return null;
    const target = getResolution(cell) > grid.res ? cellToParent(cell, grid.res) : cell;
    return hexRows.find((r) => r.h3 === target) || null;
  }, [grid, hexRows]);

  useEffect(() => {
    if (!overlayRef.current) return;
    overlayRef.current.setProps({
      layers: [
        new H3HexagonLayer({
          id: "hexes",
          data: hexRows,
          getHexagon: (d) => d.h3,
          getFillColor: (d) => (d.kind === "predicted" ? [...predColor(d.pm, d.c), 215] : [...gapColor(d.d), 200]),
          filled: true,
          stroked: false,
          extruded: false,
          highPrecision: "auto",
          pickable: true,
          onHover: (info) => setHover(info.object ? { x: info.x, y: info.y, ...info.object } : null),
          onClick: (info) => {
            if (!info.object || info.object.kind !== "predicted") return;
            setSelected({ kind: "hex", entry: explanationFor(info.object.h3), fallback: info.object });
          },
          updateTriggers: { getFillColor: [grid] },
        }),
        new ScatterplotLayer({
          id: "stations",
          data: stationsByHealth,
          getPosition: (d) => [d.lon, d.lat],
          getFillColor: (d) => HEALTH[healthOf(d)].fill,
          getLineColor: (d) => HEALTH[healthOf(d)].line,
          lineWidthUnits: "pixels",
          getLineWidth: (d) => HEALTH[healthOf(d)].width,
          stroked: true,
          filled: true,
          radiusUnits: "pixels",
          getRadius: (d) => HEALTH[healthOf(d)].radius,
          pickable: true,
          onHover: (info) => setHover(info.object ? { kind: "station", x: info.x, y: info.y, ...info.object } : null),
        }),
        new ScatterplotLayer({
          id: "explained",
          data: explained,
          getPosition: (d) => [d.lon, d.lat],
          getFillColor: [252, 252, 251, 0],
          getLineColor: [252, 252, 251, 255],
          stroked: true,
          filled: true,
          lineWidthUnits: "pixels",
          getLineWidth: 3,
          radiusUnits: "pixels",
          getRadius: 11,
          pickable: true,
          onHover: (info) => setHover(info.object ? { kind: "explained", x: info.x, y: info.y, ...info.object } : null),
          onClick: (info) => info.object && setSelected({ kind: "hex", entry: info.object, fallback: null }),
        }),
        new IconLayer({
          id: "photos",
          data: photos,
          getPosition: (d) => { const [lat, lng] = cellToLatLng(d.h3); return [lng, lat]; },
          getIcon: () => CAMERA_ICON,
          getSize: 30,
          sizeUnits: "pixels",
          pickable: true,
          onClick: (info) => info.object && setSelected({ kind: "photo", obs: info.object }),
        }),
      ],
    });
  }, [hexRows, stationsByHealth, grid, explained, photos, explanationFor]);

  const flyTo = (city) => {
    setActive(city.id);
    mapRef.current?.flyTo({ center: [city.longitude, city.latitude], zoom: city.zoom, duration: 1600 });
  };

  const applyPhoto = (obs) => {
    setPhotos((prev) => (prev.some((p) => p.id === obs.id) ? prev : [...prev, obs]));
    setSelected({ kind: "photo", obs });
    setUploadMsg("");
    const [lat, lng] = cellToLatLng(obs.h3);
    mapRef.current?.flyTo({ center: [lng, lat], zoom: 8.2, duration: 1600 });
  };

  const onUpload = async (event) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file || !sensors) return;
    const hash = await sha256Hex(file);
    const match = sensors.observations.find((o) => o.sha256 === hash);
    if (match) {
      applyPhoto(match);
    } else {
      setUploadMsg("This photo isn't in the demo cache. Live analysis needs the Gemini backend, which this static demo "
        + "doesn't run. Try one of the demo photos above.");
    }
  };

  // Deep links for the demo video and screenshots: ?explain=<h3 cell> opens that hexagon's analysis,
  // ?photo=<id> applies a demo photo. Each fires once, when its data has loaded.
  const deepLinked = useRef({ explain: false, photo: false });
  useEffect(() => {
    const cell = params.get("explain");
    if (!cell || deepLinked.current.explain || !explained.length) return;
    const entry = explanationFor(cell);
    if (entry) {
      deepLinked.current.explain = true;
      setSelected({ kind: "hex", entry, fallback: null });
    }
  }, [explained, explanationFor, params]);
  useEffect(() => {
    const id = params.get("photo");
    if (!id || deepLinked.current.photo || !sensors) return;
    const obs = sensors.observations.find((o) => o.id === id);
    if (obs) {
      deepLinked.current.photo = true;
      applyPhoto(obs);
    }
  }, [sensors, params]); // eslint-disable-line react-hooks/exhaustive-deps

  const selectedPrediction = selected?.kind === "photo" ? predictionAt(selected.obs.h3) : null;
  const selectedFusion = selected?.kind === "photo" ? fusePhoto(selectedPrediction, selected.obs) : null;

  return (
    <div className="app">
      <div ref={mapContainer} className="map" />

      <aside className="panel">
        <h1>AeroGap</h1>
        <p className="tagline">Predicting the pollution your monitors miss.</p>

        {hasPredictions && (
          <div className="toggle" role="group" aria-label="Map view">
            <button className={showPredicted ? "active" : ""} onClick={() => setView("predicted")}>Predicted PM2.5</button>
            <button className={!showPredicted ? "active" : ""} onClick={() => setView("gap")}>Monitoring gaps</button>
          </div>
        )}

        <p className="lede">
          {showPredicted ? (
            <>Predicted daily PM2.5 for every ~5&nbsp;km hexagon, from satellite NO₂, aerosol and CO, wind, fires and
              the monitors that reported that day. Colour fades to grey where the model is less sure. Click a ringed
              hexagon for Gemini's source analysis and a drafted alert.</>
          ) : (
            <>Each hexagon is ~5&nbsp;km across, shaded by the distance to the nearest <em>working</em> air quality
              monitor. Stuck or silent monitors don't count. The darker it is, the less anyone is really measuring there.</>
          )}
        </p>

        <div className="cities" role="group" aria-label="Jump to region">
          {cities.map((c) => (
            <button key={c.id} className={c.id === active ? "active" : ""} onClick={() => flyTo(c)}>
              {c.label}
            </button>
          ))}
        </div>

        {showPredicted && (
          <div className="cities dates" role="group" aria-label="Date">
            {predictions.dates.map((d) => (
              <button key={d.date} className={d.date === activeDate.date ? "active" : ""} onClick={() => setDate(d.date)}>
                {new Date(`${d.date}T00:00:00`).toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" })}
              </button>
            ))}
          </div>
        )}

        <div className="legend">
          {showPredicted ? (
            <>
              <div className="legend-title">Predicted daily PM2.5 (µg/m³), CPCB category</div>
              <div className="ramp ramp-6">
                {CATEGORIES.map((c) => (
                  <div key={c.label} className="ramp-step">
                    <span className="swatch" style={{ background: c.color }} />
                    <span className="ramp-label">{c.label}</span>
                    <span className="ramp-label">{c.range}</span>
                  </div>
                ))}
              </div>
              <div className="legend-title legend-monitors">Confidence, from validation on hidden monitors</div>
              <div className="ramp ramp-3">
                {predictions.confidence_levels.map((lvl) => (
                  <div key={lvl.level} className="ramp-step">
                    <span className="swatch" style={{ background: rgbCss(mix(CAT_RGB[4], CONFIDENCE_MIX[lvl.level])) }} />
                    <span className="ramp-label">{CONFIDENCE_LABEL[lvl.level]} ±{Math.round(lvl.rel_error_pct)}%</span>
                    <span className="ramp-label">{CONFIDENCE_RANGE[lvl.level]}</span>
                  </div>
                ))}
              </div>
              <div className="legend-station legend-monitors">
                <span className="marker marker-explained" />
                <span className="legend-label">Gemini source analysis and alert: click</span>
                <span className="legend-count">{explained.length}</span>
              </div>
            </>
          ) : (
            <>
              <div className="legend-title">Distance to nearest working monitor (km)</div>
              <div className="ramp">
                {BINS.map((b) => (
                  <div key={b.label} className="ramp-step">
                    <span className="swatch" style={{ background: b.color }} />
                    <span className="ramp-label">{b.label}</span>
                  </div>
                ))}
              </div>
            </>
          )}
          <div className="legend-title legend-monitors">Monitors ({stations.length.toLocaleString()})</div>
          {HEALTH_ORDER.map((k) => (
            <div key={k} className="legend-station">
              <span className={`marker marker-${k}`} />
              <span className="legend-label">{HEALTH[k].label}</span>
              <span className="legend-count">{(healthCounts[k] || 0).toLocaleString()}</span>
            </div>
          ))}
        </div>

        {sensors && (
          <div className="sensor">
            <div className="legend-title legend-monitors">
              Virtual sensor: citizen sky photos <span className="gemini">Gemini</span>
            </div>
            <div className="thumbs">
              {sensors.observations.map((o) => (
                <button key={o.id} className="thumb" onClick={() => applyPhoto(o)} title={`${o.city}: ${o.reviewer_label}`}>
                  <img src={`${STATIC_BASE}/${o.photo}`} alt={`Sky photo, ${o.city}`} loading="lazy" />
                </button>
              ))}
            </div>
            <label className="upload">
              Upload a sky photo
              <input type="file" accept="image/jpeg,image/png" onChange={onUpload} />
            </label>
            {uploadMsg && <p className="status">{uploadMsg}</p>}
          </div>
        )}

        <p className="status">{status}</p>
        <p className="footnote">
          Stations: OpenAQ (CPCB/state networks, low-cost sensors), last 12 months. A monitor stuck on a placeholder
          value or silent for weeks counts as a gap, not coverage.
          {showPredicted && " Confidence comes from hiding 518 real monitors and scoring the model on them: "
            + "typical error is about ±31% within 20 km of a reporting monitor, ±38% at 20–100 km and ±59% beyond."}
        </p>
      </aside>

      {selected?.kind === "hex" && (
        <ExplanationPanel entry={selected.entry} fallback={selected.fallback} onClose={() => setSelected(null)} />
      )}
      {selected?.kind === "photo" && (
        <PhotoPanel obs={selected.obs} prediction={selectedPrediction} fusion={selectedFusion}
          photoUrl={`${STATIC_BASE}/${selected.obs.photo}`} onClose={() => setSelected(null)} />
      )}

      {hover && (
        <div className="tooltip" style={{ left: hover.x + 14, top: hover.y + 14 }}>
          {hover.kind === "predicted" && (
            <>
              <strong>{Math.round(hover.pm)} µg/m³</strong> predicted PM2.5 · {CATEGORIES[categoryOf(hover.pm)].label}
              <div className="muted">
                ±{Math.round(hover.u)} µg/m³ typical error · {CONFIDENCE_LABEL[hover.c]} confidence
              </div>
              <div className="muted">Nearest reporting monitor {hover.dist.toFixed(0)} km away</div>
            </>
          )}
          {hover.kind === "explained" && (
            <>
              <strong>{hover.signals.location.district}</strong>: click for Gemini's source analysis
            </>
          )}
          {hover.kind === "gap" && (
            <>
              <strong>{hover.d.toFixed(1)} km</strong> to nearest working monitor
              {Math.abs(hover.d - hover.dn) >= 0.1 && (
                <div className="muted">{hover.dn.toFixed(1)} km to nearest listed monitor</div>
              )}
              <div className="muted">
                {hover.s > 0 ? `${hover.s} listed station${hover.s > 1 ? "s" : ""} in this hex` : "No station in this hex"}
              </div>
            </>
          )}
          {hover.kind === "station" && (
            <>
              <strong>{hover.station_name}</strong>
              <div className="muted">{[hover.city, hover.state].filter(Boolean).join(", ")}</div>
              <div>
                <span className={`marker marker-${healthOf(hover)} marker-inline`} /> {HEALTH[healthOf(hover)].label}
              </div>
              {hover.health_reason && <div className="muted">{hover.health_reason}</div>}
              {hover.rows != null && (
                <div className="muted">
                  {hover.rows.toLocaleString()} readings
                  {hover.last_seen && hover.last_seen !== "NaT" ? ` · last ${hover.last_seen.slice(0, 10)}` : ""}
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
