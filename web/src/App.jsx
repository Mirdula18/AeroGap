import { useEffect, useMemo, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import { MapboxOverlay } from "@deck.gl/mapbox";
import { H3HexagonLayer } from "@deck.gl/geo-layers";
import { ScatterplotLayer } from "@deck.gl/layers";
import cities from "./cities.json";

const API = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");

// Free, keyless basemap (CARTO Positron, OSM data). Light so the sequential
// ramp reads "further from a monitor = darker".
const BASEMAP = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json";

// Sequential single-hue ramp (blue 100 -> 700). Binned because distance is
// heavily skewed: most of the story is between 10 and 100 km.
const BINS = [
  { max: 5, label: "< 5", color: "#cde2fb" },
  { max: 10, label: "5–10", color: "#9ec5f4" },
  { max: 25, label: "10–25", color: "#6da7ec" },
  { max: 50, label: "25–50", color: "#3987e5" },
  { max: 100, label: "50–100", color: "#256abf" },
  { max: 200, label: "100–200", color: "#184f95" },
  { max: Infinity, label: "200+", color: "#0d366b" },
];
const STATION_COLOR = [235, 104, 52]; // categorical slot 2 (orange), distinct from the blue ramp

const hexToRgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
const BIN_RGB = BINS.map((b) => hexToRgb(b.color));
const colorFor = (km) => BIN_RGB[BINS.findIndex((b) => km < b.max)];

export default function App() {
  const mapContainer = useRef(null);
  const mapRef = useRef(null);
  const overlayRef = useRef(null);
  const [grid, setGrid] = useState(null);
  const [stations, setStations] = useState([]);
  const [hover, setHover] = useState(null);
  const [status, setStatus] = useState("Loading…");
  const [active, setActive] = useState("india");

  useEffect(() => {
    const map = new maplibregl.Map({
      container: mapContainer.current,
      style: BASEMAP,
      center: [cities[0].longitude, cities[0].latitude],
      zoom: cities[0].zoom,
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
    const load = () => {
      clearTimeout(timer);
      timer = setTimeout(async () => {
        controller?.abort();
        controller = new AbortController();
        const b = map.getBounds();
        const bbox = [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()].map((v) => v.toFixed(3)).join(",");
        try {
          setStatus("Loading hexes…");
          const res = await fetch(`${API}/grid?bbox=${bbox}`, { signal: controller.signal });
          if (!res.ok) throw new Error(`API ${res.status}`);
          const data = await res.json();
          setGrid(data);
          setStatus(`${data.count.toLocaleString()} hexes · H3 res ${data.res}`);
        } catch (err) {
          if (err.name !== "AbortError") setStatus(`Could not reach the API (${err.message}). It may be waking up — retrying on the next move.`);
        }
      }, 250);
    };
    // Don't wait for the basemap's "load" event: if the tile/style host is slow
    // or blocked, the gap layer (the actual product) must still render.
    load();
    map.on("moveend", load);

    fetch(`${API}/stations`)
      .then((r) => r.json())
      .then((d) => setStations(d.stations))
      .catch(() => {});

    return () => {
      clearTimeout(timer);
      map.remove();
    };
  }, []);

  const hexRows = useMemo(() => {
    if (!grid) return [];
    return grid.h3.map((h, i) => ({ h3: h, d: grid.dist_km[i], s: grid.station_count[i] }));
  }, [grid]);

  useEffect(() => {
    if (!overlayRef.current) return;
    overlayRef.current.setProps({
      layers: [
        new H3HexagonLayer({
          id: "gaps",
          data: hexRows,
          getHexagon: (d) => d.h3,
          getFillColor: (d) => [...colorFor(d.d), 200],
          filled: true,
          stroked: false,
          extruded: false,
          highPrecision: "auto",
          pickable: true,
          onHover: (info) => setHover(info.object ? { kind: "hex", x: info.x, y: info.y, ...info.object } : null),
          updateTriggers: { getFillColor: [grid] },
        }),
        new ScatterplotLayer({
          id: "stations",
          data: stations,
          getPosition: (d) => [d.lon, d.lat],
          getFillColor: STATION_COLOR,
          getLineColor: [252, 252, 251],
          lineWidthUnits: "pixels",
          getLineWidth: 2,
          stroked: true,
          radiusUnits: "pixels",
          getRadius: 5,
          pickable: true,
          onHover: (info) => setHover(info.object ? { kind: "station", x: info.x, y: info.y, ...info.object } : null),
        }),
      ],
    });
  }, [hexRows, stations, grid]);

  const flyTo = (city) => {
    setActive(city.id);
    mapRef.current?.flyTo({ center: [city.longitude, city.latitude], zoom: city.zoom, duration: 1600 });
  };

  return (
    <div className="app">
      <div ref={mapContainer} className="map" />

      <aside className="panel">
        <h1>AeroGap</h1>
        <p className="tagline">Predicting the pollution your monitors miss.</p>
        <p className="lede">
          Each hexagon is ~5&nbsp;km across, shaded by the distance to the nearest air quality monitor. The darker
          it is, the less anyone is measuring there.
        </p>

        <div className="cities" role="group" aria-label="Jump to region">
          {cities.map((c) => (
            <button key={c.id} className={c.id === active ? "active" : ""} onClick={() => flyTo(c)}>
              {c.label}
            </button>
          ))}
        </div>

        <div className="legend">
          <div className="legend-title">Distance to nearest monitor (km)</div>
          <div className="ramp">
            {BINS.map((b) => (
              <div key={b.label} className="ramp-step">
                <span className="swatch" style={{ background: b.color }} />
                <span className="ramp-label">{b.label}</span>
              </div>
            ))}
          </div>
          <div className="legend-station">
            <span className="dot" /> Monitoring station ({stations.length.toLocaleString()})
          </div>
        </div>

        <p className="status">{status}</p>
        <p className="footnote">
          Stations: OpenAQ (CPCB/state networks, low-cost sensors), last 12 months. No predictions yet: this view
          shows the gap, not the pollution.
        </p>
      </aside>

      {hover && (
        <div className="tooltip" style={{ left: hover.x + 14, top: hover.y + 14 }}>
          {hover.kind === "hex" ? (
            <>
              <strong>{hover.d.toFixed(1)} km</strong> to nearest monitor
              <div className="muted">
                {hover.s > 0 ? `${hover.s} station${hover.s > 1 ? "s" : ""} in this hex` : "No station in this hex"}
              </div>
            </>
          ) : (
            <>
              <strong>{hover.station_name}</strong>
              <div className="muted">
                {[hover.city, hover.state].filter(Boolean).join(", ")}
              </div>
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
