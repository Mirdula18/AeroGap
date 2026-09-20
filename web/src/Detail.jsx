import { useState } from "react";

export const SOURCE_LABEL = {
  vehicular: "Vehicular traffic",
  industrial: "Industrial combustion",
  biomass_burning: "Biomass burning",
  dust: "Dust",
  mixed_combustion: "Mixed combustion",
  insufficient_signal: "Insufficient signal",
  unclear: "Unclear",
};
const ROLE_LABEL = {
  delhi_centre: "Delhi city centre",
  coimbatore_centre: "Coimbatore city centre",
  dark_zone: "Dark zone: far from any working monitor",
  low_confidence: "Low-confidence area: over 100 km from a reporting monitor",
};
// Status colours are reserved for state and always carry a text label.
const URGENCY = {
  routine: { label: "Routine", color: "#0ca30c" },
  elevated: { label: "Elevated", color: "#fab219" },
  urgent: { label: "Urgent", color: "#d03b3b" },
};
const LANGS = [
  { id: "en", label: "English" },
  { id: "ta", label: "தமிழ்" },
  { id: "hi", label: "हिंदी" },
];
const pretty = (key) => key.replace(/_/g, " ").replace(/\bkm\b/, "km").replace(/\bmol m2\b/, "(mol/m²)");

function Badge({ urgency }) {
  const u = URGENCY[urgency] || URGENCY.routine;
  return (
    <span className="badge">
      <span className="badge-dot" style={{ background: u.color }} />
      {u.label}
    </span>
  );
}

function AlertTabs({ alert }) {
  const [audience, setAudience] = useState("authority");
  const [lang, setLang] = useState("en");
  if (!alert || alert.status !== "ok") {
    return <p className="muted small">No alert drafted{alert?.reason ? `: ${alert.reason}` : ""}.</p>;
  }
  return (
    <div className="alert">
      <div className="tabs" role="tablist">
        {["authority", "citizen"].map((a) => (
          <button key={a} role="tab" aria-selected={audience === a} className={audience === a ? "active" : ""}
            onClick={() => setAudience(a)}>
            {a === "authority" ? "Authority" : "Citizens"}
          </button>
        ))}
        <span className="tabs-spacer" />
        {LANGS.map((l) => (
          <button key={l.id} className={lang === l.id ? "active" : ""} onClick={() => setLang(l.id)} lang={l.id}>
            {l.label}
          </button>
        ))}
      </div>
      {audience === "authority" && <p className="alert-subject">{alert.subject_en}</p>}
      <p className="alert-text" lang={lang}>{alert[audience][lang]}</p>
      <p className="muted small">
        {alert.would_send
          ? `Mock dispatch, not sent. Would go to: ${alert.authority_recipients.join("; ")}.`
          : "Draft only: urgency is routine, so nothing would be dispatched."}
      </p>
      {alert.checks?.length > 0 && <p className="warn small">Check: {alert.checks.join("; ")}</p>}
    </div>
  );
}

export function ExplanationPanel({ entry, fallback, onClose }) {
  if (!entry) {
    return (
      <aside className="detail">
        <button className="close" onClick={onClose} aria-label="Close">×</button>
        <h2>This hexagon</h2>
        {fallback && (
          <p>
            <strong>{Math.round(fallback.pm)} µg/m³</strong> predicted PM2.5, ±{Math.round(fallback.u)} µg/m³ typical
            error. Nearest reporting monitor {Math.round(fallback.dist)} km away.
          </p>
        )}
        <p className="muted">
          No Gemini explanation was precomputed for this hexagon. The static demo replays cached explanations only, for
          the hexagons marked with a white ring.
        </p>
      </aside>
    );
  }
  const s = entry.signals;
  const a = entry.attribution;
  return (
    <aside className="detail">
      <button className="close" onClick={onClose} aria-label="Close">×</button>
      <p className="eyebrow">{ROLE_LABEL[entry.role] || entry.role}</p>
      <h2>{[s.location.district, s.location.state].filter(Boolean).join(", ")}</h2>
      <p className="headline">
        <strong>{Math.round(s.predicted_pm25_ug_m3)} µg/m³</strong> {s.predicted_category} · {s.model_confidence} confidence
        (±{Math.round(s.prediction_typical_error_pct)}%)
      </p>
      <p className="muted small">
        Model estimate for {s.date}. Nearest reporting monitor {s.distance_to_nearest_reporting_monitor_km} km, nearest
        working monitor {s.distance_to_nearest_working_monitor_km} km.
      </p>

      {a.status === "pending" ? (
        <p className="muted">Gemini source analysis for this hexagon has not been generated yet.</p>
      ) : a.status !== "ok" ? (
        <p className="warn">Gemini attribution unavailable ({a.status}).</p>
      ) : (
        <>
          <div className="section-title">
            Likely source <span className="gemini">Gemini</span>
          </div>
          <p className="source">
            {SOURCE_LABEL[a.dominant_source] || a.dominant_source}
            <span className="muted"> · {a.source_confidence} confidence</span>
            <Badge urgency={a.urgency} />
          </p>
          <p>{a.reasoning}</p>
          <ul className="evidence">
            {a.evidence.map((e, i) => (
              <li key={i}>
                <span className="evidence-signal">{pretty(e.signal)}</span> <span className="evidence-value">{e.value}</span>
                <span className="muted"> · {e.interpretation}</span>
              </li>
            ))}
          </ul>
          <div className="section-title">Recommended action</div>
          <p>{a.recommended_action}</p>
          <div className="uncertainty">
            <strong>Uncertainty.</strong> {a.uncertainty_statement}
          </div>
          {a.grounding_warnings?.length > 0 && (
            <p className="warn small">Grounding check: {a.grounding_warnings.join("; ")}</p>
          )}
          <div className="section-title">Drafted alert</div>
          <AlertTabs alert={entry.alert} />
        </>
      )}
    </aside>
  );
}

export function PhotoPanel({ obs, fusion, prediction, photoUrl, onClose, onRemove }) {
  const c = obs.credit || {};
  return (
    <aside className="detail">
      <button className="close" onClick={onClose} aria-label="Close">×</button>
      <p className="eyebrow">
        Virtual sensor <span className="gemini">Gemini</span>
      </p>
      <h2>Citizen photo, {obs.city}</h2>
      <img className="photo" src={photoUrl} alt={`Sky photo from ${obs.city}`} />
      <p className="muted small">
        {c.title} · {c.artist || "unknown author"} ·{" "}
        <a href={c.license_url} target="_blank" rel="noreferrer">{c.license}</a> ·{" "}
        <a href={c.source_url} target="_blank" rel="noreferrer">Wikimedia Commons</a>. Archive photo placed at city level
        for the demo; not taken on the selected date.
      </p>
      {obs.status === "no_sky" ? (
        <p className="warn">Gemini found no outdoor sky in this image, so it is not used as an observation.</p>
      ) : obs.status !== "ok" ? (
        <p className="warn">This photo could not be analysed ({obs.status}).</p>
      ) : (
        <>
          <p className="headline">
            Haze <strong>{obs.haze_severity}/100</strong> · visibility ~{obs.visibility_km_estimate} km ·{" "}
            {SOURCE_LABEL[obs.likely_source] || obs.likely_source} · confidence {obs.confidence}
          </p>
          <p>{obs.reasoning}</p>
          <p>
            Estimated PM2.5 <strong>{obs.pm25_low}–{obs.pm25_high} µg/m³</strong> ({obs.pm25_category}).{" "}
            <span className="muted">Uncalibrated: a coarse mapping from haze to PM2.5, not a measurement.</span>
          </p>
          {obs.consistency_warning && (
            <p className="warn small">Gemini's haze and visibility disagree, so this photo carries very little weight.</p>
          )}
          <div className="section-title">Folded into the grid</div>
          {prediction && fusion ? (
            <p>
              Model prediction here: {Math.round(prediction.pm)} ±{Math.round(prediction.u)} µg/m³. With this photo as one
              low-confidence observation: <strong>{Math.round(fusion.fused)} µg/m³</strong> (photo weight{" "}
              {Math.round(fusion.weight * 100)}%).
            </p>
          ) : (
            <p className="muted">Switch to the Predicted PM2.5 view to see how this photo shifts the estimate for its hexagon.</p>
          )}
        </>
      )}
      {onRemove && (
        <button className="withdraw" onClick={onRemove}>Remove this reading from the map</button>
      )}
    </aside>
  );
}
