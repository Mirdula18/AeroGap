"""Virtual sensor: a citizen's sky photo -> structured haze reading -> a low-confidence PM2.5 observation.

Gemini reads the photo (utils/llm_cache via gemini.common, schema-enforced JSON). The reading
becomes an observation in the H3 res-7 hex where the photo was taken, which the map fuses with
the model prediction by inverse-variance weighting, so a single photo is one noisy vote.

Privacy (DPDP Act / Digital Public Good):
  - EXIF is stripped by re-encoding before anything is sent to Gemini or stored.
  - Only the H3 res-7 cell of the location is kept (about 5 km2), never coordinates.

The PM2.5 mapping is an UNCALIBRATED HEURISTIC. Haze severity bands are mapped onto CPCB
24-hour PM2.5 categories as an ordinal prior. There are no photo-monitor pairs to calibrate
against yet, visual haze is confounded by humidity, fog and light, and Gemini never abstains
by itself (docs/gemini-notes.md), hence the sky_visible guard and the deliberately wide ranges.

Usage:
    python -m gemini.photo_sensor                    # analyse the demo corpus (live once, then cached)
    python -m gemini.photo_sensor --offline          # replay only; fails on any cache miss
    python -m gemini.photo_sensor --image photo.jpg --lat 11.02 --lon 76.96
Outputs: data/virtual_sensors/observations.json, web/public/data/virtual_sensors.json
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
from pathlib import Path

import h3
from PIL import Image

from gemini.common import MODEL, generate_json
from utils.llm_cache import usage

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "gemini" / "demo_photos.json"
OUT = ROOT / "data" / "virtual_sensors" / "observations.json"
WEB_OUT = ROOT / "web" / "public" / "data" / "virtual_sensors.json"
H3_RES = 7

# photo-v1 omitted the 0-100 scale: Gemini answered on ~0-10 (severe smog came back haze 5 with 0.5 km
# visibility). Its cached replies are never used; v2 states every scale explicitly.
PROMPT_VERSION = "photo-v2"
SYSTEM = """You estimate visible air pollution from one photograph taken outdoors by a member of the public in India.

Fields and scales:
- sky_visible: true only if the photo shows outdoor sky or a distant outdoor view.
- haze_severity: integer from 0 to 100. 0 = perfectly clear air, distant detail crisp. 25 = light haze. 50 = distinctly hazy, distant buildings or hills washed out. 75 = heavy smog, nearby buildings faded. 100 = near-opaque smog, only close objects visible.
- visibility_km_estimate: how far you can see, in kilometres.
- likely_source: vehicular, industrial, biomass_burning, dust or unclear.
- confidence: number from 0 to 1.
- reasoning: one sentence.
haze_severity and visibility must agree: visibility under 2 km means haze_severity of at least 60.

Rules:
- Judge only what is visible in THIS image: sky colour, haze, how far detail stays sharp, whether distant objects fade.
- If the image does not show outdoor sky or a distant view (a satellite image, map, chart, indoor scene or close-up), set sky_visible to false, haze_severity to 0, visibility_km_estimate to 0, likely_source to "unclear" and confidence to 0.
- Fog, mist, sunrise or sunset glow, and camera exposure can look like haze. When you cannot tell them apart, lower your confidence.
- likely_source must come from visual cues only: a grey-brown layer over traffic suggests vehicular; smoke plumes or an orange-brown pall over farmland suggests biomass_burning; a yellow-tan cast suggests dust; stacks or plumes suggest industrial. Use "unclear" when there is no specific cue.
- Do not use what you know about a city's usual pollution. Judge the image.
- reasoning: one sentence naming the visual cues you used."""
USER_PROMPT = "Assess the air quality visible in this photo."
SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "sky_visible": {"type": "BOOLEAN", "description": "true only if outdoor sky or a distant outdoor view is visible"},
        "haze_severity": {"type": "INTEGER", "description": "0 (perfectly clear) to 100 (near-opaque smog)"},
        "visibility_km_estimate": {"type": "NUMBER", "description": "estimated visibility in kilometres"},
        "likely_source": {"type": "STRING", "enum": ["vehicular", "industrial", "biomass_burning", "dust", "unclear"]},
        "confidence": {"type": "NUMBER", "description": "0 to 1"},
        "reasoning": {"type": "STRING", "description": "one sentence naming the visual cues used"},
    },
    "required": ["sky_visible", "haze_severity", "visibility_km_estimate", "likely_source", "confidence", "reasoning"],
}
REQUIRED = SCHEMA["required"]

# Uncalibrated ordinal prior: haze severity band -> CPCB PM2.5 range (µg/m³).
HAZE_TO_PM25 = [(20, 0, 30, "Good"), (40, 31, 60, "Satisfactory"), (60, 61, 120, "Moderate to Poor"),
                (80, 121, 250, "Very poor"), (101, 251, 400, "Severe")]


def prepare_image(path: str | Path, max_side: int = 1024) -> bytes:
    """Re-encode as JPEG without EXIF (location, device, time), downscaled. The only bytes ever sent or stored."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def pm25_range(haze_severity: float) -> tuple[int, int, str]:
    for upper, low, high, label in HAZE_TO_PM25:
        if haze_severity < upper:
            return low, high, label
    return HAZE_TO_PM25[-1][1], HAZE_TO_PM25[-1][2], HAZE_TO_PM25[-1][3]


def analyse(image_bytes: bytes, use_cache: bool = True, offline: bool | None = None) -> dict:
    from utils.llm_cache import image_part, text_part

    result = generate_json([text_part(USER_PROMPT), image_part(image_bytes, "image/jpeg")], schema=SCHEMA,
                           prompt_version=PROMPT_VERSION, system_instruction=SYSTEM, required=REQUIRED,
                           thinking_budget=0, use_cache=use_cache, offline=offline)
    if result["status"] != "ok":
        return {"status": result["status"], "detail": result.get("error") or result.get("failures")}

    r = result["data"]
    sky = bool(r["sky_visible"])
    haze = max(0, min(100, int(r["haze_severity"])))
    conf = max(0.0, min(1.0, float(r["confidence"])))
    out = {"status": "ok" if sky else "no_sky", "cached": result["cached"], "model": MODEL,
           "prompt_version": PROMPT_VERSION, "sky_visible": sky, "haze_severity": haze,
           "visibility_km_estimate": round(float(r["visibility_km_estimate"]), 1),
           "likely_source": r["likely_source"], "confidence": round(conf, 2), "reasoning": r["reasoning"]}
    if sky:
        low, high, label = pm25_range(haze)
        # Self-consistency guard: near-zero visibility with a low haze score means the reply is off-scale or
        # confused (exactly how photo-v1 failed). Keep it, but let it carry almost no weight.
        inconsistent = out["visibility_km_estimate"] <= 2 and haze < 50
        out |= {"pm25_low": low, "pm25_high": high, "pm25_mid": (low + high) / 2, "pm25_category": label,
                "calibrated": False, "consistency_warning": inconsistent,
                "effective_confidence": round(conf * (0.2 if inconsistent else 1.0), 2)}
    else:
        out |= {"effective_confidence": 0.0}  # no sky, no observation - whatever Gemini's numbers say
    return out


def fuse(prediction: float, prediction_uncertainty: float, obs: dict) -> dict:
    """Inverse-variance blend of the model prediction and one photo observation.

    The photo's spread is half its PM2.5 range, inflated by 1/confidence, so a vague or doubtful
    photo barely moves the prediction.
    """
    if obs.get("status") != "ok" or obs.get("effective_confidence", 0) <= 0:
        return {"fused_pm25": prediction, "photo_weight": 0.0}
    sigma_pred = max(prediction_uncertainty, 1.0)
    sigma_obs = ((obs["pm25_high"] - obs["pm25_low"]) / 2) / max(obs["effective_confidence"], 0.05)
    w_obs, w_pred = 1 / sigma_obs ** 2, 1 / sigma_pred ** 2
    weight = w_obs / (w_obs + w_pred)
    return {"fused_pm25": round(weight * obs["pm25_mid"] + (1 - weight) * prediction, 1),
            "fused_uncertainty": round(math.sqrt(1 / (w_obs + w_pred)), 1), "photo_weight": round(weight, 3)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", help="analyse one photo instead of the demo corpus")
    parser.add_argument("--lat", type=float)
    parser.add_argument("--lon", type=float)
    parser.add_argument("--offline", action="store_true", help="replay from cache only")
    parser.add_argument("--no-cache", action="store_true", help="force fresh Gemini calls")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    offline = True if args.offline else None

    if args.image:
        res = analyse(prepare_image(args.image), use_cache=not args.no_cache, offline=offline)
        if args.lat is not None and args.lon is not None:
            res["h3"] = h3.latlng_to_cell(args.lat, args.lon, H3_RES)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        print("gemini usage:", usage())
        return 0

    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    observations = []
    print(f"{'photo':<22}{'reviewer':<10}{'sky':<5}{'haze':>5}{'vis km':>8}  {'source':<16}{'conf':>5}  "
          f"{'PM2.5 range':<12}{'cached':<7}")
    for photo in corpus:
        path = ROOT / photo["file"]
        image_bytes = path.read_bytes()  # already EXIF-free; hashing these exact bytes lets the map match uploads
        res = analyse(image_bytes, use_cache=not args.no_cache, offline=offline)
        cell = h3.latlng_to_cell(photo["lat"], photo["lon"], H3_RES)
        obs = {"id": photo["id"], "sha256": hashlib.sha256(image_bytes).hexdigest(), "photo": photo["web_path"],
               "h3": cell, "city": photo["city"], "captured": photo.get("captured", ""),
               "reviewer_label": photo["reviewer_label"], "credit": photo["credit"], **res}
        observations.append(obs)
        pm = f"{res.get('pm25_low', '-')}-{res.get('pm25_high', '-')}" if res.get("status") == "ok" else "none"
        print(f"{photo['id']:<22}{photo['reviewer_label']:<10}{str(res.get('sky_visible', '?'))[:5]:<5}"
              f"{res.get('haze_severity', '-'):>5}{res.get('visibility_km_estimate', '-'):>8}  "
              f"{str(res.get('likely_source', res.get('status'))):<16}{res.get('confidence', '-'):>5}  "
              f"{pm:<12}{str(res.get('cached', '-')):<7}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(observations, indent=2, ensure_ascii=False), encoding="utf-8")
    WEB_OUT.write_text(json.dumps({"model": MODEL, "prompt_version": PROMPT_VERSION, "calibrated": False,
                                   "note": "Haze severity mapped onto CPCB PM2.5 bands as an uncalibrated prior; "
                                           "one photo is one noisy vote.",
                                   "haze_to_pm25": [{"below": u, "low": lo, "high": hi, "label": lab}
                                                    for u, lo, hi, lab in HAZE_TO_PM25],
                                   "observations": observations}, ensure_ascii=False, indent=1), encoding="utf-8")
    ok = sum(o.get("status") == "ok" for o in observations)
    print(f"\n{ok} usable observations, {sum(o.get('status') == 'no_sky' for o in observations)} no-sky, "
          f"{sum(o.get('status') not in ('ok', 'no_sky') for o in observations)} failed")
    print("gemini usage:", usage())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
