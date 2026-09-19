"""Multilingual alert drafts from a source attribution: authority and citizen, in English, Tamil and Hindi.

Input: data/explanations/<date>.json (signals + attribution per demo hex, from gemini/attribute.py).
Gemini drafts, in one schema-enforced call per hex (through utils/llm_cache via gemini.common):
  - an alert for the responsible authority (formal): location, date, predicted level, confidence,
    likely source, recommended action, urgency - in English, Tamil and Hindi;
  - a citizen-facing version in plainer language, in the same three languages.
Every draft must say it is a model estimate, and may use only the facts supplied.

Checks after the call: the predicted PM2.5 value must appear (Western digits) in all six texts.

Dispatch is MOCKED. Nothing is sent. Payloads addressed to roles (never named people) are printed
and written to data/alerts/; only "elevated" or "urgent" hexes are marked would_send.

Usage:
    python -m gemini.alert                      # the three demo dates
    python -m gemini.alert --offline            # replay only
Outputs: data/alerts/<date>_<h3>.json, alert blocks merged into web/public/data/explanations_<date>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from gemini.common import MODEL, generate_json
from utils.llm_cache import text_part, usage

ROOT = Path(__file__).resolve().parents[1]
EXPLANATIONS = ROOT / "data" / "explanations"
ALERTS = ROOT / "data" / "alerts"
WEB_DATA = ROOT / "web" / "public" / "data"
DEFAULT_DATES = "2025-11-12,2025-11-22,2026-09-10"
LANGS = ["en", "ta", "hi"]

# Batched: one call per date for all hexes (20 requests/day per model on the free tier).
PROMPT_VERSION = "alert-batch-v1"
SYSTEM = """You draft short air-quality alerts for India. You receive a JSON array; each item has a hex_id and the facts about one ~5 km area on one day. Draft each item INDEPENDENTLY from its own facts only, and return exactly one result per hex_id with the same hex_id.

Rules:
- Use ONLY the facts in the JSON. Do not add places, causes, events, statistics or health figures that are not there.
- This is a MODEL ESTIMATE, not a measurement: every text must say so plainly.
- Copy numbers exactly. Always write numbers with Western Arabic digits (0-9), including in Tamil and Hindi.
- If likely_source is "insufficient_signal", say the source could not be determined; do not guess one.
- If model_confidence is "low", say the estimate is uncertain and that verification comes first.
- Tamil and Hindi must be natural, fluent translations of the same content, not transliterations. Keep district and state names in the form normally used in that language.

authority (for the district administration and state pollution control board), formal, at most 90 words per language. Include: location (district, state), date, predicted PM2.5 in µg/m³ with its category, model confidence with its typical error, likely source, recommended action, urgency.
citizen (public advisory), plain and calm, at most 60 words per language. Include: area, date, the predicted level in everyday words plus the number, what people can do today proportionate to the category, and that it is an estimate. No alarmism when confidence is low.
subject_en: a subject line of at most 12 words for the authority alert."""
HEX_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "hex_id": {"type": "STRING"},
        "subject_en": {"type": "STRING"},
        "authority": {"type": "OBJECT", "properties": {l: {"type": "STRING"} for l in LANGS}, "required": LANGS},
        "citizen": {"type": "OBJECT", "properties": {l: {"type": "STRING"} for l in LANGS}, "required": LANGS},
    },
    "required": ["hex_id", "subject_en", "authority", "citizen"],
}
SCHEMA = {"type": "OBJECT", "properties": {"results": {"type": "ARRAY", "items": HEX_SCHEMA}}, "required": ["results"]}
REQUIRED = ["results"]
HEX_REQUIRED = HEX_SCHEMA["required"]


def alert_facts(entry: dict) -> dict:
    s, a = entry["signals"], entry["attribution"]
    return {
        "district": s["location"]["district"], "state": s["location"]["state"], "date": s["date"],
        "predicted_pm25_ug_m3": round(s["predicted_pm25_ug_m3"]), "predicted_category": s["predicted_category"],
        "model_confidence": s["model_confidence"], "typical_error_pct": round(s["prediction_typical_error_pct"]),
        "likely_source": a.get("dominant_source", "insufficient_signal"),
        "source_confidence": a.get("source_confidence"),
        "recommended_action": a.get("recommended_action"), "urgency": a.get("urgency", "routine"),
        "uncertainty_statement": a.get("uncertainty_statement"),
    }


def number_check(facts: dict, draft: dict) -> list[str]:
    value = str(facts["predicted_pm25_ug_m3"])
    return [f"{audience}.{lang} is missing the predicted value {value}"
            for audience in ("authority", "citizen") for lang in LANGS if value not in draft[audience].get(lang, "")]


def draft_alerts(facts_by_hex: dict[str, dict], use_cache: bool = True, offline: bool | None = None) -> dict[str, dict]:
    """One Gemini call for all hexes of a date. Returns hex_id -> draft (or a per-hex failure status)."""
    items = [{"hex_id": cell, "facts": f} for cell, f in facts_by_hex.items()]
    result = generate_json([text_part("Areas:\n" + json.dumps(items, ensure_ascii=False, indent=1))], schema=SCHEMA,
                           prompt_version=PROMPT_VERSION, system_instruction=SYSTEM, required=REQUIRED,
                           use_cache=use_cache, offline=offline)
    if result["status"] != "ok":
        failure = {"status": result["status"], "detail": result.get("error") or result.get("failures")}
        return {cell: failure for cell in facts_by_hex}
    by_id = {r.get("hex_id"): r for r in result["data"]["results"] if isinstance(r, dict)}
    out = {}
    for cell, facts in facts_by_hex.items():
        d = by_id.get(cell)
        if d is None or not all(k in d for k in HEX_REQUIRED):
            out[cell] = {"status": "missing_from_batch"}
            continue
        d = {k: v for k, v in d.items() if k != "hex_id"}
        out[cell] = {"status": "ok", "cached": result["cached"], "model": MODEL, "prompt_version": PROMPT_VERSION,
                     **d, "checks": number_check(facts, d)}
    return out


def mock_dispatch(day: str, cell: str, facts: dict, draft: dict) -> dict:
    would_send = draft.get("status") == "ok" and facts["urgency"] in ("elevated", "urgent")
    payload = {
        "mock": True, "sent": False, "would_send": would_send,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "h3": cell, "date": day, "urgency": facts["urgency"],
        "authority_recipients": [f"District Collector, {facts['district']}", f"{facts['state']} Pollution Control Board"],
        "citizen_channel": "public advisory (SMS / WhatsApp broadcast), not wired",
        "facts": facts, "draft": draft,
    }
    ALERTS.mkdir(parents=True, exist_ok=True)
    (ALERTS / f"{day}_{cell}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dates", default=DEFAULT_DATES)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    offline = True if args.offline else None

    for text in args.dates.split(","):
        day = text.strip()
        doc = json.loads((EXPLANATIONS / f"{day}.json").read_text(encoding="utf-8"))
        print(f"\n=== {day}")
        facts_by_hex = {}
        for cell, entry in doc["hexes"].items():
            if entry["attribution"].get("status") != "ok":
                entry["alert"] = {"status": "skipped", "reason": "attribution unavailable"}
            else:
                facts_by_hex[cell] = alert_facts(entry)
        drafts = draft_alerts(facts_by_hex, use_cache=not args.no_cache, offline=offline) if facts_by_hex else {}
        for cell, facts in facts_by_hex.items():
            entry, draft = doc["hexes"][cell], drafts[cell]
            payload = mock_dispatch(day, cell, facts, draft)
            entry["alert"] = {**draft, "would_send": payload["would_send"],
                              "authority_recipients": payload["authority_recipients"]}
            ok = draft.get("status") == "ok"
            state = ("WOULD SEND" if payload["would_send"] else "draft only") if ok else f"FAILED ({draft.get('status')})"
            print(f"  [{state:<10}] {facts['district']}, {facts['state']}"
                  f" | {facts['predicted_pm25_ug_m3']} µg/m³ {facts['predicted_category']} | {facts['urgency']}"
                  + (f" | cached={draft.get('cached')} | checks: {draft.get('checks') or 'ok'}" if ok else ""))
            if draft.get("status") == "ok":
                print(f"    subject: {draft['subject_en']}")
        (EXPLANATIONS / f"{day}.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        (WEB_DATA / f"explanations_{day}.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    print("\ngemini usage:", usage())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
