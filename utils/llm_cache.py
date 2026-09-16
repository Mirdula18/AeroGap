"""Disk cache for every Gemini call. No Gemini call may bypass this module.

The demo and the recorded video must replay from disk with ZERO API calls, so
they can never fail on stage or trip a rate limit. A cached call is a plain file
read: no network, no key needed.

Key = sha256 over model + prompt_version + endpoint + the request body, with
image bytes folded in as their own digest. Changing the prompt, the model or the
image therefore changes the key; re-running the same call does not.

    from utils.llm_cache import gemini, text_part, image_part

    resp = gemini([text_part(PROMPT), image_part("photo.jpg")], prompt_version="haze-v1")
    print(resp.json["candidates"][0]["content"]["parts"][0]["text"], resp.cached)

CLI:
    python -m utils.llm_cache --stats          # what is cached
    python -m utils.llm_cache --verify         # replay every entry offline
    python -m utils.llm_cache --prune-errors   # drop cached non-200 responses

Flags for scripts: add_cache_args(parser) gives --no-cache and --offline.
AEROGAP_GEMINI_OFFLINE=1 forces offline mode everywhere (use it for the demo).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import mimetypes
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "gemini_cache"
API_BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-3.6-flash"  # 2.0/2.5 are retired for new keys (404 with an upgrade note)

RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 5
TIMEOUT = (20, 180)

log = logging.getLogger("utils.llm_cache")


class OfflineCacheMiss(RuntimeError):
    """Offline mode is on and this call is not cached."""


@dataclass
class GeminiResponse:
    json: dict
    cached: bool
    key: str
    path: Path
    latency_s: float

    @property
    def text(self) -> str:
        parts = self.json.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts)

    def parsed_json(self) -> dict | list | None:
        """The model's JSON reply. Plain prompts come back fenced; schema mode doesn't."""
        raw = self.text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0] if "\n" in raw else raw.strip("`")
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            return None


def text_part(text: str) -> dict:
    return {"text": text}


def image_part(source: str | Path | bytes, mime_type: str | None = None) -> dict:
    if isinstance(source, (str, Path)):
        path = Path(source)
        data = path.read_bytes()
        mime_type = mime_type or mimetypes.guess_type(path.name)[0] or "image/jpeg"
    else:
        data = source
        mime_type = mime_type or "image/jpeg"
    return {"inline_data": {"mime_type": mime_type, "data": base64.b64encode(data).decode()}}


def _digest_body(body: dict) -> dict:
    """Body with inline image data replaced by a digest, for hashing and for the stored record."""
    def walk(node):
        if isinstance(node, dict):
            if "inline_data" in node and isinstance(node["inline_data"], dict):
                inline = dict(node["inline_data"])
                raw = base64.b64decode(inline.get("data", ""))
                inline["data"] = f"sha256:{hashlib.sha256(raw).hexdigest()}:{len(raw)}"
                return {**{k: v for k, v in node.items() if k != "inline_data"}, "inline_data": inline}
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node
    return walk(body)


def cache_key(model: str, prompt_version: str, endpoint: str, body: dict) -> str:
    payload = {"model": model, "prompt_version": prompt_version, "endpoint": endpoint,
               "body": _digest_body(body)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _load_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or dotenv_values(ROOT / ".env").get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set (.env). Cached calls work without it; live calls don't.")
    return key


def _retry_delay(payload: dict, default: float) -> float:
    for detail in payload.get("error", {}).get("details", []):
        if detail.get("@type", "").endswith("RetryInfo"):
            raw = str(detail.get("retryDelay", "")).rstrip("s")
            try:
                return max(float(raw), default)
            except ValueError:
                pass
    return default


def gemini(parts: list[dict], *, model: str = DEFAULT_MODEL, prompt_version: str = "v1",
           generation_config: dict | None = None, system_instruction: str | None = None,
           use_cache: bool = True, offline: bool | None = None, max_attempts: int = MAX_ATTEMPTS,
           cache_dir: Path | None = None) -> GeminiResponse:
    """One Gemini generateContent call, cached to disk by content.

    use_cache=False forces a fresh call and overwrites the cached entry.
    offline=True (or AEROGAP_GEMINI_OFFLINE=1) refuses to touch the network.
    """
    endpoint = f"models/{model}:generateContent"
    body: dict = {"contents": [{"parts": parts}]}
    if generation_config:
        body["generationConfig"] = generation_config
    if system_instruction:
        body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    directory = cache_dir or CACHE_DIR
    key = cache_key(model, prompt_version, endpoint, body)
    path = directory / f"{key}.json"
    if offline is None:
        offline = os.environ.get("AEROGAP_GEMINI_OFFLINE", "") not in ("", "0", "false", "False")

    if use_cache and path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        return GeminiResponse(record["response"], True, key, path, record.get("latency_s", 0.0))
    if offline:
        raise OfflineCacheMiss(f"offline mode: {key[:12]}… not cached ({model}, prompt_version={prompt_version})")

    api_key = _load_key()
    started = time.time()
    last_error = None
    for attempt in range(1, max_attempts + 1):
        resp = requests.post(f"{API_BASE}/{endpoint}", params={"key": api_key}, json=body, timeout=TIMEOUT)
        if resp.status_code == 200:
            payload = resp.json()
            break
        payload = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        last_error = f"HTTP {resp.status_code}: {str(payload.get('error', {}).get('message', ''))[:200]}"
        if resp.status_code not in RETRY_STATUS or attempt == max_attempts:
            raise RuntimeError(f"Gemini call failed, {last_error}")
        wait = _retry_delay(payload, min(2 ** attempt, 60))
        log.warning("%s; retrying in %.0fs (attempt %d/%d)", last_error, wait, attempt, max_attempts)
        time.sleep(wait)

    latency = time.time() - started
    directory.mkdir(parents=True, exist_ok=True)
    record = {"key": key, "model": model, "prompt_version": prompt_version, "endpoint": endpoint,
              "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "latency_s": round(latency, 3), "request": _digest_body(body), "response": payload}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return GeminiResponse(payload, False, key, path, latency)


def add_cache_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--no-cache", action="store_true", help="force a fresh Gemini call and overwrite the cache")
    parser.add_argument("--offline", action="store_true", help="replay from cache only; fail on a cache miss")
    return parser


def stats(directory: Path | None = None) -> dict:
    directory = directory or CACHE_DIR
    files = sorted(directory.glob("*.json")) if directory.exists() else []
    by_model: dict[str, int] = {}
    errors = 0
    for f in files:
        rec = json.loads(f.read_text(encoding="utf-8"))
        by_model[f"{rec.get('model')} / {rec.get('prompt_version')}"] = by_model.get(
            f"{rec.get('model')} / {rec.get('prompt_version')}", 0) + 1
        errors += "error" in rec.get("response", {})
    return {"entries": len(files), "bytes": sum(f.stat().st_size for f in files),
            "by_model_and_prompt": by_model, "cached_errors": errors, "dir": str(directory)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--verify", action="store_true", help="re-read every entry offline and report unusable ones")
    parser.add_argument("--prune-errors", action="store_true", help="delete cached responses that carry an error")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.stats or not (args.verify or args.prune_errors):
        for k, v in stats().items():
            print(f"{k}: {v}")
    if args.verify:
        bad = []
        for f in sorted(CACHE_DIR.glob("*.json")):
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
                assert rec["response"]["candidates"][0]["content"]["parts"]
            except Exception as exc:
                bad.append((f.name, str(exc)[:80]))
        print(f"verify: {len(bad)} unusable of {len(list(CACHE_DIR.glob('*.json')))}")
        for name, why in bad:
            print(f"  {name}: {why}")
    if args.prune_errors:
        removed = 0
        for f in sorted(CACHE_DIR.glob("*.json")):
            if "error" in json.loads(f.read_text(encoding="utf-8")).get("response", {}):
                f.unlink()
                removed += 1
        print(f"pruned {removed} cached error responses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
