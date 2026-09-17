"""Shared Gemini call path for AeroGap. Every call goes through utils.llm_cache.

- Structured output only: responseMimeType application/json + responseSchema. Plain prompts
  always came back wrapped in markdown fences (docs/gemini-notes.md); schema mode never did.
  parsed_json() strips fences anyway, as a second line of defence.
- One retry on an unparseable or incomplete reply (a fresh call that overwrites the cached
  bad one), then a graceful {"status": "unparseable"} instead of an exception.
- HTTP failures after the wrapper's own retries come back as {"status": "error"}.
- In offline mode (the demo) nothing is retried and a cache miss raises OfflineCacheMiss.
"""

from __future__ import annotations

from utils.llm_cache import OfflineCacheMiss, gemini

MODEL = "gemini-3.6-flash"  # 2.0 / 2.5 flash are retired for this key (404)

__all__ = ["MODEL", "OfflineCacheMiss", "generate_json"]


def generate_json(parts: list[dict], *, schema: dict, prompt_version: str, system_instruction: str,
                  required: list[str], thinking_budget: int | None = None, use_cache: bool = True,
                  offline: bool | None = None) -> dict:
    config: dict = {"responseMimeType": "application/json", "responseSchema": schema}
    if thinking_budget is not None:
        config["thinkingConfig"] = {"thinkingBudget": thinking_budget}

    failures = []
    for attempt in (1, 2):
        try:
            resp = gemini(parts, model=MODEL, prompt_version=prompt_version, generation_config=config,
                          system_instruction=system_instruction,
                          use_cache=use_cache if attempt == 1 else False, offline=offline)
        except OfflineCacheMiss:
            raise
        except Exception as exc:  # one failed photo or hex must not take down a batch
            return {"status": "error", "error": f"{type(exc).__name__}: {str(exc)[:300]}", "attempts": attempt}

        candidates = resp.json.get("candidates") or [{}]
        finish = candidates[0].get("finishReason")
        data = resp.parsed_json()
        if isinstance(data, dict) and all(k in data for k in required):
            return {"status": "ok", "data": data, "cached": resp.cached, "cache_key": resp.key,
                    "attempts": attempt, "finish_reason": finish}
        failures.append({"finish_reason": finish, "cached": resp.cached, "raw": resp.text[:300]})
        if offline:  # the demo replays only; never reach the network to repair a cached reply
            break
    return {"status": "unparseable", "attempts": len(failures), "failures": failures}
